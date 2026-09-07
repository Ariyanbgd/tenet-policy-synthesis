# Code backbone: Prompt-DT https://github.com/mxu34/prompt-dt
# Prompt-DT builds on Decision Transformer: https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

"""Prompt-DT trajectory encoding and TENET text-to-policy components.

The inherited transformer encodes prompted trajectories. TENET adds a frozen or
fine-tunable language encoder, embedding-alignment options, and a hypernetwork
that instantiates compact task-conditioned policies.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Union

import transformers

from .trajectory_gpt2 import GPT2Model

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

import copy

class TextEncoder(nn.Module):
    """Encode task descriptions and project them into policy-embedding space."""

    def __init__(
        self,
        model_name,
        hidden_size,
        llm_finetune_method,
        normalize_embeddings,
        pooling_type, 
        projection_type,  
        num_projection_layers,
        device,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        transformer_nhead=4,
        transformer_nlayer=4
    ):
        super().__init__()
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.pooling_type = pooling_type.lower()
        self.projection_type = projection_type.lower()
        self.num_projection_layers = num_projection_layers
        self.finetune_method = llm_finetune_method.lower()

        # Tokenization uses EOS as padding because LLaMA has no pad token.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load the pretrained causal language model before configuring updates.
        base_model = AutoModelForCausalLM.from_pretrained(model_name)

        if self.finetune_method == "lora":
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )
            self.encoder = get_peft_model(base_model, peft_config)
            self.encoder.print_trainable_parameters()
        elif self.finetune_method == "full":
            self.encoder = base_model
        elif self.finetune_method == "none":
            for param in base_model.parameters():
                param.requires_grad = False
            self.encoder = base_model
        else:
            raise ValueError(f"Invalid LLM fine-tuning method: {self.finetune_method}")

        # Optional learned pooling modules operate over token features.
        if self.pooling_type == 'attention':
            self.attention_pool = nn.Linear(self.encoder.config.hidden_size, 1)
        elif self.pooling_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.encoder.config.hidden_size, nhead=transformer_nhead)
            self.transformer_pool = nn.TransformerEncoder(encoder_layer, num_layers=transformer_nlayer)

        # Project pooled language features to the policy-conditioning size.
        self.projector = self._build_projection_head(self.encoder.config.hidden_size, hidden_size)
        

    def _build_projection_head(self, input_dim, output_dim):
        """Construct the configured linear or multilayer projection head."""
        # No learned projection is needed when the dimensions already match.
        if input_dim == output_dim:
            return nn.Identity()

        if self.projection_type == 'linear':
            return nn.Linear(input_dim, output_dim)
        elif self.projection_type == 'mlp':
            layers = []
            dims = [input_dim] + [output_dim] * self.num_projection_layers
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                if i < len(dims) - 2:
                    layers.append(nn.ReLU())
            return nn.Sequential(*layers)
        else:
            raise ValueError(f"Invalid projection type: {self.projection_type}")

    def forward(self, text_batch, already_encoded=False):
        """Pool and project raw text, or project cached LLM embeddings."""
        
        if not already_encoded:
            if isinstance(text_batch, str):
                text_batch = [text_batch]

            if self.pooling_type == 'eos':
                eos_token = self.tokenizer.eos_token
                text_batch = [text + " " + eos_token for text in text_batch]

            encoded = self.tokenizer(
                text_batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(self.device)

            outputs = self.encoder.model(**encoded, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # [B, T, D]

            if self.pooling_type == 'eos':
                eos_token_id = self.tokenizer.eos_token_id
                eos_positions = (encoded['input_ids'] == eos_token_id).int()
                eos_indices = eos_positions.argmax(dim=1)
                batch_indices = torch.arange(hidden_states.size(0), device=self.device)
                pooled = hidden_states[batch_indices, eos_indices]  # [B, D]

            elif self.pooling_type == 'mean':
                mask = encoded['attention_mask'].unsqueeze(-1)
                masked_hidden = hidden_states * mask
                pooled = masked_hidden.sum(dim=1) / mask.sum(dim=1)

            elif self.pooling_type == 'attention':
                attn_scores = self.attention_pool(hidden_states).squeeze(-1)  # [B, T]
                attn_scores = attn_scores.masked_fill(encoded['attention_mask'] == 0, float('-inf'))
                attn_weights = torch.softmax(attn_scores, dim=-1)
                pooled = (attn_weights.unsqueeze(-1) * hidden_states).sum(dim=1)

            elif self.pooling_type == 'transformer':
                mask = encoded['attention_mask'].unsqueeze(-1)
                pooled_hidden = self.transformer_pool(hidden_states.transpose(0, 1), src_key_padding_mask=~mask.squeeze(-1).bool())
                pooled_hidden = pooled_hidden.transpose(0, 1)  # [B, T, D]
                masked_hidden = pooled_hidden * mask
                pooled = masked_hidden.sum(dim=1) / mask.sum(dim=1)

            else:
                raise ValueError(f"Invalid pooling type: {self.pooling_type}")
            
        else:
            pooled = text_batch

        text_embedding = self.projector(pooled)
        
            
        if self.normalize_embeddings:
            text_embedding = F.normalize(text_embedding, dim=-1)

        return text_embedding





class PromptDecisionTransformer(nn.Module):
    """Encode trajectories with the prompt-augmented Decision Transformer."""

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            act_hidden_size,
            max_length,
            max_ep_len,
            normalize_embeddings,
            config
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.act_hidden_size = act_hidden_size
        self.normalize_embeddings = normalize_embeddings
        # This GPT-2 variant omits positional embeddings; explicit timestep embeddings
        # supply temporal position information below.
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.prompt_embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.prompt_embed_return = torch.nn.Linear(1, hidden_size)
        self.prompt_embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.prompt_embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # State and return heads are inherited; policy learning uses action features.
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_return = torch.nn.Linear(hidden_size, 1)
        
        if self.hidden_size != self.act_hidden_size:
            self.action_embedding = torch.nn.Linear(hidden_size, act_hidden_size)

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None, prompt=None):
        """Return per-step action features and a trajectory-level embedding."""
        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            # GPT attention uses one for valid tokens and zero for padding.
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # Embed returns, states, and actions with separate modality projections.
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # Add the same timestep embedding to each modality at that step.
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # Interleave tokens as (return, state, action) for autoregressive prediction.
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Repeat each timestep mask for its return, state, and action tokens.
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # Encode prompt trajectories with separate modality embeddings.
        if prompt is not None:
            prompt_states, prompt_actions, prompt_rewards, prompt_dones, prompt_returns_to_go, prompt_timesteps, prompt_attention_mask = prompt
            prompt_seq_length = prompt_states.shape[1]
            prompt_state_embeddings = self.prompt_embed_state(prompt_states)
            prompt_action_embeddings = self.prompt_embed_action(prompt_actions)
            if prompt_returns_to_go.shape[1] % 10 == 1:
                prompt_returns_embeddings = self.prompt_embed_return(prompt_returns_to_go[:,:-1])
            else:
                prompt_returns_embeddings = self.prompt_embed_return(prompt_returns_to_go)
            prompt_time_embeddings = self.prompt_embed_timestep(prompt_timesteps)

            prompt_state_embeddings = prompt_state_embeddings + prompt_time_embeddings
            prompt_action_embeddings = prompt_action_embeddings + prompt_time_embeddings
            prompt_returns_embeddings = prompt_returns_embeddings + prompt_time_embeddings

            prompt_stacked_inputs = torch.stack(
                (prompt_returns_embeddings, prompt_state_embeddings, prompt_action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(prompt_states.shape[0], 3 * prompt_seq_length, self.hidden_size)

            # to make the attention mask fit the stacked inputs, have to stack it as well
            prompt_stacked_attention_mask = torch.stack(
                (prompt_attention_mask, prompt_attention_mask, prompt_attention_mask), dim=1
            ).permute(0, 2, 1).reshape(prompt_states.shape[0], 3 * prompt_seq_length)

            # Prepend either one shared prompt or one prompt per batch item.
            if prompt_stacked_inputs.shape[1] == 3 * seq_length: # if only smaple one prompt
                prompt_stacked_inputs = prompt_stacked_inputs.reshape(1, -1, self.hidden_size)
                prompt_stacked_attention_mask = prompt_stacked_attention_mask.reshape(1, -1)
                stacked_inputs = torch.cat((prompt_stacked_inputs.repeat(batch_size, 1, 1), stacked_inputs), dim=1)
                stacked_attention_mask = torch.cat((prompt_stacked_attention_mask.repeat(batch_size, 1), stacked_attention_mask), dim=1)
            else: # if sample one prompt for each traj in batch
                stacked_inputs = torch.cat((prompt_stacked_inputs, stacked_inputs), dim=1)
                stacked_attention_mask = torch.cat((prompt_stacked_attention_mask, stacked_attention_mask), dim=1)
        # Pass continuous trajectory embeddings rather than token IDs.
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        if prompt is None:
            # reshape x so that the second dimension corresponds to the original
            # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
            x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        else:
            x = x.reshape(batch_size, -1, 3, self.hidden_size).permute(0, 2, 1, 3)

        # Prompts occupy the prefix; predictions retain only the current trajectory.
        return_preds = self.predict_return(x[:,2])[:, -seq_length:, :]  # predict next return given state and action
        state_preds = self.predict_state(x[:,2])[:, -seq_length:, :]    # predict next state given state and action
        action_trajectory_embeddings = x[:, 1,-seq_length:, :]
        trajectory_embedding =  x[:,2,-1,:]
        
        if self.hidden_size != self.act_hidden_size:
            action_trajectory_embeddings = self.action_embedding(action_trajectory_embeddings)
            trajectory_embedding = self.action_embedding(trajectory_embedding)
            
            
        if self.normalize_embeddings:
            action_trajectory_embeddings = F.normalize(action_trajectory_embeddings, dim=-1) 
            trajectory_embedding =  F.normalize(trajectory_embedding, dim=-1) 
            
            
            

        return state_preds, action_trajectory_embeddings, return_preds, trajectory_embedding


class DTPolicy(nn.Module):
    """Map trajectory embeddings directly to bounded actions."""

    def __init__(self, state_dim, act_dim, hidden_size, action_tanh=True):
        super().__init__()

        self.policy = nn.Sequential(
            *([nn.Linear(hidden_size, act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        

    def forward(self, embedding, state):
        
        return self.policy(embedding)
    
class FeatureExtractor(nn.Module):
    """Project raw states into the generated policy's feature space."""

    def __init__(self, state_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
        )
        self.hidden_size = hidden_size

    def forward(self, x):
        # Accept batched state sequences or individual batched states.
        if x.dim() == 3:
            B, S, D = x.shape
            x = x.reshape(B * S, D)
            h = self.net(x)  # (B*S, H)
            return h.reshape(B, S, -1)
        else:
            return self.net(x)  # (B, H)

def calculate_gain(name: str):
    """Return a supported PyTorch initialization gain."""
    safe = name.lower()
    if safe not in ['linear','conv1d','conv2d','conv3d','sigmoid','tanh','relu','leaky_relu','selu','gelu']:
        safe = 'tanh'
    return nn.init.calculate_gain(safe)



class PolicySpec:
    """Describe the dimensions of every compact-policy layer."""

    def __init__(self, state_feat_dim: int, act_dim: int, hidden_layers: List[int]):
        self.state_feat_dim = state_feat_dim
        self.act_dim = act_dim
        self.hidden_layers = hidden_layers

    @property
    def layer_dims(self) -> List[Tuple[int, int]]:
        """Return input/output dimensions for all generated layers."""
        dims = []
        in_dim = self.state_feat_dim
        for h in self.hidden_layers:
            dims.append((in_dim, h))
            in_dim = h
        dims.append((in_dim, self.act_dim))
        return dims

LayerParam = Dict[str, torch.Tensor]
ParamsType = List[LayerParam]

class HyperPolicy(nn.Module):
    """Execute a compact policy from hypernetwork-generated parameters.

    Parameters may be static per task or vary across sequence steps. Hidden
    layers use ReLU and the final action layer uses tanh.
    """
    def __init__(self, spec: PolicySpec, state_feature_extractor: nn.Module,
                 params: ParamsType):
        super().__init__()
        self.spec = spec
        self.state_feature_extractor = state_feature_extractor
        self.act = nn.ReLU()   # Hidden activation used by the released model.
        self.final_act = nn.Tanh()  # Bound generated-policy actions.
        self.params = params

    @torch.no_grad()
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Execute the instantiated policy on one state or state sequence."""
        x = self.state_feature_extractor(state)  # (B,S,H) or (B,H)

        if x.dim() == 2:
            # Individual states require a static parameter set.
            for i, layer in enumerate(self.params):
                W, b = layer['W'], layer['b']
                if W.dim() != 2:
                    raise ValueError("Time-varying params need (B,S,...) state features.")
                x = x @ W + b
                if i < len(self.params) - 1:
                    x = self.act(x)
            return self.final_act(x)

        # Apply static or time-varying parameters to a state sequence.
        for i, layer in enumerate(self.params):
            W, b = layer['W'], layer['b']
            if W.dim() == 2:
                x = torch.einsum('bsi,io->bso', x, W) + b
            elif W.dim() == 3:
                x = torch.einsum('bsi,sio->bso', x, W) + b.unsqueeze(0)
            else:
                raise ValueError("Param W must be rank 2 or 3.")
            if i < len(self.params) - 1:
                x = self.act(x)
        return self.final_act(x)

class HyperNetwork(nn.Module):
    """Generate all weights and biases of a compact task policy.

    The conditioning trunk uses tanh; generated policy hidden layers use ReLU;
    and generated actions are bounded with tanh.
    """
    def __init__(self, state_dim, act_dim, hidden_size,
                 hypernet_layers=[128, 128],
                 hidden_layers=[128, 128]):
        super().__init__()

        self.hyper_act = nn.Tanh()  # Hypernetwork trunk activation.
        self.actor_hidden_act = nn.ReLU()  # Compact-policy hidden activation.
        self.actor_final_act = nn.Tanh()   # Bound compact-policy actions.
        self.gain = calculate_gain('tanh')

        self.state_feature_extractor = FeatureExtractor(state_dim, hidden_size)

        # The hypernetwork trunk preserves all leading batch/sequence dimensions.
        self.hyper_layers = nn.ModuleList()
        cur_dim = hidden_size
        for next_sz in hypernet_layers:
            fc = nn.Linear(cur_dim, next_sz)
            self._init_normc_(fc.weight, gain=1.0)
            nn.init.constant_(fc.bias, 0)
            self.hyper_layers.append(fc)
            cur_dim = next_sz
        self.final_hyper_hidden_sz = cur_dim

        self.spec = PolicySpec(
            state_feat_dim=hidden_size,
            act_dim=act_dim,
            hidden_layers=hidden_layers,
        )

        # Each compact-policy layer receives its own weight and bias heads.
        self.hyper_W_heads = nn.ModuleList()
        self.hyper_b_heads = nn.ModuleList()
        for (in_dim, out_dim) in self.spec.layer_dims:
            w_sz = in_dim * out_dim
            b_sz = out_dim
            # Hidden heads use tanh gain; the action head uses unit gain.
            w_gain = 1.0 if (out_dim == self.spec.act_dim) else self.gain
            W = nn.Linear(self.final_hyper_hidden_sz, w_sz)
            b = nn.Linear(self.final_hyper_hidden_sz, b_sz)
            nn.init.orthogonal_(W.weight, gain=w_gain); nn.init.constant_(W.bias, 0)
            nn.init.zeros_(b.weight); nn.init.zeros_(b.bias)
            self.hyper_W_heads.append(W)
            self.hyper_b_heads.append(b)

    def _init_normc_(self, weight, gain=1.0):
        """Apply the column-normalized initialization used in the experiments."""
        nn.init.normal_(weight, mean=0, std=1)
        weight.data /= torch.sqrt(weight.pow(2).sum(0, keepdim=True) + 1e-8)
        weight.data *= gain

    def _hyper_features(self, embedding: torch.Tensor) -> torch.Tensor:
        """Transform conditioning embeddings through the hypernetwork trunk."""
        z = embedding  # (B,E) or (B,S,E)
        for layer in self.hyper_layers:
            z = self.hyper_act(layer(z))
        return z

    @torch.no_grad()
    def generate_params(self, embedding: torch.Tensor) -> Union[ParamsType, List[ParamsType]]:
        """
        If embedding is (B,E): static params.
        If embedding is (B,S,E): time-varying params (S,in,out).
        If B==1 -> return single ParamsType, else list per batch.
        """
        z = self._hyper_features(embedding)
        B = z.shape[0]

        def build(z_slice: torch.Tensor) -> ParamsType:
            params: ParamsType = []
            if z_slice.dim() == 1:
                # One static parameter set.
                for (in_dim, out_dim), W_head, b_head in zip(self.spec.layer_dims, self.hyper_W_heads, self.hyper_b_heads):
                    W = W_head(z_slice).view(in_dim, out_dim)
                    b = b_head(z_slice).view(out_dim)
                    params.append({'W': W, 'b': b})
            else:
                # One parameter set per sequence step.
                S = z_slice.shape[0]
                for (in_dim, out_dim), W_head, b_head in zip(self.spec.layer_dims, self.hyper_W_heads, self.hyper_b_heads):
                    W_flat = W_head(z_slice)    # (S, in*out)
                    b_vec  = b_head(z_slice)    # (S, out)
                    W = W_flat.view(S, in_dim, out_dim)
                    params.append({'W': W, 'b': b_vec})
            return params

        if B == 1:
            return build(z.squeeze(0))
        else:
            return [build(z[b]) for b in range(B)]

    @torch.no_grad()
    def build_policy(self, embedding: torch.Tensor) -> Union[nn.Module, List[nn.Module]]:
        """Instantiate inference policies from generated parameters."""
        params = self.generate_params(embedding)
        if isinstance(params, list) and params and isinstance(params[0], list):
            return [HyperPolicy(self.spec, self.state_feature_extractor, p) for p in params]
        return HyperPolicy(self.spec, self.state_feature_extractor, params)

    # Joint generation/application path used during optimization.
    def forward(self, embedding: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Generate parameters and apply them jointly during training."""
        z = self._hyper_features(embedding)  # (B,E) or (B,S,E)
        x = self.state_feature_extractor(state)  # (B,S,H) or (B,H)

        layer_dims = self.spec.layer_dims

        if x.dim() == 3:
            B, S, H = x.shape
            if z.dim() == 2:
                # One generated parameter set per batch item.
                for i, ((in_dim, out_dim), W_head, b_head) in enumerate(zip(layer_dims, self.hyper_W_heads, self.hyper_b_heads)):
                    W = W_head(z).view(B, in_dim, out_dim)
                    b = b_head(z).view(B, 1, out_dim)
                    x = torch.einsum('bsi,bio->bso', x, W) + b
                    if i < len(layer_dims) - 1:
                        x = self.actor_hidden_act(x)
                return self.actor_final_act(x)
            else:
                # One generated parameter set per sequence step.
                for i, ((in_dim, out_dim), W_head, b_head) in enumerate(zip(layer_dims, self.hyper_W_heads, self.hyper_b_heads)):
                    W = W_head(z).view(B, S, in_dim, out_dim)
                    b = b_head(z).view(B, S, out_dim)
                    x = torch.einsum('bsi,bsio->bso', x, W) + b
                    if i < len(layer_dims) - 1:
                        x = self.actor_hidden_act(x)
                return self.actor_final_act(x)
        else:
            # Individual states require one static conditioning embedding.
            if z.dim() != 2:
                raise ValueError("For (B,H) states, embedding must be (B,E).")
            for i, ((in_dim, out_dim), W_head, b_head) in enumerate(zip(layer_dims, self.hyper_W_heads, self.hyper_b_heads)):
                W = W_head(z).view(x.shape[0], in_dim, out_dim)
                b = b_head(z).view(x.shape[0], out_dim)
                x = torch.einsum('bi,bio->bo', x, W) + b
                if i < len(layer_dims) - 1:
                    x = self.actor_hidden_act(x)
            return self.actor_final_act(x)



class Predictor(nn.Module):
    """Combine trajectory and text encoders with shared or separate policies."""

    def __init__(
            self,
            args,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            **kwargs
    ):
        super().__init__()
        self.args = args
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        
        act_hidden_size = args.hn_embed_dim if self.args.hyper_network else hidden_size
        
        config = transformers.GPT2Config(vocab_size=1, n_embd=hidden_size, **kwargs)


        self.trajectory_encoder = PromptDecisionTransformer(state_dim,act_dim,hidden_size,act_hidden_size,max_length,max_ep_len,args.normalize_embeddings,config)
        
        self.policy = HyperNetwork(state_dim,act_dim,act_hidden_size,hypernet_layers=self.args.hypernet_layers,hidden_layers=self.args.policy_hidden_layers) if self.args.hyper_network else DTPolicy(state_dim,act_dim,act_hidden_size)

            
        if self.args.llm:
            self.text_encoder = TextEncoder(
                self.args.llm_model, act_hidden_size, args.llm_finetune_method, args.normalize_embeddings,
                self.args.llm_pooling_type, self.args.llm_projection_type,self.args.num_projection_layers,
                device=kwargs['device'], 
            )
            
            if self.args.dual_policy:
                self.llm_policy = HyperNetwork(state_dim,act_dim,act_hidden_size)
                
                
            if self.args.llm_preencoded:
                self.llm_projector = copy.deepcopy(self.text_encoder.projector)
                
            if self.args.llm_goal_prediction:
                self.llm_goal_head = nn.Sequential(
                    nn.Linear(act_hidden_size, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 3),
                )
                

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None, prompt=None, text=None):
        """Compute trajectory/text actions and embeddings for enabled losses."""
        
        state_preds, action_trajectory_embeddings, return_preds, trajectory_embedding = self.trajectory_encoder(states, actions, rewards, returns_to_go, timesteps, attention_mask, prompt)
        
        action_preds =  self.policy(action_trajectory_embeddings,states)
        goal_pred_llm = None
        if text is not None:
            if self.args.llm_preencoded:
                text = torch.stack(text,dim=0)
                text_embedding = self.llm_projector(text)
                if self.args.normalize_embeddings:
                    text_embedding = F.normalize(text_embedding, dim=-1)
            else:
                text_embedding = self.text_encoder(text)
            action_text_embeddings = text_embedding.unsqueeze(dim=1)
            length = action_trajectory_embeddings.shape[1]
            action_text_embeddings = action_text_embeddings.repeat(1,length,1)
            if self.args.dual_policy:
                action_preds_llm = self.llm_policy(action_text_embeddings,states)
            else:
                action_preds_llm = self.policy(action_text_embeddings,states)
                
            
            if self.args.llm_goal_prediction:
                goal_pred_llm = self.llm_goal_head(text_embedding) 
        else:
            text_embedding = torch.zeros_like(trajectory_embedding)
            action_preds_llm = torch.zeros_like(action_preds)
            
        

        return state_preds, action_preds, return_preds, action_trajectory_embeddings, trajectory_embedding, text_embedding, action_preds_llm, goal_pred_llm

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, prompt):
        """Predict the next action from a padded trajectory context."""
        # Rewards are retained for the inherited interface but are not encoded.

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # Left-pad each modality to the configured context length.
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None
            

        # The prompt is prepended inside the trajectory encoder.
        _, action_embeddings, _,_  = self.trajectory_encoder(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, prompt=prompt)
        
        
        action_preds =  self.policy(action_embeddings,states)
            

        return action_preds[0,-1]
