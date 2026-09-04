# Code backbone: Prompt-DT https://github.com/mxu34/prompt-dt
# Prompt-DT builds on Decision Transformer: https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

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

from vector_quantize_pytorch import FSQ

import copy

class TextEncoder(nn.Module):
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

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Base LLM
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
            # print("[LoRA] Using LoRA adapters.")
            self.encoder.print_trainable_parameters()
        elif self.finetune_method == "full":
            self.encoder = base_model
            # print("[Full FT] Fine-tuning all LLM parameters.")
        elif self.finetune_method == "none":
            for param in base_model.parameters():
                param.requires_grad = False
            self.encoder = base_model
            # print("[Frozen] No LLM parameters will be updated.")
        else:
            raise ValueError(f"Invalid LLM fine-tuning method: {self.finetune_method}")

        # Pooling layer
        if self.pooling_type == 'attention':
            self.attention_pool = nn.Linear(self.encoder.config.hidden_size, 1)
        elif self.pooling_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.encoder.config.hidden_size, nhead=transformer_nhead)
            self.transformer_pool = nn.TransformerEncoder(encoder_layer, num_layers=transformer_nlayer)

        # Projection head
        self.projector = self._build_projection_head(self.encoder.config.hidden_size, hidden_size)
        

    def _build_projection_head(self, input_dim, output_dim):
        # If dimensions match, return identity
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
        # config = transformers.GPT2Config(vocab_size=1, n_embd=hidden_size, **kwargs)

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)
        # change to parallelize mode for metaworld big model
        # self.transformer.parallelize()

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.prompt_embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.prompt_embed_return = torch.nn.Linear(1, hidden_size)
        self.prompt_embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.prompt_embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        # self.predict_action = nn.Sequential(
        #     *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        # )
        self.predict_return = torch.nn.Linear(hidden_size, 1)
        
        if self.hidden_size != self.act_hidden_size:
            self.action_embedding = torch.nn.Linear(hidden_size, act_hidden_size)

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None, prompt=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # process prompt the same as d-t
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

            # stacked_inputs add prompted sequence
            if prompt_stacked_inputs.shape[1] == 3 * seq_length: # if only smaple one prompt
                prompt_stacked_inputs = prompt_stacked_inputs.reshape(1, -1, self.hidden_size)
                prompt_stacked_attention_mask = prompt_stacked_attention_mask.reshape(1, -1)
                stacked_inputs = torch.cat((prompt_stacked_inputs.repeat(batch_size, 1, 1), stacked_inputs), dim=1)
                stacked_attention_mask = torch.cat((prompt_stacked_attention_mask.repeat(batch_size, 1), stacked_attention_mask), dim=1)
            else: # if sample one prompt for each traj in batch
                stacked_inputs = torch.cat((prompt_stacked_inputs, stacked_inputs), dim=1)
                stacked_attention_mask = torch.cat((prompt_stacked_attention_mask, stacked_attention_mask), dim=1)
        # we feed in the input embeddings (not word indices as in NLP) to the model
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

        # note here all the prompt are pre-append to x, but when return only return the last [:, -seq_length:, :] corresponding to batch data
        # get predictions
        return_preds = self.predict_return(x[:,2])[:, -seq_length:, :]  # predict next return given state and action
        state_preds = self.predict_state(x[:,2])[:, -seq_length:, :]    # predict next state given state and action
        # action_preds = self.predict_action(x[:,1])[:, -seq_length:, :]  # predict next action given state
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

    def __init__(self, state_dim, act_dim, hidden_size, action_tanh=True):
        super().__init__()

        self.policy = nn.Sequential(
            *([nn.Linear(hidden_size, act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        

    def forward(self, embedding, state):
        
        return self.policy(embedding)
    
class FeatureExtractor(nn.Module):
    def __init__(self, state_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
        )
        self.hidden_size = hidden_size

    def forward(self, x):
        # Accept (B,S,D) or (B,D)
        if x.dim() == 3:
            B, S, D = x.shape
            x = x.reshape(B * S, D)
            h = self.net(x)  # (B*S, H)
            return h.reshape(B, S, -1)
        else:
            return self.net(x)  # (B, H)

def calculate_gain(name: str):
    safe = name.lower()
    if safe not in ['linear','conv1d','conv2d','conv3d','sigmoid','tanh','relu','leaky_relu','selu','gelu']:
        safe = 'tanh'
    return nn.init.calculate_gain(safe)



class PolicySpec:
    def __init__(self, state_feat_dim: int, act_dim: int, hidden_layers: List[int]):
        self.state_feat_dim = state_feat_dim
        self.act_dim = act_dim
        self.hidden_layers = hidden_layers

    @property
    def layer_dims(self) -> List[Tuple[int, int]]:
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
    """
    params per layer:
      static: {'W': (in,out), 'b': (out,)}
      time-varying: {'W': (S,in,out), 'b': (S,out)}
    Actor hidden: ReLU; final: tanh (MATCH OLD).
    """
    def __init__(self, spec: PolicySpec, state_feature_extractor: nn.Module,
                 params: ParamsType):
        super().__init__()
        self.spec = spec
        self.state_feature_extractor = state_feature_extractor
        self.act = nn.ReLU()   # MATCH OLD
        self.final_act = nn.Tanh()  # MATCH OLD
        self.params = params

    @torch.no_grad()
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.state_feature_extractor(state)  # (B,S,H) or (B,H)

        if x.dim() == 2:
            # (B,H) requires static params
            for i, layer in enumerate(self.params):
                W, b = layer['W'], layer['b']
                if W.dim() != 2:
                    raise ValueError("Time-varying params need (B,S,...) state features.")
                x = x @ W + b
                if i < len(self.params) - 1:
                    x = self.act(x)
            return self.final_act(x)

        # (B,S,H)
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
    """
    Decoupled version aligned to OLD behavior.
    - Hyper trunk activation: tanh
    - Actor hidden: ReLU
    - Actor final: tanh
    """
    def __init__(self, state_dim, act_dim, hidden_size,
                 hypernet_layers=[128, 128],
                 hidden_layers=[128, 128]):
        super().__init__()

        self.hyper_act = nn.Tanh()  # MATCH OLD
        self.actor_hidden_act = nn.ReLU()  # MATCH OLD
        self.actor_final_act = nn.Tanh()   # MATCH OLD
        self.gain = calculate_gain('tanh')

        self.state_feature_extractor = FeatureExtractor(state_dim, hidden_size)

        # Hyper trunk (operates on last dim; preserves leading dims)
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

        # Heads to map hyper features -> flattened weights/biases
        self.hyper_W_heads = nn.ModuleList()
        self.hyper_b_heads = nn.ModuleList()
        for (in_dim, out_dim) in self.spec.layer_dims:
            w_sz = in_dim * out_dim
            b_sz = out_dim
            # last layer gain=1.0, else tanh gain
            w_gain = 1.0 if (out_dim == self.spec.act_dim) else self.gain
            W = nn.Linear(self.final_hyper_hidden_sz, w_sz)
            b = nn.Linear(self.final_hyper_hidden_sz, b_sz)
            nn.init.orthogonal_(W.weight, gain=w_gain); nn.init.constant_(W.bias, 0)
            nn.init.zeros_(b.weight); nn.init.zeros_(b.bias)
            self.hyper_W_heads.append(W)
            self.hyper_b_heads.append(b)

    def _init_normc_(self, weight, gain=1.0):
        nn.init.normal_(weight, mean=0, std=1)
        weight.data /= torch.sqrt(weight.pow(2).sum(0, keepdim=True) + 1e-8)
        weight.data *= gain

    def _hyper_features(self, embedding: torch.Tensor) -> torch.Tensor:
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
                # static
                for (in_dim, out_dim), W_head, b_head in zip(self.spec.layer_dims, self.hyper_W_heads, self.hyper_b_heads):
                    W = W_head(z_slice).view(in_dim, out_dim)
                    b = b_head(z_slice).view(out_dim)
                    params.append({'W': W, 'b': b})
            else:
                # time-varying
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
        params = self.generate_params(embedding)
        if isinstance(params, list) and params and isinstance(params[0], list):
            return [HyperPolicy(self.spec, self.state_feature_extractor, p) for p in params]
        return HyperPolicy(self.spec, self.state_feature_extractor, params)

    # Training-time joint forward (aligned to OLD behavior)
    def forward(self, embedding: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        z = self._hyper_features(embedding)  # (B,E) or (B,S,E)
        x = self.state_feature_extractor(state)  # (B,S,H) or (B,H)

        layer_dims = self.spec.layer_dims

        if x.dim() == 3:
            B, S, H = x.shape
            if z.dim() == 2:
                # static per batch
                for i, ((in_dim, out_dim), W_head, b_head) in enumerate(zip(layer_dims, self.hyper_W_heads, self.hyper_b_heads)):
                    W = W_head(z).view(B, in_dim, out_dim)
                    b = b_head(z).view(B, 1, out_dim)
                    x = torch.einsum('bsi,bio->bso', x, W) + b
                    if i < len(layer_dims) - 1:
                        x = self.actor_hidden_act(x)
                return self.actor_final_act(x)
            else:
                # time-varying per step
                for i, ((in_dim, out_dim), W_head, b_head) in enumerate(zip(layer_dims, self.hyper_W_heads, self.hyper_b_heads)):
                    W = W_head(z).view(B, S, in_dim, out_dim)
                    b = b_head(z).view(B, S, out_dim)
                    x = torch.einsum('bsi,bsio->bso', x, W) + b
                    if i < len(layer_dims) - 1:
                        x = self.actor_hidden_act(x)
                return self.actor_final_act(x)
        else:
            # (B,H) states require static z
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
                
        if self.args.quantized_embed:
            quant_level = self.args.quant_level
            quant_dim = self.args.hn_embed_dim
            self.quantizer = FSQ(
                levels = [quant_level]*quant_dim, return_indices=False
            )
            


    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None, prompt=None, text=None):
        
        state_preds, action_trajectory_embeddings, return_preds, trajectory_embedding = self.trajectory_encoder(states, actions, rewards, returns_to_go, timesteps, attention_mask, prompt)
        
        if self.args.quantized_embed:
            action_trajectory_embeddings = self.quantizer(action_trajectory_embeddings)[0]
            if self.args.quantized_embed_for_loss:
                trajectory_embedding = self.quantizer(trajectory_embedding.unsqueeze(1))[0].squeeze(1)
            
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
            if self.args.quantized_embed:
                action_text_embeddings = self.quantizer(action_text_embeddings)[0]
                if self.args.quantized_embed_for_loss:
                    text_embedding = self.quantizer(text_embedding)[0].squeeze(1)
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
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
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
            

        # Note: prompt within kwargs
        _, action_embeddings, _,_  = self.trajectory_encoder(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, prompt=prompt)
        
        
        if self.args.quantized_embed:
            action_embeddings = self.quantizer(action_embeddings)[0]
        
        action_preds =  self.policy(action_embeddings,states)
            

        return action_preds[0,-1]

    
    
