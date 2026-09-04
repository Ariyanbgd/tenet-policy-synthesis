# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

import numpy as np
import torch
import time
from wandb import env
from .prompt_utils import flatten_prompt
import copy
import torch.nn.functional as F
import random
import torch.nn as nn
import os

class PromptSequenceTrainer:

    def __init__(self, args, model, optimizer, batch_size, get_batch, loss_fn,
                 scheduler=None, eval_fns=None, get_prompt=None, get_prompt_batch=None, device=None):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.get_prompt = get_prompt
        self.prompt = self.get_prompt() # sample prompt data when initialization
        self.get_prompt_batch = get_prompt_batch
        self.device = device
        self.save = False
        
        if self.args.use_embedding_discriminator:
            self.discriminator = Discriminator(self.args.hn_embed_dim)
            self.discriminator.to(device)
            self.d_optimizer = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )

        self.start_time = time.time()


    def pure_train_iteration_mix(self, num_steps, no_prompt=False, no_llm=True):

        train_action_losses = []
        train_embedding_consistency_losses = []
        train_action_llm_losses = []
        train_contrastive_losses = []
        train_embedding_mse_losses = []
        train_discriminator_losses = []
        train_llm_goal_pred_losses = []
        train_text_contrastive_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        for _ in range(num_steps):
            train_action_loss, train_embedding_consistency_loss, train_action_llm_loss, train_contrastive_loss, train_embedding_mse_loss, train_discriminator_loss, train_llm_goal_pred_loss, train_text_contrastive_loss = self.train_step_mix(no_prompt,no_llm)
            train_action_losses.append(train_action_loss)
            train_embedding_consistency_losses.append(train_embedding_consistency_loss)
            train_action_llm_losses.append(train_action_llm_loss)
            train_contrastive_losses.append(train_contrastive_loss)
            train_embedding_mse_losses.append(train_embedding_mse_loss)
            train_discriminator_losses.append(train_discriminator_loss)
            train_llm_goal_pred_losses.append(train_llm_goal_pred_loss)
            train_text_contrastive_losses.append(train_text_contrastive_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start
        logs['training/train_action_loss_mean'] = np.mean(train_action_losses)
        logs['training/train_action_loss_std'] = np.std(train_action_losses)
        logs['training/train_embedding_consistency_loss_mean'] = np.mean(train_embedding_consistency_losses)
        logs['training/train_embedding_consistency_loss_std'] = np.std(train_embedding_consistency_losses)
        
        logs['training/train_action_llm_loss_mean'] = np.mean(train_action_llm_losses)
        logs['training/train_action_llm_loss_std'] = np.std(train_action_llm_losses)
        
        logs['training/train_contrastive_loss_mean'] = np.mean(train_contrastive_losses)
        logs['training/train_contrastive_loss_std'] = np.std(train_contrastive_losses)
        logs['training/train_embedding_mse_loss_mean'] = np.mean(train_embedding_mse_losses)
        logs['training/train_embedding_mse_loss_std'] = np.std(train_embedding_mse_losses)
        logs['training/train_discriminator_loss_mean'] = np.mean(train_discriminator_losses)
        logs['training/train_discriminator_loss_std'] = np.std(train_discriminator_losses)
        logs['training/train_llm_goal_pred_loss_mean'] = np.mean(train_llm_goal_pred_losses)
        logs['training/train_llm_goal_pred_loss_std'] = np.std(train_llm_goal_pred_losses)
        logs['training/train_text_contrastive_loss_mean'] = np.mean(train_text_contrastive_losses)
        logs['training/train_text_contrastive_loss_std'] = np.std(train_text_contrastive_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        return logs


    def train_step_mix(self, no_prompt=False, no_llm=True):
        prompt, batch, text, goal = self.get_prompt_batch()
        states, actions, rewards, dones, rtg, timesteps, attention_mask = batch
        action_target = torch.clone(actions)

        if no_prompt:
            prompt = None
        if no_llm:
            text = None

        state_preds, action_preds, reward_preds, action_trajectory_embeddings, trajectory_embedding, text_embedding, action_preds_llm, goal_pred = self.model.forward(
            states, actions, rewards, rtg[:, :-1], timesteps, attention_mask=attention_mask, prompt=prompt, text=text
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_preds_llm = action_preds_llm.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = 0

        # --- Action prediction loss from trajectory ---
        action_loss = self.loss_fn(None, action_preds, None, None, action_target, None)
        loss += action_loss

        # --- Temporal embedding consistency loss ---
        action_trajectory_embeddings_diff = action_trajectory_embeddings[:, 1:, :] - action_trajectory_embeddings[:, :-1, :]
        embedding_consistency_loss = torch.mean(action_trajectory_embeddings_diff.pow(2))
        if self.args.consistency_regularizer:
            loss += self.args.cr_coeff * embedding_consistency_loss

        action_llm_loss = torch.zeros_like(action_loss)
        contrastive_loss = torch.zeros_like(action_loss)
        embedding_mse_loss = torch.zeros_like(action_loss)
        discriminator_loss = torch.zeros_like(action_loss)
        llm_goal_pred_loss = torch.zeros_like(action_loss)
        text_contrastive_loss = torch.zeros_like(action_loss)

        if not no_llm:
            # --- LLM-predicted action loss ---
            action_llm_loss = self.loss_fn(None, action_preds_llm, None, None, action_target, None)
            loss += action_llm_loss
            
            if self.args.use_embedding_contrastive:
                # --- Normalize embeddings if requested ---
                if not self.args.normalize_embeddings:
                    text_embedding = F.normalize(text_embedding, dim=-1)
                    trajectory_embedding = F.normalize(trajectory_embedding, dim=-1)

                # --- Contrastive Loss ---
                logits = torch.matmul(trajectory_embedding, text_embedding.T) / self.args.contrastive_temperature
                labels = torch.arange(logits.shape[0], device=logits.device)
                loss_i2t = F.cross_entropy(logits, labels)
                loss_t2i = F.cross_entropy(logits.T, labels)
                contrastive_loss = (loss_i2t + loss_t2i) / 2
                loss += self.args.contrastive_coeff * contrastive_loss

                if self.args.use_text_embedding_contrastive:
                    logits_tt = torch.matmul(text_embedding, text_embedding.T) / self.args.contrastive_temperature
                    labels_tt = torch.arange(logits_tt.shape[0], device=logits.device)
                    text_contrastive_loss = F.cross_entropy(logits_tt, labels_tt)
                    loss += self.args.text_contrastive_coeff * text_contrastive_loss

            # === MSE Embedding Alignment Loss ===
            if self.args.use_embedding_mse:
                embedding_mse_loss = F.mse_loss(trajectory_embedding, text_embedding)
                loss += self.args.embedding_mse_coeff * embedding_mse_loss

            # === Discriminator Loss ===
            if self.args.use_embedding_discriminator:
                # Train discriminator: detach embeddings
                z_t = trajectory_embedding.detach()
                z_s = text_embedding.detach()
                logits_D = torch.cat([self.discriminator(z_t), self.discriminator(z_s)], dim=0).squeeze()
                labels_D = torch.cat([torch.ones(z_t.size(0)), torch.zeros(z_s.size(0))], dim=0).to(logits_D.device)
                d_loss = F.binary_cross_entropy_with_logits(logits_D, labels_D)
                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Train encoders to fool discriminator
                logits_fake = torch.cat([self.discriminator(trajectory_embedding), self.discriminator(text_embedding)], dim=0).squeeze()
                fool_labels = torch.cat([torch.zeros(trajectory_embedding.size(0)), torch.ones(text_embedding.size(0))], dim=0).to(logits_fake.device)
                discriminator_loss = F.binary_cross_entropy_with_logits(logits_fake, fool_labels)
                loss += self.args.discriminator_coeff * discriminator_loss
                
            if self.args.llm_goal_prediction and (goal_pred is not None) and (goal is not None):
                goal = goal.to(goal_pred.device)
                llm_goal_pred_loss = F.mse_loss(goal_pred, goal)
                loss += self.args.llm_goal_pred_coeff * llm_goal_pred_loss

        # === Backward ===
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        # === Logging ===
        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds - action_target) ** 2).detach().cpu().item()
            self.diagnostics['training/action_error_llm'] = torch.mean((action_preds_llm - action_target) ** 2).detach().cpu().item()


        return action_loss.detach().cpu().item(), embedding_consistency_loss.detach().cpu().item(), action_llm_loss.detach().cpu().item(), contrastive_loss.detach().cpu().item(), embedding_mse_loss.detach().cpu().item(), discriminator_loss.detach().cpu().item(), llm_goal_pred_loss.detach().cpu().item(), text_contrastive_loss.detach().cpu()


    def finetune_eval_iteration_multienv(self, get_prompt, get_batch, test_prompt_trajectories_list, test_trajectories_list, 
                                eval_episodes, env_name_list, info, 
                                variant, env_list, iter_num=0, print_logs=False, 
                                no_prompt=False, group='test-finetune',
                                finetune_opt=False):
        print('evaluate at tasks: ', env_name_list)
        logs = dict()
        print('start evaluating...')
        self.model.eval()
        self.current_model_dict = copy.deepcopy(self.model.state_dict())

        eval_start = time.time()
        if finetune_opt:
            fintune_optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=variant['finetune_lr'],
                weight_decay=1e-4,
            )
        else:
            fintune_optimizer = None
        for env_id, env_name in enumerate(env_name_list):
            self.eval_fns = [eval_episodes(tar, info[env_name], variant, env_list[env_id], env_name) for tar in info[env_name]['env_targets']]
            self.get_prompt = get_prompt(test_prompt_trajectories_list[env_id], info[env_name], variant)
            self.get_batch = get_batch(test_trajectories_list[env_id], info[env_name], variant)
            if not no_prompt:
                self.prompt = flatten_prompt(self.get_prompt(), batch_size=1) # one prompt for the whole batch now
            else:
                self.prompt = None

            self.model.train()
            # finetune the model on the data for this task 
            finetune_losses = []
            for _ in range(variant['finetune_steps']):
                finetune_loss = self.train_step(
                    batch_size_overwrite=variant['finetune_batch_size'],
                    optimizer=fintune_optimizer)
                finetune_losses.append(finetune_loss)

            self.model.eval()
            # need to sample eval_fn and prompt together 
            for eval_fn in self.eval_fns:
                outputs = eval_fn(self.model, prompt=self.prompt)
                for k, v in outputs.items():
                    logs[f'{group}-evaluation/{k}'] = v
            
            self.model.load_state_dict(self.current_model_dict)

        logs['time/evaluation'] = time.time() - eval_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs


    def train_step(self, batch_size_overwrite=None, optimizer=None):
        if batch_size_overwrite is not None:
            states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(batch_size_overwrite)
        else:
            states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)
        # print('state shape after batch', states.shape)
        # print('self.batch_size', self.batch_size)
        # print('enter train step')
        # input()
        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask, prompt=self.prompt
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        if optimizer is None:
            self.optimizer.zero_grad()
        else:
            optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)

        if optimizer is None:
            self.optimizer.step()
        else:
            optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()


    def eval_iteration_multienv(self, get_prompt, prompt_trajectories_list, eval_episodes, eval_episodes_llm, env_name_list,
                                env_name_grouped, info, variant, base_env, iter_num=0,
                                print_logs=False, no_prompt=False, group='test',
                                log_per_task_instance=False):
        print(f"[Eval-{group}] Evaluating {len(env_name_list)} task instances...")
        logs = dict()
        self.model.eval()
        eval_start = time.time()

        all_envs_avg_returns = []
        all_envs_avg_successes = []
        all_envs_avg_returns_llm = []
        all_envs_avg_successes_llm = []

        for env_id, env_name in enumerate(env_name_list):

            # ---- Build env ----

            task_name = info[env_name]["task_name"]
            task_idx = info[env_name]["task_idx"]

            env = base_env.train_classes[task_name]()
            task = base_env.train_tasks[task_idx]
            env.set_task(task)


            # ---- Build prompt once per env (works for both branches) ----
            get_prompt_fn = get_prompt(prompt_trajectories_list[env_id], info[env_name], variant)
            prompt = None if no_prompt else flatten_prompt(get_prompt_fn(), batch_size=1)

            # Placeholders
            dt_policy_ret = dt_policy_succ = None
            llm_ret = llm_succ = None

            # ---- Policy branch (non-LLM or full eval) ----
            run_dt_policy = (not self.args.llm) or (self.args.llm and not self.args.only_llm_evaluation)
            if run_dt_policy:
                eval_fns = [
                    eval_episodes(tar, info[env_name], variant, env, env_name)
                    for tar in info[env_name]['env_targets']
                ]
                per_target_returns, per_target_successes = [], []
                for eval_fn in eval_fns:
                    _, avg_return, avg_success = eval_fn(self.model, prompt=prompt)
                    per_target_returns.append(avg_return)
                    per_target_successes.append(avg_success)
                # Aggregate to one number per env
                dt_policy_ret = float(np.mean(per_target_returns)) if per_target_returns else None
                dt_policy_succ = float(np.mean(per_target_successes)) if per_target_successes else None

            # ---- LLM branch ----
            if self.args.llm:
                eval_fn_llm = eval_episodes_llm(info[env_name], variant, env, env_name)
                text = info[env_name]['text_embeddings'] if self.args.llm_preencoded else info[env_name]['text_descriptions']
                llm_ret, llm_succ = eval_fn_llm(self.model, text=text)

            # ---- Fallbacks: copy whichever is missing ----
            if dt_policy_ret is None: dt_policy_ret = llm_ret
            if dt_policy_succ is None: dt_policy_succ = llm_succ
            if llm_ret is None: llm_ret = dt_policy_ret
            if llm_succ is None: llm_succ = dt_policy_succ

            # ---- Append parallel arrays (always populated now) ----
            all_envs_avg_returns.append(dt_policy_ret)
            all_envs_avg_successes.append(dt_policy_succ)
            all_envs_avg_returns_llm.append(llm_ret)
            all_envs_avg_successes_llm.append(llm_succ)




            if log_per_task_instance:
                logs[f'{group}/task_instance/{env_name}/return'] = float(dt_policy_ret)
                logs[f'{group}/task_instance/{env_name}/success'] = float(dt_policy_succ)
                logs[f'{group}/task_instance/{env_name}/return_llm'] = float(llm_ret)
                logs[f'{group}/task_instance/{env_name}/success_llm'] = float(llm_succ)

        # === Log overall averages ===
        logs[f'{group}/avg/return'] = float(np.mean(all_envs_avg_returns))
        logs[f'{group}/avg/success'] = float(np.mean(all_envs_avg_successes))
        logs[f'{group}/avg/return_llm'] = float(np.mean(all_envs_avg_returns_llm))
        logs[f'{group}/avg/success_llm'] = float(np.mean(all_envs_avg_successes_llm))

        # === Per-task-type average ===
        for task_type, indices in env_name_grouped.items():
            returns = [all_envs_avg_returns[i] for i in indices]
            successes = [all_envs_avg_successes[i] for i in indices]
            returns_llm = [all_envs_avg_returns_llm[i] for i in indices]
            successes_llm = [all_envs_avg_successes_llm[i] for i in indices]

            logs[f'{group}/task/{task_type}/avg_return'] = float(np.mean(returns))
            logs[f'{group}/task/{task_type}/avg_success'] = float(np.mean(successes))
            logs[f'{group}/task/{task_type}/avg_return_llm'] = float(np.mean(returns_llm))
            logs[f'{group}/task/{task_type}/avg_success_llm'] = float(np.mean(successes_llm))

        # Timing and diagnostics
        logs['time/evaluation'] = time.time() - eval_start
        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num} Evaluation Summary:')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs


 
    def save_model(self, save_path):
        model_path = os.path.join(save_path, 'model.pt')
        torch.save(self.model,model_path)  # model save
    
    def save_text_encoder(self, save_path):
        model_path = os.path.join(save_path, 'model_text_encoder.pt')
        torch.save(self.model.text_encoder,model_path)  # model save



class Discriminator(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # outputs logits
        )

    def forward(self, x):
        return self.net(x)