# Code backbone: Prompt-DT https://github.com/mxu34/prompt-dt
# Prompt-DT builds on Decision Transformer: https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

"""Episode-evaluation helpers for trajectory- and text-conditioned policies.

The trajectory-conditioned evaluators maintain the history consumed by
Decision Transformer/Prompt-DT. The text-conditioned evaluator executes the
compact policy instantiated by TENET directly from a task description.
"""

import numpy as np
import torch
import torch.nn.functional as F
import time
import random

def prompt_evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
        prompt=None,
        no_r=False,
        no_rtg=False,
        no_state_normalize=False,
):
    """Evaluate a trajectory-conditioned policy with the legacy Gym API."""

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    # Keep the complete rollout history on the target device. At every step,
    # the final action and reward entries are placeholders filled after acting.
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # Append placeholders so the model predicts the current action.
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        if no_state_normalize:
            action = model.get_action(
                states.to(dtype=torch.float32),
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return=target_return,
                prompt=prompt
            )
        else:
            action = model.get_action(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return=target_return,
                prompt=prompt
            )
            
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward
        if no_r:
            rewards[-1] = 0.0

        episode_return += reward
        episode_length += 1
            
        if done:
            break

    return episode_return, episode_length


def prompt_evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        prompt=None,
        no_r=False,
        no_rtg=False,
        no_state_normalize=False
    ):
    """Evaluate Prompt-DT while updating return-to-go after each action."""

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)
    
    state, _ = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # Keep the complete rollout history on the target device. At every step,
    # the final action and reward entries are placeholders filled after acting.
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    success = False
    for t in range(max_ep_len):
        # Append placeholders so the model predicts the current action.
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        if no_state_normalize:
            action = model.get_action(
                states.to(dtype=torch.float32),
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
                prompt=prompt,
            )
        else:
            action = model.get_action(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
                prompt=prompt,
            )
            
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, terminate, truncate, infos = env.step(action)
        done = terminate or truncate

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = torch.tensor(reward, dtype=torch.float32, device=device)
        if no_r:
            rewards[-1] = 0.0

        # Standard evaluation subtracts observed reward from the desired
        # return; delayed mode holds the desired return fixed.
        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        if no_rtg:
            target_return = torch.ones_like(target_return)*ep_return
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1
        

        infos['episode_length'] = episode_length

        if infos.get("success", 0.0) > 0:
            success = True
            break
        if done:
            break
    return episode_return, success, infos



def prompt_evaluate_episode_llm(
        env,
        policy,
        max_ep_len=1000,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        text=None,
        no_state_normalize=False
    ):
    """Evaluate the compact policy instantiated from a text embedding."""

    
    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state, _ = env.reset()

    episode_return, episode_length = 0, 0
    success = False
    for t in range(max_ep_len):
        state = torch.from_numpy(state).to(device=device, dtype=torch.float32).unsqueeze(0)

        if no_state_normalize:
            action = policy(state)
        else:
            action = policy((state - state_mean) / state_std)
            
        action = action.detach().cpu().numpy()[0]

        state, reward, terminate, truncate, infos = env.step(action)
        done = terminate or truncate

        episode_return += reward
        episode_length += 1
        

        infos['episode_length'] = episode_length

        if infos.get("success", 0.0) > 0:
            success = True
            break
        if done:
            break
    return episode_return, success, infos
