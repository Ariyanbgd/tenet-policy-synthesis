import numpy as np
import json, pickle, random, os, torch
from collections import namedtuple
from .prompt_evaluate_episodes import prompt_evaluate_episode, prompt_evaluate_episode_rtg, prompt_evaluate_episode_llm
import random
from collections import defaultdict

import metaworld
import math
import torch.nn.functional as F


MT50_TASK_DESCRIPTIONS = {
    "assembly-v3": ["Place the nut onto the peg."],
    "basketball-v3": ["Dunk the basketball into the hoop."],
    "bin-picking-v3": ["Move the object from one bin to another."],
    "box-close-v3": ["Close the box with its lid."],
    "button-press-topdown-v3": ["Press the button from the top."],
    "button-press-topdown-wall-v3": ["Reach over the wall and press the button from the top."],
    "button-press-v3": ["Press the button."],
    "button-press-wall-v3": ["Reach over the wall and press the button."],
    "coffee-button-v3": ["Push the button on the coffee machine."],
    "coffee-pull-v3": ["Pull the mug away from the coffee machine."],
    "coffee-push-v3": ["Push the mug under the coffee machine."],
    "dial-turn-v3": ["Turn the dial 180 degrees."],
    "disassemble-v3": ["Remove the nut from the peg."],
    "door-close-v3": ["Close the door."],
    "door-lock-v3": ["Lock the door by turning the handle."],
    "door-open-v3": ["Open the door."],
    "door-unlock-v3": ["Unlock the door by turning the handle."],
    "hand-insert-v3": ["Insert the gripper into the opening."],
    "drawer-close-v3": ["Close the drawer."],
    "drawer-open-v3": ["Open the drawer."],
    "faucet-open-v3": ["Turn the faucet counter-clockwise to open it."],
    "faucet-close-v3": ["Turn the faucet clockwise to close it."],
    "hammer-v3": ["Use the hammer to drive the nail."],
    "handle-press-side-v3": ["Press the side handle down."],
    "handle-press-v3": ["Press the handle down."],
    "handle-pull-side-v3": ["Pull the side handle up."],
    "handle-pull-v3": ["Pull the handle up."],
    "lever-pull-v3": ["Pull the lever down."],
    "pick-place-wall-v3": ["Pick up the object and place it over the wall."],
    "pick-out-of-hole-v3": ["Pick the object out of the hole."],
    "pick-place-v3": ["Pick up the object and place it at the target."],
    "plate-slide-v3": ["Slide the plate into the cabinet."],
    "plate-slide-side-v3": ["Slide the plate into the cabinet from the side."],
    "plate-slide-back-v3": ["Pull the plate out of the cabinet."],
    "plate-slide-back-side-v3": ["Pull the plate out of the cabinet from the side."],
    "peg-insert-side-v3": ["Insert the peg into the side hole."],
    "peg-unplug-side-v3": ["Remove the peg from the side hole."],
    "soccer-v3": ["Kick the ball into the goal."],
    "stick-push-v3": ["Push the box using the stick."],
    "stick-pull-v3": ["Pull the box using the stick."],
    "push-v3": ["Push the object to the target."],
    "push-wall-v3": ["Push the object to the target past the wall."],
    "push-back-v3": ["Pull the object to the target."],
    "reach-v3": ["Reach the target position."],
    "reach-wall-v3": ["Reach the target position behind the wall."],
    "shelf-place-v3": ["Place the object on the shelf."],
    "sweep-into-v3": ["Sweep the object into the hole."],
    "sweep-v3": ["Sweep the object off the table."],
    "window-open-v3": ["Open the window."],
    "window-close-v3": ["Close the window."]
}

MT10_TASK_DESCRIPTIONS = {
    "reach-v3": ["Reach the target position."],
    "push-v3": ["Push the object to the target."],
    "pick-place-v3": ["Pick up the object and place it at the target."],
    "door-open-v3": ["Open the door."],
    "drawer-open-v3": ["Open the drawer."],
    "drawer-close-v3": ["Close the drawer."],
    "button-press-topdown-v3": ["Press the button from the top."],
    "peg-insert-side-v3": ["Insert the peg into the side hole."],
    "window-open-v3": ["Open the window."],
    "window-close-v3": ["Close the window."]
}

def describe_pick_and_place(goal):
    tgt_pos = goal[3:]

    templates = [
        "Pick the object and place it at ({tx:.3f}, {ty:.3f}, {tz:.3f})."
    ]

    return [t.format(
        tx=tgt_pos[0], ty=tgt_pos[1], tz=tgt_pos[2]
    ) for t in templates]

""" constructing envs """

def describe_reach(goal):
    # Assumes the last 3 elements are the target position (x, y, z)
    tx, ty, tz = goal[-3:]

    templates = [
        "Reach the target at ({tx:.3f}, {ty:.3f}, {tz:.3f})."
    ]

    return [t.format(tx=tx, ty=ty, tz=tz) for t in templates]

def gen_env(env_name, base_env):
    if env_name.startswith('mt50-v3-'):
        # Example: 'mt50-v3-reach-v2-0'
        parts = env_name.split('-')
        task_name = '-'.join(parts[2:-1])  # 'reach-v2'
        task_idx = int(parts[-1])          # 0
        
        # Initialize MT50 benchmark with fixed seed
        mt50 = base_env

        # Create environment instance
        env = mt50.train_classes[task_name]()
        
        # Retrieve and set task
        task = mt50.train_tasks[task_idx]
        env.set_task(task)

        # Default return values
        goal = None
        max_ep_len = 500 
        env_targets = [650]
        scale = 650.0
        text_descriptions = MT50_TASK_DESCRIPTIONS.get(task_name, f"Perform the task: {task_name}")
    elif env_name.startswith('mt50-ml45split-v3-'):
        # Example: 'mt50-ml45split-v3-reach-v2-0'
        parts = env_name.split('-')
        task_name = '-'.join(parts[3:-1])  # 'reach-v2'
        task_idx = int(parts[-1])          # 0
        
        # Initialize MT50 benchmark with fixed seed
        mt50 = base_env

        # Create environment instance
        env = mt50.train_classes[task_name]()
        
        # Retrieve and set task
        task = mt50.train_tasks[task_idx]
        env.set_task(task)

        # Default return values
        goal = None
        max_ep_len = 500 
        env_targets = [650]
        scale = 650.0
        text_descriptions = MT50_TASK_DESCRIPTIONS.get(task_name, f"Perform the task: {task_name}")
    elif env_name.startswith('mt10-v3-'):
        parts = env_name.split('-')
        task_name = '-'.join(parts[2:-1])  # 'reach-v2'
        task_idx = int(parts[-1])          # 0
        
        mt10 = base_env

        # Create environment instance
        env = mt10.train_classes[task_name]()
        
        # Retrieve and set task
        task = mt10.train_tasks[task_idx]
        env.set_task(task)

        # Default return values
        goal = None
        max_ep_len = 500 
        env_targets = [650]
        scale = 650.0
        text_descriptions = MT10_TASK_DESCRIPTIONS.get(task_name, f"Perform the task: {task_name}")
    
    elif env_name.startswith('ml1-v3-pick-place-v3-'):
        parts = env_name.split('-')
        task_name = '-'.join(parts[2:-2])
        task_idx = int(parts[-1])          # 0
        
        # Create environment instance
        env = base_env.train_classes[task_name]()
        
        # Retrieve and set task
        task = base_env.train_tasks[task_idx]
        env.set_task(task)

        # Default return values
        goal = env._last_rand_vec
        max_ep_len = 500 
        env_targets = [650]
        scale = 650.0
        text_descriptions = describe_pick_and_place(goal)
        
    elif env_name.startswith('ml1-v3-reach-v3-'):
        parts = env_name.split('-')
        task_name = '-'.join(parts[2:-2])
        task_idx = int(parts[-1])          # 0
        
        # Create environment instance
        env = base_env.train_classes[task_name]()
        
        # Retrieve and set task
        task = base_env.train_tasks[task_idx]
        env.set_task(task)

        # Default return values
        goal = env._last_rand_vec
        max_ep_len = 500 
        env_targets = [650]
        scale = 650.0
        text_descriptions = describe_reach(goal)


    else:
        raise NotImplementedError(f"Unknown env name format: {env_name}")

    return env, task_name, task_idx, max_ep_len, env_targets, scale, goal, text_descriptions


def get_env_list(env_name_list, config_save_path, device):
    info = {} 
    
    if len(env_name_list)>0 and env_name_list[0].startswith('mt50-v3-'):
        base_env = metaworld.MT50(seed=1)
    elif len(env_name_list)>0 and env_name_list[0].startswith('mt10-v3-'):
        base_env = metaworld.MT10(seed=1)
    elif len(env_name_list)>0 and env_name_list[0].startswith('ml1-'):
        parts = env_name_list[0].split("-")
        env_name = "-".join(parts[2:-2])
        base_env = metaworld.ML1(seed=1, env_name=env_name)
    elif len(env_name_list)>0 and env_name_list[0].startswith('mt50-ml45split-v3-'):
        base_env = metaworld.MT50(seed=1)
    else:
        base_env = None
    
    for env_name in env_name_list:
        info[env_name] = {}
        env, task_name, task_idx, max_ep_len, env_targets, scale, goal, text_descriptions = gen_env(env_name=env_name, base_env=base_env)
        info[env_name]['max_ep_len'] = max_ep_len
        info[env_name]['env_targets'] = env_targets
        info[env_name]['scale'] = scale
        info[env_name]['state_dim'] = env.observation_space.shape[0]
        info[env_name]['act_dim'] = env.action_space.shape[0] 
        info[env_name]['device'] = device
        info[env_name]['goal'] = goal
        info[env_name]['text_descriptions'] = text_descriptions
        info[env_name]['task_name'] = task_name
        info[env_name]['task_idx'] = task_idx
    return info, base_env


""" prompts """

def flatten_prompt(prompt, batch_size):
    p_s, p_a, p_r, p_d, p_rtg, p_timesteps, p_mask = prompt
    p_s = p_s.reshape((batch_size, -1, p_s.shape[-1]))
    p_a = p_a.reshape((batch_size, -1, p_a.shape[-1]))
    p_r = p_r.reshape((batch_size, -1, p_r.shape[-1]))
    p_d = p_d.reshape((batch_size, -1))
    p_rtg = p_rtg[:,:-1,:]
    p_rtg = p_rtg.reshape((batch_size, -1, p_rtg.shape[-1]))
    p_timesteps = p_timesteps.reshape((batch_size, -1))
    p_mask = p_mask.reshape((batch_size, -1)) 
    return p_s, p_a, p_r, p_d, p_rtg, p_timesteps, p_mask


def get_prompt(prompt_trajectories, info, variant):
    num_trajectories, p_sample, sorted_inds = info['num_trajectories'], info['p_sample'], info['sorted_inds']
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    num_episodes, max_len = variant['prompt_episode'], variant['prompt_length']

    def fn(sample_size=1):
        # random sample prompts with fixed length (prompt-length) in num episodes (prompt-episode)
        batch_inds = np.random.choice(
            np.arange(len(prompt_trajectories)),
            size=int(num_episodes*sample_size),
            replace=True,
            # p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(int(num_episodes*sample_size)):
            if variant["stochastic_prompt"]:
                traj = prompt_trajectories[int(batch_inds[i])] # random select traj
            else:
                traj = prompt_trajectories[int(sorted_inds[-i])] # select the best traj with highest rewards
                # traj = prompt_trajectories[i]
            si = max(0, traj['rewards'].shape[0] - max_len -1) # select the last traj with length max_len

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            # if tlen !=args.K:
            #     print('tlen not equal to k')
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            if not variant['no_state_normalize']:
                s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        return s, a, r, d, rtg, timesteps, mask

    return fn


def get_text(info, variant):
    batch_size = variant['batch_size']
    use_preencoded = variant.get('llm_preencoded', False)

    if use_preencoded:
        text_embeddings = info['text_embeddings']  # Tensor [m, n]

        def fn(batch_size=batch_size):
            indices = torch.randint(0, text_embeddings.size(0), (batch_size,))
            # Return a list of tensors (each [n]-dim row)
            return [text_embeddings[i] for i in indices]

    else:
        text_descriptions = info['text_descriptions']  # List[str]

        def fn(batch_size=batch_size):
            return [random.choice(text_descriptions) for _ in range(batch_size)]  # List[str]

    return fn

def get_goal(info, variant):
    batch_size = variant['batch_size']
    device = info['device']

    goal = info['goal']                       # e.g., np.array or list
    target_pos = torch.tensor(goal[3:],       # take last 3 as target
                              dtype=torch.float32,
                              device=device).unsqueeze(0)  # [1,3]

    def fn(batch_size=batch_size):
        return target_pos.repeat(batch_size, 1)  # no reassignment

    return fn

def get_prompt_batch(trajectories_list, prompt_trajectories_list, info, variant, train_env_name_list, train_env_name_grouped):
    per_env_batch_size = 1 
    llm_goal_prediction = variant['llm_goal_prediction']

    def fn(batch_size=variant['batch_size']):  # use total batch size to define how many envs to sample
        p_s_list, p_a_list, p_r_list, p_d_list, p_rtg_list, p_timesteps_list, p_mask_list = [], [], [], [], [], [], []
        s_list, a_list, r_list, d_list, rtg_list, timesteps_list, mask_list = [], [], [], [], [], [], []
        text_list, goal_list = [], []

        def sample_unique_task_type_indices(grouped_indices, batch_size):
            all_task_types = list(grouped_indices.keys())
            assert batch_size <= len(all_task_types), "Batch size exceeds number of unique task types!"

            selected_task_types = random.sample(all_task_types, batch_size)
            selected_indices = [random.choice(grouped_indices[task]) for task in selected_task_types]
            return selected_indices
        
        env_indices = sample_unique_task_type_indices(train_env_name_grouped, variant['batch_size'])

        for env_id in env_indices:
            env_name = train_env_name_list[env_id]

            if prompt_trajectories_list:
                get_prompt_fn = get_prompt(prompt_trajectories_list[env_id], info[env_name], variant)
            else:
                get_prompt_fn = get_prompt(trajectories_list[env_id], info[env_name], variant)

            get_batch_fn = get_batch(trajectories_list[env_id], info[env_name], variant) 
            get_text_fn = get_text(info[env_name], variant)
            
            prompt = flatten_prompt(get_prompt_fn(per_env_batch_size), per_env_batch_size)
            p_s, p_a, p_r, p_d, p_rtg, p_timesteps, p_mask = prompt
            p_s_list.append(p_s)
            p_a_list.append(p_a)
            p_r_list.append(p_r)
            p_d_list.append(p_d)
            p_rtg_list.append(p_rtg)
            p_timesteps_list.append(p_timesteps)
            p_mask_list.append(p_mask)

            batch = get_batch_fn(batch_size=per_env_batch_size)
            s, a, r, d, rtg, timesteps, mask = batch
            if variant['no_r']:
                r = torch.zeros_like(r)
            if variant['no_rtg']:
                rtg = torch.zeros_like(rtg)
            s_list.append(s)
            a_list.append(a)
            r_list.append(r)
            d_list.append(d)
            rtg_list.append(rtg)
            timesteps_list.append(timesteps)
            mask_list.append(mask)

            text = get_text_fn(batch_size=per_env_batch_size)
            text_list.extend(text)
            
            if llm_goal_prediction:
                get_goal_fn = get_goal(info[env_name], variant)
                goal = get_goal_fn(batch_size=per_env_batch_size)
                goal_list.append(goal)
                

        # Concatenate all collected samples
        p_s, p_a, p_r, p_d = torch.cat(p_s_list, dim=0), torch.cat(p_a_list, dim=0), torch.cat(p_r_list, dim=0), torch.cat(p_d_list, dim=0)
        p_rtg, p_timesteps, p_mask = torch.cat(p_rtg_list, dim=0), torch.cat(p_timesteps_list, dim=0), torch.cat(p_mask_list, dim=0)
        s, a, r, d = torch.cat(s_list, dim=0), torch.cat(a_list, dim=0), torch.cat(r_list, dim=0), torch.cat(d_list, dim=0)
        rtg, timesteps, mask = torch.cat(rtg_list, dim=0), torch.cat(timesteps_list, dim=0), torch.cat(mask_list, dim=0)

        prompt = p_s, p_a, p_r, p_d, p_rtg, p_timesteps, p_mask
        batch = s, a, r, d, rtg, timesteps, mask
        
        
        goals = torch.cat(goal_list, dim=0) if llm_goal_prediction else None
        return prompt, batch, text_list, goals

    return fn

""" batches """

def get_batch(trajectories, info, variant):
    num_trajectories, p_sample, sorted_inds = info['num_trajectories'], info['p_sample'], info['sorted_inds']
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    batch_size, K = variant['batch_size'], variant['K']

    def fn(batch_size=batch_size, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            # si = random.randint(0, traj['rewards'].shape[0] - 1)
            # si = random.randint(0, traj['rewards'].shape[0] - 1 - max_len)

            if traj['rewards'].shape[0] <= max_len:
                si = 0
            else:
                si = random.randint(0, traj['rewards'].shape[0] - 1 - max_len)

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            # if tlen !=args.K:
            #     print('tlen not equal to k')
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            if not variant['no_state_normalize']:
                s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device) # TODO: why mask only has several zeros

        return s, a, r, d, rtg, timesteps, mask

    return fn


def get_batch_finetune(trajectories, info, variant):
    num_trajectories, p_sample, sorted_inds = info['num_trajectories'], info['p_sample'], info['sorted_inds']
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    batch_size, K = variant['batch_size'], variant['prompt_length'] # use the same amount of data for funetuning

    def fn(batch_size=batch_size, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)
            si = max(0, traj['rewards'].shape[0] - max_len -1) # select the last traj with length max_len

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            # if tlen !=args.K:
            #     print('tlen not equal to k')
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            if not variant['no_state_normalize']:
                s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device) # TODO: why mask only has several zeros

        return s, a, r, d, rtg, timesteps, mask

    return fn

""" data processing """

def process_total_data_mean(trajectories, mode):

    # save all path information into separate lists
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    return state_mean, state_std


def process_dataset(trajectories, mode, env_name, dataset, pct_traj):
    # save all path information into separate lists
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])
    reward_info = [np.mean(returns), np.std(returns), np.max(returns), np.min(returns)]

    return trajectories, num_trajectories, sorted_inds, p_sample, state_mean, state_std, reward_info


def load_data_prompt(env_name_list, data_save_path, dataset, prompt_mode, args):
    trajectories_list = []
    prompt_trajectories_list = []
    for env_name in env_name_list:
        dataset_path = data_save_path+f'/{args.env}/{env_name}-{dataset}.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        prompt_dataset_path = data_save_path+f'/{args.env}/{env_name}-prompt-{prompt_mode}.pkl'
        with open(prompt_dataset_path, 'rb') as f:
            prompt_trajectories = pickle.load(f)
        trajectories_list.append(trajectories)
        prompt_trajectories_list.append(prompt_trajectories)
    
    # print('traj path:')
    # print(dataset_path)
    # print('prompt traj path')
    # print(prompt_dataset_path)
    # print()
    return trajectories_list, prompt_trajectories_list


def round_to_100(x):
    return min(500, int(math.ceil(x / 100.0) * 100))

def round_to_50(x):
    return int(np.round(x / 50.0) * 50)

def process_info(env_name_list, env_name_group, trajectories_list, info, mode, dataset, pct_traj, variant):
    # Step 1: Per env_name_list[i], process stats
    for i, env_name in enumerate(env_name_list):
        trajectories, num_trajectories, sorted_inds, p_sample, state_mean, state_std, reward_info = process_dataset(
            trajectories=trajectories_list[i],
            mode=mode,
            env_name=env_name,
            dataset=dataset,
            pct_traj=pct_traj
        )

        info[env_name]['num_trajectories'] = num_trajectories
        info[env_name]['sorted_inds'] = sorted_inds
        info[env_name]['p_sample'] = p_sample
        info[env_name]['state_mean'] = variant['total_state_mean'] if variant['average_state_mean'] else state_mean
        info[env_name]['state_std'] = variant['total_state_std'] if variant['average_state_mean'] else state_std
        info[env_name]['trajectories'] = trajectories  # temp for aggregation

    # Step 2: Per-task-type aggregation
    # for task_type, indices in env_name_group.items():
    #     all_returns = []
    #     all_lengths = []

    #     for idx in indices:
    #         env_name = env_name_list[idx]
    #         trajectories = info[env_name]['trajectories']
    #         for traj in trajectories:
    #             all_lengths.append(len(traj['observations']))
    #             all_returns.append(np.sum(traj['rewards']))

    #     max_ep_len_raw = np.max(all_lengths)
    #     target_return_raw = np.max(all_returns)
    #     max_ep_len = round_to_50(max_ep_len_raw + 50)
    #     target_return = round_to_50(target_return_raw)

    #     for idx in indices:
    #         env_name = env_name_list[idx]
    #         info[env_name]['max_ep_len'] = max_ep_len
    #         info[env_name]['env_targets'] = [target_return]
    #         info[env_name]['scale'] = target_return
    #         del info[env_name]['trajectories']

    return info


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum

""" evaluation """

def eval_episodes(target_rew, info, variant, env, env_name):
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    num_eval_episodes = variant['num_eval_episodes']
    mode = variant.get('mode', 'normal')

    def fn(model, prompt=None):
        returns = []
        successes = []
        for _ in range(num_eval_episodes):
            with torch.no_grad():
                ret, success, infos = prompt_evaluate_episode_rtg(
                    env,
                    state_dim,
                    act_dim,
                    model,
                    max_ep_len=max_ep_len,
                    scale=scale,
                    target_return=target_rew / scale,
                    mode=mode,
                    state_mean=state_mean,
                    state_std=state_std,
                    device=device,
                    prompt=prompt,
                    no_r=variant['no_r'],
                    no_rtg=variant['no_rtg'],
                    no_state_normalize=variant['no_state_normalize']                
                    )
            returns.append(ret)
            successes.append(success)
        return {
            f'{env_name}_target_{target_rew}_return_mean': np.mean(returns),
            f'{env_name}_target_{target_rew}_return_std': np.std(returns),
            f'{env_name}_target_{target_rew}_success_mean': np.mean(successes),
            f'{env_name}_target_{target_rew}_success_std': np.std(successes),
            }, np.mean(returns), np.mean(successes)
    return fn

def eval_episodes_llm(info, variant, env, env_name):
    max_ep_len, state_mean, state_std = info['max_ep_len'], info['state_mean'], info['state_std']
    device = info['device']
    num_eval_episodes = variant['num_eval_episodes']

    def fn(model, text=None):
        model.eval()
        returns = []
        successes = []
        if model.args.llm_preencoded:
            index = torch.randint(0, text.size(0),(1,))
            text_embedding = text[index]
            text_embedding = model.llm_projector(text_embedding)
            if model.args.normalize_embeddings:
                text_embedding = F.normalize(text_embedding, dim=-1)
        else:
            text_embedding = model.text_encoder(random.choice(text))
            

        if model.args.quantized_embed:
            text_embedding = model.quantizer(text_embedding)[0]
        if model.args.dual_policy:
            policy = model.llm_policy.build_policy(text_embedding)
        else:
            policy = model.policy.build_policy(text_embedding)

        for _ in range(num_eval_episodes):
            with torch.no_grad():
                ret, success, infos = prompt_evaluate_episode_llm(
                    env,
                    policy,
                    max_ep_len=max_ep_len,
                    state_mean=state_mean,
                    state_std=state_std,
                    device=device,
                    text=text,
                    no_state_normalize=variant['no_state_normalize']                
                    )
            returns.append(ret)
            successes.append(success)
        return np.mean(returns), np.mean(successes)
    return fn


def text_encoding(model, info, env_name_list,device):

    all_text_embeddings = []
    
    for i, env_name in enumerate(env_name_list):
        text_descriptions = info[env_name]['text_descriptions']
        model.eval() 
        with torch.no_grad():
            eos_token = model.text_encoder.tokenizer.eos_token
            text_batch = [text + " " + eos_token for text in text_descriptions]
            encoded = model.text_encoder.tokenizer(
                text_batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(device)
            outputs = model.text_encoder.encoder.model(**encoded, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # [B, T, D]
            eos_token_id = model.text_encoder.tokenizer.eos_token_id
            eos_positions = (encoded['input_ids'] == eos_token_id).int()
            eos_indices = eos_positions.argmax(dim=1)
            batch_indices = torch.arange(hidden_states.size(0), device=device)
            text_embeddings = hidden_states[batch_indices, eos_indices] 
        info[env_name]['text_embeddings'] = text_embeddings
        all_text_embeddings.append(text_embeddings)
    return info, all_text_embeddings



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    
    
def group_task_indices_by_type(train_env_name_list):
    grouped = defaultdict(list)
    for idx, env_name in enumerate(train_env_name_list):
        # Extract task type: 'mt50-v3-assembly-v3-0' → 'assembly-v3'
        task_type = '-'.join(env_name.split('-')[2:-1])
        grouped[task_type].append(idx)
    return grouped