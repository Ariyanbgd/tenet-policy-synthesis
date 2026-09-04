import argparse
import os
import pickle
import json
import numpy as np
import metaworld
from metaworld.policies import ENV_POLICY_MAP

# ======================= Config =======================
SEED = 1
SUPPORTED_BENCHMARKS = (
    "ml1-v3-pick-place-v3",
    "mt10-v3",
    "mt50-v3",
    "mt50-ml45split-v3",
)

parser = argparse.ArgumentParser(description="Generate Meta-World expert datasets.")
parser.add_argument(
    "--bench",
    choices=SUPPORTED_BENCHMARKS,
    default="mt50-ml45split-v3",
    help="Benchmark to generate (default: mt50-ml45split-v3).",
)
BENCH = parser.parse_args().bench

TAIL_STEPS_AFTER_SUCCESS = 2
PROMPT_LAST_STEPS = 5
TRAIN_COUNT_ML1 = 45   # 45 train / 5 test for ML1 (90/10 for success-filtered)
GOAL_DIM = 3           # the last 3 elements are the goal slot

# Dirs
config_dir = f"config/{BENCH}"
data_dir   = f"data/{BENCH}"
os.makedirs(config_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# =================== Helpers ==========================
def get_benchmark(bench_name: str):
    """
    Returns (benchmark_obj, bench_tag).
    - mt10-v3                  -> (MT10(seed), "mt10-v3")
    - mt50-v3                  -> (MT50(seed), "mt50-v3")
    - mt50-ml45split-v3        -> (MT50(seed), "mt50-ml45split-v3")
    - ml1-v3-<env_name>        -> (ML1(seed, env_name=<env_name>), "ml1-v3-<env_name>")
    """
    bench_name = bench_name.lower().strip()

    if bench_name.startswith("mt50-ml45split"):
        return metaworld.MT50(seed=SEED), "mt50-ml45split-v3"

    if bench_name.startswith("mt50"):
        return metaworld.MT50(seed=SEED), "mt50-v3"

    if bench_name.startswith("mt10"):
        return metaworld.MT10(seed=SEED), "mt10-v3"

    if bench_name.startswith("ml1"):
        parts = bench_name.split("-")
        if len(parts) < 4:
            raise ValueError(f"ML1 tag must be like 'ml1-v3-pick-place-v3', got '{bench_name}'")
        env_name = "-".join(parts[2:])  # e.g., 'pick-place-v3'
        return metaworld.ML1(seed=SEED, env_name=env_name), bench_name

    raise ValueError(f"Unknown benchmark: {bench_name}")

def get_expert_policy_class(env_name: str):
    if env_name not in ENV_POLICY_MAP:
        raise ValueError(f"No expert policy found for env: {env_name}")
    return ENV_POLICY_MAP[env_name]

def safe_mean(xs):
    return float(np.mean(xs)) if len(xs) else 0.0
def safe_min(xs):
    return int(np.min(xs)) if len(xs) else 0
def safe_max(xs):
    return int(np.max(xs)) if len(xs) else 0

def get_goal_vec_from_env(env) -> np.ndarray:
    """
    Get the 3D goal associated with the current task WITHOUT modifying Meta-World.
    Priority:
      1) env._last_rand_vec[-3:]  (task's goal from rand_vec)
      2) env._target_pos          (goal set at reset_model)
    """
    if hasattr(env, "_last_rand_vec") and env._last_rand_vec is not None:
        vec = np.asarray(env._last_rand_vec).reshape(-1)
        if vec.size >= GOAL_DIM:
            return vec[-GOAL_DIM:].astype(np.float32).copy()
    if hasattr(env, "_target_pos") and env._target_pos is not None:
        tgt = np.asarray(env._target_pos).reshape(-1)
        if tgt.size >= GOAL_DIM:
            return tgt[:GOAL_DIM].astype(np.float32).copy()
    raise RuntimeError("Could not determine goal vector from env (_last_rand_vec or _target_pos).")

rng = np.random.RandomState(SEED)

# =================== Init benchmark ====================
bench, bench_tag = get_benchmark(BENCH)
train_classes = bench.train_classes
train_tasks   = bench.train_tasks
env_names     = list(train_classes.keys())  # e.g., ['pick-place-v3'] for ML1

# If using mt50-ml45split-v3, derive env-name split from ML45
ml45_train_envs, ml45_test_envs = set(), set()
if bench_tag == "mt50-ml45split-v3":
    ml45 = metaworld.ML45(seed=SEED)   # only to read the split
    ml45_train_envs = set(ml45.train_classes.keys())
    ml45_test_envs  = set(ml45.test_classes.keys())

    overlap = ml45_train_envs & ml45_test_envs
    if overlap:
        raise RuntimeError(f"ML45 train/test sets overlap: {sorted(overlap)}")

    missing = set(env_names) - (ml45_train_envs | ml45_test_envs)
    if missing:
        # Not fatal. Commonly MT50 contains all ML45 envs (plus extra), or naming differences across versions.
        print(f"[WARN] MT50 envs not covered by ML45 split: {sorted(missing)} (will default to train)")

# Map: env_name -> list of task indices
env_to_task_indices_raw = {name: [] for name in env_names}
for idx, task in enumerate(train_tasks):
    env_to_task_indices_raw[task.env_name].append(idx)

# =================== Outputs/Trackers ==================
successful_task_ids = []
task_id_to_env = {}   # task_id -> env_name (MT) or short tXXX (ML1)
env_to_task_ids = {}  # env_name -> [task_ids]

# Stats (successful episodes only)
env_stats = {name: {"returns": [], "steps": [], "count": 0} for name in env_names}
overall_returns, overall_steps = [], []

is_ml1 = bench_tag.startswith("ml1-v3")

# To be filled later
train_tasks_list, test_tasks_list = [], []

# =================== Main Loop =========================
for env_name in env_names:
    print(f"\n=== {env_name.upper()} ===")
    env_cls = train_classes[env_name]
    expert  = get_expert_policy_class(env_name)()

    for task_id in env_to_task_indices_raw[env_name]:
        env = env_cls()
        env.set_task(train_tasks[task_id])        # fixes the task (contains rand_vec)
        reset_out = env.reset()
        obs = reset_out[0] if (isinstance(reset_out, tuple) and len(reset_out) == 2) else reset_out
        obs = np.asarray(obs, dtype=np.float32)

        # Two obs variants:
        # - obs_for_policy: overwrite last 3 slots with true goal (so expert policy works)
        # - obs_to_store: keep as-is (last 3 zeros in ML1)
        if is_ml1:
            goal_vec = get_goal_vec_from_env(env)
            obs_for_policy = obs.copy()
            obs_for_policy[-GOAL_DIM:] = goal_vec
            obs_to_store   = obs.copy()
        else:
            obs_for_policy = obs
            obs_to_store   = obs

        success_seen = False
        post_success_steps = 0
        steps = 0
        ep_return = 0.0

        traj = {
            "observations": [],
            "next_observations": [],
            "actions": [],
            "rewards": [],
            "terminates": [],
            "truncates": [],
            "dones": [],
            "success": []
        }

        while True:
            action = expert.get_action(obs_for_policy)
            next_obs, reward, terminate, truncate, info = env.step(action)
            next_obs = np.asarray(next_obs, dtype=np.float32)
            done = bool(terminate or truncate)

            if is_ml1:
                next_obs_for_policy = next_obs.copy()
                next_obs_for_policy[-GOAL_DIM:] = goal_vec
                next_obs_to_store   = next_obs.copy()
            else:
                next_obs_for_policy = next_obs
                next_obs_to_store   = next_obs

            traj["observations"].append(obs_to_store)
            traj["next_observations"].append(next_obs_to_store)
            traj["actions"].append(np.asarray(action, dtype=np.float32))
            traj["rewards"].append(float(reward))
            traj["terminates"].append(bool(terminate))
            traj["truncates"].append(bool(truncate))
            traj["dones"].append(done)
            s = float(info.get("success", 0.0))
            traj["success"].append(s)

            ep_return += float(reward)
            steps += 1

            if (not success_seen) and (s > 0.0):
                success_seen = True
                post_success_steps = 0
            elif success_seen:
                post_success_steps += 1

            obs = next_obs
            obs_for_policy = next_obs_for_policy
            obs_to_store   = next_obs_to_store

            if done or (success_seen and post_success_steps >= TAIL_STEPS_AFTER_SUCCESS):
                break

        if success_seen:
            # MT/ML1 naming
            if is_ml1:
                fabricated_env_type_short = f"t{task_id:03d}"  # e.g., "t017"
                env_type_for_maps   = fabricated_env_type_short
                env_type_for_files  = fabricated_env_type_short
            else:
                env_type_for_maps   = env_name
                env_type_for_files  = env_name

            successful_task_ids.append(task_id)
            task_id_to_env[task_id] = env_type_for_maps
            env_to_task_ids.setdefault(env_type_for_maps, []).append(task_id)

            # Save trajectories
            traj_np = {k: np.array(v) for k, v in traj.items()}
            base_name = f"{bench_tag}-{env_type_for_files}-{task_id}"

            expert_path = os.path.join(data_dir, f"{base_name}-expert.pkl")
            with open(expert_path, "wb") as f:
                pickle.dump([traj_np], f)

            last_k = PROMPT_LAST_STEPS if PROMPT_LAST_STEPS > 0 else len(traj_np["observations"])
            traj_short = {k: v[-last_k:] for k, v in traj_np.items()}
            prompt_path = os.path.join(data_dir, f"{base_name}-prompt-expert.pkl")
            with open(prompt_path, "wb") as f:
                pickle.dump([traj_short], f)

            # Stats
            env_stats[env_name]["returns"].append(ep_return)
            env_stats[env_name]["steps"].append(steps)
            env_stats[env_name]["count"] += 1
            overall_returns.append(ep_return)
            overall_steps.append(steps)

# =================== Train/Test Split ==================
if is_ml1:
    # Random split for ML1 (preserve your original behavior)
    all_ids = np.array(successful_task_ids, dtype=int)
    rng.shuffle(all_ids)
    train_tasks_list = all_ids[:TRAIN_COUNT_ML1].tolist()
    test_tasks_list  = all_ids[TRAIN_COUNT_ML1:].tolist()

elif bench_tag == "mt50-ml45split-v3":
    # Use ML45 env-name split to assign MT50 task_ids
    train_tasks_list, test_tasks_list = [], []
    for tid in successful_task_ids:
        env_name = task_id_to_env[tid]  # for MT50 this is the env_name
        if env_name in ml45_train_envs:
            train_tasks_list.append(tid)
        elif env_name in ml45_test_envs:
            test_tasks_list.append(tid)
        else:
            # Fallback: env not present in ML45 split -> assign to train
            train_tasks_list.append(tid)
else:
    # Plain MT50/MT10: all train
    train_tasks_list = successful_task_ids
    test_tasks_list  = []

# =================== Config file =======================
task_id_to_env_str_keys = {str(k): v for k, v in task_id_to_env.items()}

config = {
    "env": bench_tag,  # e.g., "mt50-ml45split-v3"
    "total_tasks": len(successful_task_ids),
    "train_tasks": train_tasks_list,
    "test_tasks": test_tasks_list,
    "task_id_to_env": task_id_to_env_str_keys,  # e.g., {"13": "reach-v3"}
    "env_to_task_ids": env_to_task_ids          # e.g., {"reach-v3": [13, ...]}
}
with open(os.path.join(config_dir, f"{bench_tag}.json"), "w") as f:
    json.dump(config, f, indent=4)

# =================== Stats (per-env & overall) =========
stats = {
    "per_env": {
        env: {
            "successful_episodes": env_stats[env]["count"],
            "avg_return": safe_mean(env_stats[env]["returns"]),
            "avg_steps": safe_mean(env_stats[env]["steps"]),
            "min_steps": safe_min(env_stats[env]["steps"]),
            "max_steps": safe_max(env_stats[env]["steps"]),
        }
        for env in env_names
    },
    "overall": {
        "total_successful_episodes": int(sum(s["count"] for s in env_stats.values())),
        "avg_return": safe_mean(overall_returns),
        "avg_steps": safe_mean(overall_steps),
        "min_steps": safe_min(overall_steps),
        "max_steps": safe_max(overall_steps),
    },
}
with open(os.path.join(config_dir, f"{bench_tag}-stats.json"), "w") as f:
    json.dump(stats, f, indent=4)

print(f"\n✅ Done. Saved {len(successful_task_ids)} successful tasks for {bench_tag}.")
if bench_tag == "mt50-ml45split-v3":
    print(f"  ↳ Train (ML45 envs): {len(train_tasks_list)} tasks | Test (ML45 envs): {len(test_tasks_list)} tasks")
print("📊 Stats summary (successful episodes only):")
print(json.dumps(stats["overall"], indent=2))
