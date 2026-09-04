"""Generate expert and prompt trajectories for the supported benchmarks."""

import argparse
import json
import os
import pickle

import numpy as np

import metaworld
from metaworld.policies import ENV_POLICY_MAP


SEED = 1
TAIL_STEPS_AFTER_SUCCESS = 2
PROMPT_LAST_STEPS = 5
TRAIN_COUNT_ML1 = 45
GOAL_DIM = 3

SUPPORTED_BENCHMARKS = (
    "ml1-v3-pick-place-v3",
    "mt10-v3",
    "mt50-v3",
    "mt50-ml45split-v3",
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bench",
        choices=SUPPORTED_BENCHMARKS,
        default="mt50-ml45split-v3",
        help="Benchmark to generate (default: mt50-ml45split-v3).",
    )
    return parser


def get_benchmark(bench_name, seed=SEED):
    """Construct the requested Meta-World benchmark and its output tag."""
    bench_name = bench_name.lower().strip()

    if bench_name.startswith("mt50-ml45split"):
        return metaworld.MT50(seed=seed), "mt50-ml45split-v3"
    if bench_name.startswith("mt50"):
        return metaworld.MT50(seed=seed), "mt50-v3"
    if bench_name.startswith("mt10"):
        return metaworld.MT10(seed=seed), "mt10-v3"
    if bench_name.startswith("ml1"):
        parts = bench_name.split("-")
        if len(parts) < 4:
            raise ValueError(
                "ML1 tag must be like 'ml1-v3-pick-place-v3', "
                f"got '{bench_name}'"
            )
        env_name = "-".join(parts[2:])
        return metaworld.ML1(seed=seed, env_name=env_name), bench_name

    raise ValueError(f"Unknown benchmark: {bench_name}")


def get_expert_policy_class(env_name):
    """Return the scripted Meta-World expert policy for an environment."""
    if env_name not in ENV_POLICY_MAP:
        raise ValueError(f"No expert policy found for env: {env_name}")
    return ENV_POLICY_MAP[env_name]


def safe_mean(values):
    return float(np.mean(values)) if values else 0.0


def safe_min(values):
    return int(np.min(values)) if values else 0


def safe_max(values):
    return int(np.max(values)) if values else 0


def get_goal_vec_from_env(env):
    """Read the current task's 3D goal without modifying Meta-World."""
    if hasattr(env, "_last_rand_vec") and env._last_rand_vec is not None:
        vector = np.asarray(env._last_rand_vec).reshape(-1)
        if vector.size >= GOAL_DIM:
            return vector[-GOAL_DIM:].astype(np.float32).copy()
    if hasattr(env, "_target_pos") and env._target_pos is not None:
        target = np.asarray(env._target_pos).reshape(-1)
        if target.size >= GOAL_DIM:
            return target[:GOAL_DIM].astype(np.float32).copy()
    raise RuntimeError(
        "Could not determine goal vector from env "
        "(_last_rand_vec or _target_pos)."
    )


def group_task_indices_by_environment(env_names, tasks):
    """Map each environment name to its task indices in benchmark order."""
    grouped = {name: [] for name in env_names}
    for task_id, task in enumerate(tasks):
        grouped[task.env_name].append(task_id)
    return grouped


def get_ml45_environment_split(env_names, seed):
    """Return the ML45 environment-name split used by the extra MT50 variant."""
    ml45 = metaworld.ML45(seed=seed)
    train_envs = set(ml45.train_classes.keys())
    test_envs = set(ml45.test_classes.keys())

    overlap = train_envs & test_envs
    if overlap:
        raise RuntimeError(f"ML45 train/test sets overlap: {sorted(overlap)}")

    missing = set(env_names) - (train_envs | test_envs)
    if missing:
        print(
            f"[WARN] MT50 envs not covered by ML45 split: {sorted(missing)} "
            "(will default to train)"
        )
    return train_envs, test_envs


def initialize_trajectory():
    """Create the trajectory structure expected by the training data loader."""
    return {
        "observations": [],
        "next_observations": [],
        "actions": [],
        "rewards": [],
        "terminates": [],
        "truncates": [],
        "dones": [],
        "success": [],
    }


def collect_expert_trajectory(env_cls, task, expert, is_ml1):
    """Run one scripted expert episode and return it only when successful."""
    env = env_cls()
    env.set_task(task)
    reset_output = env.reset()
    if isinstance(reset_output, tuple) and len(reset_output) == 2:
        observation = reset_output[0]
    else:
        observation = reset_output
    observation = np.asarray(observation, dtype=np.float32)

    # ML1 hides its goal in the stored observation. The scripted expert needs
    # the true goal, while training must retain the original observation.
    if is_ml1:
        goal_vector = get_goal_vec_from_env(env)
        observation_for_policy = observation.copy()
        observation_for_policy[-GOAL_DIM:] = goal_vector
        observation_to_store = observation.copy()
    else:
        goal_vector = None
        observation_for_policy = observation
        observation_to_store = observation

    trajectory = initialize_trajectory()
    success_seen = False
    post_success_steps = 0
    steps = 0
    episode_return = 0.0

    while True:
        action = expert.get_action(observation_for_policy)
        next_observation, reward, terminate, truncate, info = env.step(action)
        next_observation = np.asarray(next_observation, dtype=np.float32)
        done = bool(terminate or truncate)

        if is_ml1:
            next_observation_for_policy = next_observation.copy()
            next_observation_for_policy[-GOAL_DIM:] = goal_vector
            next_observation_to_store = next_observation.copy()
        else:
            next_observation_for_policy = next_observation
            next_observation_to_store = next_observation

        trajectory["observations"].append(observation_to_store)
        trajectory["next_observations"].append(next_observation_to_store)
        trajectory["actions"].append(np.asarray(action, dtype=np.float32))
        trajectory["rewards"].append(float(reward))
        trajectory["terminates"].append(bool(terminate))
        trajectory["truncates"].append(bool(truncate))
        trajectory["dones"].append(done)
        success = float(info.get("success", 0.0))
        trajectory["success"].append(success)

        episode_return += float(reward)
        steps += 1

        if not success_seen and success > 0.0:
            success_seen = True
            post_success_steps = 0
        elif success_seen:
            post_success_steps += 1

        observation_for_policy = next_observation_for_policy
        observation_to_store = next_observation_to_store

        if done or (
            success_seen and post_success_steps >= TAIL_STEPS_AFTER_SUCCESS
        ):
            break

    if not success_seen:
        return None

    trajectory = {key: np.array(values) for key, values in trajectory.items()}
    return trajectory, episode_return, steps


def save_trajectory_pair(trajectory, data_dir, base_name):
    """Save a full expert trajectory and its shortened prompt trajectory."""
    expert_path = os.path.join(data_dir, f"{base_name}-expert.pkl")
    with open(expert_path, "wb") as expert_file:
        pickle.dump([trajectory], expert_file)

    if PROMPT_LAST_STEPS > 0:
        prompt_length = PROMPT_LAST_STEPS
    else:
        prompt_length = len(trajectory["observations"])
    prompt_trajectory = {
        key: values[-prompt_length:] for key, values in trajectory.items()
    }
    prompt_path = os.path.join(data_dir, f"{base_name}-prompt-expert.pkl")
    with open(prompt_path, "wb") as prompt_file:
        pickle.dump([prompt_trajectory], prompt_file)


def collect_successful_trajectories(benchmark, bench_tag, data_dir):
    """Collect and save successful scripted-expert trajectories."""
    train_classes = benchmark.train_classes
    train_tasks = benchmark.train_tasks
    env_names = list(train_classes.keys())
    tasks_by_env = group_task_indices_by_environment(env_names, train_tasks)
    is_ml1 = bench_tag.startswith("ml1-v3")

    successful_task_ids = []
    task_id_to_env = {}
    env_to_task_ids = {}
    env_stats = {
        name: {"returns": [], "steps": [], "count": 0} for name in env_names
    }
    overall_returns = []
    overall_steps = []

    for env_name in env_names:
        print(f"\n=== {env_name.upper()} ===")
        env_cls = train_classes[env_name]
        expert = get_expert_policy_class(env_name)()

        for task_id in tasks_by_env[env_name]:
            result = collect_expert_trajectory(
                env_cls, train_tasks[task_id], expert, is_ml1
            )
            if result is None:
                continue

            trajectory, episode_return, steps = result
            env_type = f"t{task_id:03d}" if is_ml1 else env_name
            successful_task_ids.append(task_id)
            task_id_to_env[task_id] = env_type
            env_to_task_ids.setdefault(env_type, []).append(task_id)

            base_name = f"{bench_tag}-{env_type}-{task_id}"
            save_trajectory_pair(trajectory, data_dir, base_name)

            env_stats[env_name]["returns"].append(episode_return)
            env_stats[env_name]["steps"].append(steps)
            env_stats[env_name]["count"] += 1
            overall_returns.append(episode_return)
            overall_steps.append(steps)

    return {
        "env_names": env_names,
        "successful_task_ids": successful_task_ids,
        "task_id_to_env": task_id_to_env,
        "env_to_task_ids": env_to_task_ids,
        "env_stats": env_stats,
        "overall_returns": overall_returns,
        "overall_steps": overall_steps,
    }


def split_tasks(
    bench_tag,
    successful_task_ids,
    task_id_to_env,
    rng,
    ml45_train_envs=None,
    ml45_test_envs=None,
):
    """Apply the benchmark-specific train/test split."""
    if bench_tag.startswith("ml1-v3"):
        all_ids = np.array(successful_task_ids, dtype=int)
        rng.shuffle(all_ids)
        return (
            all_ids[:TRAIN_COUNT_ML1].tolist(),
            all_ids[TRAIN_COUNT_ML1:].tolist(),
        )

    if bench_tag == "mt50-ml45split-v3":
        train_tasks = []
        test_tasks = []
        for task_id in successful_task_ids:
            env_name = task_id_to_env[task_id]
            if env_name in ml45_train_envs:
                train_tasks.append(task_id)
            elif env_name in ml45_test_envs:
                test_tasks.append(task_id)
            else:
                train_tasks.append(task_id)
        return train_tasks, test_tasks

    return successful_task_ids, []


def build_config(bench_tag, collected, train_tasks, test_tasks):
    """Build the task configuration consumed by training."""
    return {
        "env": bench_tag,
        "total_tasks": len(collected["successful_task_ids"]),
        "train_tasks": train_tasks,
        "test_tasks": test_tasks,
        "task_id_to_env": {
            str(key): value for key, value in collected["task_id_to_env"].items()
        },
        "env_to_task_ids": collected["env_to_task_ids"],
    }


def build_stats(collected):
    """Build per-environment and overall successful-episode statistics."""
    env_stats = collected["env_stats"]
    return {
        "per_env": {
            env_name: {
                "successful_episodes": env_stats[env_name]["count"],
                "avg_return": safe_mean(env_stats[env_name]["returns"]),
                "avg_steps": safe_mean(env_stats[env_name]["steps"]),
                "min_steps": safe_min(env_stats[env_name]["steps"]),
                "max_steps": safe_max(env_stats[env_name]["steps"]),
            }
            for env_name in collected["env_names"]
        },
        "overall": {
            "total_successful_episodes": int(
                sum(values["count"] for values in env_stats.values())
            ),
            "avg_return": safe_mean(collected["overall_returns"]),
            "avg_steps": safe_mean(collected["overall_steps"]),
            "min_steps": safe_min(collected["overall_steps"]),
            "max_steps": safe_max(collected["overall_steps"]),
        },
    }


def write_json(path, contents):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(contents, output_file, indent=4)


def generate_expert_data(bench_name, seed=SEED):
    """Generate trajectories, configuration, and statistics for a benchmark."""
    benchmark, bench_tag = get_benchmark(bench_name, seed=seed)
    config_dir = os.path.join("config", bench_tag)
    data_dir = os.path.join("data", bench_tag)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    ml45_train_envs = set()
    ml45_test_envs = set()
    if bench_tag == "mt50-ml45split-v3":
        ml45_train_envs, ml45_test_envs = get_ml45_environment_split(
            list(benchmark.train_classes.keys()), seed
        )

    collected = collect_successful_trajectories(benchmark, bench_tag, data_dir)
    rng = np.random.RandomState(seed)
    train_tasks, test_tasks = split_tasks(
        bench_tag,
        collected["successful_task_ids"],
        collected["task_id_to_env"],
        rng,
        ml45_train_envs,
        ml45_test_envs,
    )

    config = build_config(bench_tag, collected, train_tasks, test_tasks)
    stats = build_stats(collected)
    write_json(os.path.join(config_dir, f"{bench_tag}.json"), config)
    write_json(os.path.join(config_dir, f"{bench_tag}-stats.json"), stats)

    print(
        f"\nDone. Saved {len(collected['successful_task_ids'])} "
        f"successful tasks for {bench_tag}."
    )
    if bench_tag == "mt50-ml45split-v3":
        print(
            f"  Train (ML45 envs): {len(train_tasks)} tasks | "
            f"Test (ML45 envs): {len(test_tasks)} tasks"
        )
    print("Stats summary (successful episodes only):")
    print(json.dumps(stats["overall"], indent=2))
    return config, stats


def main():
    args = build_parser().parse_args()
    generate_expert_data(args.bench)


if __name__ == "__main__":
    main()
