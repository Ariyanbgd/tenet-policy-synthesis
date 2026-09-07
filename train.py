"""Experiment launcher for the Meta-World portion of the TENET release.

This module selects benchmarks and method presets, validates or generates the
required expert data, and delegates training to ``training.experiment_mix_env``.
Configuration is applied in this order, with later values taking precedence:

    shared defaults -> benchmark preset -> method preset -> CLI overrides

Examples:
    python train.py --list
    python train.py --benchmark mt10 --method tenet_contrast
    python train.py --benchmark mt10 mt50 --method dt prompt_dt
    python train.py --benchmark all --method all --dry-run

Expert data is generated automatically only when the selected dataset is
missing or incomplete. Use ``--no-generate-expert-data`` for validation only.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime


# The MT50/ML45 split is an additional experiment not reported in the paper.
BENCHMARKS = {
    "ml1-pick-place": {
        "env": "ml1-v3-pick-place-v3",
        "batch_size": 32,
        "published": True,
    },
    "mt10": {"env": "mt10-v3", "batch_size": 10, "published": True},
    "mt50": {"env": "mt50-v3", "batch_size": 50, "published": True},
    "mt50-ml45split": {
        "env": "mt50-ml45split-v3",
        "batch_size": 45,
        "published": False,
    },
}

METHOD_DESCRIPTIONS = {
    "dt": "Decision Transformer without task prompts",
    "prompt_dt": "Prompt-DT conditioned on expert trajectory prompts",
    "tenet": "direct text-conditioned hypernetwork",
    "tenet_contrast": "TENET with contrastive text-trajectory grounding",
    "tenet_mse": "TENET with MSE text-trajectory grounding",
}

# Each preset contains only settings that distinguish that method. Shared
# values belong in BASE_CONFIG so the scientific differences remain visible.
METHODS = {
    # Sequence baseline without a trajectory prompt.
    "dt": {"no_prompt": True, "log_to_wandb": True, "max_iters": 5000},
    # Sequence baseline conditioned on an expert trajectory prompt.
    "prompt_dt": {"log_to_wandb": True, "max_iters": 5000},
    # Direct TENET: text and trajectory branches use separate policies.
    "tenet": {
        "log_to_wandb": True,
        "hyper_network": True,
        "hn_embed_dim": 256,
        "consistency_regularizer": True,
        "cr_coeff": 0.0001,
        "llm": True,
        "normalize_embeddings": True,
        "llm_preencoded": True,
        "only_llm_evaluation": True,
        "dual_policy": True,
        "max_iters": 5000,
    },
    # Grounded TENET: align text and trajectories contrastively.
    "tenet_contrast": {
        "log_to_wandb": True,
        "hyper_network": True,
        "hn_embed_dim": 256,
        "consistency_regularizer": True,
        "cr_coeff": 0.0001,
        "llm": True,
        "normalize_embeddings": True,
        "llm_preencoded": True,
        "use_embedding_contrastive": True,
        "only_llm_evaluation": True,
        "max_iters": 5000,
    },
    # Grounded TENET: align text and trajectories with MSE.
    "tenet_mse": {
        "log_to_wandb": True,
        "hyper_network": True,
        "hn_embed_dim": 256,
        "consistency_regularizer": True,
        "cr_coeff": 0.0001,
        "llm": True,
        "normalize_embeddings": True,
        "llm_preencoded": True,
        "use_embedding_mse": True,
        "only_llm_evaluation": True,
        "max_iters": 5000,
    },
}

# Shared defaults passed to training.experiment_mix_env. Options inherited from
# Prompt-DT are retained for compatibility, including those not used here.
BASE_CONFIG = {
    # Dataset and trajectory-prompt settings
    "dataset_mode": "expert",
    "test_dataset_mode": "expert",
    "train_prompt_mode": "expert",
    "test_prompt_mode": "expert",
    "prompt_episode": 1,
    "prompt_length": 5,
    "stochastic_prompt": True,
    "no_prompt": False,
    "no_r": False,
    "no_rtg": False,
    # Prompt-DT fine-tuning and state preprocessing
    "finetune": False,
    "finetune_steps": 10,
    "finetune_batch_size": 256,
    "finetune_opt": True,
    "finetune_lr": 1e-4,
    "no_state_normalize": False,
    "average_state_mean": True,
    "evaluation": False,
    "load_path": None,
    # Language encoder and text-conditioned policy settings
    "llm": False,
    "delayed_llm": False,
    "delayed_llm_iteration": 2500,
    "llm_model": "meta-llama/Meta-Llama-3-8B",
    "llm_finetune_method": "none",
    "llm_preencoded": False,
    "normalize_embeddings": False,
    "llm_pooling_type": "eos",
    "llm_projection_type": "mlp",
    "num_projection_layers": 2,
    "dual_policy": False,
    "llm_goal_prediction": False,
    "llm_goal_pred_coeff": 0.1,
    "only_llm_evaluation": False,
    # Optional text-trajectory grounding objectives
    "use_embedding_contrastive": False,
    "use_text_embedding_contrastive": False,
    "contrastive_coeff": 1.0,
    "text_contrastive_coeff": 0.1,
    "contrastive_temperature": 0.07,
    "use_embedding_mse": False,
    "embedding_mse_coeff": 0.1,
    "use_embedding_discriminator": False,
    "discriminator_coeff": 0.1,
    # Generated-policy and hypernetwork settings
    "hyper_network": False,
    "hn_embed_dim": 128,
    "hypernet_layers": [128, 128],
    "policy_hidden_layers": [128, 128],
    "consistency_regularizer": False,
    "cr_coeff": 0.0001,
    # Decision Transformer architecture and optimization
    "mode": "normal",
    "K": 20,
    "pct_traj": 1.0,
    "embed_dim": 128,
    "n_layer": 3,
    "n_head": 1,
    "activation_function": "relu",
    "dropout": 0.1,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "warmup_steps": 10000,
    # Evaluation, logging, and checkpointing
    "num_eval_episodes": 1,
    "max_iters": 50000,
    "num_steps_per_iter": 10,
    "device": "cuda",
    "log_to_wandb": False,
    "train_eval_interval": 200,
    "test_eval_interval": 200,
    "save": False,
}


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run TENET experiments on Meta-World. Selecting all benchmarks "
            "and methods launches 20 runs per seed."
        )
    )
    parser.add_argument(
        "--benchmark", nargs="+", default=["mt10"], metavar="NAME",
        help="Benchmark preset(s), or 'all' (default: mt10).",
    )
    parser.add_argument(
        "--method", nargs="+", default=["tenet_contrast"], metavar="NAME",
        help="Method preset(s), or 'all' (default: tenet_contrast).",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[1001],
        help="One or more random seeds (default: 1001).",
    )
    # None means preserve the method/base setting.
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--num-eval-episodes", type=int, default=None)
    parser.add_argument("--train-eval-interval", type=int, default=None)
    parser.add_argument("--test-eval-interval", type=int, default=None)
    parser.add_argument(
        "--log-to-wandb", action=argparse.BooleanOptionalAction, default=None,
        help="Override W&B logging for every selected run.",
    )
    parser.add_argument(
        "--save", action=argparse.BooleanOptionalAction, default=None,
        help="Override checkpoint saving for every selected run.",
    )
    parser.add_argument(
        "--generate-expert-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Generate a selected benchmark when its dataset is incomplete "
            "(default: enabled)."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print selected runs without checking data or starting training.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available presets and exit."
    )
    return parser


def expand_selection(values, available, label, parser):
    """Validate preset names and expand the special value ``all``."""
    if "all" in values:
        if len(values) != 1:
            parser.error(f"'all' cannot be combined with other {label} names")
        return list(available)
    unknown = [value for value in values if value not in available]
    if unknown:
        parser.error(
            f"unknown {label}(s): {', '.join(unknown)}; "
            f"choose from: {', '.join(available)}, all"
        )
    return values


def print_available_presets():
    print("Benchmarks:")
    for name, config in BENCHMARKS.items():
        note = "paper" if config["published"] else "additional (not in paper)"
        print(f"  {name:<18} {config['env']:<28} {note}")
    print("\nMethods:")
    for name in METHODS:
        print(f"  {name:<18} {METHOD_DESCRIPTIONS[name]}")


def get_runtime_overrides(cli_args):
    values = {
        "device": cli_args.device,
        "max_iters": cli_args.max_iters,
        "num_eval_episodes": cli_args.num_eval_episodes,
        "train_eval_interval": cli_args.train_eval_interval,
        "test_eval_interval": cli_args.test_eval_interval,
        "log_to_wandb": cli_args.log_to_wandb,
        "save": cli_args.save,
    }
    return {name: value for name, value in values.items() if value is not None}


def build_experiment_config(benchmark_name, method_name, seed, overrides):
    benchmark = BENCHMARKS[benchmark_name]
    config = dict(BASE_CONFIG)
    config.update(
        env=benchmark["env"], batch_size=benchmark["batch_size"], seed=seed
    )
    config.update(METHODS[method_name])
    config.update(overrides)
    return config


def find_missing_expert_data(env):
    """Return missing or invalid config/data paths for a benchmark."""
    config_dir = os.path.join("config", env)
    config_path = os.path.join(config_dir, f"{env}.json")
    stats_path = os.path.join(config_dir, f"{env}-stats.json")
    problems = []

    if not os.path.isfile(config_path):
        return [config_path]
    if not os.path.isfile(stats_path) or os.path.getsize(stats_path) == 0:
        problems.append(stats_path)

    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            task_config = json.load(config_file)
        if task_config.get("env") != env:
            problems.append(f"{config_path} (incorrect env field)")
        train_tasks = task_config["train_tasks"]
        test_tasks = task_config["test_tasks"]
        task_id_to_env = task_config["task_id_to_env"]
        if not isinstance(train_tasks, list) or not isinstance(test_tasks, list):
            raise TypeError("train_tasks and test_tasks must be lists")
        if set(train_tasks) & set(test_tasks):
            problems.append(f"{config_path} (train/test tasks overlap)")
        task_ids = train_tasks + test_tasks
        if len(task_ids) != len(set(task_ids)):
            problems.append(f"{config_path} (duplicate task IDs)")
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        return [f"{config_path} (invalid: {error})"]

    data_dir = os.path.join("data", env)
    for task_id in task_ids:
        env_type = task_id_to_env.get(str(task_id))
        if not env_type:
            problems.append(f"{config_path} (no environment for task {task_id})")
            continue
        stem = f"{env}-{env_type}-{task_id}"
        for suffix in ("expert.pkl", "prompt-expert.pkl"):
            data_path = os.path.join(data_dir, f"{stem}-{suffix}")
            if not os.path.isfile(data_path) or os.path.getsize(data_path) == 0:
                problems.append(data_path)
    return problems


def ensure_expert_data(env, generate_if_missing=False):
    """Validate a dataset and optionally invoke the expert-data generator."""
    missing = find_missing_expert_data(env)
    if not missing:
        print(f"Expert data ready: {env}")
        return

    if generate_if_missing:
        print(f"Expert data incomplete for {env}; starting generation.")
        subprocess.run(
            [sys.executable, "generate_expert_data.py", "--bench", env],
            check=True,
        )
        missing = find_missing_expert_data(env)
        if not missing:
            print(f"Expert data generated successfully: {env}")
            return

    preview = "\n".join(f"  - {path}" for path in missing[:5])
    remainder = len(missing) - min(len(missing), 5)
    if remainder:
        preview += f"\n  ... and {remainder} more missing file(s)"
    raise FileNotFoundError(
        f"Expert data for '{env}' is missing or incomplete:\n{preview}\n"
        "Automatic generation was disabled. Generate it with:\n"
        f"  {sys.executable} train.py --benchmark "
        f"{next(name for name, item in BENCHMARKS.items() if item['env'] == env)} "
        "--method tenet_contrast --generate-expert-data"
    )


def print_run(run_number, total_runs, benchmark_name, method_name, config):
    print(f"\n[{run_number}/{total_runs}] {benchmark_name} / {method_name}")
    print(
        f"  env={config['env']} seed={config['seed']} "
        f"batch_size={config['batch_size']} max_iters={config['max_iters']}"
    )
    print(
        f"  device={config['device']} wandb={config['log_to_wandb']} "
        f"save={config['save']}"
    )


def main():
    parser = build_parser()
    cli_args = parser.parse_args()
    if cli_args.list:
        print_available_presets()
        return

    benchmarks = expand_selection(
        cli_args.benchmark, BENCHMARKS, "benchmark", parser
    )
    methods = expand_selection(cli_args.method, METHODS, "method", parser)
    overrides = get_runtime_overrides(cli_args)
    total_runs = len(benchmarks) * len(methods) * len(cli_args.seeds)
    if cli_args.dry_run:
        print(f"Dry run: {total_runs} experiment(s) selected.")
    else:
        # Validate each benchmark once, regardless of method or seed count.
        # Generation runs only when validation reports missing/invalid files.
        for benchmark_name in benchmarks:
            ensure_expert_data(
                BENCHMARKS[benchmark_name]["env"],
                generate_if_missing=cli_args.generate_expert_data,
            )

    run_number = 0
    for benchmark_name in benchmarks:
        env = BENCHMARKS[benchmark_name]["env"]
        project = f"project_metaworld_tenet_{env}"
        base_result_dir = os.path.join("results", env)
        for method_name in methods:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            for seed in cli_args.seeds:
                run_number += 1
                config = build_experiment_config(
                    benchmark_name, method_name, seed, overrides
                )
                print_run(run_number, total_runs, benchmark_name, method_name, config)
                if cli_args.dry_run:
                    continue

                # Keep listing/dry runs lightweight by importing training lazily.
                from training import experiment_mix_env

                run_dir = os.path.join(
                    base_result_dir, method_name, f"seed{seed}_{timestamp}"
                )
                model_name = f"{method_name}_seed{seed}_{timestamp}"
                experiment_args = argparse.Namespace(**config)
                experiment_mix_env(
                    experiment_args,
                    variant=vars(experiment_args),
                    run_dir=run_dir,
                    model_name=model_name,
                    project=project,
                )


if __name__ == "__main__":
    main()
