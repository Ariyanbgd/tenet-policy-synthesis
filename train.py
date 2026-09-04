"""Experiment launcher for the Meta-World portion of the TENET release.

This module assembles configurations only. Training remains implemented in
``main.experiment_mix_env``.
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

# Each preset contains only settings that distinguish that method. Values are
# unchanged from the original launcher.
METHODS = {
    "dt": {"no_prompt": True, "log_to_wandb": True, "max_iters": 5000},
    "prompt_dt": {"log_to_wandb": False, "max_iters": 5000},
    "tenet": {
        "log_to_wandb": False,
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
    "tenet_mse": {
        "log_to_wandb": False,
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

# Shared defaults passed to main.experiment_mix_env. Options inherited from
# Prompt-DT are retained for compatibility, including those not used here.
BASE_CONFIG = {
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
    "finetune": False,
    "finetune_steps": 10,
    "finetune_batch_size": 256,
    "finetune_opt": True,
    "finetune_lr": 1e-4,
    "no_state_normalize": False,
    "average_state_mean": True,
    "evaluation": False,
    "load_path": None,
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
    "use_embedding_contrastive": False,
    "use_text_embedding_contrastive": False,
    "contrastive_coeff": 1.0,
    "text_contrastive_coeff": 0.1,
    "contrastive_temperature": 0.07,
    "use_embedding_mse": False,
    "embedding_mse_coeff": 0.1,
    "use_embedding_discriminator": False,
    "discriminator_coeff": 0.1,
    "hyper_network": False,
    "hn_embed_dim": 128,
    "hypernet_layers": [128, 128],
    "policy_hidden_layers": [128, 128],
    "consistency_regularizer": False,
    "cr_coeff": 0.0001,
    "quantized_embed": False,
    "quantized_embed_for_loss": False,
    "quant_level": 21,
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
        description="Run one or more TENET experiments on Meta-World."
    )
    parser.add_argument(
        "--benchmark", nargs="+", default=["ml1-pick-place"], metavar="NAME",
        help="Benchmark preset(s), or 'all' (default: ml1-pick-place).",
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
        action="store_true",
        help="Generate a selected benchmark when its dataset is incomplete.",
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
        print(f"  {name}")


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
    """Return missing config/data paths for a benchmark."""
    config_path = os.path.join("config", env, f"{env}.json")
    if not os.path.isfile(config_path):
        return [config_path]

    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            task_config = json.load(config_file)
        task_ids = task_config["train_tasks"] + task_config["test_tasks"]
        task_id_to_env = task_config["task_id_to_env"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError(f"Invalid benchmark configuration {config_path}: {error}") from error

    missing = []
    data_dir = os.path.join("data", env)
    for task_id in task_ids:
        env_type = task_id_to_env[str(task_id)]
        stem = f"{env}-{env_type}-{task_id}"
        for suffix in ("expert.pkl", "prompt-expert.pkl"):
            path = os.path.join(data_dir, f"{stem}-{suffix}")
            if not os.path.isfile(path):
                missing.append(path)
    return missing


def ensure_expert_data(env, generate_if_missing=False):
    """Validate a dataset and optionally invoke the expert-data generator."""
    missing = find_missing_expert_data(env)
    if not missing:
        print(f"Expert data ready: {env}")
        return

    if generate_if_missing:
        print(f"Expert data incomplete for {env}; starting generation.")
        subprocess.run(
            [sys.executable, "expert_data_generation.py", "--bench", env],
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
        "Generate it with --generate-expert-data, for example:\n"
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
                from main import experiment_mix_env

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
