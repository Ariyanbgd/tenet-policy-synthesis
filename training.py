"""Training orchestration shared by the TENET experiment launcher.

The trajectory loading, prompting, and training flow is inherited from
Prompt-DT and extended here with TENET's text-conditioned policy options.
"""

import argparse
import gc
import itertools
import json
import os
import pickle
import uuid
from dataclasses import dataclass

import torch
import wandb

from tenet.prompt_seq_trainer import PromptSequenceTrainer
from tenet.prompt_utils import (
    eval_episodes,
    eval_episodes_llm,
    get_batch,
    get_env_list,
    get_prompt,
    get_prompt_batch,
    group_task_indices_by_type,
    load_data_prompt,
    process_info,
    process_total_data_mean,
    set_seed,
    text_encoding,
)
from tenet.tenet import Predictor


CONFIG_PATHS = {
    "mt50-v3": "mt50-v3/mt50-v3.json",
    "mt10-v3": "mt10-v3/mt10-v3.json",
    "ml1-v3-pick-place-v3": (
        "ml1-v3-pick-place-v3/ml1-v3-pick-place-v3.json"
    ),
    "ml1-v3-reach-v3": "ml1-v3-reach-v3/ml1-v3-reach-v3.json",
    "mt50-ml45split-v3": (
        "mt50-ml45split-v3/mt50-ml45split-v3.json"
    ),
}


@dataclass
class ExperimentData:
    """Environment, trajectory, and metadata objects used during training."""

    base_env: object
    test_base_env: object
    info: dict
    test_info: dict
    train_env_names: list
    test_env_names: list
    trajectories: list
    test_trajectories: list
    prompt_trajectories: list
    test_prompt_trajectories: list
    train_env_groups: dict
    test_env_groups: dict


def load_task_config(env_name, config_root):
    """Load the generated task split for one benchmark."""
    config_path = os.path.join(config_root, CONFIG_PATHS[env_name])
    with open(config_path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def build_environment_names(env_name, task_config, split):
    """Build the full environment identifiers expected by Prompt-DT."""
    task_id_to_env = task_config["task_id_to_env"]
    return [
        f"{env_name}-{task_id_to_env[str(task_id)]}-{task_id}"
        for task_id in task_config[split]
    ]


def load_experiment_data(args, variant, project_root, device):
    """Construct environments and load/process train and test trajectories."""
    config_root = os.path.join(project_root, "config")
    data_root = os.path.join(project_root, "data")
    task_config = load_task_config(args.env, config_root)

    train_env_names = build_environment_names(
        args.env, task_config, "train_tasks"
    )
    test_env_names = build_environment_names(
        args.env, task_config, "test_tasks"
    )
    info, base_env = get_env_list(train_env_names, config_root, device)
    test_info, test_base_env = get_env_list(test_env_names, config_root, device)

    trajectories, prompt_trajectories = load_data_prompt(
        train_env_names,
        data_root,
        variant["dataset_mode"],
        variant["train_prompt_mode"],
        args,
    )
    test_trajectories, test_prompt_trajectories = load_data_prompt(
        test_env_names,
        data_root,
        variant["test_dataset_mode"],
        variant["test_prompt_mode"],
        args,
    )

    mode = variant.get("mode", "normal")
    if variant["average_state_mean"]:
        train_total = list(itertools.chain.from_iterable(trajectories))
        test_total = list(itertools.chain.from_iterable(test_trajectories))
        total_trajectories = train_total + test_total
        print(len(total_trajectories))
        total_state_mean, total_state_std = process_total_data_mean(
            total_trajectories, mode
        )
        variant["total_state_mean"] = total_state_mean
        variant["total_state_std"] = total_state_std

    train_env_groups = group_task_indices_by_type(train_env_names)
    test_env_groups = group_task_indices_by_type(test_env_names)
    pct_traj = variant.get("pct_traj", 1.0)
    info = process_info(
        train_env_names,
        train_env_groups,
        trajectories,
        info,
        mode,
        variant["dataset_mode"],
        pct_traj,
        variant,
    )
    test_info = process_info(
        test_env_names,
        test_env_groups,
        test_trajectories,
        test_info,
        mode,
        variant["test_dataset_mode"],
        pct_traj,
        variant,
    )

    return ExperimentData(
        base_env=base_env,
        test_base_env=test_base_env,
        info=info,
        test_info=test_info,
        train_env_names=train_env_names,
        test_env_names=test_env_names,
        trajectories=trajectories,
        test_trajectories=test_trajectories,
        prompt_trajectories=prompt_trajectories,
        test_prompt_trajectories=test_prompt_trajectories,
        train_env_groups=train_env_groups,
        test_env_groups=test_env_groups,
    )


def build_default_model_name(variant):
    """Reproduce the legacy descriptive model name."""
    model_name = (
        f"seed_{variant['seed']}_TRAIN_{variant['train_prompt_mode']}"
        f"_TEST_{variant['test_prompt_mode']}"
    )
    if variant["no_prompt"]:
        model_name += "_NO_PROMPT"
    if variant["finetune"]:
        model_name += "_FINETUNE"
    if variant["no_r"]:
        model_name += "_NO_R"
    if variant["hyper_network"]:
        model_name += "_HN"
    if variant["consistency_regularizer"]:
        model_name += "_CR"
    if variant["llm"]:
        model_name += (
            f"_LLM_{variant['llm_finetune_method']}"
            f"_{variant['llm_projection_type']}"
        )
        if variant["llm_projection_type"] == "mlp":
            model_name += str(variant["num_projection_layers"])
        if variant["llm_preencoded"]:
            model_name += "_preencoded"
        if variant["use_embedding_contrastive"]:
            model_name += f"_contrastive{variant['contrastive_coeff']}"
        if variant["use_embedding_mse"]:
            model_name += f"_mse{variant['embedding_mse_coeff']}"
        if variant["use_embedding_discriminator"]:
            model_name += f"_discriminator{variant['discriminator_coeff']}"
    return model_name


def save_experiment_metadata(args, variant, run_dir, data):
    """Save the legacy checkpoint metadata before training starts."""
    checkpoint = {
        "args": args,
        "variant": variant,
        "env": (
            data.base_env,
            data.test_base_env,
            data.info,
            data.test_info,
            data.train_env_names,
            data.test_env_names,
            data.trajectories,
            data.test_trajectories,
            data.prompt_trajectories,
            data.test_prompt_trajectories,
            data.train_env_groups,
            data.test_env_groups,
        ),
    }
    with open(os.path.join(run_dir, "checkpoint.pkl"), "wb") as checkpoint_file:
        pickle.dump(checkpoint, checkpoint_file)


def build_model(args, variant, data, device, run_dir):
    """Construct TENET and optionally pre-encode task descriptions."""
    first_env = data.train_env_names[0]
    state_dim = data.info[first_env]["state_dim"]
    act_dim = data.info[first_env]["act_dim"]
    model = Predictor(
        args,
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=variant["K"],
        max_ep_len=1000,
        hidden_size=variant["embed_dim"],
        n_layer=variant["n_layer"],
        n_head=variant["n_head"],
        n_inner=4 * variant["embed_dim"],
        activation_function=variant["activation_function"],
        n_positions=1024,
        resid_pdrop=variant["dropout"],
        attn_pdrop=variant["dropout"],
        device=device,
    ).to(device=device)

    if variant["llm_preencoded"]:
        assert (
            variant.get("llm")
            and variant.get("llm_finetune_method") == "none"
            and variant.get("llm_pooling_type") == "eos"
        ), "Invalid variant config for text encoding"
        data.info, _ = text_encoding(
            model, data.info, data.train_env_names, device
        )
        data.test_info, _ = text_encoding(
            model, data.test_info, data.test_env_names, device
        )
        if variant["save"]:
            text_encoder = model.text_encoder.to("cpu")
            torch.save(text_encoder, os.path.join(run_dir, "model_text_encoder.pt"))
            del text_encoder

        model.text_encoder = None
        gc.collect()
        torch.cuda.empty_cache()
    return model


def build_trainer(args, variant, data, model, device):
    """Construct the optimizer, schedule, and Prompt-DT sequence trainer."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant["learning_rate"],
        weight_decay=variant["weight_decay"],
    )
    warmup_steps = variant["warmup_steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1),
    )
    first_env = data.train_env_names[0]
    return PromptSequenceTrainer(
        args,
        model=model,
        optimizer=optimizer,
        batch_size=variant["batch_size"],
        get_batch=get_batch(
            data.trajectories[0], data.info[first_env], variant
        ),
        scheduler=scheduler,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean(
            (a_hat - a) ** 2
        ),
        eval_fns=None,
        get_prompt=get_prompt(
            data.prompt_trajectories[0], data.info[first_env], variant
        ),
        get_prompt_batch=get_prompt_batch(
            data.trajectories,
            data.prompt_trajectories,
            data.info,
            variant,
            data.train_env_names,
            data.train_env_groups,
        ),
        device=device,
    )


def evaluate_split(
    trainer,
    prompt_trajectories,
    env_names,
    env_groups,
    info,
    variant,
    base_env,
    iteration,
    group,
    no_prompt,
):
    """Evaluate one train/test environment split."""
    return trainer.eval_iteration_multienv(
        get_prompt,
        prompt_trajectories,
        eval_episodes,
        eval_episodes_llm,
        env_names,
        env_groups,
        info,
        variant,
        base_env,
        iter_num=iteration + 1,
        print_logs=True,
        no_prompt=no_prompt,
        group=group,
    )


def maybe_save_best_model(trainer, outputs, variant, args, run_dir, iteration):
    """Apply the checkpoint-selection behavior from the original loop."""
    if not (
        variant["save"]
        and iteration % args.test_eval_interval == 0
        and iteration % args.train_eval_interval == 0
    ):
        return

    if variant["llm"]:
        test_success = outputs.get("test/avg/success_llm", 0)
        train_success = outputs.get("train/avg/success_llm", 0)
    else:
        test_success = outputs.get("test/avg/success", 0)
        train_success = outputs.get("train/avg/success", 0)
    current_success_score = test_success + train_success

    # Keep the original comparison behavior. The score was not updated after a
    # save in the research code, so every evaluated score exceeds this value.
    best_success_score = float("-inf")
    if current_success_score > best_success_score:
        trainer.save_model(run_dir)


def run_training_loop(trainer, args, variant, data, run_dir, log_to_wandb):
    """Run training, periodic evaluation, checkpointing, and metric logging."""
    for iteration in range(variant["max_iters"]):
        no_llm = not args.llm
        if args.delayed_llm and iteration < args.delayed_llm_iteration:
            no_llm = True
        outputs = trainer.pure_train_iteration_mix(
            num_steps=variant["num_steps_per_iter"],
            no_prompt=args.no_prompt,
            no_llm=no_llm,
        )

        if (
            data.test_env_names
            and iteration % args.test_eval_interval == 0
        ):
            outputs.update(
                evaluate_split(
                    trainer,
                    data.test_prompt_trajectories,
                    data.test_env_names,
                    data.test_env_groups,
                    data.test_info,
                    variant,
                    data.test_base_env,
                    iteration,
                    "test",
                    args.no_prompt,
                )
            )
        if iteration % args.train_eval_interval == 0:
            outputs.update(
                evaluate_split(
                    trainer,
                    data.prompt_trajectories,
                    data.train_env_names,
                    data.train_env_groups,
                    data.info,
                    variant,
                    data.base_env,
                    iteration,
                    "train",
                    args.no_prompt,
                )
            )

        maybe_save_best_model(
            trainer, outputs, variant, args, run_dir, iteration
        )
        outputs.update({"global_step": iteration})
        if log_to_wandb:
            wandb.log(outputs)


def experiment_mix_env(
    args,
    variant,
    run_dir=None,
    model_name=None,
    project=None,
    wandb_mode="online",
):
    """Prepare and run one configured Meta-World training experiment."""
    project_root = os.getcwd()
    set_seed(variant["seed"])
    device = variant["device"]
    log_to_wandb = variant["log_to_wandb"]

    data = load_experiment_data(args, variant, project_root, device)
    if model_name is None:
        model_name = build_default_model_name(variant)
    if run_dir is None:
        run_dir = os.path.join(project_root, "model_saved", args.env, model_name)
    if log_to_wandb or variant["save"]:
        os.makedirs(run_dir, exist_ok=True)
    if variant["save"]:
        save_experiment_metadata(args, variant, run_dir, data)

    model = build_model(args, variant, data, device, run_dir)
    trainer = build_trainer(args, variant, data, model, device)

    if log_to_wandb:
        if project is None:
            project = f"Demo_{variant['env']}"
        wandb.init(
            name=model_name,
            project=project,
            config=variant,
            dir=run_dir,
            mode="online",
            id=str(uuid.uuid4()),
            resume="never",
        )

    run_training_loop(trainer, args, variant, data, run_dir, log_to_wandb)
    if log_to_wandb:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='mt10-v3')
    parser.add_argument('--dataset_mode', type=str, default='expert')
    parser.add_argument('--test_dataset_mode', type=str, default='expert')
    parser.add_argument('--train_prompt_mode', type=str, default='expert')
    parser.add_argument('--test_prompt_mode', type=str, default='expert')
    parser.add_argument("--seed", type=int, default=1001, help="random seed")

    parser.add_argument('--prompt-episode', type=int, default=1)
    parser.add_argument('--prompt-length', type=int, default=5) # 
    parser.add_argument('--stochastic-prompt', action='store_true', default=True)
    parser.add_argument('--no-prompt', action='store_true', default=False)
    parser.add_argument('--no-r', action='store_true', default=False)
    parser.add_argument('--no-rtg', action='store_true', default=False)
    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--finetune_steps', type=int, default=10)
    parser.add_argument('--finetune_batch_size', type=int, default=256)
    parser.add_argument('--finetune_opt', action='store_true', default=True)
    parser.add_argument('--finetune_lr', type=float, default=1e-4)
    parser.add_argument('--no_state_normalize', action='store_true', default=False) 
    parser.add_argument('--average_state_mean', action='store_true', default=True) 
    parser.add_argument('--evaluation', action='store_true', default=False) 
    parser.add_argument('--load-path', type=str, default= 'ML1-pick-place-v2/seed_1001_TRAIN_expert_TEST_expert_HN_CR_LLM_none_mlp2_preencoded_contrastive1') 
    
    parser.add_argument('--llm', action='store_true', default=True) # False
    parser.add_argument('--delayed-llm', action='store_true', default=False)
    parser.add_argument('--delayed-llm-iteration', type=int, default=2500) 
    parser.add_argument('--llm-model', type=str, default= "meta-llama/Meta-Llama-3-8B")
    parser.add_argument('--llm_finetune_method', type=str, default= "none") # none, lora, full
    parser.add_argument('--llm-preencoded', action='store_true', default=False) # False
    parser.add_argument('--normalize-embeddings', action='store_true', default=False) # False
    parser.add_argument('--llm_pooling_type', type=str, default= "eos") # 'eos', 'attention', 'mean', 'transformer'
    parser.add_argument('--llm_projection_type', type=str, default= "mlp") # 'linear', 'mlp'
    parser.add_argument('--num_projection_layers', type=int, default=2) 
    parser.add_argument('--dual-policy', action='store_true', default=False) # False
    parser.add_argument('--llm_goal_prediction', action='store_true', default=False) # False goal_pred_coeff
    parser.add_argument('--llm_goal_pred_coeff', type=float, default=0.1)
    parser.add_argument('--only_llm_evaluation', action='store_true', default=True)
    
    
    parser.add_argument('--use_embedding_contrastive', action='store_true', default=False) # False
    parser.add_argument('--use_text_embedding_contrastive', action='store_true', default=False) 
    parser.add_argument('--contrastive-coeff', type=float, default=1)
    parser.add_argument('--text_contrastive_coeff', type=float, default=0.1)
    parser.add_argument('--contrastive-temperature', type=float, default=0.07)
    parser.add_argument('--use_embedding_mse', action='store_true', default=False) # False
    parser.add_argument('--embedding_mse_coeff', type=float, default=0.1)
    parser.add_argument('--use_embedding_discriminator', action='store_true', default=False) # False
    parser.add_argument('--discriminator_coeff', type=float, default=0.1)
    
    parser.add_argument('--hyper-network', action='store_true', default=True) # False
    parser.add_argument('--hn_embed_dim', type=int, default=128)
    parser.add_argument('--hypernet-layers', type=lambda s: [int(x) for x in s.split(',')],default=[128, 128]) # 128,128
    parser.add_argument('--policy-hidden-layers', type=lambda s: [int(x) for x in s.split(',')],default=[128, 128])
    parser.add_argument('--consistency-regularizer', action='store_true', default=False) # False
    parser.add_argument('--cr-coeff', type=float, default=0.0001)
    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=32) # 32
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000) 
    parser.add_argument('--num_eval_episodes', type=int, default=1) #50
    parser.add_argument('--max_iters', type=int, default=50000) 
    parser.add_argument('--num_steps_per_iter', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb',  action='store_true', default=False) # True
    parser.add_argument('--train_eval_interval', type=int, default=1000)
    parser.add_argument('--test_eval_interval', type=int, default=100)
    parser.add_argument('--save',  action='store_true', default=False) #500

    args = parser.parse_args()
    experiment_mix_env(args,variant=vars(args))
    
    
#######
