import os
import argparse
import shutil
from types import SimpleNamespace
from main import experiment_mix_env 
from datetime import datetime

# models = [
#     {
#         "name": "prompt_dt",
#         "log_to_wandb": False,
#         "max_iters":5000,
#     },
# ]

# models = [
#     {
#         "name": "dt",
#         "log_to_wandb": True,
#         "no_prompt": True,
#         "max_iters":5000,
#     },
# ]


models = [
    {
        "name": "tenet_contrast",
        "log_to_wandb": True,
        "hyper_network": True,
        "hn_embed_dim": 256,
        "consistency_regularizer": True,
        "cr_coeff": 0.0001,
        "llm": True,
        "normalize_embeddings": True,
        "llm_preencoded": True,
        "use_embedding_contrastive": True,
        "only_llm_evaluation":True,
        "max_iters":5000,
    }
]

# models = [
#     {
#         "name": "tenet",
#         "log_to_wandb": Flase,
#         "hyper_network": True,
#         "hn_embed_dim": 256,
#         "consistency_regularizer": True,
#         "cr_coeff": 0.0001,
#         "llm": True,
#         "normalize_embeddings": True,
#         "llm_preencoded": True,
#         "only_llm_evaluation":True,
#         "dual_policy": True,
#         "max_iters":5000,
#     }
# ]


# models = [
#     {
#         "name": "tenet_mse",
#         "log_to_wandb": False,
#         "hyper_network": True,
#         "hn_embed_dim": 256,
#         "consistency_regularizer": True,
#         "cr_coeff": 0.0001,
#         "llm": True,
#         "normalize_embeddings": True,
#         "llm_preencoded": True,
#         "use_embedding_mse": True,
#         "only_llm_evaluation":True,
#         "max_iters":5000,
#     }
# ]

seeds = [1001]
envs = ['ml1-v3-pick-place-v3'] # mt10-v3 , mt50-v3, mt50-ml45split-v3, 'ml1-v3-pick-place-v3'

BATCH_SIZE_MAP = {
    "mt10-v3": 10,
    "mt50-v3": 50,
    "ml1-v3": 32,
    "mt50-ml45split-v3": 45
}

for env in envs:
    project = f"project_metaworld_tenet_{env}"
    base_result_dir = os.path.join("results", env)
    
    base_env_type = None
    if env.startswith("mt10-v3"):
        base_env_type = "mt10-v3"
    elif env.startswith("mt50-v3"):
        base_env_type = "mt50-v3"
    elif env.startswith("ml1-v3"):
        base_env_type = "ml1-v3"
    elif env.startswith("mt50-ml45split-v3"):
        base_env_type = "mt50-ml45split-v3"
    else:
        raise ValueError(f"Unknown env type for batch size mapping: {env}")

    for model in models:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_name = model["name"]
        for seed in seeds:
            print(f"\n=== Env: {env}, Running model: {model_name}, seed: {seed} ===")
            run_dir = os.path.join(base_result_dir, model_name, f"seed{seed}_{now}")
            

            # build args (as Namespace)
            args = argparse.Namespace(
                env=env,
                dataset_mode="expert",
                test_dataset_mode="expert",
                train_prompt_mode="expert",
                test_prompt_mode="expert",
                seed=seed,
                prompt_episode=1,
                prompt_length=5, # Here
                stochastic_prompt=True,
                no_prompt=False,
                no_r=False,
                no_rtg=False,
                finetune=False,
                finetune_steps=10,
                finetune_batch_size=256,
                finetune_opt=True,
                finetune_lr=1e-4,
                no_state_normalize=False,
                average_state_mean=True,
                evaluation=False,
                load_path=None,  # optional if not loading
                llm=False,
                delayed_llm=False,
                delayed_llm_iteration=2500,
                llm_model="meta-llama/Meta-Llama-3-8B",
                llm_finetune_method="none",
                llm_preencoded=False,
                normalize_embeddings=False,
                llm_pooling_type="eos",
                llm_projection_type="mlp",
                num_projection_layers=2,
                dual_policy=False,
                llm_goal_prediction=False,
                llm_goal_pred_coeff=0.1,
                only_llm_evaluation=False,
                use_embedding_contrastive=False,
                use_text_embedding_contrastive=False,
                contrastive_coeff=1.0,
                text_contrastive_coeff=0.1,
                contrastive_temperature=0.07,
                use_embedding_mse=False,
                embedding_mse_coeff=0.1,
                use_embedding_discriminator=False,
                discriminator_coeff=0.1,
                hyper_network=False,
                hn_embed_dim=128,
                hypernet_layers=[128, 128],
                policy_hidden_layers=[128, 128],
                consistency_regularizer=False,
                cr_coeff=0.0001,
                quantized_embed=False,
                quantized_embed_for_loss=False,
                quant_level=21,
                mode="normal",
                K=20,
                pct_traj=1.0,
                batch_size=BATCH_SIZE_MAP[base_env_type],
                embed_dim=128,
                n_layer=3,
                n_head=1,
                activation_function="relu",
                dropout=0.1,
                learning_rate=1e-4,
                weight_decay=1e-4,
                warmup_steps=10000,
                num_eval_episodes=1, # Here
                max_iters=50000,
                num_steps_per_iter=10,
                device="cuda",
                log_to_wandb=False,
                train_eval_interval=200,
                test_eval_interval=200,
                save=False,
            )

            # override with model-specific settings
            for k, v in model.items():
                setattr(args, k, v)

            # run and save in subfolder
            experiment_mix_env(args, variant=vars(args), run_dir=run_dir, model_name=f"{model_name}_seed{seed}_{now}", project=project)
