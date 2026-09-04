from ast import parse
import numpy as np
import torch
import wandb

import argparse
import pickle
import itertools
import pickle
import uuid

from tenet.prompt_decision_transformer import Predictor
from tenet.prompt_seq_trainer import PromptSequenceTrainer
from tenet.prompt_utils import get_env_list, set_seed, group_task_indices_by_type
from tenet.prompt_utils import get_prompt_batch, get_prompt, get_batch, get_batch_finetune, flatten_prompt
from tenet.prompt_utils import process_total_data_mean, load_data_prompt, process_info, text_encoding
from tenet.prompt_utils import eval_episodes, eval_episodes_llm

from collections import namedtuple
import json, pickle, os
import webbrowser

def experiment_mix_env(args, variant, run_dir=None, model_name=None, project=None, wandb_mode='online'):
    cur_dir = os.getcwd()
    set_seed(variant['seed'])
    device = variant['device']
    log_to_wandb = variant['log_to_wandb']

    ######
    # construct train and test environments
    ######
    
    config_save_path = os.path.join(cur_dir, 'config')
    data_save_path = os.path.join(cur_dir, 'data')
    

    config_path_dict = {
        'mt50-v3': "mt50-v3/mt50-v3.json",
        'mt10-v3': "mt10-v3/mt10-v3.json",
        'ml1-v3-pick-place-v3': "ml1-v3-pick-place-v3/ml1-v3-pick-place-v3.json",
        'ml1-v3-reach-v3': "ml1-v3-reach-v3/ml1-v3-reach-v3.json",
        'mt50-ml45split-v3': "mt50-ml45split-v3/mt50-ml45split-v3.json"
    }
    
    task_config = os.path.join(config_save_path, config_path_dict[args.env])
    with open(task_config, 'r') as f:
        task_config = json.load(f)

    train_env_name_list, test_env_name_list = [], []

    # Use the task_id_to_env mapping
    task_id_to_env = task_config["task_id_to_env"]

    # Add full env name (e.g., mt50-v3-reach-v2-0)
    for task_ind in task_config["train_tasks"]:
        env_type = task_id_to_env[str(task_ind)]  # keys are strings in JSON
        train_env_name_list.append(f"{args.env}-{env_type}-{task_ind}")

    for task_ind in task_config["test_tasks"]:
        env_type = task_id_to_env[str(task_ind)]
        test_env_name_list.append(f"{args.env}-{env_type}-{task_ind}")
    # training envs
    info, base_env = get_env_list(train_env_name_list, config_save_path, device)
    # testing envs
    test_info, test_base_env = get_env_list(test_env_name_list, config_save_path, device)


    K = variant['K']
    batch_size = variant['batch_size']
    pct_traj = variant.get('pct_traj', 1.)
    mode = variant.get('mode', 'normal')
    dataset_mode = variant['dataset_mode']
    test_dataset_mode = variant['test_dataset_mode']
    train_prompt_mode = variant['train_prompt_mode']
    test_prompt_mode = variant['test_prompt_mode']

    # load training dataset
    trajectories_list, prompt_trajectories_list = load_data_prompt(train_env_name_list, data_save_path, dataset_mode, train_prompt_mode, args)
    # load testing dataset
    test_trajectories_list, test_prompt_trajectories_list = load_data_prompt(test_env_name_list, data_save_path, test_dataset_mode, test_prompt_mode, args)

    # change to total train trajecotry 
    if variant['average_state_mean']:
        train_total = list(itertools.chain.from_iterable(trajectories_list))
        test_total = list(itertools.chain.from_iterable(test_trajectories_list))
        total_traj_list = train_total + test_total
        print(len(total_traj_list))
        total_state_mean, total_state_std= process_total_data_mean(total_traj_list, mode)
        variant['total_state_mean'] = total_state_mean
        variant['total_state_std'] = total_state_std
        
    train_env_name_grouped = group_task_indices_by_type(train_env_name_list)
    test_env_name_grouped = group_task_indices_by_type(test_env_name_list)

    # process train info
    info = process_info(train_env_name_list, train_env_name_grouped, trajectories_list, info, mode, dataset_mode, pct_traj, variant)
    # process test info
    test_info = process_info(test_env_name_list, test_env_name_grouped, test_trajectories_list, test_info, mode, test_dataset_mode, pct_traj, variant)

    # construct model post fix
    if model_name is None:
        model_name = 'seed_' + str(variant['seed']) + '_TRAIN_'+variant['train_prompt_mode']+'_TEST_'+variant['test_prompt_mode']
        if variant['no_prompt']:
            model_name += '_NO_PROMPT'
        if variant['finetune']:
            model_name += '_FINETUNE'
        if variant['no_r']:
            model_name += '_NO_R'
        if variant['hyper_network']:
            model_name += '_HN'
        if variant['consistency_regularizer']:
            model_name += '_CR'
        if variant['llm']:
            model_name += '_LLM_' + variant['llm_finetune_method'] + '_' + variant['llm_projection_type']
            if variant['llm_projection_type'] == 'mlp':
                model_name += str(variant['num_projection_layers'])
            if variant['llm_preencoded']:
                model_name += '_preencoded'
            if variant['use_embedding_contrastive']:
                model_name += '_contrastive' + str(variant['contrastive_coeff'])
            if variant['use_embedding_mse']:
                model_name += '_mse' + str(variant['embedding_mse_coeff'])
            if variant['use_embedding_discriminator']:
                model_name += '_discriminator' + str(variant['discriminator_coeff'])
        
    
    if run_dir is None:
        run_dir = os.path.join(cur_dir, 'model_saved',args.env,model_name)

    if log_to_wandb or variant["save"]:
        os.makedirs(run_dir, exist_ok=True)


    ######
    # saving info
    ######
    
    if variant['save']:
        checkpoint = {}
        checkpoint['args'] = args
        checkpoint['variant'] = variant
        checkpoint['env'] = (
            base_env, test_base_env,
            info, test_info, 
            train_env_name_list, test_env_name_list,
            trajectories_list, test_trajectories_list , 
            prompt_trajectories_list, test_prompt_trajectories_list,
            train_env_name_grouped, test_env_name_grouped
        )
    
        with open(os.path.join(run_dir, 'checkpoint.pkl'), 'wb') as f:
            pickle.dump(checkpoint, f)

    ######
    # construct dt model and trainer
    ######

    state_dim = info[train_env_name_list[0]]['state_dim']
    act_dim = info[train_env_name_list[0]]['act_dim']

    model = Predictor(
        args,
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=K,
        max_ep_len=1000,
        hidden_size=variant['embed_dim'],
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        n_inner=4 * variant['embed_dim'],
        activation_function=variant['activation_function'],
        n_positions=1024,
        resid_pdrop=variant['dropout'],
        attn_pdrop=variant['dropout'],
        device=device
    )
    model = model.to(device=device)
    
    if variant['llm_preencoded']:
        assert (
            variant.get('llm') and
            variant.get('llm_finetune_method') == 'none' and
            variant.get('llm_pooling_type') == 'eos'
        ), "Invalid variant config for text encoding"
        info, text_embeddings = text_encoding(model,info,train_env_name_list,device)
        test_info, test_text_embeddings = text_encoding(model,test_info,test_env_name_list,device)
        if variant['save']:
            enc_cpu = model.text_encoder.to('cpu')
            model_path = os.path.join(run_dir, 'model_text_encoder.pt')
            torch.save(enc_cpu, model_path)
            del enc_cpu  

        model.text_encoder = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    env_name = train_env_name_list[0]
    trainer = PromptSequenceTrainer(
        args,
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=get_batch(trajectories_list[0], info[env_name], variant),
        scheduler=scheduler,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
        eval_fns=None,
        get_prompt=get_prompt(prompt_trajectories_list[0], info[env_name], variant),
        get_prompt_batch=get_prompt_batch(trajectories_list, prompt_trajectories_list, info, variant, train_env_name_list, train_env_name_grouped),
        device=device
    )

    ######
    # start training
    ######

                
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
            resume="never"
        )
    
    best_success_score = float('-inf')
    for iter in range(variant['max_iters']):
        no_llm = not args.llm
        if args.delayed_llm and iter < args.delayed_llm_iteration:
            no_llm = True
        outputs = trainer.pure_train_iteration_mix(
            num_steps=variant['num_steps_per_iter'], 
            no_prompt=args.no_prompt,
            no_llm=no_llm
            )

        
        # start evaluation
        if len(test_env_name_list)>0 and (iter % args.test_eval_interval) == 0:
            # evaluate test
            test_eval_logs = trainer.eval_iteration_multienv(
                get_prompt, test_prompt_trajectories_list,
                eval_episodes, eval_episodes_llm, test_env_name_list, test_env_name_grouped, test_info, variant, test_base_env, iter_num=iter + 1, 
                print_logs=True, no_prompt=args.no_prompt, group='test')
            outputs.update(test_eval_logs)
        if iter % args.train_eval_interval == 0:
            # evaluate train
            train_eval_logs = trainer.eval_iteration_multienv(
                get_prompt, prompt_trajectories_list,
                eval_episodes, eval_episodes_llm, train_env_name_list, train_env_name_grouped, info, variant, base_env, iter_num=iter + 1, 
                print_logs=True, no_prompt=args.no_prompt, group='train')
            outputs.update(train_eval_logs)
            
        if variant['save'] and iter % args.test_eval_interval == 0 and iter % args.train_eval_interval == 0:
            if variant['llm']:
                test_success = outputs.get('test/avg/success_llm', 0)
                train_success = outputs.get('train/avg/success_llm', 0)
            else:
                test_success = outputs.get('test/avg/success', 0)
                train_success = outputs.get('train/avg/success', 0)
            current_success_score = test_success + train_success
            if current_success_score > best_success_score:
                trainer.save_model(run_dir)
                


        outputs.update({"global_step": iter}) # set global step as iteration

        if log_to_wandb:
            wandb.log(outputs)
            
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
    parser.add_argument('--quantized_embed', action='store_true', default=False)
    parser.add_argument('--quantized_embed_for_loss', action='store_true', default=False)
    parser.add_argument('--quant-level', type=int, default=21)

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
