# TeNet: Text-to-Network for Compact Policy Synthesis

Official Meta-World implementation for **TeNet: Text-to-Network for Compact
Policy Synthesis**, by Ariyan Bighashdel and Kevin Sebastian Luck
(IJCAI-ECAI 2026).

TeNet uses a pretrained language model to encode a task description and a
hypernetwork to instantiate a compact, task-specific control policy. The
language model is used when the policy is created; the generated policy then
acts directly on low-dimensional states without running the language model in
the control loop.

Project page: <https://sites.google.com/view/tenet-policy-synthesis>

## Scope of this release

The paper evaluates TeNet on MuJoCo locomotion and Meta-World manipulation.
This repository contains the **Meta-World experiments only**:

| Launcher name | Environment tag | Setting | In the paper |
| --- | --- | --- | --- |
| `ml1-pick-place` | `ml1-v3-pick-place-v3` | ML1 Pick-and-Place | Yes |
| `mt10` | `mt10-v3` | MT10 | Yes |
| `mt50` | `mt50-v3` | MT50 | Yes |
| `mt50-ml45split` | `mt50-ml45split-v3` | 45 training environments and 5 test environments | No; additional experiment |

The repository does not reproduce every experiment or ablation reported in the
paper.

## Implemented methods

| Launcher name | Method |
| --- | --- |
| `dt` | Decision Transformer without task prompts |
| `prompt_dt` | Prompt-DT conditioned on expert trajectory prompts |
| `tenet` | Direct TeNet without text-trajectory grounding |
| `tenet_contrast` | TeNet with contrastive text-trajectory grounding |
| `tenet_mse` | TeNet with MSE text-trajectory grounding |

The trajectory-processing and transformer backbone builds on
[Prompt-DT](https://github.com/mxu34/prompt-dt), which itself builds on
[Decision Transformer](https://github.com/kzl/decision-transformer). TeNet adds
the language encoder, text-conditioned hypernetwork, generated compact policy,
and optional text-trajectory grounding objectives. Attribution headers are
retained in the relevant source files.

## Repository structure

```text
train.py                    Public launcher for benchmarks and methods
training.py                 Data loading, model construction, and training loop
generate_expert_data.py     Scripted-expert dataset generation
tenet/                      Models, trainers, evaluation, and data utilities
metaworld/                  Meta-World environments and scripted policies
vector_quantize_pytorch/    Optional quantization components
config/                     Generated benchmark metadata (not tracked)
data/                       Generated expert trajectories (not tracked)
results/                    Run outputs and checkpoints (not tracked)
```

## Installation

The code was tested with Python 3.9 and the following central dependencies:

```text
torch==2.7.1+cu118
transformers==4.33.0
peft==0.13.2
wandb==0.19.9
numpy==2.0.2
gymnasium==1.1.1
mujoco==3.3.4
einops==0.8.1
scipy==1.13.1
```

MuJoCo, PyTorch, and CUDA installation details depend on the operating system
and GPU. A reproducible environment file will be added to this repository; in
the meantime, use the versions above because the inherited transformer and
Meta-World code are version-sensitive.

The TeNet presets use `meta-llama/Meta-Llama-3-8B` as a frozen text encoder.
Ensure that the model is accessible from the machine running the experiment
and that enough memory and storage are available for its initial download and
encoding pass.

## Quick start

List all available benchmarks and methods without importing the training
dependencies:

```bash
python train.py --list
```

Preview an experiment without generating data or starting training:

```bash
python train.py --benchmark mt10 --method tenet_contrast --dry-run
```

Run the default experiment (MT10 with TeNet-Contrast, seed 1001):

```bash
python train.py
```

`train.py` checks the required `config/` and `data/` files before training. If
they are missing or incomplete, the launcher generates them once with the
scripted Meta-World expert. Existing complete datasets are reused.

Expert generation can also be run explicitly:

```bash
python generate_expert_data.py --bench mt10-v3
```

To validate the data without automatically regenerating missing files:

```bash
python train.py --benchmark mt10 --method tenet_contrast \
  --no-generate-expert-data
```

Generated files are written to:

```text
config/<environment-tag>/
data/<environment-tag>/
```

For each successfully collected task, the generator saves a full expert
trajectory and a five-step prompt trajectory. Generation is deterministic with
the generator's fixed seed, but it can take substantial time for MT10 and MT50.

## Running experiments

Run one method on one benchmark:

```bash
python train.py --benchmark ml1-pick-place --method tenet
python train.py --benchmark mt10 --method prompt_dt
python train.py --benchmark mt50 --method tenet_mse
```

Run several methods or benchmarks sequentially:

```bash
python train.py --benchmark mt10 mt50 \
  --method dt prompt_dt tenet tenet_contrast tenet_mse
```

Run the full supported matrix for three seeds:

```bash
python train.py --benchmark all --method all --seeds 1001 2001 3001
```

The launcher applies configuration in the following order:

```text
shared defaults -> benchmark preset -> method preset -> command-line overrides
```

Useful runtime overrides include:

```bash
# Short smoke test without W&B
python train.py --benchmark mt10 --method tenet_contrast \
  --max-iters 1 --num-eval-episodes 1 --no-log-to-wandb

# Run on a different device
python train.py --benchmark mt10 --method dt --device cpu

# Save experiment metadata and model checkpoints
python train.py --benchmark mt10 --method tenet_contrast --save
```

Use `python train.py --help` for the complete launcher interface.

## Logging and outputs

W&B logging is enabled by the five method presets. Disable it with
`--no-log-to-wandb` when running locally or when W&B is unavailable.

Runs use the following directory layout when logging or checkpoint saving is
enabled:

```text
results/<environment-tag>/<method>/seed<seed>_<timestamp>/
```

Checkpoint saving is disabled by default. Enable it with `--save`. Depending
on the method, a run directory can contain experiment metadata, the trained
model, and a saved text encoder.

## Notes on evaluation

- The paper reports averages over three seeds and evaluates each task with 50
  rollouts.
- The launcher defaults to one evaluation episode to keep individual runs less
  expensive. Set `--num-eval-episodes 50` for the paper evaluation count.
- TeNet uses language descriptions at policy-instantiation time. The generated
  compact policy then executes without demonstrations or language-model calls.
- Prompt-DT uses expert prompt trajectories to identify the task.

## Acknowledgements

This implementation was developed from the Prompt-DT codebase and retains
components derived from Decision Transformer. It also includes Meta-World,
Hugging Face GPT-2-derived trajectory-model code, and optional components from
`vector-quantize-pytorch`. Please consult the upstream projects and the source
headers for their respective copyright and license terms.

Before redistributing a modified version of this repository, preserve all
applicable upstream notices. A repository-level license and consolidated
third-party notice are planned as part of the release preparation.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{bighashdel2026tenet,
  title     = {{TeNet}: Text-to-Network for Compact Policy Synthesis},
  author    = {Bighashdel, Ariyan and Luck, Kevin Sebastian},
  booktitle = {Proceedings of the Thirty-Fifth International Joint Conference
               on Artificial Intelligence and the Twenty-Ninth European
               Conference on Artificial Intelligence},
  year      = {2026}
}
```

The final proceedings citation and persistent paper identifier should replace
this provisional entry when they become available.
