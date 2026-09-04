import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import wandb

import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 18,          # base font size
    "axes.titlesize": 20,     # title
    "axes.labelsize": 18,     # x/y labels
    "xtick.labelsize": 14,    # tick labels
    "ytick.labelsize": 14,
    "legend.fontsize": 14,    # legend
})

# =========================
# Configuration
# =========================
entity = "ariyanbgd-vrije-universiteit-amsterdam"
base_dir = "./results"
target_metric = "train/avg/success_llm"

ALLOWED_METHODS = {"dt", "prompt_dt", "tenet", "tenet_NC", "tenet_MSE"}

STEP_SPACING = {
    "dt": 200,
    "prompt_dt": 200,
    "tenet": 200,
    "tenet_contrast": 200,
    "tenet_mse": 200,
}

DISPLAY_NAMES = {
    "dt": "DT",
    "prompt_dt": "Prompt-DT",
    "tenet": "TeNet",
    "tenet_contrast": "TeNet-Contrast",
    "tenet_mse": "TeNet-MSE",
}

METHOD_COLORS = {
    "dt": "#1f77b4",        # blue
    "prompt_dt": "#ff7f0e", # orange
    "tenet": "#2ca02c",  # green
    "tenet_contrast": "#d62728",     # red
    "tenet_mse": "#9467bd", # purple
}


GAME_ORDER = ["mt10-v3", "mt50-v3","mt50-ml45split-v3","ml1-v3-pick-place-v3"]
GAME_TITLES = {
    "mt10-v3": "MT10",
    "mt50-v3": "MT50",
    "mt50-ml45split-v3": "MT50-ML45Split",
    "ml1-v3-pick-place-v3": "ML1",
}

# =========================
# Fetch histories
# =========================
api = wandb.Api()
history_data = {}

available_games = [g for g in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, g))]
games_to_plot = [g for g in GAME_ORDER if g in available_games]

for game in games_to_plot:
    game_path = os.path.join(base_dir, game)
    project = f"project_metaworld_tenet_{game}"
    history_data[game] = {}

    print(f"🔍 Fetching runs from project: {project}")
    try:
        runs = api.runs(f"{entity}/{project}")
    except Exception as e:
        print(f"❌ Could not fetch runs for {project}: {e}")
        continue

    for method in os.listdir(game_path):
        if method not in ALLOWED_METHODS:
            continue

        method_path = os.path.join(game_path, method)
        if not os.path.isdir(method_path):
            continue

        all_runs = []
        for seed in os.listdir(method_path):
            expected_name = f"{method}_{seed}"
            matched = [r for r in runs if r.name == expected_name]
            if not matched:
                print(f"⚠️ Run {expected_name} not found in {project}")
                continue

            run = matched[0]
            try:
                history = run.history(keys=[target_metric])
                values = history[target_metric].dropna().values
                if len(values) > 0:
                    all_runs.append(values)
            except Exception as e:
                print(f"⚠️ Failed to fetch history for {expected_name}: {e}")

        if all_runs:
            history_data[game][method] = all_runs

# =========================
# Plot each game separately (without legend)
# =========================
for game in games_to_plot:
    methods = history_data.get(game, {})
    if not methods:
        print(f"ℹ️ No data found for game: {game}")
        continue

    plt.figure(figsize=(6, 4))
    plt.title(GAME_TITLES.get(game, game))
    plt.xlabel("Training Iteration")
    plt.ylabel("Success Rate")
    plt.grid(True, alpha=0.3)

    for method, runs in methods.items():
        if method not in ALLOWED_METHODS or not runs:
            continue

        min_len = min(len(r) for r in runs)
        if min_len == 0:
            continue

        aligned = np.array([r[:min_len] for r in runs], dtype=float)
        mean = np.mean(aligned, axis=0)
        std = np.std(aligned, axis=0)

        spacing = STEP_SPACING.get(method, 100)
        steps = np.arange(min_len) * spacing

        color = METHOD_COLORS.get(method, None)
        plt.plot(steps, mean, color=color)
        plt.fill_between(steps, mean - std, mean + std, alpha=0.25, color=color)

    plt.xlim(0, 5000)
    plt.tight_layout()
    plt.savefig(f"{game}.pdf")  # save each figure
    plt.close()
