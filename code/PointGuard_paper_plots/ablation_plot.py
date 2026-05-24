#%%
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

json_path = r"G:\我的云端硬盘\THESIS\Pointnet_Pointnet2_pytorch-master\log\pointguard\pointguard_classification_mix\epoch_10_npoint_16_bsize_64\numerical result\ablation.json"

with open(json_path, "r") as f:
    data = json.load(f)

scenario_names = {
    "mixed": "Mixed Attack",
    "small": "Small Attack",
    "extreme": "Extreme Attack",
    "doppler": "Unseen Attack"
}

scenario_order = ["mixed", "small", "extreme", "doppler"]
models = ["pointnet_seg", "pointguard"]
model_colors = ["#4C72B0", "#DD8452"]

#%%
#   AUC
# ==================================================
# collect values from json
plot_values = {m: [] for m in models}

for scenario in scenario_order:
   for model in models:
    item = next(
        (
            d for d in data
            if d["attack_scenario"] == scenario
            and d["baseline"] == model
        ),
        None
    )

    if item is None:
        print(f"Missing: scenario={scenario}, baseline={model}")
        plot_values[model].append(0)
        continue

    plot_values[model].append(item["mean_auc"])

x = np.arange(len(scenario_order))
width = 0.35

plt.figure(figsize=(12, 8))

for i, model in enumerate(models):
    offset = (i - 0.5) * width

    plt.bar(
        x + offset,
        plot_values[model],
        width,
        color=model_colors[i],
        label=model
    )

plt.xticks(
    x,
    [scenario_names[s] for s in scenario_order],
    fontsize=22
)

plt.yticks(fontsize=24)

plt.ylabel("AUC", fontsize=24)

plt.legend(fontsize=20)

plt.gca().set_axisbelow(True)
plt.grid(axis='y', linestyle='-', alpha=0.5)

plt.tight_layout()
plt.show()


#%%
#   F1
# ==================================================
# collect values from json
plot_values = {m: [] for m in models}

for scenario in scenario_order:
   for model in models:
    item = next(
        (
            d for d in data
            if d["attack_scenario"] == scenario
            and d["baseline"] == model
        ),
        None
    )

    if item is None:
        print(f"Missing: scenario={scenario}, baseline={model}")
        plot_values[model].append(0)
        continue

    plot_values[model].append(item["mean_f1"])

x = np.arange(len(scenario_order))
width = 0.35

plt.figure(figsize=(12, 8))

for i, model in enumerate(models):
    offset = (i - 0.5) * width

    plt.bar(
        x + offset,
        plot_values[model],
        width,
        color=model_colors[i],
        label=model
    )

plt.xticks(
    x,
    [scenario_names[s] for s in scenario_order],
    fontsize=22
)

plt.yticks(fontsize=22)
plt.ylim([0,1])

plt.ylabel("F1", fontsize=24)

plt.legend(fontsize=20)

plt.gca().set_axisbelow(True)
plt.grid(axis='y', linestyle='-', alpha=0.5)

plt.tight_layout()
plt.show()

#%%
#%%
# same color for same model
model_colors = {
    "pointnet_seg": "#5FB581",
    "pointguard": "#E77D45"
}

# different hatch for AUC / F1
metric_hatch = {
    "auc": "//",
    "f1": ".."
}

# ==================================================
# Collect values
# ==================================================
auc_values = {m: [] for m in models}
f1_values = {m: [] for m in models}

for scenario in scenario_order:
    for model in models:

        item = next(
            (
                d for d in data
                if d["attack_scenario"] == scenario
                and d["baseline"] == model
            ),
            None
        )

        if item is None:
            print(f"Missing: scenario={scenario}, baseline={model}")
            auc_values[model].append(0)
            f1_values[model].append(0)
            continue

        auc_values[model].append(item["mean_auc"])
        f1_values[model].append(item["mean_f1"])

# ==================================================
# Plot
# ==================================================
x = np.arange(len(scenario_order))
width = 0.18
fig, ax1 = plt.subplots(figsize=(12, 8))
ax2 = ax1.twinx()

# AUC bars (left axis)
for i, model in enumerate(models):

    offset = (-1.5 + i) * width

    ax1.bar(
        x + offset,
        auc_values[model],
        width=width,
        color=model_colors[model],
        hatch=metric_hatch["auc"],
        edgecolor='black',
        linewidth=1.2,
        label=f"{model} AUC"
    )

# F1 bars (right axis)
for i, model in enumerate(models):

    offset = (0.5 + i) * width

    ax2.bar(
        x + offset,
        f1_values[model],
        width=width,
        color=model_colors[model],
        hatch=metric_hatch["f1"],
        edgecolor='black',
        linewidth=1.2,
        alpha=0.9,
        label=f"{model} F1"
    )

ax1.set_xticks(x)
ax1.set_xticklabels(
    [scenario_names[s] for s in scenario_order],
    fontsize=22
)

# Left Y-axis (AUC)
ax1.set_ylabel("AUC", fontsize=24)
ax1.tick_params(axis='y', labelsize=22)
ax1.set_ylim(0, 1)

# Right Y-axis (F1)
ax2.set_ylabel("F1 Score", fontsize=24)
ax2.tick_params(axis='y', labelsize=22)
ax2.set_ylim(0, 1)

# Grid
ax1.grid(axis='y', linestyle='-', alpha=0.4)
ax1.set_axisbelow(True)

# color legend -> models
model_legend = [
    Patch(
        facecolor=model_colors["pointnet_seg"],
        edgecolor='black',
        label='PointNet_seg'
    ),
    Patch(
        facecolor=model_colors["pointguard"],
        edgecolor='black',
        label='PointGuard'
    )
]

# hatch legend -> metrics
metric_legend = [
    Patch(
        facecolor='white',
        edgecolor='black',
        hatch='//',
        label='AUC'
    ),
    Patch(
        facecolor='white',
        edgecolor='black',
        hatch='..',
        label='F1'
    )
]

legend1 = ax1.legend(
    handles=model_legend,
    fontsize=20,
    title_fontsize=17,
    bbox_to_anchor=(0.53, 0.84)
)

legend2 = ax1.legend(
    handles=metric_legend,
    fontsize=20,
    title_fontsize=17,
    bbox_to_anchor=(0.8, 0.84)
)

# keep first legend
ax1.add_artist(legend1)

plt.tight_layout()
plt.show()