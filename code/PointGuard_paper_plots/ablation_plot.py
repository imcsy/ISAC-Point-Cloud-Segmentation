#%%
import json
import numpy as np
import matplotlib.pyplot as plt

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

plt.yticks(fontsize=18)

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

plt.yticks(fontsize=18)

plt.ylabel("F1", fontsize=24)

plt.legend(fontsize=20)

plt.gca().set_axisbelow(True)
plt.grid(axis='y', linestyle='-', alpha=0.5)

plt.tight_layout()
plt.show()