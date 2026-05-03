#%%
import json
import numpy as np
import matplotlib.pyplot as plt

#%%
vanilla_color = ["#9D9090", 's']   # Vanilla
noitrain_color = ['#4472C4', '^']   # NoiseTrain
# advtrain_color = ["#44C451", '*']    # AdvTrain
pointnet_color = ['#ED7D31', 'o']    # PointNet 


#%%
json_path = r"G:\我的云端硬盘\THESIS\Pointnet_Pointnet2_pytorch-master\log\classification\acc_vs_scenario.json"
with open(json_path,"r") as f:
    data = json.load(f)

scenarios = ["no_attack", "mix_attack", "extreme_perturbation", "doppler_attack"]
scenario_names = ["No Attack", "Mixed Attack", "Extreme Attack ", "Unseen Attack"]
models = ["Vanilla","NoiseTrain","PointGuard"]

acc = {m:[] for m in models}
conf = {m:[] for m in models}
conf_std = {m:[] for m in models}

for s in scenarios:
    for m in models:
        record = next(
            item for item in data
            if item["model"]==m and item["attack_scenario"]==s
        )

        acc[m].append(record["class_acc"])
        conf[m].append(record["confidence_avg"])
        conf_std[m].append(record["confidence_std"]/3)

#%%
x = np.arange(len(scenarios))
width = 0.25

plt.figure(figsize=(12,8))

plt.bar(x-width, acc["Vanilla"], width, color=vanilla_color[0], label="Vanilla")
plt.bar(x, acc["NoiseTrain"], width, color=noitrain_color[0], label="NoiseTrain")
plt.bar(x+width, acc["PointGuard"], width, color=pointnet_color[0], label="PointGuard")

plt.xticks(x, scenario_names, fontsize=16)
plt.yticks(fontsize=14)
plt.ylabel("Class Accuracy", fontsize=18)
plt.ylim(0.6,1)
# plt.title("Attack-Agnostic Robustness Comparison")
plt.legend(fontsize=14)

plt.gca().set_axisbelow(True)
plt.grid(axis='y', linestyle='-', alpha=0.5)

# # value labels (optional)
# for i,m in enumerate(models):
#     offset = (-width,0,width)[i]
#     for j,v in enumerate(acc[m]):
#         plt.text(
#             x[j]+offset,
#             v+0.01,
#             f"{v:.3f}",
#             ha='center',
#             fontsize=9
#         )

plt.tight_layout()
plt.show()


#%%
x = np.arange(len(scenarios))
width = 0.25

plt.figure(figsize=(12,8))

plt.bar(x-width, conf["Vanilla"], width, color=vanilla_color[0], label="Vanilla")
plt.bar(x, conf["NoiseTrain"], width, color=noitrain_color[0], label="NoiseTrain")
plt.bar(x+width, conf["PointGuard"], width, color=pointnet_color[0], label="PointGuard")

plt.xticks(x, scenario_names, fontsize=16)
plt.yticks(fontsize=14)
plt.ylabel("Prediction Confidence", fontsize=18)
plt.ylim(0.6,1)
plt.legend(fontsize=14)

plt.gca().set_axisbelow(True)
plt.grid(axis='y', linestyle='-', alpha=0.5)

plt.tight_layout()
plt.show()


#%%
# Plotting
x = np.arange(len(scenarios))
width = 0.25
colors = ['#9b8e8e', '#4472c4', '#ed7d31'] # Matching your figure colors

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

for i, m in enumerate(models):
    ax.bar(x + (i - 1) * width, 
           conf[m], 
           width, 
           label=m, 
           color=colors[i],
           yerr=conf_std[m],    # <--- THIS ADDS THE STD
           capsize=5,            # Adds the horizontal 'cap' on the error bar
           error_kw={'elinewidth': 1.5, 'ecolor': '#333333'}) # Styling the line

# Aesthetics
ax.set_ylabel('Prediction Confidence', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(scenarios, fontsize=12)
ax.set_ylim(0.6, 1.0)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()