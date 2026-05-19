#%%
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay
)

#%%
baseline_ls = ["PointGuard", "SOR"]
path_inj_ls = ["pointguard_targets_preds_inj.npz", "sor_targets_preds_inj.npz"]
# path_per_ls = ["pointguard_targets_preds_per.npz", "sor_targets_preds_per.npz"]
N_baselines = len(baseline_ls)

plt.figure(figsize=(12,8))
# Load saved results
for i in range(N_baselines):
    path = os.path.join(
        r"G:\我的云端硬盘\THESIS\Pointnet_Pointnet2_pytorch-master\log\pointguard\pointguard_classification_mix\epoch_10_npoint_16_bsize_64\numerical result",
        path_inj_ls[i]
    )
    data = np.load(path, allow_pickle=True)

    all_targets = data['all_targets']
    all_preds = data['all_preds']
    all_targets[all_targets < 1] = 0

    print(all_preds)

    # =========================================================
    # ROC CURVE
    # =========================================================

    fpr, tpr, thresholds = roc_curve(1-all_targets, 1-all_preds, pos_label=1)
    roc_auc = auc(fpr, tpr)

    gaps = tpr - fpr
    best_idx = np.argmax(gaps)
    best_threshold = 1 - thresholds[best_idx]
    print(f"{baseline_ls[i]} best_threshold: {best_threshold:.4f}")

    # Plot ROC
    plt.plot(fpr, tpr, linewidth=2, label=f'{baseline_ls[i]}  AUC = {roc_auc:.4f}')

    # Mark best point
    plt.scatter(fpr[best_idx], tpr[best_idx], s=150, marker='^')

# Random classifier line
plt.plot([0,1], [0,1], linestyle='--', linewidth=2)

plt.xlabel("False Positive Rate", fontsize=20)
plt.ylabel("True Positive Rate", fontsize=20)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.legend(loc="lower right", fontsize=20)

plt.grid(True)

plt.tight_layout()

plt.show()

    # plt.plot(fpr, tpr, label= baseline_ls[i] + f' AUC = {roc_auc:.4f}')
    # plt.plot([0,1], [0,1], linestyle='--')

    # plt.xlabel("False Positive Rate", fontsize=18)
    # plt.ylabel("True Positive Rate", fontsize=18)
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    # plt.legend(loc="lower right", fontsize=18)

    # plt.axvline(
    #     x=fpr[best_idx],
    #     color='red',
    #     linestyle='--'
    # )

    # plt.grid(True)
    # plt.show()

#%%
# =========================================================
# CONFUSION MATRIX 
# =========================================================
pred_binary = (all_preds > best_threshold).astype(int)

tp = ((pred_binary == 0) & (all_targets == 0)).sum()
fp = ((pred_binary == 0) & (all_targets == 1)).sum()
tn = ((pred_binary == 1) & (all_targets == 1)).sum()
fn = ((pred_binary == 1) & (all_targets == 0)).sum()

tpr = tp / (tp + fn + 1e-8)
fpr = fp / (fp + tn + 1e-8)

print("TPR:", tpr)
print("FPR:", fpr)

cm = np.array([
    [tp, fn],
    [fp, tn]
], dtype=np.float32)

cm_norm = cm / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(8,6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_norm,
    display_labels=["Attacked", "Clean"]
)

disp.plot(
    cmap='Blues',
    values_format='.3f',
    ax=ax,
    colorbar=True
)

ax.set_xlabel("Predicted Label", fontsize=16)
ax.set_ylabel("True Label", fontsize=16)
ax.tick_params(axis='both', labelsize=16)
ax.tick_params(axis='y', labelrotation=90)
ax.set_yticklabels(
    ax.get_yticklabels(),
    va='center'
)
# Make numbers inside matrix larger
for text in ax.texts:
    text.set_fontsize(18)


plt.show()