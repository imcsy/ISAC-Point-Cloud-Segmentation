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
baseline_ls = ["SOR", "SPR", "Instance-Dis", "PointGuard"]
baseline_colors = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"]
path_inj_ls = ["sor_targets_preds_inj.npz", "spr_targets_pred_per0.0_inj1.0.npz", "discriminator_targets_pred_per0.0_inj0.8.npz", "pointguard_targets_preds_inj.npz"]
path_per_ls = ["sor_targets_preds_per.npz", "spr_targets_pred_per1.0_inj0.0.npz", "discriminator_targets_pred_per1.0_inj0.0.npz", "pointguard_targets_preds_per.npz"]
path_mix_ls = ["sor_targets_preds_per0.4_inj0.4.npz", "spr_targets_pred_per0.4_inj0.4.npz", "discriminator_targets_pred_per0.4_inj0.4.npz", "pointguard_targets_preds_per0.4_inj0.4.npz"]
N_baselines = len(baseline_ls)

all_results = []

plt.figure(figsize=(12,8))
# Load saved results
for i in range(N_baselines):
    path = os.path.join(
        r"G:\我的云端硬盘\THESIS\Pointnet_Pointnet2_pytorch-master\log\pointguard\pointguard_classification_mix\epoch_10_npoint_16_bsize_64\numerical result",
        path_mix_ls[i]
    )
    data = np.load(path, allow_pickle=True)

    all_targets = data['all_targets']
    all_preds = data['all_preds']
    all_targets[all_targets < 1] = 0

    # =========================================================
    # ROC CURVE
    # =========================================================
    fpr, tpr, thresholds = roc_curve(1-all_targets, 1-all_preds, pos_label=1)
    roc_auc = auc(fpr, tpr)

    gaps = tpr - fpr
    best_idx = np.argmax(gaps)
    best_threshold = 1 - thresholds[best_idx]
    print(f"{baseline_ls[i]} best_threshold: {best_threshold:.4f}")

    # Save for confusion matrix later
    all_results.append({
        "targets": all_targets,
        "preds": all_preds,
        "threshold": best_threshold
    })

    # Plot ROC
    width = 13
    plt.plot(fpr, tpr, linewidth=3, label=f'{baseline_ls[i]}', color=baseline_colors[i])
    # plt.plot(fpr, tpr, linewidth=2, label=f'{baseline_ls[i]:<{width}} AUC = {roc_auc:.4f}', color=baseline_colors[i])

    # Mark best point
    plt.scatter(fpr[best_idx], tpr[best_idx], s=200, marker='^', color=baseline_colors[i])

# Random classifier line
plt.plot([0,1], [0,1], linestyle='--', linewidth=2, color="#BEBBAA")
plt.xlabel("False Positive Rate", fontsize=30)
plt.ylabel("True Positive Rate", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc="lower right", fontsize=26) #, prop={'family': 'monospace', 'size': 20})

plt.grid(True)
plt.tight_layout()
plt.show()

#%%
# =========================================================
# 4 CONFUSION MATRIX FIGURES
# =========================================================
for i in range(N_baselines):

    all_targets = all_results[i]["targets"]
    all_preds = all_results[i]["preds"]
    best_threshold = all_results[i]["threshold"]

    pred_binary = (all_preds > best_threshold).astype(int)

    tp = ((pred_binary == 0) & (all_targets == 0)).sum()
    fp = ((pred_binary == 0) & (all_targets == 1)).sum()
    tn = ((pred_binary == 1) & (all_targets == 1)).sum()
    fn = ((pred_binary == 1) & (all_targets == 0)).sum()

    # F1-score
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
    print(baseline_ls[i])
    print(f"F1-Score: {f1_score:.3f}")
    print(f"Recall (TPR): {recall:.3f}")
    print(f"Precision: {precision:.3f}")

    cm = np.array([
        [tp, fn],
        [fp, tn]
    ], dtype=np.float32)

    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    print(cm_norm)

    # # New figure for each CM
    # fig, ax = plt.subplots(figsize=(8,6))

    # disp = ConfusionMatrixDisplay(
    #     confusion_matrix=cm_norm,
    #     display_labels=["Attacked", "Clean"]
    # )

    # disp.plot(
    #     cmap='Blues',
    #     values_format='.3f',
    #     ax=ax,
    #     colorbar=True
    # )

    # ax.set_xlabel("Predicted Label", fontsize=16)
    # ax.set_ylabel("True Label", fontsize=16)

    # ax.tick_params(axis='both', labelsize=16)
    # ax.tick_params(axis='y', labelrotation=90)

    # ax.set_yticklabels(
    #     ax.get_yticklabels(),
    #     va='center'
    # )

    # for text in ax.texts:
    #     text.set_fontsize(18)

    # plt.tight_layout()
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