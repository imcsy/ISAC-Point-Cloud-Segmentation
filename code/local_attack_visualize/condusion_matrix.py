#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#%%
# True Positives (Predicted Car, is Car)
# False Positives (Predicted Car, is Clutter)
# False Negatives (Predicted Clutter, is Car)
# True Negatives (Predicted Clutter, is Clutter)

# vanilla
# tp, fn, fp, tn = 9110, 286, 2555, 8120
# NoiseTrain
# tp, fn, fp, tn =  8983, 422, 1301, 9374
# PointGuard
tp, fn, fp, tn = 8503, 842, 497, 10178

#%%
cm = np.array([[tp, fn], 
               [fp, tn]])
# row_sums = [TN+FP, FN+TP]
row_sums = cm.sum(axis=1, keepdims=True)
cm_normalized = cm / row_sums

labels = ['Car', 'Clutter']

#%%
plt.figure(figsize=(6, 5), dpi=150)
sns.set_style("whitegrid", {'axes.grid': False})

# Use fmt='.2f' for decimals or '.0%' for percentages
ax = sns.heatmap(cm_normalized, 
                 annot=True, 
                 fmt='.2f', 
                 cmap='Blues',
                 xticklabels=labels, 
                 yticklabels=labels,
                 # Adding 'color': 'black' or 'white' can force it, 
                 # but usually, just providing a larger figure size helps.
                 annot_kws={"size": 20, "weight": "bold"}, 
                 cbar=True, 
                 vmin=0, 
                 vmax=1)

plt.xlabel('Predicted Class', fontsize=20, labelpad=10)
plt.ylabel('True Class', fontsize=20, labelpad=10)
ax.tick_params(axis='both', labelsize=20)

# Consistency: Add the frame to match your bar chart
for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_linewidth(0.8)
    spine.set_color('#333333')

plt.tight_layout()
plt.show()