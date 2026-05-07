#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#%%
labels = ['car', 'building', 'pole', 'clutter']
# vanilla
vanilla_matrix = [[ 11850,    361,   1745,   3121],
                  [ 35764, 787880,  34359,  54849],
                  [  4000,   6961,  47331,   2253],
                  [116820,  49158,  13551, 310625]]
noistrain_matrix = [[ 38667,    874,    555,   3662],
                    [ 63453, 764398,  12651,  71436],
                    [ 11532,  16300,  30098,   2902],
                    [116452,  57466,   6471, 310411]]
pointgurad_matrix = [[ 36082,     25,    156,   7465],
                     [   346, 854257,   9446,  48754],
                     [   267,   2411,  54133,   3994],
                     [  2883,  45217,   3974, 437918]]

print(vanilla_matrix)

#%%
cm = np.array(vanilla_matrix)
# row_sums = [TN+FP, FN+TP]
row_sums = cm.sum(axis=1, keepdims=True)
cm_normalized = cm / row_sums

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
                 annot_kws={"size": 18, "weight": "bold"}, 
                 cbar=True, 
                 vmin=0, 
                 vmax=1)

plt.xlabel('Predicted Class', fontsize=16, labelpad=10)
plt.ylabel('True Class', fontsize=16, labelpad=10)
ax.tick_params(axis='both', labelsize=16)

# Consistency: Add the frame to match your bar chart
for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_linewidth(0.8)
    spine.set_color('#333333')

plt.tight_layout()
plt.show()