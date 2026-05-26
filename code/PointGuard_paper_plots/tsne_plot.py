#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.metrics import roc_auc_score, f1_score, roc_curve

#%%
def compute_tsne(data_path, max_points=5000):
    """Loads, flattens, subsamples, and computes 2D t-SNE coordinates 
    for both PointNet and PointGuard feature spaces."""
    
    print("Loading saved tensors...")
    data = torch.load(data_path, map_location=torch.device('cpu'))
    
    pg_features = data['pointguard_features'].float() 
    pn_features = data['pointnet_features'].float()   
    scores_pg = data['pred_scores_pointguard'].numpy().flatten()      
    scores_pn = data['pred_scores_pointnet'].numpy().flatten() 
    gt_scores = data['scores'].flatten()
    labels = data['labels'].numpy().flatten()          

    print("Reshaping and flattening features...")
    c_pg = pg_features.shape[1]    # 64    (Total_B, 64, N)
    c_pn = pn_features.shape[1]    # 128   (Total_B, 128, N)
    
    pg_flat = pg_features.permute(0, 2, 1).reshape(-1, c_pg).numpy()      # (Total_B*N, 64)
    pn_flat = pn_features.permute(0, 2, 1).reshape(-1, c_pn).numpy()      # (Total_B*N, 128)

    total_points = pg_flat.shape[0]
    if total_points > max_points:
        print(f"Subsampling {max_points} random points out of {total_points}...")
        np.random.seed(42) 
        indices = np.random.choice(total_points, max_points, replace=False)
        
        pg_subset = pg_flat[indices]
        pn_subset = pn_flat[indices]
        scores_pg_subset = scores_pg[indices]
        scores_pn_subnet = scores_pn[indices]
        labels_subset = labels[indices]
        gt_scores_subset = gt_scores[indices]
    else:
        pg_subset = pg_flat
        pn_subset = pn_flat
        scores_pg_subset = scores_pg
        scores_pn_subnet = scores_pn
        labels_subset = labels
        gt_scores_subset = gt_scores

    print("Computing t-SNE for standard PointNet features...")
    tsne_pn = TSNE(n_components=2, perplexity=30, max_iter=2000, random_state=42, n_jobs=-1)
    embedding_pn = tsne_pn.fit_transform(pn_subset)

    print("Computing t-SNE for your PointGuard features...")
    tsne_pg = TSNE(n_components=2, perplexity=30, max_iter=2000, random_state=42, n_jobs=-1)
    embedding_pg = tsne_pg.fit_transform(pg_subset)
    
    return embedding_pn, embedding_pg, scores_pg_subset, scores_pn_subnet, labels_subset, gt_scores_subset


def plot_results(embedding_pn, embedding_pg, scores_pg_subset, scores_pn_subnet, labels_subset, gt_scores):
    """Generates and saves two separate figure windows with custom color maps."""
    print("Generating plots...")
    
    # ─── FIGURE 1: PointNet ───
    fig1, ax1 = plt.subplots(figsize=(9, 8))
    scatter1 = ax1.scatter(
        embedding_pn[:, 0], embedding_pn[:, 1],
        c=scores_pn_subnet, cmap='plasma', s=5, alpha=0.7
    )

    cb_ax = inset_axes(
        ax1, 
        width="30%", 
        height="2%", 
        loc='lower left', 
        bbox_to_anchor=(0.67, 0.95, 1, 1), 
        bbox_transform=ax1.transAxes,
        borderpad=0
    )

    cbar1 = fig1.colorbar(scatter1, cax=cb_ax, orientation='horizontal')

    cbar1.set_label('Softmax Probablity', color='black', fontsize=14, labelpad=10)
    cb_ax.yaxis.set_tick_params(labelsize=9)

    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_title("PointNet_seg")
    fig1.tight_layout()

    plt.show()

    # ─── FIGURE 2: PointGuard ───
    fig2, ax2 = plt.subplots(figsize=(9, 8))
    scatter2 = ax2.scatter(
        embedding_pg[:, 0], embedding_pg[:, 1],
        c=scores_pg_subset, cmap='plasma', s=5, alpha=0.7
    )

    cb_ax = inset_axes(
        ax2, 
        width="30%", 
        height="2%", 
        loc='lower left', 
        bbox_to_anchor=(0.67, 0.95, 1, 1), 
        bbox_transform=ax2.transAxes,
        borderpad=0
    )

    # 2. Draw the color bar onto the inset axes
    cbar2 = fig2.colorbar(scatter2, cax=cb_ax, orientation='horizontal')

    # 3. Format the text labels on the color bar so they look great inside the grid
    cbar2.set_label('Reliability Score', color='black', fontsize=14, labelpad=10)
    cb_ax.yaxis.set_tick_params(labelsize=9)

    # 4. Final plot configurations
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_title("PointGuard")
    fig2.tight_layout()

    plt.show()


#%%
data_path = r"G:\我的云端硬盘\THESIS\Pointnet_Pointnet2_pytorch-master\log\pointguard\pointguard_classification_mix\epoch_20_npoint_16_bsize_64\numerical result\tsne_visualization_data.pt"
# data = torch.load(data_path, map_location=torch.device('cpu'))
    
# pg_features = data['pointguard_features'].float() 
# pn_features = data['pointnet_features'].float()   
# scores = data['scores'].numpy().flatten()          
# labels = data['labels'].numpy().flatten()      

# print(scores.mean(), labels.mean())

# print(labels)

# print(scores)

#%%
embedding_pn, embedding_pg, scores_pg_subset, scores_pn_subnet, labels, gt_scores_subset = compute_tsne(data_path, max_points=5000)
    
#%%
plot_results(embedding_pn, embedding_pg, scores_pg_subset, scores_pn_subnet, labels, gt_scores_subset)


#%%
data_path = r"G:\我的云端硬盘\THESIS\Pointnet_Pointnet2_pytorch-master\log\pointguard\pointguard_classification_mix_small\epoch_20_npoint_16_bsize_64\numerical result\ablation_data.pt"

data = torch.load(data_path, map_location=torch.device('cpu'))

scores_pg = data['pred_scores_pointguard'].numpy().flatten()      
scores_pn = data['pred_scores_pointnet'].numpy().flatten() 
scores_pg_dis = data['pred_scores_pg_discriminator'].numpy().flatten()
gt_scores = data['scores'].numpy().flatten()
labels = data['labels'].numpy().flatten()    

def check_scores():
    # scores_pg_dis = np.exp(scores_pg_dis).reshape(-1,2)[:,1]
    print("MSE of PointGuard:", np.mean((scores_pg-gt_scores)**2))
    print("MSE of PointNet:", np.mean((scores_pn-gt_scores)**2))
    print("MSE of PointGuard + Discriminator:", np.mean((scores_pg_dis-gt_scores)**2))

    delta_pg = scores_pg - gt_scores
    delta_pn = scores_pn - gt_scores
    delta_pg_dis = scores_pg_dis - gt_scores

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot overlapping histograms with transparency (alpha)
    ax.hist(delta_pg, bins=30, alpha=0.6, label=r'$\delta_{pg}$ (scores_pg - gt_scores)', color='steelblue', edgecolor='black')
    ax.hist(delta_pn, bins=30, alpha=0.6, label=r'$\delta_{pn}$ (scores_pn - gt_scores)', color='darkorange', edgecolor='black')
    ax.hist(delta_pg_dis, bins=30, alpha=0.6, label=r'$\delta_{pndis}$ (scores_pn_dis - gt_scores)', color='pink', edgecolor='black')

    # 4. Add formatting and labels
    ax.set_xlabel(r'Difference ($\delta$)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(r'Distribution of $\delta_{pg}$ and $\delta_{pn}$', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig('delta_distribution.png')

def evaluate_scores(labels, scores, name="Model"):
    # AUC
    auc = roc_auc_score(labels, scores)

    # ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Best threshold (largest vertical gap)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]

    # Binary prediction using best threshold
    preds = (scores >= best_thresh).astype(int)

    # F1
    f1 = f1_score(labels, preds)

    print(f"{name}")
    print(f"  AUC            : {auc:.4f}")
    print(f"  Best Threshold : {best_thresh:.4f}")
    print(f"  F1 Score       : {f1:.4f}")
    print()

    return auc, best_thresh, f1


evaluate_scores(labels, scores_pg, "PointGuard")
evaluate_scores(labels, scores_pn, "PointNet")
evaluate_scores(labels, scores_pg_dis, "PG Discriminator")