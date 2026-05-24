#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
    else:
        pg_subset = pg_flat
        pn_subset = pn_flat
        scores_pg_subset = scores_pg
        scores_pn_subnet = scores_pn
        labels_subset = labels

    print("Computing t-SNE for standard PointNet features...")
    tsne_pn = TSNE(n_components=2, perplexity=30, max_iter=2000, random_state=42, n_jobs=-1)
    embedding_pn = tsne_pn.fit_transform(pn_subset)

    print("Computing t-SNE for your PointGuard features...")
    tsne_pg = TSNE(n_components=2, perplexity=30, max_iter=2000, random_state=42, n_jobs=-1)
    embedding_pg = tsne_pg.fit_transform(pg_subset)
    
    return embedding_pn, embedding_pg, scores_pg_subset, scores_pn_subnet, labels_subset


def plot_results(embedding_pn, embedding_pg, scores_pg_subset, scores_pn_subnet, labels_subset):
    """Generates and saves two separate figure windows with custom color maps."""
    print("Generating plots...")
    
    # ─── FIGURE 1: PointNet (Binary Colors) ───
    fig1, ax1 = plt.subplots(figsize=(9, 8))
    scatter1 = ax1.scatter(
        embedding_pn[:, 0], embedding_pn[:, 1],
        c=1-labels_subset, cmap='plasma', s=5, alpha=0.7
    )
    
    legend_elements = [
    Line2D([0], [0], marker='o', color='w',
           label='Clean',
           markerfacecolor='#F0F921',
           markersize=8),

    Line2D([0], [0], marker='o', color='w',
           label='Attacked',
           markerfacecolor='#0D0887',
           markersize=8)
    ]

    ax1.legend(handles=legend_elements, fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.3)
    fig1.tight_layout()

    # # ─── FIGURE 2: PointGuard (Continuous Colors) ───
    # fig2, ax2 = plt.subplots(figsize=(10, 8))
    # scatter2 = ax2.scatter(
    #     embedding_pg[:, 0], embedding_pg[:, 1],
    #     c=scores_subset, cmap='plasma', s=5, alpha=0.7
    # )
    # fig2.colorbar(scatter2, ax=ax2, label='Reliability Score')
    
    # ax2.grid(True, linestyle='--', alpha=0.3)
    # fig2.tight_layout()

    # plt.show()

    # ─── FIGURE 2: PointGuard (Continuous Colors with Inset Color Bar) ───
    fig2, ax2 = plt.subplots(figsize=(9, 8))
    scatter2 = ax2.scatter(
        embedding_pg[:, 0], embedding_pg[:, 1],
        c=scores_subset, cmap='plasma', s=5, alpha=0.7
    )

    # 1. Create an inset axis inside the grid area
    # width="3%" of the plot width, height="30%" of the plot height
    # loc='lower left' positions it inside. (You can also change it to 'upper right', etc.)
    # borderpad=3 pushes it slightly away from the absolute corner grid boundaries
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
embedding_pn, embedding_pg, scores_pg_subset, scores_pn_subnet, labels = compute_tsne(data_path, max_points=5000)
    
#%%
plot_results(embedding_pn, embedding_pg, scores_pg_subset, scores_pn_subnet, labels)
