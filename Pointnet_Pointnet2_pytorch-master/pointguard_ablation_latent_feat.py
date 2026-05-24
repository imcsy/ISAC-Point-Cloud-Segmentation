"""
Ablation Study:
PointGuard design and 
PointNet directly for binary segmentation
show the latent feature + expanded channel using t-SNE
"""
from data_utils.ModelNetDataLoader_clean_per_inj import ModelNetDataLoader_clean_per_inj
from defense_utils.generative_adversarial_network import perturbation_attack, weighted_dist_per, add_ADchannel
import torch.nn.functional as F
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
import json
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, f1_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from sklearn.metrics import (
    roc_curve,
    auc,
    f1_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size in training')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--num_channel', type=int, default=4, help='Input Channel Number')
    # parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    parser.add_argument('--epoch', default=20, type=int, help='number of epoch in training')
    parser.add_argument('--num_point', type=int, default=16, help='Point Number')
    # probability of two attacks
    parser.add_argument('--per_prob', type=float, default=0.8, help='Data proportion of Perturbation')
    parser.add_argument('--inject_prob', type=float, default=0, help='Data Proportion of Injection')
    # add parameters for injection attack
    parser.add_argument('--npoints_inj', type=int, default=4, help='Number of Points Injected')
    parser.add_argument('--clutter_size_inj', type=int, default=2, help='The approximate number od points for the injected clutter')
    # add parameters for perturbation attack
    parser.add_argument('--channels_per', type=int, nargs='+', default=[0, 1, 2, 3], help='Channels of Perturbation')
    parser.add_argument('--eps_per', type=float, default=1, help='Eps of Perturbation')
    # keep some parameters just to pass to ModelLoader
    parser.add_argument('--num_category', default=2, type=int, choices=[2, 10, 40],  help='training on ModelNet10/40')
    # # choose test baseline model
    # parser.add_argument("--baseline", type=str, required=True, help='baseline model')
    # parser.add_argument("--attack_scenario", type=str, required=True, help='Attack Scenario')

    return parser.parse_args()

#   Main
# ==================================================
def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/pointguard/' + args.log_dir
    param_name = f"/epoch_{args.epoch}_npoint_{args.num_point}_bsize_{args.batch_size}"
    experiment_dir = experiment_dir + param_name + '/'

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = '/content/drive/MyDrive/THESIS_dataset/mmw/MyModelNet_cls'

    test_dataset = ModelNetDataLoader_clean_per_inj(root=data_path, args=args, split='test', process_data=False,  per_prob=args.per_prob, inject_prob=args.inject_prob)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''PointGuard MODEL LOADING'''
    path = os.listdir(experiment_dir + '/logs')
    txt_files = [f for f in path if f.endswith('.txt')]
    model_name = txt_files[0].split('.')[0]
    model = importlib.import_module(model_name)

    scorer = model.get_model(num_channels=args.num_channel)
    if not args.use_cpu:
        scorer = scorer.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', weights_only=False)
    scorer.load_state_dict(checkpoint['model_state_dict'])

    '''PointNet MODEL LOADING'''
    MODEL = importlib.import_module("pointnet_sem_seg")
    classifier = MODEL.get_model(2).cuda()
    path = '/content/drive/MyDrive/THESIS/Pointnet_Pointnet2_pytorch-master/log/sem_seg/pointnet_ablation_pointguard/epoch_5_npoint_16_bsize_64/checkpoints/best_model.pth'
    checkpoint = torch.load(path, weights_only=False)
    classifier.load_state_dict(checkpoint['model_state_dict'])

    
    scorer.eval()
    classifier = classifier.eval()
    all_feat_pointguard = []
    all_feat_pointnet = []
    all_scores = []     # continuous score between 0 and 1
    all_labels = []     # hard label 0 or 1
    all_scores_pred_pg = []
    all_scores_pred_pn = []

    with torch.no_grad():
        for j, (points_aug, _, _) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
            points_aug = points_aug.float()

            if not args.use_cpu:
                points_aug = points_aug.cuda()
            points, scores = torch.split(points_aug, [4,1], dim=2)    # target (B,N,1)
            scores = scores.squeeze(-1)
            labels = (scores == 1.0).float()

            points = points.transpose(2,1)

            # pointguard
            scores_pred, _, latent_pointguard = scorer(points)            # pred (B,N); latent_pointguard (B,64,N)
            all_feat_pointguard.append(latent_pointguard.cpu())    # (B, 64, N)
            all_scores.append(scores.cpu())      # (B, N)
            all_scores_pred_pg.append(scores_pred.squeeze(-1))
            
            # pointnet
            cls_preds, _, latent_pointnet = classifier(points)
            cls_probs = torch.exp(cls_preds)[:,:,1]
            all_feat_pointnet.append(latent_pointnet.cpu())        # (B, 64, N)
            all_labels.append(labels.cpu())      # (B, N)
            all_scores_pred_pn.append(cls_probs.cpu())

    all_feat_pointguard = torch.cat(all_feat_pointguard)       # (Total_B, 64, N)
    all_feat_pointnet = torch.cat(all_feat_pointnet)       # (Total_B, 128, N)
    all_scores = torch.cat(all_scores)           # (Total_B, N)
    all_labels = torch.cat(all_labels)           # (Total_B, N)
    all_scores_pred_pg = torch.cat(all_scores_pred_pg)
    all_scores_pred_pn = torch.cat(all_scores_pred_pn)

    # save
    data_to_save = {
        'pointguard_features': all_feat_pointguard.float(), # Optional: Convert to half-precision to save disk space
        'pointnet_features': all_feat_pointnet.float(),
        'scores': all_scores.float(),
        'labels': all_labels.float(),
        'pred_scores_pointguard': all_scores_pred_pg.float(),
        'pred_scores_pointnet': all_scores_pred_pn.float()
    }

    # Save locally to your server path
    save_path = '/content/drive/MyDrive/THESIS/Pointnet_Pointnet2_pytorch-master/log/pointguard/pointguard_classification_mix/epoch_20_npoint_16_bsize_64/numerical result/tsne_visualization_data.pt'
    torch.save(data_to_save, save_path)
    print(f" Successfully saved visualization data to {save_path}!")

#    #   t-SNE
#    # ==================================================
#    # 1. Permute and reshape features from (Total_B, 64, N) -> (Total_Points, 64)
#     latent_features_flat = all_features.permute(0, 2, 1).reshape(-1, 64).numpy()

#     # Flatten target arrays to match: (Total_Points,)
#     all_scores_flat = all_scores.reshape(-1).numpy()
#     all_labels_flat = all_labels.reshape(-1).numpy()

#     # 2. Subsample points to prevent t-SNE from freezing your machine!
#     MAX_POINTS_FOR_TSNE = 5000 
#     total_available_points = latent_features_flat.shape[0]

#     if total_available_points > MAX_POINTS_FOR_TSNE:
#         print(f"Subsampling {MAX_POINTS_FOR_TSNE} points out of {total_available_points} for fast t-SNE calculation...")
#         np.random.seed(42) # For reproducible plots
#         random_indices = np.random.choice(total_available_points, MAX_POINTS_FOR_TSNE, replace=False)
        
#         # Slice the data
#         features_subset = latent_features_flat[random_indices]
#         scores_subset = all_scores_flat[random_indices]
#         labels_subset = all_labels_flat[random_indices]
#     else:
#         features_subset = latent_features_flat
#         scores_subset = all_scores_flat
#         labels_subset = all_labels_flat

#     # 3. Compute 2D t-SNE (Usually takes 30-90 seconds for 5000 points)
#     print("Computing t-SNE embeddings (this might take a moment)...")
#     tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, n_jobs=-1)
#     tsne_results = tsne.fit_transform(features_subset) # Shape: (MAX_POINTS_FOR_TSNE, 2)
#     print("t-SNE calculation complete!")

#     # 4. Generate Side-by-Side Plots
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

#     # ─── LEFT PLOT: Your Soft Score Method ───
#     scatter1 = ax1.scatter(
#         tsne_results[:, 0], tsne_results[:, 1],
#         c=scores_subset, cmap='plasma', s=4, alpha=0.7
#     )
#     cbar1 = fig.colorbar(scatter1, ax=ax1, label='Continuous Reliability Score')
#     ax1.set_title('Our Method (Continuous Latent Space)', fontsize=13, fontweight='bold')
#     ax1.set_xlabel('t-SNE Dimension 1')
#     ax1.set_ylabel('t-SNE Dimension 2')
#     ax1.grid(True, linestyle='--', alpha=0.3)

#     # ─── RIGHT PLOT: The Hard Label Baseline ───
#     # Using 'bwr' (Blue-White-Red) color map to sharply distinguish 0 and 1
#     scatter2 = ax2.scatter(
#         tsne_results[:, 0], tsne_results[:, 1],
#         c=labels_subset, cmap='bwr', s=4, alpha=0.7
#     )
#     cbar2 = fig.colorbar(scatter2, ax=ax2, label='Binary Target Label')
#     cbar2.set_ticks([0, 1])
#     cbar2.set_ticklabels(['Attacked (0)', 'Clean (1)'])

#     ax2.set_title('Baseline Method (Binary Latent Space)', fontsize=13, fontweight='bold')
#     ax2.set_xlabel('t-SNE Dimension 1')
#     ax2.set_ylabel('t-SNE Dimension 2')
#     ax2.grid(True, linestyle='--', alpha=0.3)

#     # Overall Layout Details
#     plt.suptitle('t-SNE Distribution Analysis of PointGuard Latent Space', fontsize=16, fontweight='bold', y=0.98)
#     plt.tight_layout()

#     # Save the figure to your directory
#     plt.savefig('pointguard_tsne_comparison.png', dpi=300)
#     plt.show()


if __name__ == '__main__':
    args = parse_args() 
    main(args)
