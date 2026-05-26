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

    '''PointGuard Discriminator LOADING'''
    MODEL = importlib.import_module("pointguard_discriminator")
    pg_discriminator = MODEL.get_model().cuda()
    path = '/content/drive/MyDrive/THESIS/Pointnet_Pointnet2_pytorch-master/log/sem_seg/pointguard_downstream_discriminator/epoch_5_npoint_16_bsize_64/checkpoints/best_model.pth'
    checkpoint = torch.load(path, weights_only=False)
    pg_discriminator.load_state_dict(checkpoint['model_state_dict'])

    
    scorer.eval()
    classifier = classifier.eval()
    pg_discriminator = pg_discriminator.eval()
    all_feat_pointguard = []
    all_feat_pointnet = []
    all_scores = []     # continuous score between 0 and 1
    all_labels = []     # hard label 0 or 1
    all_scores_pred_pg = []
    all_scores_pred_pn = []
    all_scores_pred_pg_dis = []

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
            scores_pred_pg, _, latent_pointguard = scorer(points)            # pred (B,N); latent_pointguard (B,64,N)
            all_feat_pointguard.append(latent_pointguard.cpu())    # (B, 64, N)
            all_scores.append(scores.cpu())      # (B, N)
            all_scores_pred_pg.append(scores_pred_pg.squeeze(-1))

            # # fake pointguard + discriminator
            # points_aug = points_aug.transpose(2,1)
            # scores_pred_pgdis = pg_discriminator(points_aug)
            # all_scores_pred_pg_dis.append(scores_pred_pgdis.squeeze(-1))

            # true pointguard + discriminator
            points_aug_pg = torch.cat([points.transpose(2,1), scores_pred_pg.unsqueeze(-1)], dim=2)     # (B, N, 5)
            points_aug_pg = points_aug_pg.transpose(2,1)
            preds_pgdis = pg_discriminator(points_aug_pg)
            scores_pred_pgdis = torch.exp(preds_pgdis)[:, :, 1]
            all_scores_pred_pg_dis.append(scores_pred_pgdis.squeeze(-1))
            
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
    all_scores_pred_pg_dis = torch.cat(all_scores_pred_pg_dis)

    # save
    data_to_save = {
        'pointguard_features': all_feat_pointguard.float(), # Optional: Convert to half-precision to save disk space
        'pointnet_features': all_feat_pointnet.float(),
        'scores': all_scores.float(),
        'labels': all_labels.float(),
        'pred_scores_pointguard': all_scores_pred_pg.float(),
        'pred_scores_pointnet': all_scores_pred_pn.float(),
        'pred_scores_pg_discriminator': all_scores_pred_pg_dis.float()
    }

    # Save locally to your server path
    save_path = '/content/drive/MyDrive/THESIS/Pointnet_Pointnet2_pytorch-master/log/pointguard/pointguard_classification_mix_small/epoch_20_npoint_16_bsize_64/numerical result/ablation_data.pt'
    torch.save(data_to_save, save_path)
    print(f" Successfully saved visualization data to {save_path}!")


if __name__ == '__main__':
    args = parse_args() 
    main(args)
