"""
test PointGuard and baselines under different intensity
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

from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay
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
    parser.add_argument('--epoch', default=10, type=int, help='number of epoch in training')
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
    # choose test baseline model
    parser.add_argument("--baseline", type=str, required=True, help='baseline model')

    return parser.parse_args()

#   PointGuard
# ==================================================
def test_pointguard(scorer, loader):
    scorer = scorer.eval()
    all_preds = []
    all_targets = []    

    for j, (points_aug, _, _) in tqdm(enumerate(loader), total=len(loader)):
        points_aug = points_aug.float()

        if not args.use_cpu:
            points_aug = points_aug.cuda()
        points, target = torch.split(points_aug, [4,1], dim=2)    # target (B,N,1)
        target = target.squeeze(2)          # target (B,N)

        points = points.transpose(2,1)
        pred, _ = scorer(points)            # pred (B,N)

        # Move to CPU and flatten
        all_preds.append(pred.detach().cpu().reshape(-1))
        all_targets.append(target.detach().cpu().reshape(-1))

    all_preds = torch.cat(all_preds)            # (num_batchs * B * N,)
    all_targets = torch.cat(all_targets)

    return all_targets, all_preds


#   SOR (Statistical Outlier Removal)
# ==================================================
def SOR_detector(S, k, weights=[1,1,1,5]):
    '''
    input: S -- a point cloud batch (B,N,4)
    output: continuous scores in [0,1] -- (B,N)
    '''
    weights = torch.tensor(weights, dtype=S.dtype, device=S.device)

    diff = S[:, :, None, :] - S[:, None, :, :]   # (B, N, N, 4)
    dist2 = torch.sum((diff ** 2) * weights, dim=3)   # (B, N, N)
    B, N, _ = dist2.shape

    # Ignore self-distance
    idx = torch.arange(N, device=S.device)
    dist2[:, idx, idx] = float('inf')
    # k nearest neighbors
    knn_dist, _ = torch.topk(dist2, k=k, dim=2, largest=False)
    # Average kNN distance
    SOR = knn_dist.mean(dim=2)   # (B, N)

    # Normalize to [0,1]
    SOR_min = torch.min(SOR, dim=1).values.unsqueeze(1)  # (B,1)
    SOR_max = torch.max(SOR, dim=1).values.unsqueeze(1)  # (B,1)
    SOR_norm = (SOR - SOR_min) / (SOR_max - SOR_min + 1e-8)
    # Higher score = more normal / less adversarial
    SOR_norm = 1 - SOR_norm

    return SOR_norm


def test_SOR(loader, k=5):
    all_preds = []
    all_targets = []

    for j, (points_aug, _, _) in tqdm(enumerate(loader), total=len(loader)):
        points_aug = points_aug.float().cpu()
        points, target = torch.split(points_aug, [4,1], dim=2)    # target (B,N,1)
        
        pred = SOR_detector(points, k)
        
        # Move to CPU and flatten
        all_targets.append(target.reshape(-1))
        all_preds.append(pred.reshape(-1))

    all_preds = torch.cat(all_preds)            # (num_batchs * B * N,)
    all_targets = torch.cat(all_targets)

    return all_targets, all_preds

#   SPR (Salient Point Removal)  (use PointNet cls model)
# ==================================================
def test_SPR(loader):
    # load cls model
    cls_model = importlib.import_module("pointnet_cls")
    classifier = cls_model.get_model(2, num_channel=args.num_channel)
    checkpoint_path = '/content/drive/MyDrive/THESIS/Pointnet_Pointnet2_pytorch-master/log/classification/pointnet_cls_mymodelnet/epoch_10_npoint_16_bsize_64/checkpoints/best_model.pth'
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    if not args.use_cpu:
        classifier = classifier.cuda()
    classifier.eval()

    all_targets = []
    all_saliency = []

    for j, (points_aug, _, _) in tqdm(enumerate(loader), total=len(loader)):
        points_aug = points_aug.float()
        points, target = torch.split(points_aug, [4,1], dim=2)    # points (B, N, 4); target (B,N,1)
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()
        
        points.requires_grad_(True)
        points_input = points.permute(0, 2, 1)
        pred, _ = classifier(points_input)    # pred: (B, 2)  (either car or clutter)
        pred_choice = pred.max(dim=1)[1]      # (B,)

        selected_logits = pred[torch.arange(pred.shape[0]), pred_choice]    # (B,) (prediction confidence)
        classifier.zero_grad()
        selected_logits.sum().backward()
        grads = points.grad                    # grads (B, N, 4)
        saliency = torch.norm(grads, dim=2)     # (B, N)
        
        all_targets.append(target.reshape(-1))
        all_saliency.append(saliency.detach().cpu().reshape(-1))
            
    all_saliency = torch.cat(all_saliency)
    all_targets = torch.cat(all_targets)
    
    saliency_log = torch.log1p(all_saliency ** 0.5)
    saliency_norm = (
        saliency_log - saliency_log.min()
    ) / (
        saliency_log.max() - saliency_log.min() + 1e-8
    )
    saliency_norm = 1 - saliency_norm

    return all_targets, saliency_norm

#   Discriminator
# ==================================================
def test_discriminator(loader):
    # load cls model
    cls_model = importlib.import_module("pointnet_cls")
    classifier = cls_model.get_model(2, num_channel=args.num_channel)
    checkpoint_path = '/content/drive/MyDrive/THESIS/Pointnet_Pointnet2_pytorch-master/log/classification/pointnet_discriminator_mymodelnet/epoch_10_npoint_16_bsize_64/checkpoints/best_model.pth'
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    if not args.use_cpu:
        classifier = classifier.cuda()
    classifier.eval()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for j, (points_aug, _, _) in tqdm(enumerate(loader), total=len(loader)):
            points_aug = points_aug.float()
            points, target = torch.split(points_aug, [4,1], dim=2)    # points (B, N, 4); target (B,N,1)
            B, N, _ = points.shape
            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()
            
            points_input = points.permute(0, 2, 1)
            pred, _ = classifier(points_input)    # pred: (B, 2)  (either attack or clean)
            prob = F.softmax(pred, dim=1)         # prob: (B, 2)

            pred = prob[:, 1].unsqueeze(1).expand(B, N)      # pred: (B, 1)

            # Move to CPU and flatten
            all_targets.append(target.reshape(-1))
            all_preds.append(pred.reshape(-1))

    all_preds = torch.cat(all_preds)            # (num_batchs * B * N,)
    all_targets = torch.cat(all_targets)

    return all_targets, all_preds

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

    '''MODEL LOADING'''
    path = os.listdir(experiment_dir + '/logs')
    txt_files = [f for f in path if f.endswith('.txt')]
    model_name = txt_files[0].split('.')[0]
    model = importlib.import_module(model_name)

    scorer = model.get_model(num_channels=args.num_channel)
    if not args.use_cpu:
        scorer = scorer.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', weights_only=False)
    scorer.load_state_dict(checkpoint['model_state_dict'])

    per_intensities = [0.2, 0.5, 1.0, 2.0, 5.0]
    N_runs = 5
    for val in per_intensities:
        args.eps_per = val
        all_aucs = []
        
        for run in range(N_runs):
            test_dataset = ModelNetDataLoader_clean_per_inj(root=data_path, args=args, split='test', process_data=False,  per_prob=args.per_prob, inject_prob=args.inject_prob)
            testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
            
            with torch.no_grad():
                if args.baseline == "pointguard":
                    all_targets, all_preds = test_pointguard(scorer, testDataLoader)
                elif args.baseline == "sor":
                    all_targets, all_preds = test_SOR(testDataLoader, k=7)
                elif args.baseline == "spr":
                    all_targets, all_preds = test_SPR(testDataLoader)
                elif args.baseline == "discriminator":
                    all_targets, all_preds = test_discriminator(testDataLoader)

                all_targets[all_targets < 1] = 0
                fpr, tpr, thresholds = roc_curve(1-all_targets, 1-all_preds, pos_label=1)
                roc_auc = auc(fpr, tpr)
                all_aucs.append(roc_auc)

        all_aucs = np.array(all_aucs)
        results = {
            "percentage": 0.20,
            "baseline": args.baseline,
            "intensity": val,
            "mean_auc": float(all_aucs.mean()),
            "std_auc": float(all_aucs.std()),
            "min_auc": float(all_aucs.min()),
            "max_auc": float(all_aucs.max()),
            "all_aucs": all_aucs.tolist()
        }
        
        save_path = '/content/drive/MyDrive/THESIS/Pointnet_Pointnet2_pytorch-master/log/pointguard/pointguard_classification_mix/epoch_10_npoint_16_bsize_64/numerical result/per_sensitivity.json'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        new_df = pd.DataFrame([results])
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            try:
                existing_df = pd.read_json(save_path)
                final_df = pd.concat([existing_df, new_df], ignore_index=True)
            except Exception:
                final_df = new_df
        else:
            final_df = new_df
        final_df.to_json(save_path, orient='records', indent=4)

        print(f"Saved results cleanly using Pandas to {save_path}")

            # np.savez(
            #     experiment_dir + f'/numerical result/pointguard_targets_preds_per{args.per_prob}_inj{args.inject_prob}.npz',
            #     all_targets=all_targets.numpy(),
            #     all_preds=all_preds.numpy()
            # )
    # elif args.baseline == "sor":
    #     with torch.no_grad():
    #         all_targets, all_preds = test_SOR(testDataLoader, k=7)
    #         np.savez(
    #             experiment_dir + f'/numerical result/sor_targets_preds_per{args.per_prob}_inj{args.inject_prob}.npz',
    #             all_targets=all_targets.numpy(),
    #             all_preds=all_preds.numpy()
    #         )
    # elif args.baseline == "spr":
    #     all_targets, all_preds = test_SPR(testDataLoader)
    #     np.savez(
    #             experiment_dir + f'/numerical result/spr_targets_pred_per{args.per_prob}_inj{args.inject_prob}.npz',
    #             all_targets=all_targets.cpu().numpy(),
    #             all_preds=all_preds.cpu().numpy()
    #         )
    # elif args.baseline == "discriminator":
    #     all_targets, all_preds = test_discriminator(testDataLoader)
    #     np.savez(
    #             experiment_dir + f'/numerical result/discriminator_targets_pred_per{args.per_prob}_inj{args.inject_prob}.npz',
    #             all_targets=all_targets.cpu().numpy(),
    #             all_preds=all_preds.cpu().numpy()
            # )

if __name__ == '__main__':
    args = parse_args() 
    main(args)
