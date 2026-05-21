"""
Ablation Study:
PointGuard design and 
PointNet directly for binary segmentation
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
    parser.add_argument('--epoch', default=10, type=int, help='number of epoch in training')
    parser.add_argument('--num_point', type=int, default=16, help='Point Number')
    # probability of two attacks
    parser.add_argument('--per_prob', type=float, default=0, help='Data proportion of Perturbation')
    parser.add_argument('--inject_prob', type=float, default=0.8, help='Data Proportion of Injection')
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
    parser.add_argument("--attack_scenario", type=str, required=True, help='Attack Scenario')

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

#   PointNet semseg
# ==================================================
def test_pointnet_seg(loader):
    MODEL = importlib.import_module("pointnet_sem_seg")
    classifier = MODEL.get_model(2).cuda()
    path = '/content/drive/MyDrive/THESIS/Pointnet_Pointnet2_pytorch-master/log/sem_seg/pointnet_ablation_pointguard/epoch_5_npoint_16_bsize_64/checkpoints/best_model.pth'
    checkpoint = torch.load(path, weights_only=False)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    all_preds = []
    all_targets = []    

    for j, (points_aug, _, _) in tqdm(enumerate(loader), total=len(loader)):
        points_aug = points_aug.float()

        points_aug = points_aug.float()
        points, target = torch.split(points_aug, [4,1], dim=2)      # (B,N,4) and (B,N,1)
        
        points = points.data.numpy()
        points = torch.Tensor(points)
        points, target = points.float().cuda(), target.long().cuda()
        points = points.transpose(2, 1)

        pred, trans_feat = classifier(points)
        pred = F.softmax(pred, dim=2)[:,:,1]

        # Move to CPU and flatten
        all_preds.append(pred.detach().cpu().reshape(-1))
        all_targets.append(target.detach().cpu().reshape(-1))
    
    all_preds = torch.cat(all_preds).float()           # (num_batchs * B * N,)
    all_targets = torch.cat(all_targets).float()

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

    N_runs = 5
    all_aucs = []
    all_F1 = []
    
    for run in range(N_runs):
        test_dataset = ModelNetDataLoader_clean_per_inj(root=data_path, args=args, split='test', process_data=False,  per_prob=args.per_prob, inject_prob=args.inject_prob)
        testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
        
        if args.baseline == "pointguard":
            with torch.no_grad():
                all_targets, all_preds = test_pointguard(scorer, testDataLoader)
        elif args.baseline == "pointnet_seg":
            with torch.no_grad():
                all_targets, all_preds = test_pointnet_seg(testDataLoader)

        all_targets[all_targets < 1] = 0
        all_targets, all_preds = all_targets.cpu(), all_preds.cpu()
        fpr, tpr, thresholds = roc_curve(1-all_targets, 1-all_preds, pos_label=1)
        roc_auc = auc(fpr, tpr)
        all_aucs.append(roc_auc)

        best_idx = np.argmax(tpr - fpr)
        pred_label = (all_preds >= thresholds[best_idx]).int()
        best_f1 = f1_score(all_targets.cpu().numpy(), pred_label.cpu().numpy())
        all_F1.append(best_f1)

    all_aucs = np.array(all_aucs)   
    all_F1 = np.array(all_F1)
    results = {
        "attack_scenario": args.attack_scenario,
        "baseline": args.baseline,
        "inject_prob": args.inject_prob,
        "per_prob": args.per_prob,
        "channels_per": args.channels_per,
        "eps_per": args.eps_per,
        "npoints_inj": args.npoints_inj,
        "clutter_size_inj": args.clutter_size_inj,
        "mean_auc": float(all_aucs.mean()),
        "std_auc": float(all_aucs.std()),
        "min_auc": float(all_aucs.min()),
        "max_auc": float(all_aucs.max()),
        "all_aucs": all_aucs.tolist(),
        "mean_f1": float(all_F1.mean()),
        "std_f1": float(all_F1.std()),
        "min_f1": float(all_F1.min()),
        "max_f1": float(all_F1.max()),
        "all_f1": all_F1.tolist()
    }
    
    save_path = '/content/drive/MyDrive/THESIS/Pointnet_Pointnet2_pytorch-master/log/pointguard/pointguard_classification_mix/epoch_10_npoint_16_bsize_64/numerical result/ablation.json'
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

if __name__ == '__main__':
    args = parse_args() 
    main(args)
