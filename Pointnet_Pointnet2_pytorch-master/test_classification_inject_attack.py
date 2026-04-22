"""
Test vanilla PointNet (cls)
under adversarial injection attack
"""
from data_utils.ModelNetDataLoader_clean_per_inj import ModelNetDataLoader_clean_per_inj
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size in training')
    parser.add_argument('--num_category', default=2, type=int, choices=[2, 10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--num_channel', type=int, default=4, help='Input Channel Number')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    # add dropout, shift or not
    parser.add_argument('--dropout', action='store_true', default=False, help='use dropout when training')
    parser.add_argument('--shift', action='store_true', default=False, help='use shift when training')
    # add epoch and npoint for tracking
    parser.add_argument('--epoch', default=5, type=int, help='number of epoch in training')
    parser.add_argument('--num_point', type=int, default=16, help='Point Number')
    # add parameters for injection attack
    parser.add_argument('--npoints_inj', type=int, default=4, help='Number of Points Injected')
    parser.add_argument('--clutter_size_inj', type=int, default=2, help='The approximate number od points for the injected clutter')
    # add parameters for perturbation attack
    parser.add_argument('--channels_per', type=int, nargs='+', default=[0, 1, 2, 3], help='Channels of Perturbation')
    parser.add_argument('--eps_per', type=float, default=0, help='Eps of Perturbation')
    # add number of testing runs
    parser.add_argument('--num_runs', type=int, default=10, help='Number of Testing Runs')
    # FLAG of whether test using PointGuard
    parser.add_argument('--is_PointGuard', action='store_true', required=False, help='Whether use PointGuard')
    return parser.parse_args()

def test(model, loader, num_class=2, vote_num=1):
    classifier = model.eval()
    cd_ls = []
    target_ls = []
    pred_ls = []

    for j, (points_aug, target, cd) in tqdm(enumerate(loader), total=len(loader)):
        points = points_aug[:,:,:4].float()
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()   # points (B,N,4)
    
        points = points.transpose(2, 1)
        vote_pool = torch.zeros(target.size()[0], num_class).cuda()
        for _ in range(vote_num):
            pred, _ = classifier(points)
            vote_pool += pred
        pred = vote_pool / vote_num
        pred_choice = pred.data.max(1)[1]

        cd_ls.extend(cd.cpu().numpy())
        target_ls.extend(target.cpu().numpy())
        pred_ls.extend(pred_choice.cpu().numpy())

    return cd_ls, target_ls, pred_ls

def test_PointGuard(cls_model, scorer_model, loader, num_class=2, vote_num=1):
    classifier = cls_model.eval()
    scorer = scorer_model.eval()
    cd_ls = []
    target_ls = []
    pred_ls = []

    for j, (p, target, cd) in tqdm(enumerate(loader), total=len(loader)):
        points = p[:,:,:4].float()
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()   # points (B,N,4)
        
        # use scorer to predict scores
        scores, _ = scorer(points.transpose(2,1))        # scores (B,N)
        scores = scores.unsqueeze(-1)       # scores (B,N,1)

        points_aug = torch.cat([points, scores], dim=2) 

        # use classifier to redict labels
        if not args.use_cpu:
            points_aug = points_aug.cuda()
        points_aug = points_aug.transpose(2, 1)
        vote_pool = torch.zeros(target.size()[0], num_class).cuda()
        for _ in range(vote_num):
            pred, _ = classifier(points_aug)
            vote_pool += pred
        pred = vote_pool / vote_num
        pred_choice = pred.data.max(1)[1]

        cd_ls.extend(cd.cpu().numpy())
        target_ls.extend(target.cpu().numpy())
        pred_ls.extend(pred_choice.cpu().numpy())

    return cd_ls, target_ls, pred_ls

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir
    param_name = f"/epoch_{args.epoch}_npoint_{args.num_point}_bsize_{args.batch_size}"
    if args.dropout:
        param_name = param_name + "_dropout"
    if args.shift:
        param_name = param_name + "_shift"
    experiment_dir = experiment_dir + param_name + '/'

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval_injection_attack.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = '/content/drive/MyDrive/THESIS_dataset/mmw/MyModelNet_cls'

    test_dataset = ModelNetDataLoader_clean_per_inj(root=data_path, args=args, split='test', process_data=False, per_prob=0, inject_prob=1)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    path = os.listdir(experiment_dir + '/logs')
    txt_files = [f for f in path if f.endswith('.txt')]
    model_name = "pointnet_cls_pointguard" if args.is_PointGuard else "pointnet_cls"
    model = importlib.import_module(model_name)

    if args.is_PointGuard:
        classifier = model.get_model(num_class, num_channel=args.num_channel+1)      # (num_channel=5)
    else:
        classifier = model.get_model(num_class, num_channel=args.num_channel)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', weights_only=False)
    classifier.load_state_dict(checkpoint['model_state_dict'])

    '''PointGuard (scorer) MODEL LOADING'''
    if args.is_PointGuard:
        model_pointguard = importlib.import_module("pointnet_pointguard")
        scorer = model_pointguard.get_model(num_channels=args.num_channel)
        if not args.use_cpu:
            scorer = scorer.cuda()
        path_pointguard_weights = "/content/drive/MyDrive/THESIS/Pointnet_Pointnet2_pytorch-master/log/pointguard/pointguard_classification_mix/epoch_10_npoint_16_bsize_64/checkpoints/best_model.pth"
        checkpoint_pointguard = torch.load(path_pointguard_weights, weights_only=False)
        scorer.load_state_dict(checkpoint_pointguard['model_state_dict'])


    with torch.no_grad():
        bins = list(np.arange(0, 5.0, 0.1)) + [float('inf')]
        all_runs_acc = []
        num_runs = args.num_runs
        print(num_runs, " runs in total")
        for _ in range(num_runs ):
            if args.is_PointGuard:
                cd_ls, target_ls, pred_ls = test_PointGuard(classifier, scorer, testDataLoader, vote_num=args.num_votes, num_class=num_class)
            else:
                cd_ls, target_ls, pred_ls = test(classifier, testDataLoader, vote_num=args.num_votes, num_class=num_class)


            df = pd.DataFrame({
                'cd': cd_ls,
                'target': target_ls,
                'pred': pred_ls
            })
            
            df['cd_bin'] = pd.cut(df['cd'], bins=bins, right=False)

            results = df.groupby(['cd_bin', 'target'], observed=False).apply(
                lambda x: (x['target'] == x['pred']).mean() if len(x) > 0 else np.nan
            ).unstack()
            current_run_avg = results.mean(axis=1)
            all_runs_acc.append(current_run_avg)

        class_acc = pd.concat(all_runs_acc, axis=1).mean(axis=1).values
        cd_upper = results.index.map(lambda x: x.right)
        cd_upper = [x if x != float('inf') else 5.1 for x in cd_upper]

        out_df = pd.DataFrame({
            'cd_upper': cd_upper,
            'class_acc': class_acc
        })

        # save the data
        filename =  f"/inj_attack_comparison_npointsinj_{args.npoints_inj}_cluttersizeinj_{args.clutter_size_inj}"
        filename = filename + ".csv"
        out_df.to_csv(experiment_dir + filename, index=False)
        print(f"Data saved to {experiment_dir + filename}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
