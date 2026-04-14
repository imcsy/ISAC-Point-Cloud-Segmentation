"""
test pointguard to provide 5th channel data
generate AD channel using NN
"""
from data_utils.ModelNetDataLoader_pointguard import ModelNetDataLoader_pointguard
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
    parser.add_argument('--epoch', default=5, type=int, help='number of epoch in training')
    parser.add_argument('--num_point', type=int, default=16, help='Point Number')
    # # add dropout, shift or not
    # parser.add_argument('--dropout', action='store_true', default=False, help='use dropout when training')
    # parser.add_argument('--shift', action='store_true', default=False, help='use shift when training')
    # add epoch and npoint for tracking files, not for testing
    return parser.parse_args()


def test(model, loader):
    mse_ls = []
    scorer = model.eval()

    for j, (points, _) in tqdm(enumerate(loader), total=len(loader)):
        # generate data (include points and the AD channel)
        clean_data = add_ADchannel(points, is_perturbed=False)  # (B, N, 5)
        clean_points, clean_target = torch.split(clean_data, [4,1], dim=2)  # (B, N, 4)  (B, N, 1) 
        # generate perturbed data 
        # channels = [0,1,2,3]
        # eps = 1
        # perturbed_data = add_ADchannel(points, is_perturbed=True, channels=channels, eps=eps)   # (B,N,5)
        # per_points, per_target = torch.split(perturbed_data, [4,1], dim=2)

        channels = [[0,1,2,3], [0,1,2,3]]
        eps = [1, 2]
        per_points_ls = []
        per_target_ls = []
        for i in range(len(channels)):
            perturbed_data = add_ADchannel(points, is_perturbed=True, channels=channels[i], eps=eps[i])   # (B,N,5)
            p, t = torch.split(perturbed_data, [4,1], dim=2)
            per_points_ls.append(p)
            per_target_ls.append(t)
        
        # concatenate
        com_points = torch.cat([clean_points] + per_points_ls, dim=1)   # (B, 2N, 4)
        com_target = torch.cat([clean_target] + per_target_ls, dim=1).squeeze(-1)   # (B, 2N)

        # # concatenate
        # com_points = torch.cat([clean_points, per_points], dim=1)   # (B, 2N, 4)
        # com_target = torch.cat([clean_target, per_target], dim=1).squeeze(-1)   # (B, 2N)
        if not args.use_cpu:
            com_points, com_target = com_points.cuda(), com_target.cuda()

        # predict and compute the loss
        com_points = com_points.transpose(2, 1)     # (B, 4, 2N)
        pred, _ = scorer(com_points)       # pred: (B,2N)   trans_feat: (B,64,64)
        mse = F.mse_loss(pred, com_target.float()).item()
        mse_ls.append(mse)

    test_mse_mean = np.mean(mse_ls)
    return test_mse_mean


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/pointguard/' + args.log_dir
    param_name = f"/epoch_{args.epoch}_npoint_{args.num_point}_bsize_{args.batch_size}"
    # if args.dropout:
    #     param_name = param_name + "_dropout"
    # if args.shift:
    #     param_name = param_name + "_shift"
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

    test_dataset = ModelNetDataLoader_pointguard(root=data_path, args=args, split='test', process_data=False)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

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

    with torch.no_grad():
        test_mse_mean = test(scorer.eval(), testDataLoader)
        log_string('Test MSE loss: %f' % test_mse_mean)


if __name__ == '__main__':
    args = parse_args()
    main(args)
