"""
train pointguard to provide 5th channel data
generate AD channel using NN
"""
import os
import sys
import torch
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader_pointguard import ModelNetDataLoader_pointguard
from defense_utils.generative_adversarial_network import perturbation_attack, weighted_dist_per, add_ADchannel
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=16, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--num_channel', type=int, default=4, help='Input Channel Number')  
    # keep some parameters just to pass to ModelLoader
    return parser.parse_args()

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def test(model, loader):
    mse_ls = []
    scorer = model.eval()

    for j, (points, _) in tqdm(enumerate(loader), total=len(loader)):

        # generate data (include points and the AD channel)
        clean_data = add_ADchannel(points, is_perturbed=False)  # (B, N, 5)
        clean_points, clean_target = torch.split(clean_data, [4,1], dim=2)  # (B, N, 4)  (B, N, 1) 
        # generate perturbed data 
        channels = [0,1,2,3]
        eps = 1
        perturbed_data = add_ADchannel(points, is_perturbed=True, channels=channels, eps=eps)   # (B,N,5)
        per_points, per_target = torch.split(perturbed_data, [4,1], dim=2)

        # concatenate
        com_points = torch.cat([clean_points, per_points], dim=1)   # (B, 2N, 4)
        com_target = torch.cat([clean_target, per_target], dim=1).squeeze(-1)   # (B, 2N)
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
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('pointguard')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
        
    param_name = f"epoch_{args.epoch}_npoint_{args.num_point}_bsize_{args.batch_size}"
    exp_dir = exp_dir.joinpath(param_name)

    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = '/content/drive/MyDrive/THESIS_dataset/mmw/MyModelNet_cls'

    train_dataset = ModelNetDataLoader_pointguard(root=data_path, args=args, split='train', process_data=False)
    test_dataset = ModelNetDataLoader_pointguard(root=data_path, args=args, split='test', process_data=False)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification.py', str(exp_dir))

    scorer = model.get_model(num_channels=args.num_channel)
    criterion = model.get_loss()
    scorer.apply(inplace_relu)

    if not args.use_cpu:
        scorer = scorer.cuda()
        criterion = criterion.cuda()

    start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            scorer.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(scorer.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    # best_loss = 0.0
    best_mse = 100.0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mse_ls = []
        scorer = scorer.train()

        scheduler.step()
        for batch_id, (points, _) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            
            # generate data (include points and the AD channel)
            clean_data = add_ADchannel(points, is_perturbed=False)  # (B, N, 5)
            clean_points, clean_target = torch.split(clean_data, [4,1], dim=2)  # (B, N, 4)  (B, N, 1) 

            channels = [0,1,2,3]
            eps = 1
            perturbed_data = add_ADchannel(points, is_perturbed=True, channels=channels, eps=eps)   # (B,N,5)
            per_points, per_target = torch.split(perturbed_data, [4,1], dim=2)
            
            # concatenate
            com_points = torch.cat([clean_points, per_points], dim=1)   # (B, 2N, 4)
            com_target = torch.cat([clean_target, per_target], dim=1).squeeze(-1)   # (B, 2N)
            if not args.use_cpu:
                com_points, com_target = com_points.cuda(), com_target.cuda()

            # predict and compute the loss
            com_points = com_points.transpose(2, 1)     # (B, 4, 2N)
            pred, trans_feat = scorer(com_points)       # pred: (B,2N)   trans_feat: (B,64,64)
            loss = criterion(pred, com_target.float(), trans_feat)

            # calculate mse loss
            mse_ls.append(F.mse_loss(pred, com_target))

            loss.backward()
            optimizer.step()
            global_step += 1

        train_mse_mean = np.mean([t.detach().cpu().numpy() for t in mse_ls])
        log_string('Train MSE loss: %f' % train_mse_mean)

        with torch.no_grad():
            test_mse_mean = test(scorer.eval(), testDataLoader)
            
            if (test_mse_mean <= best_mse):
                best_mse = test_mse_mean
                best_epoch = epoch + 1
            log_string('Test MSE loss: %f' % test_mse_mean)
            log_string('Best test MSE loss: %f' % best_mse)

            if (test_mse_mean <= best_mse):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'mse': test_mse_mean,
                    'model_state_dict': scorer.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
