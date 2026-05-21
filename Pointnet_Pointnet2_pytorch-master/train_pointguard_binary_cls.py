"""
train pointguard for binary cls on point levels directly (segmentation)
NO reliability scores
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
from data_utils.ModelNetDataLoader_clean_per_inj import ModelNetDataLoader_clean_per_inj
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
    parser.add_argument('--model', default='pointnet_sem_seg', help='model name')
    parser.add_argument('--epoch', default=10, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=16, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--num_channel', type=int, default=4, help='Input Channel Number')  
    # probability of two attacks
    parser.add_argument('--per_prob', type=float, default=0.4, help='Data proportion of Perturbation')
    parser.add_argument('--inject_prob', type=float, default=0.4, help='Data Proportion of Injection')
    # add parameters for injection attack
    parser.add_argument('--npoints_inj', type=int, default=4, help='Number of Points Injected')
    parser.add_argument('--clutter_size_inj', type=int, default=2, help='The approximate number od points for the injected clutter')
    # add parameters for perturbation attack
    parser.add_argument('--channels_per', type=int, nargs='+', default=[0, 1, 2, 3], help='Channels of Perturbation')
    parser.add_argument('--eps_per', type=float, default=1, help='Eps of Perturbation')
    # keep some parameters just to pass to ModelLoader
    parser.add_argument('--num_category', default=2, type=int, choices=[2, 10, 40],  help='training on ModelNet10/40')
    return parser.parse_args()

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def test(model, loader):
    mse_ls = []
    scorer = model.eval()

    for j, (points_aug, _, _) in tqdm(enumerate(loader), total=len(loader)):   
        points_aug = points_aug.float()

        if not args.use_cpu:
                points_aug = points_aug.cuda()                      # points_aug (B, N, 5)
        points, target = torch.split(points_aug, [4,1], dim=2)      # (B,N,4) and (B,N,1)
        target = target.squeeze(2)

        points = points.transpose(2,1)
        pred, _ = scorer(points)
        
        mse = F.mse_loss(pred, target.float()).item()
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
    exp_dir = exp_dir.joinpath('part_seg')
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

    train_dataset = ModelNetDataLoader_clean_per_inj(root=data_path, args=args, split='train', process_data=False, per_prob=args.per_prob, inject_prob=args.inject_prob)
    test_dataset = ModelNetDataLoader_clean_per_inj(root=data_path, args=args, split='test', process_data=False, per_prob=args.per_prob, inject_prob=args.inject_prob)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet_utils.py', str(exp_dir))
    shutil.copy('./train_pointguard_binary_cls.py', str(exp_dir))

    NUM_CLASSES = 2

    classifier = model.get_model(2).cuda()
    criterion = model.get_loss().cuda()
    classifier.apply(inplace_relu)
    
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

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
        classifier = classifier.train()

        scheduler.step()
        for batch_id, (points_aug, _, _) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points_aug = points_aug.float()     # (B, N, 5)
            optimizer.zero_grad()

            if not args.use_cpu:
                points_aug = points_aug.cuda()  # points_aug (B, N, 5)
            points, target = torch.split(points_aug, [4,1], dim=2)      # (B,N,4) and (B,N,1)
            target[target<1] =0
            print(points.shape, target.shape)
            points, target = torch.Tensor(points), torch.Tensor(target)

            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, trans_feat)
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            loss_sum += loss
            

        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))

        # with torch.no_grad():
        #     test_mse_mean = test(scorer, testDataLoader)
            
        #     if (test_mse_mean <= best_mse):
        #         best_mse = test_mse_mean
        #         best_epoch = epoch + 1
        #     log_string('Test MSE loss: %f' % test_mse_mean)
        #     log_string('Best test MSE loss: %f' % best_mse)

        #     if (test_mse_mean <= best_mse):
        #         logger.info('Save model...')
        #         savepath = str(checkpoints_dir) + '/best_model.pth'
        #         log_string('Saving at %s' % savepath)
        #         state = {
        #             'epoch': best_epoch,
        #             'mse': test_mse_mean,
        #             'model_state_dict': scorer.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #         }
        #         torch.save(state, savepath)
        #     global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
