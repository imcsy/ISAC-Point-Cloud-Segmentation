"""
Add SentryNet section
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
from data_utils.ModelNetDataLoader import ModelNetDataLoader

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
    parser.add_argument('--num_category', default=40, type=int, choices=[2, 10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    # parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_channel', type=int, default=3, help='Input Channel Number')   #############
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    # add dropout, shift or not
    # parser.add_argument('--dropout', action='store_true', default=False, help='use dropout when training')
    # parser.add_argument('--shift', action='store_true', default=False, help='use shift when training')
    return parser.parse_args()

def perturbation_attack(points, channels, eps):
    """
    Adds Gaussian jitter to specified channels of a point cloud tensor.
    
    Args:
        points: Input tensor of shape (batch_size, npoints, dim_input)
        channels: List of indices to perturb, e.g., [0, 1, 2] for XYZ
        eps: The standard deviation of the Gaussian noise
        
    Returns:
        perturbed_points: A new tensor with noise added
    """
    perturbed_points = points.clone()
    target_data = points[:, :, channels].reshape(-1, len(channels)) 

    sigma = torch.std(target_data, axis=0).to(points.device)

    noise_shape = (points.shape[0], points.shape[1], len(channels))
    jitter = torch.randn(noise_shape, device=points.device) * eps * sigma
    
    perturbed_points[:, :, channels] += jitter
    return perturbed_points, sigma

def weighted_dist_per(clean_points, per_points, weights):
    '''
    calculate the Weighted Euclidean Distance between clean_points and per_points
    input: clean_points / per_points # (batch_size, npoints, 4)
           weights # (4)
    output: weighted distance between clean_point and per_points # ((batch_size, npoints, 1)
    '''
    # ps_ref, ps_att = ps_ref[0], ps_att[0]       # add loop for batch
    ps_ref = clean_points.reshape(-1,4)
    ps_att = per_points.reshape(-1,4)
    dist_vec = torch.zeros(ps_ref.shape[0], dtype=torch.float32)
    for i, p_ref in enumerate(ps_ref):
        p_att = ps_att[i] 
        diff = (p_ref - p_att) ** 2
        dist_vec[i] = torch.dot(diff, weights)

    dist_vec = dist_vec.reshape(clean_points.shape[0], clean_points.shape[1], 1)
    dist_vec = torch.tensor(dist_vec)
    return dist_vec

def generative_network(clean_points, is_perturbed, channels=[0,1,2,3], eps=0):
    '''
    Add 5-th channel (abnormal detector) to the points data using generative model

    Input: clean points # (batch_size, npoints, 4)
    Return: clean/perturbed points + abnormal detetcor # (batch_size, npoints, 5)
    '''
    if is_perturbed:
        per_points, sigma = perturbation_attack(clean_points, channels, eps)
        weights = torch.zeros(4, dtype=torch.float32, device=clean_points.device)
        weights[channels] = 1 / (sigma**2)  # **2 or not
        weights[-1] = weights[-1] * 3
        dist = weighted_dist_per(clean_points, per_points, weights)
        lam = 1
        ad_channel = torch.exp(-lam*dist).to(clean_points.device)
        out = torch.cat([per_points, ad_channel], dim=2)
    else:
        ad_channel = torch.ones(clean_points.shape[0], clean_points.shape[1], 1, device=clean_points.device)
        out = out = torch.cat([clean_points, ad_channel], dim=2)
        
    return out

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


'''

            # clean
            clean_points = generative_network(points, is_perturbed=False)  # (B, N, 5)
            clean_points = clean_points.transpose(2, 1)                    # (B, 5, N)
            pred_clean, trans_feat_clean = classifier(clean_points)
            loss_clean = criterion(pred_clean, target.long(), trans_feat_clean)

            pred_choice_clean = pred_clean.data.max(1)[1]
            correct_clean = pred_choice_clean.eq(target.long().data).cpu().sum()
            mean_correct_clean.append(correct_clean.item() / float(points.size()[0]))

            # perturbed
            per_points = generative_network(points, is_perturbed=True,
                                    channels=[0, 1, 2, 3], eps=1)  # (B, N, 5)
            per_points = per_points.transpose(2, 1)                               # (B, 5, N)
            pred_per, trans_feat_per = classifier(per_points)
            loss_per = criterion(pred_per, target.long(), trans_feat_per)

            pred_choice_per = pred_per.data.max(1)[1]
            correct_per = pred_choice_per.eq(target.long().data).cpu().sum()
            mean_correct_per.append(correct_per.item() / float(points.size()[0]))

'''

def test(model, loader, num_class=40):
    mean_correct_clean = []
    class_acc_clean = np.zeros((num_class, 3))
    mean_correct_per = []
    class_acc_per = np.zeros((num_class, 3))
    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        # clean data
        clean_points = generative_network(points, is_perturbed=False)  # (B, N, 5)
        clean_points = clean_points.transpose(2, 1)                    # (B, 5, N)
        pred_clean, _ = classifier(clean_points)
        pred_choice_clean = pred_clean.data.max(1)[1]

        # perturbed
        per_points = generative_network(points, is_perturbed=True,
                                channels=[0, 1, 2, 3], eps=1)  # (B, N, 5)
        per_points = per_points.transpose(2, 1)                               # (B, 5, N)
        pred_per, _ = classifier(per_points)
        pred_choice_per = pred_per.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            # clean data
            classacc_clean = pred_choice_clean[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc_clean[cat, 0] += classacc_clean.item() / float(points[target == cat].size()[0]) # sum of accs across batches
            class_acc_clean[cat, 1] += 1        # number of batches it shows up
            # perturbed data
            classacc_per = pred_choice_per[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc_per[cat, 0] += classacc_per.item() / float(points[target == cat].size()[0]) # sum of accs across batches
            class_acc_per[cat, 1] += 1        # number of batches it shows up

        # clean data
        correct_clean = pred_choice_clean.eq(target.long().data).cpu().sum()
        mean_correct_clean.append(correct_clean.item() / float(points.size()[0]))
        # perturbed data
        correct_per = pred_choice_per.eq(target.long().data).cpu().sum()
        mean_correct_per.append(correct_per.item() / float(points.size()[0]))

    # clean data
    class_acc_clean[:, 2] = class_acc_clean[:, 0] / class_acc_clean[:, 1] # mean class acc over batches
    class_acc_clean = np.mean(class_acc_clean[:, 2])
    instance_acc_clean = np.mean(mean_correct_clean)
    # perturbed data
    class_acc_per[:, 2] = class_acc_per[:, 0] / class_acc_per[:, 1] # mean class acc over batches
    class_acc_per = np.mean(class_acc_per[:, 2])
    instance_acc_per = np.mean(mean_correct_per)

    return instance_acc_clean, class_acc_clean, instance_acc_per, class_acc_per


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
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
        
    param_name = f"epoch_{args.epoch}_npoint_{args.num_point}_bsize_{args.batch_size}"
    # if args.dropout:
    #     param_name = param_name + "_dropout"
    # if args.shift:
    #     param_name = param_name + "_shift"
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
    # '/content/drive/MyDrive/THESIS_dataset/modelnet40_normal_resampled/'
    # '/content/drive/MyDrive/THESIS_dataset/mmw/MyModelNet_cls'
    

    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification.py', str(exp_dir))

    classifier = model.get_model(num_class, num_channel=args.num_channel+1)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    # try:
    #     checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
    #     start_epoch = checkpoint['epoch']
    #     classifier.load_state_dict(checkpoint['model_state_dict'])
    #     log_string('Use pretrain model')
    # except:
    #     log_string('No existing model, starting training from scratch...')
    #     start_epoch = 0

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
    best_instance_acc_clean = 0.0
    best_class_acc_clean = 0.0
    best_instance_acc_per = 0.0
    best_class_acc_per = 0.0

    '''TRANING'''
    logger.info('Start training...')
    # test_ins_acc_log = np.zeros(args.epoch, dtype=float)
    # test_class_acc_log = np.zeros(args.epoch, dtype=float)
    # train_ins_acc_log_clean = np.zeros(args.epoch, dtype=float)
    # train_ins_acc_log_per = np.zeros(args.epoch, dtype=float)
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct_clean = []
        mean_correct_per = []
        classifier = classifier.train()

        scheduler.step()
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            
            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            # clean
            clean_points = generative_network(points, is_perturbed=False)  # (B, N, 5)
            clean_points = clean_points.transpose(2, 1)                    # (B, 5, N)
            pred_clean, trans_feat_clean = classifier(clean_points)
            loss_clean = criterion(pred_clean, target.long(), trans_feat_clean)

            pred_choice_clean = pred_clean.data.max(1)[1]
            correct_clean = pred_choice_clean.eq(target.long().data).cpu().sum()
            mean_correct_clean.append(correct_clean.item() / float(points.size()[0]))

            # perturbed
            per_points = generative_network(points, is_perturbed=True,
                                    channels=[0, 1, 2, 3], eps=1)  # (B, N, 5)
            per_points = per_points.transpose(2, 1)                               # (B, 5, N)
            pred_per, trans_feat_per = classifier(per_points)
            loss_per = criterion(pred_per, target.long(), trans_feat_per)

            pred_choice_per = pred_per.data.max(1)[1]
            correct_per = pred_choice_per.eq(target.long().data).cpu().sum()
            mean_correct_per.append(correct_per.item() / float(points.size()[0]))

            # combined loss
            loss = loss_clean + 0.5 * loss_per
            loss.backward()
            optimizer.step()
            global_step += 1
        
        # log
        train_instance_acc_clean = np.mean(mean_correct_clean)
        # train_ins_acc_log_clean[epoch] = train_instance_acc_clean
        log_string('Train Instance Accuracy for clean data: %f' % train_instance_acc_clean)

        train_instance_acc_per = np.mean(mean_correct_per)
        # train_ins_acc_log_per[epoch] = train_instance_acc_per
        log_string('Train Instance Accuracy for pertuebed data: %f' % train_instance_acc_per)

        with torch.no_grad():
            instance_acc_clean, class_acc_clean, instance_acc_per, class_acc_per = test(classifier.eval(), testDataLoader, num_class=num_class)
            # test_ins_acc_log[epoch] = instance_acc
            # test_class_acc_log[epoch] = class_acc

            if (instance_acc_per >= best_instance_acc_per):
                best_instance_acc_per = instance_acc_per
                best_epoch = epoch + 1

            if (class_acc_per >= best_class_acc_per):
                best_class_acc_per = class_acc_per
            log_string('Test Instance Accuracy for pertuebed data: %f, Class Accuracy for pertuebed data: %f' % (instance_acc_per, class_acc_per))
            log_string('Best Instance Accuracy for pertuebed data: %f, Class Accuracy for pertuebed data: %f' % (best_instance_acc_per, best_class_acc_per))

            if (instance_acc_per >= best_instance_acc_per):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc_clean,
                    'class_acc': class_acc_clean,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    # np.save(str(exp_dir) + '/logs/train_ins_acc_log.npy', train_ins_acc_log_clean)
    # np.save(str(exp_dir) + '/logs/test_ins_acc_log.npy', test_ins_acc_log)
    # np.save(str(exp_dir) + '/logs/test_class_acc_log.npy', test_class_acc_log)
    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
