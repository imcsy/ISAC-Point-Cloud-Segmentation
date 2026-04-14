"""
test classification using 5-channel models
generate AD channel using pointguard models
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from defense_utils.generative_adversarial_network import perturbation_attack, weighted_dist_per, pred_ADchannel
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
    parser.add_argument('--num_votes', type=int, default=1, help='Aggregate classification scores with voting')
    # add dropout, shift or not
    parser.add_argument('--dropout', action='store_true', default=False, help='use dropout when training')
    parser.add_argument('--shift', action='store_true', default=False, help='use shift when training')
    # add epoch and npoint for tracking
    parser.add_argument('--epoch', default=5, type=int, help='number of epoch in training')
    parser.add_argument('--num_point', type=int, default=16, help='Point Number')
    return parser.parse_args()


def test(classifier, loader, scorer, num_class=2, vote_num=1, perturb_channels=[0,1,2], perturb_eps=0):
    mean_correct = []
    classifier = classifier.eval()
    class_acc = np.zeros((num_class, 3))

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        # add perturbated attack
        if perturb_eps != 0:
            points = pred_ADchannel(scorer=scorer, clean_points=points, is_perturbed=True,
                                channels=perturb_channels, eps=perturb_eps)  # (B, N, 5)
        else:
            points = pred_ADchannel(scorer=scorer, clean_points=points, is_perturbed=False)

        points = points.transpose(2, 1)
        vote_pool = torch.zeros(target.size()[0], num_class).cuda()

        for _ in range(vote_num):
            pred, _ = classifier(points)
            vote_pool += pred
        pred = vote_pool / vote_num
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir
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
    file_handler = logging.FileHandler('%s/eval_pointguard.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = '/content/drive/MyDrive/THESIS_dataset/mmw/MyModelNet_cls'
    # '/content/drive/MyDrive/THESIS_dataset/mmw/MyModelNet_cls'
    # '/content/drive/MyDrive/THESIS_dataset/modelnet40_normal_resampled/'

    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    path = os.listdir(experiment_dir + '/logs')
    txt_files = [f for f in path if f.endswith('.txt')]
    model_name = txt_files[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, num_channel=args.num_channel+1)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', weights_only=False)
    classifier.load_state_dict(checkpoint['model_state_dict'])

    '''PointGuard (scorer) MODEL LOADING'''
    model_pointguard = importlib.import_module("pointnet_pointguard")
    scorer = model_pointguard.get_model(num_channels=args.num_channel)
    if not args.use_cpu:
        scorer = scorer.cuda()
    path_pointguard_weights = "/content/drive/MyDrive/THESIS/Pointnet_Pointnet2_pytorch-master/log/pointguard/pointguard_classification_shift_attack/epoch_10_npoint_16_bsize_64/checkpoints/best_model.pth"
    checkpoint_pointguard = torch.load(path_pointguard_weights, weights_only=False)
    scorer.load_state_dict(checkpoint_pointguard['model_state_dict'])

    with torch.no_grad():
        # # run a single test
        # perturb_channels = [0, 1, 2, 3]
        # eps = 1
        # instance_acc, class_acc = test(classifier=classifier, loader=testDataLoader, scorer=scorer, vote_num=args.num_votes, num_class=num_class, perturb_channels=perturb_channels, perturb_eps=eps)
        # log_string(f'Perturbation channels: {perturb_channels}, Perturbation epsilon: %f' % (eps))
        # log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))

        # attack comparison by running multiple tests
        # pertubation attack
        eps_values = np.round(np.arange(0, 4.1, 0.1), 2).tolist()
        acc_perturb_pos = np.zeros(len(eps_values), dtype=float)
        acc_perturb_vel = np.zeros(len(eps_values), dtype=float)
        acc_perturb_pos_vel = np.zeros(len(eps_values), dtype=float)
        for i, eps in enumerate(eps_values):
            # perturb positions only (x,y,z)
            perturb_channels = [0,1,2]
            instance_acc, class_acc = test(classifier=classifier, loader=testDataLoader, scorer=scorer, vote_num=args.num_votes, num_class=num_class, perturb_channels=perturb_channels, perturb_eps=eps)
            log_string(f'Perturbation channels: {perturb_channels}, Perturbation epsilon: %f' % (eps))
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            acc_perturb_pos[i] = class_acc

            # perturb velocity only (v)
            perturb_channels = [3]
            instance_acc, class_acc = test(classifier=classifier, loader=testDataLoader, scorer=scorer, vote_num=args.num_votes, num_class=num_class, perturb_channels=perturb_channels, perturb_eps=eps)
            log_string(f'Perturbation channels: {perturb_channels}, Perturbation epsilon: %f' % (eps))
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            acc_perturb_vel[i] = class_acc

            # perturb positions and velocity (x,y,z,v)
            perturb_channels = [0,1,2,3]
            instance_acc, class_acc = test(classifier=classifier, loader=testDataLoader, scorer=scorer, vote_num=args.num_votes, num_class=num_class, perturb_channels=perturb_channels, perturb_eps=eps)
            log_string(f'Perturbation channels: {perturb_channels}, Perturbation epsilon: %f' % (eps))
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            acc_perturb_pos_vel[i] = class_acc

        df = pd.DataFrame({
                'epsilon': eps_values,
                'accuracy_perturb_pos': acc_perturb_pos,
                'accuracy_perturb_vel': acc_perturb_vel,
                'accuracy_perturb_pos_vel': acc_perturb_pos_vel
            })
        df.to_csv(experiment_dir + "/attack_comparison.csv", index=False)

if __name__ == '__main__':
    args = parse_args()
    main(args)
