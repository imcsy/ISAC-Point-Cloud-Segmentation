"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.S3DISDataLoader_mix import S3DISDataset_mix
from data_utils.S3DISDataLoader import S3DISDataset
from data_utils.indoor3d_util import g_label2color
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['car', 'building', 'pole', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in testing [default: 32]')
    parser.add_argument('--epoch', default=10, type=int, help='number of epoch in training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--npoint', type=int, default=1024, help='point number [default: 1024]')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    # parser.add_argument('--test_area', type=int, default=5, help='area for testing, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting [default: 3]')
    parser.add_argument('--samples_per_frame', type=int, default=4, help='Number of samples obtained from each frame')
    # add dropout, shift or not
    parser.add_argument('--dropout', action='store_true', default=False, help='use dropout when training')
    parser.add_argument('--shift', action='store_true', default=False, help='use shift when training')
    # Attack Probs (scenario)
    parser.add_argument('--per_prob', type=float, default=0, help='Data proportion of Perturbation')
    parser.add_argument('--inj_prob', type=float, default=0, help='Data Proportion of Injection')
    # add parameters for perturbation attack 
    parser.add_argument('--per_channels', type=int, nargs='+', default=[0, 1, 2, 3], help='Channels of Perturbation')
    parser.add_argument('--per_eps', type=float, default=1.5, help='Eps of Perturbation (range:0-eps_per)')
    # add parameters for injection attack
    parser.add_argument('--inj_npoint_max', type=int, default=100, help='Number of Points Injected')
    parser.add_argument('--inj_clutter_size', type=int, default=10, help='The approximate number od points for the injected clutter')
    # FLAG of whether test using PointGuard / AdvTrain
    parser.add_argument('--is_PointGuard', action='store_true', required=False, help='Whether use PointGuard')
    return parser.parse_args() 


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sem_seg/' + args.log_dir

    param_name = f"/epoch_{args.epoch}_npoint_{args.npoint}_bsize_{args.batch_size}"
    if args.dropout:
        param_name = param_name + "_dropout"
    if args.shift:
        param_name = param_name + "_shift"
    experiment_dir = experiment_dir + param_name

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

    NUM_CLASSES = 4
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.npoint

    root = '/content/drive/MyDrive/THESIS_dataset/mmw/MyS3DIS_seg'

    # TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(root, split='test', block_points=NUM_POINT)
    TEST_DATASET = S3DISDataset_mix(split='test', data_root=root, num_point=NUM_POINT, sample_rate=1.0, transform=None, samples_per_frame=args.samples_per_frame, num_classes=NUM_CLASSES,
                                    per_prob=args.per_prob, inj_prob=args.inj_prob,
                                    per_channels=args.per_channels, per_eps=args.per_eps,
                                    inj_npoint_max=args.inj_npoint_max, inj_clutter_size=args.inj_clutter_size)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=10,
                                                 pin_memory=True, drop_last=True)
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', weights_only=False)
    log_string(f'Model Path: {experiment_dir}/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()


    '''Evaluate on UNchopped scenes'''
    with torch.no_grad():
        num_batches = len(testDataLoader)
        total_correct = 0
        total_seen = 0
        # loss_sum = 0
        labelweights = np.zeros(NUM_CLASSES)
        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
        cd_ls = []
        # IoU_ls = []
        # mIoU_ls = []
        # OA_ls = []
        union_ls = []
        intersection_ls = []
        classifier = classifier.eval()

        log_string('---- EVALUATION----')
        for i, (points_aug, target, cd, frame_idx) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
        # for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            # points_aug (B, N, 4/5)
            # target_np (B, N, 1)
            # cd (B)

            if not args.is_PointGuard:
                points = points_aug[:, :, :4]
            else:
                points = points_aug

            points_np = points.data.numpy()
            points = torch.Tensor(points_np)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            pred_val = seg_pred.contiguous().cpu().data.numpy()
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            pred_label = np.argmax(pred_val, 2)
            correct = np.sum((pred_label == batch_label))
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
            labelweights += tmp

            for l in range(NUM_CLASSES):
                total_seen_class[l] += np.sum((batch_label == l))           # calculate the no. of points each class (ground truth)
                total_correct_class[l] += np.sum((pred_label == l) & (batch_label == l))      # intersection
                total_iou_deno_class[l] += np.sum(((pred_label == l) | (batch_label == l)))   # union
            
            if i == 0:
                b = 10
                p = points_aug[b].transpose(1, 0).cpu().numpy()  # Result: (N, 5)   (x,y,z,v,a)
                t = batch_label[b].reshape(-1, 1)          # Ground Truth: (N, 1)
                p_l = pred_label[b].reshape(-1, 1) # Prediction: (N, 1)
                vis_combined = np.concatenate([p, t, p_l], axis=1)    # (x,y,z,v,a, ground-truth label, predicted labels)
                # idx = frame_idx[b]

                path = f"/content/drive/MyDrive/THESIS/code/local_visualize/data/segmentation_injection_point_cloud/inj_sample_{b}_mixloader_injnpoint_{args.inj_npoint_max}_injsize_{args.inj_clutter_size}.npy"
                np.save(path, vis_combined)
                log_string(f'Saved visualization sample {b} to {path}')

            # # points_np = points.data.numpy()
            # points = torch.Tensor(points)           # (B, 1024, 4/5)
            # # points, target = points.float().cuda(), target_np.long().cuda()
            # points = points.float().cuda()
            # points = points.transpose(2, 1)

            # pred_val, _ = classifier(points)
            # pred_val = pred_val.detach().cpu().numpy()
            # pred_label = np.argmax(pred_val, 2)   # [B, N]
            # gt = target.cpu().numpy().astype(np.int64)               # [B, N]

            # # one-hot encoding for entire batch
            # pred_onehot = np.eye(NUM_CLASSES)[pred_label]   # [B, N, C]
            # gt_onehot = np.eye(NUM_CLASSES)[gt]             # [B, N, C]

            # # intersection & union per sample per class
            # intersection = np.sum(pred_onehot * gt_onehot, axis=1)   # [B, C];  and operator
            # union = np.sum(pred_onehot + gt_onehot - pred_onehot * gt_onehot, axis=1)  # [B, C];  or operator

            # IoU = intersection / (union + 1e-6)   # [B, C]
            # mIoU_batch = IoU.mean(axis=1)         # [B]

            # # append results
            # cd_ls.extend(cd.cpu().numpy().tolist())
            # mIoU_ls.extend(mIoU_batch.tolist())
            # IoU_ls.extend(IoU.tolist())    # [N*B,C]
            # if i == 1:
            #     print(cd_ls.shape, mIoU_ls.shape, IoU_ls.shape)

            # Overall point Accuracy
            # OA_batch = np.mean(pred_label == gt, axis=1)   # [B]
            # OA_ls.extend(OA_batch.tolist())

            # save to visualize prediction result
            # if i == 0:
            #     # print(IoU)
            #     # print(mIoU_batch)

            #     b = 2
            #     p = points[b].transpose(1, 0).cpu().numpy()  # Result: (N, 4) or (N, 5)
            #     t = gt[b].reshape(-1, 1)          # Ground Truth: (N, 1)
            #     p_l = pred_label[b].reshape(-1, 1) # Prediction: (N, 1)
            #     vis_combined = np.concatenate([p, t, p_l], axis=1)    # (x,y,z, ground-truth label, predicted labels)
            #     idx = frame_idx[b]

            #     path = f"/content/drive/MyDrive/THESIS/code/local_visualize/data/segmentation_injection_point_cloud/clean_sample_{idx}.npy"
            #     np.save(path, vis_combined)
            #     log_string(f'Saved visualization sample {b} to {path}')

        # class IoU
        # class_iou_df = pd.DataFrame(
        #     IoU_ls, 
        #     columns=[f'IoU_{i}' for i in range(NUM_CLASSES)]
        # )
        # # mIoU and OA
        # df = pd.DataFrame({
        #     'cd': cd_ls,
        #     'mIoU': mIoU_ls,
        #     'OA': OA_ls
        # })
        # df = pd.concat([df, class_iou_df], axis=1)

        # bins = list(np.arange(0, 4.5, 0.5)) + [float('inf')]
        # df['cd_bin'] = pd.cut(df['cd'], bins=bins, right=False)
        # agg_dict = {
        #     'mIoU': 'mean',
        #     'OA': 'mean'
        # }
        # for i in range(NUM_CLASSES):
        #     agg_dict[f'IoU_{i}'] = 'mean'

        # results = df.groupby('cd_bin', observed=False).agg(agg_dict).reset_index()
        # results['cd_lower'] = results['cd_bin'].apply(lambda x: x.left)
        # class_cols = [f'IoU_{i}' for i in range(NUM_CLASSES)]
        # out_df = results[['cd_lower', 'mIoU', 'OA'] + class_cols]

        # print(out_df)
        # print("Done!")

        labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
        mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6)) # mean IoU over all classes

        IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6)
        iou_per_class_str = '------- IoU --------\n'
        for l in range(NUM_CLASSES):
            iou_per_class_str += 'class %s, IoU: %.3f \n' % (
                seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])),
                total_correct_class[l] / float(total_iou_deno_class[l]))
        log_string(iou_per_class_str)
        log_string('eval point avg class IoU: %f' % np.mean(IoU))
        log_string('eval whole scene point avg class acc: %f' % (
            np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float64) + 1e-6))))
        log_string('eval whole scene point accuracy: %f' % (
                np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))

        print("Done!")


if __name__ == '__main__':
    args = parse_args()
    main(args)
