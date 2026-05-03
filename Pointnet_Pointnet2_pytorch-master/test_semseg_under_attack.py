"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.S3DISDataLoader_mix import S3DISDataset_mix
from data_utils.indoor3d_util import g_label2color
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np

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
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=1024, help='point number [default: 1024]')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    # parser.add_argument('--test_area', type=int, default=5, help='area for testing, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting [default: 3]')
    parser.add_argument('--samples_per_frame', type=int, default=4, help='Number of samples obtained from each frame')
    # Attack Probs (scenario)
    parser.add_argument('--per_prob', type=float, default=0, help='Data proportion of Perturbation')
    parser.add_argument('--inj_prob', type=float, default=0, help='Data Proportion of Injection')
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
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

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
    NUM_POINT = args.num_point

    root = '/content/drive/MyDrive/THESIS_dataset/mmw/MyS3DIS_seg'

    # TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(root, split='test', block_points=NUM_POINT)
    TEST_DATASET = S3DISDataset_mix(split='test', data_root=root, num_point=NUM_POINT, sample_rate=1.0, transform=None, samples_per_frame=args.samples_per_frame, num_classes=NUM_CLASSES,
                                    per_prob=args.per_prob, inj_prob=args.inj_prob)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=10,
                                                 pin_memory=True, drop_last=True)
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', weights_only=False)
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
        classifier = classifier.eval()

        log_string('---- EVALUATION----')
        for i, (points_aug, target, cd) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            if not args.is_PointGuard:
                points = points_aug[:, :, :4]

            points_np = points.data.numpy()
            points = torch.Tensor(points)           # (batch_size, 1024, 4/5)
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

            # save pred result for visualization
            if args.visual:
                data_n_pred = np.concatenate((
                    points_np, 
                    batch_label.reshape(batch_label.shape[0], batch_label.shape[1], 1), 
                    pred_label.reshape(pred_label.shape[0], pred_label.shape[1], 1)
                ), axis=2)
                np.save(os.path.join(visual_dir, rf"eval_{i}.npy"), data_n_pred)

        # labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
        # mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6)) # mean IoU over all classes
        # # log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
        # log_string('eval point avg class IoU: %f' % (mIoU))
        # # log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))    
        # # log_string('eval point avg class acc: %f' % (
        # #     np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float64) + 1e-6))))

        # IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6)
        # iou_per_class_str = '------- IoU --------\n'
        # for l in range(NUM_CLASSES):
        #     iou_per_class_str += 'class %s, IoU: %.3f \n' % (
        #         seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])),
        #         total_correct_class[l] / float(total_iou_deno_class[l]))
        # log_string(iou_per_class_str)
        # log_string('eval point avg class IoU: %f' % np.mean(IoU))
        # log_string('eval whole scene point avg class acc: %f' % (
        #     np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float64) + 1e-6))))
        # log_string('eval whole scene point accuracy: %f' % (
        #         np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))

        # print("Done!")

        IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6)
        iou_per_class_str = '------- IoU --------\n'
        for l in range(NUM_CLASSES):
            iou_per_class_str += 'class %s, IoU: %.3f \n' % (
                seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])),
                total_correct_class[l] / float(total_iou_deno_class[l]))
        log_string(iou_per_class_str)
        log_string('eval point avg class IoU: %f' % np.mean(IoU))
        log_string('eval whole scene point avg class acc: %f' % (
            np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
        log_string('eval whole scene point accuracy: %f' % (
                np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))

        print("Done!")


if __name__ == '__main__':
    args = parse_args()
    main(args)
