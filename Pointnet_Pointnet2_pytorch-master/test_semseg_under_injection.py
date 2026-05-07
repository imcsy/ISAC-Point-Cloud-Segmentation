"""
only for injection attack
mIoU vs n_clutter
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
    parser.add_argument('--inj_prob', type=float, default=1, help='Data Proportion of Injection')
    # add parameters for perturbation attack 
    parser.add_argument('--per_channels', type=int, nargs='+', default=[0, 1, 2, 3], help='Channels of Perturbation')
    parser.add_argument('--per_eps', type=float, default=1.5, help='Eps of Perturbation (range:0-eps_per)')
    # add parameters for injection attack
    # parser.add_argument('--inj_npoint_max', type=int, default=100, help='Number of Points Injected')
    parser.add_argument('--inj_clutter_size', type=int, default=20, help='The approximate number od points for the injected clutter')
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

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', weights_only=False)
    log_string(f'Model Path: {experiment_dir}/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    cd_csv = []
    mIoU_csv = []

    inj_npoint_ls = np.array([0, 20, 60, 80, 100, 140, 160, 200, 220])
    for inj_npoint in inj_npoint_ls:
        # TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(root, split='test', block_points=NUM_POINT)
        TEST_DATASET = S3DISDataset_mix(split='test', data_root=root, num_point=NUM_POINT, sample_rate=1.0, transform=None, samples_per_frame=args.samples_per_frame, num_classes=NUM_CLASSES,
                                        per_prob=args.per_prob, inj_prob=args.inj_prob,
                                        per_channels=args.per_channels, per_eps=args.per_eps,
                                        inj_npoint=inj_npoint, inj_clutter_size=args.inj_clutter_size)
        testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=10,
                                                    pin_memory=True, drop_last=True)
        log_string("The number of test data is: %d" % len(TEST_DATASET))


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
            # union_ls = []
            # intersection_ls = []
            classifier = classifier.eval()

            log_string('---- EVALUATION----')
            for i, (points_aug, target, cd, frame_idx) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            # for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                # points_aug (B, N, 5)
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

                cd_ls.extend(cd.cpu().numpy().tolist())

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))           # calculate the no. of points each class (ground truth)
                    total_correct_class[l] += np.sum((pred_label == l) & (batch_label == l))      # intersection
                    total_iou_deno_class[l] += np.sum(((pred_label == l) | (batch_label == l)))   # union

            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            cd = np.mean(np.array(cd_ls))
            IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6)
            mIoU = np.mean(IoU)

            cd_csv.append(cd)
            mIoU_csv.append(mIoU)

    df = pd.DataFrame({
        "inj_npoint": inj_npoint_ls,
        "cd": cd_csv,
        "mIoU": mIoU_csv
    })
    

    path = experiment_dir + "/inj_mIoU_vs_nclutter.csv"
    print(df)
    print(os.path.dirname(path))
    print("Path exist: ", os.path.exists(os.path.dirname(path)))

    df.to_csv(path, index=False)

    print("CSV saved successfully.")

if __name__ == '__main__':
    args = parse_args() 
    main(args)
