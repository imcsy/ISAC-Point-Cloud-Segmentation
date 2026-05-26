"""
train pointguard downstream discriminator (binary)
"""
import argparse
import os
from data_utils.S3DISDataLoader import S3DISDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time
from data_utils.ModelNetDataLoader_clean_per_inj import ModelNetDataLoader_clean_per_inj

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
#            'board', 'clutter']
classes = ['attacked', 'clean']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointguard_discriminator', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=5, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--num_point', type=int, default=16, help='Point Number [default: 1024]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    # parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--samples_per_frame', type=int, default=4, help='Number of samples obtained from each frame')
    # add dropout, shift or not
    parser.add_argument('--dropout', action='store_true', default=False, help='use dropout when training')
    parser.add_argument('--shift', action='store_true', default=False, help='use shift when training') 
    # probability of two attacks
    parser.add_argument('--per_prob', type=float, default=0.4, help='Data proportion of Perturbation')
    parser.add_argument('--inject_prob', type=float, default=0.4, help='Data Proportion of Injection')
    # add parameters for injection attack
    parser.add_argument('--npoints_inj', type=int, default=4, help='Number of Points Injected')
    parser.add_argument('--clutter_size_inj', type=int, default=2, help='The approximate number od points for the injected clutter')
    # add parameters for perturbation attack
    parser.add_argument('--channels_per', type=int, nargs='+', default=[0, 1, 2, 3], help='Channels of Perturbation')
    parser.add_argument('--eps_per', type=float, default=1.5, help='Eps of Perturbation')
    # keep some parameters just to pass to ModelLoader
    parser.add_argument('--num_category', default=2, type=int, choices=[2, 10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_channel', type=int, default=5, help='Input Channel Number')  

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)

    param_name = f"epoch_{args.epoch}_npoint_{args.num_point}_bsize_{args.batch_size}"
    if args.dropout:
        param_name = param_name + "_dropout"
    if args.shift:
        param_name = param_name + "_shift"
    experiment_dir = experiment_dir.joinpath(param_name)

    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
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

    data_path = '/content/drive/MyDrive/THESIS_dataset/mmw/MyModelNet_cls'

    NUM_CLASSES = 2
    NUM_POINT = args.num_point
    BATCH_SIZE = args.batch_size

    train_dataset = ModelNetDataLoader_clean_per_inj(root=data_path, args=args, split='train', process_data=False, per_prob=args.per_prob, inject_prob=args.inject_prob)
    test_dataset = ModelNetDataLoader_clean_per_inj(root=data_path, args=args, split='test', process_data=False, per_prob=args.per_prob, inject_prob=args.inject_prob)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
    weights = torch.tensor([0.7, 0.3]).cuda()

    log_string("The number of training data is: %d" % len(train_dataset))
    log_string("The number of test data is: %d" % len(test_dataset))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))

    classifier = MODEL.get_model().cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_point_acc = 0

    for epoch in range(start_epoch, args.epoch):
        '''Train on UNchopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()

        for i, (points_aug, _, _) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            points_aug = points_aug.float()
            target = points_aug[:, :, 4]
            target[target<1] = 0

            points_aug = points_aug.data.numpy()
            points_aug = torch.Tensor(points_aug)
            points_aug, target = points_aug.float().cuda(), target.long().cuda()
            points_aug = points_aug.transpose(2, 1)

            seg_pred = classifier(points_aug)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, weights)
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss
        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))

        # '''Evaluate on UNchopped scenes'''
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            classifier = classifier.eval()

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (points_aug, _, _) in tqdm(enumerate(testDataLoader), total=len(trainDataLoader), smoothing=0.9):
                points_aug = points_aug.float()
                target = points_aug[:, :, 4]
                target[target<1] = 0

                points_aug = points_aug.data.numpy()
                points_aug = torch.Tensor(points_aug)
                points_aug, target = points_aug.float().cuda(), target.long().cuda()
                points_aug = points_aug.transpose(2, 1)
                
                seg_pred = classifier(points_aug)
                pred_val = seg_pred.contiguous().view(-1, NUM_CLASSES).cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, weights)
                loss_sum += loss
                pred_val = np.argmax(pred_val, -1)
                
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp

            point_acc = total_correct / float(total_seen)
            log_string('eval point accuracy: %f' % (point_acc)) 

            if point_acc >= best_point_acc:
                best_point_acc = point_acc
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'point_acc': best_point_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best Point Accuracy: %f' % best_point_acc)
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
