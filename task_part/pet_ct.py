import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter   
import numpy as np
from petCTDataset import PetCTDataset
import torch.nn.functional as F

# Import my custom dataset
from rgrDataset import RgrDataset
import csv

from datetime import datetime,timezone,timedelta


# All model names for chosing
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# Parameters
parser = argparse.ArgumentParser(description='TL pretrain on MRI')

# Useless parameters
parser.add_argument('data', metavar='DIR',
                    help='path to dataset (USELESS)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--usesmall', default=0, type=int,
                    help='1 for use and 0 for not use')
parser.add_argument('--findlr', default=None, type=int,
                    help='any type for find mode')
parser.add_argument('--dropout', default=0, type=float,
                    help='dropout ratio')

# Key (hyper) parameters
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

# Paths for data
parser.add_argument('--train-dir', type=str, default='./../../../datasets/final_mr_ct/trainB/',
                    help='path to the train folder, each class has a single folder')
parser.add_argument('--val-dir', type=str, default='./../../../datasets/final_mr_ct/valB/',
                    help='path to the validation folder, each class has a single folder')
parser.add_argument('--test-dir', type=str, default='./../../../datasets/final_mr_ct/testB/',
                    help='path to the test folder, each class has a single folder')
parser.add_argument('--auge-dir', type=str, default='./../../../datasets/final_mr_ct/pet_cyc/pet_fake_imgs/',
                    help='path to the train folder, each class has a single folder')
parser.add_argument('--augement', default='pure', type=str,
                    help='choose pure or mr_pure or toge or expe')

# Log and save/load path
parser.add_argument('--logpath', default='logs/log_pet_ct', type=str,
                    help='Path to store tensorboard log files.')
parser.add_argument('--resumedir', default='', type=str, metavar='PATH',
                    help='path of DIR to latest checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to load latest checkpoint (default: none)')

# Other useful parameters
parser.add_argument('--times', default=1, type=int,
                    help='run the whole file for times')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')

# My own params >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
parser.add_argument('--filename', default='save_number_file.csv', type=str,
                    help='file to save loss')


# Parameter assignment
args = parser.parse_args()

# Tensorboard writer
writer = SummaryWriter(args.logpath)


# Other variables initialization
best_loss = 0
dt = datetime.utcnow()
dt = dt.replace(tzinfo=timezone.utc) 
tzutc_8 = timezone(timedelta(hours=8))
TIMESTAMP = str(dt.astimezone(tzutc_8))
dirName=args.resumedir + 'checkpoint-' + TIMESTAMP + '/'
os.mkdir(dirName)


def main():
    if args.train_dir is None or args.val_dir is None:
        print('[Error!] Enter the data(train & test) path')

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)
    

def main_worker(gpu, ngpus_per_node, args):
    global best_loss
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        
    # Device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device {}".format(device))

    # Create model
    print('[INFO] Random Initialize ...')
    model = models.__dict__[args.arch](num_classes=1, pretrained=False)
    if args.dropout != 0:
        model.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=args.dropout, training=m.training))

    
    # Data parallel for multiple GPU usage
    if torch.cuda.device_count() > 1:
        print("[INFO] Let's use ", torch.cuda.device_count(), " GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Define loss function (criterion)
    criterion = nn.MSELoss()

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), 
                                 eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
    
    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_loss = best_loss.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = args.train_dir
    valdir = args.val_dir
    
    # Set normalize parameters
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # Train data augmentations
    train_trans = transforms.Compose(
                        [
                            transforms.Resize(256),
#                             transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize
                        ]
                    )
    
    # Validation & test data augmentations
    val_test_trans = transforms.Compose(
                        [
                            transforms.Resize(256),
#                             transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize
                        ]
                    )
    
    # Datasets (train, validation and test)
    trainset = PetCTDataset(pet_csv='./pet.csv', ct_csv='./ct.csv', 
                            pet_root_dir=args.auge_dir, ct_root_dir=args.train_dir, transform=train_trans)
    
    valset = RgrDataset('ct.csv', args.val_dir, args.augement, transform=val_test_trans)
    
    testset = RgrDataset('ct.csv', args.test_dir, args.augement, transform=val_test_trans)
 
    # Dataloaders (train, validation and test)
    train_loader = torch.utils.data.DataLoader(
                        trainset,
                        batch_size=args.batch_size,
                        num_workers=8,
                        sampler=None,
                        shuffle=True
                    )
    
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size)
    
    test_loader = torch.utils.data.DataLoader(testset,batch_size=args.batch_size)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    
    with open(args.filename, "w") as csvfile: 
        writer1 = csv.writer(csvfile)
        time_cnt = 1
        for i in range(args.times):
            outcome = 0
            best_loss = 9999999
            for epoch in range(args.start_epoch, args.epochs):
                lr = adjust_learning_rate(optimizer, epoch, args)

                # Train for one epoch
                loss_tr = train(train_loader, model, criterion, optimizer, epoch, args)

                # Evaluate on validation set
                loss_val = validate(val_loader, model, criterion, args, epoch)

                writer.add_scalars('Loss' + str(i), {
                    'Train': loss_tr,
                    'Val': loss_val
                }, epoch)
                writer.flush()

                # Remember best acc@1 and save checkpoint
                is_best = loss_val < best_loss
                best_loss = min(loss_val, best_loss)
                
                if (args.epochs - epoch) <= 10:
                    outcome += loss_val

                if is_best and epoch % 10 != 0:
                    save_checkpoint(model.state_dict(), is_best, epoch, optimizer)

                if epoch % 10 == 0:
                    save_checkpoint(model.state_dict(), is_best, epoch, optimizer)
            
            # Start test
            print('...........Testing Start..........')
            print('Loading best checkpoint ...')

            # Load best checkpont
            load_resume(args, model, optimizer, args.resumedir)

            # Test the best checkpoint
            loss_test = validate(test_loader, model, criterion, args, epoch)

            # Record the test loss
            record = {
                'loss_test': loss_test,
            }
            torch.save(record, os.path.join(args.resumedir, 'test_info.pth.tar'))

            print('[INFO] TEST LOSS: ' + str(loss_test))
            print('...........Testing End..........')
            
            outcome /= 10    
            writer1.writerow(str(loss_test) + "\n")
            print("[INFO] Time " + str(time_cnt) + ", test: " + str(loss_test))
            model = models.__dict__[args.arch](num_classes=1, pretrained=False)
            model = model.cuda(args.gpu)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                         eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
            criterion = nn.MSELoss()
            time_cnt += 1


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        else:
            images = images.cuda(0, non_blocking=True)
        target = target.type(torch.FloatTensor).view(-1, 1).cuda(args.gpu, non_blocking=True)
    
        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.item(), images.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            
    return losses.avg


def validate(val_loader, model, criterion, args, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.view(-1, 1).type(torch.FloatTensor).cuda(args.gpu, non_blocking=True)
            # compute output
            output = model(images)
            loss = criterion(output, target)
            
            # measure accuracy and record loss
            losses.update(loss.item(), images.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print('[INFO] Validation MSE: {loss}'.format(loss=losses.avg))
        
    return losses.avg

# Util function: load resume
def load_resume(args, model, optimizer, load_path):
    if load_path:
        # Default load best.pth.tar
        load_path = os.path.join(load_path, 'model_best.pth.tar')
        if os.path.isfile(load_path):
            print("=> loading checkpoint '{}'".format(load_path))
            checkpoint = torch.load(load_path)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(load_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(load_path))
    else:
        print('[ERROR] No load path provided ...') 
    
def save_checkpoint(state, is_best, epoch, optimizer, checkpoint_path=args.resumedir):
    record = {
        'epoch': epoch + 1,
        'state_dict': state,
        'optimizer': optimizer.state_dict(),
    }
    filename = os.path.join(checkpoint_path, 'record_epoch{}.pth.tar'.format(epoch))
    torch.save(record, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_path, 'model_best.pth.tar'))


        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.findlr:
        lr = args.lr * 1.05
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    main()
