import argparse
import os
import random
import shutil
import time
import warnings
import sys
import csv
import distutils
from contextlib import redirect_stdout
from collections import OrderedDict

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

from torchsummary import summary
import copy

import tensorly as tl
import tensorly
from decompositions import decompose_model

import resnet_cifar10

'''
Unfortunately, none of the pytorch repositories with ResNets on CIFAR10 provides an 
implementation as described in the original paper. If you just use the torchvision's 
models on CIFAR10 you'll get the model that differs in number of layers and parameters. 
This is unacceptable if you want to directly compare ResNet-s on CIFAR10 with the 
original paper. The purpose of resnet_cifar10 (which has been obtained from https://github.com/akamaster/pytorch_resnet_cifar10
is to provide a valid pytorch implementation of ResNet-s for CIFAR10 as described in the original paper. 
'''

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and not name.startswith("resnet")
    and callable(models.__dict__[name]))
model_names.extend(sorted(name for name in resnet_cifar10.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet_cifar10.__dict__[name])))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--model', default='', type=str, metavar='MODEL_PATH',
                    help='path to model file to load both its architecture and weights (default: none)')
parser.add_argument("--decompose", dest="decompose", action="store_true")
parser.add_argument("--cp", dest="cp", action="store_true", \
                    help="Use cp decomposition. uses tucker by default")
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-schedule', dest='lr_schedule', default=True, type=lambda x:bool(distutils.util.strtobool(x)), 
                    help='using learning rate schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--opt-ckpt', default='', type=str, metavar='OPT_PATH',
                    help='path to checkpoint file to load optimizer state (default: none)')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='CHECKPOINT_PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='only evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default="none", 
                    help='choose whether model is not pre-trained, or pre-trained on ImageNet, or on CIFAR10')
parser.add_argument('--freeze', dest='freeze', default=False, type=lambda x:bool(distutils.util.strtobool(x)), 
                    help='freeze pre-trained weights')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--desc', type=str, default=None,
                    help='description to append to model directory name')


best_acc1 = 0


def main():
    tl.set_backend("pytorch")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create or load model
    if args.model:
        if args.arch or (args.pretrained and args.pretrained != "none"):
            print("WARNING: Ignoring arguments \"arch\" and \"pretrained\" when creating model...")
        model = None
        saved_checkpoint = torch.load(args.model)
        if isinstance(saved_checkpoint, nn.Module):
            model = saved_checkpoint
        elif "model" in saved_checkpoint:
            model = saved_checkpoint["model"]
        else:
            raise Exception("Unable to load model from " + args.model)
        
    # TODO: Clean up this code. Create separate function to load each CIFAR10 version of a model. And then load its weights if needed
    elif args.arch in ["resnet20", "resnet32", "resnet44", "resnet56", "resnet110", "resnet1202", "vgg11_bn"] and args.pretrained != "imagenet":
        if args.pretrained == "none": 
            model = resnet_cifar10.__dict__[args.arch]()
        elif args.pretrained == "cifar10":
            if args.arch.startswith("resnet"):
                model = resnet_cifar10.__dict__[args.arch]()
                
                # original saved file with DataParallel
                saved_checkpoint = torch.load("./models/cifar10/original_pretrained_models/" + args.arch + ".th")
                state_dict = saved_checkpoint["state_dict"]
                
                # create new OrderedDict that does not contain module.
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove module.
                    new_state_dict[name] = v
                
                # load params
                model.load_state_dict(new_state_dict)
            elif args.arch in ["vgg11_bn"]:
                model = torch.load("./models/cifar10/original_pretrained_models/" + args.arch + ".th")
        else:
            raise Exception("Currently model {} does not support weights {}".format(args.arch, args.pretrained))
    else:
        if args.pretrained == "none": 
            model = models.__dict__[args.arch]()
        elif args.pretrained == "imagenet":
            model = models.__dict__[args.arch](pretrained=True)
        elif os.path.exists(args.pretrained):
            model = models.__dict__[args.arch]()
            state_dict = None
            
            saved_checkpoint = torch.load(args.pretrained)
            if "state_dict" in saved_checkpoint:
                state_dict = saved_checkpoint["state_dict"]
            elif "model" in saved_checkpoint:
                model = saved_checkpoint["model"]
                state_dict = model.state_dict()
            else:
                state_dict = saved_checkpoint
						
			# load params
            model.load_state_dict(state_dict)
        else:
            raise Exception("Currently model {} does not support weights {}".format(args.arch, args.pretrained))

    #TODO: add option for finetune vs. feature extraction that only work if pretrained weights are imagenet    
    if args.freeze and args.pretrained != "none":
        for param in model.parameters():
            param.requires_grad = False

    num_classes = 10
    if not args.model:
        # Parameters of newly constructed modules have requires_grad=True by default
        # TODO: Check https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html to handle different models
        if args.arch.startswith("resnet"):
            # TODO: handle if model came from imagenet or cifar10
            # change FC layer to accomodate dataset labels
            '''
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            ''' 
        elif args.arch == "alexnet":
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs,num_classes)
        elif args.arch == "vgg11_bn":
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs,num_classes)
        elif args.arch == "vgg16":
            model.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes))
        elif args.arch == "squeezenet1_0":
            model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
            model.num_classes = num_classes
        elif args.arch == "densenet121":
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)
        elif args.arch == "inception_v3":
            # Handle the auxilary net
            num_ftrs = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs,num_classes)
        else:
            raise ValueError("Unfortunately ", args.arch, " is not yet supported for CIFAR10")

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.opt_ckpt:
        print("WARNING: Ignoring arguments \"lr\", \"momentum\", \"weight_decay\", and \"lr_schedule\"")

        checkpoint = torch.load(args.opt_ckpt)
        optimizer = checkpoint['optimizer']
        if 'lr_schedule' in checkpoint:
            lr_scheduler = checkpoint['lr_schedule']
        elif (args.lr_schedule):
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                milestones=[100, 150], last_epoch=args.start_epoch - 1)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        if (args.lr_schedule):
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                milestones=[100, 150], last_epoch=args.start_epoch - 1)

    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this implementation it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'lr_schedule' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    print("Original Model:")
    print(model)
    print("\n\n")
        

    if args.decompose:
        print("Decomposing...")
        model = decompose_model(model, args.cp)
        print("\n\n")

        print("Decompose Model:")
        print(model)
        print("\n\n")

    cudnn.benchmark = True

    # TODO: make this summary function deal with parameters that are not named "weight" and "bias"
    model_tmp_copy = copy.deepcopy(model) # we noticed calling summary() on original model degrades it's accuracy. So we will call summary() on a copy of the model
    summary(model_tmp_copy, input_size=(3, 32, 32))
    print("WARNING: The summary function is not counting properly parameters in custom layers")

    # name model directory
    if (args.decompose):
        decompose_label = "tucker_decompose"
        if (args.cp):
            decompose_label += "cp_decompose"
    else:
        decompose_label = "no_decompose"

    if args.desc is not None and len(args.desc) > 0:
        model_name = '%s/%s_%s' % (args.arch, args.desc, decompose_label)
    else:
        model_name = '%s/%s' % (args.arch, decompose_label)

    model_dir = os.path.join(os.path.join(os.path.join(os.getcwd(), "models"), "cifar10"), model_name)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, 'command_args.txt'), 'w') as command_args_file:
        for arg, value in sorted(vars(args).items()):
            command_args_file.write(arg + ": " + str(value) + "\n")

    with open(os.path.join(model_dir, 'model_summary.txt'), 'w') as summary_file:
        with redirect_stdout(summary_file):
            # TODO: make this summary function deal with parameters that are not named "weight" and "bias"
            summary(model_tmp_copy, input_size=(3, 32, 32))
            print("WARNING: The summary function is not counting properly parameters in custom layers")

    del model_tmp_copy # to save memory

    # Data loading code
    data_dir = "~/pytorch_datasets"
    os.makedirs(model_dir, exist_ok=True)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root=data_dir, 
            train=False, 
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        val_log = validate(val_loader, model, criterion, args)
        val_log = [val_log]

        with open(os.path.join(model_dir, "test_log.csv"), "w") as test_log_file:
            test_log_csv = csv.writer(test_log_file)
            test_log_csv.writerow(['test_loss', 'test_top1_acc', 'test_time'])
            test_log_csv.writerows(val_log)
    else:
        train_log = []

        with open(os.path.join(model_dir, "train_log.csv"), "w") as train_log_file:
            train_log_csv = csv.writer(train_log_file)
            train_log_csv.writerow(['epoch', 'train_loss', 'train_top1_acc', 'train_time', 'test_loss', 'test_top1_acc', 'test_time'])

        print(args.start_epoch, " ", args.epochs)

        for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            adjust_learning_rate(optimizer, epoch, args)

            # train for one epoch
            print('current lr {:.4e}'.format(optimizer.param_groups[0]['lr']))
            train_epoch_log = train(train_loader, model, criterion, optimizer, epoch, args)
            if (args.lr_schedule):
                lr_scheduler.step()

            # evaluate on validation set
            val_epoch_log = validate(val_loader, model, criterion, args)
            acc1 = val_epoch_log[2]

            # append to log
            with open(os.path.join(model_dir, "train_log.csv"), "a") as train_log_file:
                train_log_csv = csv.writer(train_log_file)
                train_log_csv.writerow(((epoch,) + train_epoch_log + val_epoch_log)) 

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                if is_best:
                    try:
                        torch.save(model, os.path.join(model_dir, "model.pth"))
                    except: 
                        print("WARNING: Unable to save model.pth")
                
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                    'lr_scheduler' : lr_scheduler,
                }, is_best, model_dir)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)
    
    return (losses.avg, top1.avg.cpu().numpy(), batch_time.avg)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f}'
              .format(top1=top1))

    return (losses.avg, top1.avg.cpu().numpy(), batch_time.avg)


def save_checkpoint(state, is_best, dir_path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(dir_path, filename))
    if is_best:
        shutil.copyfile(os.path.join(dir_path, filename), os.path.join(dir_path, 'model_best.pth.tar'))


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
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
