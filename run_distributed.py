import os
import time
import random
import shutil
import argparse
import pickle
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.optim import lr_scheduler

from utils import *
from utils import Config as cf
from resnet import resnet50
from wideresnet import wideresnet28_10
from policy_controller import Policy_Controller
from data_loader import init_dataloader, My_Collator


best_acc1 = 0

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--classifier', default='wideresnet28-10',
                    choices=['wideresnet28-10', 'resnet-50'],help='classifier architecture')
    parser.add_argument('--dataset', default='cifar10',
                    choices=['cifar10', 'cifar100', 'imagenet'],help='dataset name')

    parser.add_argument('--train', help='train adv-augment', action='store_true')
    parser.add_argument('--validate',help='validate img_classifier', action='store_true')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('--with_GAN', action='store_true', default=False,
                        help='Use GAN as fixed augmentation')

    parser.add_argument('--dist-url', default='', type=str,
                    help='url used to set up distributed training')

    parser.add_argument('-n_id', '--node_id', default=0, type=int,help='ranking within the nodes')
    parser.add_argument('-n', '--nodes', default=1, type=int, help='number of nodes')
    parser.add_argument('-g_id', '--gpu_id', default=0, type=int, help='specific gpu_id to use')
    parser.add_argument('-n_w', '--num_workers', default=4, type=int,
                                               help='number of data loading workers (default: 4)')

    parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
    args = parser.parse_args()


    ngpus_per_node = torch.cuda.device_count()
    print("Available GPUS: {}".format(ngpus_per_node))

    # use 127.0.0.1 if you are using single node multi-gpu training
    # This ip address used only when args.dist_url is None
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8888'

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.nodes
        print(args.world_size)
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        # Make sure number of processes equals the number of gpus per node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        # Used when only one gpu is avilable
        main_worker(args.gpu_id, ngpus_per_node, args)


def main_worker(gpu_id, ngpus_per_node, args):

    """
    Main worker function that spawns multiple processes  
    Args:

    gpu_id : which gpu or cuda device id to use
    ngpus_per_node: total visible cuda devices
    args: parsed args from main function

    This function can be used for multi-node multi-gpu/ 
    single-node multi-gpu/single-node single-gpu training scenarios

    """
 
    global best_acc1 

    args.gpu_id = gpu_id

    print("Use GPU: {} for training".format(args.gpu_id))

    rank = args.node_id * ngpus_per_node + args.gpu_id

    # initialize distributed process group
    if args.multiprocessing_distributed:
        if args.dist_url:
            dist.init_process_group(backend='nccl', init_method=args.dist_url,
                                          world_size=args.world_size, rank = rank)
        else:
            dist.init_process_group(backend='nccl', init_method='env://',
                                          world_size=args.world_size, rank = rank)

        args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)

    torch.manual_seed(0)       # all the processes should start from same state

    torch.cuda.set_device(args.gpu_id)

    train_set, test_loader, num_classes = init_dataloader(args)

    if args.classifier == 'wideresnet28-10':
        classifier = wideresnet28_10( num_classes=num_classes)

    elif args.classifier == 'resnet-50':
        classifier = resnet50(num_classes=num_classes)

    classifier = classifier.cuda(args.gpu_id)

    criterion = nn.CrossEntropyLoss(reduction ='none')
    criterion.cuda(args.gpu_id)

    optimizer_clf = optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=cf.clf_init_lr, 
             betas=(0.9, 0.999), eps=1e-08)

    controller = Policy_Controller()
    controller.cuda(args.gpu_id)

    optimizer_ctl = optim.Adam(filter(lambda p: p.requires_grad, controller.parameters()), lr=cf.ctl_init_lr, 
             betas=(0.9, 0.999), eps=1e-08)

    # distribute classifier and polciy controller module across GPUs
    if args.multiprocessing_distributed:
        classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[args.gpu_id])
        controller = torch.nn.parallel.DistributedDataParallel(controller, device_ids=[args.gpu_id])
        
    cudnn.benchmark = True

    # resumes training from previous checkpoint if args.resume is true
    if args.resume:

        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))

            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu_id)
            checkpoint = torch.load(args.resume, map_location=loc)

            start_epoch = checkpoint['epoch']
            best_acc1   = checkpoint['best_acc1']
            if args.gpu_id is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu_id)
            classifier.load_state_dict(checkpoint['classifier_state_dict'])
            optimizer_clf.load_state_dict(checkpoint['optimizer_clf'])
            controller.load_state_dict(checkpoint['controller_state_dict'])
            optimizer_ctl.load_state_dict(checkpoint['optimizer_ctl'])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        start_epoch = 0


    if args.multiprocessing_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set,
                                                                num_replicas=args.world_size,rank=rank)
    else:
        train_sampler = None


    # custom collate function to be used with dataloader
    # this collate function applies augmentation policies on the mini-batch images
    my_collator = My_Collator(args)

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size =cf.clf_batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.num_workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               collate_fn = my_collator,
                                               drop_last = True)


    if args.validate:
        validate(test_loader, classifier, criterion, args)
        return

    controller.train()

    for epoch in range(start_epoch, cf.total_epochs):

        if args.multiprocessing_distributed:
            train_sampler.set_epoch(epoch)

        # policy controller forward call
        x = torch.tensor([0]).cuda(args.gpu_id)
        logprobs, entropies, policies = controller(x=x)

        if policies.is_cuda:
            policies.cpu()

        # set policies attribute to sampled policies so that dataloader can generate augmented imgs
        my_collator.policies = policies

        print(my_collator.policies)

        controller(rewards_reset=True)

        adjust_learning_rate(optimizer_clf, epoch)

        train(train_loader, classifier, criterion, controller, optimizer_clf, epoch, args)

        acc1 = validate(test_loader, classifier, criterion, args)

        # remember best acc@1 and save checkpoint

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        pg_loss = torch.mean(logprobs * controller(rewards_infer=True))  
        entropy_loss = torch.mean(entropies)

        loss_ctl = -( pg_loss + cf.entropy_penalty * entropy_loss )

        print("Policy_Controller Loss:{:.4e}".format(loss_ctl))

        optimizer_ctl.zero_grad() 
        loss_ctl.backward()     
        optimizer_ctl.step()

        if (rank % ngpus_per_node == 0):

            save_checkpoint({
                'epoch': epoch + 1,
                'classifier_state_dict': classifier.state_dict(),
                'best_acc1': best_acc1,
                'optimizer_clf' : optimizer_clf.state_dict(),
                'controller_state_dict':controller.state_dict(),
                'optimizer_ctl' : optimizer_ctl.state_dict(),
                'last_policies' : my_collator.policies,
                }, is_best)
        if rank == 0:
            filename = 'augmentations_epoch{}.jpg'.format(epoch)
            save_augmented_images(train_set, my_collator.policies, filename, args)

def train(train_loader, classifier, criterion, controller, optimizer_clf, epoch, args):

    """ Train function for classifier. """

    batch_time = AverageMeter('Time', ':6.3f')
    data_time  = AverageMeter('Data', ':6.3f')
    loss_vals  = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(
        len(train_loader),[batch_time, data_time, loss_vals],
        prefix="Epoch: [{}]".format(epoch))

    classifier.train()

    end = time.time()

    for i,  (input_imgs, target) in enumerate(train_loader):

        data_time.update(time.time() - end)

        input_imgs = input_imgs.cuda(args.gpu_id, non_blocking=True)
        target = target.cuda(args.gpu_id, non_blocking=True)

        #print("max: ",input_imgs.max()," min: ", input_imgs.min())

        logits = classifier(input_imgs)
        losses = criterion(logits, target)
            
        clf_loss = torch.mean(losses)
        optimizer_clf.zero_grad()
        clf_loss.backward()

        optimizer_clf.step()

        rewards = []
        for p, loss in enumerate(losses.split(cf.clf_batch_size)):
            loss_sum = torch.sum(loss)
            loss_p = ((loss_sum.detach())/(cf.clf_batch_size * cf.no_policies))
            rewards.append(loss_p)

        rewards = torch.tensor(rewards).cuda(args.gpu_id)
    
        controller(rewards_update_vals = rewards)   #moving average update

        loss_vals.update(clf_loss.item(), input_imgs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % cf.print_freq == 0:
            progress.display(i)
        

def validate(val_loader, classifier, criterion, args):

    """ validate function for classifier"""

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    # switch to evaluate mode
    classifier.eval()

    with torch.no_grad():
        end = time.time()

        for i, (input_imgs, target) in enumerate(val_loader):

            input_imgs = input_imgs.cuda(args.gpu_id, non_blocking=True)
            target = target.cuda(args.gpu_id, non_blocking=True)

            output = classifier(input_imgs)
            loss = criterion(output, target)
            loss = torch.mean(loss)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input_imgs.size(0))
            top1.update(acc1[0], input_imgs.size(0))
            top5.update(acc5[0], input_imgs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % cf.print_freq == 0:
                progress.display(i)

    # TODO: this should also be done with the ProgressMeter
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint_cifar100_2_200_wgan.pth.tar'):
    """saves checkpoint and best_model"""
    ckpt_dir = cf.checkpoint_dir 
    torch.save(state, ckpt_dir + filename)
    if is_best:
        shutil.copyfile(ckpt_dir + filename, ckpt_dir + 'model_best_cifar100_2_200_wgan.pth.tar')

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = cf.clf_init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()
