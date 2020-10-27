import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from utils import Config as cf
from augmentations import get_operations, apply_policy

# custom collator class used for collate_fn of dataloader
# performs data augmentation policies to the batched images
class My_Collator:

    def __init__(self, args):

        self.args = args
        self.policies = list(np.ones(cf.no_policies*4, dtype=np.uint8))

    def _batch_transform(self, batch):

        input_imgs = [item[0] for item in batch]
        targets = [item[1] for item in batch]

        pil_imgs = []
        for input_img in input_imgs:
            pil_imgs.append(transforms.ToPILImage()(input_img))

        if self.args.dataset in ['cifar10', 'cifar100']:
            operations = get_operations(pil_imgs, cutout_default=True)
        else:
            operations = get_operations(pil_imgs, cutout_default=False)

        augmented_imgs = []
        augmented_targets = []

        normalize_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)),])

        # augmentation policies are applied for each image in the mini batch
        # print("Policies >>>>>>  ", self.policies)
        for i in range(int(len(self.policies)/4)):
            i = i * 4
            # fix self.policies[i] to be a constant ops
            self.policies[i] = 16   # random.choice([16,17])
            policy = [self.policies[i], self.policies[i+1], self.policies[i+2], self.policies[i+3]]
            # print("\n Policy ******** ", policy)
            for pil_img, target in zip(pil_imgs, targets):
                augmented_imgs.append(normalize_transform(apply_policy(pil_img, policy, operations)))
                augmented_targets.append(target)

        augmented_targets = torch.tensor(augmented_targets)

        return torch.stack(augmented_imgs, dim=0), augmented_targets

    def __call__(self, batch):
        return self._batch_transform(batch)

# dataset initialization for cifar10/100 and imagenet
def init_dataloader(args):

    assert args.dataset in ['cifar10', 'cifar100', 'imagenet']

    if args.dataset == 'cifar10':

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5)),
            ])

        train_set = datasets.CIFAR10(
            root='~/data',
            train=True,
            download=True,
            transform=transform_train)

        test_set = datasets.CIFAR10(
            root ='~/data',
            train=False,
            download=True,
            transform = transform_test)

        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size = cf.clf_batch_size,
            shuffle    = False,
            num_workers= args.num_workers,
            pin_memory = True)

        num_classes = 10

    elif args.dataset == 'cifar100':

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5)),
            ])

        train_set = datasets.CIFAR100(
            root='~/data',
            train=True,
            download=True,
            transform=transform_train)

        test_set = datasets.CIFAR100(
            root ='~/data',
            train=False,
            download=True,
            transform = transform_test)

        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size = cf.clf_batch_size,
            shuffle    = False,
            num_workers= args.num_workers,
            pin_memory = True)

        num_classes = 100

    elif args.dataset =='imagenet':

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])

        train_set = datasets.ImageFolder(
            cf.traindir,
            transform = transform_train,
        )

        normalize = transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        
        test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(cf.valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=cf.clf_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

        num_classes = 1000


    return train_set, test_loader, num_classes