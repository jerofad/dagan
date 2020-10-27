import torch
import torch.nn as nn
from torchvision import transforms

import numpy as np
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont
from augmentations import get_operations, apply_policy

class Config:

    """if you use distributed training 
      effective batch_size  per node after augmentations becomes 

      = clf_batch_size * gpus_per_node * no_policies

      if you use single gpu 
      effective batch_size becomes = clf_batch_size * no_policies

      so configure based on available gpu memory
    """
    clf_batch_size = 128

    # classifier initial learning rate
    clf_init_lr = 0.0001

    # controller params
    ctl_init_lr = 0.00035
    no_policies = 2
    entropy_penalty = 0.00001

    total_epochs = 1

    # reward_moving_average_mometum
    momentum = 0.1

    #print_frequency
    print_freq = 10

    checkpoint_dir = 'ckpts/'
    logdir = 'training_logs/'

    #imagenet directories
    traindir = 'train/'
    valdir =  'val/'

class ProgressMeter(object):
    """ Prints metrics during training and validation """

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

def save_augmented_images(train_set, policies, filename, args):

    """Saves augmented images after applying policies
       For each policy 4 images are saved
       if there are n policies a grid image with n rows and 
       4 images per row will be saved
    """

    visualize_data_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                        batch_size = 8,
                                                        shuffle=False,
                                                        num_workers=1,
                                                        drop_last = True)


    for (input_imgs, target) in visualize_data_loader:

        pil_imgs=[]
        for input_img in input_imgs:
            pil_imgs.append(transforms.ToPILImage()(input_img))

        if args.dataset in ['cifar10', 'cifar100']:
            operations = get_operations(pil_imgs, cutout_default=True)
        else:
            operations = get_operations(pil_imgs, cutout_default=False)

        augmented_imgs = []
        for i in range(int(len(policies)/4)):
            i = i * 4
            policy = [policies[i], policies[i+1], policies[i+2], policies[i+3]]
            
            for j, pil_img in enumerate(pil_imgs):
                if j==4:
                    break
                augmented_img = apply_policy(pil_img, policy, operations)
                text = operations[policy[0]][0].__name__ + " " + str(policy[1].item()) + "\n" + \
                        operations[policy[2]][0].__name__ + " " + str(policy[3].item())

                augmented_img = add_text_toImage(augmented_img, text)
                augmented_imgs.append(np.asarray(augmented_img))

        augmented_imgs = np.stack(augmented_imgs, axis=0)
        save_image_grid(augmented_imgs, Config.logdir+filename)
        break

def add_text_toImage(image, text):
    """ Utility function to write text at the bottom of image"""

    old_im_size = image.size
    if image.size[0] <=80:
        new_size = (image.size[0]+40, image.size[1]+40)
    else:
        new_size = (image.size[0], image.size[1]+40)

    new_im = Image.new("RGB", new_size)   ## luckily, this is already black!
    new_im.paste(image, (0,0))

    fnt = ImageFont.truetype(fm.findfont(fm.FontProperties(family='Sans Serif')), 10)
    d = ImageDraw.Draw(new_im)
    d.text((0, image.size[1]), text, font=fnt, fill=(255, 255, 255))
    return new_im


def make_grid(images, grid_size=(Config.no_policies, 4)):
    """ Helper function to make a grid-image from images"""
    grid_h, grid_w = grid_size
    print(images.shape)
    img_h, img_w = images.shape[1], images.shape[2]
    grid = np.zeros([grid_h*img_h, grid_w*img_w, 3], dtype=images.dtype)
    for idx in range(images.shape[0]):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[y : y + img_h, x : x + img_w, :] = images[idx]
    return grid

def convert_to_pil_image(image):
    """ Helper function to convert to PIL image from numpy array"""
    assert image.ndim == 3
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    format = 'RGB' if image.ndim == 3 else 'L'
    return Image.fromarray(image, format)

def save_image(image, filename, quality=95):

    img = convert_to_pil_image(image)
    if '.jpg' in filename:
        img.save(filename,"JPEG", quality=quality, optimize=False)
    else:
        img.save(filename)

def save_image_grid(images, filename, grid_size=(Config.no_policies, 4)):
    """Saves image_grid at specified location"""
    save_image(make_grid(images, grid_size), filename)
