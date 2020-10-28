import random
import numpy as np
from PIL import Image
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
from cgan import generate_sample

# Constructing Augmentation Search Space
# Utility functions for performing data augmentation on training images

def magnitude_selection(min_val, max_val, bin_val, total_bins):

    magnitudes = np.linspace(min_val, max_val, total_bins+1)
    magnitude = random.uniform(magnitudes[bin_val], magnitudes[bin_val+1])
    return magnitude

def shear_x(img, mag):

    return img.transform(img.size, PIL.Image.AFFINE, (1, mag, 0, 0, 1, 0))

def shear_y(img, mag):  

    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, mag, 1, 0))

def translate_x(img, mag): 

    mag = mag*img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, mag, 0, 1, 0))

def translate_y(img, mag): 

    mag = mag*img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, mag))

def rotate(img, mag):  

    return img.rotate(mag)

def auto_contrast(img, mag):

    return PIL.ImageOps.autocontrast(img)

def invert(img, mag):

    return PIL.ImageOps.invert(img)

def equalize(img, mag):

    return PIL.ImageOps.equalize(img)

def solarize(img, mag):

    return PIL.ImageOps.solarize(img, mag)

def posterize(img, mag):  

    mag = int(round(mag))
    return PIL.ImageOps.posterize(img, mag)

def contrast(img, mag):  

    return PIL.ImageEnhance.Contrast(img).enhance(mag)

def color(img, mag):  

    return PIL.ImageEnhance.Color(img).enhance(mag)

def brightness(img, mag):  

    return PIL.ImageEnhance.Brightness(img).enhance(mag)

def sharpness(img, mag):  

    return PIL.ImageEnhance.Sharpness(img).enhance(mag)

def cutout(img, mag, default=False):  

    mag = int(round(mag))

    if default:
        mask_size = 16
    else:
        mask_size = mag*img.size[0]   

    img = np.asarray(img).copy()
    h, w= img.shape[:2]

    x0 = np.random.randint(0, w-mask_size)
    y0 = np.random.randint(0, h-mask_size)

    img[y0:y0+mask_size, x0:x0+mask_size, :].fill(0)

    image = PIL.Image.fromarray(img)
    return image


def sample_pairing(imgs):  

    def f(img1, mag):
        i = np.random.choice(len(imgs))
        img2 = imgs[i]
        return PIL.Image.blend(img1, img2, mag)
    return f


# Conditional GAN
def cgan(img, label):
    label = random.randint(0, 100)
    gen_images = generate_sample(label)
    # convert the numpy array back to PIL image
    gen_images = 0.5 * gen_images + 0.5
    im = gen_images.squeeze()
    new_im = Image.fromarray((im * 255).astype(np.uint8))
    
    return new_im


def dagan():

    return


def get_operations(imgs, cutout_default=False):
    return [
        (shear_x, -0.3, 0.3),
        (shear_y, -0.3, 0.3),
        (translate_x, -0.45, 0.45),
        (translate_y, -0.45, 0.45),
        (rotate, -30, 30),
        (auto_contrast, 0, 1),
        (invert, 0, 1),
        (equalize, 0, 1),
        (solarize, 0, 256),
        (posterize, 4, 8),
        (contrast, 0.1, 1.9),
        (color, 0.1, 1.9),
        (brightness, 0.1, 1.9),
        (sharpness, 0.1, 1.9),
        (cutout, 0, 0.25, cutout_default),
        (sample_pairing(imgs), 0, 0.4),
        (cgan, 0, 1),
        # (dagan())
    ]

# Applying augmentation policy to an image
# One augmentation policy is sequential application of two transformations    

def apply_policy(img, policy, operations):

    op1, op1_binval, op2, op2_binval = policy

    op1_mag = magnitude_selection(operations[op1][1], operations[op1][2], op1_binval, 10)
    img = operations[op1][0](img, op1_mag)

    op2_mag = magnitude_selection(operations[op2][1], operations[op2][2], op2_binval, 10)
    img = operations[op2][0](img, op2_mag)

    return img
