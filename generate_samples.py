"""Python script to generate samples and generated images
"""

import numpy as np
from numpy.random import randint
from keras.models import load_model

import imageio


# generate points in latent space as input for the generator
def generate_cgan_sample(class_for, n_samples=1):
    '''
    Generates new random but very realistic features using
    a trained generator model

    Params:
        class_for: Int - features for this class
        n_samples: Int - how many samples to generate
    '''

    noise = np.random.uniform(0, 1, (n_samples, 100))
    label = np.full((n_samples,), fill_value=class_for)
    # load model
    model = load_model('cgan_generator.h5')
    # incase to reshape
    # x.reshape(32,32,3)
    return model.predict([noise, label])


def save_sample(n_samples):
    label = randint(0, 100)
    gen_imgs = generate_cgan_sample(label, n_samples=n_samples)
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    # images = np.zeros((176,128,128,3))
    for i in range(len(gen_imgs)):
        imageio.imwrite('cgan_set_' + str(i) + '.jpg',
                        gen_imgs[i, :, :, :])