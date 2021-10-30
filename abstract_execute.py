import os

import chainer
from chainer import cuda, serializers
from skimage import img_as_float
from skimage.io import imread, imsave
from skimage.transform import resize

from abstract.gp_gan import gp_gan
from abstract.model import EncoderDecoder, DCGAN_G

import cv2
import glob
import random
import numpy as np

class Abstract():
  def __init__(self):    
    if True: #args.supervised:
        self.G = EncoderDecoder(64, 64, 3, 4000, image_size=64)
        serializers.load_npz('./abstract/models/blending_gan.npz', self.G)
    else:
        chainer.config.use_cudnn = 'never'
        self.G = DCGAN_G(64, 3, 64)
        serializers.load_npz('./abstract/models/unsupervised_blending_gan.npz', self.G)

    #cuda.get_device(-1).use()  # Make a specified GPU current
    #self.G.to_gpu()  # Copy the model to the GPU

    self.mask_paths = glob.glob('./abstract/masks/*.png')


  def abstract_excute(self, img_paths):
    imgs = []

    #img_paths= ['1.jpg','2.jpg']

    for path in img_paths:
      t_img = cv2.imread(path)
      imgs.append(t_img)

    obj = img_as_float(imgs[1])
    bg = img_as_float(imgs[0])

    mask = imread(random.choice(self.mask_paths), as_gray=True).astype(obj.dtype)
    #mask = imread('./abstract/masks/4.png', as_gray=True).astype(obj.dtype)

    obj = resize(obj, (512, 512), anti_aliasing=True)
    bg = resize(bg, (512, 512), anti_aliasing=True)
    mask = resize(mask, (512, 512))


    #bg = resize(bg, (obj.shape[0], obj.shape[1]), anti_aliasing=True)
    #mask = resize(mask, (obj.shape[0], obj.shape[1]))
    
    with chainer.using_config("train", False):
      result = gp_gan(obj, bg, mask, self.G, 64, -1, color_weight=1,
                          sigma=0.1,
                          gradient_kernel='normal', smooth_sigma=1,
                          supervised=True,
                          nz=100, n_iteration=1000)
    result = cv2.cvtColor(np.array(result), cv2.COLOR_BGR2RGB)
    imsave('result.jpg', result)
    

    #imgs[1] = cv2.resize(imgs[1], dsize=(obj.shape[0], obj.shape[1]),interpolation=cv2.INTER_LINEAR)
    #result = cv2.seamlessClone(imgs[1], imgs[0], np.array(mask), (int(obj.shape[0]/2), int(obj.shape[1]/2)), cv2.NORMAL_CLONE)
    #cv2.imwrite('result1.jpg', result)
    
    return 0, result
