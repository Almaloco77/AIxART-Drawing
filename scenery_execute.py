from __future__ import division, print_function
import argparse
from PIL import Image
import numpy as np
import cv2 
import os
from scenery.smooth_local_affine import smooth_local_affine

import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
#if type(tf.contrib) != type(tf): tf.contrib._warning = None
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from scenery.vgg19.vgg import Vgg19
from PIL import Image
import time
from scenery.closed_form_matting import getLaplacian
import math
from functools import partial
import copy
import os 


try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

VGG_MEAN = [103.939, 116.779, 123.68]


parser = argparse.ArgumentParser()
# Input Options
parser.add_argument("--content_image_path", dest='content_image_path',  nargs='?', help="Path to the content image")
parser.add_argument("--style_image_path",   dest='style_image_path',    nargs='?', help="Path to the style image")
parser.add_argument("--content_seg_path",   dest='content_seg_path',    nargs='?', help="Path to the style segmentation")
parser.add_argument("--style_seg_path",     dest='style_seg_path',      nargs='?', help="Path to the style segmentation")
parser.add_argument("--init_image_path",    dest='init_image_path',     nargs='?', help="Path to init image", default="")
parser.add_argument("--output_image",       dest='output_image',        nargs='?', help='Path to output the stylized image', default="best_stylized.png")
parser.add_argument("--serial",             dest='serial',              nargs='?', help='Path to save the serial out_iter_X.png', default='./')

# Training Optimizer Options
parser.add_argument("--max_iter",           dest='max_iter',            nargs='?', type=int, help='maximum image iteration', default=1200)
parser.add_argument("--learning_rate",      dest='learning_rate',       nargs='?', type=float, help='learning rate for adam optimizer', default=1.0)
parser.add_argument("--print_iter",         dest='print_iter',          nargs='?', type=int, help='print loss per iterations', default=400)
# Note the result might not be smooth enough since not applying smooth for temp result
parser.add_argument("--save_iter",          dest='save_iter',           nargs='?', type=int, help='save temporary result per iterations', default=100)
parser.add_argument("--lbfgs",              dest='lbfgs',               nargs='?', help="True=lbfgs, False=Adam", default=False)

# Weight Options
parser.add_argument("--content_weight",     dest='content_weight',      nargs='?', type=float, help="weight of content loss", default=5e0)
parser.add_argument("--style_weight",       dest='style_weight',        nargs='?', type=float, help="weight of style loss", default=1e2)
parser.add_argument("--tv_weight",          dest='tv_weight',           nargs='?', type=float, help="weight of total variational loss", default=1e-3)
parser.add_argument("--affine_weight",      dest='affine_weight',       nargs='?', type=float, help="weight of affine loss", default=1e4)

# Style Options
parser.add_argument("--apply_smooth",       dest='apply_smooth',        nargs='?', help="if apply local affine smooth", default=True)

# Smoothing Argument
parser.add_argument("--f_radius",           dest='f_radius',            nargs='?', type=int, help="smooth argument", default=15)
parser.add_argument("--f_edge",             dest='f_edge',              nargs='?', type=float, help="smooth argument", default=1e-1)

args = parser.parse_args()

class Scenery():
  
  def __init__ (self):
      print("init")
      #self.config = tf.ConfigProto()
      
     

  def rgb2bgr(self, rgb, vgg_mean=True):
    if vgg_mean:
        return rgb[:, :, ::-1] - VGG_MEAN
    else:
        return rgb[:, :, ::-1]

  def bgr2rgb(self, bgr, vgg_mean=False):
    if vgg_mean:
        return bgr[:, :, ::-1] + VGG_MEAN
    else:
        return bgr[:, :, ::-1]

  def load_seg(self, content_seg_path, style_seg_path, content_shape, style_shape):
    color_codes = ['BLUE', 'GREEN', 'BLACK', 'WHITE', 'RED', 'YELLOW', 'GREY', 'LIGHT_BLUE', 'PURPLE']
    def _extract_mask(seg, color_str):
        h, w, c = np.shape(seg)
        if color_str == "BLUE":
            mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
            mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
            mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
        elif color_str == "GREEN":
            mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
            mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
            mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
        elif color_str == "BLACK":
            mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
            mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
            mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
        elif color_str == "WHITE":
            mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
            mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
            mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
        elif color_str == "RED":
            mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
            mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
            mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
        elif color_str == "YELLOW":
            mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
            mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
            mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
        elif color_str == "GREY":
            mask_r = np.multiply((seg[:, :, 0] > 0.4).astype(np.uint8),
                                 (seg[:, :, 0] < 0.6).astype(np.uint8))
            mask_g = np.multiply((seg[:, :, 1] > 0.4).astype(np.uint8),
                                 (seg[:, :, 1] < 0.6).astype(np.uint8))
            mask_b = np.multiply((seg[:, :, 2] > 0.4).astype(np.uint8),
                                 (seg[:, :, 2] < 0.6).astype(np.uint8))
        elif color_str == "LIGHT_BLUE":
            mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
            mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
            mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
        elif color_str == "PURPLE":
            mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
            mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
            mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
        return np.multiply(np.multiply(mask_r, mask_g), mask_b).astype(np.float32)

    # PIL resize has different order of np.shape
    content_seg = np.array(Image.open(content_seg_path).convert("RGB").resize(content_shape, resample=Image.BILINEAR), dtype=np.float32) / 255.0
    style_seg = np.array(Image.open(style_seg_path).convert("RGB").resize(style_shape, resample=Image.BILINEAR), dtype=np.float32) / 255.0

    color_content_masks = []
    color_style_masks = []
    for i in xrange(len(color_codes)):
        color_content_masks.append(tf.expand_dims(tf.expand_dims(tf.constant(_extract_mask(content_seg, color_codes[i])), 0), -1))
        color_style_masks.append(tf.expand_dims(tf.expand_dims(tf.constant(_extract_mask(style_seg, color_codes[i])), 0), -1))

    return color_content_masks, color_style_masks

  def gram_matrix(self, activations):
    height = tf.shape(activations)[1]
    width = tf.shape(activations)[2]
    num_channels = tf.shape(activations)[3]
    gram_matrix = tf.transpose(activations, [0, 3, 1, 2])
    gram_matrix = tf.reshape(gram_matrix, [num_channels, width * height])
    gram_matrix = tf.matmul(gram_matrix, gram_matrix, transpose_b=True)
    return gram_matrix

  def content_loss(self, const_layer, var_layer, weight):
    return tf.reduce_mean(tf.compat.v1.squared_difference(const_layer, var_layer)) * weight

  def style_loss(self, CNN_structure, const_layers, var_layers, content_segs, style_segs, weight):
    loss_styles = []
    layer_count = float(len(const_layers))
    layer_index = 0

    _, content_seg_height, content_seg_width, _ = content_segs[0].get_shape().as_list()
    _, style_seg_height, style_seg_width, _ = style_segs[0].get_shape().as_list()
    for layer_name in CNN_structure:
      layer_name = layer_name[layer_name.find("/") + 1:]

      # downsampling segmentation
      if "pool" in layer_name:
        content_seg_width, content_seg_height = int(math.ceil(content_seg_width / 2)), int(math.ceil(content_seg_height / 2))
        style_seg_width, style_seg_height = int(math.ceil(style_seg_width / 2)), int(math.ceil(style_seg_height / 2))

        for i in xrange(len(content_segs)):
            content_segs[i] = tf.compat.v1.image.resize_bilinear(content_segs[i], tf.constant((content_seg_height, content_seg_width)))
            style_segs[i] = tf.compat.v1.image.resize_bilinear(style_segs[i], tf.constant((style_seg_height, style_seg_width)))

      elif "conv" in layer_name:
        for i in xrange(len(content_segs)):
            # have some differences on border with torch
            content_segs[i] = tf.nn.avg_pool(tf.pad(content_segs[i], [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), \
            ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID')
            style_segs[i] = tf.nn.avg_pool(tf.pad(style_segs[i], [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), \
            ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID')

      if layer_name == var_layers[layer_index].name[var_layers[layer_index].name.find("/") + 1:]:
        print("Setting up style layer: <{}>".format(layer_name))
        const_layer = const_layers[layer_index]
        var_layer = var_layers[layer_index]

        layer_index = layer_index + 1

        layer_style_loss = 0.0
        for content_seg, style_seg in zip(content_segs, style_segs):
            gram_matrix_const = self.gram_matrix(tf.multiply(const_layer, style_seg))
            style_mask_mean   = tf.reduce_mean(style_seg)
            gram_matrix_const = tf.cond(tf.greater(style_mask_mean, 0.),
                                    lambda: gram_matrix_const / (tf.compat.v1.to_float(tf.size(const_layer)) * style_mask_mean),
                                    lambda: gram_matrix_const
                                )

            gram_matrix_var   = self.gram_matrix(tf.multiply(var_layer, content_seg))
            content_mask_mean = tf.reduce_mean(content_seg)
            gram_matrix_var   = tf.cond(tf.greater(content_mask_mean, 0.),
                                    lambda: gram_matrix_var / (tf.compat.v1.to_float(tf.size(var_layer)) * content_mask_mean),
                                    lambda: gram_matrix_var
                                )

            diff_style_sum    = tf.reduce_mean(tf.compat.v1.squared_difference(gram_matrix_const, gram_matrix_var)) * content_mask_mean

            layer_style_loss += diff_style_sum

        loss_styles.append(layer_style_loss * weight)
    return loss_styles

  def total_variation_loss(self, output, weight):
    shape = output.get_shape()
    tv_loss = tf.reduce_sum((output[:, :-1, :-1, :] - output[:, :-1, 1:, :]) * (output[:, :-1, :-1, :] - output[:, :-1, 1:, :]) + \
              (output[:, :-1, :-1, :] - output[:, 1:, :-1, :]) * (output[:, :-1, :-1, :] - output[:, 1:, :-1, :])) / 2.0
    return tv_loss * weight

  def affine_loss(self, output, M, weight):
    loss_affine = 0.0
    output_t = output / 255.
    for Vc in tf.unstack(output_t, axis=-1):
        Vc_ravel = tf.reshape(tf.transpose(Vc), [-1])
        loss_affine += tf.matmul(tf.expand_dims(Vc_ravel, 0), tf.compat.v1.sparse_tensor_dense_matmul(M, tf.expand_dims(Vc_ravel, -1)))

    return loss_affine * weight

  def save_result(self, img_, str_):
    result = Image.fromarray(np.uint8(np.clip(img_, 0, 255.0)))
    result.save(str_)

  
  def print_loss(self, args, loss_content, loss_styles_list, loss_tv, loss_affine, overall_loss, output_image):
    #global iter_count, min_loss, best_image
    if self.iter_count % args.print_iter == 0:
        print('Iteration {} / {}\n\tContent loss: {}'.format(self.iter_count, args.max_iter, loss_content))
        for j, style_loss in enumerate(loss_styles_list):
            print('\tStyle {} loss: {}'.format(j + 1, style_loss))
        print('\tTV loss: {}'.format(loss_tv))
        print('\tAffine loss: {}'.format(loss_affine))
        print('\tTotal loss: {}'.format(overall_loss - loss_affine))

    if overall_loss < self.min_loss:
        min_loss, best_image = overall_loss, output_image

    if self.iter_count % args.save_iter == 0 and self.iter_count != 0:
        self.save_result(best_image[:, :, ::-1], os.path.join(args.serial, 'out_iter_{}.png'.format(self.iter_count)))

    self.iter_count += 1

  def stylize(self, args, Matting):
    tf.compat.v1.disable_eager_execution()
    self.config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
      
    self.config.gpu_options.allow_growth = True
    
    self.iter_count = 0
    self.min_loss, self.best_image = float("inf"), None
    self.sess = tf.compat.v1.Session(config=self.config)
   
    # prepare input images
    #tt = Image.open(args.content_image_path).convert("RGB")
    tt = cv2.imread(args.content_image_path)
    tt = cv2.resize(tt, dsize=(512, 512), interpolation=cv2.INTER_AREA)
    tt = cv2.cvtColor(tt,cv2.COLOR_RGB2BGR)
    #tt = tt.resize((512, 512))
    #cv2.imwrite('tt.png',tt)
    content_image = np.array(tt, dtype=np.float32)
    content_width, content_height = content_image.shape[1], content_image.shape[0]
   
    if Matting:
        M = tf.compat.v1.to_float(getLaplacian(content_image / 255.))
    
    
    content_image = self.rgb2bgr(content_image)
    content_image = content_image.reshape((1, content_height, content_width, 3)).astype(np.float32)
    
    #ss = Image.open(args.style_image_path).convert("RGB")
    #ss = ss.resize((512, 512))
    #ss.save('ss.png')
    
    ss = cv2.imread(args.style_image_path)
    ss = cv2.resize(ss, dsize=(512, 512), interpolation=cv2.INTER_AREA)
    #ss = cv2.cvtColor(ss,cv2.COLOR_RGB2BGR)
    #cv2.imwrite('ss.png',ss)


    style_image = self.rgb2bgr(np.array(ss, dtype=np.float32))
    style_width, style_height = style_image.shape[1], style_image.shape[0]
    style_image = style_image.reshape((1, style_height, style_width, 3)).astype(np.float32)

    content_masks, style_masks = self.load_seg(args.content_seg_path, args.style_seg_path, [content_width, content_height], [style_width, style_height])
    if not args.init_image_path:
        if Matting:
            print("<WARNING>: Apply Matting with random init")
        init_image = np.random.randn(1, content_height, content_width, 3).astype(np.float32) * 0.0001
    else:
        init_image = np.expand_dims(self.rgb2bgr(np.array(Image.open(args.init_image_path).convert("RGB"), dtype=np.float32)).astype(np.float32), 0)

    mean_pixel = tf.constant(VGG_MEAN)
    input_image = tf.Variable(init_image)
    with tf.name_scope("constant"):
        vgg_const = Vgg19()
        vgg_const.build(tf.constant(content_image), clear_data=False)
        content_fv = self.sess.run(vgg_const.conv4_2)
        content_layer_const = tf.constant(content_fv)
    
        vgg_const.build(tf.constant(style_image))
        style_layers_const = [vgg_const.conv1_1, vgg_const.conv2_1, vgg_const.conv3_1, vgg_const.conv4_1, vgg_const.conv5_1]
        style_fvs = self.sess.run(style_layers_const)
        style_layers_const = [tf.constant(fv) for fv in style_fvs]

    with tf.name_scope("variable"):
        vgg_var = Vgg19()
        vgg_var.build(input_image)

    # which layers we want to use?
    style_layers_var = [vgg_var.conv1_1, vgg_var.conv2_1, vgg_var.conv3_1, vgg_var.conv4_1, vgg_var.conv5_1]
    content_layer_var = vgg_var.conv4_2
    # The whole CNN structure to downsample mask
    layer_structure_all = [layer.name for layer in vgg_var.get_all_layers()]

    # Content Loss
    loss_content = self.content_loss(content_layer_const, content_layer_var, float(args.content_weight))

    # Style Loss
    loss_styles_list = self.style_loss(layer_structure_all, style_layers_const, style_layers_var, content_masks, style_masks, float(args.style_weight))
    loss_style = 0.0
    for loss in loss_styles_list:
        loss_style += loss

    input_image_plus = tf.squeeze(input_image + mean_pixel, [0])

    # Affine Loss
    if Matting:
        loss_affine = self.affine_loss(input_image_plus, M, args.affine_weight)
    else:
        loss_affine = tf.constant(0.00001)  # junk value

    # Total Variational Loss
    loss_tv = self.total_variation_loss(input_image, float(args.tv_weight))

    if args.lbfgs:
        if not Matting:
            overall_loss = loss_content + loss_tv + loss_style
        else:
            overall_loss = loss_content + loss_style + loss_tv + loss_affine

        optimizer = tf.compat.v1.contrib.opt.ScipyOptimizerInterface(overall_loss, method='L-BFGS-B', options={'maxiter': args.max_iter, 'disp': 0})
        self.sess.run(tf.compat.v1.global_variables_initializer())
        print_loss_partial = partial(self.print_loss, args)
        optimizer.minimize(self.sess, fetches=[loss_content, loss_styles_list, loss_tv, loss_affine, overall_loss, input_image_plus], loss_callback=print_loss_partial)

        global min_loss, best_image, iter_count
        best_result = copy.deepcopy(self.best_image)
        min_loss, best_image = float("inf"), None
        return best_result
    else:
        VGGNetLoss = loss_content + loss_tv + loss_style
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)
        VGG_grads = optimizer.compute_gradients(VGGNetLoss, [input_image])

        if Matting:
            b, g, r = tf.unstack(input_image_plus / 255., axis=-1)
            b_gradient = tf.transpose(tf.reshape(2 * tf.compat.v1.sparse_tensor_dense_matmul(M, tf.expand_dims(tf.reshape(tf.transpose(b), [-1]), -1)), [content_width, content_height]))
            g_gradient = tf.transpose(tf.reshape(2 * tf.compat.v1.sparse_tensor_dense_matmul(M, tf.expand_dims(tf.reshape(tf.transpose(g), [-1]), -1)), [content_width, content_height]))
            r_gradient = tf.transpose(tf.reshape(2 * tf.compat.v1.sparse_tensor_dense_matmul(M, tf.expand_dims(tf.reshape(tf.transpose(r), [-1]), -1)), [content_width, content_height]))

            Matting_grad = tf.expand_dims(tf.stack([b_gradient, g_gradient, r_gradient], axis=-1), 0) / 255. * args.affine_weight
            VGGMatting_grad = [(VGG_grad[0] + Matting_grad, VGG_grad[1]) for VGG_grad in VGG_grads]

            train_op = optimizer.apply_gradients(VGGMatting_grad)
        else:
            train_op = optimizer.apply_gradients(VGG_grads)

        self.sess.run(tf.compat.v1.global_variables_initializer())
        min_loss, best_image = float("inf"), None
        for i in xrange(1, args.max_iter):
            _, loss_content_, loss_styles_list_, loss_tv_, loss_affine_, overall_loss_, output_image_ = self.sess.run([
                train_op, loss_content, loss_styles_list, loss_tv, loss_affine, VGGNetLoss, input_image_plus
            ])
            if i % args.print_iter == 0:
                print('Iteration {} / {}\n\tContent loss: {}'.format(i, args.max_iter, loss_content_))
                for j, style_loss_ in enumerate(loss_styles_list_):
                    print('\tStyle {} loss: {}'.format(j + 1, style_loss_))
                print('\tTV loss: {}'.format(loss_tv_))
                if Matting:
                    print('\tAffine loss: {}'.format(loss_affine_))
                print('\tTotal loss: {}'.format(overall_loss_ - loss_tv_))

            if overall_loss_ < min_loss:
                min_loss, best_image = overall_loss_, output_image_

            if i % args.save_iter == 0 and i != 0:
                self.save_result(best_image[:, :, ::-1], os.path.join(args.serial, './temp_images/temp_image.png'.format(i)))

        return best_image



  def scenery_excute(self, img_paths, dest_filename=None, width=512, height=512, out_filename='result.png'):
    
    args.content_image_path = img_paths[0]
    args.style_image_path = img_paths[1]
    args.content_seg_path = './data/mask.png'
    args.style_seg_path = './data/mask.png'
     
    #args.max_iter = 2 * args.max_iter
    tmp_image_bgr = self.stylize(args, True)
    ##############################################################################################################
    #result = Image.fromarray(np.uint8(np.clip(tmp_image_bgr[:, :, ::-1], 0, 255.0)))
    #args.init_image_path = os.path.join(args.serial, "./temp_images/temp_image.png")
    #result.save(args.init_image_path)
    #best_image_bgr = self.stylize(args, True)

    ##############################################################################################################
    args.init_image_path = ""
   
    """
    # Pycuda runtime incompatible with Tensorflow
    
    content_input = np.array(Image.open(args.content_image_path).convert("RGB"), dtype=np.float32)
    # RGB to BGR
    content_input = content_input[:, :, ::-1]
    # H * W * C to C * H * W
    content_input = content_input.transpose((2, 0, 1))
    input_ = np.ascontiguousarray(content_input, dtype=np.float32) / 255.

    _, H, W = np.shape(input_)

    output_ = np.ascontiguousarray(best_image_bgr.transpose((2, 0, 1)), dtype=np.float32) / 255.
    best_ = smooth_local_affine(output_, input_, 1e-7, 3, H, W, args.f_radius, args.f_edge).transpose(1, 2, 0)
    result = Image.fromarray(np.uint8(np.clip(output_ * 255., 0, 255.)))
    result.save(args.output_image)

    # use numpy to convert the pil_image into a numpy array
    numpy_image=np.array(result)  
    # convert to a openCV2 image and convert from RGB to BGR format
    result=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('result.jpg', result)
    """
    return 0, cv2.imread('./temp_images/temp_image.png')
