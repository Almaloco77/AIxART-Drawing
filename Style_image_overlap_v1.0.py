from __future__ import print_function
import sys
sys.path.insert(0, 'src')
import transform, numpy as np, vgg, pdb, os
import scipy.misc
import tensorflow as tf
from utils import save_img, get_img, exists, list_files
from argparse import ArgumentParser
from collections import defaultdict
import time
import json
import subprocess
import numpy
from moviepy.video.io.VideoFileClip import VideoFileClip
import moviepy.video.io.ffmpeg_writer as ffmpeg_writer

import cv2
import random

BATCH_SIZE = 4
DEVICE = '/gpu:0'

position = np.array([
                    [[0, 0, 720, 960], [0, 0, 400, 260], [30, 240, 340, 370], [10, 300, 220, 630], [400, 90, 520, 410], 
                    [490, 310, 600, 450], [395, 495, 610, 715], [70, 705, 720, 960], [315, 319, 425, 465], [210, 330, 510, 485]
                    ]
                    )

def position_overlap(i, k, img, result_img):
    h, w, c = img.shape

    mask = img[position[k][i][1]:position[k][i][3], position[k][i][0]:position[k][i][2]]
   
    result_img[position[k][i][1]:position[k][i][3], position[k][i][0]:position[k][i][2]] = mask
    #cv2.imshow('mask', mask)
    #cv2.imshow('result_img', result_img)
    #cv2.waitKey(0)

def img_load(source_image_folder):

    #result_img = 255 * np.ones((960, 720, 3), np.uint8)
    source_images = os.listdir(source_image_folder)

    #shape_num = random.randint(0, 6) # 8개 :7  //  현재, 7개:6
    #style_num = []
    
    # for ii in range(100):
    #     visit = True
    #     style_num_temp = random.randint(0, 9)
    #     for jj in range(len(style_num)):
    #         if style_num[jj] == style_num_temp:
    #             visit = False
    #             break
    #     if visit : 
    #         style_num.append(style_num_temp)
    #         if len(style_num) == 10:
    #             break 
    
    images_names = []
    if source_images:
        for index, source_image in enumerate(source_images):
            
            images_names.append(os.path.join(source_image_folder, source_image))
    #         img = cv2.imread(image)
    #         img = cv2.resize(img, dsize=(720, 960), interpolation=cv2.INTER_AREA)
    #         position_overlap(index, shape_num, img, result_img)
            

    #         if (shape_num == 2 or shape_num == 3) and index == 4: break
    #         if shape_num == 4 and index == 7: break
    #         if shape_num == 6 and index == 8: break

    # cv2.imwrite('result_img.png', result_img)
    # cv2.waitKey(0)
    return images_names


def ffwd_video(path_in, path_out, checkpoint_dir, device_t='/gpu:0', batch_size=4):
    video_clip = VideoFileClip(path_in, audio=False)
    video_writer = ffmpeg_writer.FFMPEG_VideoWriter(path_out, video_clip.size, video_clip.fps, codec="libx264",
                                                    preset="medium", bitrate="2000k",
                                                    audiofile=path_in, threads=None,
                                                    ffmpeg_params=None)

    g = tf.Graph()
    soft_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with g.as_default(), g.device(device_t), \
            tf.compat.v1.Session(config=soft_config) as sess:
        batch_shape = (batch_size, video_clip.size[1], video_clip.size[0], 3)
        img_placeholder = tf.compat.v1.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')

        preds = transform.net(img_placeholder)
        saver = tf.compat.v1.train.Saver()
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, checkpoint_dir)

        X = np.zeros(batch_shape, dtype=np.float32)

        def style_and_write(count):
            for i in range(count, batch_size):
                X[i] = X[count - 1]  # Use last frame to fill X
            _preds = sess.run(preds, feed_dict={img_placeholder: X})
            for i in range(0, count):
                video_writer.write_frame(np.clip(_preds[i], 0, 255).astype(np.uint8))

        frame_count = 0  # The frame count that written to X
        for frame in video_clip.iter_frames():
            X[frame_count] = frame
            frame_count += 1
            if frame_count == batch_size:
                style_and_write(frame_count)
                frame_count = 0

        if frame_count != 0:
            style_and_write(frame_count)

        video_writer.close()


# get img_shape
def ffwd(data_in, paths_out, checkpoint_dir, device_t='/gpu:0', batch_size=4):
    assert len(paths_out) > 0
    is_paths = type(data_in[0]) == str
    if is_paths:
        assert len(data_in) == len(paths_out)
        img_shape = get_img(data_in[0]).shape
    else:
        assert data_in.size[0] == len(paths_out)
        img_shape = X[0].shape

    g = tf.Graph()
    batch_size = min(len(paths_out), batch_size)
    curr_num = 0
    soft_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with g.as_default(), g.device(device_t), \
            tf.compat.v1.Session(config=soft_config) as sess:
        batch_shape = (batch_size,) + img_shape
        img_placeholder = tf.compat.v1.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')

        preds = transform.net(img_placeholder)
        saver = tf.compat.v1.train.Saver()
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, checkpoint_dir)

        num_iters = int(len(paths_out)/batch_size)
        for i in range(num_iters):
            pos = i * batch_size
            curr_batch_out = paths_out[pos:pos+batch_size]
            if is_paths:
                curr_batch_in = data_in[pos:pos+batch_size]
                X = np.zeros(batch_shape, dtype=np.float32)
                for j, path_in in enumerate(curr_batch_in):
                    img = get_img(path_in)
                    assert img.shape == img_shape, \
                        'Images have different dimensions. ' +  \
                        'Resize images or use --allow-different-dimensions.'
                    X[j] = img
            else:
                X = data_in[pos:pos+batch_size]

            _preds = sess.run(preds, feed_dict={img_placeholder:X})
            for j, path_out in enumerate(curr_batch_out):
                save_img(path_out, _preds[j])
                
        remaining_in = data_in[num_iters*batch_size:]
        remaining_out = paths_out[num_iters*batch_size:]
    if len(remaining_in) > 0:
        ffwd(remaining_in, remaining_out, checkpoint_dir, 
            device_t=device_t, batch_size=1)

def ffwd_to_img(in_path, out_path, checkpoint_dir, device='/cpu:0'):

    #print(in_path)
    paths_in, paths_out = [in_path], [out_path]
    ffwd(paths_in, paths_out, checkpoint_dir, batch_size=1, device_t=device)

def ffwd_different_dimensions(in_path, out_path, checkpoint_dir, 
            device_t=DEVICE, batch_size=4):
    in_path_of_shape = defaultdict(list)
    out_path_of_shape = defaultdict(list)
    for i in range(len(in_path)):
        in_image = in_path[i]
        out_image = out_path[i]
        shape = "%dx%dx%d" % get_img(in_image).shape
        in_path_of_shape[shape].append(in_image)
        out_path_of_shape[shape].append(out_image)
    for shape in in_path_of_shape:
        print('Processing images of shape %s' % shape)
        ffwd(in_path_of_shape[shape], out_path_of_shape[shape], 
            checkpoint_dir, device_t, batch_size)

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        dest='checkpoint_dir',
                        help='dir or .ckpt file to load checkpoint from',
                        metavar='CHECKPOINT', required=False)

    parser.add_argument('--in-path', type=str,
                        dest='in_path',help='dir or file to transform',
                        metavar='IN_PATH', required=False)

    help_out = 'destination (dir or file) of transformed file or files'
    parser.add_argument('--out-path', type=str,
                        dest='out_path', help=help_out, metavar='OUT_PATH',
                        required=False)

    parser.add_argument('--device', type=str,
                        dest='device',help='device to perform compute on',
                        metavar='DEVICE', default=DEVICE)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size',help='batch size for feedforwarding',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--allow-different-dimensions', action='store_true',
                        dest='allow_different_dimensions', 
                        help='allow different image dimensions')

    return parser

def check_opts(opts):
    exists(opts.checkpoint_dir, 'Checkpoint not found!')
    exists(opts.in_path, 'In path not found!')
    if os.path.isdir(opts.out_path):
        exists(opts.out_path, 'out dir not found!')
        assert opts.batch_size > 0

def main():
    parser = build_parser()
    opts = parser.parse_args()
    #check_opts(opts)
    images_names = img_load("image_align")

    if not os.path.isdir(opts.in_path):
        out_file  = ["result/0.png", "result/1.png", "result/2.png", "result/3.png", "result/4.png", "result/5.png", "result/6.png", "result/7.png", "result/8.png", "result/9.png"]
        checkpoint_file = ["weight/la_muse.ckpt", "weight/rain_princess.ckpt", "weight/scream.ckpt", "weight/udnie.ckpt", "weight/wave.ckpt", "weight/wreck.ckpt", "weight/bull", "weight/laundry", "weight/vangogh", "weight/woman"]
        
        shape_num = random.randint(0, 6) # 8개 :7  //  현재, 7개:6
        style_num = []
        for ii in range(100):
            visit = True
            style_num_temp = random.randint(0, 9)
            for jj in range(len(style_num)):
                if style_num[jj] == style_num_temp:
                    visit = False
                    break
            if visit : 
                style_num.append(style_num_temp)
                if len(style_num) == 10:
                    break 


        for i in range(len(checkpoint_file)): 
            #ffwd_to_img(opts.in_path, out_file[i], checkpoint_file[i], device=opts.device)
            img = ffwd_to_img(images_names[i], out_file[style_num[i]], checkpoint_file[style_num[i]], device=opts.device)

        
        result_img = 255 * np.ones((960, 720, 3), np.uint8)        
        source_images = os.listdir("result")

        print(len(source_images))
        print(source_images)
        
        if source_images[0] == 'Thumbs.db':
            os.remove(os.path.join("result", 'Thumbs.db'))

        print(len(source_images))
        print(source_images)
        
        if source_images:
            for index, source_image in enumerate(source_images):
                if (source_image.lower().endswith('.jpg') or source_image.lower().endswith('.png') or source_image.lower().endswith('.jpeg')): 
                    image = os.path.join("result", source_image)            
                    img = cv2.imread(image)


                    img = cv2.resize(img, dsize=(720, 960), interpolation=cv2.INTER_AREA)
                    position_overlap(index, shape_num, img, result_img)
                    
                    if (shape_num == 2 or shape_num == 3) and index == 4: break
                    if shape_num == 4 and index == 7: break
                    if shape_num == 6 and index == 8: break

        cv2.imwrite('result_img.png', result_img)
        cv2.waitKey(0)
 

if __name__ == '__main__':
    main()
