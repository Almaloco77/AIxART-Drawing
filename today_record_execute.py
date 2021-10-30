from __future__ import print_function
import sys
sys.path.insert(0, 'today_record/src')
sys.path.insert(0, 'today_record')
import transform, numpy as np, vgg, pdb, os
import tensorflow as tf
from today_record.src.utils import save_img, get_img, exists, list_files

import imageio
import random
import glob

import os
import cv2 as cv
import numpy as np
from PIL import Image

from facemorpher import locator_y
from facemorpher import aligner
from facemorpher import warper



position = np.array([
                    [[0, 0, 720, 960], [0, 0, 400, 260], [30, 240, 340, 370], [10, 300, 220, 630], [400, 90, 520, 410], 
                    [490, 310, 600, 450], [395, 495, 610, 715], [70, 705, 720, 960], [315, 319, 425, 465], [210, 330, 510, 485]
                    ], # 1 - 10 image
                    [[0, 0, 720, 960], [265, 350, 430, 565], [270, 580, 690, 850], [25, 15, 430, 240], [0, 0, 350, 270], 
                    [570, 310, 720, 530], [390, 75, 720, 350], [135, 235, 355, 380], [45, 300, 135, 740], [135, 475, 360, 825]
                    ], #2 - 10 image
                    [[0, 0, 720, 960], [265, 0, 720, 960], [365, 0, 720, 960], [430, 0, 720, 960],[510, 0, 720, 960]
                    ], #3 - 5 image
                    [[0, 0, 720, 960], [0, 290, 720, 960], [0, 390, 720, 960], [0, 560, 720, 960], [0, 680, 720, 960]
                    ], #4 - 5 image
                    [[0, 0, 720, 960], [0, 240, 360, 480], [0, 480, 360, 720], [0, 720, 360, 960],
                    [360, 0, 720, 240], [360, 240, 720, 480], [360, 480, 720, 720], [360, 720, 720, 960]
                    ], #5 - 8 image
                    [[0, 0, 720, 960], [0, 285, 360, 560], [0, 560, 360, 700], [0, 700, 360, 960], 
                    [360, 0, 720, 200], [360, 200, 720, 370], [360, 370, 720, 595], [360, 595, 720, 645], [360, 645, 720, 750], [360, 750, 720, 960]
                    ], #6 - 10 image
                    [[0, 0, 720, 960], [0, 340, 360, 540], [0, 540, 360, 715], [0, 715, 360, 960], 
                    [360, 0, 720, 285], [360, 285, 720, 495], [360, 495, 720, 645], [360, 645, 720, 750], [360, 750, 720, 960]
                    ], #7 - 9 image
                    ]
                    )



def alphablend(mode, result_img, temp_img, img, width, position, max_val):
    alpha_width_rate = 2

    if position[mode] == max_val:
        return result_img
    
    middle = position[mode]
    alpha_width = width * alpha_width_rate // 100 
    start = middle - alpha_width // 2
    step = 100/alpha_width

    for j in range(alpha_width+1 ):
        alpha = (100 - step * j) / 100  
        beta = 1 - alpha

        if mode == 0:
            result_img[position[1]:position[3], start+j] = temp_img[position[1]:position[3], start+j] * \
                                    alpha + img[position[1]:position[3], start+j] * beta
        elif mode == 1:
            result_img[start+j,position[0]:position[2]] = temp_img[start+j,position[0]:position[2]] * \
                                    alpha + img[start+j,position[0]:position[2]] * beta       
        elif mode == 2:
            result_img[position[1]:position[3], start+j] = temp_img[position[1]:position[3], start+j] * \
                                    beta + img[position[1]:position[3], start+j] * alpha
        elif mode == 3:
            result_img[start+j,position[0]:position[2]] = temp_img[start+j,position[0]:position[2]] * \
                                    beta + img[start+j,position[0]:position[2]] * alpha

    return result_img
  
def position_overlap(i, k, img, result_img):
    h, w, c = img.shape
    temp_img = result_img.copy()

    mask = img[position[k][i][1]:position[k][i][3], position[k][i][0]:position[k][i][2]]
    result_img[position[k][i][1]:position[k][i][3], position[k][i][0]:position[k][i][2]] = mask
    # if i != 0:
    #     height, width = img.shape[:2]
    #     result_img = alphablend(0, result_img, temp_img, img, w, position[k][i], 0)
    #     result_img = alphablend(1, result_img, temp_img, img, w, position[k][i], 0)
    #     result_img = alphablend(2, result_img, temp_img, img, w, position[k][i], 720)
    #     result_img = alphablend(3, result_img, temp_img, img, w, position[k][i], 960)

def sharpen(img):
  blured = cv.GaussianBlur(img, (0, 0), 2.5)
  return cv.addWeighted(img, 1.4, blured, -0.4, 0)

def load_image_points(img, size):
  points = locator_y.face_points(img)

  if len(points) == 0:
    print('No face')
    return None, None
  else:
    return aligner.resize_align(img, points, size)

def face_excute(imgs, dest_filename=None, width=500, height=600, background='average',
             blur_edges=False, out_filename='result.png', plot=False):
  size = (height, width)

  images = []
  point_set = []
  for path in imgs:

    img, points = load_image_points(path, size)
    if img is not None:
      images.append(img)
      point_set.append(points)
    else:
        return 1, None
  
  return 0, images


class TodayRecord():
    def __init__(self):
        with tf.device('/gpu:0'):
            tf.compat.v1.disable_eager_execution()
            soft_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
            soft_config.gpu_options.allow_growth = True
            self.sess = tf.compat.v1.Session(config=soft_config)
            self.batch_shape  =  (1, 600, 500, 3)
            self.img_placeholder = tf.compat.v1.placeholder(tf.float32, shape=self.batch_shape, name='img_placeholder')

            self.preds = transform.net(self.img_placeholder)
            self.saver = tf.compat.v1.train.Saver()
            print('='*50)
     
        

    def ffwd_to_img(self, in_path, checkpoint_dir, device='/gpu:0'):
        g = tf.Graph()
        batch_size = min(1, 4)

        with g.as_default(), g.device(device):
            if os.path.isdir(checkpoint_dir):
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                else:
                    raise Exception("No checkpoint found...")
            else:
                self.saver.restore(self.sess, checkpoint_dir)
            num_iters = int(1/batch_size)
            for i in range(num_iters):
                pos = i * batch_size
                X = [np.array(Image.fromarray(in_path))]
                _preds = self.sess.run(self.preds, feed_dict={self.img_placeholder:X})
                return _preds[0]




    def today_record_execute(self, img_path):
        # 이미지 배열 선택
        shape_num = 0 #random.randint(0, 6) # 8개 :7  //  현재, 7개:6
        #shape_num = 0
        num_images = len(position[shape_num])

        """
        ori_ran =[]
        ran_num = random.randint(0,num_images)

        for path in range(0, int(num_images * 0.5)):
            while ran_num in ori_ran:
                ran_num = random.randint(0,num_images)
            ori_ran.append(ran_num)
        """

        # Check point 선택
        checkpoint_file = ["./today_record/weight/la_muse.ckpt", "./today_record/weight/rain_princess.ckpt", "./today_record/weight/scream.ckpt", "./today_record/weight/udnie.ckpt", "./today_record/weight/wave.ckpt", "./today_record/weight/wreck.ckpt", "./today_record/weight/bull", "./today_record/weight/laundry", "./today_record/weight/vangogh", "./today_record/weight/woman"]
        random.shuffle(checkpoint_file)
        checkpoint_file = checkpoint_file[0:num_images]
        #checkpoint_file = checkpoint_file[0:new_num_images]
        
        # 이미지 로드
        imgs = []
        
        input_img = cv.imread(img_path)
        #input_img = cv.imread("./test/0.png")

        input_img = cv.cvtColor(input_img,cv.COLOR_BGR2RGB)
        imgs.append(input_img)

        pre_img_paths = glob.glob('./today_record/input/*.png')
        #pre_img_paths = glob.glob('./test/input/*.png')
        random.shuffle(pre_img_paths)
        pre_img_paths = pre_img_paths[0:num_images-1]
        #pre_img_paths = pre_img_paths[0:new_num_images-1]

        for path in pre_img_paths:
            imgs.append(cv.imread(path))
        random.shuffle(imgs)

        # 얼굴 검출 수행
        code, aligned_img = face_excute(imgs)
        
        if code != 0:
            print('ERROR 2: face detection error')
            return code, None, None
        
        if len(imgs) != len(aligned_img):
            print('ERROR 3: face detection error')
            return 2, None, None
        
        
        # Style 변환
        """
        style_img = []

        for i in range(len(checkpoint_file)):
            visit = True
            for j in range(len(ori_ran)):
                if ori_ran[j] == i : 
                    cv_image = np.array(aligned_img[i])     
                    cv.imwrite('./temp_images/test.png', cv_image)
                    style_img.append(cv.imread('./temp_images/test.png'))
                    visit = False
                    
            if visit:
                temp_img = self.ffwd_to_img(aligned_img[i], checkpoint_file[i], device='/gpu:0')
                cv.imwrite('./temp_images/test.png', temp_img)
                style_img.append(cv.imread('./temp_images/test.png'))
        if os.path.exists('./temp_images/test.png'):
            os.remove('./temp_images/test.png') 
       
        result_img = 255 * np.ones((960, 720, 3), np.uint8)
        """
        # Style 변환
        style_img = []
        for i in range(len(checkpoint_file)):
            temp_img = self.ffwd_to_img(aligned_img[i], checkpoint_file[i], device='/gpu:0')
            cv.imwrite('./temp_images/test.png', temp_img)
            style_img.append(cv.imread('./temp_images/test.png'))
        if os.path.exists('./temp_images/test.png'):
            os.remove('./temp_images/test.png') 
        
        result_img = 255 * np.ones((960, 720, 3), np.uint8)

        #print(len(style_img))

        # 변환된 얼굴 합성
        for idx, val in enumerate(style_img):
            val = cv.resize(val, dsize=(720, 960), interpolation=cv.INTER_AREA)
            #cv.imwrite('./{}.png'.format(idx), val)
            position_overlap(idx, shape_num, val, result_img)

        cv.imwrite('result.jpg', result_img)

        return 0, result_img, input_img


