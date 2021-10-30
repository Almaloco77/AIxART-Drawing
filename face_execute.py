from docopt import docopt
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from face import locator
from face import aligner
from face import warper
from face import blender

def list_imgpaths(imgfolder):
  for fname in os.listdir(imgfolder):
    if (fname.lower().endswith('.jpg') or
       fname.lower().endswith('.png') or
       fname.lower().endswith('.jpeg')):
      yield os.path.join(imgfolder, fname)

def sharpen(img):
  blured = cv2.GaussianBlur(img, (0, 0), 2.5)
  return cv2.addWeighted(img, 1.4, blured, -0.4, 0)

def load_image_points(img, size):
  points = locator.face_points(img)

  if len(points) == 0:
    try:
      height, width, channel = img.shape
      matrix = cv2.getRotationMatrix2D((width/2, height/2), -90, 1)
      img = cv2.warpAffine(img, matrix, (width, height))
      points = locator.face_points(img)

      return aligner.resize_align(img, points, size)
    except:
      return None, None
  else:
    return aligner.resize_align(img, points, size)


def face_excute(img_paths, dest_filename=None, width=500, height=600, background='average',
             blur_edges=False, out_filename='result.png', plot=False):
  size = (height, width)
  images = []
  point_set = []
  for path in img_paths:
    t_img = cv2.imread(path)
    h_, w_, c_ = t_img.shape

    #if w_ > 1080:
    #  t_img = cv2.resize(t_img, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    t_img = cv2.cvtColor(np.array(t_img), cv2.COLOR_BGR2RGB)
    print(path)
    img, points = load_image_points(t_img, size)
    if img is not None:
      images.append(img)
      point_set.append(points)

  if len(images) < 2:
    print('No face or detected face points')
    return 2, None

  if dest_filename is not None:
    dest_img, dest_points = load_image_points(dest_filename, size)
    if dest_img is None or dest_points is None:
      print('No face or detected face points in dest img: ' + dest_filename)
      return 2, None
  else:
    dest_img = np.zeros(images[0].shape, np.uint8)
    dest_points = locator.average_points(point_set)

  num_images = len(images)
  result_images = np.zeros(images[0].shape, np.float32)
  for i in range(num_images):
    result_images += warper.warp_image(images[i], point_set[i],
                                       dest_points, size, np.float32)

  result_image = np.uint8(result_images / num_images)
  face_indexes = np.nonzero(result_image)
  dest_img[face_indexes] = result_image[face_indexes]

  mask = blender.mask_from_points(size, dest_points)
  if blur_edges:
    blur_radius = 5
    mask = cv2.blur(mask, (blur_radius, blur_radius))

  if background in ('transparent', 'average'):
    dest_img = np.dstack((dest_img, mask))

    if background == 'average':
      average_background = locator.average_points(images)
      dest_img = blender.overlay_image(dest_img, mask, average_background)

      dest_img= dest_img.astype(np.uint8)
      center = (int(width / 2), int(height / 2))

      
      result = cv2.seamlessClone(dest_img, images[0], mask, center, cv2.NORMAL_CLONE)
      #result1 = cv2.seamlessClone(dest_img, images[1], mask, center, cv2.NORMAL_CLONE)
      cv2.imwrite('result.jpg', result)


  return 0, result

