#!/usr/bin/python3

# Copyright 2017 Andres Milioto, Cyrill Stachniss. All Rights Reserved.
#
#  This file is part of Bonnet.
#
#  Bonnet is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  Bonnet is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with Bonnet. If not, see <http://www.gnu.org/licenses/>.

'''
  Given an RGB plant image, the functions in this file
  calculate the extra features for addition to CNN.
'''

import cv2
import numpy as np


def contrast_stretch(img):
  """
  Performs a simple contrast stretch of the given image, in order to remove
  extreme outliers.
  """
  in_min = np.percentile(img, 0.05)
  in_max = np.percentile(img, 99.95)

  out_min = 0.0
  out_max = 255.0

  out = img - in_min
  out *= out_max / (in_max - in_min)

  out[out < out_min] = 0.0
  out[out > out_max] = 255.0

  return out


def contrast_stretch_const(img, in_min, in_max):
  """
  Performs a simple contrast stretch of the given image, in order to remove
  extreme outliers.
  """
  out_min = 0.0
  out_max = 255.0

  out = img - in_min
  out *= out_max / (in_max - in_min)

  out[out < out_min] = 0.0
  out[out > out_max] = 255.0

  return out


def thresh(img, conservative=0, min_blob_size=50):
  '''
    Get threshold to make mask using the otsus method, and apply a correction
    passed in conservative (-100;100) as a percentage of th.
  '''

  # blur and get level using otsus
  blur = cv2.GaussianBlur(img, (13, 13), 0)
  level, _ = cv2.threshold(
      blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

  # print("Otsus Level: ",level)

  # change with conservative
  level += conservative / 100.0 * level

  # check boundaries
  level = 255 if level > 255 else level
  level = 0 if level < 0 else level

  # mask image
  _, mask = cv2.threshold(blur, level, 255, cv2.THRESH_BINARY)

  # morph operators
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

  # remove small connected blobs
  # find connected components
  n_components, output, stats, centroids = cv2.connectedComponentsWithStats(
      mask, connectivity=8)
  # remove background class
  sizes = stats[1:, -1]
  n_components = n_components - 1

  # remove blobs
  mask_clean = np.zeros((output.shape))
  # for every component in the image, keep it only if it's above min_blob_size
  for i in range(0, n_components):
    if sizes[i] >= min_blob_size:
      mask_clean[output == i + 1] = 255

  return mask_clean


def exgreen(img):
  '''
    Returns the excess green of the image as:
      exgreen = 2 * G - R - B
  '''

  # get channels
  B, G, R = cv2.split(img)

  # normalize
  B_ = B.astype(float) / np.median(B.astype(float))
  G_ = G.astype(float) / np.median(G.astype(float))
  R_ = R.astype(float) / np.median(R.astype(float))

  E = B_ + G_ + R_ + 0.001
  b = B_ / E
  g = G_ / E
  r = R_ / E

  # calculate exgreen
  exgr = 2.8 * g - r - b

  # expand contrast
  exgr = contrast_stretch(exgr)

  # convert to saveable image
  exgr = exgr.astype(np.uint8)

  return exgr


def cive(img):
  '''
    Returns the inverse color index of vegetation extraction of the image as:
      cive = 0.881 * g - 0.441 * r - 0.385 * b - 18.78745
  '''

  # get channels
  B, G, R = cv2.split(img)

  # normalize
  B_ = B.astype(float) / np.median(B.astype(float))
  G_ = G.astype(float) / np.median(G.astype(float))
  R_ = R.astype(float) / np.median(R.astype(float))

  E = B_ + G_ + R_ + 0.001
  b = B_ / E
  g = G_ / E
  r = R_ / E

  # calculate cive
  c = 0.881 * g - 0.441 * r - 0.385 * b - 18.78745

  # expand contrast
  c = contrast_stretch(c)

  # convert to saveable image
  c = c.astype(np.uint8)

  return c


def exred(img):
  '''
    Returns the excess green (inverted, to comply with other masks) of the image as:
      exred = 1.4 * R - G
  '''

  # get channels
  B, G, R = cv2.split(img)

  # normalize
  B_ = B.astype(float) / np.median(B.astype(float))
  G_ = G.astype(float) / np.median(G.astype(float))
  R_ = R.astype(float) / np.median(R.astype(float))

  E = B_ + G_ + R_ + 0.001
  b = B_ / E
  g = G_ / E
  r = R_ / E

  # calculate exgreen
  exr = 1.4 * r - g

  # expand contrast
  exr = contrast_stretch(exr)

  # convert to saveable image
  exr = exr.astype(np.uint8)

  return exr


def ndi(img):
  '''
    Get the normalized diference index
  '''
  # get channels
  B, G, R = cv2.split(img)

  # normalize
  B_ = B.astype(float) / np.median(B.astype(float))
  G_ = G.astype(float) / np.median(G.astype(float))
  R_ = R.astype(float) / np.median(R.astype(float))

  E = B_ + G_ + R_ + 0.001
  b = B_ / E
  g = G_ / E
  r = R_ / E

  # calculate ndi
  idx = (g - r) / (g + r)

  # expand contrast
  idx = contrast_stretch(idx)

  # convert to saveable image
  idx = idx.astype(np.uint8)

  return idx


def hsv(img):
  '''
    Convert image to hsv
  '''

  ret = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  return ret


def edges(mask):
  '''
    Get edges with canny detector
  '''
  # blur
  mask = cv2.GaussianBlur(mask, (5, 5), 0)

  edges = cv2.Canny(mask, 100, 200)

  # stretch
  edges = contrast_stretch(edges)

  # cast
  edges = np.uint8(edges)

  return edges


def laplacian(mask):
  '''
    Get 2nd order gradients using the Laplacian
  '''

  # blur
  mask = cv2.GaussianBlur(mask, (5, 5), 0)

  # edges with laplacian
  laplacian = cv2.Laplacian(mask, cv2.CV_64F, 5)

  # stretch
  laplacian = contrast_stretch(laplacian)

  # cast
  laplacian = np.uint8(laplacian)

  return laplacian


def gradients(mask, direction='x'):
  '''
    Get gradients using sobel operator
  '''
  mask = cv2.GaussianBlur(mask, (5, 5), 0)

  if direction == 'x':
    # grad x
    sobel = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=7)
  elif direction == 'y':
    # grad y
    sobel = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=7)
  else:
    print("Invalid gradient direction. Must be x or y")
    quit()

  # sobel = np.absolute(sobel)
  sobel = contrast_stretch(sobel)   # expand contrast
  sobel = np.uint8(sobel)

  return sobel


def watershed(rgb, idx, mask):
  '''
    Get watershed transform from image
  '''

  # kernel definition
  kernel = np.ones((3, 3), np.uint8)

  # sure background area
  sure_bg = cv2.dilate(mask, kernel)
  sure_bg = np.uint8(sure_bg)
  # util.im_gray_plt(sure_bg,"sure back")

  # Finding sure foreground area
  dist_transform = cv2.distanceTransform(np.uint8(mask), cv2.DIST_L2, 3)
  # util.im_gray_plt(dist_transform,"dist transform")
  ret, sure_fg = cv2.threshold(
      dist_transform, 0.5 * dist_transform.max(), 255, 0)

  # Finding unknown region
  sure_fg = np.uint8(sure_fg)
  # util.im_gray_plt(sure_fg,"sure fore")

  unknown = cv2.subtract(sure_bg, sure_fg)
  # util.im_gray_plt(unknown,"unknown")

  # marker labelling
  ret, markers = cv2.connectedComponents(sure_fg)

  # add one to all labels so that sure background is not 0, but 1
  markers = markers + 1

  # mark the region of unknown with zero
  markers[unknown == 255] = 0

  # util.im_gray_plt(np.uint8(markers),"markers")

  # apply watershed
  markers = cv2.watershed(rgb, markers)

  # create limit mask
  mask = np.zeros(mask.shape, np.uint8)
  mask[markers == -1] = 255

  return mask


def mask_multidim(img, mask):
  '''
    mask an image with a mask (0,255)
  '''
  ret = np.array(img)
  if len(img.shape) == 3:
    ret[mask == 0, :] = 0
  elif len(img.shape) == 2:
    ret[mask == 0] = 0
  else:
    # unknown shape
    ret = img

  return ret


def chanelwise_norm(img):
  '''
    Returns the normalized image:
  '''
  ret = np.array(img)

  # expand contrast
  for i in range(img.shape[2]):
    ret[:, :, i] = contrast_stretch(ret[:, :, i].astype(float))

  # convert to saveable image
  ret = ret.astype(np.uint8)

  return ret
