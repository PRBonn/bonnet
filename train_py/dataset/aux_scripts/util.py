# Copyright 2017 Andres Milioto. All Rights Reserved.
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
  Some auxiliary functions to do some opencv stuff
'''

import matplotlib.pyplot as plt
import cv2
import numpy as np


def im_plt(img, title=None):
  """
    Open image and print it on screen
  """
  plt.ion()
  plt.figure()
  plt.imshow(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
  if title is not None:
    plt.title(str(title))


def im_tight_plt(img):
  """
    Open image and print it without borders on screen
  """
  plt.ion()
  fig, ax = plt.subplots()
  fig.subplots_adjust(0, 0, 1, 1)
  ax.axis("off")
  ax.imshow(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))


def im_gray_plt(img, title=None):
  """
    Open grayscale image and print it on screen
  """
  plt.ion()
  plt.figure()
  plt.imshow(img.astype(np.uint8), cmap=plt.get_cmap('gist_gray'))
  if title is not None:
    plt.title(str(title))


def hist_plot(img, title=None):
  """
    Calculate histogram and plot it
  """
  plt.ion()
  plt.figure()
  plt.hist(img.ravel(), 256)
  if title is not None:
    plt.title(str(title))


def im_block():
  """
  Blocks thread until windows are closed
  """
  plt.show(block=True)


def transparency(img, mask):
  alpha = 1
  beta = 0.5
  gamma = 0
  rows, cols, depth = mask.shape
  img = cv2.resize(img, (cols, rows)).astype(np.uint8)
  mask = mask.astype(np.uint8)
  transparent_mask = cv2.addWeighted(img, alpha, mask, beta, gamma, dtype=-1)
  return img, transparent_mask


def prediction_to_color(predicted_mask, label_remap, color_map):
  # get prediction and make it color
  # map to color
  color_mask = np.zeros([predicted_mask.shape[0], predicted_mask.shape[1], 3])
  for key in label_remap:
    color_mask[np.where((predicted_mask == label_remap[key]))] = color_map[key]
  return color_mask
