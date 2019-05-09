# coding: utf-8

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

""" Abstraction for getting cityscapes dataset and putting it in an abstract
    class that can handle any segmentation problem :)
"""

# import the abstract dataset classes
import dataset.abstract_dataset as abs_data

# opencv stuff for images and numpy for matrix representation
import cv2

# file and folder handling
import os
from os import listdir
from os.path import isfile, join
from colorama import Fore, Back, Style
import shutil


def dir_to_data(directory, label_map, label_remap, new_shape=None, force_remap=False):
  """Get the dataset in the format that we need it
     and give it to dataset generator
  Args:
    directory: Folder where the dataset is stored
    label_map: all the classes we know
    label_remap: remap to classes we need for the crossentropy loss
    new_shape: New shape for the images to use
    force_remap: Delete remap folders, so that they are populated again
  Returns:
    images: images file names
    labels: remaped label filenames
    n_data = amount of data in the dataset
    content_perc = dict of percentage content for each class in the entire set
  """
  print("---------------------------------------------------------------------")
  print("Parsing directory %s" % directory)

  # make lists strings

  # create the content dictionary and other statistics in data
  content_perc = {}
  for key in label_map:
    content_perc[key] = 0.0
  total_pix = 0.0
  n_data = 0

  img_dir = directory + '/img/'
  img_remap_dir = img_dir + '/remap/'
  label_dir = directory + '/lbl/'
  label_remap_dir = label_dir + '/remap/'

  # get the file list in the folder
  images = [f for f in listdir(img_dir)
            if isfile(join(img_dir, f))]
  labels = [f for f in listdir(label_dir)
            if isfile(join(label_dir, f))]

  # check if image has a corresponding label, otherwise warn and continue
  for f in images[:]:
    if f not in labels[:]:
      # warn
      print("Image file %s has no label, GIMME DAT LABEL YO! Ignoring..." % f)
      # ignore image
      images.remove(f)
    else:
      # success!
      n_data += 1
      # calculate class content in the image
      # print("Calculating label content of label %s"%f)
      l = cv2.imread(label_dir + f, 0)  # open label as grayscale
      h, w = l.shape
      total_pix += h * w  # add to the total count of pixels in images
      # print("Number of pixels in image %s"%(h*w))
      # create histogram
      hist = cv2.calcHist([l], [0], None, [256], [0, 256])
      for key in content_perc:
        # look known class
        content_perc[key] += hist[key]
        hist[key] = 0
        # print("Content of class %s: %f"%(label_map[key],content_perc[key]))
      # report unknowkn class
      flag = 0
      for i in range(0, 256):
        if hist[i] > 0:
          if not flag:
            print(Back.RED + "Achtung! Img %s" % f + Style.RESET_ALL)
            flag = 1
          print(Fore.RED + "   ├─── labels contains %d unmapped class pixels with value %d" %
                (hist[i], i) + Style.RESET_ALL)
      if flag:
        # there were pixels that don't belong to our classes, so drop before
        # breaking our crossentropy
        print("Dropping image %s" % f)
        labels.remove(f)
        images.remove(f)

  # loop labels checking rogue labels with no images (magic label from ether)
  for f in labels[:]:
    if f not in images[:]:
      # warn
      print("Label file %s has no image, IS THIS MAGIC?! Ignoring..." % f)
      # ignore image
      labels.remove(f)

  # remove remap folders to create them again
  if force_remap and os.path.exists(label_remap_dir):
    shutil.rmtree(label_remap_dir)
  if force_remap and os.path.exists(img_remap_dir):
    shutil.rmtree(img_remap_dir)

  # remap all labels to [0,num_classes], otherwise it breaks the crossentropy
  if not os.path.exists(label_remap_dir):
    print("Cross Entropy remap non existent, creating...")
    os.makedirs(label_remap_dir)
    for f in labels:
      lbl = cv2.imread(label_dir + f, 0)
      if(new_shape is not None):
        lbl = cv2.resize(lbl, new_shape, interpolation=cv2.INTER_NEAREST)
      for key in label_remap:
        lbl[lbl == key] = label_remap[key]
      cv2.imwrite(directory + '/lbl/remap/' + f, lbl)

  # remap all images to jpg and resized to proper size, so that we open faster
  new_images = []
  if not os.path.exists(img_remap_dir):
    print("Jpeg remap non existent, creating...")
    os.makedirs(img_remap_dir)
    for f in images:
      img = cv2.imread(img_dir + f, cv2.IMREAD_UNCHANGED)
      f = os.path.splitext(f)[0] + '.jpg'
      new_images.append(f)
      if(new_shape is not None):
        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)
      cv2.imwrite(img_remap_dir + f, img)
  else:
    for f in images:
      f = os.path.splitext(f)[0] + '.jpg'
      new_images.append(f)

  # final percentage calculation
  print("Total number of pixels: %d" % (total_pix))
  for key in content_perc:
    print("Total number of pixels of class %s: %d" %
          (label_map[key], content_perc[key]))
    content_perc[key] /= total_pix
  # report number of class labels
  for key in label_map:
    print("Content percentage of class %s in dataset: %f" %
          (label_map[key], content_perc[key]))
  print("Total amount of images: %d" % n_data)
  print("---------------------------------------------------------------------")

  print(' SPECIFIC TO CITYSCAPES '.center(80, '*'))
  print("Don't weigh the 'crap' class (key 255)")
  # this is a hack, and needs to be done more elegantly
  content_perc[255] = float("inf")
  print("Content percentage of class %s in dataset: %f" %
        (label_map[255], content_perc[255]))
  print(' SPECIFIC TO CITYSCAPES '.center(80, '*'))

  # prepend the folder to each file name
  new_images = [directory + '/img/remap/' + name for name in new_images]
  labels = [directory + '/lbl/remap/' + name for name in labels]

  # order to ensure matching (necessary?)
  new_images.sort()
  labels.sort()

  return new_images, labels, n_data, content_perc


def read_data_sets(DATA):
  """Get the dataset in the format that we need it
     and give it to the main app
  -*- coding: utf-8 -*-
  Args:
    DATA: Dictionary with dataset information.
          Structure for the input dir:
            Dataset
              ├── test
              │   ├── img
              │   │   └── pic3.jpg
              │   └── lbl
              │       └── pic3.jpg
              ├── train
              │   ├── img
              │   │   └── pic1.jpg
              │   └── lbl
              │       └── pic1.jpg
              └── valid
                  ├── img
                  │   └── pic2.jpg
                  └── lbl
                      └── pic2.jpg
  Returns:
    data_sets: Object of class dataset where all training, validation and
               test data is stored, along with other information to train
               the desired net.
  """

  # get the datasets from the folders
  data_dir = DATA['data_dir']

  directories = ["/train", "/valid", "/test"]
  types = ["/img", "/lbl"]

  # check that all folders exist:
  for d in directories:
    for t in types:
      if not os.path.exists(data_dir + d + t):
        print("%s dir does not exist. Check dataset folder" %
              (data_dir + d + t))
        quit()

  print("Data depth: ", DATA["img_prop"]["depth"])

  # if force resize, do it properly
  if "force_resize" in DATA and DATA["force_resize"]:
    new_rows = DATA["img_prop"]["height"]
    new_cols = DATA["img_prop"]["width"]
    new_shape = (new_cols, new_rows)
  else:
    new_shape = None

  # for backward compatibility
  if "force_remap" in DATA:
    force_remap = DATA["force_remap"]
  else:
    force_remap = False

  # train data
  train_img, train_lbl, train_n, train_cont = dir_to_data(join(data_dir, "train"),
                                                          DATA["label_map"],
                                                          DATA["label_remap"],
                                                          new_shape=new_shape,
                                                          force_remap=force_remap)
  train_data = abs_data.Dataset(train_img, train_lbl, train_n, train_cont,
                                "train", DATA)

  # validation data
  valid_img, valid_lbl, valid_n, valid_cont = dir_to_data(join(data_dir, "valid"),
                                                          DATA["label_map"],
                                                          DATA["label_remap"],
                                                          new_shape=new_shape,
                                                          force_remap=force_remap)
  valid_data = abs_data.Dataset(valid_img, valid_lbl, valid_n, valid_cont,
                                "valid", DATA)

  # test data
  test_img, test_lbl, test_n, test_cont = dir_to_data(join(data_dir, "test"),
                                                      DATA["label_map"],
                                                      DATA["label_remap"],
                                                      new_shape=new_shape,
                                                      force_remap=force_remap)
  test_data = abs_data.Dataset(test_img, test_lbl, test_n, test_cont,
                               "test", DATA)

  data_sets = abs_data.FullDataset(train_data, valid_data, test_data, DATA)

  print("Successfully imported datasets")
  print("Train data samples: ", train_n)
  print("Validation data samples: ", valid_n)
  print("Test data samples: ", test_n)

  return data_sets
