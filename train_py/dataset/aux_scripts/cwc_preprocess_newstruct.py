#!/usr/bin/python3
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

'''
  Use the output format extracted from the BAG that uses images and color
  labels created by Philipp's label creator.

  Input:

     - InputDirectory
          ├── images
          │   └── rgb
          │       ├── bonirob_2016-05-10-11-29-18_13_frame0.png
          │       ├── bonirob_2016-05-10-11-29-18_13_frame1.png
          │       └── etc...
          └── annotations
              └── dlp
                   └──colorCleaned
                        ├── bonirob_2016-05-10-11-29-18_13_frame0.png
                        ├── bonirob_2016-05-10-11-29-18_13_frame1.png OR
                        ├── bonirob_2016-05-05-13-20-54_20_frame0_GroundTruth_color
                        └──etc...

      - OutputDirectory: Where to put the images

      - Yaml config: - Mapping from colors in GroundTruth_color images to class
                       numbers in grayscale image we output
                     - Split for the randomization of the dataset

  Output:

      - OutputDirectory
              ├── train
              │   ├── img
              │   │   ├── xxxxxxxxxx.png
              │   │   └── xxxxxxxxxx.png
              │   └── lbl
              │       ├── xxxxxxxxxx.png
              │       └── xxxxxxxxxx.png
              ├── valid
              │   ├── img
              │   │   ├── xxxxxxxxxx.png
              │   │   └── xxxxxxxxxx.png
              │   └── lbl
              │       ├── xxxxxxxxxx.png
              │       └── xxxxxxxxxx.png
              └──test
                  ├── img
                  │   ├── xxxxxxxxxx.png
                  │   └── xxxxxxxxxx.png
                  └── lbl
                      ├── xxxxxxxxxx.png
                      └── xxxxxxxxxx.png

      - The directory is randomized, split according to desired percentages, and
        all the labels are mapped according to the mapping we want.
'''

import os
import yaml
from shutil import copyfile
import argparse
import shutil
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import random
import math

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./cwc_preprocess.py")
  parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      help='Dataset location. No default!',
  )
  parser.add_argument(
      '--cfg', '-c',
      type=str,
      required=True,
      help='Config yaml location. No default!',
  )
  parser.add_argument(
      '--out', '-o',
      type=str,
      required=True,
      help='Organized dataset location. No default!',
  )
  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("----------")
  print("INTERFACE:")
  print("dataset: %s" % FLAGS.dataset)
  print("cfg: %s" % FLAGS.cfg)
  print("out: %s" % FLAGS.out)
  print("----------")

  # try to open dataset yaml
  try:
    print("Opening cfg file %s" % FLAGS.cfg)
    f = open(FLAGS.cfg, 'r')
    CFG = yaml.load(f)
  except:
    print("Error opening cfg yaml file. Check! Exiting...")
    quit()

  # try to get the dataset:
  print("Parsing the dataset")
  rgb_img_path = "/images/rgb/"
  annotations_path = "/annotations/dlp/colorCleaned/"
  directories = [rgb_img_path, annotations_path]
  for d in directories:
    if not os.path.exists(FLAGS.dataset + d):
      print("%s dir does not exist. Check dataset folder" %
            (FLAGS.dataset + d))
      quit()
    else:
      print("Found dir dataset->%s" % (d))

  # create the output directory structure
  try:
    if os.path.exists(FLAGS.out):
      shutil.rmtree(FLAGS.out)
    print("Creating output dir")
    os.makedirs(FLAGS.out)
  except:
    print("Cannot create output dir")

  # create temporary output directory to put all the images and labels
  tmpdir = "/tmp/cwctmp"
  # try to remove
  try:
    shutil.rmtree(tmpdir)
  except:
    print("no tmp dir to remove, passing")
    pass
  # try to create
  try:
    print("Creating /tmp/cwctmp dir")
    os.makedirs(tmpdir)
    os.makedirs(tmpdir + "/images/")
    os.makedirs(tmpdir + "/labels/")
  except:
    print("Cannot create tmp dirs")
    quit()

  # check which files have an image and a corresponding label, and put both in
  # a list
  images = [f for f in listdir(FLAGS.dataset + rgb_img_path) if isfile(join(FLAGS.dataset + rgb_img_path, f))]
  labels = [f for f in listdir(FLAGS.dataset + annotations_path)
            if isfile(join(FLAGS.dataset + annotations_path, f))]
  ord_images = []
  ord_labels = []

  print("Matching images to labels when possible")
  for img_name in images:
    # get img stem for finding alt label
    stem = os.path.splitext(img_name)[0]
    # convert into label name
    alt_label_name = stem + "_GroundTruth_color.png"
    alt_label_name_2 = stem + "_GroundTruth_iMap.png"
    # is it in list?
    if (img_name in labels):
      # put in ordered list
      # print(img_name,label_name)
      ord_images.append(img_name)
      ord_labels.append(img_name)
    elif (alt_label_name in labels):
      # put in ordered list
      # print(img_name,label_name)
      ord_images.append(img_name)
      ord_labels.append(alt_label_name)
    elif (alt_label_name_2 in labels):
      # put in ordered list
      # print(img_name,label_name)
      ord_images.append(img_name)
      ord_labels.append(alt_label_name_2)
    else:
      print("No label for img %s" % (img_name))
  # zip them to access both at the same time
  duples = zip(ord_images, ord_labels)

  # Map each RGB label to a monochrome label and save it and its image to tmp dir
  # naming them more nicely (to match at least)
  for t in duples:
    copyfile(FLAGS.dataset + rgb_img_path +
             t[0], tmpdir + "/images/" + t[0])  # copy image
    copyfile(FLAGS.dataset + annotations_path +
             t[1], tmpdir + "/labels/" + t[0])  # copy image
    # map labels to monochrome
    lbl_name = tmpdir + "/labels/" + t[0]
    print(lbl_name)
    lbl = cv2.imread(lbl_name)
    h, w, d = lbl.shape
    graylbl = np.zeros((h, w), np.uint8)
    for key in CFG["color_map"]:
      graylbl[np.where((lbl == CFG["color_map"][key]).all(2))] = key
    cv2.imwrite(lbl_name, graylbl)

  # shuffle temp dir files (only images, we know now that labels have the same name)
  images = [f for f in listdir(tmpdir + "/images/")
            if isfile(join(tmpdir + "/images/", f))]
  files_num = len(images)
  random.shuffle(images)

  # split according to desired split
  # create train, valid and test dirs
  directories = ["/train", "/valid", "/test"]
  types = ["/img", "/lbl"]
  for d in directories:
    print("Creating %s dir." % (FLAGS.out + d))
    if not os.path.exists(FLAGS.out + d):
      os.makedirs(FLAGS.out + d)
    for t in types:
      print("Creating %s dir." % (FLAGS.out + d + t))
      if not os.path.exists(FLAGS.out + d + t):
        os.makedirs(FLAGS.out + d + t)

  # move into output structure
  train_split = CFG["split"][0]
  valid_split = CFG["split"][1]
  test_split = CFG["split"][2]
  if train_split + valid_split + valid_split > 100:
    print("Watch out, split is inconsistent. Doing my best.")
    # normalize
    n = (train_split + valid_split + test_split) / 100.00
    print("modified train split %f->%f" % (train_split, train_split / n))
    train_split /= n
    print("modified valid split %f->%f" % (valid_split, valid_split / n))
    valid_split /= n
    print("modified test split %f->%f" % (test_split, test_split / n))
    test_split /= n
  files_num = len(images)

  # train
  train_num = int(math.floor(files_num * float(train_split) / 100.0))
  train_set = images[0:train_num]
  print("Copying train files to %s" % (FLAGS.out + "/train/"))
  for f in train_set:
    copyfile(tmpdir + "/images/" + f, FLAGS.out +
             "/train/img/" + f)  # copy images
    copyfile(tmpdir + "/labels/" + f, FLAGS.out +
             "/train/lbl/" + f)  # copy images

  # valid
  valid_num = int(math.floor(files_num * float(valid_split) / 100.0))
  valid_set = images[train_num:train_num + valid_num]
  for f in valid_set:
    copyfile(tmpdir + "/images/" + f, FLAGS.out +
             "/valid/img/" + f)  # copy images
    copyfile(tmpdir + "/labels/" + f, FLAGS.out +
             "/valid/lbl/" + f)  # copy images

  # test
  test_num = int(math.floor(files_num * float(test_split) / 100.0))
  test_set = images[train_num + valid_num:train_num + valid_num + test_num]
  for f in test_set:
    copyfile(tmpdir + "/images/" + f, FLAGS.out +
             "/test/img/" + f)  # copy images
    copyfile(tmpdir + "/labels/" + f, FLAGS.out +
             "/test/lbl/" + f)  # copy images

  # delete tmp dir
  shutil.rmtree(tmpdir)
