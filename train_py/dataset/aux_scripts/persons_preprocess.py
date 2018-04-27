#!/usr/bin/python3
# coding: utf-8

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
  Use the output format extracted from the BAG that uses images, labels created
  by Philipp's label creator and the YAML to create a consistent 1 to 1 mapping
  dataset of image->label

  Input:

     - InputDirectory
          ├── persons__ds1/
          │        ├── img
          │        └── masks_machine
          ├── persons__ds10/
          ├── persons__ds11/
          ├── persons__ds12/
          ├── persons__ds13/
          ├── persons__ds2/
          ├── persons__ds3/
          ├── persons__ds4/
          ├── persons__ds5/
          ├── persons__ds6/
          ├── persons__ds7/
          ├── persons__ds8/
          └── persons__ds9/

      - OutputDirectory: Where to put the images

  Output:

      - OutputDirectory
              ├── train
              │   ├── img
              │   │   ├── 1.png
              │   │   └── 4.png
              │   └── lbl
              │       ├── 1.png
              │       └── 4.png
              ├── valid
              │   ├── img
              │   │   ├── 2.png
              │   │   └── 6.png
              │   └── lbl
              │       ├── 2.png
              │       └── 6.png
              └──test
                  ├── img
                  │   ├── 5.png
                  │   └── 3.png
                  └── lbl
                      ├── 5.png
                      └── 3.png

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
  print("out: %s" % FLAGS.out)
  print("----------")

  # try to get the dataset:
  print("Parsing the dataset")
  datasets = ["/persons__ds10/",
              "/persons__ds11/",
              "/persons__ds12/",
              "/persons__ds13/",
              "/persons__ds2/",
              "/persons__ds3/",
              "/persons__ds4/",
              "/persons__ds5/",
              "/persons__ds6/",
              "/persons__ds7/",
              "/persons__ds8/",
              "/persons__ds9/",
              "/coco_val2017/",
              "/coco_train2017/"]
  directories = ["/img", "/masks_machine"]

  for dat in datasets:
    for dire in directories:
      if not os.path.exists(FLAGS.dataset + dat + dire):
        print("%s dir does not exist. Check dataset folder" %
              (FLAGS.dataset + dat + dire))
        quit()
      else:
        print("Found dir dataset->%s" % (FLAGS.dataset + dat + dire))

  # create the output directory structure
  try:
    if os.path.exists(FLAGS.out):
      shutil.rmtree(FLAGS.out)
    print("Creating output dir")
    os.makedirs(FLAGS.out)
  except:
    print("Cannot create output dir")

  # check which files have an image and a corresponding label, and put both in
  # a list
  images = []
  labels = []
  for dataset in datasets:
    # append new images (avoiding duplicates)
    new_images = [FLAGS.dataset + dataset + '/img/' + f
                  for f in listdir(FLAGS.dataset + dataset + '/img/')
                  if isfile(join(FLAGS.dataset + dataset + '/img/', f))]
    for image in new_images:
      if os.path.splitext(os.path.basename(image))[0] not in [os.path.splitext(os.path.basename(i))[0] for i in images]:
        images.append(image)
      else:
        print("Image", image, "is duplicated. Removing...")

    # append new labels (avoiding duplicates)
    new_labels = [FLAGS.dataset + dataset + '/masks_machine/' + f
                  for f in listdir(FLAGS.dataset + dataset + '/masks_machine/')
                  if isfile(join(FLAGS.dataset + dataset + '/masks_machine/', f))]
    for label in new_labels:
      if os.path.splitext(os.path.basename(label))[0] not in [os.path.splitext(os.path.basename(l))[0] for l in labels]:
        labels.append(label)
      else:
        print("label", label, "is duplicated. Removing...")

  # order to match
  images.sort()
  labels.sort()
  assert(len(images) == len(labels))
  duples = list(zip(images, labels))

  # shuffle temp dir files (only images, we know now that labels have the same name)
  files_num = len(duples)
  random.shuffle(duples)

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
  train_split = 80
  valid_split = 10
  test_split = 10
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

  # train
  print("Copying train files to %s" % (FLAGS.out + "/train/"))
  train_num = int(math.floor(files_num * float(train_split) / 100.0))
  train_set = duples[0:train_num]
  for d in train_set:
    copyfile(d[0], FLAGS.out + "/train/img/" +
             os.path.basename(d[0]))  # copy images
    copyfile(d[1], FLAGS.out + "/train/lbl/" +
             os.path.basename(d[1]))  # copy labels

  # valid
  print("Copying valid files to %s" % (FLAGS.out + "/valid/"))
  valid_num = int(math.floor(files_num * float(valid_split) / 100.0))
  valid_set = duples[train_num:train_num + valid_num]
  for d in valid_set:
    copyfile(d[0], FLAGS.out + "/valid/img/" +
             os.path.basename(d[0]))  # copy images
    copyfile(d[1], FLAGS.out + "/valid/lbl/" +
             os.path.basename(d[1]))  # copy labels

  # test
  print("Copying test files to %s" % (FLAGS.out + "/test/"))
  test_num = int(math.floor(files_num * float(test_split) / 100.0))
  test_set = duples[train_num + valid_num:train_num + valid_num + test_num]
  for d in test_set:
    copyfile(d[0], FLAGS.out + "/test/img/" +
             os.path.basename(d[0]))  # copy images
    copyfile(d[1], FLAGS.out + "/test/lbl/" +
             os.path.basename(d[1]))  # copy labels
