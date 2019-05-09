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

  Inputs are already split in train,val,test, and put into a format we can read, so all we need to do is join them.

  Input:

     - InputDirectory (with the remaped images already there)
              ├── gtFine_trainvaltest
              │   └── gtFine
              │       ├── test
              │       │   ├── berlin
              │       │   ├── bielefeld
              │       │   ├── bonn
              │       │   ├── leverkusen
              │       │   ├── mainz
              │       │   └── munich
              │       ├── train
              │       │   ├── aachen
              │       │   ├── bochum
              │       │   ├── bremen
              │       │   ├── cologne
              │       │   ├── darmstadt
              │       │   ├── dusseldorf
              │       │   ├── erfurt
              │       │   ├── hamburg
              │       │   ├── hanover
              │       │   ├── jena
              │       │   ├── krefeld
              │       │   ├── monchengladbach
              │       │   ├── strasbourg
              │       │   ├── stuttgart
              │       │   ├── tubingen
              │       │   ├── ulm
              │       │   ├── weimar
              │       │   └── zurich
              │       │         ├── zurich_000000_000019_gtFine_labelTrainIds.png
              │       │         ├── zurich_000001_000019_gtFine_labelTrainIds.png
              │       │         ├── zurich_000002_000019_gtFine_labelTrainIds.png
              │       │         ├── zurich_000003_000019_gtFine_labelTrainIds.png
              │       │         └── zurich_000004_000019_gtFine_labelTrainIds.png
              │       │ 
              │       └── val
              │           ├── frankfurt
              │           ├── lindau
              │           └── munster
              └── leftImg8bit_trainvaltest
                  └── leftImg8bit
                      ├── test
                      │   ├── berlin
                      │   ├── bielefeld
                      │   ├── bonn
                      │   ├── leverkusen
                      │   ├── mainz
                      │   └── munich
                      ├── train
                      │   ├── aachen
                      │   ├── bochum
                      │   ├── bremen
                      │   ├── cologne
                      │   ├── darmstadt
                      │   ├── dusseldorf
                      │   ├── erfurt
                      │   ├── hamburg
                      │   ├── hanover
                      │   ├── jena
                      │   ├── krefeld
                      │   ├── monchengladbach
                      │   ├── strasbourg
                      │   ├── stuttgart
                      │   ├── tubingen
                      │   ├── ulm
                      │   ├── weimar
                      │   └── zurich
                      └── val
                          ├── frankfurt
                          ├── lindau
                          └── munster
                                ├── munster_000169_000019_leftImg8bit.png
                                ├── munster_000170_000019_leftImg8bit.png
                                ├── munster_000171_000019_leftImg8bit.png
                                ├── munster_000172_000019_leftImg8bit.png
                                └── munster_000173_000019_leftImg8bit.png

      - OutputDirectory: Where to put the images

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
'''

import os
import yaml
from shutil import copyfile
import argparse
import shutil
import cv2
import numpy as np
import random
import math

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./synthia_preprocess.py")
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

  # rgb image stuff
  rgb_img_path = "leftImg8bit_trainvaltest/leftImg8bit/"
  rgb_img_tail = "_leftImg8bit.png"

  # annotation stuff
  annotations_path = "gtFine_trainvaltest/gtFine/"
  annotation_tail = "_gtFine_labelTrainIds.png"

  # check directories
  directories = [rgb_img_path, annotations_path]
  for d in directories:
    if not os.path.exists(os.path.join(FLAGS.dataset, d)):
      print("%s dir does not exist. Check dataset folder" %
            (os.path.join(FLAGS.dataset, d)))
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

  # create train, valid and test dirs
  directories = ["train", "valid", "test"]
  types = ["img", "lbl"]
  for d in directories:
    print("Creating %s dir." % (os.path.join(FLAGS.out, d)))
    if not os.path.exists(os.path.join(FLAGS.out, d)):
      os.makedirs(os.path.join(FLAGS.out, d))
    for t in types:
      print("Creating %s dir." % (os.path.join(FLAGS.out, d, t)))
      if not os.path.exists(os.path.join(FLAGS.out, d, t)):
        os.makedirs(os.path.join(FLAGS.out, d, t))

  # # train
  train_cities = ["aachen", "bochum", "bremen", "cologne", "darmstadt", "dusseldorf", "erfurt", "hamburg",
                  "hanover", "jena", "krefeld", "monchengladbach", "strasbourg", "stuttgart", "tubingen", "ulm", "weimar", "zurich"]
  for city in train_cities:
    images = [f for f in os.listdir(os.path.join(FLAGS.dataset, rgb_img_path, "train", city)) if (
        rgb_img_tail in f and os.path.isfile(os.path.join(FLAGS.dataset, rgb_img_path, "train", city, f)))]
    labels = [f for f in os.listdir(os.path.join(FLAGS.dataset, annotations_path, "train", city)) if (
        annotation_tail in f and os.path.isfile(os.path.join(FLAGS.dataset, annotations_path, "train", city, f)))]
    for img in images:
      src = os.path.join(FLAGS.dataset, rgb_img_path, "train", city, img)
      dst = os.path.join(FLAGS.out, "train/img",
                         img.replace(rgb_img_tail, ".png"))
      print("copying ", src, " to ", dst)
      copyfile(src, dst)  # copy image
    for lbl in labels:
      src = os.path.join(FLAGS.dataset, annotations_path, "train", city, lbl)
      dst = os.path.join(FLAGS.out, "train/lbl",
                         lbl.replace(annotation_tail, ".png"))
      print("copying ", src, " to ", dst)
      copyfile(src, dst)  # copy image

  # # valid
  val_cities = ["frankfurt", "lindau", "munster"]
  for city in val_cities:
    images = [f for f in os.listdir(os.path.join(FLAGS.dataset, rgb_img_path, "val", city)) if (
        rgb_img_tail in f and os.path.isfile(os.path.join(FLAGS.dataset, rgb_img_path, "val", city, f)))]
    labels = [f for f in os.listdir(os.path.join(FLAGS.dataset, annotations_path, "val", city)) if (
        annotation_tail in f and os.path.isfile(os.path.join(FLAGS.dataset, annotations_path, "val", city, f)))]
    for img in images:
      src = os.path.join(FLAGS.dataset, rgb_img_path, "val", city, img)
      dst = os.path.join(FLAGS.out, "valid/img",
                         img.replace(rgb_img_tail, ".png"))
      print("copying ", src, " to ", dst)
      copyfile(src, dst)  # copy image
    for lbl in labels:
      src = os.path.join(FLAGS.dataset, annotations_path, "val", city, lbl)
      dst = os.path.join(FLAGS.out, "valid/lbl",
                         lbl.replace(annotation_tail, ".png"))
      print("copying ", src, " to ", dst)
      copyfile(src, dst)  # copy image

  # test
  test_cities = ["berlin", "bielefeld",
                 "bonn", "leverkusen", "mainz", "munich"]
  # create black labels, because we have nothing here.
  black_label = np.full((100, 100), 255, dtype=np.uint8)
  for city in test_cities:
    images = [f for f in os.listdir(os.path.join(FLAGS.dataset, rgb_img_path, "test", city)) if (
        rgb_img_tail in f and os.path.isfile(os.path.join(FLAGS.dataset, rgb_img_path, "test", city, f)))]
    for img in images:
      src = os.path.join(FLAGS.dataset, rgb_img_path, "test", city, img)
      dst = os.path.join(FLAGS.out, "test/img",
                         img.replace(rgb_img_tail, ".png"))
      print("copying ", src, " to ", dst)
      copyfile(src, dst)  # copy image
      lbl_dst = os.path.join(FLAGS.out, "test/lbl",
                             img.replace(rgb_img_tail, ".png"))
      cv2.imwrite(lbl_dst, black_label)
