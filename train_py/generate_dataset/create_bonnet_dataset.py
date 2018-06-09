# this script is helping to create dataset from image and labeled image folders. By create a new folder for Training,
# Validation, and Testing. AS default the image of validation and Testing dataset will be 10% for each, however, it
# could be change that by but the value they want for the arguments --v and --t.

import argparse
import os
import numpy as np
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument('--img', action='store', dest='images_path', help='the path of the images folder.')
parser.add_argument('--lbl', action='store', dest='labels_path', help='the path of the labels folder.')
parser.add_argument('--dis', action='store', dest='des_path', help='the path of the save folder.'
                    , default=False)
parser.add_argument('--v', action='store', dest='val_per', help='the percent of the validation set'
                    , default=False, type=int)
parser.add_argument('--t', action='store', dest='test_per', help='the percent of the test set'
                    , default=False, type=float)

args = parser.parse_args()
if not args.des_path:
    args.des_path = "."

if not args.val_per:
    args.val_per = 10

if not args.test_per:
    args.test_per = 10

img_list = np.array(os.listdir(args.images_path))
lbl_list = np.array(os.listdir(args.labels_path))

# create random array
random_array = np.arange(len(img_list))
np.random.shuffle(random_array)

# find the test_set
num_of_test_set = int(len(img_list) * args.test_per / 100)
test_data_set = img_list[random_array[:num_of_test_set]]
# get index of test dataset
test_data_set_idx = list()
for f in test_data_set:
    test_data_set_idx.append(list(img_list).index(f))

# delete the test data set from img_list
img_list = np.delete(img_list, np.array(test_data_set_idx))
lbl_list = np.delete(lbl_list, np.array(test_data_set_idx))

# create random array
random_array = np.arange(len(img_list))
np.random.shuffle(random_array)
# find the valid set
num_of_valid_set = int(len(img_list) * args.val_per / 100)
valid_data_set = img_list[random_array[:num_of_valid_set]]

# create folders
# create Bonnet_dataset
args.des_path = os.path.join(args.des_path, "Bonnet_dataset")
if not os.path.exists(args.des_path):
    os.makedirs(args.des_path)

# create train
train_path = os.path.join(args.des_path, "train")
if not os.path.exists(train_path):
    os.makedirs(train_path)

train_img_path = os.path.join(train_path, "img")
if not os.path.exists(train_img_path):
    os.makedirs(train_img_path)

train_lbl_path = os.path.join(train_path, "lbl")
if not os.path.exists(train_lbl_path):
    os.makedirs(train_lbl_path)

# copy train files
for f in img_list:
    copyfile(os.path.join(args.images_path, f), os.path.join(train_img_path, f))
    copyfile(os.path.join(args.labels_path, f), os.path.join(train_lbl_path, f))

# create validation
valid_path = os.path.join(args.des_path, "valid")
if not os.path.exists(valid_path):
    os.makedirs(valid_path)

valid_img_path = os.path.join(valid_path, "img")
if not os.path.exists(valid_img_path):
    os.makedirs(valid_img_path)

valid_lbl_path = os.path.join(valid_path, "lbl")
if not os.path.exists(valid_lbl_path):
    os.makedirs(valid_lbl_path)

# copy valid files
for f in valid_data_set:
    copyfile(os.path.join(args.images_path, f), os.path.join(valid_img_path, f))
    copyfile(os.path.join(args.labels_path, f), os.path.join(valid_lbl_path, f))

# create test
test_path = os.path.join(args.des_path, "test")
if not os.path.exists(test_path):
    os.makedirs(test_path)

test_img_path = os.path.join(test_path, "img")
if not os.path.exists(test_img_path):
    os.makedirs(test_img_path)

test_lbl_path = os.path.join(test_path, "lbl")
if not os.path.exists(test_lbl_path):
    os.makedirs(test_lbl_path)

# copy valid files
for f in test_data_set:
    copyfile(os.path.join(args.images_path, f), os.path.join(test_img_path, f))
    copyfile(os.path.join(args.labels_path, f), os.path.join(test_lbl_path, f))
