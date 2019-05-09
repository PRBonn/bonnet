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

""" Abstraction for every dataset. All of the datasets we use should have
    this overall structure
"""

# queue for the pre-fetching
import queue
import threading
import cv2
import numpy as np
import os
import random


class ImgFetcher(threading.Thread):
  def __init__(self, name, dataset):
    threading.Thread.__init__(self)
    self.name = name
    self.dataset = dataset
    self.die = False
    if self.name == "ImgBufftrain":
      # randomize data at the beginning
      combined = list(zip(self.dataset.images, self.dataset.labels))
      random.shuffle(combined)
      self.dataset.images[:], self.dataset.labels[:] = zip(*combined)

  def augment(self, img, lbl):
    # define the augmentations

    # flip horizontally?
    flip = bool(random.getrandbits(1))
    if flip:
      img = cv2.flip(img, 1)
      lbl = cv2.flip(lbl, 1)

    # gamma shift
    gamma = bool(random.getrandbits(1))
    if gamma:
      # build a lookup table mapping the pixel values [0, 255] to
      # their adjusted gamma values
      randomGamma = np.random.uniform(low=0.8, high=1.2)
      invGamma = 1.0 / randomGamma
      table = np.array([((i / 255.0) ** invGamma) * 255
                        for i in np.arange(0, 256)]).astype("uint8")
      img = cv2.LUT(img, table)

    # blur
    blur = bool(random.getrandbits(1))
    if blur:
      ksize = random.randint(3, 7)
      img = cv2.blur(img,(ksize,ksize))

    return img, lbl

  def run(self):
    # loop infinitely, the queue will block
    while not self.die:
      # Fetch images
      # print self.dataset.images[self.dataset.idx]
      img = cv2.imread(
          self.dataset.images[self.dataset.idx], cv2.IMREAD_COLOR)
      lbl = cv2.imread(self.dataset.labels[self.dataset.idx], 0)
      name = os.path.basename(self.dataset.images[self.dataset.idx])

      # augment
      if self.name == "ImgBufftrain":
        img, lbl = self.augment(img, lbl)

      # queue if there is still room, otherwise block
      self.dataset.img_q.put(img)  # blocking
      self.dataset.lbl_q.put(lbl)  # blocking
      self.dataset.name_q.put(name)  # blocking
      self.dataset.idx += 1
      if self.dataset.idx == self.dataset.num_examples:
        self.dataset.idx = 0  # begin again
        if self.name == "ImgBufftrain":
          # randomize data after each epoch
          combined = list(zip(self.dataset.images, self.dataset.labels))
          random.shuffle(combined)
          self.dataset.images[:], self.dataset.labels[:] = zip(*combined)

    print("Exiting Thread %s" % self.name)

  def cleanup(self):
    self.die = True
    if not self.dataset.img_q.empty():
      self.dataset.img_q.get()
    if not self.dataset.lbl_q.empty():
      self.dataset.lbl_q.get()


class Dataset:
  """ Dataset class, for abstraction
      Contains all images and labels for each set (train, validation or test)
  """

  def __init__(self, images, labels, num_examples, content, name, DATA):
    self.images = images  # name of files, not data!
    self.labels = labels  # name of files, not data!
    self.num_examples = num_examples
    self.idx = 0
    self.img_width = DATA["img_prop"]["width"]
    self.img_height = DATA["img_prop"]["height"]
    self.img_depth = DATA["img_prop"]["depth"]
    self.content = content  # matches label_map keys, but contains class
    # percentage as ratio of pixels in entire dataset.
    self.name = name
    # if true, we spawn a thread that prefetches batches for training
    self.buff = DATA["buff"]
    # init prefetch thread
    self.init(DATA["buff_nr"])

  def init(self, img_nr):  # img_nr should be bigger than 1 batch for it to make sense
    # only start the thread if buffering was enabled
    if self.buff:
      # create the fifo
      self.img_q = queue.Queue(maxsize=img_nr)
      self.lbl_q = queue.Queue(maxsize=img_nr)
      self.name_q = queue.Queue(maxsize=img_nr)
      # start a thread pre-fetching images to get them fast from ram in a batch
      self.imgfetcher = ImgFetcher("ImgBuff" + self.name, self)
      self.imgfetcher.setDaemon(True)
      self.imgfetcher.start()
    else:
      print("Batch buff has been disabled, so images will be opened on the fly")
    return

  def next_batch(self, size):
    '''
      Return size items (wraps around if the last elements are less than a
      batch size. Be careful with this for evaluation)
    '''
    # different if images are being buffered or not
    images = []
    labels = []
    names = []
    if self.buff:
      for i in range(0, size):
        images.append(self.img_q.get())  # blocking
        labels.append(self.lbl_q.get())  # blocking
        names.append(self.name_q.get())  # blocking
    else:
      for i in range(0, size):
        img = cv2.imread(self.images[self.idx], cv2.IMREAD_UNCHANGED)
        lbl = cv2.imread(self.labels[self.idx], 0)
        images.append(img)
        labels.append(lbl)
        names.append(os.path.basename(self.images[self.idx]))
        self.idx += 1
        if self.idx == self.num_examples:
          self.idx = 0
    return images, labels, names

  def cleanup(self):
    if self.buff:
      self.imgfetcher.cleanup()


class FullDataset:
  """ FullDataset class, for abstraction
      Contains all training, validation and test data, along with image size,
      depth, and number of classes
  """

  def __init__(self, train, validation, test, DATA):
    self.train = train
    self.validation = validation
    self.test = test
    self.label_map = DATA["label_map"]  # maps value to string class
    self.num_classes = len(self.label_map)  # number of classes to classify

  def cleanup(self):
    self.train.cleanup()
    self.validation.cleanup()
    self.test.cleanup()
