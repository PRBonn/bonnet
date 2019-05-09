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

"""
This code uses opencv to apply different augmentations to a single image or a
training dataset, which is a common practice in machine learning. Functions are
pretty self-explanatory.

General comment for all functions:
  We don't do anything with labels in this functions, they take images as
  inputs, and return images as output, but in the same order, so label
  management can be done easily in a higher level.

  e.g: images: [...,img1,img2,...]
       labels: [...,lbl1,lbl2,...]
       index:  [...,i,   i+1 ,...]

       n_transformations: 2
       returns: [...,img1_1,img1_2,img1_3,img2_1,img2_2,img2_3,...]
       labels:  [...,lbl1,  lbl1,  lbl1,  lbl2,  lbl2,  lbl2,  ...]
       index:   [...,i,     i+1,    ...   i+3,   i+4     ...      ]
                                           ^---> (i+n_transformations+1)

  Indexes for labels should therefore travel in increments of
  (n_transformations+1) in the higher level to recover the labeled dataset
  structure. Randomize before feeding this to the training, even if the dataset
  has been shuffled before, since all the transformed images will be highly
  correlated with each other, and they will be of the same class!

  Unless otherwise noted, the above structure is used in all functions, where
  the original image is returned and the transformations requested are appended
  after it.
"""

# opencv is needed for the image transformations
import cv2
import numpy as np


def resize(img, new_size, neighbor=False):
  """
  SINGLE IMAGE FUNCTION - Takes one image, returns one image, not list!
  returns the resized img to size new size, where new_size=(rows,cols)

  When resizing labels we want to grab the nearest neighbor, not interpolate,
  so we should set neighbor to True
  """
  # get rows and cols of original image
  if len(img.shape) == 3:
    rows, cols, depth = img.shape
  else:
    rows, cols = img.shape

  # get rows and cols of new image
  new_rows = new_size[0]
  new_cols = new_size[1]

  # resizing should be done differently if we upsize or downsize, due to interpol
  if neighbor:
    interpol = cv2.INTER_NEAREST
  elif new_rows > rows:
    interpol = cv2.INTER_CUBIC
  else:
    interpol = cv2.INTER_LINEAR
  resized_img = cv2.resize(img, (cols, new_rows), interpolation=interpol)

  if neighbor:
    interpol = cv2.INTER_NEAREST
  elif new_cols > cols:
    interpol = cv2.INTER_CUBIC
  else:
    interpol = cv2.INTER_LINEAR
  resized_img = cv2.resize(
      resized_img, (new_cols, new_rows), interpolation=interpol)

  return resized_img


def extract_patch(img, corner1, corner2, resize=False, shape=None):
  """
  SINGLE IMAGE FUNCTION - Takes one image, returns one image, not list!

  Extracts patch from image img starting in corner1 and finishing in corner2,
  where both corners are given as a list of [x,y] coordinates in the original
  image. x and y start in the upper left corner of the image (opencv images)

  If shape is given, then ignore the second corner and use the first corner and
  the shape to extract the patch

  If any of the coordinates are off boundaries, the patch extracted will be
  cropped to the boundaries of the image. The boundaries can be expressed in
  any order, the function takes care of the rearrangement necessary for the
  crop.

  By default we return the patch, but if the flag resize is set to True,
  we resize the patch to the size of the original image (useful for CNN)

  Returns the extracted patch, if everything is correct, otherwise it returns
  None, for error checking
  """
  # sanity checks
  if (type(corner1) is not list or len(corner1) != 2 or
      type(corner2) is not list or len(corner2) != 2 or
          (shape and (type(shape) is not list or len(shape) != 2))):
    print("Wrong usage of the corner parameters")
    return None

  # get rows and cols of original image
  rows, cols, depth = img.shape

  # copy the corners internally to work with them
  pt1 = corner1[:]
  pt2 = corner2[:]

  # if shape is given, use instead of second corner
  if shape:
    pt2 = [pt1[0] + shape[0], pt1[1] + shape[1]]

  # limits
  minim_pt = [0, 0]
  maxim_pt = [cols, rows]

  # clip the values to the min (0,0) and max (rows,cols)
  for pt in [pt1, pt2]:
    for i in [0, 1]:
      pt[i] = minim_pt[i] if pt[i] < minim_pt[i] else pt[i]
      pt[i] = maxim_pt[i] if pt[i] > maxim_pt[i] else pt[i]

  # swap points if improperly arranged for cropping
  if pt2[0] < pt1[0]:
    pt2, pt1 = pt1, pt2
  if pt2[1] < pt1[1]:
    pt2[1], pt1[1] = pt1[1], pt2[1]

  # now crop
  patch = img[pt1[1]:pt2[1] + 1, pt1[0]:pt2[0] + 1]

  # resize?
  if resize:
    patch = cv2.resize(patch, (cols, rows), interpolation=cv2.INTER_CUBIC)

  return patch


def extract_patch_n(images, indexes, shape):
  """
  Extracts patches from the images with the indicated shape [x,y], from the
  indexes asked in the indexes list, where:
  indexes:
    1: top left
    2: top right
    3: center
    4: bottom left
    5: bottom right

  It returns a list with the original image and all the patches, unless
  something is wrong, in which case we return None for error checking
  """
  # if we only have 1 image, transform into a list to work with same script
  if type(images) is not list:
    images = [images]

  # if we only have 1 index, transform into a list to work with same script
  if type(indexes) is not list:
    indexes = [indexes]

  if not all((idx > 0 and idx < 6) for idx in indexes):
    print("Wrong usage of indexes -> Off boundaries")
    return None

  if type(shape) is not list or len(shape) != 2:
    print("Wrong usage of shape parameter")
    return None

  # container for patches
  patches = []

  # extract desired patches from each image
  for img in images:
    # get rows and cols to rotate
    rows, cols, depth = img.shape

    # append original at the beginning
    patches.append(img)

    # extract one patch per index
    for idx in indexes:
      patch = {
          1: lambda x: extract_patch(x, [0, 0], [None, None], shape=shape),
          2: lambda x: extract_patch(x, [cols - shape[0], 0], [None, None], shape=shape),
          3: lambda x: extract_patch(x, [cols / 2 - shape[0] / 2, rows / 2 - shape[1] / 2], [cols / 2 + shape[0] / 2, rows / 2 + shape[1] / 2]),
          4: lambda x: extract_patch(x, [0, rows - shape[1]], [None, None], shape=shape),
          5: lambda x: extract_patch(x, [cols - shape[0], rows - shape[1]], [None, None], shape=shape)
      }[idx](img)

      patches.append(patch)

  return patches


def rotations(images, n_rot, ccw_limit, cw_limit):
  """
  Rotates every image in the list "images" n_rot times, between 0 and cw_limit
  (clockwise limit) n_rot times and between 0 and ccw_limit (counterclockwise
  limit) n_rot times more. The limits are there to make sense of the data
  augmentation. E.g: Rotating an mnist digit 180 degrees turns a 6 into a 9,
  which makes no sense at all.

  cw_limit and ccw_limit are in degrees!

  Returns a list with all the rotated samples. Size will be 2*n_rot+1, because
  we also want the original sample to be included

  Example: images=[img],n_rot=3,ccw_limit=90,cw_limit=90
  Returns: [img1: original,
            img2: 90 degrees rot ccw,
            img3: 60 degrees rot ccw,
            img4: 30 degrees rot ccw,
            img5: 30 degrees rot cw,
            img5: 60 degrees rot cw
            img5: 90 degrees rot cw]
  """
  # if we only have 1 image, transform into a list to work with same script
  if type(images) is not list:
    images = [images]

  # calculate the initial angle and the step
  cw_step_angle = float(cw_limit) / float(n_rot)
  ccw_step_angle = float(ccw_limit) / float(n_rot)

  # container for rotated images
  rotated_images = []

  # get every image and apply the number of desired rotations
  for img in images:
    # get rows and cols to rotate
    rows, cols, depth = img.shape

    # append the original one too
    rotated_images.append(img)

    # rotate the amount of times we want them rotated
    for i in range(1, n_rot + 1):
      # create rotation matrix with center in the center of the image,
      # scale 1, and the desired angle (we travel counter clockwise first, and
      # then clockwise
      M_ccw = cv2.getRotationMatrix2D(
          (cols / 2, rows / 2), i * ccw_step_angle, 1)
      # rotate using the matrix (using bicubic interpolation)
      rot_img = cv2.warpAffine(img, M_ccw, (cols, rows), flags=cv2.INTER_CUBIC)
      # append to rotated images container
      rotated_images.append(rot_img)

      M_cw = cv2.getRotationMatrix2D(
          (cols / 2, rows / 2), -i * cw_step_angle, 1)
      # rotate using the matrix (using bicubic interpolation)
      rot_img = cv2.warpAffine(img, M_cw, (cols, rows), flags=cv2.INTER_CUBIC)
      # append to rotated images container
      rotated_images.append(rot_img)

  return rotated_images


def horiz_stretch(images, n_stretch, max_stretch, crop_center=True):
  """
  Applies a horizontal stretch transform to every image in the list "images"
  n_stretch times, with the max stretch passed in the max_stretch argument.

  Stretch > 1 expands the image, and < 1 compresses the image.

  By default, we crop the center of the images so that all images have the same
  shape, but this can be changed with the arg crop_center set to False.

  Returns a list with all the stretch samples. Size will be n_stretch+1, because
  we also want the original sample to be included.

  Example: images=[img],n_stretch=2, max_stretch=1.5
  Returns: [img1=img,
           img2=img with stretch of 1.25 in x,
           img3=img with stretch of 1.5 in x]
  """
  # if we only have 1 image, transform into a list to work with same script
  if type(images) is not list:
    images = [images]

  # calculate the stretch steps
  step_rel = (max_stretch - 1) / float(n_stretch)  # relative to image size

  # container for stretched images
  stretched_images = []

  # get every image and apply the number of desired stretches
  for img in images:
    # get rows and cols to stretch
    rows, cols, depth = img.shape

    stretched_images.append(img)

    # stretch the amount of times we want
    for i in range(1, n_stretch + 1):
      # create stretch matrix by mapping points
      # abs increase in size (or decrease if <1)
      new_size = int((step_rel * i + 1) * cols)

      # compress or stretch? (neg vs pos)
      pts1 = np.float32([[0, 0], [cols, 0], [cols, rows]])
      pts2 = np.float32([[0, 0], [new_size, 0], [new_size, rows]])
      M = cv2.getAffineTransform(pts1, pts2)
      stretch_img = cv2.warpAffine(
          img, M, (new_size, rows), flags=cv2.INTER_CUBIC)

      if crop_center:
        # different if I stretch or compress.
        if(max_stretch >= 1):
          # if stretch, then cut out center
          row_start = (stretch_img.shape[0] / 2) - (rows / 2)
          col_start = (stretch_img.shape[1] / 2) - (cols / 2)
          stretch_img = stretch_img[row_start:row_start +
                                    rows, col_start:col_start + cols]
        else:
          # if compress image will be smaller than original
          # fill image with zeros and copy the compressed one in the center
          fix_size_stretch_img = np.zeros(img.shape).astype(np.uint8)
          row_start = (rows - stretch_img.shape[0]) / 2
          col_start = (cols - stretch_img.shape[1]) / 2
          fix_size_stretch_img[row_start:row_start + stretch_img.shape[0],
                               col_start:col_start + stretch_img.shape[1]] = stretch_img
          stretch_img = fix_size_stretch_img
      # append to stretched images container
      stretched_images.append(stretch_img)

  return stretched_images


def vert_stretch(images, n_stretch, max_stretch, crop_center=True):
  """
  Applies a vertical stretch transform to every image in the list "images"
  n_stretch times, with the max stretch passed in the max_stretch argument.

  Stretch > 1 expands the image, and < 1 compresses the image.

  By default, we crop the center of the images so that all images have the same
  shape, but this can be changed with the arg crop_center set to False.

  Returns a list with all the stretch samples. Size will be n_stretch+1, because
  we also want the original sample to be included.

  Example: images=[img],n_stretch=2, max_stretch=1.5
  Returns: [img1=img,
           img2=img with stretch of 1.25 in y,
           img3=img with stretch of 1.5 in y]
  """
  # if we only have 1 image, transform into a list to work with same script
  if type(images) is not list:
    images = [images]

  # calculate the stretch steps
  step_rel = (max_stretch - 1) / float(n_stretch)  # relative to image size

  # container for stretched images
  stretched_images = []

  # get every image and apply the number of desired stretches
  for img in images:
    # get rows and cols to stretch
    rows, cols, depth = img.shape

    stretched_images.append(img)

    # stretch the amount of times we want
    for i in range(1, n_stretch + 1):
      # create stretch matrix by mapping points
      # abs increase in size (or decrease if <1)
      new_size = int((step_rel * i + 1) * rows)

      # compress or stretch? (neg vs pos)
      pts1 = np.float32([[0, 0], [0, rows], [cols, rows]])
      pts2 = np.float32([[0, 0], [0, new_size], [cols, new_size]])
      M = cv2.getAffineTransform(pts1, pts2)
      stretch_img = cv2.warpAffine(
          img, M, (cols, new_size), flags=cv2.INTER_CUBIC)

      if crop_center:
        # different if I stretch or compress.
        if(max_stretch >= 1):
          # if stretch, then cut out center
          row_start = (stretch_img.shape[0] / 2) - (rows / 2)
          col_start = (stretch_img.shape[1] / 2) - (cols / 2)
          stretch_img = stretch_img[row_start:row_start +
                                    rows, col_start:col_start + cols]
        else:
          # if compress image will be smaller than original
          # fill image with zeros and copy the compressed one in the center
          fix_size_stretch_img = np.zeros(img.shape).astype(np.uint8)
          row_start = (rows - stretch_img.shape[0]) / 2
          col_start = (cols - stretch_img.shape[1]) / 2
          fix_size_stretch_img[row_start:row_start + stretch_img.shape[0],
                               col_start:col_start + stretch_img.shape[1]] = stretch_img
          stretch_img = fix_size_stretch_img
      # append to stretched images container
      stretched_images.append(stretch_img)

  return stretched_images


def horiz_shear(images, n_shear, max_shear, crop_center=True):
  """
  Applies a horizontal shear transform to every image in the list "images"
  n_shear times, with the max shear passed in the max_shear argument. By
  default, we crop the center of the images so that all images have the same
  shape, but this can be changed with the arg crop_center set to False.

  Returns a list with all the shear samples. Size will be n_shear+1, because
  we also want the original sample to be included.

  Example: images=[img],n_shear=2, max_shear=0.5
  Returns: [img1=img,
           img2=img with shear of 0.25 in x,
           img3=img with shear of 0.5 in x]
  """
  # if we only have 1 image, transform into a list to work with same script
  if type(images) is not list:
    images = [images]

  # calculate the shear steps
  step_rel = max_shear / float(n_shear)  # relative to image size

  # container for sheared images
  sheared_images = []

  # get every image and apply the number of desired shears
  for img in images:
    # get rows and cols to shear
    rows, cols, depth = img.shape
    step_abs = step_rel * cols  # absolute to image size

    # shear the amount of times we want
    for i in range(0, n_shear + 1):
      # create shear matrix by mapping points
      # increases in size in each dim of the img
      size_inc = abs(int(step_abs * i))

      # shear is different if is positive or negative
      if int(step_abs * i) > 0:
        pts1 = np.float32([[0, 0], [0, rows], [cols, rows]])
        pts2 = np.float32([[0, 0], [size_inc, rows], [cols + size_inc, rows]])
        M = cv2.getAffineTransform(pts1, pts2)
        shear_img = cv2.warpAffine(
            img, M, (cols + size_inc, rows), flags=cv2.INTER_CUBIC)
      elif int(step_abs * i) < 0:
        pts1 = np.float32([[0, rows], [0, 0], [cols, rows]])
        pts2 = np.float32([[0, rows], [size_inc, 0], [cols, rows]])
        M = cv2.getAffineTransform(pts1, pts2)
        shear_img = cv2.warpAffine(
            img, M, (cols + size_inc, rows), flags=cv2.INTER_CUBIC)
      else:
        shear_img = img
      # shear using the matrix (and bicubic interpolation)

      if crop_center:
        row_start = (shear_img.shape[0] / 2) - (rows / 2)
        col_start = (shear_img.shape[1] / 2) - (cols / 2)
        shear_img = shear_img[row_start:row_start +
                              rows, col_start:col_start + cols]

      # append to sheared images container
      sheared_images.append(shear_img)

  return sheared_images


def vert_shear(images, n_shear, max_shear, crop_center=True):
  """
  Applies a vertical shear transform to every image in the list "images"
  n_shear times, with the max shear passed in the max_shear argument. By
  default, we crop the center of the images so that all images have the same
  shape, but this can be changed with the arg crop_center set to False.

  Returns a list with all the shear samples. Size will be n_shear+1, because
  we also want the original sample to be included.

  Example: images=[img],n_shear=2, max_shear=0.5
  Returns: [img1=img,
           img2=img with shear of 0.25 in y,
           img3=img with shear of 0.5 in y]
  """
  # if we only have 1 image, transform into a list to work with same script
  if type(images) is not list:
    images = [images]

  # calculate the shear steps
  step_rel = max_shear / float(n_shear)  # relative to image size

  # container for sheared images
  sheared_images = []

  # get every image and apply the number of desired shears
  for img in images:
    # get rows and cols to shear
    rows, cols, depth = img.shape
    step_abs = step_rel * rows  # absolute to image size

    # shear the amount of times we want
    for i in range(0, n_shear + 1):
      # create shear matrix by mapping points
      # increases in size in each dim of the img
      size_inc = abs(int(step_abs * i))

      # shear is different if is positive or negative
      if int(step_abs * i) > 0:
        pts1 = np.float32([[0, 0], [cols, 0], [cols, rows]])
        pts2 = np.float32([[0, 0], [cols, size_inc], [cols, rows + size_inc]])
        M = cv2.getAffineTransform(pts1, pts2)
        shear_img = cv2.warpAffine(
            img, M, (cols, rows + size_inc), flags=cv2.INTER_CUBIC)
      elif int(step_abs * i) < 0:
        pts1 = np.float32([[0, 0], [cols, 0], [0, rows]])
        pts2 = np.float32([[0, size_inc], [cols, 0], [0, rows + size_inc]])
        M = cv2.getAffineTransform(pts1, pts2)
        shear_img = cv2.warpAffine(
            img, M, (cols, rows + size_inc), flags=cv2.INTER_CUBIC)
      else:
        shear_img = img
      # shear using the matrix (and bicubic interpolation)

      if crop_center:
        row_start = (shear_img.shape[0] / 2) - (rows / 2)
        col_start = (shear_img.shape[1] / 2) - (cols / 2)
        shear_img = shear_img[row_start:row_start +
                              rows, col_start:col_start + cols]

      # append to sheared images container
      sheared_images.append(shear_img)

  return sheared_images


def horiz_flip(images):
  """
  Applies a horizontal flip to every image in the list "images"
  Returns a list with all the original and flipped samples.
  """
  # if we only have 1 image, transform into a list to work with same script
  if type(images) is not list:
    images = [images]

  # container for sheared images
  flipped_images = []

  # get every image and apply the number of desired shears
  for img in images:
    # append original and flipped images to container
    flipped_images.append(img)
    flipped_images.append(cv2.flip(img, 1))

  return flipped_images


def vert_flip(images):
  """
  Applies a vertical flip to every image in the list "images"
  Returns a list with all the original and flipped samples.
  """
  # if we only have 1 image, transform into a list to work with same script
  if type(images) is not list:
    images = [images]

  # container for sheared images
  flipped_images = []

  # get every image and apply the number of desired shears
  for img in images:
    # append original and flipped images to container
    flipped_images.append(img)
    flipped_images.append(cv2.flip(img, 0))

  return flipped_images


def gaussian_noise(images, mean, std):
  """
  Applies gaussian noise to every image in the list "images" with the desired

  Returns a list with all the original and noisy images.
  """
  # if we only have 1 image, transform into a list to work with same script
  if type(images) is not list:
    images = [images]

  # container for sheared images
  noisy_images = []

  # get every image and apply the number of desired shears
  for img in images:
    # get rows and cols apply noise to
    rows, cols, depth = img.shape

    # append original image
    noisy_images.append(img)

    # fill in the per-channel mean and std
    m = np.full((1, depth), mean)
    s = np.full((1, depth), std)

    # add noise to image
    # noisy_img = img.copy()
    noisy_img = np.zeros((rows, cols, depth), dtype=np.uint8)
    noisy_img = cv2.randn(noisy_img, m, s)
    noisy_img = img + noisy_img

    # append noisy image to container
    noisy_images.append(noisy_img)

  return noisy_images


def occlusions(images, grid_x, grid_y, selection):
  """
  Applies a grid to each image and removes a block from selection (zeroing it).
  Returns a list with all the original and occluded images.

  Example: grid_x=3,grid_y=3,selection=[1,3].

  This divides the image in 9, and returns the original image with the 1st
  and 3rd quadrant occluded (in separate images).

  Grid for this case:
  -------------------------
  |   0   |   1-< |   2   |
  -------------------------
  |   3<- |   4   |   5   |
  -------------------------
  |   6   |   7   |   8   |
  -------------------------

  """
  # if we only have 1 image, transform into a list to work with same script
  if type(images) is not list:
    images = [images]

  # container for sheared images
  occluded_images = []

  # get every image and apply the number of desired shears
  for img in images:
    # append original image
    occluded_images.append(img)

    # get rows and cols
    rows, cols, depth = img.shape

    # number of rows and cols in subsections
    x_subsec = cols / grid_x
    y_subsec = rows / grid_y

    for idx in selection:
      # select x_box and y_box
      x_box = idx % grid_x
      y_box = idx / grid_x

      # generate the mask
      mask = np.full((rows, cols), 255).astype(np.uint8)
      mask[y_box * y_subsec:(y_box + 1) * y_subsec,
           x_box * x_subsec:(x_box + 1) * x_subsec] = 0

      # occlude image
      occ_img = cv2.bitwise_and(img, img, mask=mask)

      # append occluded image to container
      occluded_images.append(occ_img)

  return occluded_images


if __name__ == "__main__":
  print("Config as app not done!")
