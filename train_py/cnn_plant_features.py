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

import cv2
import argparse
import dataset.aux_scripts.util as util
import dataset.plant_features as pf
import dataset.augment_data as ad

if __name__ == "__main__":
  parser = argparse.ArgumentParser("Feature Calculator")
  parser.add_argument(
      '--rgb',
      nargs='+',
      type=str,
      help='path of rgb image'
  )
  parser.add_argument(
      '--t',
      type=float,
      help='Threshold to generate mask',
      default=0
  )
  filter_options = ["exgr", "exr", "cive", "ndi", "hsv",
                    "gradients", "laplace", "edges", "water", "mask", "norm"]
  parser.add_argument(
      '--filters',
      type=str,
      nargs='+',
      help='Filters to apply',
      default=filter_options,
      choices=filter_options
  )
  FLAGS, unparsed = parser.parse_known_args()

  # check that I have the inputs
  if FLAGS.rgb is None:
    print("No input image. Gimme dat image")
    quit()

  print("---------------------------------------------------------------------")
  print("ARGS:")
  print("input rgb: ", FLAGS.rgb)
  print("thresh: ", FLAGS.t)
  print("filters: ", FLAGS.filters)
  print("---------------------------------------------------------------------")

  # opening rgb
  for img in FLAGS.rgb:
    print("Opening RGB image")
    rgb_img = cv2.imread(img, cv2.IMREAD_UNCHANGED)  # open image
    proper_h = 384
    proper_w = 512
    # resize to see how it works with kernels
    rgb_img = ad.resize(rgb_img, [proper_h, proper_w])
    if rgb_img is None:
      print("Rgb image doesn't exist")
      quit()

    # get exgreen
    exgr = None
    exgr_mask = None
    if "exgr" in FLAGS.filters:
      exgr = pf.exgreen(rgb_img)
      print("Exgr shape: ", exgr.shape)
      util.im_gray_plt(exgr, "exgreen (%s)" % img)
      exgr_mask = pf.thresh(exgr, FLAGS.t)
      util.im_gray_plt(exgr_mask, "exgreen mask (%s)" % img)
      # util.hist_plot(exgr,"exgreen histogram (%s)"%img)

    # get cive
    if "cive" in FLAGS.filters:
      c = pf.cive(rgb_img)
      print("cive shape: ", c.shape)
      util.im_gray_plt(c, "cive (%s)" % img)
      c_mask = pf.thresh(c, FLAGS.t)
      util.im_gray_plt(c_mask, "cive mask (%s)" % img)
      # util.hist_plot(c,"cive histogram (%s)"%img)

    # get exred
    if "exr" in FLAGS.filters:
      exr = pf.exred(rgb_img)
      print("Exred shape: ", exr.shape)
      util.im_gray_plt(exr, "exred (%s)" % img)
      exr_mask = pf.thresh(exr, FLAGS.t)
      util.im_gray_plt(exr_mask, "exred mask (%s)" % img)

    # get ndi
    if "ndi" in FLAGS.filters:
      n = pf.ndi(rgb_img)
      print("NDI shape: ", n.shape)
      util.im_gray_plt(n, "ndi (%s)" % img)
      ndi_mask = pf.thresh(n, FLAGS.t)
      util.im_gray_plt(ndi_mask, "ndi mask (%s)" % img)

    # get hsv
    if "hsv" in FLAGS.filters:
      h = pf.hsv(rgb_img)
      print("HSV shape: ", h.shape)
      # threshold hue
      util.im_gray_plt(h[:, :, 0], "hsv hue (%s)" % img)
      h_mask = pf.thresh(h[:, :, 0], FLAGS.t)
      util.im_gray_plt(h_mask, "hsv hue mask (%s)" % img)

    if "gradients" in FLAGS.filters:
      # x gradient
      if exgr is None:
        exgr = pf.exgreen(rgb_img)
      g = pf.gradients(exgr, 'x')
      print("Gradient x shape: ", g.shape)
      util.im_gray_plt(g, "Gradient mask x (%s)" % img)
      # y gradient
      g = pf.gradients(exgr, 'y')
      print("Gradient y shape: ", g.shape)
      util.im_gray_plt(g, "Gradient mask y (%s)" % img)

    if "laplace" in FLAGS.filters:
      if exgr is None:
        exgr = pf.exgreen(rgb_img)
      lplc = pf.laplacian(exgr)
      print("Laplacian shape: ", lplc.shape)
      util.im_gray_plt(lplc, "laplacian mask (%s)" % img)

    if "edges" in FLAGS.filters:
      if exgr is None:
        exgr = pf.exgreen(rgb_img)
      e = pf.edges(exgr)
      print("Edge shape: ", e.shape)
      util.im_gray_plt(e, "edges mask (%s)" % img)

    if "water" in FLAGS.filters:
      if exgr_mask is None:
        exgr = pf.exgreen(rgb_img)
        exgr_mask = pf.thresh(exgr, FLAGS.t)
      w = pf.watershed(rgb_img, exgr, exgr_mask)
      print("Watershed shape: ", w.shape)
      util.im_gray_plt(w, "watershed (%s)" % img)

    if "mask" in FLAGS.filters:
      if exgr_mask is None:
        exgr = pf.exgreen(rgb_img)
        exgr_mask = pf.thresh(exgr, FLAGS.t)
      m = pf.mask_multidim(rgb_img, exgr_mask)
      print("m shape: ", m.shape)
      util.im_plt(m, "watershed (%s)" % img)
      mgray = pf.mask_multidim(exgr, exgr_mask)
      print("mgray shape: ", mgray.shape)
      util.im_gray_plt(mgray, "watershed (%s)" % img)

    if "norm" in FLAGS.filters:
      # print("rgb_img maxes: ",rgb_img[:,:,0].max(),rgb_img[:,:,1].max(),rgb_img[:,:,2].max())
      # print("rgb_img mins: ",rgb_img[:,:,0].min(),rgb_img[:,:,1].min(),rgb_img[:,:,2].min())
      n = pf.chanelwise_norm(rgb_img)
      print("n shape: ", n.shape)
      util.im_plt(n, "normalized per channel (%s)" % img)

  # block thread until images are done
  util.im_block()
