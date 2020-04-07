#!/usr/bin/env python
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2018
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Daniel DeTone (ddetone)
#                       Tomasz Malisiewicz (tmalisiewicz)
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

import argparse
import glob
import numpy as np
import os
import time

import cv2
import torch
import joblib
import scipy

from model import SuperPointFrontend
from util import read_image

class SuperPointBackend(object):

  def __init__(self, clusterer, metrics):
    self.clusterer = clusterer
    self.metrics = getattr(scipy.spatial.distance, metrics)

  def predict(self, descs):
    histogram = np.zeros(self.clusterer.n_clusters)
    clusters = self.clusterer.predict(descs)
    for i in clusters:
      histogram[i] += 1
    return histogram

  def compare(self, x, y):
    hx = self.predict(x.transpose(1, 0))
    hy = self.predict(y.transpose(1, 0))
    return self.metrics(hx, hy)

if __name__ == '__main__':
  # Parse command line arguments.
  parser = argparse.ArgumentParser(description='PyTorch SuperPoint-Based Image Comparison.')
  parser.add_argument('base', type=str, default='base.jpg',
      help='Directory of the base image')
  parser.add_argument('query', type=str, default='query.jpg',
      help='Directory of the query image')
  parser.add_argument('--metrics', type=str, default='euclidean',
      help='Metrics of histogram comparison')
  parser.add_argument('--weights_path', type=str, default='superpoint_v1.pth',
      help='Path to pretrained weights file (default: superpoint_v1.pth).')
  parser.add_argument('--dict_path', type=str, default='superpoint_v1.joblib',
      help='Path to pretrained dictionary file (default: superpoint_v1.joblib).')
  parser.add_argument('--H', type=int, default=120,
      help='Input image height (default: 120).')
  parser.add_argument('--W', type=int, default=160,
      help='Input image width (default:160).')
  parser.add_argument('--nms_dist', type=int, default=4,
      help='Non Maximum Suppression (NMS) distance (default: 4).')
  parser.add_argument('--conf_thresh', type=float, default=0.015,
      help='Detector confidence threshold (default: 0.015).')
  parser.add_argument('--nn_thresh', type=float, default=0.7,
      help='Descriptor matching threshold (default: 0.7).')
  parser.add_argument('--cuda', action='store_true',
      help='Use cuda GPU to speed up network processing speed (default: False)')
  opt = parser.parse_args()
  print(opt)

  print('==> Loading pre-trained network.')
  # This class runs the SuperPoint network and processes its outputs.
  fe = SuperPointFrontend(weights_path=opt.weights_path,
                          nms_dist=opt.nms_dist,
                          conf_thresh=opt.conf_thresh,
                          nn_thresh=opt.nn_thresh,
                          cuda=opt.cuda)
  print('==> Successfully loaded pre-trained network.')

  print('==> Loading pre-trained dictionary.')
  dictionary = joblib.load(opt.dict_path)
  be = SuperPointBackend(dictionary, opt.metrics)
  print('==> Successfully loaded pre-trained dictionary.')

  # This class runs the SuperPoint network and processes its outputs.
  print('==> Running Comparison.')

  # Get the images in grayscale
  img_x = read_image(opt.base, (opt.H, opt.W))
  img_y = read_image(opt.query, (opt.H, opt.W))

  # Get points and descriptors.
  pts_x, desc_x, heatmap_x = fe.run(img_x)
  pts_y, desc_y, heatmap_y = fe.run(img_y)

  # Compare the descriptors
  dist = be.compare(desc_x, desc_y)
  print("Distance: {}".format(dist))

  print('==> Finshed Demo.')
