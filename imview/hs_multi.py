#! /usr/bin/env python

#Create weight rasters needed for multi-directional hillshade

import os, sys

import numpy as np
import gdal

from pygeotools.lib import iolib

az_list = (225, 270, 315, 360)
aspect_fn = sys.argv[1]
aspect_ds = gdal.Open(aspect_fn)
aspect = iolib.ds_getma(aspect_ds)

for az in az_list:
    w_fn = os.path.splitext(aspect_fn)[0]+'_w%i.tif' % az
    w = np.sin(np.radians(aspect - az))**2
    iolib.writeGTiff(w, w_fn, aspect_ds)
