#! /usr/bin/env python
"""
Create RGB raster from an input 1-band raster
Can be used to create color shaded relief maps

Note: currently uses imagemagick command-line for composite, should move to matplotlib

"""
#To do:
#Might want to create as new function in imview
#Modify to accept clim and hs filename as command line inputs
#Do trim tight with colorbar output
#Proper ndv handling

import os
import argparse
import subprocess
import shutil

import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal

from pygeotools.lib import iolib
from pygeotools.lib import malib

from imview.lib import gmtColormap
cpt_rainbow = gmtColormap.get_rainbow()
plt.register_cmap(cmap=cpt_rainbow)
cpt_rainbow_r = gmtColormap.get_rainbow(rev=True)
plt.register_cmap(cmap=cpt_rainbow_r)

def make_kmz(fn):
    print("Generating kmz")
    kmz_fn = os.path.splitext(fn)[0]+'.kmz'
    #Note, only PNG format supports transparency at this point
    cmd = ['gdal_translate', '-of', 'KMLSUPEROVERLAY', '-co', 'FORMAT=PNG', fn, kmz_fn]
    print(' '.join(cmd))
    subprocess.call(cmd, shell=False)

def getparser():
    parser = argparse.ArgumentParser(description="Create color RGB image from input 1-band raster values")
    parser.add_argument('-cmap', default='cpt_rainbow', help='Matplotlib color ramp')
    parser.add_argument('-clim', nargs=2, type=float, default=None, help='Color ramp limits (min max)')
    parser.add_argument('-hs_overlay', action='store_true', help='Overlay on shaded relief map')
    parser.add_argument('-kmz', action='store_true', help='Output kmz for Google Earth')
    parser.add_argument('-alpha', type=float, default=0.6, help='Opacity for color overlay')
    parser.add_argument('fn', type=str, help='Input raster filename')
    return parser
    
def main():
    parser = getparser()
    args = parser.parse_args()
    hs_overlay = args.hs_overlay
    kmz = args.kmz
    opacity = args.alpha
    cmap = args.cmap

    fn = args.fn
    print fn
    ds = gdal.Open(fn) 
    b = ds.GetRasterBand(1)
    ndv = iolib.get_ndv_b(b)

    print("Loading input raster")
    a = iolib.b_getma(b)

    clim = args.clim
    if clim is None:
        clim = malib.calcperc(a, (2, 98))

    print("Generating color ramp")
    cramp_fn = os.path.splitext(fn)[0]+'_ramp.txt'
    ncolors = 21 
    csteps = np.linspace(0, 1, ncolors)
    cm = plt.get_cmap(cmap)
    #Compute raster values between specified min/max
    vals = np.linspace(clim[0], clim[1], ncolors)
    #Compute rgba for these values on the given color ramp
    cvals = cm(csteps, bytes=True)    
    #Combine into single array
    cramp = np.vstack((vals, cvals.T)).T
    #Set alpha to desired transparency
    cramp[:,-1] = opacity * 255
    header = '#val r g b a'
    footer = 'nv %s %s %s 0' % (ndv, ndv, ndv)
    np.savetxt(cramp_fn, cramp, fmt='%f %i %i %i %i', header=header, footer=footer, comments='')

    print("Generating gdaldem color-relief tif")
    color_fn = os.path.splitext(fn)[0]+'_color.tif'
    if not os.path.exists(color_fn):
        #cmd = 'gdaldem color-relief -nearest_color_entry -alpha %s %s %s' % (fn, cramp_fn, color_fn)
        cmd = ['gdaldem', 'color-relief', '-alpha']
        cmd.extend(iolib.gdal_opt_co)
        cmd.extend([fn, cramp_fn, color_fn])
        print(' '.join(cmd))
        subprocess.call(cmd, shell=False)

    if kmz:
        make_kmz(color_fn)

    if hs_overlay:
        print("Generating shaded relief")
        hs_fn = os.path.splitext(fn)[0]+'_hs_az315.tif'
        #Check to see if file exists, or if provided as input
        if not os.path.exists(hs_fn):
            cmd = ['gdaldem', 'hillshade']
            #cmd.extend('-compute_edges')
            cmd.extend(iolib.gdal_opt_co)
            cmd.extend([fn, hs_fn])
            print(' '.join(cmd))
            subprocess.call(cmd, shell=False)

        print("Loading shaded relief and calculating percentile stretch")
        hs = iolib.fn_getma(hs_fn)
        hs_clim = malib.calcperc(hs, (1, 99))
        #Since imagemagick was compiled with quantum depth 16, need to scale levels
        hs_clim = (hs_clim[0]*65535/255., hs_clim[1]*65535/255.)

        print("Generating color composite shaded relief")
        overlay_fn = os.path.splitext(color_fn)[0]+'_hs.tif'
        if not os.path.exists(overlay_fn):
            #Can also try hsvmerge.py
            #cmd = 'composite %s %s -dissolve "%i" %s' % (color_fn, hs_fn, opacity*100, overlay_fn)
            #This uses imagemagick composite function
            #For some reason, this level adjustment is not working
            #cmd = ['convert', hs_fn, color_fn, '-compose', 'dissolve', \
            cmd = ['convert', hs_fn, '-level', '%i,%i' % hs_clim, color_fn, '-compose', 'dissolve', \
            '-define', 'compose:args=%i' % int(opacity*100), '-composite', '-compress', 'LZW', overlay_fn]
            #cmd = ['composite', color_fn, hs_fn, '-dissolve', str(int(opacity*100)), '-compress', 'LZW', overlay_fn]
            print(' '.join(cmd))
            subprocess.call(cmd, shell=False)

            print("Updating georeferencing metadata")
            out_ndv = 0
            overlay_ds = gdal.Open(overlay_fn, gdal.GA_Update)
            overlay_ds.SetProjection(ds.GetProjection())
            overlay_ds.SetGeoTransform(ds.GetGeoTransform())
            for n in range(overlay_ds.RasterCount):
                overlay_ds.GetRasterBand(n+1).SetNoDataValue(out_ndv)
            overlay_ds = None

            #Rewrite with blocks and LZW-compression
            print("Creating tiled and compressed version")
            tmp_fn = '/tmp/temp_%s.tif' % os.getpid()
            cmd = ['gdal_translate',]
            cmd.extend(iolib.gdal_opt_co)
            cmd.extend((overlay_fn, tmp_fn))
            print(' '.join(cmd))
            subprocess.call(cmd, shell=False)
            shutil.move(tmp_fn, overlay_fn)

        if not os.path.exists(overlay_fn+'.ovr'):
            print("Generating overviews")
            cmd = ['gdaladdo', '-ro', '-r', 'average', '--config', \
            'COMPRESS_OVERVIEW', 'LZW', '--config', 'BIGTIFF_OVERVIEW', 'YES', \
            overlay_fn, '2', '4', '8', '16', '32', '64']
            print(' '.join(cmd))
            subprocess.call(cmd, shell=False)

        if kmz:
            make_kmz(overlay_fn)

if __name__ == '__main__':
    main()
