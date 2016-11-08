#! /usr/bin/env python

#David Shean
#dshean@gmail.com

#Image viewer built on matplotlib
#Needs to be cleaned up, but should work out of the box

#Todo
#access from malib
#Clean up overlay handling
#pyproj or geolib to go from projected coord to lat/lon
#Incorporate basemap support to print lat/lon, add scalebar, etc
#Put bma into ndarray, use figure number with format_coord - display all values for multiband image

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox
from osgeo import gdal, ogr, osr

from pygeotools.lib import iolib
from pygeotools.lib import malib
from pygeotools.lib import geolib
from pygeotools.lib import timelib
from pygeotools.lib import warplib

from lib import pltlib

#These are matplotlib 2.0 colormaps
#To use:
#cmap=cmaps.inferno
#cmap=cmaps.viridis
import colormaps as cmaps
plt.register_cmap(name='inferno', cmap=cmaps.inferno)
plt.register_cmap(name='inferno_r', cmap=cmaps.inferno_r)
plt.register_cmap(name='magma', cmap=cmaps.magma)
plt.register_cmap(name='plasma', cmap=cmaps.plasma)
plt.register_cmap(name='viridis', cmap=cmaps.viridis)

import gmtColormap
cpt_rainbow = gmtColormap.get_rainbow()
plt.register_cmap(cmap=cpt_rainbow)
cpt_rainbow_r = gmtColormap.get_rainbow(rev=True) 
plt.register_cmap(cmap=cpt_rainbow_r)

#Global variable holding array under cursor
gbma = None

#Use to specify a constant set of contours
#c_fn = 'raster.tif'
#bma_c = iolib.fn_getma(c_fn)

#This adds the z-value to cursor coordinate info
def format_coord(x, y):
    x = int(x + 0.5)
    y = int(y + 0.5)

    #Convert x and y into mapped coordinates using geotransform
    #Also, show WGS84
    #Need geotransform for this - implement the geoma class containing fn and ds with bma

    try:
        coord_str = "%s @ [%4i, %4i]" % (gbma[y,x], x, y)
        #if ggt:
        #    mx, my = geolib.pixelToMap(x, y, ggt)
        #    coord_str = "%0.6f %0.6 %s @ [%4i, %4i]" % (mx, my, gbma[y,x], x, y)
        return coord_str 
    except IndexError:
        return ""

#When mouse enters a new axis, set global ma so format_coord(x,y) shows correct values
def enter_axis(event):
    ax = event.inaxes
    #Need to make sure we're only updated over valid image axes 
    if len(ax.images) > 0:
        global gbma
        gbma = ax.images[-1].get_array()

#Click to print x, y, z coordinates
def onclick(event):
    if event.inaxes != None:
        xpx = event.xdata
        ypx = event.ydata
        try:
            out = (xpx, ypx, gbma[ypx, xpx])
            if ggt is not None:
                #Note matplotlib (0,0) is lower left
                #gdal (0,0) is upper left
                mx, my = geolib.pixelToMap(xpx, ypx, ggt)
                out = (xpx, ypx), (mx, my, gbma[ypx, xpx])
            print out
        except IndexError:
            pass

def scale_ticks(ax, ds, latlon=False):
    gt = ds.GetGeoTransform()
    if gt != (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
        x_ticks = np.array(ax.get_xticks())
        #x_tick_labels = np.around(gt[0]+x_ticks*gt[1], decimals=-1).astype(int)
        x_tick_labels = np.around(gt[0]+x_ticks*gt[1]).astype(int)
        y_ticks = np.array(ax.get_yticks())
        #y_tick_labels = np.around(gt[3]+y_ticks*gt[5], decimals=-1).astype(int)
        y_tick_labels = np.around(gt[3]+y_ticks*gt[5]).astype(int)
        ax.set_xlabel('X coord (meters)')
        ax.set_ylabel('Y coord (meters)')
        if latlon: 
            #Want to automatically determine number of decimals based on extent here
            srs = geolib.get_ds_srs(ds)
            ct=osr.CoordinateTransformation(srs, geolib.wgs_srs)
            lbl = np.array(ct.TransformPoints(zip(x_tick_labels, y_tick_labels)))
            x_tick_labels = np.round(lbl[:,0], decimals=3)
            x_tick_labels = ['%.3f' % a for a in x_tick_labels]
            y_tick_labels = np.round(lbl[:,1], decimals=3)
            y_tick_labels = ['%.3f' % a for a in y_tick_labels]
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
        ax.set_xticklabels(x_tick_labels, size='8')
        ax.set_yticklabels(y_tick_labels, size='8')

def ndanimate(a):
    import matplotlib.animation as animation
    a = malib.checkma(a)
    #Compute constant scale
    clim = malib.calcperc(a)
    label = 'Elev. Diff. (m)'
    fig = plt.figure()
    ims = []
    for i in a:
        #cmap = 'gist_rainbow_r'
        cmap = 'cpt_rainbow'
        im = plt.imshow(i, cmap=cmap, clim=clim)
        im.axes.patch.set_facecolor('black')
        #cbar = fig.colorbar(im, extend='both', shrink=0.5)
        #cbar.set_label(label)
        ims.append([im])
    an = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
    plt.show()
    return an

"""
#Note: scalebar loc:
best  0
upper right   1
upper left    2
lower left    3
lower right   4
right     5
center left   6
center right  7
lower center  8
upper center  9
center    10
"""
class AnchoredScaleBar(AnchoredOffsetbox):
    def __init__(self, transform, sizex=0, sizey=0, labelx=None, labely=None, loc=3,
                 pad=0.5, borderpad=0.5, sep=5, prop=None, **kwargs):
        """
        Draw a horizontal and/or vertical  bar with the size in data coordinate
        of the give axes. A label will be drawn underneath (center-aligned).

        - transform : the coordinate frame (typically axes.transData)
        - sizex,sizey : width of x,y bar, in data units. 0 to omit
        - labelx,labely : labels for x,y bars; None to omit
        - loc : position in containing axes
        - pad, borderpad : padding, in fraction of the legend font size (or prop)
        - sep : separation between labels and bars in points.
        - **kwargs : additional arguments passed to base class constructor
        """
        from matplotlib.patches import Rectangle
        from matplotlib.offsetbox import AuxTransformBox, VPacker, HPacker, TextArea, DrawingArea
        bars = AuxTransformBox(transform)
        if sizex:
            bars.add_artist(Rectangle((0,0), sizex, 0, fc="none", linewidth=2))
        if sizey:
            bars.add_artist(Rectangle((0,0), 0, sizey, fc="none",linewidth=2))
 
        if sizex and labelx:
            bars = VPacker(children=[bars, TextArea(labelx, minimumdescent=False)],
                           align="center", pad=0, sep=sep)
        if sizey and labely:
            bars = HPacker(children=[TextArea(labely), bars],
                            align="center", pad=0, sep=sep)
        if not 'frameon' in kwargs.keys():
            kwargs['frameon']=False
 
        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=bars, prop=prop, **kwargs)
        
 
def add_scalebar(ax, matchx=True, matchy=True, **kwargs):
    """ Add scalebars to axes

    Adds a set of scale bars to *ax*, matching the size to the ticks of the plot
    and optionally hiding the x and y axes

    - ax : the axis to attach ticks to
    - matchx,matchy : if True, set size of scale bars to spacing between ticks
                    if False, size should be set using sizex and sizey params
    - **kwargs : additional arguments passed to AnchoredScaleBars

    Returns created scalebar object
    """

    #Note: this is a hack using labels
    #Should properly use the ax.transData information
    def f(axis):
        loc = axis.get_majorticklocs()
        lbl = axis.get_majorticklabels()
        dl_ax = abs(loc[1] - loc[0])
        print lbl[1]
        dl = abs(float(lbl[1].get_text()) - float(lbl[0].get_text()))
        #Adjust to nice round numbers here
        scaling = dl/float(dl_ax)
        target = 500
        dl_ax = target/scaling 
        dl = target 
        return dl_ax, dl 

    #Should get unit from srs
    #unit = ' m'
    unit = ' km'

    if matchx:
        l = f(ax.xaxis)
        kwargs['sizex'] = l[0]
        #kwargs['labelx'] = str(l[1]) + unit 
        kwargs['labelx'] = str(l[1]/1000.) + unit 
    if matchy:
        l = f(ax.yaxis)
        kwargs['sizey'] = l[0]
        kwargs['labely'] = str(l[1]) + unit
        
    sb = AnchoredScaleBar(ax.transData, **kwargs)
    ax.add_artist(sb)
 
    return sb

#Note: can probably handle the cmap and clim in imshow_kwargs
#def bma_fig(bma, cmap='gray', clim=malib.calcperc(bma,perc)):
#def bma_fig(fig, bma, cmap='gist_rainbow_r', clim=None, bg=None, n_subplt=1, subplt=1, label=None, cint=None, **imshow_kwargs):
def bma_fig(fig, bma, cmap='cpt_rainbow', clim=None, clim_perc=(2,98), bg=None, bg_perc=(2,98), n_subplt=1, subplt=1, label=None, title=None, cint=None, alpha=0.5, ticks=False, scalebar=None, ds=None, shp=None, imshow_kwargs={'interpolation':'nearest'}, cbar_kwargs={'extend':'both', 'orientation':'vertical', 'shrink':0.7, 'fraction':0.12, 'pad':0.02}, **kwargs):
    #We don't use the kwargs, just there to save parsing in main
    
    if clim is None:
        clim = malib.calcperc(bma, clim_perc)
        #Deal with masked cases
        if clim[0] == clim[1]:
            clim = (bma.fill_value, 255)
        print "Colorbar limits (%0.1f-%0.1f%%): %0.3f %0.3f" % (clim_perc[0], clim_perc[1], clim[0], clim[1])
    else:
        print "Colorbar limits: %0.3f %0.3f" % (clim[0], clim[1])

    #Link all subplots for zoom/pan
    sharex = sharey = None
    if len(fig.get_axes()) > 0:
        sharex = sharey = fig.get_axes()[0]

    #Hack to catch situations with only 1 subplot, but a subplot number > 1
    if n_subplt == 1:
        subplt = 1

    #One row, multiple columns
    ax = fig.add_subplot(1, n_subplt, subplt, sharex=sharex, sharey=sharey)
    #This occupies the full figure
    #ax = fig.add_axes([0., 0., 1., 1., ])

    #ax.patch.set_facecolor('black')
    ax.patch.set_facecolor('white')

    cmap_name = cmap
    cmap = plt.get_cmap(cmap_name)
    #This sets the nodata background to opaque black
    if 'inferno' in cmap_name:
        #cmap.set_bad('w', alpha=1)
        cmap.set_bad('0.5', alpha=1)
    else:
        cmap.set_bad('k', alpha=1)

    #ax.set_title("Band %i" % subplt, fontsize=10)
    if title is not None:
        ax.set_title(title)

    #If a background image is provided, plot it first
    if bg is not None:
        #Note, 1 is opaque, 0 completely transparent
        #alpha = 0.6
        #bg_perc = (4,96)
        #bg_perc = (0.05, 99.95)
        #bg_perc = (1, 99)
        bg_alpha = 1.0
        #bg_alpha = 0.5 
        bg_clim = malib.calcperc(bg, bg_perc)
        bg_cmap_name = 'gray'
        bg_cmap = plt.get_cmap(bg_cmap_name)
        bg_cmap.set_bad('k', alpha=1)
        #Set the overlay bad values to completely transparent, otherwise darkens the bg
        cmap.set_bad(alpha=0)
        bgplot = ax.imshow(bg, cmap=bg_cmap, clim=bg_clim, alpha=bg_alpha)
        imgplot = ax.imshow(bma, alpha=alpha, cmap=cmap, clim=clim, **imshow_kwargs)
    else:
        imgplot = ax.imshow(bma, cmap=cmap, clim=clim, **imshow_kwargs)
 
    gt = None
    if ds is not None:
        gt = np.array(ds.GetGeoTransform())
        gt_scale_factor = min(np.array([ds.RasterYSize, ds.RasterXSize])/np.array(bma.shape,dtype=float))
        gt[1] *= gt_scale_factor
        gt[5] *= gt_scale_factor
        ds_srs = geolib.get_ds_srs(ds)
        if ticks:
            scale_ticks(ax, ds)
        else:
            pltlib.hide_ticks(ax)
    else:
        pltlib.hide_ticks(ax)
    #This forces the black line outlining the image subplot to snap to the actual image dimensions
    ax.set_adjustable('box-forced')

    cbar = True 
    if cbar:
        #Had to turn off the ax=ax for overlay to work
        #cbar = fig.colorbar(imgplot, ax=ax, extend='both', shrink=0.5) 
        #Should set the format based on dtype of input data 
        #cbar_kwargs['format'] = '%i'
        #cbar_kwargs['format'] = '%0.1f'
        #cbar_kwargs['orientation'] = 'horizontal'
        #cbar_kwargs['shrink'] = 0.8

        cbar = pltlib.add_cbar(ax, imgplot, label=label, cbar_kwargs=cbar_kwargs)
   
    #Plot contours every cint interval and update colorbar appropriately
    if cint is not None:
        if bma_c is not None:
            bma_clim = malib.calcperc(bma_c)
            #PIG bed ridge contours
            #bma_clim = (-1300, -300)
            #Jak front shear margin contours
            #bma_clim = (2000, 4000)
            cstart = int(np.floor(bma_clim[0] / cint)) * cint 
            cend = int(np.ceil(bma_clim[1] / cint)) * cint
        else:
            #cstart = int(np.floor(bma.min() / cint)) * cint 
            #cend = int(np.ceil(bma.max() / cint)) * cint
            cstart = int(np.floor(clim[0] / cint)) * cint 
            cend = int(np.ceil(clim[1] / cint)) * cint

        #Turn off dashed negative (beds are below sea level)
        #matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

        clvl = np.arange(cstart, cend+1, cint)
        #contours = ax.contour(bma_c, colors='k', levels=clvl, alpha=0.5)
        contours = ax.contour(bma_c, cmap='gray', linestyle='--', levels=clvl, alpha=1.0)

        #Update the cbar with contour locations
        cbar.add_lines(contours)
        cbar.set_ticks(contours.levels)

    #Plot shape overlay, moved code to pltlib
    if shp is not None:
        pltlib.shp_overlay(ax, ds, shp, gt=gt)

    if scalebar is not None:
        scale_ticks(ax, ds)
        if scalebar=='xy':
            add_scalebar(ax,frameon=True)
        elif scalebar=='x':
            add_scalebar(ax,matchy=False,frameon=True)
        elif scalebar=='y':
            add_scalebar(ax,matchx=False,frameon=True)
        if not ticks:
            pltlib.hide_ticks(ax)

    #imgplot.set_cmap(cmap)
    #imgplot.set_clim(clim)
  
    global gbma
    gbma = bma
    global ggt
    ggt = gt

    #Clicking on a subplot will make it active for z-coordinate display
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('axes_enter_event', enter_axis)
    
    #Add support for interactive z-value display 
    ax.format_coord = format_coord

    #Attempts to add a histogram
    #fig.add_subplot(gs[0,-1])
    #plt.hist(bma.compressed(),256)

#Wrapper for malib.gdal_getma 
def get_bma(src_ds, bn, full):
    if full:
        return iolib.ds_getma(src_ds, bn)
    else:
        return iolib.gdal_getma_sub(src_ds, bn)

def main():

    #Generate list of valid mpl colormaps
    maps=[m for m in plt.cm.datad]
    #maps=[m for m in plt.cm.datad if not m.endswith("_r")]
    maps.sort()
    maps.append('cpt_rainbow')
    maps.append('cpt_rainbow_r')
    maps.append('inferno')
    maps.append('inferno_r')
    maps.append('magma')
    maps.append('plasma')
    maps.append('viridis')

    #Parse input arguments
    parser = argparse.ArgumentParser(description='A lightweight matplotlib image viewer')
    parser.add_argument('-cmap', default=None, choices=maps, help='set colormap type')
    #Check to make sure these are ordered correctly
    parser.add_argument('-clim', nargs=2, type=float, default=None, help='set colormap limits (min max)')
    parser.add_argument('-clim_perc', nargs=2, type=float, default=(2.0, 98.0), help='set colormap percentile limits (min max)')
    parser.add_argument('-coord', default=None, choices=['None', 'proj', 'latlon'], help='set coordinate label type')
    parser.add_argument('-cint', type=float, default=None, help='set contour interval')
    parser.add_argument('-label', type=str, default=None, help='colorbar label')
    parser.add_argument('-full', action='store_true', help='do not subsample for display') 
    #Eventually, allow for of filename specification
    parser.add_argument('-of', default=None, choices=['pdf', 'ps', 'jpg', 'png', 'tif'], help='save output to specified file type') 
    #Note: should bind this to -of above
    parser.add_argument('-dpi', type=float, default=None, help='specify output dpi') 
    parser.add_argument('-outsize', nargs=2, type=float, default=None, help='specify output dimensions in inches (w h)') 
    parser.add_argument('-overlay', default=None, help='specify basemap for overlay')
    parser.add_argument('-clipped',action='store_true',help='Do not warp to match overlay')
    parser.add_argument('-shp', default=None, help='specify shapefile for overlay')
    parser.add_argument('-alpha', type=float, default=0.5, help='Overlay transparency (0 is transparent, 1 opaque)')
    parser.add_argument('-link', action='store_true', help='share axes for all input images')
    parser.add_argument('-no_cbar', action='store_true', help='no colorbar')
    parser.add_argument('-ticks', action='store_true', help='display ticks')
    parser.add_argument('-scalebar',type=str,default=None,choices=['xy','x','y'],help='Show scalebar in x and y, x, or y')
    parser.add_argument('filelist', nargs='+', help='input filenames (img1.tif img2.tif...)')

    #Create dictionary of arguments
    args = vars(parser.parse_args())
    
    #Want to enable -full when -of is specified, probably a fancy way to do this with argparse
    if args['of']:
        args['full'] = True

    #Note, imshow has many interpolation types:
    #'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 
    #'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
    #{'interpolation':'bicubic', 'aspect':'auto'}
    args['imshow_kwargs']={'interpolation':'bicubic'}

    if args['clipped'] and args['overlay'] is None:
        sys.exit("Must specify an overlay filename with option 'clipped'")

    #Set this as the background numpy array
    args['bg'] = None

    if args['shp'] is not None:
        print args['shp']

    if args['link']:
        fig = plt.figure(0)
        n_ax = len(args['filelist'])
        src_ds_list = [gdal.Open(fn) for fn in args['filelist']]
        t_srs = geolib.get_ds_srs(src_ds_list[0])
        res_stats = geolib.get_res_stats(src_ds_list, t_srs=t_srs)
        #Use min res
        res = res_stats[0]
        extent = geolib.ds_geom_union_extent(src_ds_list, t_srs=t_srs)
        #print res, extent

    for n,fn in enumerate(args['filelist']):

        if not iolib.fn_check(fn):
            print 'Unable to open input file: %s' % fn
            continue

        #Note: this won't work if img1 has 1 band and img2 has 3 bands
        #Hack for now
        if not args['link']:
            fig = plt.figure(n)
            n_ax = 1
        
        #fig.set_facecolor('black')
        fig.set_facecolor('white')
        fig.canvas.set_window_title(os.path.split(fn)[1])
        #fig.suptitle(os.path.split(fn)[1], fontsize=10)

        #Note: warplib SHOULD internally check to see if extent/resolution/projection are identical
        #This eliminates the need for a clipped flag
        #If user has already warped the background and source data 
        if args['overlay']:
            if args['clipped']: 
                src_ds = gdal.Open(fn, gdal.GA_ReadOnly)
                #Only load up the bg array once
                if args['bg'] is None:
                    #Need to check that background fn exists
                    print "%s background" % args['overlay']
                    bg_ds = gdal.Open(args['overlay'], gdal.GA_ReadOnly)
                    #Check image dimensions
                    args['bg'] = get_bma(bg_ds, 1, args['full'])
            else:
                #Clip/warp background dataset to match overlay dataset 
                #src_ds, bg_ds = warplib.memwarp_multi_fn([fn, args['overlay']], extent='union')
                src_ds, bg_ds = warplib.memwarp_multi_fn([fn, args['overlay']], extent='first')
                #src_ds, bg_ds = warplib.memwarp_multi_fn([fn, args['overlay']], res='min', extent='first')
                #Want to load up the unique bg array for each input
                args['bg'] = get_bma(bg_ds, 1, args['full'])
        else:
            src_ds = gdal.Open(fn, gdal.GA_ReadOnly)
            if args['link']:
                #Not sure why, but this still warps all linked ds, even when identical res/extent/srs
                #src_ds = warplib.warp(src_ds, res=res, extent=extent, t_srs=t_srs)
                src_ds = warplib.memwarp_multi([src_ds,], res=res, extent=extent, t_srs=t_srs)[0]

        cbar_kwargs={'extend':'both', 'orientation':'vertical', 'shrink':0.7, 'fraction':0.12, 'pad':0.02}

        nbands = src_ds.RasterCount
        b = src_ds.GetRasterBand(1)
        dt = gdal.GetDataTypeName(b.DataType)
        #Eventually, check dt of each band
        print 
        print "%s (%i bands)" % (fn, nbands)
        #Singleband raster
        if (nbands == 1):
            if args['cmap'] is None:
                #Special case to handle ASP float32 grayscale data
                if '-L_sub' in fn or '-R_sub' in fn:
                    args['cmap'] = 'gray'
                else:
                    if (dt == 'Float64') or (dt == 'Float32') or (dt == 'Int32'):
                        args['cmap'] = 'cpt_rainbow'
                    #This is for WV images
                    elif (dt == 'UInt16'):
                        args['cmap'] = 'gray'
                    elif (dt == 'Byte'):
                        args['cmap'] = 'gray'
                    else:
                        args['cmap'] = 'cpt_rainbow'
                """
                if 'count' in fn:
                    args['clim_perc'] = (0,100)
                    cbar_kwargs['extend'] = 'neither'
                    args['cmap'] = 'cpt_rainbow'
                if 'mask' in fn:
                    args['clim'] = (0, 1)
                    #Could be (0, 255)
                    #args['clim_perc'] = (0,100)
                    #Want absolute clim of 0, then perc of 100
                    cbar_kwargs['extend'] = 'neither'
                    args['cmap'] = 'gray'
                """
            args['cbar_kwargs'] = cbar_kwargs
            bma = get_bma(src_ds, 1, args['full'])   
            #Note n+1 here ensures we're assigning subplot correctly here (n is 0-relative, subplot is 1)
            bma_fig(fig, bma, n_subplt=n_ax, subplt=n+1, ds=src_ds, **args)
        #3-band raster, likely disparity map
        #This doesn't work when alpha band is present
        elif (nbands == 3) and (dt == 'Byte'):
            #For some reason, tifs are vertically flipped
            if (os.path.splitext(fn)[1] == '.tif'):
                args['imshow_kwargs']['origin'] = 'lower'
            #Use gdal dataset here instead of imread(fn)?
            imgplot = plt.imshow(plt.imread(fn), **args['imshow_kwargs'])
            pltlib.hide_ticks(imgplot.axes)
        #Handle the 3-band disparity map case here
        #elif ((dt == 'Float32') or (dt == 'Int32')):
        else: 
            if args['cmap'] is None:
                args['cmap'] = 'cpt_rainbow'
            bn = 1
            while bn <= nbands:
                bma = get_bma(src_ds, bn, args['full'])
                bma_fig(fig, bma, n_subplt=nbands, subplt=bn, ds=src_ds, **args)
                bn += 1
        #Want to be better about this else case - lazy for now
        #else:
        #    bma = get_bma(src_ds, 1, args['full'])
        #    bma_fig(fig, bma, **args)

        ts = timelib.fn_getdatetime_list(fn) 

        if ts:
            print "Timestamp list: ", ts

        """
        if len(ts) == 1:
            plt.title(ts[0].date())
        elif len(ts) == 2:
            plt.title("%s to %s" % (ts[0].date(), ts[1].date()))
        """
            
        plt.tight_layout()
        
        #Write out the file 
        #Note: make sure display is local for savefig
        if args['of']:
            outf = str(os.path.splitext(fn)[0])+'_fig.'+args['of'] 
            #outf = str(os.path.splitext(fn)[0])+'_'+str(os.path.splitext(args['overlay'])[0])+'_fig.'+args['of'] 

            #Note: need to account for colorbar (12%) and title - some percentage of axes beyond bma dimensions
            #Should specify minimum text size for output

            max_size = np.array((10.0,10.0))
            max_dpi = 300.0
            #If both outsize and dpi are specified, don't try to change, just make the figure
            if (args['outsize'] is None) and (args['dpi'] is None):
                args['dpi'] = 150.0

            #Unspecified out figure size for a given dpi
            if (args['outsize'] is None) and (args['dpi'] is not None):
                args['outsize'] = np.array(bma.shape[::-1])/args['dpi']
                if np.any(np.array(args['outsize']) > max_size):
                    args['outsize'] = max_size
            #Specified output figure size, no specified dpi 
            elif (args['outsize'] is not None) and (args['dpi'] is None):
                args['dpi'] = np.min([np.max(np.array(bma.shape[::-1])/np.array(args['outsize'])), max_dpi])
                
            print
            print "Saving output figure:"
            print "Filename: ", outf
            print "Size (in): ", args['outsize']
            print "DPI (px/in): ", args['dpi']
            print "Input dimensions (px): ", bma.shape[::-1]
            print "Output dimensions (px): ", tuple(np.array(args['outsize'])*args['dpi'])
            print

            fig.set_size_inches(args['outsize'])
            #fig.set_size_inches(54.427, 71.87)
            #fig.set_size_inches(40, 87)
            fig.savefig(outf, dpi=args['dpi'], bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor(), edgecolor='none')
    #Show the plot - want to show all at once
    if not args['of']: 
        plt.show()

if __name__ == '__main__':
    main()
