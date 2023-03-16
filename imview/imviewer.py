#! /usr/bin/env python

#David Shean
#dshean@gmail.com

#Image viewer built on matplotlib

#Todo
#Clean up overlay handling
#pyproj or geolib to go from projected coord to lat/lon
#Incorporate cartopy support to print lat/lon tick labels
#Put bma into ndarray, use figure number with format_coord - display all values for multiband image

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox
from osgeo import gdal, ogr, osr

from pygeotools.lib import iolib, malib, geolib, timelib, warplib

from imview.lib import pltlib

#Global variable holding array under cursor
#Note: A lot of this functionality is now possible with matplotlib widget or more modern packages
gbma = None

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
            print(out)
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

def bma_fig(fig, bma, cmap='cpt_rainbow', clim=None, clim_perc=(2,98), bg=None, bg_perc=(2,98), n_subplt=1, subplt=1, label=None, title=None, contour_int=None, contour_fn=None, alpha=0.5, ticks=False, scalebar=None, ds=None, shp=None, imshow_kwargs={'interpolation':'none'}, cbar_kwargs={'orientation':'vertical'}, **kwargs):
    #We don't use the kwargs, just there to save parsing in main
    
    if clim is None:
        clim = pltlib.get_clim(bma, clim_perc=clim_perc)

    print("Colorbar limits: %0.3f %0.3f" % (clim[0], clim[1]))

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

    #Set appropriate nodata value color
    cmap_name = cmap
    cmap = pltlib.cmap_setndv(cmap_name)
    
    #ax.set_title("Band %i" % subplt, fontsize=10)
    if title is not None:
        ax.set_title(title)

    #If a background image is provided, plot it first
    if bg is not None:
        #Note, alpha=1 is opaque, 0 completely transparent
        #alpha = 0.6
        bg_perc = (4,96)
        bg_alpha = 1.0
        bg_clim = malib.calcperc(bg, bg_perc)
        #bg_clim = (1, 255)
        bg_cmap_name = 'gray'
        bg_cmap = pltlib.cmap_setndv(bg_cmap_name, cmap_name)
        #bg_cmap = plt.get_cmap(bg_cmap_name)
        #if 'inferno' in cmap_name:
        #    bg_cmap.set_bad('0.5', alpha=1)
        #else:
        #    bg_cmap.set_bad('k', alpha=1)
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
        #xres = geolib.get_res(ds)[0]
        xres = gt[1]
    else:
        pltlib.hide_ticks(ax)
    #This forces the black line outlining the image subplot to snap to the actual image dimensions
    #depreciated in 2.2
    #ax.set_adjustable('box-forced')

    if cbar_kwargs:
        #Should set the format based on dtype of input data 
        #cbar_kwargs['format'] = '%i'
        #cbar_kwargs['format'] = '%0.1f'
        #cbar_kwargs['orientation'] = 'horizontal'

        #Determine whether we need to add extend triangles to colorbar
        cbar_kwargs['extend'] = pltlib.get_cbar_extend(bma, clim)

        #Add the colorbar to the axes
        cbar = pltlib.add_cbar(ax, imgplot, label=label, cbar_kwargs=cbar_kwargs)
   
    #Plot contours every contour_int interval and update colorbar appropriately
    if contour_int is not None:
        if contour_fn is not None:
            contour_bma = iolib.fn_getma(contour_fn)
            contour_bma_clim = malib.calcperc(contour_bma)
        else:
            contour_bma = bma
            contour_bma_clim = clim

        #PIG bed ridge contours
        #bma_clim = (-1300, -300)
        #Jak front shear margin contours
        #bma_clim = (2000, 4000)
        contour_bma_clim = (100, 250)
        cstart = int(np.floor(contour_bma_clim[0] / contour_int)) * contour_int 
        cend = int(np.ceil(contour_bma_clim[1] / contour_int)) * contour_int

        #Turn off dashed negative (beds are below sea level)
        #matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

        clvl = np.arange(cstart, cend+1, contour_int)
        contour_prop = {'levels':clvl, 'linestyle':'-', 'linewidths':0.5, 'alpha':1.0}
        #contours = ax.contour(contour_bma, colors='k', **contour_prop)
        #contour_cmap = 'gray'
        contour_cmap = 'gray_r'
        #This prevents white contours
        contour_cmap_clim = (0, contour_bma_clim[-1])
        contours = ax.contour(contour_bma, cmap=contour_cmap, vmin=contour_cmap_clim[0], \
                vmax=contour_cmap_clim[-1], **contour_prop)

        #Add labels
        ax.clabel(contours, inline=True, inline_spacing=0, fontsize=4, fmt='%i')

        #Update the cbar with contour locations
        #cbar.add_lines(contours)
        #cbar.set_ticks(contours.levels)

    #Plot shape overlay, moved code to pltlib
    if shp is not None:
        pltlib.shp_overlay(ax, ds, shp, gt=gt, color='k')

    if scalebar:
        scale_ticks(ax, ds)
        sb_loc = pltlib.best_scalebar_location(bma)
        #Force scalebar position
        #sb_loc = 'lower right'
        pltlib.add_scalebar(ax, xres, location=sb_loc)
        if not ticks:
            pltlib.hide_ticks(ax)

    #Set up interactive display
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
        return iolib.ds_getma_sub(src_ds, bn)

def getparser():
    #Generate list of valid mpl colormaps
    maps=[m for m in plt.colormaps()]
    #maps=[m for m in plt.cm.datad if not m.endswith("_r")]
    maps.sort()

    #Parse input arguments
    parser = argparse.ArgumentParser(description='A lightweight matplotlib image viewer')
    #parser.add_argument('-cmap', default=None, choices=maps, help='set colormap type')
    parser.add_argument('-cmap', default=None, help='set colormap name (see https://matplotlib.org/examples/color/colormaps_reference.html)')
    #Check to make sure these are ordered correctly
    parser.add_argument('-clim', nargs=2, type=float, default=None, help='set colormap limits (min max)')
    parser.add_argument('-clim_perc', nargs=2, type=float, default=(2.0, 98.0), \
            help='set colormap percentile limits (min max)')
    parser.add_argument('-coord', default=None, choices=['None', 'proj', 'latlon'], \
            help='set coordinate label type')
    parser.add_argument('-contour_fn', default=None, \
            help='Filename of raster to use for contour. Default is input filename')
    parser.add_argument('-contour_clim', nargs=2, type=float, default=None, help='set contour limits')
    parser.add_argument('-contour_int', type=float, default=None, help='set contour interval')
    parser.add_argument('-label', type=str, default=None, help='colorbar label')
    parser.add_argument('-full', action='store_true', help='do not subsample for display') 
    #Eventually, allow for of filename specification
    parser.add_argument('-of', default=None, choices=['pdf', 'ps', 'jpg', 'png', 'tif'], \
            help='save output to specified file type') 
    #Note: should bind this to -of above
    parser.add_argument('-dpi', type=float, default=None, help='specify output dpi') 
    parser.add_argument('-outsize', nargs=2, type=float, default=None, \
            help='specify output dimensions in inches (w h)') 
    parser.add_argument('-overlay', default=None, help='specify basemap for overlay')
    parser.add_argument('-shp', default=None, help='specify shapefile for overlay')
    parser.add_argument('-alpha', type=float, default=0.5, \
            help='Overlay transparency (0 is transparent, 1 opaque)')
    parser.add_argument('-link', action='store_true', help='share axes for all input images')
    parser.add_argument('-no_cbar', action='store_true', help='no colorbar')
    parser.add_argument('-ticks', action='store_true', help='display ticks')
    parser.add_argument('-scalebar',action='store_true', help='Show scalebar')
    parser.add_argument('-title', type=str, default=None, help='Specify title, ts=timestamp, fn=filename, or user-specified string"')
    parser.add_argument('-invert', action='store_true', help='Multiply values by -1')
    parser.add_argument('filelist', nargs='+', help='input filenames (img1.tif img2.tif...)')
    return parser

def main():
    parser = getparser()
    #Create dictionary of arguments
    args = vars(parser.parse_args())
    
    #Want to enable -full when -of is specified, probably a fancy way to do this with argparse
    if args['of']:
        args['full'] = True

    args['imshow_kwargs'] = pltlib.imshow_kwargs 

    #Need to implement better extent handling for link and overlay
    #Can use warplib extent parsing
    extent = 'first'
    #extent = 'union'

    #Should accept 'ts' or 'fn' or string here, default is 'ts'
    #Can also accept list for subplots
    title = args['title']

    if args['link']:
        fig = plt.figure(0)
        n_ax = len(args['filelist'])
        src_ds_list = [gdal.Open(fn) for fn in args['filelist']]
        t_srs = geolib.get_ds_srs(src_ds_list[0])
        res_stats = geolib.get_res_stats(src_ds_list, t_srs=t_srs)
        #Use min res
        res = res_stats[0]
        #Use max res
        #res = res_stats[1]
        extent = 'intersection'
        extent = geolib.ds_geom_union_extent(src_ds_list, t_srs=t_srs)
        #extent = geolib.ds_geom_intersection_extent(src_ds_list, t_srs=t_srs)
        #print(res, extent)
        
    for n,fn in enumerate(args['filelist']):
        if not iolib.fn_check(fn):
            print('Unable to open input file: %s' % fn)
            continue

        if title == 'ts':
            ts = timelib.fn_getdatetime_list(fn) 

            if ts:
                print("Timestamp list: ", ts)
                if len(ts) == 1:
                    args['title'] = ts[0].date()
                elif len(ts) > 1:
                    args['title'] = "%s to %s" % (ts[0].date(), ts[1].date())
            else:
                print("Unable to extract timestamp")
                args['title'] = None
        elif title == 'fn':
            args['title'] = fn
        
        #Note: this won't work if img1 has 1 band and img2 has 3 bands
        #Hack for now
        if not args['link']:
            fig = plt.figure(n)
            n_ax = 1
        
        #fig.set_facecolor('black')
        fig.set_facecolor('white')
        fig.canvas.manager.set_window_title(os.path.split(fn)[1])
        #fig.suptitle(os.path.split(fn)[1], fontsize=10)

        if args['overlay']:
            #Should automatically search for shaded relief with same base fn
            #bg_fn = os.path.splitext(fn)[0]+'_hs_az315.tif'
            #Clip/warp background dataset to match overlay dataset 
            src_ds, bg_ds = warplib.memwarp_multi_fn([fn, args['overlay']], extent=extent, res='max')
            #Want to load up the unique bg array for each input
            args['bg'] = get_bma(bg_ds, 1, args['full'])
        else:
            src_ds = gdal.Open(fn, gdal.GA_ReadOnly)
            if args['link']:
                src_ds = warplib.memwarp_multi([src_ds,], res=res, extent=extent, t_srs=t_srs)[0]

        args['cbar_kwargs'] = pltlib.cbar_kwargs
        if args['no_cbar']: 
            args['cbar_kwargs'] = None 

        nbands = src_ds.RasterCount
        b = src_ds.GetRasterBand(1)
        dt = gdal.GetDataTypeName(b.DataType)
        #Eventually, check dt of each band
        print("%s (%i bands)" % (fn, nbands))
        #Singleband raster
        if (nbands == 1):
            if args['cmap'] is None:
                #Special case to handle ASP float32 grayscale data
                if '-L_sub' in fn or '-R_sub' in fn:
                    args['cmap'] = 'gray'
                elif '-D_sub' in fn:
                    args['cmap'] = 'cpt_rainbow'
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
            bma = get_bma(src_ds, 1, args['full'])   
            if args['invert']:
                bma *= -1
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
                
            print()
            print("Saving output figure:")
            print("Filename: ", outf)
            print("Size (in): ", args['outsize'])
            print("DPI (px/in): ", args['dpi'])
            print("Input dimensions (px): ", bma.shape[::-1])
            print("Output dimensions (px): ", tuple(np.array(args['outsize'])*args['dpi']))
            print()

            fig.set_size_inches(args['outsize'])
            #fig.set_size_inches(54.427, 71.87)
            #fig.set_size_inches(40, 87)
            fig.savefig(outf, dpi=args['dpi'], bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor(), edgecolor='none')
            #fig.savefig(outf, dpi=args['dpi'], facecolor=fig.get_facecolor(), edgecolor='none')
    #Show the plot - want to show all at once
    if not args['of']: 
        plt.show()

if __name__ == '__main__':
    main()
