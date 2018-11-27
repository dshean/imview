#! /usr/bin/env python

#David Shean
#dshean@gmail.com

#Library of common matplotlib functions
#Should migrate much of imview functionality here, update to work with ax

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from pygeotools.lib import malib, geolib, iolib

from imview.lib import gmtColormap
cpt_rainbow = gmtColormap.get_rainbow()
plt.register_cmap(cmap=cpt_rainbow)
cpt_rainbow_r = gmtColormap.get_rainbow(rev=True) 
plt.register_cmap(cmap=cpt_rainbow_r)

#import itertools
#color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
#colors = itertools.cycle(color_list)

#Global imshow keyword arguments
#Note: matpltolib v2.0+ interpolates across masked values, disable interpolation for now
imshow_kwargs = {'interpolation':'none'}

#Global cbar keyword arguments
#cbar_kwargs={'extend':'both', 'orientation':'vertical', 'fraction':0.046, 'pad':0.04}
#cbar_kwargs={'extend':'both', 'orientation':'vertical', 'shrink':0.7, 'fraction':0.12, 'pad':0.02}
cbar_kwargs={'orientation':'vertical'}

def iv_fn(fn, full=False, **kwargs):
    ds = iolib.fn_getds(fn)
    return iv_ds(ds, full=full, **kwargs)

def iv_ds(ds, full=False, **kwargs):
    if full:
        a = iolib.ds_getma(ds)
    else:
        a = iolib.ds_getma_sub(ds)
    return iv(a, **kwargs)

def iv(a, ax=None, clim=None, clim_perc=(2,98), cmap='cpt_rainbow', label=None, title=None, \
        ds=None, res=None, hillshade=False, scalebar=True):
    """
    Quick image viewer with standardized display settings
    """
    if ax is None:
        f,ax = plt.subplots()
    ax.set_aspect('equal')
    if clim is None:
        clim = get_clim(a, clim_perc)
    cm = cmap_setndv(cmap, cmap)
    alpha=1.0
    if hillshade:
        if ds is not None:
            hs = geolib.gdaldem_mem_ds(ds, processing='hillshade', computeEdges=True, returnma=True)
            b_cm = cmap_setndv('gray', cmap)
            #Set the overlay bad values to completely transparent, otherwise darkens the bg
            cm.set_bad(alpha=0)
            bg_clim_perc = (2,98)
            bg_clim = get_clim(hs, bg_clim_perc)
            #bg_clim = (1, 255)
            bgplot = ax.imshow(hs, cmap=b_cm, clim=bg_clim)
            alpha = 0.5
    if scalebar:
        if ds is not None:
            #Get resolution at center of dataset
            ccoord = geolib.get_center(ds)
            c_srs = geolib.localortho(*ccoord)
            res = geolib.get_res(ds, c_srs)[0]
        if res is not None:
            sb_loc = best_scalebar_location(a)
            add_scalebar(ax, res, location=sb_loc)
    imgplot = ax.imshow(a, cmap=cm, clim=clim, alpha=alpha, **imshow_kwargs)
    cbar_kwargs['extend'] = get_cbar_extend(a, clim=clim)
    cbar_kwargs['format'] = get_cbar_format(a)
    cbar = add_cbar(ax, imgplot, label=label)
    hide_ticks(ax)
    if title is not None:
        ax.set_title(title)
    plt.tight_layout()
    return ax

def get_clim(a, clim_perc=(2,98)):
    """
    Computer percentile stretch for input array
    """
    #print("Colorbar limits (%0.1f-%0.1f%%)" % clim_perc)
    clim = malib.calcperc(a, clim_perc)
    if clim[0] == clim[1]:
        if clim[0] > a.fill_value:
            clim = (a.fill_value, clim[0])
        else:
            clim = (clim[0], a.fill_value)
    return clim

def get_cbar_extend(a, clim=None):
    """
    Determine whether we need to add triangles to ends of colorbar
    """
    if clim is None:
        clim = get_clim(a)
    extend = 'both'
    if a.min() >= clim[0] and a.max() <= clim[1]:
        extend = 'neither'
    elif a.min() >= clim[0] and a.max() > clim[1]:
        extend = 'max'
    elif a.min() < clim[0] and a.max() <= clim[1]:
        extend = 'min'
    return extend

def get_cbar_format(a):
    format = None
    #Check dtype
    #format = '%i'
    return format

def cmap_setndv(cmap1, cmap2=None):
    """
    Set default nodata mapping for colorbar
    """
    #Get matplotlib colormap objet
    cmap = plt.get_cmap(cmap1)
    if cmap2 is None: 
        cmap2 = cmap1
    if 'inferno' in cmap2:
        #Set nodata to opaque gray
        cmap.set_bad('0.5', alpha=1)
    else:
        #Set nodata to opaque black
        cmap.set_bad('k', alpha=1)
    return cmap

def hide_ticks(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def best_scalebar_location(a, length_pad=0.2, height_pad=0.1):
    """
    Attempt to determine best corner for scalebar based on number of unmasked pixels
    """
    a = malib.checkma(a)
    length = int(a.shape[1]*length_pad)
    height = int(a.shape[0]*height_pad)
    d = {}
    d['upper right'] = a[0:height,-length:].count()
    d['upper left'] = a[0:height,0:length].count()
    d['lower right'] = a[-height:,-length:].count()
    d['lower left'] = a[-height:,0:length].count()
    loc = min(d, key=d.get)
    return loc

def add_scalebar(ax, res, location='lower right', arr=None):
    from matplotlib_scalebar.scalebar import ScaleBar
    if arr is not None:
        location=best_scalebar_location(arr)
    sb = ScaleBar(res, location=location, border_pad=0.5)
    ax.add_artist(sb)

def add_cbar(ax, mappable, label=None, arr=None, clim=None, cbar_kwargs=cbar_kwargs, fontsize=10):
    """
    Add colorbar to axes for previously plotted mappable (output from imshow)
    
    Note that there are still issues with some backends like %matplotlib inline
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    #from mpl_toolkits.axes_grid1.colorbar import colorbar
    fig = ax.figure
    divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    cax = divider.append_axes("right", size="5%", pad="2%")
    if arr is not None and clim is not None:
        cbar_kwargs['extend'] = get_cbar_extend(arr, clim=clim)
    cbar = fig.colorbar(mappable, cax=cax, **cbar_kwargs) 
    if label is not None:
        cbar.set_label(label, size=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    #Set colorbar to be opaque, even if image is transparent
    cbar.set_alpha(1)
    #Need this to update after alpha change
    cbar.draw_all()
    return cbar

#This is old
def add_colorbar(ax, mappable, loc='center left', label=None):
    from matplotlib_colorbar.colorbar import Colorbar
    cbar = Colorbar(mappable, location=loc, border_pad=0.5)
    if label is not None:
        cbar.set_label(label)
    cbar.set_alpha(1)
    cbar.draw_all()
    ax.add_artist(cbar)

def minorticks_on(ax, x=True, y=True):
    if x:
        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    if y:
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    #ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1000))

def pad_xaxis(ax, abspad=None, percpad=0.02):
    xmin, xmax = ax.get_xlim()
    x_ptp = xmax - xmin
    if abspad is not None:
        pad = abs(abspad)
    else:
        pad = abs(x_ptp) * percpad
    xmin -= pad
    xmax += pad
    ax.set_xlim(xmin, xmax)

def fmt_date_ax(ax, minor=3):
    minorticks_on(ax)
    #months = range(1,13,minor)
    months = [4, 7, 10]
    ax.xaxis.set_minor_locator(matplotlib.dates.MonthLocator(months))
    #ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=6))
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    #Rotate and align tick labels
    fig = ax.get_figure()
    fig.autofmt_xdate()
    #ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
    #Update interactive display
    date_str = '%Y-%m-%d %H:%M'
    date_fmt = matplotlib.dates.DateFormatter(date_str)
    ax.fmt_xdata = date_fmt
    #trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    #ax.fill_between(x, 0, 1, facecolor='gray', alpha=0.5, transform=trans)
    ax.xaxis.grid(True, 'major')

#This overlays a shapefile
#Should probably isolate to geometry
#Currently needs the ds associated with the imshow ax
#Should probably set this in the ax transform - need to look into this
#def shp_overlay(ax, ds, shp_fn, gt=None, color='w'):
def shp_overlay(ax, ds, shp_fn, gt=None, color='darkgreen'):
    from osgeo import ogr
    from pygeotools.lib import geolib
    #ogr2ogr -f "ESRI Shapefile" output.shp input.shp -clipsrc xmin ymin xmax ymax
    shp_ds = ogr.Open(shp_fn)
    lyr = shp_ds.GetLayer()
    lyr_srs = lyr.GetSpatialRef()
    lyr.ResetReading()
    nfeat = lyr.GetFeatureCount()
    #Note: this is inefficient for large numbers of features
    #Should produce collections of points or lines, then have single plot call
    for n, feat in enumerate(lyr):
        geom = feat.GetGeometryRef()
        geom_type = geom.GetGeometryType()
        #Points
        if geom_type == 1:
            mX, mY, z = geom.GetPoint()
            attr = {'marker':'o', 'markersize':5, 'linestyle':'None'}
        #Line
        elif geom_type == 2:
            l, mX, mY = geolib.line2pts(geom)
            z = 0
            #attr = {'marker':None, 'linestyle':'-', 'linewidth':0.5, 'alpha':0.8}
            attr = {'marker':None, 'linestyle':'-', 'linewidth':1.0, 'alpha':0.8}
            #attr = {'marker':'.', 'markersize':0.5, 'linestyle':'None'}
        #Polygon, placeholder
        #Note: this should be done with the matplotlib patch functionality
        #http://matplotlib.org/users/path_tutorial.html
        elif geom_type == 3:
            print("Polygon support not yet implemented")
            #ogr2ogr -nlt LINESTRING out.shp in.shp
            l, mX, mY = geolib.line2pts(geom)
            z = 0
            attr = {'marker':None, 'linestyle':'-', 'facecolor':'w'}

        ds_srs = geolib.get_ds_srs(ds) 
        if gt is None:
            gt = ds.GetGeoTransform()
        if not lyr_srs.IsSame(ds_srs):
            mX, mY, z = geolib.cT_helper(mX, mY, z, lyr_srs, ds_srs)

        #ds_extent = geolib.ds_extent(ds)
        ds_extent = geolib.ds_geom_extent(ds)
      
        mX = np.ma.array(mX)
        mY = np.ma.array(mY)

        mX[mX < ds_extent[0]] = np.ma.masked
        mX[mX > ds_extent[2]] = np.ma.masked
        mY[mY < ds_extent[1]] = np.ma.masked
        mY[mY > ds_extent[3]] = np.ma.masked

        mask = np.ma.getmaskarray(mY) | np.ma.getmaskarray(mX)
        mX = mX[~mask]
        mY = mY[~mask]

        if mX.count() > 0:
            ax.set_autoscale_on(False)
            if geom_type == 1: 
                pX, pY = geolib.mapToPixel(np.array(mX), np.array(mY), gt)
                ax.plot(pX, pY, color=color, **attr)
            else:
                l = np.ma.array(l)
                l = l[~mask]

                lmed = np.ma.median(np.diff(l))
                lbreaks = (np.diff(l) > lmed*2).nonzero()[0]
                if lbreaks.size: 
                    a = 0
                    lbreaks = list(lbreaks)
                    lbreaks.append(l.size)
                    for b in lbreaks:
                        mmX = mX[a:b+1]
                        mmY = mY[a:b+1]
                        a = b+1
                        #import ipdb; ipdb.set_trace()
                        #pX, pY = geolib.mapToPixel(np.array(mX), np.array(mY), gt)
                        pX, pY = geolib.mapToPixel(mmX, mmY, gt)
                        print(n, np.diff(pX).max(), np.diff(pY).max())
                        #ax.plot(pX, pY, color='LimeGreen', **attr)
                        #ax.plot(pX, pY, color='LimeGreen', alpha=0.5, **attr)
                        #ax.plot(pX, pY, color='w', alpha=0.5, **attr)
                        ax.plot(pX, pY, color=color, **attr)
                else:
                    pX, pY = geolib.mapToPixel(np.array(mX), np.array(mY), gt)
                    ax.plot(pX, pY, color=color, **attr)

#Added this here for convenience, needs further testing
def plot_2dhist(ax, x, y, xlim=None, ylim=None, xint=None, yint=None, nbins=(128,128), log=False, maxline=True, trendline=False):
    #Should compute number of bins automatically based on input values, xlim and ylim
    common_mask = ~(malib.common_mask([x,y]))
    x = x[common_mask]
    y = y[common_mask]
    if xlim is None:
        #xlim = (x.min(), x.max())
        xlim = malib.calcperc(x, (0.1, 99.9))
    if ylim is None:
        #ylim = (y.min(), y.max())
        ylim = malib.calcperc(y, (0.1, 99.9))
    #Note, round to nearest meter here
    #xlim = np.rint(np.array(xlim))
    xlim = np.array(xlim)
    ylim = np.array(ylim)
    if xint is not None:
        xedges = np.arange(xlim[0], xlim[1]+xint, xint)
    else:
        xedges = nbins[0]

    if yint is not None:
        yedges = np.arange(ylim[0], ylim[1]+yint, yint)
    else:
        yedges = nbins[1]

    H, xedges, yedges = np.histogram2d(x,y,range=[xlim,ylim],bins=[xedges, yedges])
    #H, xedges, yedges = np.histogram2d(x,y,range=[xlim,ylim],bins=nbins)
    #H = np.rot90(H)
    #H = np.flipud(H)
    H = H.T
    #Mask any empty bins
    Hmasked = np.ma.masked_where(H==0,H)
    #Hmasked = H
    H_clim = malib.calcperc(Hmasked, (2,98))
    if log:
        import matplotlib.colors as colors
        ax.pcolormesh(xedges,yedges,Hmasked,cmap='inferno',norm=colors.LogNorm(vmin=H_clim[0],vmax=H_clim[1]))
    else:
        ax.pcolormesh(xedges,yedges,Hmasked,cmap='inferno',vmin=H_clim[0],vmax=H_clim[1])
    if maxline:
        #Add line for max values in each x bin
        Hmed_idx = np.ma.argmax(Hmasked, axis=0)
        ymax = (yedges[:-1]+np.diff(yedges))[Hmed_idx]
        ax.plot(xedges[:-1]+np.diff(xedges), ymax, color='dodgerblue',lw=1.0)
    if trendline:
        #Add trendline
        import scipy.stats
        y_slope, y_intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
        y_f = y_slope * xlim + y_intercept
        ax.plot(xlim, y_f, color='limegreen', ls='--', lw=0.5)
