#! /usr/bin/env python

#David Shean
#dshean@gmail.com

#Viewer for DEMStack objects, stored in npz format
#Lots of custom stuff still in here - need to clean up

import sys
import os
import itertools
from datetime import datetime, timedelta
 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import colormaps as cmaps
import gmtColormap
cpt_rainbow = gmtColormap.get_rainbow()
cmap = cpt_rainbow
#cmap = plt.get_cmap('gist_rainbow_r')

from lib import pltlib
from pygeotools.lib import timelib
from pygeotools.lib import iolib
from pygeotools.lib import malib
from pygeotools.lib import geolib
from pygeotools.lib import filtlib 

#This is kind of a mess
#Generate dictionary of different data sources
def get_source_dict():
    #Use OrderedDict here to remember order - this is important for cascade to identify source from fn
    from collections import OrderedDict
    source_dict = OrderedDict() 
    source_dict['ATM'] = {'fn_pattern':'_atm_', 'label':'ATM', 'marker':'^', 'error':0.1, 'type':'DEM'}
    source_dict['LVIS'] = {'fn_pattern':'_lvis_', 'label':'LVIS', 'marker':'v', 'error':0.1, 'type':'DEM'}
    source_dict['GLAS'] = {'fn_pattern':'_glas_', 'label':'GLAS', 'marker':'>', 'error':0.2, 'type':'DEM'}
    #source_dict['TDM'] = {'fn_pattern':'TDM', 'label':'TDM', 'marker':'H', 'error':1.0, 'type':'DEM'}
    source_dict['TDM'] = {'fn_pattern':'_DEM-trans_reference-DEM', 'label':'TDM', 'marker':'H', 'error':1.0, 'type':'DEM'}
    source_dict['SPIRIT'] = {'fn_pattern':'SP', 'label':'SPIRIT', 'marker':'p', 'error':3.0, 'type':'DEM'}
    source_dict['GLISTIN'] = {'fn_pattern':'GLISTIN', 'label':'GLISTIN', 'marker':'<', 'error':2.0, 'type':'DEM'}
    source_dict['DG_mono'] = {'fn_pattern':'-DEM_mono_32m_trans', 'label':'DG mono', 'marker':'D', 'error':1.0, 'type':'DEM'}
    source_dict['DG_stereo'] = {'fn_pattern':'-DEM_32m_trans', 'label':'DG stereo', 'marker':'s', 'error':0.5, 'type':'DEM'}
    source_dict['DG_stereo_tiltcorr'] = {'fn_pattern':'-DEM_32m_trans', 'label':'DG stereo tiltcorr', 'marker':'s', 'error':0.5, 'type':'DEM'}
    source_dict['DG_stereo_nocorr_tiltcorr'] = {'fn_pattern':'-DEM_32m_trans', 'label':'DG stereo nocorr tiltcorr', 'marker':'s', 'alpha':0.5, 'error':0.5, 'type':'DEM'}
    source_dict['DG_mono_tiltcorr'] = {'fn_pattern':'-DEM_32m_trans', 'label':'DG mono tiltcorr', 'marker':'D', 'error':0.5, 'type':'DEM'}
    source_dict['DG_mono_nocorr_tiltcorr'] = {'fn_pattern':'-DEM_32m_trans', 'label':'DG mono nocorr tiltcorr', 'marker':'D', 'error':0.5, 'alpha':0.5, 'type':'DEM'}

    #Missing stereo align tiltcorr
    source_dict['DG_stereo_nocorr_reftrend_medcorr'] = {'fn_pattern':'-DEM_32m_reftrend_medcorr', 'label':'DG stereo lint', 'marker':'+', 'error':1.5, 'type':'DEM'}
    source_dict['DG_stereo_nocorr_reftrend_tiltcorr'] = {'fn_pattern':'-DEM_32m_reftrend_tiltcorr', 'label':'DG stereo lint+tilt', 'marker':'+', 'error':2.0, 'type':'DEM'}
    source_dict['DG_stereo_nocorr'] = {'fn_pattern':'-DEM_32m', 'label':'DG stereo nocorr', 'marker':'o', 'error':4.0, 'type':'DEM'}
    source_dict['DG_mono_reftrend_tiltcorr'] = {'fn_pattern':'-DEM_mono_32m_trans_reftrend_tiltcorr', 'label':'DG mono tilt', 'marker':'+', 'error':2.0, 'type':'DEM'}
    source_dict['DG_mono_nocorr_reftrend_tiltcorr'] = {'fn_pattern':'-DEM_mono_32m_reftrend_tiltcorr', 'label':'DG mono lint+tilt', 'marker':'+', 'error':2.5, 'type':'DEM'}
    source_dict['DG_mono_nocorr'] = {'fn_pattern':'-DEM_mono_32m', 'label':'DG mono nocorr', 'marker':'d', 'error':5.0, 'type':'DEM'}

    #Velocity data (easier to add these to same source_dict
    source_dict['TSX'] = {'fn_pattern':'_tsx', 'label':'TSX', 'marker':'o', 'error':np.nan, 'error_perc':0.03, 'type':'velocity'}
    source_dict['ALOS'] = {'fn_pattern':'_alos', 'label':'ALOS', 'marker':'s', 'error':np.nan, 'error_perc':0.03, 'type':'velocity'}
    source_dict['RS1'] = {'fn_pattern':'_rsat1', 'label':'RS1', 'marker':'H', 'error':np.nan, 'error_perc':0.03, 'type':'velocity'}
    source_dict['RS2'] = {'fn_pattern':'_rsat2', 'label':'RS2', 'marker':'p', 'error':np.nan, 'error_perc':0.03, 'type':'velocity'}
    source_dict['LS8'] = {'fn_pattern':'_ls8', 'label':'LS8', 'marker':'p', 'error':np.nan, 'error_perc':0.03, 'type':'velocity'}
    source_dict['None'] = {'fn_pattern':'None', 'label':'Other', 'marker':'+', 'error':0.0, 'type':'None'}
    return source_dict

stack_fn = sys.argv[1]

print "Loading stack"
s = malib.DEMStack(stack_fn=stack_fn, stats=True, trend=True, save=False)
d = s.date_list_o

min_dt = d[0]
max_dt = d[-1]
#Use these to set bounds to hardcode min/max of all stacks
#import pytz
#min_dt = datetime(1999,1,1)
#min_dt = datetime(2007,1,1, tzinfo=pytz.utc)
#max_dt = datetime(2015,12,31, tzinfo=pytz.utc)

source = np.ma.array(s.source)
source_dict = get_source_dict()
error = s.error
gt = s.gt
m = s.ma_stack
val = s.stack_mean
count = s.stack_count
std = s.stack_std
trend = s.stack_trend
detrended_std = s.stack_detrended_std
stack_type = 'dem'
filter_outliers = False 

if 'TSX' in source or 'ALOS' in source or 'RS1' in source or 'RS2' in source:
    stack_type = 'velocity' 

if 'zs' in stack_fn:
    stack_type = 'racmo'

if 'meltrate' in stack_fn:
    stack_type = 'meltrate'

if stack_type == 'velocity':
    #pad = 3
    #Use this for Jak stack with RADARSAT data
    pad = 0
    ylabel = 'Velocity (m/yr)'
    ylabel_rel = 'Relative Velocity (m/yr)'
    ylabel_resid = 'Detrended Velocity (m/yr)'
    plot4_label = 'Detrended std (m/yr)'
    hs = None
    alpha = 1.0
    geoid_offset = False
    plot_trend = False
    plot_resid = False
    errorbars = False
    if 'RS' in source:
        filter_outliers = True
elif stack_type == 'racmo':
    pad = 0
    ylabel = 'RACMOFDM zs (m)'
    ylabel_rel = 'Relative RACMOFDM zs (m)'
    ylabel_resid = 'Detrended RACMOFDM zs (m)'
    plot4_label = 'Detrended std (m)'
    hs = None
    alpha = 1.0
    geoid_offset = False
    plot_trend = True 
    plot_resid = True 
    errorbars = False
elif stack_type == 'meltrate':
    pad = 3
    ylabel = 'Melt Rate (m/yr)'
    ylabel_rel = 'Relative Melt Rate (m/yr)'
    ylabel_resid = 'Detrended Melt Rate (m/yr)'
    plot4_label = 'Detrended std (m/yr)'
    hs = None
    alpha = 1.0
    geoid_offset = False
    plot_trend = True 
    plot_resid = False 
    errorbars = False
else:
    #pad = 5
    #pad = 1
    pad = 3
    ylabel = 'Elevation (m EGM2008)'
    ylabel_rel = 'Relative Elevation (m)'
    ylabel_resid = 'Detrended Elevation (m)'
    #plot4_label = 'Detrended std (m)'
    plot4_label = 'Elevation std (m)'
    s.mean_hillshade()
    hs = s.stack_mean_hs
    hs_clim = malib.calcperc(hs, (2,98))
    alpha = 0.6
    geoid_offset = False 
    plot_trend = True
    plot_resid = True 
    errorbars = True

color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
colors = itertools.cycle(color_list)
ms = 5

#This overloads the default matplotlib click handler
def onclick(event):
    b = event.button
    #Left button picks points
    if b is 1:
        ex = event.xdata
        ey = event.ydata
        #Need to make sure we're in valid axes
        if ex is not None and ey is not None:
            plot_point(ex, ey)
    elif b is 2:
        save_figure()
    #Right button clears figures
    elif b is 3:
        clear_figure()

def save_figure():
    print "Not yet implemented"

#This creates a static legend for all point types
#Could put constant error bars on points in legend, not on plot 
def create_legend(ax, loc='lower left'):
    lines = []
    labels = []
    uniq_src = np.ma.unique(source)
    if uniq_src.size == 1 and (uniq_src[0] == 'Other' or uniq_src[0] == 'None'):
        pass
    else:
        #Should create unique markers for each source
        #marker = itertools.cycle(('o', 's', '*', '+', '.', '*')) 
        for i in np.ma.unique(source):
            alpha = 1.0
            if i in source_dict:
                marker = source_dict[i]['marker']
                if 'alpha' in source_dict[i]:
                    alpha = source_dict[i]['alpha']
                label = source_dict[i]['label']
            else:
                marker = 'o'
                #marker = markers.next()
                label = i
            lines.append(plt.Line2D(range(1), range(1), linewidth=None, linestyle='', color='k', marker=marker, alpha=alpha, markersize=ms, markerfacecolor='k'))
            labels.append(label)
        marker_key = plt.legend(lines, labels, ncol=2, numpoints=1, loc=loc, prop={'size':8})
        ax.add_artist(marker_key)

def create_legend_interactive(ax):
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), numpoints=1, loc='lower left')
    legend = ax.get_legend()
    for h in legend.legendHandles:
        h.set_color('k')

def fmt_ax(ax, ylabel=None, legend=True):
    #ax.set_ylim(-1, 1)
    ax.set_xlim(min_dt, max_dt)
    pltlib.fmt_date_ax(ax)
    pltlib.pad_xaxis(ax)
    ax.set_ylabel(ylabel)
    if legend:
        create_legend(ax)

def linregress(v):
    import scipy.stats
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(d[~(np.ma.getmaskarray(v))], v.compressed())
    vm = d*slope + intercept
    r = v - vm
    #malib.print_stats(r)
    print slope*365.25, intercept, r_value**2, std_err
    return vm, r, slope*365.25

def sample_stack(ex, ey, geoid_offset=False, pad=3):
    if ex > m.shape[2]-1 or ey > m.shape[1]-1:
        print "Input coordinates are outside stack extent:"
        print ex, ey
        print m.shape
        v = None
    else:
        print "Sampling with pad: %i" % pad
        if pad == 0:
            v = m[:,ey,ex]
        else:
            window_x = np.around(np.clip([ex-pad, ex+pad+1], 0, m.shape[2]-1)).astype(int)
            window_y = np.around(np.clip([ey-pad, ey+pad+1], 0, m.shape[1]-1)).astype(int)
            print window_x
            print window_y
            v = m[:,window_y[0]:window_y[1],window_x[0]:window_x[1]].reshape(m.shape[0], np.ptp(window_x)*np.ptp(window_y))
            #v = v.mean(axis=1)
            v = np.ma.median(v, axis=1)
        if v.count() == 0:
            print "No valid values"
        else:
            mx, my = geolib.pixelToMap(ex, ey, gt)
            print ex, ey, mx, my
            print "Count: %i" % v.count()
            #Hack to get elevations relative to geoid
            #Note: this can be added multiple times if clicked quickly
            if geoid_offset:
                #geoid_offset = geolib.sps2geoid(mx, my, 0.0)[2]
                geoid_offset = geolib.nps2geoid(mx, my, 0.0)[2]
                print "Removing geoid offset: %0.1f" % geoid_offset
                v += geoid_offset
        #Should filter here
        #RS1 has some values that are many 1000s of m/yr below neighbors
        if filter_outliers:
            if True:
                med = malib.fast_median(v)
                mad = malib.mad(v)
                min_v = med - mad*4
                f_idx = (v < min_v).filled(False)
                if np.any(f_idx):
                    print med, mad
                    print "Outliers removed by absolute filter: (val < %0.1f)" % min_v
                    print timelib.o2dt(d[f_idx])
                    print v[f_idx]
                    v[f_idx] = np.ma.masked
            if True:
                v_idx = (~np.ma.getmaskarray(v)).nonzero()[0]
                #This tries to maintain fixed window in time
                f = filtlib.rolling_fltr(v, size=7)
                #This uses fixed number of neighbors
                f = filtlib.rolling_fltr(v[v_idx], size=7)
                #f_diff = np.abs(f - v)
                #Note: the issue is usually that the velocity values are too low
                #f_diff = f - v
                f_diff = f - v[v_idx]
                diff_thresh = 2000
                #f_idx = (f_diff > diff_thresh).filled(False)
                #f_idx = (f_diff < diff_thresh).filled(False)
                f_idx = np.zeros_like(v.data).astype(bool)
                f_idx[v_idx] = (f_diff > diff_thresh)
                if np.any(f_idx):
                    print "Outliers removed by rolling median filter: (val < %0.1f)" % diff_thresh
                    print timelib.o2dt(d[f_idx])
                    print v[f_idx]
                    v[f_idx] = np.ma.masked
    return v

#Return lists of valid points for each source type
#This is necessary to plot with different marker symbols
def split_sample(v):
    out = []
    v_idx = (~np.ma.getmaskarray(v)).nonzero()[0]
    good_v = v[v_idx].data
    good_d = d[v_idx].data
    good_error = error[v_idx].data
    good_source = source[v_idx].data
    for i in np.unique(good_source):
        alpha = 1.0
        idx = (good_source == i)
        if i in source_dict:
            #If error is defined as percentage, recompute for local sample value
            if 'error_perc' in source_dict:
                error_perc = source_dict[i]['error_perc']
                if ~np.isnan(error_perc):
                    good_error[idx] = good_v[idx]*error_perc
            im = source_dict[i]['marker']
            if 'alpha' in source_dict[i]:
                alpha = source_dict[i]['alpha']
            label = source_dict[i]['label']
        else:
            im = 'o'
            label = i
        il = [good_d[idx], good_v[idx], good_error[idx], label, im, alpha]
        out.append(il)
    return out

#Split sample by year, doy
def split_sample_doy(dt_list, v):
    out = []
    v_idx = (~np.ma.getmaskarray(v)).nonzero()[0]
    good_v = np.ma.array(v)[v_idx].data
    good_d = np.ma.array(dt_list)[v_idx].data
    good_y = np.array([myd.year for myd in good_d])
    unique_year = np.unique(good_y)
    good_doy = timelib.np_dt2j(good_d)
    full_y_mm = np.array([min(dt_list).year, max(dt_list).year])
    #Hack for plotting
    #full_y_mm = np.array([2007, 2015])
    full_y_ptp = full_y_mm.ptp()
    if full_y_ptp == 0:
        full_y_ptp = 1
    good_y_mm = np.array([min(good_d).year, max(good_d).year])
    #good_y_mm = np.array([2007, 2015])
    good_y_ptp = good_y_mm.ptp()
    if good_y_ptp == 0:
        good_y_ptp = 1
    #Note: could also assign color based on decimal year rather than integer year
    #Need scatter plot for variable c
    for i in unique_year:
        idx = (good_y == i)
        c_full = float(i-full_y_mm[0])/full_y_ptp
        c_good = float(i-good_y_mm[0])/good_y_ptp
        il = [i, good_doy[idx], good_v[idx], c_full, c_good]
        out.append(il)
    return out

def doy_plot(dt_list, v, ylabel=None, title=None):
    samp_list = split_sample_doy(dt_list, v)
    #fig = plt.figure(figsize=(5,7.5))
    fig = plt.figure()
    doy_ax = fig.add_subplot(111)
    doy_ax.set_title(title)
    if ylabel is not None:
        doy_ax.set_ylabel(ylabel)
    for i in samp_list:
        c = cmap(i[3])
        doy_ax.plot(i[1], i[2], color=c, label=i[0], marker='o', markersize=5, linestyle='-', linewidth=0.6)
    doy_ax.set_xlim(0,365)
    month_ndays = np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    month_ticks = np.cumsum(month_ndays)
    doy_ax.xaxis.set_ticks(month_ticks)
    month_ticklabels = doy_ax.xaxis.get_ticklabels()
    #for label in month_ticklabels[::2]:
    for label in month_ticklabels:
        label.set_visible(False)
    doy_ax.set_xlabel('Day of year')
    #month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    doy_ax.xaxis.set_ticks(month_ticks[:-1] + 15, minor=True)
    doy_ax.xaxis.set_ticklabels(month_names, minor=True)
    for tick in doy_ax.xaxis.get_minor_ticks():
        tick.tick1line.set_markersize(0)
        tick.tick2line.set_markersize(0)
        tick.label1.set_horizontalalignment('center')
    doy_ax.set_xlabel('Month')
    plt.legend(loc='upper right', prop={'size':8})
    #doy_ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    doy_ax.xaxis.grid(True, 'major')
    pltlib.minorticks_on(doy_ax, x=False, y=True)
    if title is not None:
        mx,my = title.split(', ')
        fig_fn = '%s_doyplot_%s_%s_fig.png' % (os.path.splitext(s.stack_fn)[0], mx, my) 
    else:
        fig_fn = '%s_doyplot_terminus_fig.png' % os.path.splitext(s.stack_fn)[0]
    #doy_ax.set_ylim(100,260)
    plt.tight_layout()
    #plt.savefig(fig_fn, dpi=300)
    plt.draw()

def plot_point_map(mx, my):
    ex, ey = geolib.mapToPixel(mx, my, gt)
    plot_point(ex, ey)

def plot_point(ex, ey, errorbars=True):
    if ex > m.shape[2]-1 or ey > m.shape[1]-1:
        print "Input coordinates are outside stack extent:"
        print ex, ey
        print m.shape
    else:
        v = sample_stack(ex, ey, geoid_offset=geoid_offset, pad=pad)
        v_idx = (~np.ma.getmaskarray(v)).nonzero()[0]
        v_error = error[v_idx].data
        v_source = source[v_idx].data
        samp_list = split_sample(v)
        #v = m[:,ey,ex]
        #if not samp_list:
        if v.count() > 0:
            c = next(colors)
            v_mean = v.mean()
            v_valid = v.compressed()
            v_rel_valid = v_valid - v_mean
            d_valid = np.ma.array(d, mask=np.ma.getmaskarray(v)).compressed()
            #Plot trendline
            if plot_trend:
                vm, r, slope = linregress(v)
                vm_rel = vm - v_mean 
                plt.figure(1)
                ax_rel.plot_date(d, vm_rel, label='%0.1f m/yr' % slope, marker=None, markersize=ms, color=c, linestyle='--', linewidth=0.6)
                #v_rel_abs_lim = int(max(abs(v_rel.min()), abs(v_rel.max())) + 0.5)
                #if np.all(np.abs(ax_rel.get_ylim()) < vn_abs_lim):
                #    ax_rel.set_ylim(-v_rel_abs_lim, v_rel_abs_lim)
                if plot_resid:
                    r_samp_list = split_sample(r)
                    plt.figure(3)
                    for i in r_samp_list:
                        r_line, = ax_resid.plot_date(i[0], i[1], marker=i[4], alpha=i[5], markersize=ms, color=c, linestyle='None') 
                        if errorbars:
                           if np.any(i[2] != 0): 
                                ax_resid.errorbar(i[0], i[1], yerr=i[2], color=c, linestyle='None')
                    plt.draw()
            for i in samp_list:
                plt.figure(1)
                #Don't label, as we want legend to contain trend values
                v_rel_line, = ax_rel.plot_date(i[0], i[1]-v_mean, marker=i[4], alpha=i[5], markersize=ms, color=c, linestyle='None') 
                if errorbars:
                    if np.any(i[2] != 0): 
                        ax_rel.errorbar(i[0], i[1]-v_mean, yerr=i[2], color=c, linestyle='None')
                plt.figure(2)
                v_line, = ax_abs.plot_date(i[0], i[1], marker=i[4], alpha=i[5], markersize=ms, color=c, linestyle='None')
                if errorbars:
                    if np.any(i[2] != 0): 
                        ax_abs.errorbar(i[0], i[1], yerr=i[2], color=c, linestyle='None')
            #Now draw lines connecting points
            v_line, = ax_abs.plot_date(d_valid, v_valid, marker=None, color=c, linestyle='-', linewidth=0.6, alpha=0.5) 
            plt.draw()
            plt.figure(1)
            #Don't really need/want lines between points on this one - too busy
            v_rel_line, = ax_rel.plot_date(d_valid, v_rel_valid, marker=None, color=c, linestyle='-', linewidth=0.6, alpha=0.5) 
            if plot_trend:
                #Add legend containing trend values
                plt.legend(loc='upper right', prop={'size':10})
            plt.draw()
            #Plot doy
            #Need to add title to doy plots
            mx, my = geolib.pixelToMap(ex, ey, gt)
            title='%0.1f, %0.1f' % (mx, my)
            #doy_plot(s.date_list, v, ylabel, title=title)
            #create_legend_interactive(ax_abs)
            #Now add point to context maps
            plt.figure(0)
            #Could get fancy here and scale the marker as unfilled square scaled to size of padded sample
            ax0_pt_list.extend(ax0.plot(ex, ey, 'o', color=c))
            ax1_pt_list.extend(ax1.plot(ex, ey, 'o', color=c))
            ax2_pt_list.extend(ax2.plot(ex, ey, 'o', color=c))
            ax3_pt_list.extend(ax3.plot(ex, ey, 'o', color=c))
            plt.draw()
            if False:
                out_fn = 'stack_sample_%0.1f_%0.1f.csv' % (mx, my)
                #np.savetxt(out_fn, np.array([d_valid, v_valid, v_error, v_source]).T, fmt='%0.6f, %0.1f, %0.1f, %s', delimiter=',')
                np.savetxt(out_fn, np.array([d_valid, v_valid, v_error]).T, fmt='%0.6f, %0.1f, %0.1f', delimiter=',')

def clear_figure():
    print "Clearing figures"
    for i in (1,2,3):
        plt.figure(i)
        ax = plt.gca()
        ylabel = ax.get_ylabel()
        ax.cla()
        fmt_ax(ax, ylabel)
        plt.draw()
    plt.figure(0)
    for ax_pt_list in (ax0_pt_list, ax1_pt_list, ax2_pt_list, ax3_pt_list):
        for i in ax_pt_list:
            i.remove() 
        del ax_pt_list[:]
    plt.draw()
    #Reset the color cycle
    global colors
    colors = itertools.cycle(color_list)

#fig = plt.figure(0, figsize=(14,12), facecolor='white')
fig = plt.figure(0, figsize=(14,12))

#These record all points plotted on the context plots
ax0_pt_list = []
ax1_pt_list = []
ax2_pt_list = []
ax3_pt_list = []

interp = 'none'
#interp = 'bicubic'

#Overlay on mean_hs
#Add colorbars
imshow_kwargs = {'interpolation':interp}

val_clim = malib.calcperc(val, (2,98))
ax0 = fig.add_subplot(221)
if hs is not None:
    ax0.imshow(hs, cmap='gray', clim=hs_clim, **imshow_kwargs)
im0 = ax0.imshow(val, cmap=cpt_rainbow, clim=val_clim, alpha=alpha, **imshow_kwargs)
#This was used for Stanton et al figure
#val_clim = (0, 50)
#im0 = ax0.imshow(val, cmap=cmaps.inferno, clim=val_clim, alpha=alpha, **imshow_kwargs)
ax0.set_adjustable('box-forced')
pltlib.hide_ticks(ax0)
pltlib.add_cbar(ax0, im0, ylabel)

count_clim = malib.calcperc(count, (2,98))
#count_clim = malib.calcperc(count, (4,100))
ax1 = fig.add_subplot(222, sharex=ax0, sharey=ax0)
if hs is not None:
    ax1.imshow(hs, cmap='gray', clim=hs_clim, **imshow_kwargs)
im1 = ax1.imshow(count, cmap=cmaps.inferno, clim=count_clim, alpha=alpha, **imshow_kwargs)
ax1.set_adjustable('box-forced')
pltlib.hide_ticks(ax1)
pltlib.add_cbar(ax1, im1, 'Count')

#clim=(-20, 20)
#trend_clim = malib.calcperc(trend, (1,99))
#trend_clim = malib.calcperc(trend, (2,98))
trend_clim = malib.calcperc(trend, (4,96))
#trend_clim = malib.calcperc(trend, (10,90))
max_abs_clim = max(np.abs(trend_clim))
trend_clim = (-max_abs_clim, max_abs_clim)
ax2 = fig.add_subplot(223, sharex=ax0, sharey=ax0)
#ax0.set_title("Trend")
if hs is not None:
    ax2.imshow(hs, cmap='gray', clim=hs_clim, **imshow_kwargs)
im2 = ax2.imshow(trend, cmap='RdBu', clim=trend_clim, alpha=alpha, **imshow_kwargs)
ax2.set_adjustable('box-forced')
pltlib.hide_ticks(ax2)
pltlib.add_cbar(ax2, im2, 'Linear Trend (m/yr)')

dstd_clim = (0, malib.calcperc(std, (0,95))[1])
#dstd_clim = (0, malib.calcperc(detrended_std, (0,98))[1])
ax3 = fig.add_subplot(224, sharex=ax0, sharey=ax0)
if hs is not None:
    ax3.imshow(hs, cmap='gray', clim=hs_clim, **imshow_kwargs)
im3 = ax3.imshow(detrended_std, cmap=cpt_rainbow, clim=dstd_clim, alpha=alpha, **imshow_kwargs)
#im3 = ax3.imshow(std, cmap=cpt_rainbow, clim=dstd_clim, alpha=alpha, **imshow_kwargs)
ax3.set_adjustable('box-forced')
pltlib.hide_ticks(ax3)
#pltlib.add_cbar(ax3, im3, 'Detrended Std (m)')
pltlib.add_cbar(ax3, im3, plot4_label)

plt.autoscale(tight=True)
plt.tight_layout()

cid = fig.canvas.mpl_connect('button_press_event', onclick)

fig1 = plt.figure(1)
ax_rel = fig1.add_subplot(111)
fmt_ax(ax_rel, ylabel=ylabel_rel)

fig2 = plt.figure(2)
ax_abs = fig2.add_subplot(111)
fmt_ax(ax_abs, ylabel=ylabel)

fig3 = plt.figure(3)
ax_resid = fig3.add_subplot(111)
fmt_ax(ax_resid, ylabel=ylabel_resid)
plt.axhline(0, color='k', linestyle='-', linewidth=0.6)

#plot_terminus_position()

#Big3 termini
#plot_point_map(-180024.975, -2279364.741)
#plot_point_map(302396.059, -2576964.367)
#plot_point_map(483016.446, -2286632.427)

#Jak firstround test
#440.183345752 709.374366988 -180298.132936 -2278515.97974
#422.286048902 733.237429455 -180870.846435 -2279279.59774
#501.174681965 748.151843496 -178346.410177 -2279756.85899
#499.186093426 771.020611693 -178410.04501 -2280488.65957
#581.712517789 786.929320004 -175769.199431 -2280997.73824
#580.718223519 804.826616854 -175801.016847 -2281570.45174
#plot_point_map(-180298.132936, -2278515.97974)
#plot_point_map(-180870.846435, -2279279.59774)
#plot_point_map(-178346.410177, -2279756.85899)
#plot_point_map(-178410.04501, -2280488.65957)
#plot_point_map(-175769.199431, -2280997.73824)
#plot_point_map(-175801.016847, -2281570.45174)

#plot_point_map(-180036.266704, -2278639.54921)
#plot_point_map(-178269.349311, -2280004.89447)

"""
#Joughin 2012 velocity points
#M6 -180600,-2278600
plot_point_map(-180600,-2278600)
#M9 -178300,-2281100
plot_point_map(-178300,-2281100)
#M13 -173300,-2281500
plot_point_map(-173300,-2281500)
#M17 -169000,-2280200
plot_point_map(-169000,-2280200)
#M26 -165900,-2279500
plot_point_map(-165900,-2279500)
#M43 -160500,-2278400
plot_point_map(-160500,-2278400)
"""

#Joughin 2012 dh/dt points
#Z195 -182960,-2276400
#Z216 -182120,-2278000
#Z387 -176580,-2280900
#Z603 -170240,-2279900
#Z719 -163830,-2278600
#Zupper -157950,-2277700

"""
Jak figures from AGU
#plot_point_map(-180184.18903617435717,-2279738.798009647522122)
#00,-183668.208997432579054,-2275141.501217651646584
#plot_point_map(-183668.208997432579054,-2275141.501217651646584)
#01,-182353.737558729044395,-2277448.532722311560065
plot_point_map(-182353.737558729044395,-2277448.532722311560065)
#02,-180462.508243858901551,-2277733.558416932355613
plot_point_map(-180462.508243858901551,-2277733.558416932355613)
#03,-180304.905800953129074,-2278472.948601203970611
#plot_point_map(-180304.905800953129074,-2278472.948601203970611)
#04,-180184.18903617435717,-2279738.798009647522122
#plot_point_map(-180184.18903617435717,-2279738.798009647522122)
#05,-180080.238488725677598,-2280038.913299863692373
plot_point_map(-180080.238488725677598,-2280038.913299863692373)
#06,-178433.795946879923576,-2280238.431286093313247
plot_point_map(-178433.795946879923576,-2280238.431286093313247)
#07,-178311.402560367772821,-2280577.10887617059052
#plot_point_map(-178311.402560367772821,-2280577.10887617059052)
#08,-174000.808084720250918,-2280617.347797763068229
plot_point_map(-174000.808084720250918,-2280617.347797763068229)
#09,-168209.756618797808187,-2279658.320166463498026
plot_point_map(-168209.756618797808187,-2279658.320166463498026)
#10,-159571.801450172642944,-2277824.095990514848381
plot_point_map(-159571.801450172642944,-2277824.095990514848381)
#11,-157875.060256335273152,-2277639.667599882464856
plot_point_map(-157875.060256335273152,-2277639.667599882464856)
#12,-149548.956730055389926,-2275476.825564261060208
plot_point_map(-149548.956730055389926,-2275476.825564261060208)
#13,-149763.564311884954805,-2273448.113267276901752
plot_point_map(-149763.564311884954805,-2273448.113267276901752)
"""

"""
#rock_highcount,-188335.577014962822432,-2278825.682833711151034
plot_point_map(-188335.577014962822432,-2278825.682833711151034)
#rock_highcount2,-188340.603025859134505,-2279752.981844075955451
plot_point_map(-188340.603025859134505,-2279752.981844075955451)
#rock_lowstd,-189906.205420052603586,-2277094.222079940605909
plot_point_map(-189906.205420052603586,-2277094.222079940605909)
#rock_lowstd2,-187219.802595987217501,-2279398.648075887002051
plot_point_map(-187219.802595987217501,-2279398.648075887002051)
#margin1,-188649.702695980813587,-2283087.74007376190275
plot_point_map(-188649.702695980813587,-2283087.74007376190275)
#margin2,-186704.636479117762065,-2280036.951459716074169
plot_point_map(-186704.636479117762065,-2280036.951459716074169)
#rock_n,-190672.672081736323889,-2266245.577560304664075
plot_point_map(-190672.672081736323889,-2266245.577560304664075)
#margin_n,-190740.523228836100316,-2264883.528607411310077
plot_point_map(-190740.523228836100316,-2264883.528607411310077)
"""

#Stanton site
#plot_point_map(-1606611, -301871)
##plot_point_map(-1606742, -302164)
#20120111
#plot_point_map(-1606093.99136, -300364.168531)
#20131221
#plot_point_map(-1609115.131,-307316.722)
#PIG2
##plot_point_map(-1577500.90441, -310357.800583)
#plot_point_map(-1578160.63304, -310685.589483)

#PIG Lake center
#plot_point_map(-1592550.279,-238959.242)
#PIG off-lake spot
#plot_point_map(-1596376.138,-239460.994)

"""
#Sequence over PIG grounding zone
#Upstream
#plot_point_map(-1589871.98024, -252042.841939)
#Local high
plot_point_map(-1590627.93726, -260798.30489)
plot_point_map(-1591194.15313, -266842.566003)
plot_point_map(-1594891.71034, -274648.520111)
plot_point_map(-1595649.19602, -277318.978095)
plot_point_map(-1596740.48894, -280913.825382)
plot_point_map(-1597506.70072, -285461.777641)
plot_point_map(-1598794.68739, -288514.359645)
#plot_point_map(-1599776.10091, -291956.9575)
plot_point_map(-1617820.29908, -285567.598353)
"""

"""
#print "Saving figure"
#fig_fn = os.path.splitext(s.stack_fn)[0] + '_context_maps.pdf'
fig_fn = os.path.splitext(s.stack_fn)[0] + '_context_maps.png'
plt.figure(0)
plt.tight_layout()
plt.savefig(fig_fn, dpi=300)

fig_fn = os.path.splitext(s.stack_fn)[0] + '.png'
plt.figure(2)
#plt.ylim(70, 350)
plt.tight_layout()
plt.savefig(fig_fn, dpi=300)
"""

plt.show()
