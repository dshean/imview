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

from pygeotools.lib import timelib
from pygeotools.lib import iolib
from pygeotools.lib import malib
from pygeotools.lib import geolib
from pygeotools.lib import filtlib 

from imview.lib import pltlib
import imview.lib.colormaps as cmaps
from imview.lib import gmtColormap
cpt_rainbow = gmtColormap.get_rainbow()
cmap = cpt_rainbow
#cmap = plt.get_cmap('gist_rainbow_r')

#Reorganized to use main, so now need global variables for interactive onclick, sampling and plotting functions 
#There is definitely a cleaner way to organize this functionality
global m
global d
global min_dt
global max_dt
global source_dict
global geoid_offset
global errorbars
global pad
global error
global source
global gt
global ms
global plot_trend
global plot_resid
global filter_outliers

global ax_list
global ax_pt_list
global ax_rel
global ax_abs
global ax_resid

global colors

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

def reset_colors():
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    global colors
    colors = itertools.cycle(color_list)

def save_figure():
    print "Not yet implemented"

#This creates a static legend for all point types
#Could put constant error bars on points in legend, not on plot 
def create_legend(ax, source, loc='lower left'):
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

def fmt_ax(ax, ylabel=None, legend_source=None):
    #ax.set_ylim(-1, 1)
    ax.set_xlim(min_dt, max_dt)
    pltlib.fmt_date_ax(ax)
    pltlib.pad_xaxis(ax)
    ax.set_ylabel(ylabel)
    if legend_source is not None:
        create_legend(ax, legend_source)

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
            #Note: need to fix this - need integer indices, or interpolate
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
        fig_fn = '%s_doyplot_fig.png' % os.path.splitext(s.stack_fn)[0]
    #doy_ax.set_ylim(100,260)
    plt.tight_layout()
    #plt.savefig(fig_fn, dpi=300)
    plt.draw()

def plot_point_map(mx, my):
    ex, ey = geolib.mapToPixel(mx, my, gt)
    plot_point(ex, ey)

def plot_point(ex, ey):
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
                    plt.figure(3)
                    r_samp_list = split_sample(r)
                    for i in r_samp_list:
                        r_line, = ax_resid.plot_date(i[0], i[1], marker=i[4], alpha=i[5], markersize=ms, color=c, linestyle='None') 
                        if errorbars:
                           if np.any(i[2] != 0): 
                                ax_resid.errorbar(i[0], i[1], yerr=i[2], color=c, linestyle='None')
                    plt.draw()
            for i in samp_list:
                #Don't label, as we want legend to contain trend values
                plt.figure(1)
                v_rel_line, = ax_rel.plot_date(i[0], i[1]-v_mean, marker=i[4], alpha=i[5], markersize=ms, color=c, linestyle='None') 
                if errorbars:
                    if np.any(i[2] != 0): 
                        plt.figure(1)
                        ax_rel.errorbar(i[0], i[1]-v_mean, yerr=i[2], color=c, linestyle='None')
                plt.figure(2)
                v_line, = ax_abs.plot_date(i[0], i[1], marker=i[4], alpha=i[5], markersize=ms, color=c, linestyle='None')
                if errorbars:
                    if np.any(i[2] != 0): 
                        plt.figure(2)
                        ax_abs.errorbar(i[0], i[1], yerr=i[2], color=c, linestyle='None')
            plt.figure(2)
            #Now draw lines connecting points
            v_line, = ax_abs.plot_date(d_valid, v_valid, marker=None, color=c, linestyle='-', linewidth=0.6, alpha=0.5) 
            plt.draw()
            #Don't really need/want lines between points on this one - too busy
            plt.figure(1)
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
            ax_pt_list[0].extend(ax_list[0].plot(ex, ey, 'o', color=c))
            ax_pt_list[1].extend(ax_list[1].plot(ex, ey, 'o', color=c))
            ax_pt_list[2].extend(ax_list[2].plot(ex, ey, 'o', color=c))
            ax_pt_list[3].extend(ax_list[3].plot(ex, ey, 'o', color=c))
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
        print ylabel
        ax.cla()
        fmt_ax(ax, ylabel)
        plt.draw()
    plt.figure(0)
    for pt_list in ax_pt_list:
        for i in pt_list:
            i.remove() 
        del pt_list[:]
    plt.draw()
    #Reset the color cycle
    reset_colors()

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
    source_dict['DG_mono'] = {'fn_pattern':'-DEM_mono_[0-9]*m_trans', 'label':'DG mono', 'marker':'D', 'error':1.0, 'type':'DEM'}
    source_dict['DG_stereo'] = {'fn_pattern':'-DEM_[0-9]*m_trans', 'label':'DG stereo', 'marker':'s', 'error':0.5, 'type':'DEM'}
    source_dict['DG_stereo_tiltcorr'] = {'fn_pattern':'-DEM_[0-9]*m_trans', 'label':'DG stereo tiltcorr', 'marker':'s', 'error':0.5, 'type':'DEM'}
    source_dict['DG_stereo_nocorr_tiltcorr'] = {'fn_pattern':'-DEM_[0-9]*m_trans', 'label':'DG stereo nocorr tiltcorr', 'marker':'s', 'alpha':0.5, 'error':0.5, 'type':'DEM'}
    source_dict['DG_mono_tiltcorr'] = {'fn_pattern':'-DEM_[0-9]*m_trans', 'label':'DG mono tiltcorr', 'marker':'D', 'error':0.5, 'type':'DEM'}
    source_dict['DG_mono_nocorr_tiltcorr'] = {'fn_pattern':'-DEM_[0-9]*m_trans', 'label':'DG mono nocorr tiltcorr', 'marker':'D', 'error':0.5, 'alpha':0.5, 'type':'DEM'}

    #Missing stereo align tiltcorr
    source_dict['DG_stereo_nocorr_reftrend_medcorr'] = {'fn_pattern':'-DEM_[0-9]*m_reftrend_medcorr', 'label':'DG stereo lint', 'marker':'+', 'error':1.5, 'type':'DEM'}
    source_dict['DG_stereo_nocorr_reftrend_tiltcorr'] = {'fn_pattern':'-DEM_[0-9]*m_reftrend_tiltcorr', 'label':'DG stereo lint+tilt', 'marker':'+', 'error':2.0, 'type':'DEM'}
    source_dict['DG_stereo_nocorr'] = {'fn_pattern':'-DEM_[0-9]*m', 'label':'DG stereo nocorr', 'marker':'o', 'error':4.0, 'type':'DEM'}
    source_dict['DG_mono_reftrend_tiltcorr'] = {'fn_pattern':'-DEM_mono_[0-9]*m_trans_reftrend_tiltcorr', 'label':'DG mono tilt', 'marker':'+', 'error':2.0, 'type':'DEM'}
    source_dict['DG_mono_nocorr_reftrend_tiltcorr'] = {'fn_pattern':'-DEM_mono_[0-9]*m_reftrend_tiltcorr', 'label':'DG mono lint+tilt', 'marker':'+', 'error':2.5, 'type':'DEM'}
    source_dict['DG_mono_nocorr'] = {'fn_pattern':'-DEM_mono_[0-9]*m', 'label':'DG mono nocorr', 'marker':'d', 'error':5.0, 'type':'DEM'}

    #Velocity data (easier to add these to same source_dict
    source_dict['TSX'] = {'fn_pattern':'_tsx', 'label':'TSX', 'marker':'o', 'error':np.nan, 'error_perc':0.03, 'type':'velocity'}
    source_dict['ALOS'] = {'fn_pattern':'_alos', 'label':'ALOS', 'marker':'s', 'error':np.nan, 'error_perc':0.03, 'type':'velocity'}
    source_dict['RS1'] = {'fn_pattern':'_rsat1', 'label':'RS1', 'marker':'H', 'error':np.nan, 'error_perc':0.03, 'type':'velocity'}
    source_dict['RS2'] = {'fn_pattern':'_rsat2', 'label':'RS2', 'marker':'p', 'error':np.nan, 'error_perc':0.03, 'type':'velocity'}
    source_dict['LS8'] = {'fn_pattern':'_ls8', 'label':'LS8', 'marker':'p', 'error':np.nan, 'error_perc':0.03, 'type':'velocity'}
    source_dict['LS8_golive'] = {'fn_pattern':'_vv', 'label':'LS8_golive', 'marker':'p', 'error':np.nan, 'error_perc':0.03, 'type':'velocity'}
    source_dict['None'] = {'fn_pattern':'None', 'label':'Other', 'marker':'+', 'error':0.0, 'type':'None'}
    return source_dict

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: %s stack.npz" % os.path.basename(sys.argv[0]))

    stack_fn = sys.argv[1]

    print "Loading stack"
    s = malib.DEMStack(stack_fn=stack_fn, stats=True, trend=True, save=False)
    global d
    d = s.date_list_o

    d_ptp = d[-1] - d[0]
    d_pad = 0.03*d_ptp
    global min_dt
    min_dt = d[0]-d_pad
    global max_dt
    max_dt = d[-1]+d_pad
    #Use these to set bounds to hardcode min/max of all stacks
    #import pytz
    #min_dt = datetime(1999,1,1)
    #min_dt = datetime(2007,1,1, tzinfo=pytz.utc)
    #max_dt = datetime(2015,12,31, tzinfo=pytz.utc)

    global source
    source = np.ma.array(s.source)
    global source_dict
    source_dict = get_source_dict()
    global error 
    error = s.error
    global gt
    gt = s.gt
    global m
    m = s.ma_stack
    val = s.stack_mean
    count = s.stack_count
    std = s.stack_std
    trend = s.stack_trend
    detrended_std = s.stack_detrended_std
    stack_type = 'dem'
    global filter_outliers
    filter_outliers = False 

    global pad
    global geoid_offset
    global plot_trend
    global plot_resid
    global errorbars

    if 'TSX' in source or 'ALOS' in source or 'RS1' in source or 'RS2' in source:
        stack_type = 'velocity' 

    if '_vv' in stack_fn:
        stack_type = 'velocity' 

    if 'zs' in stack_fn:
        stack_type = 'racmo'

    if 'meltrate' in stack_fn:
        stack_type = 'meltrate'

    if stack_type == 'velocity':
        #pad = 3
        pad = 1
        #Use this for Jak stack with RADARSAT data
        #pad = 0
        #interval = 'yr'
        interval = 'day'
        ylabel = 'Velocity (m/%s)' % interval
        ylabel_rel = 'Relative Velocity (m/%s)' % interval
        ylabel_resid = 'Detrended Velocity (m/%s)' % interval
        #plot4_label = 'Detrended std (m/%s)' % interval
        plot4_label = 'Velocity std (m/%s)' % interval
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

    #Set color cycle
    reset_colors()

    global ms
    ms = 5

    #fig = plt.figure(0, figsize=(14,12), facecolor='white')
    fig = plt.figure(0, figsize=(14,12))

    #These record all points plotted on the context plots
    global ax_pt_list
    ax_pt_list = [[], [], [], []]

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
    #dstd_clim = (0,1)
    ax3 = fig.add_subplot(224, sharex=ax0, sharey=ax0)
    if hs is not None:
        ax3.imshow(hs, cmap='gray', clim=hs_clim, **imshow_kwargs)
    #im3 = ax3.imshow(detrended_std, cmap=cpt_rainbow, clim=dstd_clim, alpha=alpha, **imshow_kwargs)
    im3 = ax3.imshow(std, cmap=cpt_rainbow, clim=dstd_clim, alpha=alpha, **imshow_kwargs)
    ax3.set_adjustable('box-forced')
    pltlib.hide_ticks(ax3)
    #pltlib.add_cbar(ax3, im3, 'Detrended Std (m)')
    pltlib.add_cbar(ax3, im3, plot4_label)

    global ax_list
    ax_list = [ax0, ax1, ax2, ax3]

    plt.autoscale(tight=True)
    plt.tight_layout()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    fig1 = plt.figure(1)
    global ax_rel
    ax_rel = fig1.add_subplot(111)
    fmt_ax(ax_rel, ylabel=ylabel_rel, legend_source=source)

    fig2 = plt.figure(2)
    global ax_abs
    ax_abs = fig2.add_subplot(111)
    fmt_ax(ax_abs, ylabel=ylabel, legend_source=source)

    fig3 = plt.figure(3)
    global ax_resid
    ax_resid = fig3.add_subplot(111)
    fmt_ax(ax_resid, ylabel=ylabel_resid, legend_source=source)
    plt.axhline(0, color='k', linestyle='-', linewidth=0.6)

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

    #Ngozumpa rock points
    plot_point_map(471730.519971, 3100956.30059)
    plot_point_map(474114.751943, 3099631.72727)
    plot_point_map(469023.783192, 3100184.59266)
    plot_point_map(473273.935837, 3101520.684)
    plot_point_map(472696.889042, 3098235.39259)

    plt.show()

if __name__ == "__main__":
    main()
