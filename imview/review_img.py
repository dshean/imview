#! /usr/bin/env python
"""
Simple interactive image viewer to generate good/bad lists from input list of filenames
"""

import sys
import os
import argparse

from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from pygeotools.lib import iolib
from pygeotools.lib import malib

from imview.lib import gmtColormap
cpt_rainbow = gmtColormap.get_rainbow()
plt.register_cmap(cmap=cpt_rainbow)

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is one of "yes" or "no".
    """
    valid = {"yes":True,   "y":True,  "ye":True,
             "no":False,     "n":False}
    if default == None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "\
                             "(or 'y' or 'n').\n")

def get_parser():
    parser = argparse.ArgumentParser(description="View input images and generate good/bad lists")
    parser.add_argument('fn_list', nargs='+', help='Input filename(s)')
    parser.add_argument('-prefix', type=str, default=None, help='Output prefix for good/bad lists')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    fn_list = args.fn_list

    print
    print "Reviewing %i images" % len(fn_list)
    print

    good = []
    bad = []

    good_fn = "good_list.txt"
    if args.prefix is not None:
        good_fn = args.prefix+'_'+good_fn
    good_f=open(good_fn, 'a', 0)

    bad_fn = "bad_list.txt"
    if args.prefix is not None:
        bad_fn = args.prefix+'_'+bad_fn
    bad_f=open(bad_fn, 'a', 0)

    fig = plt.figure()
    ax  = fig.add_subplot(111)
    plt.ion()
    plt.show()

    #Use PIL Image
    basic = False 

    for fn in fn_list:
        print fn
        plt.clf()
        if basic:
            im = mpimg.imread(fn) 
            if im.ndim == 3:
                cmap = None
            plt.imshow(im, cmap=cmap)
        else:
            ds = gdal.Open(fn)
            a = iolib.gdal_getma_sub(ds)
            perc = malib.calcperc(a)
            cmap = 'cpt_rainbow'
            alpha = 1.0
            if '_hs' in fn:
                cmap = 'gray'
            else:
                hs_fn = os.path.splitext(fn)[0]+'_hs.tif'
                if os.path.exists(hs_fn):
                    hs_ds = gdal.Open(hs_fn)
                    hs = iolib.gdal_getma_sub(hs_ds)
                    hs_perc = malib.calcperc(hs)
                    plt.imshow(hs, cmap='gray', clim=hs_perc)
                    alpha = 0.5
            plt.imshow(a, cmap=cmap, clim=perc, alpha=alpha)
        fig.canvas.draw()
        
        if query_yes_no("{} good?".format(fn)):
            good.append(fn)
            good_f.write("%s\n" % fn)
        else:
            bad.append(fn)
            bad_f.write("%s\n" % fn)

    plt.close()

    print
    print "Good: %i" % (len(good))
    print good
    print
    print "Bad: %i" % (len(bad))
    print bad 
    print

    good_f.close()
    bad_f.close()

if __name__ == "__main__":
    main()
