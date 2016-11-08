#! /usr/bin/env python

#Simple image viewer

import sys
import os

import matplotlib.pyplot as plt

fn_list = sys.argv[1:]

for fn in fn_list:
    fig = plt.figure()
    fig.canvas.set_window_title(os.path.split(fn)[1])
    fig.set_facecolor('white')
    img = plt.imread(fn)
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()

plt.show()
