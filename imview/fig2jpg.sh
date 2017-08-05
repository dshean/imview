#! /bin/bash

#This creates 150 dpi jpg images from inputs
#Useful for preparing slides or documents where full-res versions just lead to filesize bloat

#in=$1
#out=${in%.*}_sm.jpg
#convert $in -units PixelsPerInch -antialias -resize 1500x1500 -density 150 -quality 85 $out

parallel "convert {} -units PixelsPerInch -antialias -resize 1500x1500 -density 150 -quality 85 {.}_sm.jpg" ::: $@
