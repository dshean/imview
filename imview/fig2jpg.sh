#! /bin/bash

#This creates 150 dpi jpg images from inputs
#Useful for preparing slides or documents where full-res versions just lead to filesize bloat

#in=$1
#out=${in%.*}_sm.jpg
#convert $in -units PixelsPerInch -antialias -resize 1500x1500 -density 150 -quality 85 $out

#This is 10" @ 150 dpi (pptx standard)
#size=1500
#This is 13.33" @ 150 dpi (pptx widescreen)
#size=2000
#parallel "convert {} -units PixelsPerInch -antialias -resize ${size}x${size} -density 150 -quality 85 {.}_sm.jpg" ::: $@

#This is 10" @ 300 dpi
size=3000
parallel "convert {} -units PixelsPerInch -antialias -resize ${size}x${size} -density 300 -quality 95 {.}_300dpi.jpg" ::: $@
