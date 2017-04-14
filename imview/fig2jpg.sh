#! /bin/bash

#in=$1
#out=${in%.*}_sm.jpg
#convert $in -units PixelsPerInch -antialias -resize 1500x1500 -density 150 -quality 85 $out

parallel "convert {} -units PixelsPerInch -antialias -resize 1500x1500 -density 150 -quality 85 {.}_sm.jpg" ::: $@
