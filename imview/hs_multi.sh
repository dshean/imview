#! /bin/bash

#Quick hack implementation of multidirectional hillshade
#Will be in GDAL 2.2

in=$1

gdal_opt='-co TILED=YES -co COMPRESS=LZW -co BIGTIFF=IF_SAFER'
gdaldem_opt='-compute_edges'

#Generate hillshade from 4 different angles
az_list="225 270 315 360"
parallel "gdaldem hillshade $gdal_opt $gdaldem_opt -alt 30 -azimuth {1} {2} {2.}_hs_az{1}.tif" ::: $az_list ::: $in 

#Prepare smoothed version of aspect map
#29-pixel window for Gaussian filter (pygeotools)
s=29
filter.py $in $s 
smooth=${in%.*}_filt${s}px.tif
gdaldem aspect $gdal_opt $gdaldem_opt $smooth ${smooth%.*}_aspect.tif

#Create weight rasters for each hs
#No sin function in image_calc
#parallel "image_calc -o ${in%.*}_w{}.tif -c 'sin(var_0 - {})**2' ${in%.*}_hs_az{}.tif" ::: $az_list
~/src/imview/imview/hs_multi.py ${smooth%.*}_aspect.tif

#Combine, convert to byte
image_calc -o ${in%.*}_hs_multi.tif -d uint8 -c '255*(var_0*var_4 + var_1*var_5 + var_2*var_6 + var_3*var_7)/2.0' ${smooth%.*}_aspect_w{225,270,315,360}.tif ${in%.*}_hs_az{225,270,315,360}.tif 

