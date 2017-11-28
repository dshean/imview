#! /bin/bash

#Wrapper around gdaldem hillshade
#Inputs can be multiple DEMs, will process in parallel

gdal_opt='-co TILED=YES -co COMPRESS=LZW -co BIGTIFF=IF_SAFER'

gdaldem_opt=''
#gdaldem_opt='-alg ZevenbergenThorne -compute_edges'
gdaldem_opt+=' -compute_edges'
#Should check to see if srs is geographic, then automatically apply scale
#gdaldem_opt+=' -s 111120'

#Create shaded relief with illumnation from the following azimuths
#az_list="000 045 090 135 180 225 270 315"
#az_list="315 000 045"
az_list="315"

#Requires GNU Parallel
parallel "gdaldem hillshade $gdal_opt $gdaldem_opt -azimuth {1} {2} {2.}_hs_az{1}.tif" ::: $az_list ::: $@
