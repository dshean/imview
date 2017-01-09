# imview
Image viewers for geospatial data

## Overview

This repo contains several utilities that I use on a daily basis for rapid data preview, detailed analysis, and figure generation.  The imviewer.py tool is probably the most useful.  It works well, but could use a rewrite, as it has been glommed together over the span of 4-5 years.

## Viewers
- imviewer - viewer for geospatial data, overlay capabilities
- stack_view - viewer for "stack.npz" time series output (see [pygeotools](https://github.com/dshean/pygeotools.git)), allowing for point sampling and plotting
- iv - lightweight viewer for standard images (jpg, png, etc.)
- review_img -   lightweight viewer to identify good and bad images

## Modules
- lib/pltlib - a collection of useful functions for matplotlib plotting, including drawing vector data over raster data

## Command-line Examples

#### Preprocessing (optional):
```
mos_fn=BigDEM.tif
gdaldem hillshade $mos_fn ${mos_fn}_hs_az315.tif
gdaladdo -ro -r average --config COMPRESS_OVERVIEW LZW --config BIGTIFF_OVERVIEW YES $mos_fn 2 4 8 16 32 64
gdaladdo -ro -r average --config COMPRESS_OVERVIEW LZW --config BIGTIFF_OVERVIEW YES ${mos_fn}_hs_az315.tif 2 4 8 16 32 64
```

#### View color DEM overlaid on shaded relief map:

`imviewer.py $mos_fn -overlay ${mos_fn}_hs_az315.tif -label 'Elevation (m WGS84)'`

* By default, this will quickly load a low-resolution preview (specify -full to load full-res image)
* Lower right corner shows coordinates and value under cursor
* Left-click to sample image coordinates, map coordinates and raster value
* Can specify transparency with `-alpha 0.5`

#### View with user-defined color map and limits

`imviewer.py -cmap 'RdYlBl' -clim -5 5 dem_dz_eul.tif -label 'Elevation difference (m)'`

#### Link several images (allows for simultaneous zoom and pan):

`imviewer.py -link dem.tif image.tif velocity.tif`

#### View polyline shapefile overlay:

`imviewer.py $mos_fn -overlay ${mos_fn}_hs_az315.tif -shp polyline.shp` 

#### Output high-quality figure with scalebar:

`imviewer.py $mos_fn -overlay ${mos_fn}_hs_az315.tif -scale x -label 'Elevation (m WGS84)' -of png -dpi 300` 

#### View time series stack:
```
make_stack.py -tr 'mean' -te 'intersection' 20080101_dem.tif 20090101_dem.tif 20100101_dem.tif
stack_view.py 20080101_dem_20100101_dem_stack_3.npz
```
* Left-click to extract time series at point on any of the context maps
* Right-click to clear all points
* Can zoom and pan on context maps

## Installation

Install the latest release from PyPI:

    pip install imview 

**Note**: by default, this will deploy executable scripts in /usr/local/bin

### Building from source

Clone the repository and install:

    git clone https://github.com/dshean/imview.git
    pip install -e imview

The -e flag ("editable mode", setuptools "develop mode") will allow you to modify source code and immediately see changes.

### Core requirements 
- [Matplotlib](http://matplotlib.org/)
- [GDAL/OGR](http://www.gdal.org/)
- [NumPy](http://www.numpy.org/)
- [pygeotools](https://github.com/dshean/pygeotools)

## License

This project is licensed under the terms of the MIT License.

