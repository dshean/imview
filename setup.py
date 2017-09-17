#!/usr/bin/env python

from distutils.core import setup

#To prepare a new release
#python setup.py sdist upload

setup(name='imview',
    version='0.3.0',
    description='Image viewers for geospatial data',
    author='David Shean',
    author_email='dshean@gmail.com',
    license='MIT',
    url='https://github.com/dshean/imview',
    packages=['imview','imview.lib'],
    long_description=open('README.md').read(),
    install_requires=['numpy','gdal','matplotlib', 'pygeotools', 'matplotlib-scalebar', 'matplotlib-colorbar'],
    #Note: this will write to /usr/local/bin
    scripts=['imview/imviewer.py','imview/iv.py','imview/stack_view.py', \
    'imview/review_img.py', 'imview/color_hs.py', \
    'imview/gdaladdo_ro.sh', 'imview/hs.sh', 'imview/fig2jpg.sh']
)

