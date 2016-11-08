#!/usr/bin/env python

from distutils.core import setup

#To prepare a new release
#python setup.py sdist upload

setup(name='imview',
    version='0.1.0',
    description='Image viewers for geospatial data',
    author='David Shean',
    author_email='dshean@gmail.com',
    license='MIT',
    url='https://github.com/dshean/imview',
    packages=['imview', 'imview.lib'],
    long_description=open('README.md').read(),
    install_requires=['numpy','gdal','matplotlib', 'pygeotools'],
    #Note: this will write to /usr/local/bin
    #scripts=['imview/imview.py','imview/iv.py','imview/stack_view.py', 'imview/review_img.py']
)

