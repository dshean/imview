#!/usr/bin/env python

from distutils.core import setup

# To prepare a new release
# python setup.py sdist upload

setup(name='imview',
      version='0.3.0',
      description='Image viewers for geospatial data',
      author='David Shean',
      author_email='dshean@gmail.com',
      license='MIT',
      url='https://github.com/dshean/imview',
      packages=['imview', 'imview.lib'],
      long_description=open('README.md').read(),
      install_requires=['numpy', 'gdal', 'matplotlib', 'pygeotools',
                        'matplotlib-scalebar', 'matplotlib-colorbar'],
      entry_points={
          'console_scripts': [
              'imviewer = imview.imviewer:main',
          ]
      }
      )
