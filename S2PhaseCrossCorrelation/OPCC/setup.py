"""

 @title: Optimized Sentinel-2 Coregistration using Phase Cross Correlation
 @author: Matthew Tralka
 @date: May 2021
 @version: 0.1

"""

from distutils.core import Extension
from distutils.core import setup

from Cython.Build import cythonize
import numpy

sourcefiles = ["OptimizedPhaseCrossCorrelation.pyx"]

extensions = [Extension("OptimizedPhaseCrossCorrelation", sourcefiles)]

setup(ext_modules=cythonize(extensions), include_dirs=[numpy.get_include()])
