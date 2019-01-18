# -*- coding: utf-8 -*-
from distutils.core import setup, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs

setup(ext_modules=[Extension("testa", ["test.cpp"],
      include_dirs=get_numpy_include_dirs()+['./'])])
# from distutils.core import setup, Extension
# import numpy

# # define the extension module
# cos_module_np = Extension('cos_module_np', sources=['test.cpp'],
#                           include_dirs=[numpy.get_include()])

# # run the setup
# setup(ext_modules=[cos_module_np])