#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("Cosine_Similarity_Cython.pyx"),
    include_dirs=[np.get_include()]
)