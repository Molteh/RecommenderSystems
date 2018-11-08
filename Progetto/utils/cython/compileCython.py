#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16/07/2017

@author: Maurizio Ferrari Dacrema
"""


from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("cosine_similarity.pyx"),
    include_dirs=[numpy.get_include()]
)