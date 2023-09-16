"""
Connectomics
============

Provides
  1. An array object of arbitrary homogeneous items
  2. Fast mathematical operations over arrays
  3. Linear Algebra, Fourier Transforms, Random Number Generation

How to use the documentation
"""
print("brain initialized")

# To get submodules
#from . import conn
from .brain import *
__all__ = brain.__all__.copy()

#__all__.extend(['brain'])