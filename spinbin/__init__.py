"""
-------
spinbin
-------
Welcome to `spinbin`, a Python package for fast 2 dimensional data binning
using `cython` and `numpy`.

Provides:

1.  Fast `cython` functions which escape the `gil`.

2.  Fast rotation matrix for tri-axial rotations.

3.  Several support functions for processing I/O.

------------
Installation
------------
See below for instructions for installing spinbin with either setup.py or PIP.
If for any reason you can not use either of these methods, please contact the
author (Sol Courtney) via email at sol.courtney@gmail.com and a solution can be
figured out.

Using PIP
---------
Below are the instruction examples for installing spinbin with PIP.
.. _PIP : https://packaging.python.org/tutorials/installing-packages

    From the spinbin top level directory, run the following:

    >>> pip install ./ -v

Using setup.py
--------------
Below are the instruction examples for installing spinbin using the basic
setup.py script.
.. _setup.py : https://docs.python.org/2/install

    From the spinbin top level directory, run the following:

    >>> python setup.py


"""
from __future__ import division
from __future__ import print_function

import cython
import numpy as np
import os
import sys

from cython.parallel import parallel
from cython.parallel import prange

from libc.stdlib cimport rand
from libc.math cimport cos
from libc.math cimport sin
from libc.math cimport M_PI

cimport numpy as np

