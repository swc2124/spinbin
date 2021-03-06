--------------------
Welcome to `spinbin`
--------------------
Welcome to `spinbin`, a Python package for fast multiprocessing 2 and 3 dimensional data binning using `cython` and `numpy`.  This package supports my `skysurvey` package and is an expanded version of the included `c_functions` extension module within `skysurvey`.

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
Below are the instruction examples for installing spinbin with [PIP](https://packaging.python.org/tutorials/installing-packages).

From the spinbin top level directory, run the following:

    >>> pip install ./ -v

Using setup.py
--------------
Below are the instruction examples for installing spinbin using the basic
[setup.py](https://docs.python.org/2/install) script.

From the spinbin top level directory, run the following:

    >>> python setup.py
