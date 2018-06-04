import os
from numpy import get_include

from setuptools import find_packages
from setuptools import setup
from setuptools.extension import Extension
try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except:
    print("You don't seem to have Cython installed. Please get a")
    print("copy from www.cython.org and install it")
    import sys
    sys.exit(1)

ext_modules = Extension('spinbin',
                        [
                        os.path.join('spinbin', 'core', 'bin_data.pyx'),
                        os.path.join('spinbin', 'core', 'integerize.pyx'),
                        os.path.join('spinbin', 'core', 'limits.pyx'),
                        os.path.join('spinbin', 'gridbin', 'deposit.pyx'),
                        os.path.join('spinbin', 'rotate', 'triax_rotation.pyx')
                        ],
                        extra_compile_args=['-fopenmp'],
                        extra_link_args=['-fopenmp'])

setup(
    name='spinbin',
    version='0.1.0',
    author='Sol W. Courtney',
    author_email='swc2124@Columbia.edu',
    maintainer='Sol W. Courtney',
    maintainer_email='swc2124@Columbia.edu',
    url='https://github.com/swc2124/spinbin',
    description='Python tools for working with stellar data.',
    download_url='https://github.com/swc2124/spinbin.git',
    license='MIT',
    include_package_data=True,
    include_dirs=[
        get_include()
    ],
    packages=find_packages(),
    install_requires=[
        'numpy>=0.x',
        'cython>=0.x'
    ],
    cmdclass={
        'build_ext': build_ext
    },
    ext_modules=cythonize(
        ext_modules,
        include_path=[
            get_include()
        ]
    )
)