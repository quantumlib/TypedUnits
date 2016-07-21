from distutils.core import setup
from Cython.Build import cythonize

setup(
    name="fastunits",
    version="1.0",
    packages=['fastunits'],
    ext_modules=cythonize("fastunits/unitarray.pyx"),
    requires=['Cython', 'numpy'])
