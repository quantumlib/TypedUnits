from distutils.core import setup
from setuptools import Extension
from Cython.Distutils import build_ext

setup(
    ext_modules=[Extension(
        "fastunits.__all_cythonized",
        ["fastunits/cython/__all_cythonized.pyx"])],
    requires=['Cython', 'numpy'],
    cmdclass={'build_ext': build_ext})
