from distutils.core import setup
from Cython.Build import cythonize
import os.path

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

setup(ext_modules=cythonize("pyfu/_all_cythonized.pyx"), requires=['Cython', 'numpy'])
