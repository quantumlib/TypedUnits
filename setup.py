from distutils.core import setup, Extension
import numpy as np

setup(name="fastunits", version="1.0",
      packages = ['fastunits'],
      include_dirs = [np.get_include()],
      ext_modules=[Extension("fastunits.unitarray", 
                             sources = ["fastunits/unitarray.c"])])
