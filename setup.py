from distutils.core import setup, Extension
setup(name="fastunits", version="1.0",
      packages = ['fastunits'],
      ext_modules=[Extension("fastunits.unitarray", 
                             sources = ["fastunits/unitarray.c"])])
