from distutils.core import setup, Extension
setup(name="unit_array", version="1.0",
      ext_modules=[Extension("unit_array", ["unit_array.c"])])
