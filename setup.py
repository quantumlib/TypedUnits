from distutils.core import setup, Extension
setup(name="unit_array", version="1.0",
      py_modules=['test_unit_array', 'unit_test_cmds', 'fast_units', 'unit_grammar'],
      ext_modules=[Extension("unit_array", ["unit_array.c"])])
