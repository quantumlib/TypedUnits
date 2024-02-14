import os.path
import setuptools  # type: ignore

from Cython.Build import cythonize

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

setuptools.setup(
    name="pyfu",
    version="0.0.2",
    packages=setuptools.find_packages(),
    ext_modules=cythonize("pyfu/_all_cythonized.pyx", compiler_directives={'language_level': 3}),
    requires=['Cython', 'numpy'],
    python_requires=">=3.10.0",
)
