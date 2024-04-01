# Copyright 2024 The TUnits Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os.path
import setuptools  # type: ignore

from Cython.Build import cythonize

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

requirements = open('../requirements.txt', 'r').readlines()

setuptools.setup(
    name="pyfu",
    version="0.0.2",
    packages=setuptools.find_packages(),
    ext_modules=cythonize("pyfu/_all_cythonized.pyx", compiler_directives={'language_level': 3}),
    install_requires=requirements,
    setup_requires=requirements,
    python_requires=">=3.10.0",
)
