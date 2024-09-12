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

import os
import setuptools
from Cython.Build import cythonize

requirements = open('requirements.txt', 'r').readlines()



__version__ = ''
if os.path.exists('version/version.txt'):
    __version__ = open('version/version.txt').read().strip()

if not __version__:
    # Local build
    import _version

    __version__ = _version.__version__ + '.dev'


setuptools.setup(
    name="typedunits",
    version=__version__,
    packages=['tunits.proto', 'tunits.core', 'tunits'],
    include_package_data=True,
    ext_modules=cythonize(
        [
            setuptools.Extension(
                'tunits_core',
                ['tunits/core/_all_cythonized.pyx'],
                extra_compile_args=["-O3"],
            )
        ],
        compiler_directives={
            'language_level': 3,
            'embedsignature': True,
        },
    ),
    install_requires=requirements,
    setup_requires=requirements,
    python_requires=">=3.10.0",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type='text/markdown',
    url='https://github.com/quantumlib/TypedUnits',
    author='The TUnits Authors',
)
