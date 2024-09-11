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


_VERSION_KEY = 'TypedUnits_RELEASE_VERSION'
__version__ = os.environ.get(_VERSION_KEY, '')
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
    long_description='A cython based units library with protobuffers support and a notion of dimensions.',
)
