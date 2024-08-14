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

# Combines all the .pyx files together, so they are compiled and optimized as
# a single thing (and have access to their respective fast cdefs).

include "cython/proto.pyx"
include "cython/frac.pyx"
include "cython/conversion.pyx"
include "cython/unit_array.pyx"
include "cython/unit_mismatch_error.pyx"
include "cython/with_unit.pyx"
include "cython/with_unit_value.pyx"
include "cython/with_unit_value_array.pyx"
include "cython/base_unit_data.pyx"
include "cython/derived_unit_data.pyx"
include "cython/physical_constant_data.pyx"
include "cython/prefix_data.pyx"
include "cython/unit_grammar.pyx"
include "cython/unit_database.pyx"
include "cython/unit.pyx"
include "cython/dimension.pyx"