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

"""
Exposes standard units as module variables.

For example, '1/units.millisecond' is equal to 'units.kHz'.
"""

from . import unit as __unit

# Expose defined units (e.g. 'meter', 'km', 'day') as module variables.
# Note: unless you like overwriting Boltzmann's constant, keep the underscores.
for _k, _v in __unit.default_unit_database.known_units.items():
    globals()[_k] = _v
