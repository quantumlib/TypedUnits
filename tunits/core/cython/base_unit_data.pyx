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
Defines named physical units exposed in the default unit database and as
members of tunits.units (and tunits.api.like_pylabrad_units).

Base units are defined just by their name, not by other units or quantities.
"""

from attrs import frozen


@frozen
class BaseUnitData:
    """Describes the properties of a base unit.

    Attributes:
        symbol: The short name for the unit (e.g. 'm' for meter).
        name: The full name of the unit (e.g. 'meter').
        use_prefixes: Should there be 'kiloUNIT', 'milliUNIT', etc.
    """

    symbol: str
    name: str
    use_prefixes: bool = True


__SI_BASE_UNITS = [
    BaseUnitData('m', 'meter'),
    BaseUnitData('kg', 'kilogram'),  # Note: has special-cased prefix behavior.
    BaseUnitData('s', 'second'),
    BaseUnitData('A', 'ampere'),
    BaseUnitData('K', 'kelvin'),
    BaseUnitData('mol', 'mole'),
    BaseUnitData('cd', 'candela'),
    BaseUnitData('rad', 'radian'),
]

__OTHER_BASE_UNITS = [
    BaseUnitData('dB', 'decibel', use_prefixes=False),
    BaseUnitData('dBm', 'decibel-milliwatt', use_prefixes=False),
    BaseUnitData('degC', 'celsius', use_prefixes=False),
    BaseUnitData('degF', 'fahrenheit', use_prefixes=False),
]

ALL_BASE_UNITS = __SI_BASE_UNITS + __OTHER_BASE_UNITS
