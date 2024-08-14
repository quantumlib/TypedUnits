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

from attrs import frozen


@frozen
class PrefixData:
    """Describes the properties of a unit prefix.

    Attributes:
        symbol: The short name for the prefix (e.g. 'G' for giga).
        name: The full name of the prefix (e.g. 'giga').
        exp10: The power of 10 the prefix corresponds to.
    """

    symbol: str
    name: str
    exp10: int


SI_PREFIXES = [
    PrefixData('Y', 'yotta', exp10=24),
    PrefixData('Z', 'zetta', exp10=21),
    PrefixData('E', 'exa', exp10=18),
    PrefixData('P', 'peta', exp10=15),
    PrefixData('T', 'tera', exp10=12),
    PrefixData('G', 'giga', exp10=9),
    PrefixData('M', 'mega', exp10=6),
    PrefixData('k', 'kilo', exp10=3),
    PrefixData('h', 'hecto', exp10=2),
    PrefixData('da', 'deka', exp10=1),
    PrefixData('d', 'deci', exp10=-1),
    PrefixData('c', 'centi', exp10=-2),
    PrefixData('m', 'milli', exp10=-3),
    PrefixData('u', 'micro', exp10=-6),
    PrefixData('n', 'nano', exp10=-9),
    PrefixData('p', 'pico', exp10=-12),
    PrefixData('f', 'femto', exp10=-15),
    PrefixData('a', 'atto', exp10=-18),
    PrefixData('z', 'zepto', exp10=-21),
    PrefixData('y', 'yocto', exp10=-24),
]
