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

from typing import Any, TYPE_CHECKING, Union
import unittest

from tunits import Value
from .perf_testing_util import a_random_compatible_unit_val, a_random_unit_val, perf_goal

if TYPE_CHECKING:
    import numpy


@perf_goal(avg_nanos=1800, args=[a_random_compatible_unit_val] * 2)
def test_perf_add(a: Value, b: Value) -> Value:
    return a + b


@perf_goal(avg_nanos=1100, args=[a_random_unit_val])
def test_perf_scale(a: Value) -> Value:
    return a * 3.14


@perf_goal(avg_micros=3, args=[a_random_unit_val] * 2)
def test_perf_multiply(a: Value, b: Value) -> Value:
    return a * b


@perf_goal(avg_nanos=600, args=[a_random_compatible_unit_val] * 2)
def test_perf_get_item(a: Value, b: Value) -> Union[float, 'numpy.typing.NDArray[Any]']:
    return a[b]


@perf_goal(avg_nanos=1500, args=[a_random_compatible_unit_val] * 2)
def test_perf_divmod(a: Value, b: Value) -> Any:
    return divmod(a, b)


@perf_goal(avg_micros=8.5, args=[a_random_compatible_unit_val] * 2)
def test_perf_import_multiply_add_heterogeneous(a: Value, b: Value) -> Value:
    from tunits.units import kilometer, inch

    return a * kilometer + b * inch


@perf_goal(avg_nanos=1000, args=[a_random_unit_val])
def test_perf_abs(a: Value) -> Value:
    return abs(a)


@perf_goal(avg_micros=3.5, args=[a_random_unit_val])
def test_perf_pow(a: Value) -> Value:
    return abs(a) ** (2 / 3.0)


@perf_goal(avg_micros=3.9, args=[a_random_unit_val])
def test_perf_str(a: Value) -> str:
    return str(a)


@perf_goal(avg_micros=40, args=[a_random_unit_val])
def test_perf_repr(a: Value) -> str:
    return repr(a)


@perf_goal(avg_nanos=800)
def test_perf_parse_atom() -> Value:
    return Value(1, 'kilogram')


@perf_goal(avg_micros=500)
def test_perf_parse_formula() -> Value:
    return Value(1, 'm*s/kg^4')


@perf_goal(avg_nanos=400, args=[a_random_unit_val] * 2)
def test_perf_is_compatible(a: Value, b: Value) -> bool:
    return a.is_compatible(b)


@perf_goal(avg_nanos=800, args=[a_random_unit_val] * 2)
def test_perf_equate(a: Value, b: Value) -> bool:
    return a == b


@perf_goal(avg_nanos=800, args=[a_random_compatible_unit_val] * 2)
def test_perf_compare(a: Value, b: Value) -> bool:
    return a < b


if __name__ == "__main__":
    unittest.main()
