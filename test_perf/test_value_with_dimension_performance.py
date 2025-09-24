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

from tunits.units import kilometer, inch
from tunits import ValueWithDimension, Value
from .perf_testing_util import a_random_value_with_dimension, perf_goal


if TYPE_CHECKING:
    import numpy


# Using 1500 repeats so that each of 1119 different values in tunits.units get selected at least once.
@perf_goal(repeats=1500, avg_nanos=1800, args=[a_random_value_with_dimension] * 2)
def test_perf_add(a: ValueWithDimension, b: ValueWithDimension) -> ValueWithDimension:
    return a + b


@perf_goal(repeats=1500, avg_nanos=1200, args=[a_random_value_with_dimension])
def test_perf_scale(a: ValueWithDimension) -> ValueWithDimension:
    return a * 3.14


@perf_goal(repeats=1500, avg_micros=3, args=[a_random_value_with_dimension] * 2)
def test_perf_multiply(a: ValueWithDimension, b: ValueWithDimension) -> Value:
    return a * b


@perf_goal(repeats=1500, avg_nanos=600, args=[a_random_value_with_dimension] * 2)
def test_perf_get_item(
    a: ValueWithDimension, b: ValueWithDimension
) -> Union[float, 'numpy.typing.NDArray[Any]']:
    return a[b]


@perf_goal(repeats=1500, avg_nanos=1650, args=[a_random_value_with_dimension] * 2)
def test_perf_divmod(a: ValueWithDimension, b: ValueWithDimension) -> Any:
    return divmod(a, b)


@perf_goal(repeats=1500, avg_micros=8.5, args=[a_random_value_with_dimension] * 2)
def test_perf_import_multiply_add_heterogeneous(
    a: ValueWithDimension, b: ValueWithDimension
) -> Value:
    return a * kilometer + b * inch


@perf_goal(repeats=1500, avg_nanos=1000, args=[a_random_value_with_dimension])
def test_perf_abs(a: ValueWithDimension) -> ValueWithDimension:
    return abs(a)


@perf_goal(repeats=1500, avg_micros=3.5, args=[a_random_value_with_dimension])
def test_perf_pow(a: ValueWithDimension) -> Value:
    return abs(a) ** (2 / 3.0)


@perf_goal(repeats=1500, avg_micros=3.9, args=[a_random_value_with_dimension])
def test_perf_str(a: ValueWithDimension) -> str:
    return str(a)


@perf_goal(repeats=1500, avg_micros=40, args=[a_random_value_with_dimension])
def test_perf_repr(a: ValueWithDimension) -> str:
    return repr(a)


@perf_goal(repeats=1500, avg_nanos=400, args=[a_random_value_with_dimension] * 2)
def test_perf_is_compatible(a: ValueWithDimension, b: ValueWithDimension) -> bool:
    return a.is_compatible(b)


@perf_goal(repeats=1500, avg_nanos=1200, args=[a_random_value_with_dimension] * 2)
def test_perf_equate(a: ValueWithDimension, b: ValueWithDimension) -> bool:
    return a == b


@perf_goal(repeats=1500, avg_nanos=800, args=[a_random_value_with_dimension] * 2)
def test_perf_compare(a: ValueWithDimension, b: ValueWithDimension) -> bool:
    return a < b


if __name__ == "__main__":
    unittest.main()
