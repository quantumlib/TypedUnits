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

import unittest

from .perf_testing_util import (
    a_random_array_with_dimension,
    a_random_value_with_dimension,
    perf_goal,
)

from tunits import ValueArray, ValueWithDimension, ArrayWithDimension


# Using 1500 repeats so that each of 1119 different values in tunits.units get selected at least once.
@perf_goal(repeats=1500, avg_micros=15, args=[a_random_array_with_dimension] * 2)
def test_perf_array_add(a: ArrayWithDimension, b: ArrayWithDimension) -> ValueArray:
    return a + b


@perf_goal(
    repeats=1500,
    avg_micros=15,
    args=[a_random_array_with_dimension, a_random_value_with_dimension],
)
def test_perf_array_shift(a: ArrayWithDimension, b: ValueWithDimension) -> ValueArray:
    return a + b


@perf_goal(repeats=1500, avg_micros=12, args=[a_random_array_with_dimension] * 2)
def test_perf_array_multiply(
    a: ArrayWithDimension, b: ArrayWithDimension
) -> ValueArray:
    return a * b


@perf_goal(
    repeats=1500,
    avg_micros=15,
    args=[a_random_array_with_dimension, a_random_value_with_dimension],
)
def test_perf_array_scale(a: ArrayWithDimension, b: ValueWithDimension) -> ValueArray:
    return a * b


@perf_goal(repeats=1500, avg_micros=12, args=[a_random_array_with_dimension] * 2)
def test_perf_array_divide(
    a: ArrayWithDimension, b: ArrayWithDimension
) -> ValueArray:
    return a / b


@perf_goal(repeats=1500, avg_micros=135, args=[a_random_array_with_dimension])
def test_perf_array_str(a: ArrayWithDimension) -> str:
    return str(a)


@perf_goal(repeats=1500, avg_micros=180, args=[a_random_array_with_dimension])
def test_perf_array_repr(a: ArrayWithDimension) -> str:
    return repr(a)


if __name__ == "__main__":
    unittest.main()
