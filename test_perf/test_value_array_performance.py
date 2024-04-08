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

from test_perf.perf_testing_util import (
    a_random_compatible_unit_array,
    a_random_compatible_unit_val,
    a_random_unit_array,
    a_random_unit_val,
    perf_goal,
)


@perf_goal(avg_micros=15, args=[a_random_compatible_unit_array] * 2)
def test_perf_array_add(a, b):
    return a + b


@perf_goal(avg_micros=15, args=[a_random_compatible_unit_array, a_random_compatible_unit_val])
def test_perf_array_shift(a, b):
    return a + b


@perf_goal(avg_micros=15, args=[a_random_unit_array] * 2)
def test_perf_array_multiply(a, b):
    return a * b


@perf_goal(avg_micros=15, args=[a_random_unit_array, a_random_unit_val])
def test_perf_array_scale(a, b):
    return a * b


@perf_goal(avg_micros=15, args=[a_random_unit_array] * 2)
def test_perf_array_divide(a, b):
    return a / b


@perf_goal(avg_micros=750, args=[a_random_unit_array])
def test_perf_array_str(a):
    return str(a)


@perf_goal(avg_micros=900, args=[a_random_unit_array])
def test_perf_array_repr(a):
    return repr(a)


if __name__ == "__main__":
    unittest.main()
