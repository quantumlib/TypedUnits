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

from typing import Iterator
import inspect

import pytest

import numpy as np

import tunits
import tunits.core as core


def _all_dimensions() -> Iterator[type[core.ValueWithDimension]]:
    for name in dir(core):
        if name == 'ValueWithDimension':
            continue
        obj = getattr(core, name)
        if inspect.isclass(obj) and issubclass(obj, core.ValueWithDimension):
            yield obj


_ALL_DIMENSIONS = [*_all_dimensions()]


@pytest.mark.parametrize('dimension', _ALL_DIMENSIONS)
def test_arithmetic_ops_preserve_type(dimension: type[core.ValueWithDimension]) -> None:
    u = dimension(dimension.valid_base_units()[0])
    a = u * 2
    b = u * 3

    # Declare with correct type so that mypy knows the type to compare to.
    c = u * 1.0

    c = a + b
    assert c == u * 5

    c = a - b
    assert c == u * -1

    c = a * 7
    assert c == u * 14

    c = a / 11
    assert c == u * 2 / 11


@pytest.mark.parametrize('dimension', _ALL_DIMENSIONS)
def test_arithmetic_ops_preserve_type_array(
    dimension: type[core.ValueWithDimension],
) -> None:
    u = dimension(dimension.valid_base_units()[0])
    a = u * [2, 3]
    b = u * [5, 7]

    # Declare with correct type so that mypy knows the type to compare to.
    c = u * [1.0, 2.0]

    c = a + b
    assert all(c == u * [7, 10])

    c = a - b
    assert all(c == u * [-3, -4])

    c = a * 7
    assert all(c == u * [14, 21])

    c = a / 11
    assert all(c == u * [2 / 11, 3 / 11])


def test_division_type() -> None:
    t: tunits.Time = 3 * tunits.ns

    t = t / 2
    assert t == 1.5 * tunits.ns

    assert isinstance(1 / t, tunits.Value)


def test_dimension_with_wrong_unit_raises():

    with pytest.raises(ValueError, match='not a valid unit for dimension'):
        _ = tunits.Time(1)

    with pytest.raises(ValueError, match='not a valid unit for dimension'):
        _ = tunits.TimeArray([1])

    with pytest.raises(ValueError, match='not a valid unit for dimension'):
        _ = tunits.Time(1, 'm')

    with pytest.raises(ValueError, match='not a valid unit for dimension'):
        _ = tunits.TimeArray([1], 'm')
