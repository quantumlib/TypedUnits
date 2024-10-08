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

import pytest
import tunits.core as u
from test.test_utils import frac


def test_gcd() -> None:
    with pytest.raises(TypeError):
        u.gcd("a", "b")  # type: ignore[arg-type]

    assert u.gcd(0, -2) == 2
    assert u.gcd(0, -1) == 1
    assert u.gcd(0, 0) == 0
    assert u.gcd(0, 1) == 1
    assert u.gcd(0, 2) == 2

    assert u.gcd(1, -2) == 1
    assert u.gcd(1, -1) == 1
    assert u.gcd(1, 0) == 1
    assert u.gcd(1, 1) == 1
    assert u.gcd(1, 2) == 1

    assert u.gcd(2, -2) == 2
    assert u.gcd(2, -1) == 1
    assert u.gcd(2, 0) == 2
    assert u.gcd(2, 1) == 1
    assert u.gcd(2, 2) == 2

    assert u.gcd(3, -2) == 1
    assert u.gcd(3, -1) == 1
    assert u.gcd(3, 0) == 3
    assert u.gcd(3, 1) == 1
    assert u.gcd(3, 2) == 1

    assert u.gcd(2 * 3 * 5, 3 * 5 * 7) == 3 * 5
    assert u.gcd(2 * 3 * 5 * 5, 3 * 5 * 7) == 3 * 5
    assert u.gcd(2 * 3 * 5 * 5, 3 * 5 * 5 * 7) == 3 * 5 * 5
    assert u.gcd(945356, 633287) == 1
    assert u.gcd(+541838, +778063) == 11
    assert u.gcd(-541838, +778063) == 11
    assert u.gcd(+541838, -778063) == 11
    assert u.gcd(-541838, -778063) == 11


def test_frac_least_terms() -> None:
    with pytest.raises(TypeError):
        u.frac_least_terms("a", "b")  # type: ignore[arg-type]
    with pytest.raises(ZeroDivisionError):
        u.frac_least_terms(0, 0)
    with pytest.raises(ZeroDivisionError):
        u.frac_least_terms(1, 0)

    assert u.frac_least_terms(0, 3) == frac(0)
    assert u.frac_least_terms(0, -3) == frac(0)
    assert u.frac_least_terms(2, 3) == frac(2, 3)
    assert u.frac_least_terms(2, 4) == frac(denom=2)

    assert u.frac_least_terms(+4, +6) == frac(+2, 3)
    assert u.frac_least_terms(-4, +6) == frac(-2, 3)
    assert u.frac_least_terms(+4, -6) == frac(-2, 3)
    assert u.frac_least_terms(-4, -6) == frac(+2, 3)

    assert u.frac_least_terms(1, 1) == frac()
    assert u.frac_least_terms(0, 1) == frac(0)
    assert u.frac_least_terms(121, 33) == frac(11, 3)


def test_frac_times() -> None:
    assert u.frac_times(frac(0), frac(5, 7)) == frac(0)
    assert u.frac_times(frac(0), frac(-5, 7)) == frac(0)
    assert u.frac_times(frac(2, 3), frac(0)) == frac(0)
    assert u.frac_times(frac(2, 3), frac(5, 7)) == frac(10, 21)
    assert u.frac_times(frac(2, 33), frac(55, 7)) == frac(10, 21)
    assert u.frac_times(frac(22, 3), frac(5, 77)) == frac(10, 21)
    assert u.frac_times(frac(-2, 3), frac(5, 7)) == frac(-10, 21)
    assert u.frac_times(frac(2, 3), frac(-5, 7)) == frac(-10, 21)
    assert u.frac_times(frac(-2, 3), frac(-5, 7)) == frac(10, 21)


def test_frac_div() -> None:
    with pytest.raises(ZeroDivisionError):
        u.frac_div(frac(2, 3), frac(0))

    assert u.frac_div(frac(0), frac(5, 7)) == frac(0)
    assert u.frac_div(frac(0), frac(-5, 7)) == frac(0)
    assert u.frac_div(frac(2, 3), frac(5, 7)) == frac(14, 15)
    assert u.frac_div(frac(22, 3), frac(55, 7)) == frac(14, 15)
    assert u.frac_div(frac(2, 33), frac(5, 77)) == frac(14, 15)
    assert u.frac_div(frac(-2, 3), frac(5, 7)) == frac(-14, 15)
    assert u.frac_div(frac(2, 3), frac(-5, 7)) == frac(-14, 15)
    assert u.frac_div(frac(-2, 3), frac(-5, 7)) == frac(14, 15)


def test_float_to_twelths_frac() -> None:
    with pytest.raises(ValueError):
        u.float_to_twelths_frac(1.0 / 24)
    with pytest.raises(ValueError):
        u.float_to_twelths_frac(1.0 / 7)
    with pytest.raises(ValueError):
        u.float_to_twelths_frac(1.0 / 5)
    with pytest.raises(ValueError):
        u.float_to_twelths_frac(1.0 / 11)
    with pytest.raises(ValueError):
        u.float_to_twelths_frac(1.0 / 13)

    assert u.float_to_twelths_frac(0) == frac(0)
    assert u.float_to_twelths_frac(502) == frac(502)
    assert u.float_to_twelths_frac(1.0 / 12) == frac(denom=12)
    assert u.float_to_twelths_frac(-1.0 / 12) == frac(-1, 12)
    assert u.float_to_twelths_frac(501.0 / 3) == frac(167)
    assert u.float_to_twelths_frac(502.0 / 3) == frac(502, 3)

    # Precision.
    assert u.float_to_twelths_frac((1 << 55) + 1) == frac((1 << 55) + 1)
    assert u.float_to_twelths_frac(float(1 << 55) / 3.0) == frac(1 << 55, 3)


def test_frac_to_double() -> None:
    assert u.frac_to_double(frac(0)) == 0
    assert u.frac_to_double(frac(2, 3)) == 2.0 / 3
