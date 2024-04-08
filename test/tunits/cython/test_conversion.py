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

# noinspection PyProtectedMember
from tunits.core import _all_cythonized as u


def frac(numer=1, denom=1):
    return {'numer': numer, 'denom': denom}


def conv(factor=1.0, numer=1, denom=1, exp10=0):
    return {'factor': factor, 'ratio': frac(numer, denom), 'exp10': exp10}


def test_conversion_to_double():
    # Identity.
    assert u.conversion_to_double(conv()) == 1

    # Single elements.
    assert u.conversion_to_double(conv(factor=2)) == 2
    assert u.conversion_to_double(conv(numer=2)) == 2
    assert u.conversion_to_double(conv(denom=2)) == 0.5
    assert u.conversion_to_double(conv(exp10=2)) == 100

    # All elements.
    assert u.conversion_to_double(conv(factor=1.75, numer=5, denom=8, exp10=2)) == 109.375


def test_conversion_times():
    # Identity.
    assert u.conversion_times(conv(), conv()) == conv()

    # Single elements.
    assert u.conversion_times(conv(factor=2), conv(factor=3)) == conv(factor=6)
    assert u.conversion_times(conv(numer=2), conv(numer=3)) == conv(numer=6)
    assert u.conversion_times(conv(denom=2), conv(denom=3)) == conv(denom=6)
    assert u.conversion_times(conv(exp10=2), conv(exp10=3)) == conv(exp10=5)

    # All elements.
    p = u.conversion_times(
        conv(factor=3.14, numer=5, denom=12, exp10=2),
        conv(factor=2.71, numer=2, denom=15, exp10=-5),
    )
    assert p == conv(factor=3.14 * 2.71, numer=1, denom=18, exp10=-3)


def test_conversion_div():
    # Identity.
    assert u.conversion_div(conv(), conv()) == conv()

    # Single elements.
    assert u.conversion_div(conv(factor=3), conv(factor=2)) == conv(factor=1.5)
    assert u.conversion_div(conv(numer=2), conv(numer=3)) == conv(numer=2, denom=3)
    assert u.conversion_div(conv(denom=2), conv(denom=3)) == conv(denom=2, numer=3)
    assert u.conversion_div(conv(exp10=2), conv(exp10=3)) == conv(exp10=-1)

    # All elements.
    d = u.conversion_div(
        conv(factor=3.14, numer=5, denom=12, exp10=2),
        conv(factor=2.71, numer=2, denom=15, exp10=-5),
    )
    assert d == conv(factor=3.14 / 2.71, numer=25, denom=8, exp10=7)


def test_inverse_conversion():
    # Identity.
    assert u.inverse_conversion(conv()) == conv()

    # Single elements.
    assert u.inverse_conversion(conv(factor=2)) == conv(factor=0.5)
    assert u.inverse_conversion(conv(numer=3)) == conv(denom=3)
    assert u.inverse_conversion(conv(denom=5)) == conv(numer=5)
    assert u.inverse_conversion(conv(exp10=7)) == conv(exp10=-7)

    # All elements.
    c = u.inverse_conversion(conv(factor=3.14, numer=5, denom=12, exp10=2))
    assert c == conv(factor=1 / 3.14, numer=12, denom=5, exp10=-2)


def test_conversion_raise_to():
    # Identity.
    assert u.conversion_raise_to(conv(), frac()) == conv()
    assert u.conversion_raise_to(conv(2, 3, 4, 5), frac()) == conv(2, 3, 4, 5)
    assert u.conversion_raise_to(conv(), frac(500, 33)) == conv()

    # Factor.
    assert u.conversion_raise_to(conv(factor=3), frac(2)) == conv(factor=9)
    assert u.conversion_raise_to(conv(factor=4), frac(-2)) == conv(factor=0.0625)
    assert u.conversion_raise_to(conv(factor=3), frac(denom=2)) == conv(factor=1.7320508075688772)
    assert u.conversion_raise_to(conv(factor=3), frac(5, 9)) == conv(factor=1.8410575470987482)
    assert u.conversion_raise_to(conv(factor=3), frac(-5, 9)) == conv(factor=0.5431660740729487)

    # Numer.
    assert u.conversion_raise_to(conv(numer=3), frac(2)) == conv(numer=9)
    assert u.conversion_raise_to(conv(numer=3), frac(-2)) == conv(denom=9)
    assert u.conversion_raise_to(conv(numer=4), frac(denom=2)) == conv(numer=2)
    assert u.conversion_raise_to(conv(numer=512), frac(5, 9)) == conv(numer=32)
    # Lose precision.
    assert u.conversion_raise_to(conv(numer=3), frac(denom=2)) == conv(factor=1.7320508075688772)
    assert u.conversion_raise_to(conv(numer=512), frac(5, 11)) == conv(factor=17.040657431039403)
    assert u.conversion_raise_to(conv(numer=512), frac(-5, 11)) == conv(factor=0.058683181916356644)

    # Denom.
    assert u.conversion_raise_to(conv(denom=3), frac(2)) == conv(denom=9)
    assert u.conversion_raise_to(conv(denom=3), frac(-2)) == conv(numer=9)
    assert u.conversion_raise_to(conv(denom=4), frac(denom=2)) == conv(denom=2)
    assert u.conversion_raise_to(conv(denom=512), frac(5, 9)) == conv(denom=32)
    # Lose precision.
    assert u.conversion_raise_to(conv(denom=3), frac(denom=2)) == conv(factor=0.5773502691896258)
    assert u.conversion_raise_to(conv(denom=512), frac(5, 11)) == conv(factor=0.058683181916356644)
    assert u.conversion_raise_to(conv(denom=512), frac(-5, 11)) == conv(factor=17.040657431039403)

    # Exp10.
    assert u.conversion_raise_to(conv(exp10=6), frac(2)) == conv(exp10=12)
    assert u.conversion_raise_to(conv(exp10=3), frac(-2)) == conv(exp10=-6)
    assert u.conversion_raise_to(conv(exp10=6), frac(denom=2)) == conv(exp10=3)
    assert u.conversion_raise_to(conv(exp10=18), frac(5, 9)) == conv(exp10=10)
    # Lose precision.
    assert u.conversion_raise_to(conv(exp10=3), frac(denom=2)) == conv(factor=31.622776601683793)
    assert u.conversion_raise_to(conv(exp10=18), frac(5, 11)) == conv(factor=151991108.2952933)
    assert u.conversion_raise_to(conv(exp10=18), frac(-5, 11)) == conv(factor=6.579332246575682e-9)
