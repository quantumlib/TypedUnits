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

cimport cython
from libc.math cimport floor as c_floor, pow as c_pow


cdef struct conversion:
    # A conversion factor with support for fractions and power-of-10 exponents
    # to reduce floating point error in common use cases.
    #
    # Conversions should always have a proper fraction and *never* be zero.
    double factor
    frac ratio
    int exp10


cdef conversion identity_conversion():
    """Returns a conversion that does nothing."""
    cdef conversion c
    c.factor = 1.0
    c.ratio.numer = 1
    c.ratio.denom = 1
    c.exp10 = 0
    return c


@cython.cdivision(True)
cpdef double conversion_to_double(conversion c):
    """Returns a double that approximates the given conversion."""
    return c.ratio.numer * c.factor * c_pow(10, c.exp10) / c.ratio.denom


cpdef conversion conversion_times(conversion a, conversion b):
    """Returns a conversion equivalent to applying both given conversions."""
    cdef conversion c
    c.factor = a.factor * b.factor
    c.ratio = frac_times(a.ratio, b.ratio)
    c.exp10 = a.exp10 + b.exp10
    return c


@cython.cdivision(True)
cpdef conversion conversion_div(conversion a, conversion b) except *:
    """
    Returns a conversion equivalent to applying one conversion and un-applying
    another.
    """
    cdef conversion c
    c.factor = a.factor / b.factor
    c.ratio = frac_div_not_zero(a.ratio, b.ratio)
    c.exp10 = a.exp10 - b.exp10
    return c


cdef long long round(double d):
    return <long long>c_floor(0.5 + d)


@cython.cdivision(True)
cpdef conversion inverse_conversion(conversion c):
    cdef conversion result
    result.factor = 1 / c.factor
    result.ratio.numer = c.ratio.denom
    result.ratio.denom = c.ratio.numer
    result.exp10 = -c.exp10
    return result


@cython.cdivision(True)
cpdef conversion conversion_raise_to(conversion base, frac exponent) except *:
    """
    Returns a conversion that, if applied several times, would be roughly
    equivalent to the given conversion.

    Precision lose may be unavoidable when performing roots, but we do try to
    use exact results when possible.
    """
    if exponent.numer < 0:
        base = inverse_conversion(base)
        exponent.numer *= -1

    cdef conversion result
    if exponent.denom == 1:
        # No need to do fancy checking for loss of precision.
        result.factor = c_pow(base.factor, exponent.numer)
        result.ratio.numer = round(c_pow(base.ratio.numer, exponent.numer))
        result.ratio.denom = round(c_pow(base.ratio.denom, exponent.numer))
        result.exp10 = base.exp10 * exponent.numer
        return result

    cdef double exponent_double = frac_to_double(exponent)

    result.factor = c_pow(base.factor, exponent_double)

    cdef double numer_p = c_pow(base.ratio.numer, exponent_double)
    result.ratio.numer = round(numer_p)
    if (c_pow(result.ratio.numer, exponent.denom)
            != c_pow(base.ratio.numer, exponent.numer)):
        result.ratio.numer = 1
        result.factor *= numer_p

    cdef double denom_p = c_pow(base.ratio.denom, exponent_double)
    result.ratio.denom = round(denom_p)
    if (c_pow(result.ratio.denom, exponent.denom)
            != c_pow(base.ratio.denom, exponent.numer)):
        result.ratio.denom = 1
        result.factor /= denom_p

    if base.exp10 % exponent.denom == 0:
        result.exp10 = base.exp10 /  exponent.denom * exponent.numer
    else:
        result.exp10 = 0
        result.factor *= c_pow(10.0, base.exp10 * exponent_double)

    return result
