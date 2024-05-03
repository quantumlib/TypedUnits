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


# A ratio that should always be canonicalized into least terms with the sign
# on the numerator.
cdef struct frac:
    long long numer
    long long denom


# Returns the greatest divisor that both inputs share.
@cython.cdivision(True)
cpdef long long gcd(long long a, long long b):
    if a < 0 or b < 0:
       return gcd(abs(a), abs(b))
    while b != 0:
        a, b = b, a%b
    return a

# Returns an equivalent fraction, without common factors between numerator and
# denominator and with the negative sign on the numerator (if present).
@cython.cdivision(True)
cpdef frac frac_least_terms(long long numer, long long denom) except *:
    if denom < 0:
        return frac_least_terms(-numer, -denom)
    if denom == 0:
        raise ZeroDivisionError()
    cdef long long d = gcd(numer, denom)
    cdef frac f
    f.numer = numer / d
    f.denom = denom / d
    return f


# Returns the product of the two given fractions, in least terms.
@cython.cdivision(True)
cpdef frac frac_times(frac a, frac b):
    cdef long long d1 = gcd(a.numer, b.denom)
    cdef long long d2 = gcd(b.numer, a.denom)
    cdef frac f
    f.numer = (a.numer/d1) * (b.numer/d2)
    f.denom = (a.denom/d2) * (b.denom/d1)
    return f


@cython.cdivision(True)
cdef frac frac_div_not_zero(frac a, frac b):
    cdef long long d1 = gcd(a.numer, b.numer)
    cdef long long d2 = gcd(b.denom, a.denom)
    cdef frac f
    f.numer = (a.numer/d1) * (b.denom/d2)
    f.denom = (a.denom/d2) * (b.numer/d1)
    if f.denom < 0:
        f.numer *= -1
        f.denom *= -1
    return f


# Returns the quotient of the two given fractions, in least terms.
@cython.cdivision(True)
cpdef frac frac_div(frac a, frac b) except *:
    if b.numer == 0:
        raise ZeroDivisionError()
    return frac_div_not_zero(a, b)


# Recognizes floats corresponding to twelths. Returns them as a fraction.
cpdef frac float_to_twelths_frac(a) except *:
    if isinstance(a, int):
        return frac_least_terms(a, 1)

    cdef double d = float(a)
    cdef long long x = <long long>c_floor(12*d + 0.5)
    if abs(12*d - x) > 1e-5:
        raise ValueError("Not a twelfth.")

    return frac_least_terms(x, 12)


# Converts a fraction to a double approximating its value.
cpdef double frac_to_double(frac f):
    return <double>f.numer / <double>f.denom
