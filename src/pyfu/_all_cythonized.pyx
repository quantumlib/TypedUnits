# Combines all the .pyx files together, so they are compiled and optimized as
# a single thing (and have access to their respective fast cdefs).

include "cython/frac.pyx"
include "cython/conversion.pyx"
include "cython/unit_array.pyx"
include "cython/unit_mismatch_error.pyx"
include "cython/with_unit.pyx"
include "cython/with_unit_complex.pyx"
include "cython/with_unit_value.pyx"
include "cython/with_unit_value_array.pyx"
