# Combines all the .pyx files together, so they are compiled and optimized as
# a single thing (and have access to their respective fast cpdefs).

include "frac.pyx"
include "unit_array.pyx"
include "with_unit.pyx"
include "other_types.pyx"
