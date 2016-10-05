"""
Exposes standard units as module variables.

For example, '1/units.millisecond' is equal to 'units.kHz'.
"""

import unit as __unit

# Expose defined units (e.g. 'meter', 'km', 'day') as module variables.
for k, v in __unit.default_unit_database.known_units.items():
    globals()[k] = v
