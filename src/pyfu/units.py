"""
Exposes standard units as module variables.

For example, '1/units.millisecond' is equal to 'units.kHz'.
"""

from . import unit as __unit

# Expose defined units (e.g. 'meter', 'km', 'day') as module variables.
# Note: unless you like overwriting Boltzmann's constant, keep the underscores.
for _k, _v in __unit.default_unit_database.known_units.items():
    globals()[_k] = _v
