## pythonunits

Implements unit of measurement arithmetic, where a number is associated with a product of powers of base units and values with compatible units can be added.

Defines SI units, SI prefixes, and some derived units.

## Example

```python
from fastunits import Value, meter

print Value(2, 'km') / Value(3, 's')
print 3*meter + 5*meter
```
# Building

In the `pythonunits` directory, run:

    python setup.py build_ext --inplace
