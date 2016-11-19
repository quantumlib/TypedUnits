## Pyfu - Fast Python Units

| [pytest](https://matrix-reloaded.physics.ucsb.edu/teamcity/viewType.html?buildTypeId=Pyle_PythonTests)          | [pylint](https://matrix-reloaded.physics.ucsb.edu/teamcity/viewType.html?buildTypeId=Pyle_PythonPerformanceTests)    |
| ------------- |-------------|
| [![pytest status](https://matrix-reloaded.physics.ucsb.edu/teamcity/app/rest/builds/buildType:pythonunits_Pythonunits/statusIcon)](https://matrix-reloaded.physics.ucsb.edu/teamcity/project.html?projectId=pythonunits&tab=projectOverview) | [![pylint status](https://ci.sanieldank.com/buildStatus/icon?job=pyfu-pylint-all-master)](https://ci.sanieldank.com/job/pyfu-pylint-all-master) |

Implements unit of measurement arithmetic, where a number is associated with a product of powers of base units and values with compatible units can be added.

Defines SI units, SI prefixes, and some derived units.

## Example

```python
>>> import pyfu
>>> from pyfu.units import meter, km, N

>>> print(5*meter + km)
1005.0 meter

>>> print(N/meter)
N/meter

>>> print((N/meter).inBaseUnits())
kg/s^2

>>> print(2*km / pyfu.Value(3, 's'))
0.666666666667 km/s
```

# Building

1. **Install dependencies**

        pip install numpy
        pip install pytest
        pip install pyparsing
        pip install Cython

2. **Cythonize**

    `python setup.py build_ext --inplace`

3. **Test**

    `py.test`
