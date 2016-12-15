## Pyfu - Fast Python Units

| [pytest](https://ci.sanieldank.com/job/pyfu-pytest-master) | [pylint](https://ci.sanieldank.com/job/pyfu-pylint-all-master) |
| ------------- |-------------|
| [![pytest](https://ci.sanieldank.com/buildStatus/icon?job=pyfu-pytest-master)](https://ci.sanieldank.com/job/pyfu-pytest-master/lastCompletedBuild/testReport/) | [![pylint](https://ci.sanieldank.com/buildStatus/icon?job=pyfu-pylint-all-master)](https://ci.sanieldank.com/job/pyfu-pylint-all-master/lastCompletedBuild/console/) | 

Implements unit of measurement arithmetic, where a number is associated with a product of powers of base units and values with compatible units can be added.

Defines SI units, SI prefixes, and some derived units.

## Example

```python
import pyfu
from pyfu.units import meter, km, N

print(5*meter + km)
# 1005.0 meter

print(N/meter)
# N/meter

print((N/meter).inBaseUnits())
# kg/s^2

print(2*km / pyfu.Value(3, 's'))
# 0.666666666667 km/s
```

# Building

0. Clone the repository.

    ```bash
    git clone https://github.com/martinisgroup/pyfu.git

    cd pyfu
    ```

    *All future steps assume you are in the repository's directory.*

0. Install system dependencies.

    ```bash
    cat apt_dependency_list.txt | xargs sudo apt-get install --yes
    ```

0. Install python dependencies.

    ```bash
    pip install -r requirements.txt
    ```

0. Cythonize.

    ```bash
    python setup.py build_ext --inplace
    ```

0. Test.

    ```bash
    pytest
    ```
