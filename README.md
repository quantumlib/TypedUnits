## Pyfu - Fast Python Units

| GCB CI status: |
| ---------- |
| [![GCB Build Status](https://storage.googleapis.com/qh-build-badges/builds/pyfu/branches/master.svg)](https://pantheon.corp.google.com/cloud-build/builds?project=qh-build&query=trigger_id%3D%22736396b1-c130-4936-adf5-cd9c7be2b492%22) |

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
    git clone https://github.com/qh-lab/pyfu.git

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

# Installation

```
    python setup.py install
```


## Formatting

```bash
scripts/format.sh  # to format
scripts/format.sh --check  # to verify format
```
