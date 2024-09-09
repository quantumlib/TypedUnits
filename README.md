## TUnits - Fast Python Units

| GCB CI status: |
| ---------- |
| [![GCB Build Status](https://storage.googleapis.com/qh-build-badges/builds/pyfu/branches/master.svg)](https://pantheon.corp.google.com/cloud-build/builds?project=qh-build&query=trigger_id%3D%22736396b1-c130-4936-adf5-cd9c7be2b492%22) |

Implements unit of measurement arithmetic, where a number is associated with a product of powers of base units and values with compatible units can be added.

Defines SI units, SI prefixes, and some derived units.

## Example

```python
import tunits
from tunits.units import meter, km, N

print(5*meter + km)
# 1005.0 meter

print(N/meter)
# N/meter

print((N/meter).inBaseUnits())
# kg/s^2

print(2*km / tunits.Value(3, 's'))
# 0.666666666667 km/s
```

# Installation

1. To install the latest version from the main branch

    ```bash
    pip install git+https://github.com/quantumlib/TypedUnits
    ```

1. For a local editable copy

    ```bash
    git clone https://github.com/quantumlib/TypedUnits
    cd TypedUnits
    pip install .
    ```

# Development and Testing

1. Clone the repository.

    ```bash
    git clone https://github.com/quantumlib/TypedUnits

    cd TypedUnits
    ```

    *All future steps assume you are in the repository's directory.*

1. Install dev environment dependencies.

    ```bash
    pip install -r dev_tools/dev.env.txt
    ```

1. Install TUnits

    ```bash
    pip install .
    ```

1. Test.

    ```bash
    pytest
    ```


## Formatting

```bash
dev_tools/format.sh  # to format
dev_tools/format.sh --check  # to verify format
```

---

**Note:** This is not an officially supported Google product