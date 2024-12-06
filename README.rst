## TUnits - Fast Python Units

.. image:: https://github.com/quantumlib/TypedUnits/actions/workflows/ci.yml/badge.svg?branch=main
  :target: https://github.com/quantumlib/TypedUnits
  :alt: Build & Test


Implements unit of measurement arithmetic, where a number is associated with a product of powers of base units and values with compatible units can be added.

The library is written in Cython for speed. The library provides the ability to statically check dimensionality type (see below) and a limited protobuffer serialization support for select units (see below). Contributions to extend support for more units are welcome.

A precompiled wheel can be installed using `pip install typedunits [--pre]`.

## Example

```python
>> import tunits
>> from tunits.units import meter, km, N, MHz

>> print(3*MHz)
Frequency(3, 'MHz')

>> print(5*meter + km)
Length(1005.0, 'm')

>> print(N/meter)
N/m

>> print((N/meter).in_base_units())
kg/s^2

>> print(2*km / tunits.Value(3, 's'))
0.666666666667 km/s
```

## Static Type Check

TypedUnits provides the ability to statically check the dimensionality of variables and parameters. For example mypy would complain about incompatible types for the following code.

```py3
from tunits import Frequency, LengthArray
from tunits.units import meter, km, MHz

def frequency_fn(f: Frequency) -> Frequency:
    return 3*f

x = 2 * meter
x_arr = LengthArray([1, 2], km)
y = 3 * km
f = 7 * MHz

# okay
print(frequency_fn(f))
print(x + y)
print(x_arr - y)

# not okay
print(frequency_fn(x))
print(f + x)
print(x - f)
frequency_fn(x_arr)
```

```sh
$ mypy my_code.py
my_code.py:18: error: Argument 1 to "frequency_fn" has incompatible type "Length"; expected "Frequency"  [arg-type]
my_code.py:19: error: No overload variant of "__add__" of "Value" matches argument type "Length"  [operator]
my_code.py:19: note: Possible overload variants:
my_code.py:19: note:     def __add__(self, int | float | complex | number[Any], /) -> Frequency
my_code.py:19: note:     def __add__(self, ValueArray | list[Any] | tuple[Any] | ndarray[Any, dtype[Any]], /) -> ValueArray
my_code.py:19: note:     def __add__(self, Frequency, /) -> Frequency
my_code.py:20: error: No overload variant of "__sub__" of "Value" matches argument type "Frequency"  [operator]
my_code.py:20: note: Possible overload variants:
my_code.py:20: note:     def __sub__(self, int | float | complex | number[Any], /) -> Length
my_code.py:20: note:     def __sub__(self, list[Any] | tuple[Any] | ndarray[Any, dtype[Any]], /) -> ValueArray
my_code.py:20: note:     def __sub__(self, Length, /) -> Length
my_code.py:21: error: Argument 1 to "frequency_fn" has incompatible type "LengthArray"; expected "Frequency"  [arg-type]
Found 4 errors in 1 file (checked 1 source file)
```


## Serialization support
TypedUnits provides protobuffer serialization support for [selected units](https://github.com/quantumlib/TypedUnits/blob/main/tunits/proto/tunits.proto#L22). Contributions are welcome to increase serialization coverage.

```py3
>> from tunits import Frequency
>> from tunits.units import MHz
>>
>> v = 3*MHz
>> msg = v.to_proto()
>> print(msg)
units {
  unit: HERTZ
  scale: MEGA
}
real_value: 3

>> Frequency.from_proto(msg)
Frequency(3.0, 'MHz')
```

# Installation

1. To install a precompiled wheel (add `--pre` for prelease version)

    ```
    pip install typedunits # [--pre] 
    ```

1. To locally build the latest version from the main branch

    ```bash
    pip install git+https://github.com/quantumlib/TypedUnits
    ```

1. For a local editable copy

    ```bash
    git clone https://github.com/quantumlib/TypedUnits
    cd TypedUnits
    pip install -e .
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
    pip install -e .
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
