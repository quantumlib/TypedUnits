import numpy as np
import pyfu
import random
import time
import unittest

_perf_goal_results = []
def perf_goal(avg_micros, repeats=100, args=None):
    """
    A decorator that turns a function into a perf test.
    :param avg_micros: Maximum acceptable average duration, in microseconds.
    :param repeats: Number of times to sample the function's running time.
    :param args: Arguments to pass into the method.
    """
    if args is None:
        args = []
    index = len(_perf_goal_results)
    _perf_goal_results.append(None)

    def decorate(f):
        def wrapped():
            try:
                _perf_goal_results[index] = '[fail] ' + f.__name__

                total = 0.0
                for _ in range(repeats):
                    ctx = dict()
                    arg_vals = [e.sample_function(ctx)
                                if isinstance(e, Sample)
                                else e
                                for e in args]
                    start_time = time.clock()
                    f(*arg_vals)
                    duration = time.clock() - start_time
                    total += duration
                mean_duration = total / repeats

                did_fail = mean_duration * 10**6 > avg_micros
                _perf_goal_results[index] = (
                    "averaged %d%%%s of target (%s us) for %s" % (
                        mean_duration * 10 ** 8 / avg_micros,
                        ' (!)' if did_fail else '',
                        avg_micros,
                        f.__name__))

                if did_fail:
                    raise AssertionError(
                        "%s took too long. Mean (%s us) over target (%s us)." %
                        (f.__name__, mean_duration * 10 ** 6, avg_micros))
            finally:
                # Because tests can run out of order, we defer the printing
                # until we have all the results and can print in order.
                if all(e is not None for e in _perf_goal_results):
                    print('')
                    print('-------------')
                    print('perf results:')
                    print('-------------')
                    for r in _perf_goal_results:
                        print(r)
                    print('-------------')

        return wrapped
    return decorate


class Sample:
    """
    Recognized by perf_goal as an argument that should vary.
    """
    def __init__(self, sample_function):
        self.sample_function = sample_function

unit_list = [v for k, v in pyfu.unit.default_unit_database.known_units.items()]


def sample_random_unit_combo():
    r = random.random()
    r *= random.choice(unit_list)
    r *= random.choice(unit_list)
    r /= random.choice(unit_list)
    if random.random() > 0.5:
        r *= random.choice(unit_list)
    while r.isDimensionless():
        r *= random.choice(unit_list)
    return r


def sample_matching_combo_sampler(ctx):
    key = 'a_compatible_unit'
    if key not in ctx:
        ctx[key] = sample_random_unit_combo()
    return ctx[key] * random.random()

a_random_unit_val = Sample(lambda _: sample_random_unit_combo())
a_random_compatible_unit_val = Sample(sample_matching_combo_sampler)

a_random_unit_array = Sample(lambda _:
                             np.array([random.random() for _ in range(1024)]) *
                             sample_random_unit_combo())

a_random_compatible_unit_array = Sample(
    lambda ctx:
    np.array([random.random() for _ in range(1024)]) *
    sample_matching_combo_sampler(ctx))


# ----- Value tests ------

@perf_goal(avg_micros=5, args=[a_random_compatible_unit_val] * 2)
def test_perf_add(a, b):
    return a + b


@perf_goal(avg_micros=5, args=[a_random_unit_val])
def test_perf_scale(a):
    return a * 3.14


@perf_goal(avg_micros=5, args=[a_random_unit_val] * 2)
def test_perf_multiply(a, b):
    return a * b


@perf_goal(avg_micros=5, args=[a_random_compatible_unit_val] * 2)
def test_perf_get_item(a, b):
    return a[b]


@perf_goal(avg_micros=5, args=[a_random_compatible_unit_val] * 2)
def test_perf_divmod(a, b):
    return divmod(a, b)


@perf_goal(avg_micros=20, args=[a_random_compatible_unit_val] * 2)
def test_perf_import_multiply_add_heterogeneous(a, b):
    from pyfu.units import kilometer, inch
    return a * kilometer + b * inch


@perf_goal(avg_micros=20, args=[a_random_unit_val])
def test_perf_str(a):
    return str(a)


# Note: another PR improves this 20X.
@perf_goal(avg_micros=500, args=[a_random_unit_val])
def test_perf_repr(a):
    return repr(a)


# ----- Array tests ------

@perf_goal(avg_micros=15, args=[a_random_compatible_unit_array] * 2)
def test_perf_add_arrays(a, b):
    return a + b


@perf_goal(avg_micros=15, args=[a_random_compatible_unit_array,
                                a_random_compatible_unit_val])
def test_perf_add_value_array(a, b):
    return a + b


@perf_goal(avg_micros=15, args=[a_random_unit_array] * 2)
def test_perf_multiply_arrays(a, b):
    return a * b


@perf_goal(avg_micros=15, args=[a_random_compatible_unit_array,
                                a_random_compatible_unit_val])
def test_perf_multiply_value_array(a, b):
    return a * b


if __name__ == "__main__":
    unittest.main()
