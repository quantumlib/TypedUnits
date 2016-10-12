import numpy as np
import pyfu
import random
import time


def _perf_bar_text(avg, stddev, n=20):
    """
    Returns an ascii progress bar showing the average relative to 1 with '#'s
    and two standard deviations with '~'s. E.g. '#####~~~~~       '.
    """
    if avg > 1:
        return '!' * n
    n_avg = int(avg * n)
    n_2dev = min(int((avg + stddev * 2) * n), n) - n_avg
    return ('#' * n_avg + '~' * n_2dev).ljust(n, ' ')


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
        name = f.__name__.replace('test_perf_', '')

        def wrapped():
            try:
                _perf_goal_results[index] = '[fail] ' + name

                total = 0.0
                squares_total = 0.0
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
                    squares_total += duration**2

                mean_duration = total / repeats
                # Note: keep in mind that perf measurements aren't normally
                # distributed, and are more volatile than this number implies.
                std_dev = (squares_total / repeats - mean_duration**2)**0.5

                avg_ratio = mean_duration * 10**6 / avg_micros
                std_dev_ratio = std_dev * 10**6 / avg_micros
                did_fail = avg_ratio > 1
                _perf_goal_results[index] = (
                    u"[%s] %s%% \u00B1%s%% of target (%s us) for %s" % (
                        _perf_bar_text(avg_ratio, std_dev_ratio),
                        str(int(avg_ratio * 100)).rjust(3, ' '),
                        str(int(std_dev_ratio * 100)).rjust(2, ' '),
                        str(avg_micros).rjust(3),
                        name))

                if did_fail:
                    raise AssertionError(
                        "%s took too long. Mean (%s us) over target (%s us)." %
                        (name, mean_duration * 10 ** 6, avg_micros))
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
