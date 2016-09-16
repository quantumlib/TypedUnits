#!/usr/bin/python

import unittest
import random
import time
import numpy as np

import fastunits as U


def perf_goal(avg_micros, repeats=100):
    """
    A decorator that turns a function into a perf test.
    :param avg_micros: Maximum acceptable average duration, in microseconds.
    :param repeats: Number of times to sample the function's running time.
    """
    def decorate(f):
        def wrapped():
            total = 0.0
            for _ in range(repeats):
                start_time = time.clock()
                f()
                duration = time.clock() - start_time
                total += duration
            mean_duration = total / repeats
            if mean_duration*10**6 > avg_micros:
                raise AssertionError(
                    "%s took too long. Mean (%s us) exceeded target (%s us)." %
                        (f.__name__, mean_duration * 10 ** 6, avg_micros))

            print("averaged %d%% of target for %s" %
                  (mean_duration * 10 ** 8 / avg_micros, f.__name__))
        return wrapped
    return decorate


# noinspection PyProtectedMember
unit_list = [v for k, v in U.unit._unit_cache.items()]


def random_unit():
    return random.choice(unit_list)


@perf_goal(avg_micros=15)
def test_perf_multiply_units():
    u1 = random_unit()
    u2 = random_unit()
    return u1 * u2


@perf_goal(avg_micros=20)
def test_perf_multiply_values():
    u1 = random_unit()
    u2 = random_unit()
    return (1.0*u1) * (1.0*u2)


@perf_goal(avg_micros=10)
def test_perf_multiply_cached_units():
    return U.ns * U.GHz


@perf_goal(avg_micros=15)
def test_perf_multiply_cached_values():
    return (1.0 * U.ns) * (1.0 * U.GHz)


@perf_goal(avg_micros=20)
def test_perf_add_values_with_same_cached_unit():
    return (1*U.m) + (1*U.m)


@perf_goal(avg_micros=20)
def test_perf_add_values_with_different_cached_units():
    return (1*U.nm) + (1*U.m)


@perf_goal(avg_micros=100*1000, repeats=1)
def test_perf_envelope_unit():
    n = 1000
    t = np.arange(20*n)*U.ns
    t_cos = np.arange(20)*U.ns
    w_cos = np.pi * 2 * U.GHz / 20
    z = np.zeros(20*n, dtype=np.complex128)
    for i in range(n):
        w = random.random() * .1 * U.GHz
        phi = random.random()*np.pi
        z[i*20:(i+1)*20] = \
            (1-np.cos(t_cos*w_cos))*np.exp(1j*w*t[i*20:(i+1)*20]+phi)
    return z


@perf_goal(avg_micros=100*1000, repeats=1)
def test_perf_envelope_no_unit():
    n = 1000
    t = np.arange(20*n)
    t_cos = np.arange(20)
    w_cos = np.pi * 2 / 20
    z = np.zeros(20*n, dtype=np.complex128)
    for j in range(n):
        w = random.random() * .1
        phi = random.random() * np.pi
        z[j*20:(j+1)*20] = \
            (1-np.cos(t_cos*w_cos))*np.exp(1j*w*t[j*20:(j+1)*20]+phi)
    return z


@perf_goal(avg_micros=100)
def test_perf_multiply_unit_array():
    a1 = np.arange(1000)*U.ns
    a2 = np.arange(1000)*U.GHz
    return a1 * a2


if __name__ == "__main__":
    unittest.main()
