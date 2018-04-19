#!/usr/bin/env python

from math import *
import SwarmPackagePy
from SwarmPackagePy import testFunctions as tf
import pytest


def ackley_function(x):
    return -exp(-sqrt(0.5*sum([i**2 for i in x]))) - \
           exp(0.5*sum([cos(i) for i in x])) + 1 + exp(1)


def bukin_function(x):
    return 100*sqrt(abs(x[1]-0.01*x[0]**2)) + 0.01*abs(x[0] + 10)


def cross_in_tray_function(x):
    return round(-0.0001*(abs(sin(x[0])*sin(x[1])*exp(abs(100 -
                            sqrt(sum([i**2 for i in x]))/pi))) + 1)**0.1, 7)


def sphere_function(x):
    return sum([i**2 for i in x])


def bohachevsky_function(x):
    return x[0]**2 + 2*x[1]**2 - 0.3*cos(3*pi*x[0]) - 0.4*cos(4*pi*x[1]) + 0.7


def sum_squares_function(x):
    return sum([(i+1)*x[i]**2 for i in range(len(x))])


def sum_of_different_powers_function(x):
    return sum([abs(x[i])**(i+2) for i in range(len(x))])


def booth_function(x):
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2


def matyas_function(x):
    return 0.26*sphere_function(x) - 0.48*x[0]*x[1]


def mccormick_function(x):
    return sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1


def dixon_price_function(x):
    return (x[0] - 1)**2 + sum([(i+1)*(2*x[i]**2 - x[i-1])**2
                                for i in range(1, len(x))])


def six_hump_camel_function(x):
    return (4 - 2.1*x[0]**2 + x[0]**4/3)*x[0]**2 + x[0]*x[1]\
           + (-4 + 4*x[1]**2)*x[1]**2


def three_hump_camel_function(x):
    return 2*x[0]**2 - 1.05*x[0]**4 + x[0]**6/6 + x[0]*x[1] + x[1]**2


def easom_function(x):
    return -cos(x[0])*cos(x[1])*exp(-(x[0] - pi)**2 - (x[1] - pi)**2)


def michalewicz_function(x):
    return -sum([sin(x[i])*sin((i+1)*x[i]**2/pi)**20 for i in range(len(x))])


def beale_function(x):
    return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + \
           (2.625 - x[0] + x[0]*x[1]**3)**2


def drop_wave_function(x):
    return -(1 + cos(12*sqrt(sphere_function(x))))/(0.5*sphere_function(x) + 2)

def pytest_generate_tests(metafunc):
    fdata = [
        # function                          min value  dim      domain
        (tf.cross_in_tray_function,          -2.06261,   2, [-10,   10]),
        (tf.sphere_function,                        0,   2, [-5.12, 5.12]),
        (tf.bohachevsky_function,                   0,   2, [-10,   10]),
        (tf.sum_of_different_powers_function,       0,   2, [-1,    1]),
        (tf.booth_function,                         0,   2, [-10,   10]),
        (tf.matyas_function,                        0,   2, [-10,   10]),
        (tf.six_hump_camel_function,          -1.0316,   2, [-3,    3]),
        (tf.three_hump_camel_function,              0,   2, [-5,    5]),
        (tf.easom_function,                        -1,   2, [-30,   30]),
    ]
    algs = [
        # (algorithm, number of iterations, agents, additional parameters)
        (SwarmPackagePy.aba, 100,  60, []),
        (SwarmPackagePy.ba,  100,  60, []),
        (SwarmPackagePy.bfo, 100,  20, []),
        (SwarmPackagePy.ca,  500,  60, []),
        (SwarmPackagePy.cso, 50,  150, []),
        (SwarmPackagePy.chso, 150, 60, []),
        (SwarmPackagePy.fa,  50,   60, []),
        (SwarmPackagePy.fwa, 20,    3, []),
        (SwarmPackagePy.gsa, 100,  60, []),
        (SwarmPackagePy.gwo, 100,  60, []),
        (SwarmPackagePy.hs,  1500, 60, []),
        (SwarmPackagePy.pso, 50,   60, []),
        (SwarmPackagePy.ssa, 150,  60, []),
        (SwarmPackagePy.wsa, 45,   45, []),
    ]
    tdata = []
    for alg in algs:
        for f in fdata:
            tdata.append((*alg, *f))

    metafunc.parametrize(["alg", "iter", "agents", "params", "f", "val", "dim",
                          "dom"], tdata)


def test_alg(alg, agents, iter, params, f, val, dim, dom):
    alh = alg(agents, f, *dom, dim, iter, *params)
    # print(f(alh.get_Gbest()), val)
    assert abs(f(alh.get_Gbest()) - val) < 0.1
