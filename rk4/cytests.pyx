__author__ = 'nikolas'

import numpy as np
cimport numpy as cnp
cimport rk4.cyrk4 as ccrk4
import rk4.cyrk4 as crk4
from cpython.ref cimport PyObject

cdef struct params_t:
    double kk

def ode_python(y,t, kk):
    return [y[1], -kk * y[0]]

def ode_python_callback(t, y, ydot, params):
    kk, = params
    ydot[0] = y[1]
    ydot[1] = -kk * y[0]

def sde_python_callback(t, y, w, ydot, params):
    kk, = params
    ydot[0] = y[1]
    ydot[1] = -kk * y[0] + w[0]


cdef void ode_c(double t, double y[], double ydot[], void *params):
    ydot[0] = y[1]
    ydot[1] = -(<params_t*>params)[0].kk * y[0]

cdef void sde_c(double t, double y[], double w[], double ydot[], void * params):
    ode_c(t, y, ydot, params)
    ydot[1] += w[0]


from scipy.integrate import odeint


def test_ode():
    cdef params_t p
    p.kk = 5.
    times = np.linspace(0, .5, 101)
    y0 = np.array([100., 0])

    ret1 = ccrk4.integrate_ode_c_simple(ode_c, times, y0, .00001, <void*> &p)
    ret2 = odeint(ode_python, y0, times, args=(5.,))
    assert np.allclose(ret1, ret2)

def test_create_ode_callback():
    y0 = np.array([100., 0])
    cdef:
        cnp.ndarray y0a = np.ascontiguousarray(y0, np.float64)
        ccrk4.py_callback pc
    params = (5., )
    pc.fn = <PyObject*>ode_python_callback
    pc.pyparams = <PyObject*>params
    pc.m = len(y0)
    pc.q = 0

    y = np.copy(y0)
    ydot = np.zeros(2)
    ccrk4.callback_ode(0., <double *>cnp.PyArray_DATA(y), <double *> cnp.PyArray_DATA(ydot), <void*> &pc)
    assert np.allclose(ydot, ode_python(y, 0., params[0]))


def test_ode_callback():
    times = np.linspace(0, .1, 101)
    y0 = np.array([2., 0])
    ret1 = crk4.integrate_ode(ode_python_callback, times, y0, .0001, (5.,))
    ret2 = odeint(ode_python, y0, times, args=(5.,))
    assert np.allclose(ret1, ret2)

def test_sample_gaussian():
    nsamples = 100000
    vec = np.ones(nsamples)
    ccrk4.sample_gaussian(<double*>cnp.PyArray_DATA(vec), nsamples, 1.)
    assert abs(np.mean(vec)) < 1e-1

def test_sde():
    cdef params_t p
    p.kk = 5.
    times = np.linspace(0, .1, 101)
    y0 = np.array([100., 0])
    trials = 100
    retsum = np.zeros((len(times), 2))
    wtsum = np.zeros((len(times), 1))
    for kk in range(trials):
        ret = ccrk4.integrate_sde_c_simple(sde_c, times, y0, .0001, 1, <void*> &p)
        retsum += ret[0]
        wtsum += ret[1]
    ret2 = odeint(ode_python, y0, times, args=(p.kk,))
    assert np.allclose(retsum/trials, ret2, rtol=1e-1)



def test_sde_callback():
    times = np.linspace(0, .1, 101)
    y0 = np.array([100., 0])
    trials = 100
    retsum = np.zeros((len(times), 2))
    wtsum = np.zeros((len(times), 1))
    for kk in range(trials):
        ret = crk4.integrate_sde(sde_python_callback, times, y0, .0001, 1, (5.,))
        retsum += ret[0]
        wtsum += ret[1]

    ret2 = odeint(ode_python, y0, times, args=(5.,))
    assert np.allclose(retsum/trials, ret2, rtol=1e-1)
