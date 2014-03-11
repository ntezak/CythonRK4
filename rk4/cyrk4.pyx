cimport cython
import numpy as np
cimport numpy as np
from cpython.ref cimport PyObject

from libc.math cimport sqrt
from libc.stdlib cimport malloc, free
from cython_gsl cimport gsl_rng_type, gsl_rng, gsl_rng_mt19937, gsl_rng_alloc, gsl_ran_gaussian


cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void sample_gaussian(double * ptr,  Py_ssize_t Np, double sigma):
    cdef Py_ssize_t kk
    for kk in range(Np):
        ptr[kk] = gsl_ran_gaussian(r, sigma)



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void integrate_ode_c(ode_t ode, Py_ssize_t nt, double tlist[], Py_ssize_t neqs, double y0[], double yts[],  double hmax, void * params):
    """
    Integrate an ode with initial state y0[0:neqs] and fill yts[0:nt, 0:neqs] with its values at times specified in tlist[0:nt].

    The method employs a fixed stepsize Runge Kutta 4th order scheme with stepsize specified by hmax.
    """
    cdef:
        Py_ssize_t kk, ll
        double t, h
        double * yt = <double*> malloc(neqs * sizeof(double))
        double * k1 = <double*> malloc(neqs * sizeof(double))
        double * k2 = <double*> malloc(neqs * sizeof(double))
        double * k3 = <double*> malloc(neqs * sizeof(double))
        double * k4 = <double*> malloc(neqs * sizeof(double))


    t = tlist[0]
    copy_array(y0,  yt, neqs)
    copy_array(y0,  &yts[0], neqs)

    for kk in range(1, nt):
        copy_array(yt, &yts[kk * neqs], neqs)
        h = hmax
        while t < tlist[kk]:
            if tlist[kk] - t < hmax:
                h = tlist[kk] - t

            ode(t, yt, k1, params)

            add_to(yt, k1, neqs, .5 * h)
            t += .5 * h
            ode(t, yt, k2, params)

            for ll in range(neqs):
                yt[ll] = yts[kk * neqs + ll] + .5 * h * k2[ll]
            ode(t, yt, k3, params)

            t += .5 * h
            for ll in range(neqs):
                yt[ll] = yts[kk * neqs + ll] + h * k3[ll]
            ode(t, yt, k4, params)

            for ll in range(neqs):
                yt[ll] = yts[kk * neqs + ll] +  h/6. * (k1[ll] + 2. * (k2[ll] + k3[ll]) + k4[ll])

            copy_array(yt, &yts[kk * neqs], neqs)

            if h < hmax:
                break
    free(yt)
    free(k1)
    free(k2)
    free(k3)
    free(k4)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void integrate_sde_c(sde_t sde, Py_ssize_t nt, double tlist[], Py_ssize_t neqs, double y0[], double yts[], Py_ssize_t nw, double wts[], double hmax, void * params):
    """
    Integrate an sde with initial state y0[0:neqs] and fill yts[0:neqs] with its values at times specified in tlist[0:nt] driven by a nw real gaussian noises.

    The noises are held constant over each RK4 step, which is technically not correct,
    but yields a more stable method for obtaining fast results for systems with limited bandwidth.
    The method employs a fixed stepsize Runge Kutta 4th order scheme with stepsize specified by hmax.
    The noise increments integrated over each time interval are stored in wts[0:nt, 0:nw]
    """
    cdef:
        Py_ssize_t kk, ll
        double t, h, sigma

        double * yt = <double*> malloc(neqs * sizeof(double))
        double * k1 = <double*> malloc(neqs * sizeof(double))
        double * k2 = <double*> malloc(neqs * sizeof(double))
        double * k3 = <double*> malloc(neqs * sizeof(double))
        double * k4 = <double*> malloc(neqs * sizeof(double))
        double * wt = <double*> malloc(nw * sizeof(double))

    t = tlist[0]
    copy_array(y0,  yt, neqs)
    copy_array(y0,  &yts[0], neqs)
    for kk in range(1, nt):
        copy_array(yt, &yts[kk * neqs], neqs)
        h = hmax
        sigma = 1./sqrt(h)
        while t < tlist[kk]:
            if tlist[kk] - t < hmax:
                h = tlist[kk] - t
                sigma = 1./sqrt(h)
                # print(h, sigma)

            sample_gaussian(wt, nw, sigma)

            sde(t, yt, wt, k1, params)

            add_to(yt, k1, neqs, .5 * h)
            t += .5 * h
            sde(t, yt, wt, k2, params)

            for ll in range(neqs):
                yt[ll] = yts[kk * neqs + ll] + .5 * h * k2[ll]
            sde(t, yt, wt, k3, params)


            t += .5 * h
            for ll in range(neqs):
                yt[ll] = yts[kk * neqs + ll] + h * k3[ll]
            sde(t, yt, wt, k4, params)

            for ll in range(neqs):
                yt[ll] = yts[kk * neqs + ll] +  h/6. * (k1[ll] + 2. * (k2[ll] + k3[ll]) + k4[ll])

            copy_array(yt, &yts[kk * neqs], neqs)
            add_to(&wts[(kk-1) * nw], wt, nw, h)

            if h < hmax:
                t = tlist[kk]
                break
    free(yt)
    free(k1)
    free(k2)
    free(k3)
    free(k4)
    free(wt)



cdef void callback_ode(double t, double y[], double ydot[], void * params):
    cdef:
        py_callback * pp = <py_callback*> params
        cython.view.array yarr = <double[:pp[0].m]> y
        cython.view.array ydotarr = <double[:pp[0].m]> ydot
    (<object>(pp[0].fn))(t, yarr, ydotarr, <object>(pp[0].pyparams))


cdef void callback_sde(double t, double y[], double w[], double ydot[], void * params):
    cdef:
        py_callback * pp = <py_callback*> params
        cython.view.array yarr = <double[:pp[0].m]> y
        cython.view.array ydotarr = <double[:pp[0].m]> ydot
        cython.view.array warr = <double[:pp[0].q]> w
    (<object>pp[0].fn)(t, yarr, warr, ydotarr, <object>(pp[0].pyparams))


cdef np.ndarray integrate_ode_c_simple(ode_t ode, np.ndarray tlist, np.ndarray y0, double hmax, void *params):
    cdef:
        Py_ssize_t nt = len(tlist)
        Py_ssize_t neqs = len(y0)
        np.ndarray ret = np.zeros((nt, neqs))

    integrate_ode_c(ode, nt, <double*> np.PyArray_DATA(tlist),
                    neqs, <double*>np.PyArray_DATA(y0),
                    <double*>np.PyArray_DATA(ret), hmax, params)
    return ret

cdef object integrate_sde_c_simple(sde_t sde, np.ndarray tlist, np.ndarray y0, double hmax, Py_ssize_t q, void *params):
    cdef:
        Py_ssize_t nt = len(tlist)
        Py_ssize_t neqs = len(y0)
        np.ndarray ret = np.zeros((nt, neqs))
        np.ndarray wts = np.zeros((nt, q))

    integrate_sde_c(sde, nt, <double*> np.PyArray_DATA(tlist),
                    neqs, <double*>np.PyArray_DATA(y0),
                    <double*>np.PyArray_DATA(ret),
                    q, <double*>np.PyArray_DATA(wts),
                    hmax, params)
    return ret, wts


def integrate_ode(ode, tlist, y0, hmax, params=None):
    cdef:
        np.ndarray tlista = np.ascontiguousarray(tlist, np.float64)
        np.ndarray y0a = np.ascontiguousarray(y0, np.float64)
        py_callback pc
    pc.fn = <PyObject*>ode
    pc.pyparams = <PyObject*>params
    pc.m = len(y0)
    pc.q = 0
    return integrate_ode_c_simple(callback_ode, tlista, y0a, hmax, <void*>&pc)

def integrate_sde(sde, tlist, y0, hmax, q, params=None):
    cdef:
        np.ndarray tlista = np.ascontiguousarray(tlist, np.float64)
        np.ndarray y0a = np.ascontiguousarray(y0, np.float64)
        py_callback pc
    pc.fn = <PyObject*>sde
    pc.pyparams = <PyObject*>params
    pc.m = len(y0)
    pc.q = q
    return integrate_sde_c_simple(callback_sde, tlista, y0a, hmax, q, <void*>&pc)


########## COMPLEX INTERFACE ##########


cdef wrap_complex_params wrap(ode_complex_t ode, sde_complex_t sde, void * params):
    cdef wrap_complex_params p
    p.ode = ode
    p.sde = sde
    p.params = params
    return p


cdef void wrapped_complex_ode(double t, double y[], double ydot[], void * params):
    cdef wrap_complex_params * p = <wrap_complex_params*> params
    p[0].ode(t, <complex*>y, <complex*> ydot, p[0].params)

cdef void wrapped_complex_sde(double t, double y[], double w[], double ydot[], void * params):
    cdef wrap_complex_params * p = <wrap_complex_params*> params
    p[0].sde(t, <complex*>y, w, <complex*> ydot, p[0].params)


cdef void integrate_complex_ode_c(ode_complex_t ode, Py_ssize_t nt, double tlist[], Py_ssize_t neqs, complex y0[], complex yts[], double hmax, void * params):
    cdef:
        wrap_complex_params p = wrap(ode, NULL, params)
    integrate_ode_c(wrapped_complex_ode, nt, tlist, 2 * neqs, <double*> y0, <double*>yts, hmax, &p)


cdef void integrate_complex_sde_c(sde_complex_t sde, Py_ssize_t nt, double tlist[], Py_ssize_t neqs, complex y0[], complex yts[], Py_ssize_t nw, double wts[], double hmax, void * params):
    cdef:
        wrap_complex_params p = wrap(NULL, sde, params)
    integrate_sde_c(wrapped_complex_sde, nt, tlist, 2 * neqs, <double*> y0, <double*>yts, nw, wts, hmax, &p)

cdef np.ndarray integrate_complex_ode_c_simple(ode_complex_t ode, np.ndarray tlist, np.ndarray y0, double hmax, void *params):
    cdef:
        wrap_complex_params p = wrap(ode, NULL, params)
    y0 = np.astype(y0, np.complex128)
    return integrate_ode_c_simple(wrapped_complex_ode, tlist, y0.view(np.float64), hmax, &p)

cdef object integrate_complex_sde_c_simple(sde_complex_t sde, np.ndarray tlist, np.ndarray y0, double hmax, Py_ssize_t q, void *params):
    cdef:
        wrap_complex_params p = wrap(NULL, sde, params)
    y0 = np.astype(y0, np.complex128)
    ret, wts  = integrate_sde_c_simple(wrapped_complex_sde, tlist, y0.view(np.float64), hmax, q, &p)
    return ret.view(np.complex128), wts