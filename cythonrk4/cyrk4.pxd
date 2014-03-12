cimport cython
from cpython.ref cimport PyObject
cimport numpy as np


# cdef extern from "gsl/gsl_rng.h":
#    ctypedef struct gsl_rng_type:
#        pass
#    ctypedef struct gsl_rng:
#        pass
#    gsl_rng_type *gsl_rng_mt19937
#    gsl_rng *gsl_rng_alloc(gsl_rng_type * T)
#
# cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)
#
#
# cdef extern from "gsl/gsl_randist.h":
#    double gaussian "gsl_ran_gaussian"(gsl_rng *, double)


ctypedef void (* ode_t)(double t, double y[], double ydot[], void * params)
ctypedef void (* sde_t)(double t, double y[], double w[], double ydot[], void * params)
ctypedef void (* ode_complex_t)(double t, complex y[], complex ydot[], void * params)
ctypedef void (* sde_complex_t)(double t, complex y[], double w[], complex ydot[], void * params)


cdef struct wrap_complex_params:
    ode_complex_t ode
    sde_complex_t sde
    void * params

cdef wrap_complex_params wrap(ode_complex_t ode, sde_complex_t sde, void * params)

cdef void wrapped_complex_ode(double t, double y[], double ydot[], void * params)
cdef void wrapped_complex_sde(double t, double y[], double w[], double ydot[], void * params)


cdef void integrate_ode_c(ode_t ode, Py_ssize_t nt, double tlist[], Py_ssize_t neqs, double y0[], double yts[], double hmax, void * params)
cdef void integrate_sde_c(sde_t sde, Py_ssize_t nt, double tlist[], Py_ssize_t neqs, double y0[], double yts[], Py_ssize_t nw, double wts[], double hmax, void * params)

cdef np.ndarray integrate_ode_c_simple(ode_t ode, np.ndarray tlist, np.ndarray y0, double hmax, void *params)
cdef object integrate_sde_c_simple(sde_t sde, np.ndarray tlist, np.ndarray y0, double hmax, Py_ssize_t q, void *params)


cdef void integrate_complex_ode_c(ode_complex_t ode, Py_ssize_t nt, double tlist[], Py_ssize_t neqs, complex y0[], complex yts[], double hmax, void * params)
cdef void integrate_complex_sde_c(sde_complex_t sde, Py_ssize_t nt, double tlist[], Py_ssize_t neqs, complex y0[], complex yts[], Py_ssize_t nw, double wts[], double hmax, void * params)

cdef np.ndarray integrate_complex_ode_c_simple(ode_complex_t ode, np.ndarray tlist, np.ndarray y0, double hmax, void *params)
cdef object integrate_complex_sde_c_simple(sde_complex_t sde, np.ndarray tlist, np.ndarray y0, double hmax, Py_ssize_t q, void *params)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void copy_array(double * from_p, double * to_p, Py_ssize_t Np):
    cdef Py_ssize_t kk
    for kk in range(Np):
        to_p[kk] = from_p[kk]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void init_zero(double * ptr,  Py_ssize_t Np):
    cdef Py_ssize_t kk
    for kk in range(Np):
        ptr[kk] = 0.

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void add_to(double * to_p, double * from_p, Py_ssize_t Np, double factor):
    cdef Py_ssize_t kk
    for kk in range(Np):
        to_p[kk] += from_p[kk] * factor


cdef void sample_gaussian(double * ptr,  Py_ssize_t Np, double sigma)


cdef struct py_callback:
    PyObject * pyparams
    PyObject * fn
    Py_ssize_t m
    Py_ssize_t q


cdef void callback_ode(double t, double y[], double ydot[], void * params)
cdef void callback_sde(double t, double y[], double w[], double ydot[], void * params)

