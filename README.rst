CythonRK4 README
================

This project aims to provide a fast solver implemented in cython for ode's and sde's.
It provides only cython bindings to be used by other cython code.

There is also a facility to call python callback functions as the ODE/SDE (currently only for real valued functions),
but it is highly recommended to use the cython bindings.

Check out the `example ipython notebook`__ to see how to use this package.

__ http://nbviewer.ipython.org/github/ntezak/CythonRK4/blob/master/examples/Examples.ipynb
