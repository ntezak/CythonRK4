from distutils.core import setup, Extension
from Cython.Build import cythonize
import sys
import subprocess
import numpy as np
import cython_gsl

args = sys.argv[1:]

# Make a `cleanall` rule to get rid of intermediate and library files
if "cleanall" in args:
    print "Deleting cython files..."
    # Just in case the build directory was created by accident,
    # note that shell=True should be OK here because the command is constant.
    subprocess.Popen("rm -rf cythonrk4/*.c", shell=True, executable="/bin/bash")
    subprocess.Popen("rm -rf cythonrk4/*.so", shell=True, executable="/bin/bash")

    # Now do a normal clean
    sys.argv[1] = "clean"


extensions = [Extension("cythonrk4/*", ["cythonrk4/*.pyx"],
                        include_dirs=[np.get_include(),
                                      "/usr/local/include"],
                        libraries=["gsl", "gslcblas", "m"]+cython_gsl.get_libraries())]

setup(
    # ext_modules = cythonize("cythonrk4/*.pyx", include_dirs=[np.get_include()])
    ext_modules = cythonize(extensions)
)