from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("distribute_and_evolve_pulsed.pyx"),
)
