from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(['recombination_cythonized.pyx']),
    setup_requires=[
        "cython >= 0.22.1",
    ],
    # include_package_data=True,
    # zip_safe=False,
)
