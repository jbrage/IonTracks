from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(['initial_recombination.pyx', 'general_and_initial_recombination.pyx']),
    setup_requires=[
        "cython >= 0.22.1",
    ],
    # include_package_data=True,
    # zip_safe=False,
)
