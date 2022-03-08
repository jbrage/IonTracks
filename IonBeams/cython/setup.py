from distutils.core import setup
from Cython.Build import cythonize
import numpy 


setup(
    ext_modules=cythonize(['initial_recombination.pyx', 'general_and_initial_recombination.pyx']),
    include_dirs=[numpy.get_include()],
    setup_requires=[
        "cython >= 0.22.1",
    ],
    # include_package_data=True,
    # zip_safe=False,
)
