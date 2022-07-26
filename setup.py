from distutils.core import setup
from Cython.Build import cythonize
import numpy 


setup(
    ext_modules=cythonize([
        'hadrons/cython_files/initial_recombination.pyx', 
        'hadrons/cython_files/continuous_beam.pyx',
    ]),
    include_dirs=[numpy.get_include()],
    # setup_requires=[
    #     "cython",
    # ],
    # include_package_data=True,
    # zip_safe=False,
)

