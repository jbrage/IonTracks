import os
from setuptools import setup, find_packages, Extension

from setuptools.command.build_ext import build_ext as _build_ext



try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None

class build_ext(_build_ext):
    """
    From https://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py/21621689#21621689
    """
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        #__builtins__.__NUMPY_SETUP__ = False
        import numpy
        print("Building package with numpy version {}".format(numpy.__version__))
        self.include_dirs.append(numpy.get_include())


# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#distributing-cython-modules
def no_cythonize(extensions, **_ignore):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in (".pyx", ".py"):
                if extension.language == "c++":
                    ext = ".cpp"
                else:
                    ext = ".c"
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions

extensions = [
    Extension("hadrons.cython_files.continuous_beam", ["hadrons/cython_files/continuous_beam.pyx"]),
    Extension("hadrons.cython_files.initial_recombination", ["hadrons/cython_files/initial_recombination.pyx"]),
]

CYTHONIZE = bool(int(os.getenv("CYTHONIZE", 0))) and cythonize is not None


if CYTHONIZE:
    compiler_directives = {"language_level": 3, "embedsignature": True}
    extensions = cythonize(extensions, compiler_directives=compiler_directives)
else:
    extensions = no_cythonize(extensions)

with open("requirements.txt") as fp:
    install_requires = fp.read().strip().split("\n")

# nice example here https://github.com/FedericoStra/cython-package-example
setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=extensions,
    install_requires=install_requires,
    setup_requires=[
         "numpy", "cython"
     ]
)

# setup(
#     ext_modules=cythonize([
#         'hadrons/cython_files/initial_recombination.pyx', 
#         'hadrons/cython_files/continuous_beam.pyx',
#     ]),
#     include_dirs=[numpy.get_include()],
#     setup_requires=[
#         "cython",
#     ],
#     # include_package_data=True,
#     # zip_safe=False,
# )

