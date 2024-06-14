import os

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None

""" this code is inspired by  https://github.com/FedericoStra/cython-package-example """


class build_ext(_build_ext):
    """
    From https://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py/21621689#21621689
    """

    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
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


CYTHONIZE = bool(int(os.getenv("CYTHONIZE", 0))) and cythonize is not None

extensions = [
    Extension(
        "hadrons.cython_files.continuous_beam",
        ["hadrons/cython_files/continuous_beam.pyx"],
    ),
    Extension(
        "hadrons.cython_files.initial_recombination",
        ["hadrons/cython_files/initial_recombination.pyx"],
    ),
    Extension(
        "electrons.cython.continuous_e_beam", ["electrons/cython/continuous_e_beam.pyx"]
    ),
    Extension("electrons.cython.pulsed_e_beam", ["electrons/cython/pulsed_e_beam.pyx"]),
]

if CYTHONIZE:
    # description of compiler directives: https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#compiler-directives
    # language_level 3 enables and enforces Python 3 semantics
    # embedsignature  Cython will embed a textual copy of the call signature in the docstring of all Python visible functions and classes
    compiler_directives = {"language_level": 3, "embedsignature": True}
    extensions = cythonize(
        module_list=extensions, compiler_directives=compiler_directives
    )
else:
    extensions = no_cythonize(extensions)

install_requires = [
    "Cython>=3.0.10",
    "numpy>=1.24.4",
    "scipy>=1.13.1",
    "matplotlib>=3.9.0",
    "mpmath>=1.3.0",
    "pandas>=2.2.2",
    "seaborn>=0.13.2",
    "numba>=0.59.1",
    "isort>=5.13.2",
    "flake8>=7.0.0",
    "flake8-isort>=6.1.1",
]

setup(
    cmdclass={"build_ext": build_ext},
    ext_modules=extensions,
    install_requires=install_requires,
    setup_requires=["numpy", "cython"],
)
