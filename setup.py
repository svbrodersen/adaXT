# distutils: define_macros=CYTHON_TRACE=1
import numpy as np
from setuptools import setup, Extension, find_packages

NAME = "adaXT"
VERSION = "0.1.0"
DESCRIPTION = "A Python package for tree-based regression and classification"
PROJECT_URLS = {
    "Documentation": "",
    "Source Code": "https://github.com/NiklasPfister/adaXT"
}
with open("README.md", 'r') as f:
    LONG_DESCRIPTION = f.read()


USE_CYTHON = True  # TODO: get commandline input, such that a user can choose whether to compile with cython always when installing, or just the already compiled c files

# Make all pyx files for the decision_tree
ext = '.pyx' if USE_CYTHON else ".c"
include_dir = np.get_include()
extensions = [Extension("adaXT.decision_tree.*",
                        ["src/adaXT/decision_tree/*" + ext],
                        include_dirs=[include_dir])]

# If we are using cython, then compile, otherwise use the c files
if USE_CYTHON:
    from Cython.Build import cythonize
    with_debug = False
    # TODO: Annotate should be false upon release, it creates the html file,
    # where you can see what is in python
    extensions = cythonize(extensions, gdb_debug=with_debug, annotate=False)

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description= LONG_DESCRIPTION,
    project_urls = PROJECT_URLS,
    packages=find_packages(where="src"),
    package_dir={"adaXT": "./src/adaXT"},
    ext_modules=extensions,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "License :: GNU GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent",
    ],
)
