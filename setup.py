# -*- coding: utf-8 -*-

'''setup.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Nov 2016

This sets up the package.

Stolen from http://python-packaging.readthedocs.io/en/latest/everything.html and
modified by me.

'''
__version__ = '0.3.19'

import sys

from setuptools import setup

# pytesting stuff and imports copied wholesale from:
# https://docs.pytest.org/en/latest/goodpractices.html#test-discovery
from setuptools.command.test import test as TestCommand

class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        import shlex
        import pytest

        if not self.pytest_args:
            targs = []
        else:
            targs = shlex.split(self.pytest_args)

        errno = pytest.main(targs)
        sys.exit(errno)


def readme():
    with open('README.rst') as f:
        return f.read()

INSTALL_REQUIRES = [
    'numpy>=1.4.0',
    'scipy',
    'astropy>=1.3',
    'matplotlib',
    'Pillow',
    'jplephem',
    'requests',
    'tornado',
    'pyeebls',
    'tqdm',
    'scikit-learn',
    'futures;python_version<"3.2"',
]

EXTRAS_REQUIRE = {
    'all':[
        # for lcdb
        'psycopg2-binary',
        # for lcfit.mandelagol_fit_magseries
        'emcee==3.0rc1',
        'h5py',
        'batman-package',
        'corner',
    ]
}



#############################
## RUN SETUP FOR ASTROBASE ##
#############################

setup(
    name='astrobase',
    version=__version__,
    description=('Python modules and scripts '
                 'useful for variable star work in astronomy.'),
    long_description=readme(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    keywords='astronomy',
    url='https://github.com/waqasbhatti/astrobase',
    author='Waqas Bhatti',
    author_email='waqas.afzal.bhatti@gmail.com',
    license='MIT',
    packages=['astrobase',
              'astrobase.periodbase',
              'astrobase.varbase',
              'astrobase.varclass',
              'astrobase.lcmodels',
              'astrobase.fakelcs',
              'astrobase.services',
              'astrobase.hatsurveys',
              'astrobase.cpserver'],
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    tests_require=['pytest',],
    cmdclass={'test':PyTest},
    entry_points={
        'console_scripts':[
            'hatlc=astrobase.hatsurveys.hatlc:main',
            'checkplotserver=astrobase.cpserver.checkplotserver:main',
            'checkplotlist=astrobase.cpserver.checkplotlist:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
