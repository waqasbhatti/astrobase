# -*- coding: utf-8 -*-

'''setup.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Nov 2016

This sets up the package.

Stolen from http://python-packaging.readthedocs.io/en/latest/everything.html and
modified by me.

'''
__version__ = '0.2.8'

import sys, os.path

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
        #import here, cause outside the eggs aren't loaded
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
]

EXTRAS_REQUIRE = {
    'all':['psycopg2'],
}

# add extra stuff needed if we're running Python 2.7
# FIXME: need to think about fixing this because Py3 will completely
# ignore this and we usually run python setup.py dist from Py3
# for now, we'll get rid of the wheel format and see if that fixes this
if sys.version_info.major < 3:
    INSTALL_REQUIRES.append('futures')

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
              'astrobase.lcmodels'],
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    tests_require=['pytest',],
    cmdclass={'test':PyTest},
    entry_points={
        'console_scripts':[
            'checkplotserver=astrobase.checkplotserver:main',
            'checkplotlist=astrobase.checkplotlist:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
