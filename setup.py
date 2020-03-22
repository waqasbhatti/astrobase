# -*- coding: utf-8 -*-

'''setup.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Nov 2016

This sets up the package.

Stolen from http://python-packaging.readthedocs.io/en/latest/everything.html and
modified by me.

'''
__version__ = '0.5.0'

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
    with open('README.md') as f:
        return f.read()


INSTALL_REQUIRES = [
    'numpy>=1.4.0',
    'scipy>=0.17.0',
    'astropy>=1.3',
    'matplotlib>=2.0',
    'Pillow',
    'jplephem',
    'requests>=2.20',
    'tornado',
    'pyeebls>=0.1.6',
    'tqdm',
    'scikit-learn>=0.19',
]

EXTRAS_REQUIRE = {
    'all':[
        'psycopg2-binary',
        'emcee==3.0rc1',
        'h5py',
        'batman-package',
        'corner',
        'transitleastsquares>=1.0',
        'paramiko',
        'boto3',
        'awscli',
        'google-api-python-client',
        'google-cloud-storage',
        'google-cloud-pubsub',
    ],
    # for lcfit.mandelagol_fit_magseries
    'mandelagol':[
        'emcee==3.0rc1',
        'h5py',
        'batman-package',
        'corner',
    ],
    # for periodbase.tls
    'tls':[
        'emcee==3.0rc1',
        'h5py',
        'batman-package',
        'transitleastsquares>=1.0',
        'corner',
    ],
    # for lcproc_aws and awsutils
    'aws':[
        'paramiko',
        'boto3',
        'awscli',
    ],
    # for lcproc_gcp and gcputils
    'gcp':[
        'paramiko',
        'google-api-python-client',
        'google-cloud-storage',
        'google-cloud-pubsub',
    ]
}


#############################
## RUN SETUP FOR ASTROBASE ##
#############################

# run setup.
setup(
    name='astrobase',
    version=__version__,
    description=('Python modules and scripts '
                 'useful for variable star work in astronomy.'),
    long_description=readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    keywords='astronomy',
    url='https://github.com/waqasbhatti/astrobase',
    author='Waqas Bhatti',
    author_email='waqas.afzal.bhatti@gmail.com',
    license='MIT',
    packages=[
        'astrobase',
        'astrobase.checkplot',
        'astrobase.cpserver',
        'astrobase.fakelcs',
        'astrobase.hatsurveys',
        'astrobase.lcfit',
        'astrobase.lcmodels',
        'astrobase.lcproc',
        'astrobase.periodbase',
        'astrobase.services',
        'astrobase.varbase',
        'astrobase.varclass',
    ],
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    tests_require=['pytest==3.8.2',],
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
    python_requires=">=3.5"
)
