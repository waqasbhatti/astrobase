'''setup.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Nov 2016

This sets up the package.

Stolen from http://python-packaging.readthedocs.io/en/latest/everything.html and
modified by me.

'''
__version__ = '0.1.0'

import sys, os.path

from setuptools import setup

# for f2py extension building
from numpy.distutils.core import Extension, setup as npsetup

def readme():
    with open('README.md') as f:
        return f.read()

########################
## DO THE FORTRAN BIT ##
########################

# taken from github:dfm/python-bls.git/setup.py

# Define the Fortran extension.
bls = Extension("bls._bls", ["bls/bls.pyf", "bls/eebls.f"])

npsetup(
    name='bls',
    version=__version__,
    description=('Python f2py extension wrapping '
                 'eebls.f by Kovacs et al. 2002.'),
    long_description=readme(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    keywords='astronomy',
    url='https://github.com/waqasbhatti/astrobase',
    author='Waqas Bhatti',
    author_email='waqas.afzal.bhatti@gmail.com',
    license='MIT',
    packages=["bls"],
    ext_modules=[bls,],
    install_requires=[
        'numpy',
        'scipy',
        'astropy',
        'matplotlib',
        'Pillow',
        'jplephem',
        'simplejson',
        'astroquery',
    ],
)


#############################
## RUN SETUP FOR ASTROBASE ##
#############################

setup(
    name='astrobase',
    version=__version__,
    description=('A bunch of Python modules and scripts '
                 'useful for variable star work in astronomy.'),
    long_description=readme(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    keywords='astronomy',
    url='https://github.com/waqasbhatti/astrobase',
    author='Waqas Bhatti',
    author_email='waqas.afzal.bhatti@gmail.com',
    license='MIT',
    packages=['astrobase'],
    install_requires=[
        'bls',
        'numpy',
        'scipy',
        'astropy',
        'matplotlib',
        'Pillow',
        'jplephem',
        'simplejson',
        'astroquery',
    ],
    extras_require={
        'LCDB':['psycopg2'],
    },
    # entry_points={
    #     'console_scripts':[
    #         'fitshdr.py=astrobase.imageutils:fitshdr',
    #     ],
    #},
    include_package_data=True,
    zip_safe=False,
)
