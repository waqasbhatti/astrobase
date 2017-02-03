'''setup.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Nov 2016

This sets up the package.

Stolen from http://python-packaging.readthedocs.io/en/latest/everything.html and
modified by me.

'''
__version__ = '0.1.0'

import sys, os.path

from setuptools import setup

# for f2py extension building
try:
    from numpy.distutils.core import Extension, setup as npsetup
except:
    print('\nyou need to have numpy installed before running setup.py,\n'
          'because we need its Extension functionality to make a\n'
          'compiled Fortran extension for BLS!\n')
    raise


def readme():
    with open('README.md') as f:
        return f.read()

INSTALL_REQUIRES = [
    'numpy',
    'scipy',
    'astropy',
    'matplotlib',
    'Pillow',
    'jplephem',
    'astroquery',
    'tornado',
]

EXTRAS_REQUIRE = {
    'LCDB':['psycopg2'],
}

# add extra stuff needed if we're running Python 2.7
if sys.version_info.major < 3:
    INSTALL_REQUIRES.append([
        'futures'
    ])

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
    install_requires=INSTALL_REQUIRES,
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
    packages=['astrobase','astrobase.periodbase','astrobase.varbase'],
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        'console_scripts':[
            'checkplotserver=astrobase.checkplotserver:main',
            'checkplotlist=astrobase.checkplotlist:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
