'''setup.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Nov 2016

This sets up the package.

Stolen from http://python-packaging.readthedocs.io/en/latest/everything.html and
modified by me.

'''
__version__ = '0.1.7'

import sys, os.path

from setuptools import setup

def readme():
    with open('README.rst') as f:
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
    'pyeebls'
]

EXTRAS_REQUIRE = {
    'LCDB':['psycopg2'],
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
