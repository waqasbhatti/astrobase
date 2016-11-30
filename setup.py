'''setup.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Nov 2016

This sets up the package.

Stolen from http://python-packaging.readthedocs.io/en/latest/everything.html and
modified by me.

'''
__VERSION__ = 0.1.0

from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='astrobase',
    version=__VERSION__,
    description=('A bunch of Python modules and scripts '
                 'useful for variable star work in astronomy.'),
    long_description=readme(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
    ],
    keywords='astronomy',
    url='https://github.com/waqasbhatti/astrobase',
    author='Waqas Bhatti',
    author_email='waqas.afzal.bhatti@gmail.com',
    license='MIT',
    packages=['astrobase','classy'],
    install_requires=[
        'numpy',
        'scipy',
        'astropy',
        'matplotlib',
        'Pillow',
        'tornado',
        'jplephem',
        'simplejson',
        'psycopg2',
    ],
    entry_points={
        'console_scripts':[
            'classy-server=classy.classyserver:main',
        ],
    },
    include_package_data=True,
    zip_safe=False
)
