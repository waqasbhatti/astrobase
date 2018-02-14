This contains modules used for testing astrobase using the pytest package.

# Running tests

To run all the tests from the base directory of the git repository, make sure a
virtualenv is active with all the needed requirements, and then run setup.py:

```bash
$ virtualenv astrobase-testing  # or use: python3 -m venv astrobase-testing
$ source astrobase-testing/bin/activate
$ git clone https://github.com/waqasbhatti/astrobase.git
$ cd astrobase

# run these to hopefully get wheels for faster install (can skip this step)
$ pip install numpy  # do this first for f2py stuff
$ pip install -r requirements.txt

# finally run the tests (this will get and compile requirements automatically)
$ python setup.py test
```

# Test module list

## test_periodbase.py

This tests the following:

- downloads a light curve from the github repository notebooks/nb-data dir
- reads the light curve using astrobase.hatlc
- runs the GLS, WIN, PDM, AoV, BLS, AoVMH, and ACF period finders on the LC

## test_lcfit.py

This tests the following:

- downloads a light curve from the github repository notebooks/nb-data dir
- reads the light curve using astrobase.hatlc
- fits the light curve using Fourier, SavGol, Legendre, transit model, and
  eclipsing binary models

## test_checkplot.py

This tests the following:

- downloads a light curve from the github repository notebooks/nb-data dir
- reads the light curve using astrobase.hatlc
- creates a checkplot PNG, twolsp PNG, and pickle using these results
