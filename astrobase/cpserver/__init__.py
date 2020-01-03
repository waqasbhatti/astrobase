#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# checkplotserver - Waqas Bhatti (wbhatti@astro.princeton.edu) - Dec 2016
# License: MIT. See LICENSE for full text.

'''This package contains the implementation of the `checkplotserver` webapp to
review large numbers of checkplot pickle files generated as part of a variable
star classification pipeline. Also provided is a lightweight
`checkplot-viewer.html` webapp to quickly glance through large numbers of
checkplot PNGs.

If you made checkplot pickles (`checkplot-*.pkl`)
-------------------------------------------------

Invoke this command from that directory like so::

    $ checkplotlist pkl subdir/containing/the/checkplots

Then, from that directory, invoke the checkplotserver webapp (make sure the
astrobase virtualenv is active, so the command below is in your path)::

    $ checkplotserver [list of options, use --help to see these]

The webapp will start up a Tornado web server running on your computer and
listening on a local address (default: http://localhost:5225). This webapp will
read the checkplot-filelist.json file to find the checkplots.

Browse to http://localhost:5225 (or whatever port you set in checkplotserver
options) to look through or update all your checkplots. Any changes will be
written back to the checkplot .pkl files, making this method of browsing more
suited to more serious variability searches on large numbers of checkplots.

If you made checkplots PNGs (`checkplot-*.png`)
-----------------------------------------------

Copy `checkplot-viewer.html` and `checkplot-viewer.js` to the base directory
from where you intend to serve your checkplot images from. Then invoke this
command from that directory::

    $ checkplotlist png subdir/containing/the/checkplots 'optional-glob*.png'

This will generate a `checkplot-filelist.json` file containing the file paths to
the checkplots. You can then run a temporary Python web server from this base
directory to browse through all the checkplots::

    $ python -m SimpleHTTPServer # Python 2
    $ python3 -m http.server     # Python 3

then browse to http://localhost:8000/checkplot-viewer.html.

If this directory is already in a path served by a web server, then you can just
browse to the `checkplot-viewer.html` file normally. Note that a `file:///` URL
provided to the browser won't necessarily work in some browsers (especially
Google Chrome) because of security precautions.

'''
