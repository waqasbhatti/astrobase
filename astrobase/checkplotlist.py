#!/usr/bin/env python
'''checkplotlist.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Dec 2016
License: MIT. See LICENSE for full text.

DESCRIPTION
===========

This makes a checkplot file list for use with the checkplot-viewer.html or the
checkplotserver.py webapps. Checkplots are quick-views of object info, finder
charts, light curves, phased light curves, and periodograms used to examine
their stellar variability. These are produced by several functions in the
astrobase.checkplot module:

checkplot.checkplot_png: makes a checkplot PNG for a single period-finding
                         method

checkplot.twolsp_checkplot_png: does the same for two independent period-finding
                                methods

checkplot.checkplot_pickle: makes a checkplot .pkl.gz for any number of
                            independent period-finding methods


USAGE
=====

If you made checkplots in the PNG format (checkplot-*.png)
----------------------------------------------------------

Copy checkplot-viewer.html and checkplot-viewer.js to the
base directory from where you intend to serve your checkplot images from. Then
invoke this command from that directory:

$ checkplotlist png subdir/containing/the/checkplots

This will generate a checkplot-filelist.json file containing the file paths to
the checkplots.

You can then run a temporary Python web server from this base directory to
browse through all the checkplots:

$ python -m SimpleHTTPServer # Python 2
$ python3 -m http.server     # Python 3

then browse to http://localhost:8000/checkplot-viewer.html.

If this directory is already in a path served by a web server, then you can just
browse to the checkplot-viewer.html file normally. Note that a file:/// URL
provided to the browser won't necessarily work in some browsers (especially
Google Chrome) because of security precautions.

If you made checkplots in the pickle format (checkplot-*.pkl)
-------------------------------------------------------------

Invoke this command from that directory like so:

$ checkplotlist pkl subdir/containing/the/checkplots

Then, from that directory, invoke the checkplotserver webapp (make sure the
astrobase virtualenv is active, so the command below is in your path):

$ checkplotserver [list of options, use --help to see these]

The webapp will start up a Tornado web server running on your computer and
listening on a local address (default: http://localhost:5225). This webapp will
read the checkplot-filelist.json file to find the checkplots.

Browse to http://localhost:5225 (or whatever port you set in checkplotserver
options) to look through or update all your checkplots. Any changes will be
written back to the checkplot .pkl files, making this method of browsing more
suited to more serious variability searches on large numbers of checkplots.

TL;DR
=====

This makes a checkplot file list for use with the checkplot-viewer.html (for
checkplot PNGs) or the checkplotserver.py (for checkplot pickles) webapps.

checkplotlist <pkl|png> <subdir/containing/checkplots/> '[checkplot file glob]'

If you have checkplots that don't have 'checkplot' somewhere in their file name,
use the optional checkplot file glob argument to checkplotlist to provide
this. Make sure to use the quotes around this argument, otherwise the shell will
expand it, e.g.:

$ checkplot png my-project/awesome-objects '*awesome-objects*'

'''

import os
import os.path
import sys
import glob
import json


def main(args=None):

    if not args:
        args = sys.argv

    if len(args) < 3:
        docstring = __doc__
        if docstring:
            print(docstring)
        else:
            print("Usage: %s <pkl|png> <subdir/containing/the/checkplots/> "
                  "'[file glob to use]'" % args[0])
        sys.exit(2)

    checkplotbasedir = args[2]

    if len(args) == 4:
        fileglob = args[3]
    else:
        fileglob = '*checkplot*'

    if args[1] == 'pkl':
        checkplotext = 'pkl'
    elif args[1] == 'png':
        checkplotext = 'png'
    else:
        print("unknown format for checkplots: %s! can't continue!"
              % args[1])
        sys.exit(1)


    currdir = os.getcwd()

    checkplotglob = os.path.join(checkplotbasedir,
                                 '%s.%s' % (fileglob, checkplotext))

    print('searching for checkplots: %s' % checkplotglob)

    searchresults = glob.glob(checkplotglob)

    if searchresults:

        print('found %s checkplot files in %s, '
              'making checkplot-filelist.json...' %
              (len(searchresults), checkplotbasedir))

        outjson = os.path.abspath(
            os.path.join(currdir,'checkplot-filelist.json')
        )

        # ask if the checkplot list JSON should be updated
        if os.path.exists(outjson):

            answer = input('There is an existing '
                           'checkplot list file in this '
                           'directory:\n    %s\nDo you want to '
                           'overwrite it? (default: no) [y/n] ' % outjson)

            # if it's OK to overwrite, then do so
            if answer and answer == 'y':

                with open(outjson,'w') as outfd:
                    print('overwriting existing checkplot list')
                    outdict = {'checkplots':sorted(searchresults),
                               'nfiles':len(searchresults)}
                    json.dump(outdict,outfd)

            # if it's not OK to overwrite, then
            else:

                # read in the outjson, and add stuff to it for objects that
                # don't have an entry
                print('updating existing checkplot list file')

                with open(outjson,'r') as infd:
                    indict = json.load(infd)

                # update the checkplot list only
                indict['checkplots'] = sorted(searchresults)
                indict['nfiles'] = len(searchresults)
                # write the updated to back to the file
                with open(outjson,'w') as outfd:
                    json.dump(indict, outfd)

        # if this is a new output file
        else:

            with open(outjson,'w') as outfd:
                outdict = {'checkplots':sorted(searchresults),
                           'nfiles':len(searchresults)}
                json.dump(outdict,outfd)

        if os.path.exists(outjson):
            print('checkplot file list written to %s' % outjson)
        else:
            print('ERR! writing the checkplot file list failed!')

    else:

        print('ERR! no checkplots found in %s' % checkplotbasedir)


if __name__ == '__main__':

    args = sys.argv
    main(args=args)
