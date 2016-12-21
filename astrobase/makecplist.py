#!/usr/bin/env python
'''makecplist.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Dec 2016
License: MIT. See LICENSE for full text.

This makes a checkplot image file list for use with the checkplot-viewer webapp.

Copy this file along with checkplot-viewer.html and checkplot-viewer.js to the
base directory from where you intend to serve your checkplot images from. Then
invoke it from that directory like so:

$ python makecplist.py subdir/containing/the/checkplot-pngs

This will generate a checkplot-filelist.json file containing the file paths to
the checkplot PNG files. You can then run a temporary Python web server from
this base directory to browse through all the checkplots:

$ python -m SimpleHTTPServer # Python 2
$ python3 -m http.server     # Python 3

then browse to http://localhost:8000.

If this directory is already in a path served by a web server, then you can just
browse there normally. Note that a file:/// URL provided to the browser won't
necessarily work in some browsers (especially Google Chrome) because of security
precautions.

'''

import os
import os.path
import sys
import glob
try:
    import simplejson as json
except:
    import json

if __name__ == '__main__':

    if len(sys.argv) != 2:
        docstring = __doc__
        if docstring:
            print(docstring)
        else:
            print('Usage: %s <subdir/containing/the/checkplot-pngs>' %
                  sys.argv[0])
        sys.exit(2)

    checkplotbasedir = sys.argv[1]
    currdir = os.getcwd()
    searchresults = glob.glob(os.path.join(checkplotbasedir, '*checkplot*.png'))

    if searchresults:

        print('found %s checkplot PNGs in %s, '
              'making checkplot-filelist.json...' %
              (len(searchresults),checkplotbasedir))

        outjson = os.path.abspath(
            os.path.join(currdir,'checkplot-filelist.json')
        )

        with open(outjson,'wb') as outfd:

            outdict = {'checkplots':sorted(searchresults),
                       'nfiles':len(searchresults)}
            json.dump(outdict,outfd)

        if os.path.exists(outjson):
            print('checkplot file list written to %s' % outjson)
        else:
            print('ERR! writing the checkplot file list failed!')

    else:

        print('ERR! no checkplots found in %s' % checkplotbasedir)
