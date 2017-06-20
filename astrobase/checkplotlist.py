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

$ checkplotlist png subdir/containing/the/checkplots 'optional-glob*.png'

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

'''


PROGDESC = '''\
This makes a checkplot file list for use with the checkplot-viewer.html (for
checkplot PNGs) or the checkplotserver.py (for checkplot pickles) webapps.
'''

PROGEPILOG= '''\
If you have checkplots that don't have 'checkplot' somewhere in their file name,
use the optional checkplot file glob argument to checkplotlist to provide
this:

--search '<filename glob for prefix>'

Make sure to use the quotes around this argument, otherwise the shell will
expand it.

Example: search for checkplots with awesome-object in their filename:

$ checkplotlist png my-project/awesome-objects --search '*awesome-object*'

For checkplot pickles only: If you want to sort the checkplot pickle files in
the output list in some special way other than the usual filename sort order,
this requires an argument on the commandline of the form:

--sortby '<sortkey>-<asc|desc>'.

Here, sortkey is some key in the checkplot pickle. This can be a simple key:
e.g. objectid or it can be a composite key: e.g. varinfo.features.stetsonj.
sortorder is either 'asc' or desc' for ascending/descending sort. The sortkey
must exist in all checkplot pickles.

Example: sort checkplots by their 2MASS J magnitudes in ascending order:

$ checkplotlist pkl my-project/awesome-objects --sortby 'objectinfo.jmag-asc'

Example: sort checkplots by the best peak in their PDM periodograms:

$ checkplotlist pkl my-project/awesome-objects --sortby 'pdm.nbestlspvals.0-asc'
'''

import os
import os.path
import sys
import glob
import json
import argparse

# to turn a list of keys into a dict address
# from https://stackoverflow.com/a/14692747
# used to walk a checkplotdict for a specific key in the structure
from functools import reduce
from operator import getitem

from astrobase import checkplot
import numpy as np
import multiprocessing as mp

def dict_get(datadict, keylist):
    return reduce(getitem, keylist, datadict)

def sortkey_worker(task):
    cpf, key = task
    cpd = checkplot._read_checkplot_picklefile(cpf)
    return dict_get(cpd, key)


def main():

    ####################
    ## PARSE THE ARGS ##
    ####################

    aparser = argparse.ArgumentParser(
        epilog=PROGEPILOG,
        description=PROGDESC,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    aparser.add_argument(
        'cptype',
        action='store',
        choices=['pkl','png'],
        type=str,
        help=("type of checkplot to search for: pkl -> checkplot pickles, "
              "png -> checkplot PNGs")
    )
    aparser.add_argument(
        'cpdir',
        action='store',
        type=str,
        help=("directory containing the checkplots to process")
    )
    aparser.add_argument(
        '--search',
        action='store',
        default='*checkplot*',
        type=str,
        help=("file glob prefix to use when searching for checkplots "
              "(the extension is added automatically - .png or .pkl)")
    )
    aparser.add_argument(
        '--sortby',
        action='store',
        type=str,
        help=("the sort key and order to use when sorting")
    )
    aparser.add_argument(
        '--splitout',
        action='store',
        type=int,
        default=5000,
        help=("if there are more than SPLITOUT objects in "
              "the target directory (default: %(default)s), "
              "checkplotlist will split the output JSON into multiple files. "
              "this helps keep the webapps responsive.")
    )

    args = aparser.parse_args()

    checkplotbasedir = args.cpdir
    fileglob = args.search
    splitout = args.splitout

    if args.sortby:
        sortkey, sortorder = args.sortby.split('-')
    else:
        sortkey, sortorder = None, None

    if args.cptype == 'pkl':
        checkplotext = 'pkl'
    elif args.cptype == 'png':
        checkplotext = 'png'
    else:
        print("unknown format for checkplots: %s! can't continue!"
              % args.cptype)
        sys.exit(1)


    #######################
    ## NOW START WORKING ##
    #######################

    currdir = os.getcwd()

    checkplotglob = os.path.join(checkplotbasedir,
                                 '%s.%s' % (fileglob, checkplotext))

    print('searching for checkplots: %s' % checkplotglob)

    searchresults = glob.glob(checkplotglob)

    if searchresults:

        print('found %s checkplot files in %s, '
              'making checkplot-filelist.json...' %
              (len(searchresults), checkplotbasedir))

        # see if we should sort the searchresults in some special order
        # this requires an arg on the commandline of the form:
        # '<sortkey>-<asc|desc>'
        # where sortkey is some key in the checkplot pickle:
        #   this can be a simple key: e.g. objectid
        #   or it can be a composite key: e.g. varinfo.varfeatures.stetsonj
        # and sortorder is either 'asc' or desc' for ascending/descending sort
        if sortkey and sortorder:

            print('sorting checkplot pickles by %s in order: %s...' %
                  (sortkey, sortorder))

            # dereference the sort key
            sortkeys = sortkey.split('.')

            # if there are any integers in the sortkeys strings, interpret these
            # to mean actual integer indexes of lists or integer keys for dicts
            # this allows us to move into arrays easily by indexing them
            sortkeys = [(int(x) if x.isdecimal() else x) for x in sortkeys]

            pool = mp.Pool()
            tasks = [(x, sortkeys) for x in searchresults]
            sorttargets = pool.map(sortkey_worker, tasks)

            pool.close()
            pool.join()

            sorttargets = np.array(sorttargets)
            sortind = np.argsort(sorttargets)
            if sortorder == 'desc':
                sortind = sortind[::-1]
            searchresults = np.array(searchresults)
            searchresults = searchresults[sortind].tolist()

        # if there's no special sort order defined, use the usual sort order
        else:
            print('no special sort key and order specified, '
                  'sorting checkplot pickles '
                  'using usual alphanumeric sort...')
            searchresults = sorted(searchresults)
            sortkey = 'filename'
            sortorder = 'asc'

        nchunks = int(len(searchresults)/splitout) + 1

        searchchunks = [searchresults[x*splitout:x*splitout+splitout] for x
                        in range(nchunks)]

        if nchunks > 1:
            print('WRN! more than %s checkplots in this directory, '
                  'splitting into %s chunks' % (len(searchresults), nchunks))


        for chunkind, chunk in enumerate(searchchunks):

            # figure out if we need to split the JSON file
            outjson = os.path.abspath(
                os.path.join(
                    currdir,
                    'checkplot-filelist%s.json' %
                    ('-%02i' % chunkind if len(searchchunks) > 1 else '')
                )
            )

            # ask if the checkplot list JSON should be updated
            if os.path.exists(outjson):

                answer = input('There is an existing '
                               'checkplot list file in this '
                               'directory:\n    %s\nDo you want to '
                               'overwrite it completely? (default: no) [y/n] ' %
                               outjson)

                # if it's OK to overwrite, then do so
                if answer and answer == 'y':

                    with open(outjson,'w') as outfd:
                        print('WRN! completely overwriting '
                              'existing checkplot list %s' % outjson)
                        outdict = {'checkplots':chunk,
                                   'nfiles':len(chunk),
                                   'sortkey':sortkey,
                                   'sortorder':sortorder}
                        json.dump(outdict,outfd)

                # if it's not OK to overwrite, then
                else:

                    # read in the outjson, and add stuff to it for objects that
                    # don't have an entry
                    print('only updating existing checkplot list '
                          'file with any new checkplot pickles')

                    with open(outjson,'r') as infd:
                        indict = json.load(infd)

                    # update the checkplot list, sortorder, and sortkey only
                    indict['checkplots'] = chunk
                    indict['nfiles'] = len(chunk)
                    indict['sortkey'] = sortkey
                    indict['sortorder'] = sortorder

                    # write the updated to back to the file
                    with open(outjson,'w') as outfd:
                        json.dump(indict, outfd)

            # if this is a new output file
            else:

                with open(outjson,'w') as outfd:
                    outdict = {'checkplots':chunk,
                               'nfiles':len(chunk),
                               'sortkey':sortkey,
                               'sortorder':sortorder}
                    json.dump(outdict,outfd)

            if os.path.exists(outjson):
                print('checkplot file list written to %s' % outjson)
            else:
                print('ERR! writing the checkplot file list failed!')

    else:

        print('ERR! no checkplots found in %s' % checkplotbasedir)


if __name__ == '__main__':
    main()
