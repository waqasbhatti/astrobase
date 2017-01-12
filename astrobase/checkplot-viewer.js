// checkplot-viewer.js - Waqas Bhatti (wbhatti@astro.princeton.edu) - Dec 2016
// License: MIT. See LICENSE for the full text.
//
// This contains the JS to drive the checkplot quickviewer.
//
// To run the example, run python -m SimpleHTTPServer (or python3 -m
// http.server) from the terminal in this directory, then browse to
// http://localhost:8000/checkplot-viewer.html.

// For an actual deployment, you'll want to first copy over the files:

// - checkplotlist.py
// - checkplot-viewer.html
// - checkplot-viewer.js

// to the directory you plan to serve all your checkplots from. The directory
// structure should be something like:

// checkplotlist.py
// checkplot-viewer.js
// checkplot-viewer.html
// checkplots/
//   checkplot-1.png
//   checkplot-2.png
//   ...

// Then in that directory, do:

// $ python checkplotlist.py png subdir/containing/checkplots/pngs

// This will generate a checkplot-filelist.json file that the webapp will use to
// load all the checkplot images. Then, just navigate to the
// directory/checkplot-viewer.html page using a web browser. NOTE: you'll need
// an actual web server to serve this directory; using a file:/// URL won't work
// in Chrome, for example. In a pinch, you can just use Python's built-in web
// server run from the directory, then navigate to
// http://localhost:8000/checkplot-viewer.html.

// For Python 2.7: use python -m SimpleHTTPServer
// For Python 3+: use python3 -m http.server


//////////////
// JS BELOW //
//////////////

// this is the checkplot object
var checkplot = {

    filelist: [],
    nfiles: 0,
    currfile: '',

    // this loads a checkplot from an image file into an HTML canvas object
    load_checkplot: function (filename) {

        console.log('loading ' + filename);

        var imelem = $('#checkplot');
        var plottitle = $('#checkplot-current');

        // load image from data url
        imelem.attr('src',filename);

        // build the title for this current file
        var filelink = '<a href="' + filename + '" target="_blank">' +
            filename + '</a>';
        plottitle.html(filelink);

        // highlight the file in the sidebar list
        $("a[data-fname='" + filename + "']").wrap('<strong></strong>')

        if (checkplot.currfile.length > 0) {
            // un-highlight the previous file in side bar
            $("a[data-fname='" + checkplot.currfile + "']").unwrap();
        }

        // update the current file tracker
        checkplot.currfile = filename;

    },

    // this populates the sidebar's file list
    populate_web_filelist: function () {

        var outelem = $('#pnglist');

        if (checkplot.nfiles > 0) {

            var flen = checkplot.nfiles;
            var ind = 0;
            var basefname = '';

            for (ind; ind < flen; ind++) {

                basefname = checkplot.filelist[ind].split('/');
                basefname = basefname[basefname.length - 1];

                outelem.append('<li>' +
                               '<a class="checkplot-load" ' +
                               'href="#" data-fname="' +
                               checkplot.filelist[ind] +
                               '">' + basefname + '</a></li>');

            }

        }

        else {

            outelem.append('<li>Sorry, no valid checkplots found.</li>');

        }

    },


    // this loads the file list from the checkplot-filelist.json file, then
    // updates the sidebar, and loads the first checkplot
    get_file_list:  function (url) {

        $.getJSON(url, function (data) {
            checkplot.filelist = data.checkplots;
            checkplot.nfiles = data.nfiles;
        }).done(function () {
            console.log('populating sidebar file list')
            checkplot.populate_web_filelist();
        }).done(function () {
            checkplot.load_checkplot(checkplot.filelist[0]);
        });

    },

    // this binds actions to the web-app controls
    action_setup: function () {

        // the previous checkplot link
        $('.checkplot-prev').on('click',function (evt) {

            evt.preventDefault();

            // find the current index
            var currindex = checkplot.filelist.indexOf(checkplot.currfile);
            var prevfile = checkplot.filelist[currindex-1];
            if (prevfile != undefined) {
                checkplot.load_checkplot(prevfile);
            }

        });

        // the next checkplot link
        $('.checkplot-next').on('click',function (evt) {

            evt.preventDefault();

            // find the current index
            var currindex = checkplot.filelist.indexOf(checkplot.currfile);
            var nextfile = checkplot.filelist[currindex+1];
            if (nextfile != undefined) {
                checkplot.load_checkplot(nextfile);
            }

        });

        // clicking on a checkplot file in the sidebar
        $('#pnglist').on('click', '.checkplot-load', function (evt) {

            evt.preventDefault();

            var filetoload = $(this).attr('data-fname');
            console.log('file to load: ' + filetoload);

            if (checkplot.filelist.indexOf(filetoload) != -1) {
                checkplot.load_checkplot(filetoload);
            }

        });

    }

};
