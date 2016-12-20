// checkplot-viewer.js - Waqas Bhatti (wbhatti@astro.princeton.edu) - Dec 2016
// License: MIT. See LICENSE for the full text.
//
// This contains the JS to drive the checkplot quickviewer.

var checkplot = {

    filelist: [],
    nfiles: 0,
    currfile: '',

    load_checkplot: function (filename) {

        console.log('loading ' + filename);

        var canvas = document.getElementById('checkplot');
        var context = canvas.getContext('2d');
        var plottitle = $('#checkplot-current');

        // load image from data url
        var imageobject = new Image();
        imageobject.onload = function() {
          context.drawImage(this, 0, 0);
        };

        imageobject.src = filename;
        plottitle.text(filename);
        checkplot.currfile = filename;
    },


    populate_web_filelist: function () {

        var outelem = $('#pnglist');

        if (checkplot.nfiles > 0) {

            var flen = checkplot.nfiles;
            var ind = 0;

            for (ind; ind < flen; ind++) {

                outelem.append('<li>' +
                               '<a class="checkplot-load" ' +
                               'href="#" data-fname="' + checkplot.filelist[ind] +
                               '">' + checkplot.filelist[ind] + '</a></li>');

            }

        }

        else {

            outelem.append('<li>Sorry, no valid checkplots found.</li>');

        }

    },


    get_file_list:  function (url) {

        $.getJSON(url, function (data) {
            checkplot.filelist = data.checkplots;
            checkplot.nfiles = data.nfiles;
        }).done(function () {
            console.log('populating sidebar file list')
            checkplot.populate_web_filelist();
        }).done(function () {
            console.log('making checkplot with ' + checkplot.filelist);
            checkplot.load_checkplot(checkplot.filelist[0]);
        });

    },


    action_setup: function () {

        $('.checkplot-previous').on('click',function (evt) {

            evt.preventDefault();

            // find the current index
            var currindex = checkplot.filelist.indexOf(checkplot.currfile);
            var prevfile = checkplot.filelist[currindex-1];
            if (prevfile != undefined) {
                checkplot.load_checkplot(prevfile);
            }

        });

        $('.checkplot-next').on('click',function (evt) {

            evt.preventDefault();

            // find the current index
            var currindex = checkplot.filelist.indexOf(checkplot.currfile);
            var nextfile = checkplot.filelist[currindex+1];
            if (nextfile != undefined) {
                checkplot.load_checkplot(nextfile);
            }

        });

        $('.checkplot-load').on('click', function (evt) {

            evt.preventDefault();

            var filetoload = $(this).attr('data-fname').text();
            console.log('file to load: ' + filetoload);
            if (checkplot.filelist.indexOf(filetoload) != -1) {
                checkplot.load_checkplot(filetoload);
            }

        });

    }

};
