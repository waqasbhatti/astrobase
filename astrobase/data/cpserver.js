// cpserver.js - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jan 2017
// License: MIT. See LICENSE for the full text.
//
// This contains the JS to drive the checkplotserver's interface.
//

//////////////
// JS BELOW //
//////////////

// this contains utility functions
var cputils = {

    // this encodes a string to base64
    // https://developer.mozilla.org/en-US/docs/Web/API/WindowBase64/Base64_encoding_and_decoding
    b64_encode: function (str) {
        return btoa(
            encodeURIComponent(str)
                .replace(/%([0-9A-F]{2})/g,
                         function(match, p1) {
                             return String.fromCharCode('0x' + p1);
                         }));
    },

    // this turns a base64 string into an image by updating its source
    b64_to_image: function (str, targetelem) {

        var datauri = 'data:image/png;base64,' + str;
        $(targetelem).attr('src',datauri);

    }

};


// this is the checkplot object
var cpv = {

    filelist: [],
    nfiles: 0,
    currfile: '',
    currcp:'',

    // this loads a checkplot from an image file into an HTML canvas object
    load_checkplot: function (filename) {

        console.log('loading ' + filename);

        // build the title for this current file
        var plottitle = $('#checkplot-current');
        var filelink = filename;
        var objectidelem = $('#objectid');
        var twomassidelem = $('#twomassid');

        plottitle.html(filelink);

        if (cpv.currfile.length > 0) {
            // un-highlight the previous file in side bar
            $("a[data-fname='" + cpv.currfile + "']").unwrap();
        }

        // do the AJAX call to get this checkplot
        var ajaxurl = '/cp/' + cputils.b64_encode(filename);

        $.getJSON(ajaxurl, function (data) {

            cpv.currcp = data.result;
            console.log('received cp for ' + cpv.currcp.objectid);

            /////////////////////////////////////////////////
            // update the UI with elems for this checkplot //
            /////////////////////////////////////////////////

            // update the objectid header
            objectidelem.html(cpv.currcp.objectid);
            // update the twomassid header
            twomassidelem.html('2MASS J' + cpv.currcp.objectinfo.twomassid);

            // update the finder chart
            cputils.b64_to_image(cpv.currcp.finderchart,
                                 '#finderchart');

            // update the objectinfo
            var hatinfo = '<strong>' +
                (cpv.currcp.objectinfo.stations.split(',')).join(', ') +
                '</strong><br>' +
                '<strong>LC points:</strong> ' +
                cpv.currcp.objectinfo.ndet;
            $('#hatinfo').html(hatinfo);

            var coordspm =
                '<strong>RA, Dec:</strong> ' +
                math.format(cpv.currcp.objectinfo.ra,6) + ', ' +
                math.format(cpv.currcp.objectinfo.decl,6) + '<br>' +
                '<strong>Total PM:</strong> ' +
                math.format(cpv.currcp.objectinfo.propermotion,5) +
                ' mas/yr<br>' +
                '<strong>Reduced PM:</strong> ' +
                math.format(cpv.currcp.objectinfo.reducedpropermotion,4);
            $('#coordspm').html(coordspm);

            var mags = '<strong><em>g, r, i</em>:</strong> ' +
                math.format(cpv.currcp.objectinfo.sdssg,4) + ', ' +
                math.format(cpv.currcp.objectinfo.sdssr,4) + ', ' +
                math.format(cpv.currcp.objectinfo.sdssi,4) + '<br>' +
                '<strong><em>J, H, K</em>:</strong> ' +
                math.format(cpv.currcp.objectinfo.jmag,4) + ', ' +
                math.format(cpv.currcp.objectinfo.hmag,4) + ', ' +
                math.format(cpv.currcp.objectinfo.kmag,4) + '<br>' +
                '<strong><em>B, V</em>:</strong> ' +
                math.format(cpv.currcp.objectinfo.bmag,4) + ', ' +
                math.format(cpv.currcp.objectinfo.vmag,4);
            $('#mags').html(mags);

            // update the varinfo

            // update the phased magseries
            cputils.b64_to_image(cpv.currcp.magseries,
                                '#magseriesplot');


        }).done(function () {

            console.log('done with cp');

            // update the current file tracker
            cpv.currfile = filename;
            // highlight the file in the sidebar list
            $("a[data-fname='" + filename + "']").wrap('<strong></strong>')

        }).fail (function (xhr) {

            console.log('cp loading failed from ' + ajaxurl);

        });


    },


    // this binds actions to the web-app controls
    action_setup: function () {

        // the previous checkplot link
        $('.checkplot-prev').on('click',function (evt) {

            evt.preventDefault();

            // find the current index
            var currindex = cpv.filelist.indexOf(cpv.currfile);
            var prevfile = cpv.filelist[currindex-1];
            if (prevfile != undefined) {
                cpv.load_checkplot(prevfile);
            }

        });

        // the next checkplot link
        $('.checkplot-next').on('click',function (evt) {

            evt.preventDefault();

            // find the current index
            var currindex = cpv.filelist.indexOf(cpv.currfile);
            var nextfile = cpv.filelist[currindex+1];
            if (nextfile != undefined) {
                cpv.load_checkplot(nextfile);
            }

        });

        // clicking on a checkplot file in the sidebar
        $('#pnglist').on('click', '.checkplot-load', function (evt) {

            evt.preventDefault();

            var filetoload = $(this).attr('data-fname');
            console.log('file to load: ' + filetoload);

            if (cpv.filelist.indexOf(filetoload) != -1) {
                cpv.load_checkplot(filetoload);
            }

        });

    }

};
