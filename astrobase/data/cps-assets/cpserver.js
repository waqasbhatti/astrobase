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


// this contains updates to current checkplots, tracks the entire project's info
// and provides functions to save everything to CSV or JSON
var cptracker = {

    // this is the actual object that will get written to JSON or CSV
    cpdata: {},

    // this is the order of columns for generating the CSV
    infocolumns: [
        'objectid','checkplot',
        'objectinfo.hatid','objectinfo.twomassid',
        'objectinfo.network','objectinfo.stations','objectinfo.ndet',
        'objectinfo.objecttags','objectinfo.bmag','objectinfo.vmag',
        'objectinfo.sdssg','objectinfo.sdssr','objectinfo.sdssi',
        'objectinfo.jmag','objectinfo.hmag','objectinfo.kmag',
        'objectinfo.bvcolor','objectinfo.ijcolor','objectinfo.jkcolor',
        'objectinfo.pmra','objectinfo.pmra_err',
        'objectinfo.pmdecl','objectinfo.pmdecl_err',
        'objectinfo.propermotion','objectinfo.reducedpropermotion',
        'varinfo.objectisvar','varinfo.varperiod','varinfo.varepoch',
        'varinfo.vartags','comments'
    ],

    // this function generates a CSV row for a single object in the
    // cptracker.cpdata object
    cpdata_get_csvrow: function (cpdkey, separator) {

        var thisdata = cptracker.cpdata[cpdkey];
        var rowarr = [];

        cptracker.infocolumns.forEach(function (e, i, a) {

            // if the key is a compound one (firstlev.secondlev)
            if (e.indexOf('.') != -1) {

                var keys = e.split('.');
                rowarr.push(thisdata[keys[0]][keys[1]]);

            }

            // otherwise, it's just in the first level
            else {

                // special case of checkplot filename
                if (e == 'checkplot') {
                    rowarr.push(cpdkey);
                }
                else {
                    rowarr.push(thisdata[e]);
                }
            }

        });

        // join the output row
        var csvrow = rowarr.join(separator);
        return csvrow

    },

    // this generates a CSV for download
    cpdata_to_csv: function () {

        var csvarr = [cptracker.infocolumns.join('|')];

        for (obj in cptracker.cpdata) {
            csvarr.push(cptracker.cpdata_get_csvrow(obj,'|'));
        }

        csvarr = "data:text/csv;charset=utf-8," +
            encodeURIComponent(csvarr.join('\n'));

        $('#download-anchor').attr('href', csvarr);
        $('#download-anchor').attr('download', 'project-objectlist.csv');
        $('#download-anchor').html('get CSV for reviewed objects');
        $('#download-anchor').css({'display': 'inline'});

    },

    // this generates a JSON for download
    cpdata_to_json: function () {

        // we need to reverse the keys of the cpdata, so they're objectid first,
        // add the object's checkplot file into its own dict
        var jsonobj = {};

        for (obj in cptracker.cpdata) {

            var thisdata = cptracker.cpdata[obj];
            thisdata['checkplot'] = obj;
            jsonobj[thisdata['objectid']] = thisdata;

        }

        // generate the string
        var jsonstr = "data:text/json;charset=utf-8," +
            encodeURIComponent(JSON.stringify(jsonobj));

        $('#download-anchor').attr('href', jsonstr);
        $('#download-anchor').attr('download', 'project-objectlist.json');
        $('#download-anchor').html('get JSON for reviewed objects');
        $('#download-anchor').css({'display': 'inline'});

    },

    // this saves a single object back to the checkplotlist JSON
    reviewed_object_to_cplist: function () {

        var saveinfo = {checkplot: cpv.currfile,
                        varinfo: cpv.currcp.varinfo,
                        objectinfo: cpv.currcp.objectinfo,
                        comments: cpv.currcp.comments}

        var postmsg = {objectid: cpv.currcp.objectid,
                       changes: JSON.stringify(saveinfo)}


        if (cpv.readonlymode) {
            // if we're in readonly mode, inform the user
            $('#alert-box').html(
                'The checkplot server is in readonly mode. ' +
                    'Edits to object information will not be saved.'
            );
        }

        // if we're not in readonly mode, post the results
        else {

            // send this back to the checkplotserver
            $.post('/list', postmsg, function (data) {

                var success = data.status;
                var message = data.message;
                var result = data.result;

                if (!success) {

                    $('#alert-box').html('could not save info for <strong>' +
                                   postmsg.objectid + '</strong>!');
                    console.log('saving the changes back to the '+
                                'JSON file failed for ' + postmsg.objectid);

                }


            }, 'json').fail(function (xhr) {

                $('#alert-box').html('could not save info for <strong>' +
                               postmsg.objectid + '</strong>!');
                console.log('saving the changes back to the '+
                            'JSON file failed for ' + postmsg.objectid);
            });

        }

    },

    // this loads all the reviewed objects from the current project's JSON
    // checkplot file list, and updates the reviewed objects list
    all_reviewed_from_cplist: function () {

        $.getJSON('/list', function (data) {

            var reviewedobjects = data.reviewed;

            // generate the object info rows and append to the reviewed object
            // list
            for (obj in reviewedobjects) {

                var objdict = reviewedobjects[obj];

                var objectid = obj;
                var objcp = objdict.checkplot;
                var objectinfo = objdict.objectinfo;
                var varinfo = objdict.varinfo;

                // generate the new list element: this contains objectid -
                // variability flag, variability tags, object tags
                var objectidelem = '<div class="tracker-obj" data-objectid="' +
                    objectid +
                    '"><a href="#" class="objload-checkplot" ' +
                    'data-fname="' + objcp + '" ';

                if (varinfo.objectisvar == '1') {
                    objectidelem = objectidelem +
                        'data-objectisvar="1"' +
                        '>' +
                        objectid + '</a>: ' +
                        'variable';
                }
                else if (varinfo.objectisvar == '2') {
                    objectidelem = objectidelem +
                        'data-objectisvar="2"' +
                        '>' +
                        objectid + '</a>: ' +
                        'not variable';
                }
                else if (varinfo.objectisvar == '3') {
                    objectidelem = objectidelem +
                        'data-objectisvar="3"' +
                        '>' +
                        objectid + '</a>: ' +
                        'maybe variable';
                }
                else if (varinfo.objectisvar == '0') {
                    objectidelem = objectidelem +
                        'data-objectisvar="0"' +
                        '>' +
                        objectid + '</a>: ' +
                        'no varflag set';
                }

                if (varinfo.vartags != null) {

                    var thisvartags =
                        varinfo.vartags.split(', ');

                    if (thisvartags[0].length > 0) {

                        objectidelem = objectidelem + ' ';
                        thisvartags.forEach(function (e, i, a) {
                            objectidelem = objectidelem +
                                '<span class="cp-tag">' +
                                e + '</span> ';
                        });
                    }
                }

                if (objectinfo.objecttags != null) {

                    var thisobjtags =
                        objectinfo.objecttags.split(', ');

                    if (thisobjtags[0].length > 0) {

                        objectidelem = objectidelem + ' ';
                        thisobjtags.forEach(function (e, i, a) {
                            objectidelem = objectidelem +
                                '<span class="cp-tag">' +
                                e + '</span> ';
                        });
                    }

                }

                // finish the objectidelem li tag
                objectidelem = objectidelem + '</div>';
                $('#project-status').append(objectidelem);

                // update the count in saved-count
                var nsaved = $('#project-status div').length;
                $('#saved-count').html(nsaved + '/' + cpv.totalcps);


            }

        }).fail(function (xhr) {

            $('#alert-box').html('could not load the existing checkplot list');
            console.log('could not load the existing checkplot list');

        });

    }

};


// this is the container for the main functions
var cpv = {

    // these hold the current checkplot's data and filename respectively
    currentfind: 0,
    currfile: '',
    currcp: {},
    totalcps: 0,
    cpfpng: null,

    // this checks if the server is in readonly mode. disables controls if so.
    readonlymode: false,

    // these help in moving quickly through the phased LCs
    currphasedind: null,
    maxphasedind: null,

    // this function generates a spinner
    make_spinner: function (spinnermsg) {

        var spinner = spinnermsg +
            '<div class="spinner">' +
            '<div class="rect1"></div>' +
            '<div class="rect2"></div>' +
            '<div class="rect3"></div>' +
            '<div class="rect4"></div>' +
            '<div class="rect5"></div>' +
            '</div>';

        $('#alert-box').html(spinner);

    },

    // this function generates an alert box
    make_alert: function (alertmsg) {

        var xalert =
            '<div class="alert alert-warning alert-dismissible fade show" ' +
            'role="alert">' +
            '<button type="button" class="close" data-dismiss="alert" ' +
            'aria-label="Close">' +
            '<span aria-hidden="true">&times;</span>' +
            '</button>' +
            alertmsg +
            '</div>';

        $('#alert-box').html(xalert);

    },

    // this loads a checkplot from an image file into an HTML canvas object
    load_checkplot: function (filename) {

        // remove the summary download anchor from view since it's stale now
        $('#download-anchor').css({'display': 'none'});

        // start the spinny thing
        cpv.make_spinner('loading...');

        // build the title for this current file
        var plottitle = $('#checkplot-current');
        var filelink = filename;
        var objectidelem = $('#objectid');
        var twomassidelem = $('#twomassid');

        plottitle.html(filelink);

        if (cpv.currfile.length > 0) {
            // un-highlight the previous file in side bar
            $("a.checkplot-load")
                .filter("[data-fname='" + cpv.currfile + "']")
                .unwrap();
        }

        // do the AJAX call to get this checkplot
        var ajaxurl = '/cp/' + cputils.b64_encode(filename);

        // add the current object's fname to the objectid data-fname
        objectidelem.attr('data-fname', filename);

        $.getJSON(ajaxurl, function (data) {

            cpv.currcp = data.result;

            /////////////////////////////////////////////////
            // update the UI with elems for this checkplot //
            /////////////////////////////////////////////////

            // update the readonly status
            cpv.readonlymode = data.readonly;

            // update the objectid header
            objectidelem.html(cpv.currcp.objectid);

            // update the twomassid header
            if (cpv.currcp.objectinfo.twomassid != undefined) {
                twomassidelem.html('2MASS J' + cpv.currcp.objectinfo.twomassid);
            }
            else {
                twomassidelem.html('');
            }

            // update the finder chart
            cputils.b64_to_image(cpv.currcp.finderchart,
                                 '#finderchart');


            var hatstations = cpv.currcp.objectinfo.stations;
            if (hatstations != undefined && hatstations) {

                var splitstations =
                    (String(hatstations).split(',')).join(', ');
            }
            else {
                var splitstations = 'no HAT observations';
            }

            var objndet = cpv.currcp.objectinfo.ndet;

            if (objndet == undefined) {
                objndet = cpv.currcp.magseries_ndet;
            }

            // update the objectinfo
            var hatinfo = '<strong>' +
                splitstations +
                '</strong><br>' +
                '<strong>LC points:</strong> ' + objndet;
            $('#hatinfo').html(hatinfo);

            var coordspm =
                '<strong>RA, Dec:</strong> ' +
                '<a title="SIMBAD search at these coordinates" ' +
                'href="http://simbad.u-strasbg.fr/simbad/sim-coo?Coord=' +
                cpv.currcp.objectinfo.ra + '+' + cpv.currcp.objectinfo.decl +
                '&Radius=1&Radius.unit=arcmin' +
                '" rel="nofollow" target="_blank">' +
                math.format(cpv.currcp.objectinfo.ra,6) + ', ' +
                math.format(cpv.currcp.objectinfo.decl,6) + '</a><br>' +
                '<strong>Total PM:</strong> ' +
                math.format(cpv.currcp.objectinfo.propermotion,5) +
                ' mas/yr<br>' +
                '<strong>Reduced PM:</strong> ' +
                math.format(cpv.currcp.objectinfo.reducedpropermotion,4);
            $('#coordspm').html(coordspm);

            var mags = '<strong><em>gri</em>:</strong> ' +
                math.format(cpv.currcp.objectinfo.sdssg,5) + ', ' +
                math.format(cpv.currcp.objectinfo.sdssr,5) + ', ' +
                math.format(cpv.currcp.objectinfo.sdssi,5) + '<br>' +
                '<strong><em>JHK</em>:</strong> ' +
                math.format(cpv.currcp.objectinfo.jmag,5) + ', ' +
                math.format(cpv.currcp.objectinfo.hmag,5) + ', ' +
                math.format(cpv.currcp.objectinfo.kmag,5) + '<br>' +
                '<strong><em>BV</em>:</strong> ' +
                math.format(cpv.currcp.objectinfo.bmag,5) + ', ' +
                math.format(cpv.currcp.objectinfo.vmag,5);
            $('#mags').html(mags);

            var colors = '<strong><em>B - V</em>:</strong> ' +
                math.format(cpv.currcp.objectinfo.bvcolor,4) + '<br>' +
                '<strong><em>i - J</em>:</strong> ' +
                math.format(cpv.currcp.objectinfo.ijcolor,4) + '<br>' +
                '<strong><em>J - K</em>:</strong> ' +
                math.format(cpv.currcp.objectinfo.jkcolor,4);
            $('#colors').html(colors);

            // update the magseries plot
            cputils.b64_to_image(cpv.currcp.magseries,
                                '#magseriesplot');

            // update the varinfo
            if (cpv.currcp.varinfo.objectisvar == 1) {

                $('#varcheck').val(1);

            }
            else if (cpv.currcp.varinfo.objectisvar == 2) {

                $('#varcheck').val(2);

            }
            else if (cpv.currcp.varinfo.objectisvar == 3) {

                $('#varcheck').val(3);

            }
            else {

                $('#varcheck').val(0);

            }
            $('#objectperiod').val(cpv.currcp.varinfo.varperiod);
            $('#objectepoch').val(cpv.currcp.varinfo.varepoch);
            $('#objecttags').val(cpv.currcp.objectinfo.objecttags);
            $('#objectcomments').val(cpv.currcp.objectcomments);
            $('#vartags').val(cpv.currcp.varinfo.vartags);

            // update the phased light curves

            // first, count the number of methods we have in the cp
            var lspmethods = [];
            var ncols = 0;

            if ('pdm' in cpv.currcp) {
                lspmethods.push('pdm');
                ncols = ncols + 1;
            }
            if ('gls' in cpv.currcp) {
                lspmethods.push('gls');
                ncols = ncols + 1;
            }
            if ('bls' in cpv.currcp) {
                lspmethods.push('bls');
                ncols = ncols + 1;
            }
            if ('aov' in cpv.currcp) {
                lspmethods.push('aov');
                ncols = ncols + 1;
            }

            var colwidth = 12/ncols;

            // zero out previous stuff
            $('.phased-container').empty();
            cpv.currphasedind = null;
            cpv.maxphasedind = null;

            // this is the fast scroll index for moving quickly through the
            // phased plots and selecting them as the best
            var phasedplotindex = 0;

            // then go through each lsp method, and generate the containers
            for (let lspmethod of lspmethods) {

                if (lspmethod in cpv.currcp) {

                    var nbestperiods = cpv.currcp[lspmethod].nbestperiods;
                    var periodogram = cpv.currcp[lspmethod].periodogram;

                    // start putting together the container for this method
                    var mcontainer_coltop =
                        '<div class="col-sm-' + colwidth +
                        '" "data-lspmethod="' + lspmethod + '">';
                    var mcontainer_colbot = '</div>';

                    var periodogram_row =
                        '<div class="row periodogram-container">' +
                        '<div class="col-sm-12">' +
                        '<img src="data:image/png;base64,' +
                        cpv.currcp[lspmethod].periodogram + '" ' +
                        'class="img-fluid" id="periodogram-' +
                        lspmethod + '">' + '</div></div>';

                    var phasedlcrows= [];

                    // up to 5 periods are possible
                    var periodindexes = ['phasedlc0',
                                         'phasedlc1',
                                         'phasedlc2',
                                         'phasedlc3',
                                         'phasedlc4'];

                    for (let periodind of periodindexes) {

                        if (periodind in cpv.currcp[lspmethod]) {

                            var phasedlcrow =
                                '<a href="#" class="phasedlc-select" ' +
                                'title="use this period and epoch" ' +
                                'data-lspmethod="' + lspmethod + '" ' +
                                'data-periodind="' + periodind + '" ' +
                                'data-phasedind="' + phasedplotindex + '" ' +
                                'data-currentbest="no" ' +
                                'data-period="' +
                                cpv.currcp[lspmethod][periodind].period + '" ' +
                                'data-epoch="' +
                                cpv.currcp[lspmethod][periodind].epoch + '">' +
                                '<div class="row py-1 phasedlc-container-row" ' +
                                'data-periodind="' + periodind + '">' +
                                '<div class="col-sm-12">' +
                                '<img src="data:image/png;base64,' +
                                cpv.currcp[lspmethod][periodind].plot + '"' +
                                'class="img-fluid" id="plot-' +
                                periodind + '">' + '</div></div></a>';

                            phasedlcrows.push(phasedlcrow);
                            phasedplotindex = phasedplotindex + 1;

                        }

                    }

                    // now that we've collected everything, generate the
                    // container column
                    var mcontainer = mcontainer_coltop + periodogram_row +
                        phasedlcrows.join(' ') + mcontainer_colbot;

                    // write the column to the phasedlc-container
                    $('.phased-container').append(mcontainer);

                }

            }

            // write the max phasedind
            cpv.maxphasedind = phasedplotindex;


        }).done(function () {

            // update the current file trackers
            cpv.currfile = filename;
            cpv.currentfind = parseInt(
                $("a.checkplot-load")
                    .filter("[data-fname='" + filename + "']")
                    .attr('data-findex')
            );


            // FIXME: get this checkplot's cptools results
            // if no cptools results, then populate the phased LCs
            // - on the psearch tab, with GLS pgram + phasedlc0
            // - on the var tab, with phasedlc0
            // - on the lcfit tab, with phasedlc0

            // FIXME: if the checkplot's cptools results are available in either
            // the cptools.allresults object or in the temp checkplot, get them
            // from there. this will probably need a separate call to
            // cptools.load_results() and an AJAX endpoint

            // FIXME: on the psearch tab, need to populate select with peaks for
            // each period finding method in the checkplot

            // FIXME: when we load the checkplot, load it's cptools results into
            // the cptools.current object.

            // highlight the file in the sidebar list
            $("a.checkplot-load")
                .filter("[data-fname='" + filename + "']")
                .wrap('<strong></strong>')

            // fix the height of the sidebar as required
            var winheight = $(window).height();
            var docheight = $(document).height();
            var ctrlheight = $('.sidebar-controls').height()
            $('.sidebar').css({'height': docheight + 'px'});

            // get rid of the spinny thing
            $('#alert-box').empty();

            if (cpv.readonlymode) {
                // if we're in readonly mode, inform the user
                $('#alert-box').html(
                    'The checkplot server is in readonly mode. ' +
                        'Edits to object information will not be saved.'
                );
            }

        }).fail (function (xhr) {

            $('#alert-box').html('could not load checkplot <strong>' +
                           filename + '</strong>!');

        });


    },

    // this functions saves the current checkplot by doing a POST request to the
    // backend. this MUST be called on every checkplot list action (i.e. next,
    // prev, before load of a new checkplot, so changes are always saved). UI
    // elements in the checkplot list will tag the saved checkplots
    // appropriately
    save_checkplot: function (nextfunc_callback, nextfunc_arg, savetopng) {

        // make sure we're not in readonly mode
        // if we are, then just bail out immediately
        if (cpv.readonlymode) {

            $('#alert-box').html(
                'The checkplot server is in readonly mode. ' +
                    'Edits to object information will not be saved.'
            );

            // call the next function. we call this here so we can be sure the
            // next action starts correctly even if we're not saving anything
            if (!(nextfunc_callback === undefined) &&
                !(nextfunc_arg === undefined)) {
                nextfunc_callback(nextfunc_arg);
            }

        }

        // if we're not in readonly mode, go ahead and save stuff
        else {

            // do the AJAX call to get this checkplot
            var ajaxurl = '/cp/' + cputils.b64_encode(cpv.currfile);

            // get the current value of the objectisvar select box
            cpv.currcp.varinfo.objectisvar = $('#varcheck').val();

            // make sure that we've saved the input varinfo, objectinfo and
            // comments
            cpv.currcp.varinfo.vartags = $('#vartags').val();
            cpv.currcp.objectinfo.objecttags = $('#objecttags').val();
            cpv.currcp.objectcomments = $('#objectcomments').val();

            var cppayload = JSON.stringify(
                {objectid: cpv.currcp.objectid,
                 objectinfo: cpv.currcp.objectinfo,
                 varinfo: cpv.currcp.varinfo,
                 comments: cpv.currcp.objectcomments}
            );

            // first, generate the object to send with the POST request
            var postobj = {cpfile: cpv.currfile,
                           cpcontents: cppayload,
                           savetopng: savetopng};

            // this is to deal with UI elements later
            var currfile = postobj.cpfile;

            // next, do a saving animation in the alert box
            cpv.make_spinner('saving...');

            // next, send the POST request and handle anything the server
            // returns. FIXME: this should use _xsrf once we set that up
            $.post(ajaxurl, postobj, function (data) {

                // get the info from the backend
                var updatestatus = data.status;
                var updatemsg = data.message;
                var updateinfo = data.result;

                // update the cptracker with what changed so we can try to undo
                // later if necessary.
                if (updatestatus == 'success') {

                    // store only the latest update in the tracker
                    // FIXME: think about adding in update history
                    // probably a better fit for indexedDB or something
                    cptracker.cpdata[postobj.cpfile] = updateinfo.changes;

                    // check if this object is already present and remove if it
                    // so
                    var statobjcheck = $('.tracker-obj').filter(
                        '[data-objectid="' +
                            updateinfo.changes.objectid +
                            '"]'
                    );

                    // we need to update the project status widget

                    // generate the new list element: this contains objectid -
                    // variability flag, variability tags, object tags
                    var objectli =
                        '<div class="tracker-obj" ' +
                        'data-objectid="' + updateinfo.changes.objectid + '">';

                    var objectidelem =  '<a class="objload-checkplot" ' +
                        'href="#" data-fname="' + postobj.cpfile + '" ' +
                        'data-objectisvar="' +
                        updateinfo.changes.varinfo.objectisvar + '">' +
                        updateinfo.changes.objectid +
                        '</a>:';

                    if (updateinfo.changes.varinfo.objectisvar == '1') {
                        var varelem = 'variable';
                    }
                    else if (updateinfo.changes.varinfo.objectisvar == '2') {
                        var varelem = 'not variable';
                    }
                    else if (updateinfo.changes.varinfo.objectisvar == '3') {
                        var varelem = 'maybe variable';
                    }
                    else if (updateinfo.changes.varinfo.objectisvar == '0') {
                        var varelem = 'no varflag set';
                    }

                    var thisvartags =
                        updateinfo.changes.varinfo.vartags.split(', ');

                    var thisobjtags =
                        updateinfo.changes.objectinfo.objecttags.split(', ');

                    var vartaglist = [];

                    if ((thisvartags != null) && (thisvartags[0].length > 0)) {

                        thisvartags.forEach(function (e, i, a) {
                            vartaglist.push('<span class="cp-tag">' +
                                            e + '</span>');
                        });

                        vartaglist = vartaglist.join(' ');
                    }
                    else {
                        vartaglist = '';
                    }

                    var objtaglist = [];

                    if ((thisobjtags != null) && (thisobjtags[0].length > 0)) {

                        thisobjtags.forEach(function (e, i, a) {
                            objtaglist.push('<span class="cp-tag">' +
                                            e + '</span>');
                        });

                        objtaglist = objtaglist.join(' ');
                    }
                    else {
                        objtaglist = '';
                    }

                    var finelem = [objectidelem,
                                   varelem,
                                   vartaglist,
                                   objtaglist].join(' ');

                    // if this object exists in the list already
                    // replace it with the new content
                    if (statobjcheck.length > 0) {

                        statobjcheck.html(finelem);
                        console.log('updating existing entry for ' +
                                    updateinfo.changes.objectid);
                    }

                    // if this object doesn't exist, add a new row
                    else {
                        console.log('adding new entry for ' +
                                    updateinfo.changes.objectid);
                        $('#project-status').append(objectli + finelem +
                                                    '</div>');
                    }

                    // update the count in saved-count
                    var nsaved = $('#project-status div').length;
                    $('#saved-count').html(nsaved + '/' + cpv.totalcps);

                    // if we called a save to PNG, show it if it succeeded
                    if (!(savetopng === undefined) &&
                        (updateinfo.cpfpng != 'png making failed')) {

                        updatemsg = 'saved PNG to:<br>' +
                            '<textarea rows="3" ' +
                            'class="form-control" readonly>' +
                            updateinfo.cpfpng +
                            '</textarea>';
                        $('#alert-box').html(updatemsg);

                    }
                    else if (!(savetopng === undefined) &&
                        (updateinfo.cpfpng == 'png making failed')) {

                        updatemsg = 'sorry, making a PNG for ' +
                            'this object failed!';
                        $('#alert-box').html(updatemsg);

                    }

                }

                else {
                    $('#alert-box').html(updatemsg);
                }

                // on POST done, update the UI elements in the checkplot list
                // and call the next function.
            },'json').done(function (xhr) {

                // send the changes to the backend so they're present in the
                // checkplot-filelist.json file for the next time around
                cptracker.reviewed_object_to_cplist();

                // update the filter select box
                $('#reviewedfilter').change();

                // call the next function. we call this here so we can be sure
                // the save finished before the next action starts
                if (!(nextfunc_callback === undefined) &&
                    !(nextfunc_arg === undefined)) {
                    nextfunc_callback(nextfunc_arg);
                    // clean out the alert box if there's a next function
                    $('#alert-box').empty();

                }

                // if POST failed, pop up an alert in the alert box
            }).fail(function (xhr) {

                var errmsg = 'could not update ' +
                    currfile + ' because of an internal server error';
                $('#alert-box').html(errmsg);

            });

        }

    },


    // this binds actions to the web-app controls
    action_setup: function () {

        // the previous checkplot link
        $('.checkplot-prev').on('click',function (evt) {

            evt.preventDefault();

            // find the current index
            var prevfilelink = $("a.checkplot-load")
                .filter("[data-findex='" +
                        (cpv.currentfind-1) + "']");
            var prevfile = prevfilelink.attr('data-fname');

            if (prevfile != undefined) {
                cpv.save_checkplot(cpv.load_checkplot,prevfile);
                // $(prevfilelink)[0].scrollIntoView();
            }
            else {
                // make sure to save current
                cpv.save_checkplot(null,null);
            }

        });

        // the next checkplot link
        $('.checkplot-next').on('click',function (evt) {

            evt.preventDefault();

            // find the current index
            var nextfilelink = $("a.checkplot-load")
                .filter("[data-findex='" +
                        (cpv.currentfind+1) + "']");
            var nextfile = nextfilelink.attr('data-fname');

            if (nextfile != undefined) {
                cpv.save_checkplot(cpv.load_checkplot,nextfile);
                // $(nextfilelink)[0].scrollIntoView();
            }
            else {
                // make sure to save current
                cpv.save_checkplot(null,null);
            }

        });


        // clicking on the generate JSON button
        $('#save-project-json').click(function (evt) {

            // make sure we have at least one object in the saved list
            nsaved = $('#project-status div').length;

            if (nsaved > 0) {
                cptracker.cpdata_to_json();
            }

        });

        // clicking on the generate CSV button
        $('#save-project-csv').click(function (evt) {

            // make sure we have at least one object in the saved list
            nsaved = $('#project-status div').length;

            if (nsaved > 0) {
                cptracker.cpdata_to_csv();
            }

        });


        $('#reviewedfilter').on('change', function (evt) {

            // get this value
            var filterval = $(this).val();

            if (filterval == 0) {
                $('.objload-checkplot')
                    .filter('[data-objectisvar="0"]')
                    .parent()
                    .show();
                $('.objload-checkplot')
                    .filter('[data-objectisvar!="0"]')
                    .parent()
                    .hide();
            }
            else if (filterval == 1) {
                $('.objload-checkplot')
                    .filter('[data-objectisvar="1"]')
                    .parent()
                    .show();
                $('.objload-checkplot')
                    .filter('[data-objectisvar!="1"]')
                    .parent()
                    .hide();
            }
            else if (filterval == 2) {
                $('.objload-checkplot')
                    .filter('[data-objectisvar="2"]')
                    .parent()
                    .show();
                $('.objload-checkplot')
                    .filter('[data-objectisvar!="2"]')
                    .parent()
                    .hide();
            }
            else if (filterval == 3) {
                $('.objload-checkplot')
                    .filter('[data-objectisvar="3"]')
                    .parent()
                    .show();
                $('.objload-checkplot')
                    .filter('[data-objectisvar!="3"]')
                    .parent()
                    .hide();
            }
            else {
                $('.objload-checkplot').parent().show();
            }

        });


        // FIXME: fill this in later
        $('#checkplotfilter').on('change', function (evt) {

            // get this value
            var filterval = $(this).val();

            if (filterval == 0) {
                $('.checkplot-load').show();
            }

            else if (filterval == 1) {

                var reviewedcps = [];


            }

            else if (filterval == 2) {

            }

        });



        // this handles adding object tags from the dropdown
        $('.objtag-dn').click(function (evt) {

            evt.preventDefault();

            var thisobjtag = $(this).attr('data-dnobjtag');

            // get the current val for the objecttags
            var objecttags = $('#objecttags').val();

            // split by comma and strip extra spaces
            objecttags = objecttags.split(',');
            objecttags.forEach(function (item, index, arr) {
                objecttags[index] = item.trim();
            });

            // remove any item with zero length
            objecttags = objecttags
                .filter(function(val) { return val.length > 0 });

            // check if we already have this vartag in the list and append it if
            // we don't.
            if (objecttags.indexOf(thisobjtag) == -1) {
                objecttags.push(thisobjtag);
                objecttags = objecttags.join(', ');
                $('#objecttags').val(objecttags);
            }

        });

        // this handles adding object tags from the dropdown
        $('.vartag-dn').click(function (evt) {

            evt.preventDefault();

            var thisvartag = $(this).attr('data-dnvartag');

            // get the current val for the vartags
            var vartags = $('#vartags').val();

            // split by comma and strip extra spaces
            vartags = vartags.split(',');
            vartags.forEach(function (item, index, arr) {
                vartags[index] = item.trim();
            });

            // remove any item with zero length
            vartags = vartags
                .filter(function(val) { return val.length > 0 });

            // check if we already have this vartag in the list and append it if
            // we don't.
            if (vartags.indexOf(thisvartag) == -1) {
                vartags.push(thisvartag);
                vartags = vartags.join(', ');
                $('#vartags').val(vartags);
            }

        });

        // clicking on a checkplot file in the sidebar
        $('#checkplotlist').on('click', '.checkplot-load', function (evt) {

            evt.preventDefault();

            var filetoload = $(this).attr('data-fname');

            // save the currentcp if one exists, use the load_checkplot as a
            // callback to load the next one
            if (('objectid' in cpv.currcp) && (cpv.currfile.length > 0))  {
                cpv.save_checkplot(cpv.load_checkplot,filetoload);
            }

            else {
                // ask the backend for this file
                cpv.load_checkplot(filetoload);
            }

        });

        // clicking on a checkplot file in the sidebar
        $('#project-status').on('click', '.objload-checkplot', function (evt) {

            evt.preventDefault();

            var filetoload = $(this).attr('data-fname');

            // save the currentcp if one exists, use the load_checkplot as a
            // callback to load the next one
            if (('objectid' in cpv.currcp) && (cpv.currfile.length > 0))  {
                cpv.save_checkplot(cpv.load_checkplot,filetoload);
            }

            else {
                // ask the backend for this file
                cpv.load_checkplot(filetoload);
            }

        });


        // clicking on a phased LC loads its period and epoch into the boxes
        // also saves them to the currcp
        $('.phased-container').on('click','.phasedlc-select', function (evt) {

            evt.preventDefault();

            var period = $(this).attr('data-period');
            var epoch = $(this).attr('data-epoch');

            // update the boxes
            $('#objectperiod').val(period);
            $('#objectepoch').val(epoch);

            // save to currcp
            cpv.currcp.varinfo.varperiod = parseFloat(period);
            cpv.currcp.varinfo.varepoch = parseFloat(epoch);

            // add a selected class
            var selector = '[data-periodind="' +
                $(this).attr('data-periodind') +
                '"]';
            $('.phasedlc-container-row').removeClass('phasedlc-selected');
            $(this)
                .children('.phasedlc-container-row')
                .filter(selector).addClass('phasedlc-selected');

            // change the variability flag to 'probably variable' automatically.
            // since we've set a period and epoch, we probably think this is a
            // variable
            var currvarflag = $('#varcheck').val();

            // we only change if the flag is not set already
            if (currvarflag == 0) {
                $('#varcheck').val(1);
            }



        });

        // resizing the window fixes the sidebar again
        $(window).on('resize', function (evt) {

            // fix the height of the sidebar as required
            var winheight = $(window).height();
            var docheight = $(document).height();
            var ctrlheight = $('.sidebar-controls').height()

            $('.sidebar').css({'height': docheight + 'px'});

        });

        $('#save-to-png').on('click', function(evt) {

            evt.preventDefault();
            cpv.save_checkplot(undefined, undefined, true);

        });

    },

    // this does keyboard shortcut setup
    keyboard_setup: function () {

        /////////////////////////
        // SETTING VARIABILITY //
        /////////////////////////

        // alt+shift+v: object is variable
        Mousetrap.bind('alt+shift+v', function() {
            $('#varcheck').val(1);
        });

        // alt+shift+u: object is not variable
        Mousetrap.bind('alt+shift+n', function() {
            $('#varcheck').val(2);
        });

        // alt+shift+m: object is maybe a variable
        Mousetrap.bind('alt+shift+m', function() {
            $('#varcheck').val(3);
        });

        // alt+shift+u: unset variability flag
        Mousetrap.bind('alt+shift+u', function() {
            $('#varcheck').val(0);
        });


        //////////////
        // MOVEMENT //
        //////////////

        // ctrl+right: save this, move to next checkplot
        Mousetrap.bind(['ctrl+right','shift+right'], function() {
            $('.checkplot-next').click();
        });

        // ctrl+left: save this, move to prev checkplot
        Mousetrap.bind(['ctrl+left','shift+left'], function() {
            $('.checkplot-prev').click();
        });

        // ctrl+enter: save this, move to next checkplot
        Mousetrap.bind('ctrl+enter', function() {
            $('.checkplot-next').click();
        });


        // shift+enter: save this, but don't go anywhere
        Mousetrap.bind('shift+enter', function() {
            cpv.save_checkplot(undefined, undefined);
            $('#alert-box').empty();
        });

        // ctrl+shift+e: save this as a PNG
        Mousetrap.bind('alt+shift+e', function() {
            cpv.save_checkplot(undefined, undefined, true);
        });


        // ctrl+down: move to the next phased LC and set it as the best
        Mousetrap.bind(['ctrl+shift+down'], function() {

            // check the current phased index, if it's null, then set it to 0
            if (cpv.currphasedind == null) {
                cpv.currphasedind = 0;
            }
            else if (cpv.currphasedind < cpv.maxphasedind) {
                cpv.currphasedind = cpv.currphasedind + 1;
            }

            var targetelem = $('a[data-phasedind="' +
                               cpv.currphasedind + '"]');

            if (targetelem.length > 0) {

                // scroll into view if the bottom of this plot is off the screen
                if ( (targetelem.offset().top + targetelem.height()) >
                     $(window).height() ) {
                    targetelem[0].scrollIntoView(true);
                }

                // click on the target elem to select it
                targetelem.click();

            }

        });

        // ctrl+up: move to the prev phased LC and set it as the best
        Mousetrap.bind(['ctrl+shift+up'], function() {

            // check the current phased index, if it's null, then set it to 0
            if (cpv.currphasedind == null) {
                cpv.currphasedind = 0;
            }
            else if (cpv.currphasedind > 0) {
                cpv.currphasedind = cpv.currphasedind - 1;
            }

            var targetelem = $('a[data-phasedind="' +
                               cpv.currphasedind + '"]');

            if (targetelem.length > 0) {

                // scroll into view if the top of this plot is off the screen
                if ( (targetelem.offset().top) > $(window).height() ) {
                    targetelem[0].scrollIntoView(true);
                }

                // click on the target elem to select it
                targetelem.click();

            }

        });

        // ctrl+backspace: clear variability tags
        Mousetrap.bind('ctrl+backspace', function() {

            // clean out the variability info and input boxes
            $('#vartags').val('');
            $('#objectperiod').val('');
            $('#objectepoch').val('');
            $('#varcheck').val(0);

            cpv.currcp.varinfo.objectisvar = null;
            cpv.currcp.varinfo.varepoch = null;
            cpv.currcp.varinfo.varisperiodic = null;
            cpv.currcp.varinfo.varperiod = null;
            cpv.currcp.varinfo.vartags = null;
        });

        // ctrl+shift+backspace: clear all info
        Mousetrap.bind('ctrl+shift+backspace', function() {

            // clean out the all info and input boxes
            $('#vartags').val('');
            $('#objectperiod').val('');
            $('#objectepoch').val('');
            $('#objecttags').val('');
            $('#objectcomments').val('');
            $('#varcheck').val(0);

            cpv.currcp.varinfo.objectisvar = null;
            cpv.currcp.varinfo.varepoch = null;
            cpv.currcp.varinfo.varisperiodic = null;
            cpv.currcp.varinfo.varperiod = null;
            cpv.currcp.varinfo.vartags = null;
            cpv.currcp.objectinfo.objecttags = null;
            cpv.currcp.comments = null;
        });


        ///////////////////////
        // TAGGING VARIABLES //
        ///////////////////////

        // ctrl+shift+1: planet candidate
        Mousetrap.bind('ctrl+shift+1', function () {

            // get the current val for the vartags
            var vartags = $('#vartags').val();

            // split by comma and strip extra spaces
            vartags = vartags.split(',');
            vartags.forEach(function (item, index, arr) {
                vartags[index] = item.trim();
            });

            // remove any item with zero length
            vartags = vartags
                .filter(function(val) { return val.length > 0 });

            // check if we already have this vartag in the list and append it if
            // we don't.
            if (vartags.indexOf('planet candidate') == -1) {
                vartags.push('planet candidate');
                vartags = vartags.join(', ');
                $('#vartags').val(vartags);
            }

        });

        // ctrl+shift+2: RRab pulsator
        Mousetrap.bind('ctrl+shift+2', function () {

            // get the current val for the vartags
            var vartags = $('#vartags').val();

            // split by comma and strip extra spaces
            vartags = vartags.split(',');
            vartags.forEach(function (item, index, arr) {
                vartags[index] = item.trim();
            });

            // remove any item with zero length
            vartags = vartags
                .filter(function(val) { return val.length > 0 });

            // check if we already have this vartag in the list and append it if
            // we don't.
            if (vartags.indexOf('RRab pulsator') == -1) {
                vartags.push('RRab pulsator');
                vartags = vartags.join(', ');
                $('#vartags').val(vartags);
            }

        });

        // ctrl+shift+3: RRc pulsator
        Mousetrap.bind('ctrl+shift+3', function () {

            // get the current val for the vartags
            var vartags = $('#vartags').val();

            // split by comma and strip extra spaces
            vartags = vartags.split(',');
            vartags.forEach(function (item, index, arr) {
                vartags[index] = item.trim();
            });

            // remove any item with zero length
            vartags = vartags
                .filter(function(val) { return val.length > 0 });

            // check if we already have this vartag in the list and append it if
            // we don't.
            if (vartags.indexOf('RRc pulsator') == -1) {
                vartags.push('RRc pulsator');
                vartags = vartags.join(', ');
                $('#vartags').val(vartags);
            }

        });

        // ctrl+shift+4: starspot rotation
        Mousetrap.bind('ctrl+shift+4', function () {

            // get the current val for the vartags
            var vartags = $('#vartags').val();

            // split by comma and strip extra spaces
            vartags = vartags.split(',');
            vartags.forEach(function (item, index, arr) {
                vartags[index] = item.trim();
            });

            // remove any item with zero length
            vartags = vartags
                .filter(function(val) { return val.length > 0 });

            // check if we already have this vartag in the list and append it if
            // we don't.
            if (vartags.indexOf('starspot rotation') == -1) {
                vartags.push('starspot rotation');
                vartags = vartags.join(', ');
                $('#vartags').val(vartags);
            }

        });

        // ctrl+shift+5: flare star
        Mousetrap.bind('ctrl+shift+5', function () {

            // get the current val for the vartags
            var vartags = $('#vartags').val();

            // split by comma and strip extra spaces
            vartags = vartags.split(',');
            vartags.forEach(function (item, index, arr) {
                vartags[index] = item.trim();
            });

            // remove any item with zero length
            vartags = vartags
                .filter(function(val) { return val.length > 0 });

            // check if we already have this vartag in the list and append it if
            // we don't.
            if (vartags.indexOf('flare star') == -1) {
                vartags.push('flare star');
                vartags = vartags.join(', ');
                $('#vartags').val(vartags);
            }

        });

        // ctrl+shift+6: contact EB
        Mousetrap.bind('ctrl+shift+6', function () {

            // get the current val for the vartags
            var vartags = $('#vartags').val();

            // split by comma and strip extra spaces
            vartags = vartags.split(',');
            vartags.forEach(function (item, index, arr) {
                vartags[index] = item.trim();
            });

            // remove any item with zero length
            vartags = vartags
                .filter(function(val) { return val.length > 0 });

            // check if we already have this vartag in the list and append it if
            // we don't.
            if (vartags.indexOf('contact EB') == -1) {
                vartags.push('contact EB');
                vartags = vartags.join(', ');
                $('#vartags').val(vartags);
            }

        });

        // ctrl+shift+7: semi-detached EB
        Mousetrap.bind('ctrl+shift+7', function () {

            // get the current val for the vartags
            var vartags = $('#vartags').val();

            // split by comma and strip extra spaces
            vartags = vartags.split(',');
            vartags.forEach(function (item, index, arr) {
                vartags[index] = item.trim();
            });

            // remove any item with zero length
            vartags = vartags
                .filter(function(val) { return val.length > 0 });

            // check if we already have this vartag in the list and append it if
            // we don't.
            if (vartags.indexOf('semi-detached EB') == -1) {
                vartags.push('semi-detached EB');
                vartags = vartags.join(', ');
                $('#vartags').val(vartags);
            }

        });

        // ctrl+shift+8: detached EB
        Mousetrap.bind('ctrl+shift+8', function () {

            // get the current val for the vartags
            var vartags = $('#vartags').val();

            // split by comma and strip extra spaces
            vartags = vartags.split(',');
            vartags.forEach(function (item, index, arr) {
                vartags[index] = item.trim();
            });

            // remove any item with zero length
            vartags = vartags
                .filter(function(val) { return val.length > 0 });

            // check if we already have this vartag in the list and append it if
            // we don't.
            if (vartags.indexOf('detached EB') == -1) {
                vartags.push('detached EB');
                vartags = vartags.join(', ');
                $('#vartags').val(vartags);
            }

        });

        // ctrl+shift+9: weird variability
        Mousetrap.bind('ctrl+shift+9', function () {

            // get the current val for the vartags
            var vartags = $('#vartags').val();

            // split by comma and strip extra spaces
            vartags = vartags.split(',');
            vartags.forEach(function (item, index, arr) {
                vartags[index] = item.trim();
            });

            // remove any item with zero length
            vartags = vartags
                .filter(function(val) { return val.length > 0 });

            // check if we already have this vartag in the list and append it if
            // we don't.
            if (vartags.indexOf('weird variability') == -1) {
                vartags.push('weird variability');
                vartags = vartags.join(', ');
                $('#vartags').val(vartags);
            }

        });

        // ctrl+shift+0: period harmonic
        Mousetrap.bind('ctrl+shift+0', function () {

            // get the current val for the vartags
            var vartags = $('#vartags').val();

            // split by comma and strip extra spaces
            vartags = vartags.split(',');
            vartags.forEach(function (item, index, arr) {
                vartags[index] = item.trim();
            });

            // remove any item with zero length
            vartags = vartags
                .filter(function(val) { return val.length > 0 });

            // check if we already have this vartag in the list and append it if
            // we don't.
            if (vartags.indexOf('period harmonic') == -1) {
                vartags.push('period harmonic');
                vartags = vartags.join(', ');
                $('#vartags').val(vartags);
            }

        });


        //////////////////////////
        // TAGGING OBJECT TYPES //
        //////////////////////////

        // alt+shift+1: white dwarf
        Mousetrap.bind(['alt+shift+1','command+shift+1'], function () {

            // get the current val for the objecttags
            var objecttags = $('#objecttags').val();

            // split by comma and strip extra spaces
            objecttags = objecttags.split(',');
            objecttags.forEach(function (item, index, arr) {
                objecttags[index] = item.trim();
            });

            // remove any item with zero length
            objecttags = objecttags
                .filter(function(val) { return val.length > 0 });

            // check if we already have this objecttag in the list and append it
            // if we don't.
            if (objecttags.indexOf('white dwarf') == -1) {
                objecttags.push('white dwarf');
                objecttags = objecttags.join(', ');
                $('#objecttags').val(objecttags);
            }

        });

        // alt+shift+2: hot star (OB)
        Mousetrap.bind(['alt+shift+2','command+shift+2'], function () {

            // get the current val for the objecttags
            var objecttags = $('#objecttags').val();

            // split by comma and strip extra spaces
            objecttags = objecttags.split(',');
            objecttags.forEach(function (item, index, arr) {
                objecttags[index] = item.trim();
            });

            // remove any item with zero length
            objecttags = objecttags
                .filter(function(val) { return val.length > 0 });

            // check if we already have this objecttag in the list and append it
            // if we don't.
            if (objecttags.indexOf('hot star (OB)') == -1) {
                objecttags.push('hot star (OB)');
                objecttags = objecttags.join(', ');
                $('#objecttags').val(objecttags);
            }

        });

        // alt+shift+3: A star
        Mousetrap.bind(['alt+shift+3','command+shift+3'], function () {

            // get the current val for the objecttags
            var objecttags = $('#objecttags').val();

            // split by comma and strip extra spaces
            objecttags = objecttags.split(',');
            objecttags.forEach(function (item, index, arr) {
                objecttags[index] = item.trim();
            });

            // remove any item with zero length
            objecttags = objecttags
                .filter(function(val) { return val.length > 0 });

            // check if we already have this objecttag in the list and append it
            // if we don't.
            if (objecttags.indexOf('A star') == -1) {
                objecttags.push('A star');
                objecttags = objecttags.join(', ');
                $('#objecttags').val(objecttags);
            }

        });

        // alt+shift+4: F or G dwarf
        Mousetrap.bind(['alt+shift+4','command+shift+4'], function () {

            // get the current val for the objecttags
            var objecttags = $('#objecttags').val();

            // split by comma and strip extra spaces
            objecttags = objecttags.split(',');
            objecttags.forEach(function (item, index, arr) {
                objecttags[index] = item.trim();
            });

            // remove any item with zero length
            objecttags = objecttags
                .filter(function(val) { return val.length > 0 });

            // check if we already have this objecttag in the list and append it
            // if we don't.
            if (objecttags.indexOf('F or G dwarf') == -1) {
                objecttags.push('F or G dwarf');
                objecttags = objecttags.join(', ');
                $('#objecttags').val(objecttags);
            }

        });

        // alt+shift+5: red giant
        Mousetrap.bind(['alt+shift+5','command+shift+5'], function () {

            // get the current val for the objecttags
            var objecttags = $('#objecttags').val();

            // split by comma and strip extra spaces
            objecttags = objecttags.split(',');
            objecttags.forEach(function (item, index, arr) {
                objecttags[index] = item.trim();
            });

            // remove any item with zero length
            objecttags = objecttags
                .filter(function(val) { return val.length > 0 });

            // check if we already have this objecttag in the list and append it
            // if we don't.
            if (objecttags.indexOf('red giant') == -1) {
                objecttags.push('red giant');
                objecttags = objecttags.join(', ');
                $('#objecttags').val(objecttags);
            }

        });

        // alt+shift+6: K or M dwarf
        Mousetrap.bind(['alt+shift+6','command+shift+6'], function () {

            // get the current val for the objecttags
            var objecttags = $('#objecttags').val();

            // split by comma and strip extra spaces
            objecttags = objecttags.split(',');
            objecttags.forEach(function (item, index, arr) {
                objecttags[index] = item.trim();
            });

            // remove any item with zero length
            objecttags = objecttags
                .filter(function(val) { return val.length > 0 });

            // check if we already have this objecttag in the list and append it
            // if we don't.
            if (objecttags.indexOf('K or M dwarf') == -1) {
                objecttags.push('K or M dwarf');
                objecttags = objecttags.join(', ');
                $('#objecttags').val(objecttags);
            }

        });

        // alt+shift+7: giant star
        Mousetrap.bind(['alt+shift+7','command+shift+7'], function () {

            // get the current val for the objecttags
            var objecttags = $('#objecttags').val();

            // split by comma and strip extra spaces
            objecttags = objecttags.split(',');
            objecttags.forEach(function (item, index, arr) {
                objecttags[index] = item.trim();
            });

            // remove any item with zero length
            objecttags = objecttags
                .filter(function(val) { return val.length > 0 });

            // check if we already have this objecttag in the list and append it
            // if we don't.
            if (objecttags.indexOf('giant star') == -1) {
                objecttags.push('giant star');
                objecttags = objecttags.join(', ');
                $('#objecttags').val(objecttags);
            }

        });

        // alt+shift+8: dwarf star
        Mousetrap.bind(['alt+shift+8','command+shift+8'], function () {

            // get the current val for the objecttags
            var objecttags = $('#objecttags').val();

            // split by comma and strip extra spaces
            objecttags = objecttags.split(',');
            objecttags.forEach(function (item, index, arr) {
                objecttags[index] = item.trim();
            });

            // remove any item with zero length
            objecttags = objecttags
                .filter(function(val) { return val.length > 0 });

            // check if we already have this objecttag in the list and append it
            // if we don't.
            if (objecttags.indexOf('dwarf star') == -1) {
                objecttags.push('dwarf star');
                objecttags = objecttags.join(', ');
                $('#objecttags').val(objecttags);
            }

        });

        // alt+shift+9: blended with neighbors
        Mousetrap.bind(['alt+shift+9','command+shift+9'], function () {

            // get the current val for the objecttags
            var objecttags = $('#objecttags').val();

            // split by comma and strip extra spaces
            objecttags = objecttags.split(',');
            objecttags.forEach(function (item, index, arr) {
                objecttags[index] = item.trim();
            });

            // remove any item with zero length
            objecttags = objecttags
                .filter(function(val) { return val.length > 0 });

            // check if we already have this objecttag in the list and append it
            // if we don't.
            if (objecttags.indexOf('blended with neighbors') == -1) {
                objecttags.push('blended with neighbors');
                objecttags = objecttags.join(', ');
                $('#objecttags').val(objecttags);
            }

        });

        // alt+shift+0: weird object
        Mousetrap.bind(['alt+shift+0','command+shift+0'], function () {

            // get the current val for the objecttags
            var objecttags = $('#objecttags').val();

            // split by comma and strip extra spaces
            objecttags = objecttags.split(',');
            objecttags.forEach(function (item, index, arr) {
                objecttags[index] = item.trim();
            });

            // remove any item with zero length
            objecttags = objecttags
                .filter(function(val) { return val.length > 0 });

            // check if we already have this objecttag in the list and append it
            // if we don't.
            if (objecttags.indexOf('weird object') == -1) {
                objecttags.push('weird object');
                objecttags = objecttags.join(', ');
                $('#objecttags').val(objecttags);
            }

        });


    }

};


var cptools = {

    // allobjects contains objects of the form:
    // '<objectid>': {various lctool results}
    // this is indexed by objectid
    allobjects: {},

    // processing contains objects of the form:
    // '<objectid>': {'lctool':name of lctool currently running}
    running: {},

    // failed contains objects of the form:
    // '<lctool-objectid>': message from backend explaining what happened
    failed: {},

    // this holds the current checkplot's results for fast review
    current: {},


    // this loads the current checkplot's results in order of priority:
    // - from cptools.allresults.objects if objectid is present
    // - from the checkplot-<objectid>.pkl-cpserver-temp file if that is present
    // if neither are present, then this object doesn't have any results yet

    // checks the processing list in the allresults for the objectid requested.
    // if it's found there, then shows the spinner with the current tool
    // that's running. if it's found in the objects
    load_results: function () {

    },

    // this fires the POST action to save all outstanding stuff to the
    // checkplot pickle permanently
    sync_results: function (saveelem) {

    },

    // this clears out the currentresults and the allresults entry for the
    // current checkplot and deletes the -cpserver-temp pickle file.
    reset_results: function () {

    },

    run_periodsearch: function () {

        // get the current objectid and checkplot filename
        var currobjectid = $('#objectid').text();
        var currfname = $('#objectid').attr('data-fname');

        // this tracks if we're ok to proceed
        var proceed = false;

        // this tracks any errors in input. we join them all with <br>
        // and show them in the alert box.
        var messages = [];

        // get which period search to run
        var lspmethod = $('#psearch-lspmethod').val();

        // this is the lctool to call
        var lctooltocall = 'psearch-' + lspmethod;

        // see if magsarefluxes is checked
        var magsarefluxes = $('#psearch-magsarefluxes').prop('checked');

        // see if autofreq is checked
        var autofreq = $('#psearch-autofreq').prop('checked');

        if ((lspmethod == 'gls') || (lspmethod == 'bls') ||
            (lspmethod == 'pdm') || (lspmethod == 'aov')) {
            proceed = true;
        }

        else {
            messages.push("unknown periodogram method specified");
        }

        // if it's not checked, get the startp, endp, and frequency step
        if (!autofreq) {

            var startp = $('#psearch-startp').val();

            var endp = $('#psearch-endp').val();
            var freqstep = $('#psearch-freqstep').val();

            startp = parseFloat(startp);
            endp = parseFloat(endp);
            freqstep = parseFloat(freqstep);

            // check startp
            if (isNaN(startp)) {
                messages.push("no start period provided");
                proceed = false;
            }
            else if ((startp == 0.0) || (startp < 0.0)) {
                messages.push("start period cannot be 0.0 or < 0.0");
                proceed = false;
            }
            else if (startp > endp) {
                messages.push("start period cannot be larger than end period");
                proceed = false;
            }
            else {
                messages.push("using start period: " + math.format(startp, 5));
                proceed = true;
            }

            // check endp
            if (isNaN(endp)) {
                messages.push("no end period provided");
                proceed = false;
            }
            else if ((endp == 0.0) || (endp < 0.0)) {
                messages.push("end period cannot be 0.0 or < 0.0");
                proceed = false;
            }
            else {
                messages.push("using end period: " + math.format(endp, 5));
                proceed = true;
            }

            // check freqstep
            if (isNaN(freqstep)) {
                messages.push("no frequency step provided");
                proceed = false;
            }
            else if ((freqstep == 0.0) || (freqstep < 0.0)) {
                messages.push("frequency step cannot be 0.0 or < 0.0");
                proceed = false;
            }
            else {
                messages.push("using frequency step: " +
                              math.format(freqstep, 8));
                proceed = true;
            }

        }

        // proceed if we can
        if (proceed) {

            // generate the alert box messages
            messages.push("running " + lspmethod.toUpperCase() + '...');
            messages = messages.join("<br>");
            cpv.make_spinner('<span class="text-primary">' +
                             messages +
                             '</span><br>');

            // generate the tool queue box
            var tqbox = '<div class="tq-box" data-objectid="' + currobjectid +
                '" data-fname="' + currfname +
                '" data-lctool="' + lctooltocall +
                '" data-toolstate="running">' +
                '<strong>' + currobjectid + '</strong><br>' +
                lctooltocall + ' &mdash; ' +
                '<span class="tq-state">running</span></div>';

            $('.tool-queue').append(tqbox);

            // the call to the backend
            var ajaxurl = '/tools/' + cputils.b64_encode(cpv.currfile);

            if (autofreq) {

                // this is the data object to send to the backend
                var sentdata = {
                    objectid: currobjectid,
                    lctool: lctooltocall,
                    forcereload: true,
                    magsarefluxes: magsarefluxes,
                    autofreq: autofreq
                };


            }

            else {

                // this is the data object to send to the backend
                var sentdata = {
                    objectid: currobjectid,
                    lctool: lctooltocall,
                    forcereload: true,
                    magsarefluxes: magsarefluxes,
                    autofreq: autofreq,
                    startp: startp,
                    endp: endp,
                    stepsize: freqstep
                };


            }

            // make the call
            $.getJSON(ajaxurl, sentdata, function (recvdata) {

                // the received data is in the standard form
                var reqstatus = recvdata.status;
                var reqmsg = recvdata.message;
                var reqresult = recvdata.result;
                var cpobjectid = reqresult.objectid;

                // update this after we get back from the AJAX call
                var currobjectid = $('#objectid').text();

                if (reqstatus == 'success' || reqstatus == 'warning') {

                    var lsp = reqresult[lspmethod];

                    // only update if the user is still on the object
                    // we started with
                    if (cpobjectid == currobjectid) {

                        // update the select control for the periodogram peaks
                        // with the best three peaks
                        $('#psearch-pgrampeaks').empty();
                        $('#psearch-pgrampeaks').append(
                            '<option value="0|' + lsp.phasedlc0.period +
                                '|' + lsp.phasedlc0.epoch +
                                '" selected>peak 1: ' +
                                math.format(lsp.phasedlc0.period, 7) +
                                '</option>'
                        );
                        $('#psearch-pgrampeaks').append(
                            '<option value="1|' + lsp.phasedlc1.period +
                                '|' + lsp.phasedlc1.epoch +
                                '">peak 2: ' +
                                math.format(lsp.phasedlc1.period, 7) +
                                '</option>'
                        );
                        $('#psearch-pgrampeaks').append(
                            '<option value="2|' + lsp.phasedlc2.period +
                                '|' + lsp.phasedlc2.epoch +
                                '">peak 3: ' +
                                math.format(lsp.phasedlc2.period, 7) +
                                '</option>'
                        );

                        // update the period box with the best period
                        $('#psearch-plotperiod').val(lsp.phasedlc0.period);

                        // update the epoch box with the epoch of the best period
                        $('#psearch-plotepoch').val(lsp.phasedlc0.epoch);

                        // put the periodogram into #psearch-periodogram-display
                        cputils.b64_to_image(lsp.periodogram,
                                             '#psearch-periodogram-display');

                        // put the phased LC for the best period into
                        // #psearch-phasedlc-display
                        cputils.b64_to_image(lsp.phasedlc0.plot,
                                             '#psearch-phasedlc-display');

                        // update the cptools tracker with the results for this
                        // object

                        // if this objectid is in the tracker
                        if (cpobjectid in cptools.allobjects) {

                            cptools.allobjects[cpobjectid][lspmethod] = lsp;

                        }

                        // if it's not in the tracker, add it
                        else {

                            cptools.allobjects[cpobjectid] = reqresult;
                        }

                        // update the cptools currentresults with these results
                        cptools.current[lspmethod] = lsp;

                        // add a warning to the alert box if there was one
                        if (reqstatus == 'warning') {

                            // show the error if something exploded
                            // but only if we're on the right objectid.
                            var errmsg = '<span class="text-warning">' +
                                reqmsg  + '</span>';
                            $('#alert-box').html(errmsg);

                        }

                        // update the tool queue
                        // fade out and remove the matching entry
                        var tqsel = '[data-objectid="' + cpobjectid + '"]' +
                            '[data-lctool="' + lctooltocall + '"]';
                        var tqboxelem = $('.tq-box').filter(tqsel);
                        tqboxelem.children('span')
                            .html('<span class="text-primary">DONE<span>');
                        tqboxelem.fadeOut(2000, function () {
                            $(this).remove();
                        });

                        // clear out the alert box only if we're on the same
                        // objectid
                        $('#alert-box').empty();

                    }

                    // if the user has moved on,
                    // update the tool queue and allresults tracker
                    else {

                        console.log('results received for ' + cpobjectid +
                                    ' but user has moved to ' + currobjectid +
                                    ', updated tracker...');

                        // if this objectid is in the tracker
                        if (cpobjectid in cptools.allobjects) {

                            cptools.allobjects[cpobjectid][lspmethod] = lsp;

                        }

                        // if it's not in the tracker, add it
                        else {

                            cptools.allobjects[cpobjectid] = reqresult;
                        }

                        // update the tool queue
                        // fade out and remove the matching entry
                        var tqsel = '[data-objectid="' + cpobjectid + '"]' +
                            '[data-lctool="' + lctooltocall + '"]';
                        var tqboxelem = $('.tq-box').filter(tqsel);
                        tqboxelem.children('span')
                            .html('<span class="text-primary">DONE<span>');
                        tqboxelem.fadeOut(2000, function () {
                            $(this).remove();
                        });

                    }

                }

                // if the request failed
                else {


                    // show the error if something exploded but only if we're on
                    // the right objectid.
                    if (cpobjectid == currobjectid) {

                        var errmsg = '<span class="text-danger">' +
                            reqmsg  + '</span>';
                        $('#alert-box').html(errmsg);

                    }

                    // update the tool queue to show what happened
                    var tqsel = '[data-objectid="' + cpobjectid + '"]' +
                        '[data-lctool="' + lctooltocall + '"]';
                    var tqboxelem = $('.tq-box').filter(tqsel);
                    tqboxelem.children('span')
                        .html('<span class="text-danger">FAILED<span>');
                    tqboxelem.fadeOut(2500, function () {
                        $(this).remove();
                    });

                    // update the cptools.failed tracker
                    var failedkey = lctooltocall + '-' + cpobjectid;
                    cptools.failed[failedkey] = reqmsg;

                }


            }).fail(function (xhr) {

                // show the error - here we don't know which objectid was
                // returned, so show the error wherever we are
                var reqmsg = "could not run " + lctooltocall +
                    " because of a server error!";
                var errmsg = '<span class="text-danger">' +
                    reqmsg  + '</span>';
                $('#alert-box').html(errmsg);


            });


        }

        // otherwise we can't proceed, bail out and show the error messages
        else {

            // generate the alert box messages
            messages = messages.join("<br>");
            $('#alert-box').html('<span class="text-warning">' +
                                 messages +
                                 '</span>');


        }

    },

    get_varfeatures: function () {

        // get the current objectid and checkplot filename
        var currobjectid = $('#objectid').text();
        var currfname = $('#objectid').attr('data-fname');

        // this is the lctool to call
        var lctooltocall = 'var-varfeatures';

        // this tracks any errors in input. we join them all with <br>
        // and show them in the alert box.
        var messages = [];

        // generate the alert box messages
        messages.push("getting variability features...");
        messages = messages.join("<br>");
        cpv.make_spinner('<span class="text-primary">' +
                         messages +
                         '</span><br>');

        // generate the tool queue box
        var tqbox = '<div class="tq-box" data-objectid="' + currobjectid +
            '" data-fname="' + currfname +
            '" data-lctool="' + lctooltocall +
            '" data-toolstate="running">' +
            '<strong>' + currobjectid + '</strong><br>' +
            lctooltocall + ' &mdash; ' +
            '<span class="tq-state">running</span></div>';

        $('.tool-queue').append(tqbox);

        // the call to the backend
        var ajaxurl = '/tools/' + cputils.b64_encode(cpv.currfile);

        // this is the data object to send to the backend
        var sentdata = {
            objectid: currobjectid,
            lctool: lctooltocall,
            forcereload: true,
        };

        // make the call
        $.getJSON(ajaxurl, sentdata, function (recvdata) {

            // the received data is in the standard form
            var reqstatus = recvdata.status;
            var reqmsg = recvdata.message;
            var reqresult = recvdata.result;
            var cpobjectid = reqresult.objectid;

            // update this after we get back from the AJAX call
            var currobjectid = $('#objectid').text();

            if (reqstatus == 'success' || reqstatus == 'warning') {

                // only update if the user is still on the object
                // we started with
                if (cpobjectid == currobjectid) {

                    // this is the target element where we'll make a table
                    var featreselem = $('#var-varfeatures-results');

                    // these are the results
                    var vfeatures = reqresult.varinfo.varfeatures;

                    var feat_table = [
                        '<textarea class="form-control" rows="30" readonly>'
                    ];

                    feat_table.push(JSON.stringify(vfeatures, null, 2));

                    feat_table = feat_table.join('') + '</textarea>';
                    featreselem.html(feat_table);

                    // update the cptools tracker with the results for this
                    // object

                    // if this objectid is in the tracker
                    if (cpobjectid in cptools.allobjects) {

                        if ('varinfo' in cptools.allobjects[cpobjectid]) {

                            cptools.allobjects[
                                cpobjectid
                            ]['varinfo']['varfeatures'] = vfeatures;

                        }

                        else {

                            cptools.allobjects[cpobjectid] = {
                                varinfo:{
                                    varfeatures: vfeatures
                                }
                            };
                        }

                    }

                    // if it's not in the tracker, add it
                    else {

                        cptools.allobjects[cpobjectid] = reqresult;
                    }

                    // update the cptools currentresults with these results
                    if ('varinfo' in cptools.current) {
                        cptools.current['varinfo']['varfeatures'] = vfeatures;
                    }
                    else {
                        cptools.current['varinfo'] = {varfeatures: vfeatures};
                    }

                    // add a warning to the alert box if there was one
                    if (reqstatus == 'warning') {

                        // show the error if something exploded
                        // but only if we're on the right objectid.
                        var errmsg = '<span class="text-warning">' +
                            reqmsg  + '</span>';
                        $('#alert-box').html(errmsg);

                    }

                    // update the tool queue
                    // fade out and remove the matching entry
                    var tqsel = '[data-objectid="' + cpobjectid + '"]' +
                        '[data-lctool="' + lctooltocall + '"]';
                    var tqboxelem = $('.tq-box').filter(tqsel);
                    tqboxelem.children('span')
                        .html('<span class="text-primary">DONE<span>');
                    tqboxelem.fadeOut(2000, function () {
                        $(this).remove();
                    });

                    // clear out the alert box only if we're on the same
                    // objectid
                    $('#alert-box').empty();


                }

                // if the user has moved on...
                else {

                    console.log('results received for ' + cpobjectid +
                                ' but user has moved to ' + currobjectid +
                                ', updated tracker...');

                    // if this objectid is in the tracker
                    if (cpobjectid in cptools.allobjects) {

                        if ('varinfo' in cptools.allobjects[cpobjectid]) {

                            cptools.allobjects[
                                cpobjectid
                            ]['varinfo']['varfeatures'] = vfeatures;

                        }

                        else {

                            cptools.allobjects[cpobjectid] = {
                                varinfo:{
                                    varfeatures: vfeatures
                                }
                            };
                        }

                    }

                    // if it's not in the tracker, add it
                    else {

                        cptools.allobjects[cpobjectid] = reqresult;
                    }

                    // update the tool queue
                    // fade out and remove the matching entry
                    var tqsel = '[data-objectid="' + cpobjectid + '"]' +
                        '[data-lctool="' + lctooltocall + '"]';
                    var tqboxelem = $('.tq-box').filter(tqsel);
                    tqboxelem.children('span')
                        .html('<span class="text-primary">DONE<span>');
                    tqboxelem.fadeOut(2000, function () {
                        $(this).remove();
                    });


                }

            }

            // if the request failed
            else {

                // show the error if something exploded but only if we're on
                // the right objectid.
                if (cpobjectid == currobjectid) {

                    var errmsg = '<span class="text-danger">' +
                        reqmsg  + '</span>';
                    $('#alert-box').html(errmsg);

                }

                // update the tool queue to show what happened
                var tqsel = '[data-objectid="' + cpobjectid + '"]' +
                    '[data-lctool="' + lctooltocall + '"]';
                var tqboxelem = $('.tq-box').filter(tqsel);
                tqboxelem.children('span')
                    .html('<span class="text-danger">FAILED<span>');
                tqboxelem.fadeOut(2500, function () {
                    $(this).remove();
                });

                // update the cptools.failed tracker
                var failedkey = lctooltocall + '-' + cpobjectid;
                cptools.failed[failedkey] = reqmsg;

            }

        }).fail(function (xhr) {

            // show the error - here we don't know which objectid was
            // returned, so show the error wherever we are
            var reqmsg = "could not run  " + lctooltocall +
                " because of a server error!";
            var errmsg = '<span class="text-danger">' +
                reqmsg  + '</span>';
            $('#alert-box').html(errmsg);
        });


    },


    new_phasedlc_plot: function () {

        // get the current objectid and checkplot filename
        var currobjectid = $('#objectid').text();
        var currfname = $('#objectid').attr('data-fname');

        // this is the lctool to call
        var lctooltocall = 'phasedlc-newplot';

        // this tracks any errors in input. we join them all with <br>
        // and show them in the alert box.
        var messages = [];

        // this tracks if we're ok to proceed
        var proceed = true;

        // collect the form values
        var plotperiod = parseFloat($('#psearch-plotperiod').val());
        var plotepoch = parseFloat($('#psearch-plotepoch').val());

        var plotxlim = $('#psearch-plotxlim').val();
        if (plotxlim != '') {

            plotxlim = plotxlim.replace(']','').replace('[','').split(', ');
            plotxlim = [parseFloat(plotxlim[0]), parseFloat(plotxlim[1])];

            // if the plot limits aren't the default, then we probably want
            // xliminsetmode on
            if ((plotxlim[1] - plotxlim[0]) > 1.0) {
                var xliminsetmode = false;
            }
            else {
                var xliminsetmode = true;
            }
        }

        else {
            var xliminsetmode = false;
        }

        var phasebin = parseFloat($('#psearch-binphase').val());

        var periodind = $('#psearch-pgrampeaks').val();
        if (periodind === null) {
            periodind = 0;
        }
        else {
            periodind = parseInt(periodind.split('|')[0]);
        }

        var lspmethod = $('#psearch-lspmethod').val();

        var magsarefluxes = $('#psearch-magsarefluxes').prop('checked');

        // see if we can proceed

        if ((isNaN(plotperiod)) || (plotperiod < 0.0)) {
            messages.push("plot period is invalid")
            proceed = false;
        }

        if ((isNaN(plotepoch)) || (plotepoch < 0.0)) {
            messages.push("plot epoch is invalid")
            proceed = false;
        }

        if ((isNaN(plotxlim[0])) ||
            (plotxlim[0] < -1.0) ||
            (plotxlim[0] > plotxlim[1])) {
            messages.push("plot x-axis lower limit is invalid")
            proceed = false;
        }

        if ((isNaN(plotxlim[1])) ||
            (plotxlim[1] > 1.0) ||
            (plotxlim[1] < plotxlim[0])) {
            messages.push("plot x-axis upper limit is invalid")
            proceed = false;
        }

        if ((isNaN(phasebin)) ||
            (phasebin > 0.25) ||
            (phasebin < 0.0)) {
            messages.push("plot phase bin size is invalid")
            proceed = false;
        }


        // don't go any further if the input is valid
        if (!proceed) {

            // generate the alert box messages
            messages = messages.join("<br>");
            $('#alert-box').html('<span class="text-warning">' +
                                 messages +
                                 '</span>');

        }

        // otherwise, we're good to go
        else {

            // generate the alert box messages
            messages.push("making phased LC plot...");
            messages = messages.join("<br>");
            cpv.make_spinner('<span class="text-primary">' +
                             messages +
                             '</span><br>');

            // generate the tool queue box
            var tqbox = '<div class="tq-box" data-objectid="' + currobjectid +
                '" data-fname="' + currfname +
                '" data-lctool="' + lctooltocall +
                '" data-toolstate="running">' +
                '<strong>' + currobjectid + '</strong><br>' +
                lctooltocall + ' &mdash; ' +
                '<span class="tq-state">running</span></div>';

            $('.tool-queue').append(tqbox);

            // the call to the backend
            var ajaxurl = '/tools/' + cputils.b64_encode(cpv.currfile);

            // this is the data object to send to the backend
            var sentdata = {
                // common stuff
                objectid: currobjectid,
                lctool: lctooltocall,
                forcereload: true,
                // request values
                magsarefluxes: magsarefluxes,
                lspmethod: lspmethod,
                periodind: periodind,
                varperiod: plotperiod,
                varepoch: plotepoch,
                xliminsetmode: xliminsetmode,
                plotxlim: plotxlim,
                phasebin: phasebin
            };

            console.log(sentdata);

            // make the call
            $.getJSON(ajaxurl, sentdata, function (recvdata) {

                // the received data is in the standard form
                var reqstatus = recvdata.status;
                var reqmsg = recvdata.message;
                var reqresult = recvdata.result;
                var cpobjectid = reqresult.objectid;

                // update this after we get back from the AJAX call
                var currobjectid = $('#objectid').text();

                if (reqstatus == 'success' || reqstatus == 'warning') {

                    var lsp = reqresult[lspmethod];

                    // only update if the user is still on the object
                    // we started with
                    if (cpobjectid == currobjectid) {

                        var lckey = 'phasedlc' + periodind;

                        // put the phased LC for the best period into
                        // #psearch-phasedlc-display
                        cputils.b64_to_image(lsp[lckey]['plot'],
                                             '#psearch-phasedlc-display');

                        // update the global object period and epoch with the
                        // period and epoch used here if told to do so
                        var pupdate = $('#psearch-updateperiod').prop('checked');
                        var eupdate = $('#psearch-updateepoch').prop('checked');

                        if (pupdate) {
                            $('#objectperiod').val(plotperiod);
                        }
                        if (eupdate) {
                            $('#objectepoch').val(plotepoch);
                        }

                        // update current cptools object
                        if (lspmethod in cptools.current) {

                            cptools.current[lspmethod][lckey] = lsp[lckey];

                        }

                        else {
                            cptools.current[lspmethod] = {lckey: lsp[lckey]};
                        }


                        // update cptools tracker
                        if (cpobjectid in cptools.allobjects) {

                            if (lspmethod in cptools.allobjects[cpobjectid]) {

                                cptools.allobjects[
                                    cpobjectid
                                ][lspmethod][lckey] = lsp[lckey];

                            }

                            else {
                                cptools.allobjects[
                                    cpobjectid
                                ][lspmethod] = {lckey: lsp[lckey]};
                            }

                        }

                        else {

                            cptools.allobjects[cpobjectid] = reqresult;

                        }


                        // add a warning to the alert box if there was one
                        if (reqstatus == 'warning') {

                            // show the error if something exploded
                            // but only if we're on the right objectid.
                            var errmsg = '<span class="text-warning">' +
                                reqmsg  + '</span>';
                            $('#alert-box').html(errmsg);

                        }

                        // update the tool queue
                        // fade out and remove the matching entry
                        var tqsel = '[data-objectid="' + cpobjectid + '"]' +
                            '[data-lctool="' + lctooltocall + '"]';
                        var tqboxelem = $('.tq-box').filter(tqsel);
                        tqboxelem.children('span')
                            .html('<span class="text-primary">DONE<span>');
                        tqboxelem.fadeOut(2000, function () {
                            $(this).remove();
                        });

                        // clear out the alert box only if we're on the same
                        // objectid
                        $('#alert-box').empty();

                    }

                    // if the user has moved on...
                    else {

                        console.log('results received for ' + cpobjectid +
                                    ' but user has moved to ' + currobjectid +
                                    ', updated tracker...');

                        // update cptools tracker
                        if (cpobjectid in cptools.allobjects) {

                            if (lspmethod in cptools.allobjects[cpobjectid]) {

                                cptools.allobjects[
                                    cpobjectid
                                ][lspmethod][lckey] = lsp[lckey];

                            }

                            else {
                                cptools.allobjects[
                                    cpobjectid
                                ][lspmethod] = {lckey: lsp[lckey]};
                            }

                        }

                        else {

                            cptools.allobjects[cpobjectid] = reqresult;

                        }

                        // update the tool queue
                        // fade out and remove the matching entry
                        var tqsel = '[data-objectid="' + cpobjectid + '"]' +
                            '[data-lctool="' + lctooltocall + '"]';
                        var tqboxelem = $('.tq-box').filter(tqsel);
                        tqboxelem.children('span')
                            .html('<span class="text-primary">DONE<span>');
                        tqboxelem.fadeOut(2000, function () {
                            $(this).remove();
                        });

                    }

                }

                // if the request failed
                else {

                    // show the error if something exploded but only if we're on
                    // the right objectid.
                    if (cpobjectid == currobjectid) {

                        var errmsg = '<span class="text-danger">' +
                            reqmsg  + '</span>';
                        $('#alert-box').html(errmsg);

                    }

                    // update the tool queue to show what happened
                    var tqsel = '[data-objectid="' + cpobjectid + '"]' +
                        '[data-lctool="' + lctooltocall + '"]';
                    var tqboxelem = $('.tq-box').filter(tqsel);
                    tqboxelem.children('span')
                        .html('<span class="text-danger">FAILED<span>');
                    tqboxelem.fadeOut(2500, function () {
                        $(this).remove();
                    });

                    // update the cptools.failed tracker
                    var failedkey = lctooltocall + '-' + cpobjectid;
                    cptools.failed[failedkey] = reqmsg;

                }

            }).fail(function (xhr) {

                // show the error - here we don't know which objectid was
                // returned, so show the error wherever we are
                var reqmsg = "could not run  " + lctooltocall +
                    " because of a server error!";
                var errmsg = '<span class="text-danger">' +
                    reqmsg  + '</span>';
                $('#alert-box').html(errmsg);
            });


        }



    },


    prewhiten_lightcurve: function () {

        // get the current objectid and checkplot filename
        var currobjectid = $('#objectid').text();
        var currfname = $('#objectid').attr('data-fname');

        // this is the lctool to call
        var lctooltocall = 'var-prewhiten';

        // this tracks any errors in input. we join them all with <br>
        // and show them in the alert box.
        var messages = [];

        // generate the alert box messages
        messages.push("getting variability features...");
        messages = messages.join("<br>");
        cpv.make_spinner('<span class="text-primary">' +
                         messages +
                         '</span><br>');

        // generate the tool queue box
        var tqbox = '<div class="tq-box" data-objectid="' + currobjectid +
            '" data-fname="' + currfname +
            '" data-lctool="' + lctooltocall +
            '" data-toolstate="running">' +
            '<strong>' + currobjectid + '</strong><br>' +
            lctooltocall + ' &mdash; ' +
            '<span class="tq-state">running</span></div>';

        $('.tool-queue').append(tqbox);

        // the call to the backend
        var ajaxurl = '/tools/' + cputils.b64_encode(cpv.currfile);

        // this is the data object to send to the backend
        var sentdata = {
            objectid: currobjectid,
            lctool: lctooltocall,
            forcereload: true,
        };

        // make the call
        $.getJSON(ajaxurl, sentdata, function (recvdata) {

            // the received data is in the standard form
            var reqstatus = recvdata.status;
            var reqmsg = recvdata.message;
            var reqresult = recvdata.result;
            var cpobjectid = reqresult.objectid;

            // update this after we get back from the AJAX call
            var currobjectid = $('#objectid').text();

            if (reqstatus == 'success' || reqstatus == 'warning') {

                // only update if the user is still on the object
                // we started with
                if (cpobjectid == currobjectid) {

                }

                // if the user has moved on...
                else {

                }

            }

            // if the request failed
            else {

                // show the error if something exploded but only if we're on
                // the right objectid.
                if (cpobjectid == currobjectid) {

                    var errmsg = '<span class="text-danger">' +
                        reqmsg  + '</span>';
                    $('#alert-box').html(errmsg);

                }

                // update the tool queue to show what happened
                var tqsel = '[data-objectid="' + cpobjectid + '"]' +
                    '[data-lctool="' + lctooltocall + '"]';
                var tqboxelem = $('.tq-box').filter(tqsel);
                tqboxelem.children('span')
                    .html('<span class="text-danger">FAILED<span>');
                tqboxelem.fadeOut(2500, function () {
                    $(this).remove();
                });

                // update the cptools.failed tracker
                var failedkey = lctooltocall + '-' + cpobjectid;
                cptools.failed[failedkey] = reqmsg;

            }

        }).fail(function (xhr) {

            // show the error - here we don't know which objectid was
            // returned, so show the error wherever we are
            var reqmsg = "could not run  " + lctooltocall +
                " because of a server error!";
            var errmsg = '<span class="text-danger">' +
                reqmsg  + '</span>';
            $('#alert-box').html(errmsg);
        });

    },


    mask_signal: function () {

        // get the current objectid and checkplot filename
        var currobjectid = $('#objectid').text();
        var currfname = $('#objectid').attr('data-fname');

        // this is the lctool to call
        var lctooltocall = 'var-masksig';

        // this tracks any errors in input. we join them all with <br>
        // and show them in the alert box.
        var messages = [];

        // generate the alert box messages
        messages.push("getting variability features...");
        messages = messages.join("<br>");
        cpv.make_spinner('<span class="text-primary">' +
                         messages +
                         '</span><br>');

        // generate the tool queue box
        var tqbox = '<div class="tq-box" data-objectid="' + currobjectid +
            '" data-fname="' + currfname +
            '" data-lctool="' + lctooltocall +
            '" data-toolstate="running">' +
            '<strong>' + currobjectid + '</strong><br>' +
            lctooltocall + ' &mdash; ' +
            '<span class="tq-state">running</span></div>';

        $('.tool-queue').append(tqbox);

        // the call to the backend
        var ajaxurl = '/tools/' + cputils.b64_encode(cpv.currfile);

        // this is the data object to send to the backend
        var sentdata = {
            objectid: currobjectid,
            lctool: lctooltocall,
            forcereload: true,
        };

        // make the call
        $.getJSON(ajaxurl, sentdata, function (recvdata) {

            // the received data is in the standard form
            var reqstatus = recvdata.status;
            var reqmsg = recvdata.message;
            var reqresult = recvdata.result;
            var cpobjectid = reqresult.objectid;

            // update this after we get back from the AJAX call
            var currobjectid = $('#objectid').text();

            if (reqstatus == 'success' || reqstatus == 'warning') {

                // only update if the user is still on the object
                // we started with
                if (cpobjectid == currobjectid) {

                }

                // if the user has moved on...
                else {

                }

            }

            // if the request failed
            else {

                // show the error if something exploded but only if we're on
                // the right objectid.
                if (cpobjectid == currobjectid) {

                    var errmsg = '<span class="text-danger">' +
                        reqmsg  + '</span>';
                    $('#alert-box').html(errmsg);

                }

                // update the tool queue to show what happened
                var tqsel = '[data-objectid="' + cpobjectid + '"]' +
                    '[data-lctool="' + lctooltocall + '"]';
                var tqboxelem = $('.tq-box').filter(tqsel);
                tqboxelem.children('span')
                    .html('<span class="text-danger">FAILED<span>');
                tqboxelem.fadeOut(2500, function () {
                    $(this).remove();
                });

                // update the cptools.failed tracker
                var failedkey = lctooltocall + '-' + cpobjectid;
                cptools.failed[failedkey] = reqmsg;

            }

        }).fail(function (xhr) {

            // show the error - here we don't know which objectid was
            // returned, so show the error wherever we are
            var reqmsg = "could not run  " + lctooltocall +
                " because of a server error!";
            var errmsg = '<span class="text-danger">' +
                reqmsg  + '</span>';
            $('#alert-box').html(errmsg);
        });


    },


    lcfit_magseries: function () {

        // get the current objectid and checkplot filename
        var currobjectid = $('#objectid').text();
        var currfname = $('#objectid').attr('data-fname');

        // get which period search to run
        var fitmethod = $('#lcfit-fitmethod').val();

        // this is the lctool to call
        var lctooltocall = 'lcfit-' + fitmethod;

        // this tracks any errors in input. we join them all with <br>
        // and show them in the alert box.
        var messages = [];

        // this tracks if we're ok to proceed
        var proceed = false;

        // generate the alert box messages
        messages.push("getting variability features...");
        messages = messages.join("<br>");
        cpv.make_spinner('<span class="text-primary">' +
                         messages +
                         '</span><br>');

        // generate the tool queue box
        var tqbox = '<div class="tq-box" data-objectid="' + currobjectid +
            '" data-fname="' + currfname +
            '" data-lctool="' + lctooltocall +
            '" data-toolstate="running">' +
            '<strong>' + currobjectid + '</strong><br>' +
            lctooltocall + ' &mdash; ' +
            '<span class="tq-state">running</span></div>';

        $('.tool-queue').append(tqbox);

        // the call to the backend
        var ajaxurl = '/tools/' + cputils.b64_encode(cpv.currfile);

        // this is the data object to send to the backend
        var sentdata = {
            objectid: currobjectid,
            lctool: lctooltocall,
            forcereload: true,
        };

        // make the call
        $.getJSON(ajaxurl, sentdata, function (recvdata) {

            // the received data is in the standard form
            var reqstatus = recvdata.status;
            var reqmsg = recvdata.message;
            var reqresult = recvdata.result;
            var cpobjectid = reqresult.objectid;

            // update this after we get back from the AJAX call
            var currobjectid = $('#objectid').text();

            if (reqstatus == 'success' || reqstatus == 'warning') {

                // only update if the user is still on the object
                // we started with
                if (cpobjectid == currobjectid) {

                }

                // if the user has moved on...
                else {

                }

            }

            // if the request failed
            else {

                // show the error if something exploded but only if we're on
                // the right objectid.
                if (cpobjectid == currobjectid) {

                    var errmsg = '<span class="text-danger">' +
                        reqmsg  + '</span>';
                    $('#alert-box').html(errmsg);

                }

                // update the tool queue to show what happened
                var tqsel = '[data-objectid="' + cpobjectid + '"]' +
                    '[data-lctool="' + lctooltocall + '"]';
                var tqboxelem = $('.tq-box').filter(tqsel);
                tqboxelem.children('span')
                    .html('<span class="text-danger">FAILED<span>');
                tqboxelem.fadeOut(2500, function () {
                    $(this).remove();
                });

                // update the cptools.failed tracker
                var failedkey = lctooltocall + '-' + cpobjectid;
                cptools.failed[failedkey] = reqmsg;

            }

        }).fail(function (xhr) {

            // show the error - here we don't know which objectid was
            // returned, so show the error wherever we are
            var reqmsg = "could not run  " + lctooltocall +
                " because of a server error!";
            var errmsg = '<span class="text-danger">' +
                reqmsg  + '</span>';
            $('#alert-box').html(errmsg);
        });

    },


    action_setup: function () {

        ///////////////////////
        // PERIOD SEARCH TAB //
        ///////////////////////

        // periodogram search - half period
        $('#psearch-halfperiod').on('click', function (evt) {

            evt.preventDefault();
            var plotperiod = parseFloat($('#psearch-plotperiod').val());
            if (!isNaN(plotperiod)) {
                $('#psearch-plotperiod').val(plotperiod/2.0);
            }

        });

        // periodogram search - 2x period
        $('#psearch-doubleperiod').on('click', function (evt) {

            evt.preventDefault();
            var plotperiod = parseFloat($('#psearch-plotperiod').val());
            if (!isNaN(plotperiod)) {
                $('#psearch-plotperiod').val(plotperiod*2.0);
            }

        });

        // periodogram search - plot phased LC
        $('#psearch-makephasedlc').on('click', function (evt) {

            evt.preventDefault();
            cptools.new_phasedlc_plot();

        });

        // periodogram search - periodogram peak select
        $('#psearch-pgrampeaks').on('change', function (evt) {

            var newval = $(this).val().split('|');

            $('#psearch-plotperiod').val(newval[1]);
            $('#psearch-plotepoch').val(newval[2]);

        });


        // periodogram search - start
        $('#psearch-start').on('click', function (evt) {

            evt.preventDefault();
            cptools.run_periodsearch();

        });

        // periodogram search - handle autofreq
        $('#psearch-autofreq').on('click', function (evt) {

            var thischecked = $(this).prop('checked');

            if (!thischecked) {

                $('#psearch-startp').prop('disabled',false);
                $('#psearch-endp').prop('disabled',false);
                $('#psearch-freqstep').prop('disabled',false);

            }

            else {

                $('#psearch-startp').prop('disabled',true);
                $('#psearch-endp').prop('disabled',true);
                $('#psearch-freqstep').prop('disabled',true);

            }

        });

        // FIXME: bind the periodogram select so it looks for existing periods
        // for that periodogram method and loads them into the period select
        // box.




        /////////////////////
        // VARIABILITY TAB //
        /////////////////////

        // variability - get variability features
        $('#get-varfeatures').on('click', function (evt) {

            evt.preventDefault();
            cptools.get_varfeatures();

        });


        ////////////////
        // LC FIT TAB //
        ////////////////




        ////////////////////
        // SAVE/RESET TAB //
        ////////////////////



    }

};
