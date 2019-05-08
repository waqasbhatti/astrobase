/*global $, jQuery, math */

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
    return window.btoa(
      encodeURIComponent(str)
        .replace(/%([0-9A-F]{2})/g,
                 function(match, p1) {
                   return String.fromCharCode('0x' + p1);
                 }));
  },

  // this decodes a string from base64
  b64_decode: function (str) {
    return decodeURIComponent(window.atob(str).split('').map(function(c) {
      return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
    }).join(''));

  },

  // https://stackoverflow.com/a/26601101
  b64_decode2: function (s) {

    var e={},i,b=0,c,x,l=0,a,r='',w=String.fromCharCode,L=s.length;
    var A="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for(i=0;i<64;i++){e[A.charAt(i)]=i;}
    for(x=0;x<L;x++){
      c=e[s.charAt(x)];b=(b<<6)+c;l+=6;
      while(l>=8){((a=(b>>>(l-=8))&0xff)||(x<(L-2)))&&(r+=w(a));}
    }
    return r;

  },


  // this turns a base64 string into an image by updating its source
  b64_to_image: function (str, targetelem) {

    var datauri = 'data:image/png;base64,' + str;
    $(targetelem).attr('src',datauri);

  },

  // this displays a base64 encoded image on the canvas
  b64_to_canvas: function (str, targetelem) {

    var datauri = 'data:image/png;base64,' + str;
    var newimg = new Image();
    var canvas = document.getElementById(targetelem.replace('#',''));

    var imgheight = 300;
    var imgwidth = 300;
    var cnvwidth = canvas.width;
    canvas.height = cnvwidth;
    var imgscale = cnvwidth/imgwidth;

    var ctx = canvas.getContext('2d');

    // this event listener will fire when the image is loaded
    newimg.addEventListener('load', function () {
      ctx.drawImage(newimg,
                    0,
                    0,
                    imgwidth*imgscale,
                    imgheight*imgscale);
    });

    // load the image and fire the listener
    newimg.src = datauri;

  },

  // this holds imagedata for the canvas so we can restore changed parts of
  // the image
  pixeltracker: null

};


// this contains updates to current checkplots, tracks the entire project's info
// and provides functions to save everything to CSV or JSON
var cptracker = {

  // this is a list of all checkplots
  checkplotlist: [],

  // this is the actual object that will get written to JSON or CSV
  cpdata: {},

  // this is the order of columns for generating the CSV
  infocolumns: [
    'objectid',
    'checkplot',
    'objectinfo.twomassid',
    'objectinfo.network',
    'objectinfo.stations',
    'objectinfo.ndet',
    'objectinfo.objecttags',
    'objectinfo.umag',
    'objectinfo.bmag',
    'objectinfo.vmag',
    'objectinfo.rmag',
    'objectinfo.imag',
    'objectinfo.sdssu',
    'objectinfo.sdssg',
    'objectinfo.sdssr',
    'objectinfo.sdssi',
    'objectinfo.sdssz',
    'objectinfo.jmag',
    'objectinfo.hmag',
    'objectinfo.kmag',
    'objectinfo.ujmag',
    'objectinfo.uhmag',
    'objectinfo.ukmag',
    'objectinfo.irac1',
    'objectinfo.irac2',
    'objectinfo.irac3',
    'objectinfo.irac4',
    'objectinfo.wise1',
    'objectinfo.wise2',
    'objectinfo.wise3',
    'objectinfo.wise4',
    'objectinfo.deredb',
    'objectinfo.deredv',
    'objectinfo.deredu',
    'objectinfo.deredg',
    'objectinfo.deredr',
    'objectinfo.deredi',
    'objectinfo.deredz',
    'objectinfo.deredj',
    'objectinfo.deredh',
    'objectinfo.deredk',
    'objectinfo.dered_umag',
    'objectinfo.dered_bmag',
    'objectinfo.dered_vmag',
    'objectinfo.dered_rmag',
    'objectinfo.dered_imag',
    'objectinfo.dered_jmag',
    'objectinfo.dered_hmag',
    'objectinfo.dered_kmag',
    'objectinfo.dered_sdssu',
    'objectinfo.dered_sdssg',
    'objectinfo.dered_sdssr',
    'objectinfo.dered_sdssi',
    'objectinfo.dered_sdssz',
    'objectinfo.dered_ujmag',
    'objectinfo.dered_uhmag',
    'objectinfo.dered_ukmag',
    'objectinfo.dered_irac1',
    'objectinfo.dered_irac2',
    'objectinfo.dered_irac3',
    'objectinfo.dered_irac4',
    'objectinfo.dered_wise1',
    'objectinfo.dered_wise2',
    'objectinfo.dered_wise3',
    'objectinfo.dered_wise4',
    'objectinfo.bvcolor',
    'objectinfo.gjcolor',
    'objectinfo.ijcolor',
    'objectinfo.jkcolor',
    'objectinfo.gkcolor',
    'objectinfo.vkcolor',
    'objectinfo.ugcolor',
    'objectinfo.grcolor',
    'objectinfo.ricolor',
    'objectinfo.izcolor',
    'objectinfo.sdssu-sdssg',
    'objectinfo.sdssg-sdssr',
    'objectinfo.sdssr-sdssi',
    'objectinfo.sdssi-sdssz',
    'objectinfo.sdssg-sdssi',
    'objectinfo.sdssg-jmag',
    'objectinfo.sdssg-kmag',
    'objectinfo.umag-bmag',
    'objectinfo.bmag-vmag',
    'objectinfo.vmag-rmag',
    'objectinfo.vmag-imag',
    'objectinfo.rmag-imag',
    'objectinfo.vmag-kmag',
    'objectinfo.jmag-hmag',
    'objectinfo.hmag-kmag',
    'objectinfo.jmag-kmag',
    'objectinfo.dereddened',
    'objectinfo.bmagfromjhk',
    'objectinfo.vmagfromjhk',
    'objectinfo.sdssufromjhk',
    'objectinfo.sdssgfromjhk',
    'objectinfo.sdssrfromjhk',
    'objectinfo.sdssifromjhk',
    'objectinfo.sdsszfromjhk',
    'objectinfo.color_classes',
    'objectinfo.ra',
    'objectinfo.decl',
    'objectinfo.gl',
    'objectinfo.gb',
    'objectinfo.pmra',
    'objectinfo.pmra_err',
    'objectinfo.pmdecl',
    'objectinfo.pmdecl_err',
    'objectinfo.propermotion',
    'objectinfo.rpmj',
    'objectinfo.reducedpropermotion',
    'objectinfo.gaiamag',
    'objectinfo.gaia_parallax',
    'objectinfo.gaia_parallax_err',
    'objectinfo.neighbors',
    'objectinfo.closestdistarcsec',
    'objectinfo.gaia_neighbors',
    'objectinfo.gaia_closest_distarcsec',
    'objectinfo.gaia_closest_gmagdiff',
    'objectinfo.simbad_best_mainid',
    'objectinfo.simbad_best_objtype',
    'objectinfo.simbad_best_allids',
    'objectinfo.simbad_best_distarcsec',
    'objectinfo.d_ug',
    'objectinfo.d_gr',
    'objectinfo.s_color',
    'objectinfo.l_color',
    'objectinfo.v_color',
    'objectinfo.p1_color',
    'objectinfo.m_sti',
    'objectinfo.m_sts',
    'objectinfo.extinctb',
    'objectinfo.extinctv',
    'objectinfo.extinctu',
    'objectinfo.extinctg',
    'objectinfo.extinctr',
    'objectinfo.extincti',
    'objectinfo.extinctz',
    'objectinfo.extinctj',
    'objectinfo.extincth',
    'objectinfo.extinctk',
    'objectinfo.extinction_umag',
    'objectinfo.extinction_bmag',
    'objectinfo.extinction_vmag',
    'objectinfo.extinction_rmag',
    'objectinfo.extinction_imag',
    'objectinfo.extinction_sdssu',
    'objectinfo.extinction_sdssg',
    'objectinfo.extinction_sdssr',
    'objectinfo.extinction_sdssi',
    'objectinfo.extinction_sdssz',
    'objectinfo.extinction_jmag',
    'objectinfo.extinction_hmag',
    'objectinfo.extinction_kmag',
    'varinfo.objectisvar',
    'varinfo.varperiod',
    'varinfo.varepoch',
    'varinfo.vartags',
    'comments'
  ],

  // this function generates a CSV row for a single object in the
  // cptracker.cpdata object
  cpdata_get_csvrow: function (cpdkey, separator) {

    let thisdata = cptracker.cpdata[cpdkey];
    let rowarr = [];

    // iterate through all the keys to extract for this object
    for (let item of cptracker.infocolumns) {

      let key = null;
      let val = null;

      // if this is a compound key
      if (item.indexOf('.') != -1) {

        key = item.split('.');
        val = thisdata[key[0]][key[1]];

        if (val !== undefined) {
          rowarr.push(val);
        }
        else {
          rowarr.push('');
        }

      }

      // otherwise it's a simple key
      else {

        key = item;

        // special case of checkplot filename
        if (item == 'checkplot') {
          val = cpdkey;
        }

        else {

          if (thisdata[key] !== undefined) {
            val = thisdata[key];
          }

          else {
            val = '';
          }

        }

        rowarr.push(val);

      }

    }

    // join the output row
    let csvrow = rowarr.join(separator);
    return csvrow;

  },

  // this generates a CSV for download
  cpdata_to_csv: function () {

    var csvarr = [cptracker.infocolumns.join('|')];
    var obj = null;

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
    var obj = null;

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
                    comments: cpv.currcp.objectcomments};

    var postmsg = {objectid: cpv.currcp.objectid,
                   changes: JSON.stringify(saveinfo)};


    if (cpv.readonlymode) {
      // if we're in readonly mode, inform the user
      $('#alert-box').html(
        'The checkplot server is in readonly mode. ' +
          'Edits to object information will not be saved.'
      );
    }

    // if we're not in readonly mode, post the results
    else {

      var target_url = cpv.CPSERVER_BASEURL + 'list';

      // send this back to the checkplotserver
      $.post(target_url, postmsg, function (data) {

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

    var target_url = cpv.CPSERVER_BASEURL + 'list';

    $.getJSON(target_url, function (data) {

      var reviewedobjects = data.reviewed;
      var obj = null;
      // generate the object info rows and append to the reviewed object
      // list
      for (obj in reviewedobjects) {

        // update the cptracker.cpdata object with reviewed object info
        // if it's not in there
        var thiscpx = reviewedobjects[obj]['checkplot'];
        if (!(thiscpx in cptracker.cpdata)) {
          cptracker.cpdata[thiscpx] = reviewedobjects[obj];
          cptracker.cpdata[thiscpx]['objectid'] = obj;
          cptracker.cpdata[thiscpx]['comments'] =
            reviewedobjects[obj]['comments'];
        }

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
  // always start in readonly mode for safety
  readonlymode: true,

  // these help in moving quickly through the phased LCs
  currphasedind: null,
  maxphasedind: null,

  // this holds the current status of the checkplot load
  loadsuccess: false,

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
    var plot_title_elem = $('#checkplot-current');
    var basename = filename.split('/');
    var filelink = '<strong>' + filename + '</strong>';

    // put in the object's names,file names, and links
    var objectidelem = $('#objectid');
    var twomassidelem = $('#twomassid');
    plot_title_elem.html(filelink);

    if (cpv.currfile.length > 0) {
      // un-highlight the previous file in side bar
      $("a.checkplot-load")
        .filter("[data-fname='" + cpv.currfile + "']")
        .unwrap();
    }

    // do the AJAX call to get this checkplot
    // make sure to make a new call every time
    console.log('loading checkplot ' + filename);
    var ajaxurl = cpv.CPSERVER_BASEURL + 'cp/' +
        encodeURIComponent(cputils.b64_encode(filename)) +
        '?t=' + Date.now();

    // add the current object's fname to the objectid data-fname
    objectidelem.attr('data-fname', filename);

    $.getJSON(ajaxurl, function (data) {

      cpv.currcp = data.result;

      /////////////////////////////////////////////////
      // update the UI with elems for this checkplot //
      /////////////////////////////////////////////////

      // update the readonly status
      cpv.readonlymode = data.readonly;

      if (cpv.currcp === null || data.status == 'error') {

        console.log(data.message);
        $('#alert-box').html(data.message);

        cpv.loadsuccess = false;

        return null;

      }

      else {
        cpv.loadsuccess = true;
      }

      // update the objectid header
      objectidelem.html(cpv.currcp.objectid);

      // update the twomassid header
      if (cpv.currcp.objectinfo.twomassid != undefined) {
        twomassidelem.html('2MASS J' + cpv.currcp.objectinfo.twomassid);
      }
      else {
        twomassidelem.html('');
      }


      //
      // first, update the period search tab
      //

      // update the time0 placeholders
      $('.time0').html(cpv.currcp.time0);

      // set the items in the filters from existing if possible
      if (cpv.currcp.hasOwnProperty('uifilters')) {

        if (cpv.currcp.uifilters != null &&
            cpv.currcp.uifilters.psearch_timefilters != null) {
          $('#psearch-timefilters').val(
            cpv.currcp.uifilters.psearch_timefilters
          );
        }
        else {
          $('#psearch-timefilters').val('');
        }

        if (cpv.currcp.uifilters != null &&
            cpv.currcp.uifilters.psearch_magfilters != null) {
          $('#psearch-magfilters').val(
            cpv.currcp.uifilters.psearch_magfilters
          );
        }
        else {
          $('#psearch-magfilters').val('');
        }

        if (cpv.currcp.uifilters != null &&
            cpv.currcp.uifilters.psearch_sigclip != null) {
          $('#psearch-sigclip').val(
            cpv.currcp.uifilters.psearch_sigclip
          );
        }
        else {
          $('#psearch-sigclip').val('');
        }

      }

      // otherwise, use default nothing
      else {
        $('#psearch-timefilters').val('');
        $('#psearch-magfilters').val('');
        $('#psearch-sigclip').val('');
      }

      //
      // set up the rest of period-search tab
      //

      // set the period and epoch from the current period and epoch
      if (cpv.currcp.varinfo.varperiod != null &&
          cpv.currcp.varinfo.varepoch != null) {

        $('#psearch-plotperiod').val(cpv.currcp.varinfo.varperiod);
        $('#psearch-plotepoch').val(cpv.currcp.varinfo.varepoch);
        $("#psearch-pgrampeaks").html(
          '<option value="0|' + cpv.currcp.varinfo.varperiod + '|' +
            cpv.currcp.varinfo.varepoch +
            '">prev. saved period</option>'
        );

      }

      else {
        $("#psearch-pgrampeaks").html(
          '<option value="none">no tool results yet</option>'
        );
        $('#psearch-plotperiod').val('');
        $('#psearch-plotepoch').val('');
      }

      // these are the plot frames, nothing by default
      $('#psearch-periodogram-display')
        .attr('src',
              cpv.CPSERVER_BASEURL + 'static/no-tool-results.png');
      $('#psearch-phasedlc-display')
        .attr('src',
              cpv.CPSERVER_BASEURL + 'static/no-tool-results.png');


      //
      // update object information now
      //

      // update the finder chart
      // cputils.b64_to_image(cpv.currcp.finderchart,
      //                      '#finderchart');
      cputils.b64_to_canvas(cpv.currcp.finderchart,
                            '#finderchart');



      // get the number of detections
      var objndet = cpv.currcp.objectinfo.ndet;

      if (objndet == undefined) {
        objndet = cpv.currcp.magseries_ndet;
      }


      // get the observatory information
      if ('stations' in cpv.currcp.objectinfo) {

        // get the HAT stations
        var hatstations = cpv.currcp.objectinfo.stations;
        var splitstations = '';

        if (hatstations != undefined && hatstations) {
          splitstations = (String(hatstations).split(',')).join(', ');
        }

        // update the objectinfo
        var hatinfo = '<strong>' +
            splitstations +
            '</strong><br>' +
            '<strong>LC points:</strong> ' + objndet;
        $('#hatinfo').html(hatinfo);

      }
      else if ('observatory' in cpv.currcp.objectinfo) {

        var obsinfo = '<strong'> +
            cpv.currcp.objectinfo.observatory + '</strong><br>' +
            '<strong>LC points:</strong> ' + objndet;
        $('#hatinfo').html(obsinfo);

      }
      else if ('telescope' in cpv.currcp.objectinfo) {

        var telinfo = '<strong'> +
            cpv.currcp.objectinfo.telescope + '</strong><br>' +
            '<strong>LC points:</strong> ' + objndet;
        $('#hatinfo').html(telinfo);

      }
      else {
        $('#hatinfo').html('<strong>LC points:</strong> ' + objndet);
      }

      // get the GAIA status (useful for G mags, colors, etc.)
      if (cpv.currcp.objectinfo.gaia_status != undefined) {
        var gaia_ok =
            cpv.currcp.objectinfo.gaia_status.indexOf('ok') != -1;
        var gaia_message =
            cpv.currcp.objectinfo.gaia_status.split(':')[1];
      }

      else {
        var gaia_ok = false;
        var gaia_message = (
          'no GAIA cross-match information available'
        );
        console.log('no GAIA info');
      }

      // get the SIMBAD status (useful for G mags, colors, etc.)
      if (cpv.currcp.objectinfo.simbad_status != undefined) {
        var simbad_ok =
            cpv.currcp.objectinfo.simbad_status.indexOf('ok') != -1;
        var simbad_message =
            cpv.currcp.objectinfo.simbad_status.split(':')[1];
      }

      else {
        var simbad_ok = false;
        var simbad_message = (
          'no SIMBAD cross-match information available'
        );
        console.log('no SIMBAD info');
      }


      //
      // get the coordinates
      //

      // ra
      if (cpv.currcp.objectinfo.ra != undefined) {
        var objectra = math.format(
          cpv.currcp.objectinfo.ra, 6
        );

      }
      else {
        var objectra = '';
      }
      // decl
      if (cpv.currcp.objectinfo.decl != undefined) {
        var objectdecl = math.format(
          cpv.currcp.objectinfo.decl, 6
        );

      }
      else {
        var objectdecl = '';
      }
      // gl
      if (cpv.currcp.objectinfo.gl != undefined) {
        var objectgl = math.format(
          cpv.currcp.objectinfo.gl, 6
        );

      }
      else {
        var objectgl = '';
      }
      // gb
      if (cpv.currcp.objectinfo.gb != undefined) {
        var objectgb = math.format(
          cpv.currcp.objectinfo.gb, 6
        );

      }
      else {
        var objectgb = '';
      }
      // total proper motion
      if (cpv.currcp.objectinfo.propermotion != undefined) {
        var objectpm = math.format(
          cpv.currcp.objectinfo.propermotion, 5
        ) + ' mas/yr';

        if ( (cpv.currcp.objectinfo.pmra_source != undefined) &&
             (cpv.currcp.objectinfo.pmdecl_source != undefined) ) {

          var pmra_source = cpv.currcp.objectinfo.pmra_source;
          var pmdecl_source = cpv.currcp.objectinfo.pmdecl_source;

          // note if the propermotion came from GAIA
          if ( (pmra_source == pmdecl_source) &&
               (pmra_source == 'gaia') ) {
            objectpm = objectpm + ' (from GAIA)';
          }

        }

      }
      else {
        var objectpm = '';
      }

      // reduced proper motion [Jmag]
      if (cpv.currcp.objectinfo.rpmj != undefined) {
        var objectrpmj = math.format(
          cpv.currcp.objectinfo.rpmj, 4
        );

      }
      else if (cpv.currcp.objectinfo.reducedpropermotion != undefined) {
        var objectrpmj = math.format(
          cpv.currcp.objectinfo.reducedpropermotion, 4
        );

      }
      else {
        var objectrpmj = '';
      }

      // format the coordinates and PM
      var coordspm =
          '<strong>Equatorial (&alpha;, &delta;):</strong> ' +
          '<a title="SIMBAD search at these coordinates" ' +
          'href="http://simbad.u-strasbg.fr/simbad/sim-coo?Coord=' +
          cpv.currcp.objectinfo.ra + '+' + cpv.currcp.objectinfo.decl +
          '&Radius=1&Radius.unit=arcmin' +
          '" rel="nofollow" target="_blank">' +
          objectra + ', ' + objectdecl + '</a><br>' +
          '<strong>Galactic (l, b):</strong> ' +
          objectgl + ', ' + objectgb + '<br>' +
          '<strong>Total PM:</strong> ' + objectpm + '<br>' +
          '<strong>Reduced PM<sub>J</sub>:</strong> ' + objectrpmj;

      // see if we can get the GAIA parallax
      if (gaia_ok && cpv.currcp.objectinfo.gaia_parallaxes[0]) {

        var gaia_parallax = math.format(
          cpv.currcp.objectinfo.gaia_parallaxes[0], 5
        );
        coordspm = coordspm + '<br>' +
          '<strong>GAIA parallax:</strong> ' +
          gaia_parallax + ' mas';

      }

      $('#coordspm').html(coordspm);

      //
      // handle the mags
      //

      var magnotices = [];

      if (cpv.currcp.objectinfo.bmagfromjhk != undefined &&
          cpv.currcp.objectinfo.bmagfromjhk) {
        magnotices.push('B');
      }
      if (cpv.currcp.objectinfo.vmagfromjhk != undefined &&
          cpv.currcp.objectinfo.vmagfromjhk) {
        magnotices.push('V');
      }
      if (cpv.currcp.objectinfo.sdssufromjhk != undefined &&
          cpv.currcp.objectinfo.sdssufromjhk) {
        magnotices.push('u');
      }
      if (cpv.currcp.objectinfo.sdssgfromjhk != undefined &&
          cpv.currcp.objectinfo.sdssgfromjhk) {
        magnotices.push('g');
      }
      if (cpv.currcp.objectinfo.sdssrfromjhk != undefined &&
          cpv.currcp.objectinfo.sdssrfromjhk) {
        magnotices.push('r');
      }
      if (cpv.currcp.objectinfo.sdssifromjhk != undefined &&
          cpv.currcp.objectinfo.sdssifromjhk) {
        magnotices.push('i');
      }
      if (cpv.currcp.objectinfo.sdsszfromjhk != undefined &&
          cpv.currcp.objectinfo.sdsszfromjhk) {
        magnotices.push('z');
      }

      if (magnotices.length > 0) {
        $('#magnotice').html('<br>(' + magnotices.join('') +
                             ' via JHK transform)');
      }

      // set up the cmdplots property for currcp
      cpv.currcp.cmdplots = [];


      // set up GAIA info
      if (gaia_ok) {
        var gaiamag = cpv.currcp.objectinfo.gaia_mags[0];
        if (cpv.currcp.objectinfo.gaiak_colors != null) {
          var gaiakcolor = cpv.currcp.objectinfo.gaiak_colors[0];
        }
        else {
          var gaiakcolor = null;
        }
        var gaiaabsmag = cpv.currcp.objectinfo.gaia_absolute_mags[0];
      }
      else {
        var gaiamag = null;
        var gaiakcolor = null;
        var gaiaabsmag = null;
      }

      //
      // now we need to handle both generations of checkplots
      //

      // this is for the current generation of checkplots
      if (cpv.currcp.objectinfo.hasOwnProperty('available_bands')) {

        var mind = 0;
        var cind = 0;
        var mlen = cpv.currcp.objectinfo['available_bands'].length;
        var clen = cpv.currcp.objectinfo['available_colors'].length;

        var maglabel_pairs = [];
        var colorlabel_pairs = [];

        var thiskey = null;
        var thislabel = null;
        var thisval = null;

        // generate the mag-label pairs
        for (mind; mind < mlen; mind++) {

          thiskey = cpv.currcp.objectinfo['available_bands'][mind];
          thislabel =
            cpv.currcp.objectinfo['available_band_labels'][mind];
          thisval = math.format(cpv.currcp.objectinfo[thiskey],
                                5);
          maglabel_pairs.push('<span class="no-wrap-break">' +
                              '<strong><em>' +
                              thislabel +
                              '</em>:</strong> ' +
                              thisval +
                              '</span>');

        }

        // generate the color-label pairs
        for (cind; cind < clen; cind++) {

          thiskey = cpv.currcp.objectinfo['available_colors'][cind];
          thislabel =
            cpv.currcp.objectinfo['available_color_labels'][cind];
          thisval = math.format(cpv.currcp.objectinfo[thiskey],
                                4);

          if (cpv.currcp.objectinfo.dereddened != undefined &&
              cpv.currcp.objectinfo.dereddened) {
            thislabel = '(' + thislabel
              + ')<sub>0</sub>';
            $('#derednotice').html('<br>(dereddened)');
          }
          else {
            thislabel = '(' + thislabel + ')';
          }

          colorlabel_pairs.push('<span class="no-wrap-break">' +
                                '<strong><em>' +
                                thislabel +
                                '</em>:</strong> ' +
                                thisval +
                                '</span>');

        }


        // now add the GAIA information if it exists
        maglabel_pairs.push(
          '<span class="no-wrap-break">' +
            '<strong><em>GAIA G</em>:</strong> ' +
            math.format(gaiamag,5) +
            '</span>'
        );
        maglabel_pairs.push(
          '<span class="no-wrap-break">' +
            '<strong><em>GAIA M<sub>G</sub></em>:</strong> ' +
            math.format(gaiaabsmag,5) +
            '</span>'
        );


        maglabel_pairs = maglabel_pairs.join(', ');
        colorlabel_pairs = colorlabel_pairs.join(', ');

        $('#mags').html(maglabel_pairs);
        $('#colors').html(colorlabel_pairs);

      }

      // this is for the older generation of checkplots
      else {

        var mags = '<strong><em>ugriz</em>:</strong> ' +
            math.format(cpv.currcp.objectinfo.sdssu,5) + ', ' +
            math.format(cpv.currcp.objectinfo.sdssg,5) + ', ' +
            math.format(cpv.currcp.objectinfo.sdssr,5) + ', ' +
            math.format(cpv.currcp.objectinfo.sdssi,5) + ', ' +
            math.format(cpv.currcp.objectinfo.sdssz,5) + '<br>' +
            '<strong><em>JHK</em>:</strong> ' +
            math.format(cpv.currcp.objectinfo.jmag,5) + ', ' +
            math.format(cpv.currcp.objectinfo.hmag,5) + ', ' +
            math.format(cpv.currcp.objectinfo.kmag,5) + '<br>' +
            '<strong><em>BV</em>:</strong> ' +
            math.format(cpv.currcp.objectinfo.bmag,5) + ', ' +
            math.format(cpv.currcp.objectinfo.vmag,5) + '<br>' +
            '<strong><em>GAIA G</em>:</strong> ' +
            math.format(gaiamag,5) + ', ' +
            '<strong><em>GAIA M<sub>G</sub></em>:</strong> ' +
            math.format(gaiaabsmag,5);

        $('#mags').html(mags);

        //
        // handle the colors
        //

        if (cpv.currcp.objectinfo.dereddened != undefined &&
            cpv.currcp.objectinfo.dereddened) {

          $('#derednotice').html('<br>(dereddened)');

          var colors =
              '<strong><em>(B - V)<sub>0</sub></em>:</strong> ' +
              math.format(cpv.currcp.objectinfo.bvcolor,4) + ',  ' +
              '<strong><em>(V - K)<sub>0</sub></em>:</strong> ' +
              math.format(cpv.currcp.objectinfo.vkcolor,4) + '<br>' +
              '<strong><em>(J - K)<sub>0</sub></em>:</strong> ' +
              math.format(cpv.currcp.objectinfo.jkcolor,4) + ',  ' +
              '<strong><em>(i - J)<sub>0</sub></em>:</strong> ' +
              math.format(cpv.currcp.objectinfo.ijcolor,4) + '<br>' +
              '<strong><em>(g - K)<sub>0</sub></em>:</strong> ' +
              math.format(cpv.currcp.objectinfo.gkcolor,4) + ',  ' +
              '<strong><em>(g - r)<sub>0</sub></em>:</strong> ' +
              math.format(cpv.currcp.objectinfo.grcolor,4);

        }

        else {
          var colors =
              '<strong><em>(B - V)</em>:</strong> ' +
              math.format(cpv.currcp.objectinfo.bvcolor,4) + ',  ' +
              '<strong><em>(V - K)</em>:</strong> ' +
              math.format(cpv.currcp.objectinfo.vkcolor,4) + '<br>' +
              '<strong><em>(J - K)</em>:</strong> ' +
              math.format(cpv.currcp.objectinfo.jkcolor,4) + ',  ' +
              '<strong><em>(i - J)</em>:</strong> ' +
              math.format(cpv.currcp.objectinfo.ijcolor,4) + '<br>' +
              '<strong><em>(g - K)</em>:</strong> ' +
              math.format(cpv.currcp.objectinfo.gkcolor,4) + ',  ' +
              '<strong><em>(g - r)</em>:</strong> ' +
              math.format(cpv.currcp.objectinfo.grcolor,4);
        }

        // format the colors
        $('#colors').html(colors);

      }


      //
      // additional stuff
      //

      // first, empty out the extra info table
      $("#objectinfo-extra").empty();

      // add the color classification if available
      if (cpv.currcp.objectinfo.color_classes != undefined &&
          cpv.currcp.objectinfo.color_classes.length > 0) {

        var formatted_color_classes =
            cpv.currcp.objectinfo.color_classes.join(', ');
        $('#objectinfo-extra')
          .append(
            "<tr>" +
              "<th>Color classification</th>" +
              "<td>" + formatted_color_classes + "</td>" +
              "</tr>"
          );

      }


      // neighbors
      if (cpv.currcp.objectinfo.neighbors != undefined ||
          (cpv.currcp.objectinfo.gaia_ids != undefined &&
           cpv.currcp.objectinfo.gaia_ids.length > 0) ) {

        if (cpv.currcp.objectinfo.neighbors > 0) {

          var formatted_neighbors =
              '<strong><em>from LCs in collection</em>:</strong> ' +
              cpv.currcp.objectinfo.neighbors + '<br>' +
              '<em>closest distance</em>: ' +
              math.format(cpv.currcp.objectinfo.closestdistarcsec,4) +
              '&Prime;<br>';
        }
        else {
          var formatted_neighbors =
              '<strong><em>from LCs in collection</em>:</strong> 0<br>';
        }

        if (gaia_ok) {
          var formatted_gaia =
              '<strong><em>from GAIA</em>:</strong> ' +
              (cpv.currcp.objectinfo.gaia_ids.length - 1) + '<br>' +
              '<em>closest distance</em>: ' +
              math.format(
                cpv.currcp.objectinfo.gaia_closest_distarcsec, 4
              ) + '&Prime;<br>' +
              '<em>closest G mag (obj - nbr)</em>: ' +
              math.format(
                cpv.currcp.objectinfo.gaia_closest_gmagdiff, 4
              ) + ' mag';

        }
        else {
          var formatted_gaia =
              '<strong><em>GAIA query failed</em>:</strong> ' +
              gaia_message;
        }

        $('#objectinfo-extra')
          .append(
            "<tr>" +
              "<th>Neighbors within " +
              math.format(cpv.currcp.objectinfo.searchradarcsec,
                          3) + "&Prime;</th>" +
              "<td>" + formatted_neighbors +
              formatted_gaia +
              "</td>" +
              "</tr>"
          );


      }

      // get the CMDs for this object if there are any
      if (cpv.currcp.hasOwnProperty('colormagdiagram') &&
          cpv.currcp.colormagdiagram != null) {

        var cmdlist = Object.getOwnPropertyNames(
          cpv.currcp.colormagdiagram
        );

        var cmdkey = '<tr><th>' +
            'Color-magnitude diagrams' +
            '</th>';

        var cmdval = '<td>';
        var cmdimgs = [];

        // prepare the img divs
        var cmdi = 0;
        for (cmdi; cmdi < cmdlist.length; cmdi++) {

          var thiscmdlabel = cmdlist[cmdi];
          var thiscmdplot = cpv.currcp.colormagdiagram[cmdlist[cmdi]];

          var cmddd =
              '<div class="dropdown">' +
              '<a href="#" ' +
              'title="Click to see the ' +
              thiscmdlabel +
              ' color-magnitude ' +
              'diagram for this object" ' +
              'id="cmd-' + cmdi +
              '-dropdown" data-toggle="dropdown" ' +
              'aria-haspopup="true" aria-expanded="false">' +
              '<strong>' + thiscmdlabel + ' CMD</strong>' +
              '</a>' +
              '<div class="dropdown-menu text-sm-center cmd-dn" ' +
              'aria-labelledby="cmd-' + cmdi + '-dropdown">' +
              '<img id="cmd-' + cmdi +'-plot" class="img-fluid">' +
              '</div></div>';
          cmdval = cmdval + cmddd;
          cmdimgs.push('#cmd-' + cmdi + '-plot');

        }


        cmdval = cmdkey + cmdval + '</td></tr>';
        $('#objectinfo-extra').append(cmdval);

        // now populate the img divs with the actual CMD images
        cmdi = 0;
        for (cmdi; cmdi < cmdlist.length; cmdi++) {

          var thiscmdlabel = cmdlist[cmdi];
          var thiscmdplot = cpv.currcp.colormagdiagram[thiscmdlabel];
          cputils.b64_to_image(thiscmdplot, cmdimgs[cmdi]);

        }

      }

      // get SIMBAD info if possible
      if (cpv.currcp.objectinfo.simbad_status != undefined) {

        if (simbad_ok) {

          var simbad_best_allids =
              cpv.currcp.objectinfo.simbad_best_allids
              .split('|').join(', ');

          var formatted_simbad =
              '<strong><em>matching objects</em>:</strong> ' +
              (cpv.currcp.objectinfo.simbad_nmatches) + '<br>' +
              '<em>closest distance</em>: ' +
              math.format(
                cpv.currcp.objectinfo.simbad_best_distarcsec, 4
              ) + '&Prime;<br>' +
              '<em>closest object ID</em>: ' +
              cpv.currcp.objectinfo.simbad_best_mainid + '<br>' +
              '<em>closest object type</em>: ' +
              cpv.currcp.objectinfo.simbad_best_objtype + '<br>' +
              '<em>closest object other IDs</em>: ' +
              simbad_best_allids;

        }
        else {
          var formatted_simbad =
              '<strong><em>SIMBAD query failed</em>:</strong> ' +
              simbad_message;
        }

        $('#objectinfo-extra')
          .append(
            "<tr>" +
              "<th>SIMBAD information</th>" +
              "<td>" + formatted_simbad +
              "</td>" +
              "</tr>"
          );

      }

      // update the magseries plot
      cputils.b64_to_image(cpv.currcp.magseries,
                           '.magseriesplot');

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

      // get the period and epoch
      $('#objectperiod').val(cpv.currcp.varinfo.varperiod);
      $('#objectepoch').val(cpv.currcp.varinfo.varepoch);

      // update the rest of the object info
      $('#objecttags').val(cpv.currcp.objectinfo.objecttags);
      $('#objectcomments').val(cpv.currcp.objectcomments);
      $('#vartags').val(cpv.currcp.varinfo.vartags);

      // update the phased light curves

      // first, count the number of methods we have in the cp
      var lspmethods = cpv.currcp.pfmethods;
      var ncols = lspmethods.length;

      // enlarge the width of the phased container so it overflows in x
      // this should be handled correctly hopefully
      var phasedcontainer_maxwidth = ncols*375;
      $('.phased-container').width(phasedcontainer_maxwidth);

      var colwidth = math.floor(12/ncols);

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
              ' phased-container-col" data-lspmethod="' +
              lspmethod + '">';
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

            if (periodind in cpv.currcp[lspmethod] &&
                cpv.currcp[lspmethod][periodind].period != null) {

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
                  'class="img-fluid zoomable-tile" id="plot-' +
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

      //
      // handle the neighbors tab for this object
      //

      // 1. empty the rows for the gaia table and lcc neighbor list
      $('#neighbor-container').empty();

      $('#gaia-neighbor-tbody').empty();
      if (cpv.currcp.objectinfo.gaia_ids != undefined) {
        $('#gaia-neighbor-count').html(
          cpv.currcp.objectinfo.gaia_ids.length
        );
      }
      else {
        $('#gaia-neighbor-count').html('No');
      }

      // empty the neighbor container and then set its max-width
      $('#lcc-neighbor-container').empty();
      $('#lcc-neighbor-container').width(
        300 + lspmethods.length*300
      );
      if (lspmethods.length == 0) {
        $('#lcc-neighbor-container').width('100%');
      }

      if (cpv.currcp.neighbors != undefined &&
          cpv.currcp.neighbors.length > 0) {
        $("#lcc-neighbor-count").html(cpv.currcp.neighbors.length);
      }
      else {
        $("#lcc-neighbor-count").html('No');
      }

      // 2. update the search radius
      if (cpv.currcp.objectinfo.searchradarcsec != undefined) {
        $('#gaia-neighbor-maxdist').html(
          ' within ' +
            cpv.currcp.objectinfo.searchradarcsec +
            '&Prime;'
        );
        $('#lcc-neighbor-maxdist').html(
          ' within ' +
            cpv.currcp.objectinfo.searchradarcsec +
            '&Prime;'
        );
      }
      else {
        $('#gaia-neighbor-maxdist').html(
          ''
        );
        $('#lcc-neighbor-maxdist').html(
          ''
        );

      }

      // 3. update the GAIA neighbors
      if (gaia_ok) {

        // for each gaia neighbor, put in a table row
        var gi = 0;

        for (gi; gi < cpv.currcp.objectinfo.gaia_ids.length; gi++) {

          if (cpv.currcp.objectinfo.gaia_xypos != null) {
            var gaia_x = cpv.currcp.objectinfo.gaia_xypos[gi][0];
          }
          else {
            var gaia_x = 0.0;
          }

          if (cpv.currcp.objectinfo.gaia_xypos != null) {
            var gaia_y = cpv.currcp.objectinfo.gaia_xypos[gi][1];
          }
          else {
            var gaia_y = 0.0;
          }


          // special formatting for the object itself
          if (gi == 0) {
            var rowhtml = '<tr class="gaia-objectlist-row ' +
                'text-primary' +
                '" ' +
                'data-gaiaid="' +
                cpv.currcp.objectinfo.gaia_ids[gi] +
                '" data-xpos="' +
                gaia_x +
                '" data-ypos="' +
                gaia_y +
                '" >' +
                '<td>obj: ' + cpv.currcp.objectinfo.gaia_ids[gi] +
                '</td>' +
                '<td>' + math.format(
                  cpv.currcp.objectinfo.gaia_dists[gi], 3
                ) +
                '</td>' +
                '<td>' +
                math.format(
                  cpv.currcp.objectinfo.gaia_parallaxes[gi], 3
                ) + ' &plusmn; ' +
                math.format(
                  cpv.currcp.objectinfo.gaia_parallax_errs[gi], 3
                ) +
                '</td>' +
                '<td>' +
                math.format(
                  cpv.currcp.objectinfo.gaia_mags[gi], 3
                ) +
                '</td>' +
                '<td>' +
                math.format(
                  cpv.currcp.objectinfo.gaia_absolute_mags[gi], 3
                ) +
                '</td>' +
                '</tr>';
          }

          else {

            var rowhtml = '<tr class="gaia-objectlist-row" ' +
                'data-gaiaid="' +
                cpv.currcp.objectinfo.gaia_ids[gi] +
                '" data-xpos="' +
                gaia_x +
                '" data-ypos="' +
                gaia_y +
                '" >' +
                '<td>' + cpv.currcp.objectinfo.gaia_ids[gi] +
                '</td>' +
                '<td>' + math.format(
                  cpv.currcp.objectinfo.gaia_dists[gi], 3
                ) +
                '</td>' +
                '<td>' +
                math.format(
                  cpv.currcp.objectinfo.gaia_parallaxes[gi], 3
                ) + ' &plusmn; ' +
                math.format(
                  cpv.currcp.objectinfo.gaia_parallax_errs[gi], 3
                ) +
                '</td>' +
                '<td>' +
                math.format(
                  cpv.currcp.objectinfo.gaia_mags[gi], 3
                ) +
                '</td>' +
                '<td>' +
                math.format(
                  cpv.currcp.objectinfo.gaia_absolute_mags[gi], 3
                ) +
                '</td>' +
                '</tr>';

          }

          $('#gaia-neighbor-tbody').append(rowhtml);

        }

      }

      // if GAIA xmatch failed, fill in the table without special
      // formatting if possible
      else if (cpv.currcp.objectinfo.gaia_ids != undefined) {

        // for each gaia neighbor, put in a table row
        var gi = 0;

        // put in any rows of neighbors if there are any
        for (gi; gi < cpv.currcp.objectinfo.gaia_ids.length; gi++) {

          if (cpv.currcp.objectinfo.gaia_xypos != null) {
            var gaia_x = cpv.currcp.objectinfo.gaia_xypos[gi][0];
          }
          else {
            var gaia_x = 0.0;
          }

          if (cpv.currcp.objectinfo.gaia_xypos != null) {
            var gaia_y = cpv.currcp.objectinfo.gaia_xypos[gi][1];
          }
          else {
            var gaia_y = 0.0;
          }

          var rowhtml = '<tr class="gaia-objectlist-row" ' +
              'data-gaiaid="' +
              cpv.currcp.objectinfo.gaia_ids[gi] +
              '" data-xpos="' +
              gaia_x +
              '" data-ypos="' +
              gaia_y +
              '" >' +
              '<td>' + cpv.currcp.objectinfo.gaia_ids[gi] +
              '</td>' +
              '<td>' + math.format(
                cpv.currcp.objectinfo.gaia_dists[gi], 3
              ) +
              '</td>' +
              '<td>' +
              math.format(
                cpv.currcp.objectinfo.gaia_parallaxes[gi], 3
              ) + ' &plusmn; ' +
              math.format(
                cpv.currcp.objectinfo.gaia_parallax_errs[gi], 3
              ) +
              '</td>' +
              '<td>' +
              math.format(
                cpv.currcp.objectinfo.gaia_mags[gi], 3
              ) +
              '</td>' +
              '<td>' +
              math.format(
                cpv.currcp.objectinfo.gaia_absolute_mags[gi], 3
              ) +
              '</td>' +
              '</tr>';
          $('#gaia-neighbor-tbody').append(rowhtml);

        }

      }

      // 4. update the LCC neighbor list

      //
      // first, add the target in any case
      //

      var ni = 0;

      // set the column width
      var nbrcolw = math.floor(12/(1 + lspmethods.length));

      // if there aren't any PFs, reset the col width to 3
      if (lspmethods.length == 0) {
        nbrcolw = 4;
      }
      console.log(colwidth + ' ' + nbrcolw);

      // make the plots for the target object

      var rowheader = '<h6>' + 'Target: ' +
          cpv.currcp.objectid + ' at (&alpha;, &delta;) = (' +
          math.format(cpv.currcp.objectinfo.ra, 5) + ', ' +
          math.format(cpv.currcp.objectinfo.decl, 5) + ')</h6>';

      var rowplots = [
        '<div class="col-sm-' + nbrcolw + ' mx-0 px-0">' +
          '<img src="data:image/png;base64,' +
          cpv.currcp.magseries +
          '" class="img-fluid zoomable-tile">' +
          '</div>'
      ];
      var nli = 0;
      for (nli; nli < lspmethods.length; nli++) {

        if (cpv.currcp[lspmethods[nli]]['phasedlc0']['plot'] != null) {

          var thisnphased =
              '<div class="col-sm-' + nbrcolw + ' mx-0 px-0">' +
              '<img src="data:image/png;base64,' +
              cpv.currcp[lspmethods[nli]]['phasedlc0']['plot'] +
              '" class="img-fluid zoomable-tile">' +
              '</div>';
          rowplots.push(thisnphased);

        }

      }

      // put together this row of plots
      var rowplots_str = rowplots.join(' ');

      // put together this row
      var nbrrow = '<div class="row">' +
          '<div class="col-sm-12">' +
          rowheader + '</div></div>' +
          '<div class="row bot-mrg-20px">' +
          rowplots_str +
          '</div>';

      $('#lcc-neighbor-container').append(nbrrow);

      //
      // then, add the neighbors if there are any
      //

      if (cpv.currcp.neighbors.length > 0) {

        // now make the plots for the neighbors
        for (ni; ni < cpv.currcp.neighbors.length; ni++) {

          var nbrobjectid = cpv.currcp.neighbors[ni].objectid;
          var nbrra = cpv.currcp.neighbors[ni].objectinfo.ra;
          var nbrdecl = cpv.currcp.neighbors[ni].objectinfo.decl;
          var nbrdist =
              cpv.currcp.neighbors[ni].objectinfo.distarcsec;


          // check if this neighbor is present in the current
          // collection
          var nbrcp = $('#checkplot-current').text().replace(
            cpv.currcp.objectid,
            nbrobjectid
          );

          if (cptracker.checkplotlist.indexOf(nbrcp) != -1) {
            var nbrlink =
                '<a title="load the checkplot for this object" ' +
                'href="#" class="nbrload-checkplot" ' +
                'data-fname="' + nbrcp + '">' +
                nbrobjectid + '</a>';
          }

          else {
            var nbrlink = nbrobjectid;
          }

          rowheader = '<h6>' + '<span class="neighbor-num">N' +
            (ni+1) +
            '</span>: ' +
            nbrlink + ' at (&alpha;, &delta;) = <a href="' +
            'http://simbad.u-strasbg.fr/simbad/sim-coo?Coord=' +
            nbrra + '+' + nbrdecl +
            '&Radius=30&Radius.unit=arcsec" rel="nofollow" ' +
            'target="_blank" ' +
            'title="SIMBAD search at these coordinates">' +
            '(' + math.format(nbrra, 5) + ', ' +
            math.format(nbrdecl, 5) + ')</a>, distance: ' +
            math.format(nbrdist,3) + '&Prime;';

          // add in the magdiffs and colordiffs for this neighbor if
          // available
          var nbr_magdiffs =
              cpv.currcp.neighbors[ni].objectinfo.magdiffs;
          var nbr_colordiffs =
              cpv.currcp.neighbors[ni].objectinfo.colordiffs;

          // the magdiffs for this neighbor
          if (nbr_magdiffs != undefined) {

            var nbr_magdiff_dd_start =
                ', <span class="dropdown">' +
                '<a href="#" ' +
                'title="Click to see the neighbor-target magnitude ' +
                'diffs for this neighbor" ' +
                'id="magdiff-' + ni +
                '-dropdown" data-toggle="dropdown" ' +
                'aria-haspopup="true" aria-expanded="false">' +
                'target-neighbor mag diffs' +
                '</a>' +
                '<span class="dropdown-menu text-sm-center" ' +
                'aria-labelledby="magdiff-' + ni + '-dropdown">' +
                '<table id="magdiff-' + ni +
                '-table" class="table table-sm table-bordered">' +
                '<thead>' +
                '<tr><th>band</th><th>mag diff</th></tr></thead>';

            var nbr_magdiff_keys = Object.getOwnPropertyNames(
              nbr_magdiffs
            );
            var magdiffind = 0;
            var magdiff_table = [];
            var magdiff_key = null;

            for (magdiffind;
                 magdiffind < nbr_magdiff_keys.length;
                 magdiffind++) {

              magdiff_key = nbr_magdiff_keys[magdiffind];
              magdiff_table.push(
                '<tr><td>' +
                  magdiff_key +
                  '</td><td>' +
                  math.format(nbr_magdiffs[magdiff_key],4) +
                  '</td></tr>'
              );

            }

            var nbr_magdiff_dd_end =
                '</table>' +
                '</span></span>';

            magdiff_table = magdiff_table.join(' ');
            var nbr_magdiff_dd = nbr_magdiff_dd_start +
                magdiff_table +
                nbr_magdiff_dd_end;
          }

          else {
            var nbr_magdiff_dd = '';
          }

          // the colordiffs for this neighbor
          if (nbr_colordiffs != undefined) {

            var nbr_colordiff_dd_start =
                ', <span class="dropdown">' +
                '<a href="#" ' +
                'title="Click to see the neighbor-target color ' +
                'diffs for this neighbor" ' +
                'id="colordiff-' + ni +
                '-dropdown" data-toggle="dropdown" ' +
                'aria-haspopup="true" aria-expanded="false">' +
                'target-neighbor color diffs' +
                '</a>' +
                '<span class="dropdown-menu text-sm-center" ' +
                'aria-labelledby="colordiff-' + ni + '-dropdown">' +
                '<table id="colordiff-' + ni +
                '-table" class="table table-sm table-bordered">' +
                '<thead>' +
                '<tr><th>color</th><th>color diff</th></tr></thead>';

            var nbr_colordiff_keys = Object.getOwnPropertyNames(
              nbr_colordiffs
            );
            var colordiffind = 0;
            var colordiff_table = [];
            var colordiff_key = null;

            for (colordiffind;
                 colordiffind < nbr_colordiff_keys.length;
                 colordiffind++) {

              colordiff_key = nbr_colordiff_keys[colordiffind];
              colordiff_table.push(
                '<tr><td>' +
                  colordiff_key +
                  '</td><td>' +
                  math.format(nbr_colordiffs[colordiff_key],4) +
                  '</td></tr>'
              );

            }

            var nbr_colordiff_dd_end =
                '</table>' +
                '</span></span>';

            colordiff_table = colordiff_table.join(' ');
            var nbr_colordiff_dd = nbr_colordiff_dd_start +
                colordiff_table +
                nbr_colordiff_dd_end;
          }

          else {
            var nbr_colordiff_dd = '';
          }

          // finish up the header row for this neighbor by adding mag
          // and color diff info if they're available
          rowheader = rowheader +
            nbr_magdiff_dd +
            nbr_colordiff_dd +
            '</h6>';

          // get the magseries plot for this neighbor if available
          if (cpv.currcp.neighbors[ni].magseries != undefined) {

            // add the magseries plot for this neighbor
            rowplots = [
              '<div class="col-sm-' + nbrcolw + ' mx-0 px-0">' +
                '<img src="data:image/png;base64,' +
                cpv.currcp.neighbors[ni].magseries +
                '" class="img-fluid zoomable-tile">' +
                '</div>'
            ];
          }

          else {
            rowplots = [
              '<div class="col-sm-' + nbrcolw + ' mx-0 px-0">' +
                '<img src="/static/nolc-available.png"' +
                ' class="img-fluid">' +
                '</div>'
            ];

          }

          // for each lspmethod, add the phased LC for the neighbor
          for (nli = 0; nli < lspmethods.length; nli++) {

            if (cpv.currcp.neighbors[ni][lspmethods[nli]]
                != undefined) {

              thisnphased =
                '<div class="col-sm-' + nbrcolw + ' px-0">' +
                '<img src="data:image/png;base64,' +
                cpv.currcp.neighbors[ni][lspmethods[nli]]['plot'] +
                '" class="img-fluid zoomable-tile">' +
                '</div>';
              rowplots.push(thisnphased);

            }

            else {
              thisnphased =
                '<div class="col-sm-' + nbrcolw + ' px-0">' +
                '<img src="/static/nolc-available.png"' +
                '" class="img-fluid">' +
                '</div>';
              rowplots.push(thisnphased);

            }

          }

          // put together this row of plots
          rowplots_str = rowplots.join(' ');

          // put together this row
          nbrrow = '<div class="row">' +
            '<div class="col-sm-12">' +
            rowheader + '</div></div>' +
            '<div class="row bot-mrg-20px">' +
            rowplots_str +
            '</div>';

          $('#lcc-neighbor-container').append(nbrrow);

        }

      }


      // 5. put together the xmatch info

      // first, reset stuff
      $('#unmatched-catalogs').html(
        '<p>No cross-matches for this object were found.</p>'
      );
      $('#matched-catalogs').empty();

      // then, go through the xmatch content
      if (cpv.currcp.xmatch != undefined) {

        var xmcatrows = [];
        var unxmcatrows = [];

        var xmk = null;
        var tablecolkeys = null
        var tablecolnames = null;
        var tablecolunits = null;

        var xmcrow = null;
        var tci = 0;
        var thisunit = '';

        var xm = null;

        for (xm in cpv.currcp.xmatch) {

          xmk = cpv.currcp.xmatch[xm];

          // if we found a cross-match, then generate the table for it
          if (xmk['found'] == true) {

            tablecolkeys = xmk['colkeys'];
            tablecolnames = xmk['colnames'];
            tablecolunits = xmk['colunit'];

            xmcrow = [
              '<div class="row mt-1">' +
                '<div class="col-sm-12">' +
                '<h6>Matched to: <abbr ' +
                'title="' + xmk['desc'] + ' ">' +
                '<strong>' +
                xmk['name'] +
                '</strong></abbr> within ' +
                math.format(xmk['distarcsec'],3) + '&Prime;' +
                '</h6>'
            ];

            xmcrow.push(
              '<table class="table-sm objectinfo-table">' +
                '<thead><tr>'
            );


            // first, the header row
            for (tci = 0;
                 tci < tablecolkeys.length;
                 tci++) {

              if (tablecolunits[tci] != null) {
                thisunit = '[' + tablecolunits[tci] + ']';
              }
              else {
                thisunit = '&nbsp;';
              }

              xmcrow.push(
                '<th>' +
                  tablecolnames[tci] + '<br>' + thisunit +
                  '</th>'
              );

            }

            // close out this row
            xmcrow.push(
              '</tr></thead><tbody><tr>'
            );

            // next, do the value row
            // first, the header row
            for (tci = 0;
                 tci < tablecolkeys.length;
                 tci++) {

              // format float things
              if (tablecolunits[tci] != null) {
                xmcrow.push(
                  '<td>' +
                    math.format(
                      xmk['info'][tablecolkeys[tci]],
                      5
                    ) +
                    '</td>'
                );
              }

              // otherwise, leave them alone
              else {
                xmcrow.push(
                  '<td>' +
                    xmk['info'][tablecolkeys[tci]] +
                    '</td>'
                );
              }


            }

            // close out the table
            xmcrow.push(
              '</tr></tbody></table></div></div>'
            );

            xmcrow = xmcrow.join(' ');
            xmcatrows.push(xmcrow)

          }

          else {

            unxmcatrows.push(
              '<div class="row mt-1">' +
                '<div class="col-sm-12">' +
                '<h6>No matches found in <abbr ' +
                'title="' + xmk['desc'] + ' ">' +
                '<strong>' +
                xmk['name'] +
                '</strong></abbr></h6></div></div>'
            );

          }

        }

        xmcatrows = xmcatrows.join(' ');
        unxmcatrows = unxmcatrows.join(' ');

        $('#matched-catalogs').html(xmcatrows);
        $('#unmatched-catalogs').html(unxmcatrows);

      }

      //
      // end of main processing for a checkplot
      //

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

      // re-initialize all popovers
      $('[data-toggle="popover"]').popover();

      // highlight the file in the sidebar list
      $("a.checkplot-load")
        .filter("[data-fname='" + filename + "']")
        .wrap('<strong></strong>');

      // fix the height of the sidebar as required
      var winheight = $(window).height();
      var docheight = $(document).height();
      var ctrlheight = $('.sidebar-controls').height();
      $('.sidebar').css({'height': docheight + 'px'});

      // get rid of the spinny thing
      if (cpv.loadsuccess) {

        $('#alert-box').empty();

      }

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
      console.error(xhr);

    });


  },

  // this functions saves the current checkplot by doing a POST request to the
  // backend. this MUST be called on every checkplot list action (i.e. next,
  // prev, before load of a new checkplot, so changes are always saved). UI
  // elements in the checkplot list will tag the saved checkplots
  // appropriately
  save_checkplot: function (nextfunc_callback, nextfunc_arg, savetopng) {

    // make sure the current checkplot exists
    if (cpv.currcp === null) {

      console.log('checkplot not loaded successfully, nothing to save');

      // update the filter select box
      $('#reviewedfilter').change();

      // call the next function. we call this here so we can be sure
      // the save finished before the next action starts
      if ( !((nextfunc_callback === undefined) ||
             (nextfunc_callback === null)) &&
           !((nextfunc_arg === undefined) ||
             (nextfunc_arg === null)) ) {
        nextfunc_callback(nextfunc_arg);
        // clean out the alert box if there's a next function
        $('#alert-box').empty();

      }

      return null;

    }

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
      var ajaxurl = cpv.CPSERVER_BASEURL + 'cp/' +
          encodeURIComponent(cputils.b64_encode(cpv.currfile));

      // get the current value of the objectisvar select box
      cpv.currcp.varinfo.objectisvar = $('#varcheck').val();

      // make sure that we've saved the input varinfo, objectinfo and
      // comments, period, epoch, etc.
      cpv.currcp.varinfo.vartags = $('#vartags').val();
      cpv.currcp.objectinfo.objecttags = $('#objecttags').val();
      cpv.currcp.objectcomments = $('#objectcomments').val();
      cpv.currcp.varinfo.objectisvar = parseInt($('#varcheck').val());
      cpv.currcp.varinfo.varperiod = parseFloat($('#objectperiod').val());
      cpv.currcp.varinfo.varepoch = parseFloat($('#objectepoch').val());

      // make sure we also save the applied filters in the period-search
      // tab
      cpv.currcp.uifilters = {
        psearch_timefilters:$('#psearch-timefilters').val(),
        psearch_magfilters:$('#psearch-magfilters').val(),
        psearch_sigclip:$('#psearch-sigclip').val()
      }

      var cppayload = JSON.stringify(
        {objectid: cpv.currcp.objectid,
         objectinfo: cpv.currcp.objectinfo,
         varinfo: cpv.currcp.varinfo,
         comments: cpv.currcp.objectcomments,
         uifilters: cpv.currcp.uifilters}
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
              (updateinfo.cpfpng.length > 0)) {

            console.log('getting checkplot PNG from backend');

            // prepare the download
            // see https://gist.github.com/fupslot/5015897

            // base64 decode the sent string
            var pngstr = window.atob(updateinfo.cpfpng);

            // generate an array for the byte string.  we need to do
            // this because the string needs to be turned into the
            // correct (0,255) values for a PNG
            var arraybuf = new ArrayBuffer(pngstr.length);
            var pngarr = new Uint8Array(arraybuf);

            // turn the byte string to into a byte array
            for (var pngind = 0; pngind < pngstr.length; pngind++) {
              pngarr[pngind] = pngstr.charCodeAt(pngind);
            }

            // generate a blob from the byte array
            var pngblob = new Blob([pngarr],
                                   {type: 'image/png'});
            var pngurl = window.URL.createObjectURL(pngblob);

            updatemsg = '<a href="' + pngurl + '" ' +
              'download="checkplot-' + cpv.currcp.objectid +
              '.png">download checkplot PNG for this object</a>';
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
        if ( !((nextfunc_callback === undefined) ||
               (nextfunc_callback === null)) &&
             !((nextfunc_arg === undefined) ||
               (nextfunc_arg === null)) ) {
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
        $('#alert-box').html('');
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
        $('#alert-box').html('');
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
      if ((cpv.currcp != null) &&
          ('objectid' in cpv.currcp) &&
          (cpv.currfile.length > 0))  {
        cpv.save_checkplot(cpv.load_checkplot,filetoload);
      }

      else {
        // ask the backend for this file
        cpv.load_checkplot(filetoload);
      }

    });

    // this loads the neighbor's checkplot if it's in the current LC
    // collection loaded into the checkplotserver
    $('#lcc-neighbor-container').on('click','.nbrload-checkplot', function (e) {

      e.preventDefault();

      var filetoload = $(this).attr('data-fname');

      // save the currentcp if one exists, use the load_checkplot as a
      // callback to load the next one
      if ((cpv.currcp != null) &&
          ('objectid' in cpv.currcp) &&
          (cpv.currfile.length > 0))  {
        cpv.save_checkplot(cpv.load_checkplot,filetoload);
      }

      else {
        // ask the backend for this file
        cpv.load_checkplot(filetoload);
      }


    });


    // this handles the hover per objectid row to highlight the object in
    // the finder chart
    $('#gaia-neighbors').on('mouseover','.gaia-objectlist-row', function (e) {

      e.preventDefault();

      var canvas = document.getElementById('finderchart');
      var canvaswidth = canvas.width;
      var canvasheight = canvas.height;
      var ctx = canvas.getContext('2d');

      // FIXME: check if astropy.wcs returns y, x and we've been doing
      // this wrong all this time
      var thisx = $(this).attr('data-xpos');
      var thisy = $(this).attr('data-ypos');

      var cnvx = thisx * canvaswidth/300.0;

      // y is from the top of the image for canvas
      // FITS coords are from the bottom of the image
      var cnvy = (300.0 - thisy) * canvasheight/300.0;

      // save the damaged part of the image
      cputils.pixeltracker = ctx.getImageData(cnvx-20,cnvy-20,40,40);

      ctx.strokeStyle = 'green';
      ctx.lineWidth = 3.0
      ctx.strokeRect(cnvx-7.5,cnvy-7.5,12.5,12.5);

    });

    // this handles the repair to the canvas after the user mouses out of
    // the row
    $('#gaia-neighbors').on('mouseout','.gaia-objectlist-row', function (e) {

      e.preventDefault();

      var canvas = document.getElementById('finderchart');
      var canvaswidth = canvas.width;
      var canvasheight = canvas.height;
      var ctx = canvas.getContext('2d');

      var thisx = $(this).attr('data-xpos');
      var thisy = $(this).attr('data-ypos');

      var cnvx = thisx * canvaswidth/300.0;

      // y is from the top of the image for canvas
      // FITS coords are from the bottom of the image
      var cnvy = (300.0 - thisy) * canvasheight/300.0;

      // restore the imagedata if we have any
      if (cputils.pixeltracker != null) {
        ctx.putImageData(cputils.pixeltracker,
                         cnvx-20, cnvy-20);
      }

    });


    // clicking on a objectid in the sidebar
    $('#project-status').on('click', '.objload-checkplot', function (evt) {

      evt.preventDefault();

      var filetoload = $(this).attr('data-fname');

      // save the currentcp if one exists, use the load_checkplot as a
      // callback to load the next one
      if ((cpv.currcp != null) &&
          ('objectid' in cpv.currcp) &&
          (cpv.currfile.length > 0))  {
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

      // if readonlymode is true, we don't do anything
      if (!cpv.readonlymode) {

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

      }

    });

    // fancy zoom and pan effects for a phased LC tile
    // see https://codepen.io/ccrch/pen/yyaraz
    $('.phased-container')
      .on('mouseover', '.zoomable-tile', function (evt) {

        $(this).css({'transform': 'scale(1.7)',
                     'z-index':1000});

      });
    $('.phased-container')
      .on('mouseout', '.zoomable-tile', function (evt) {

        $(this).css({'transform': 'scale(1.0)',
                     'z-index':0});

      });

    // mouse zoom effects for LCC neighbor LC panels
    $('#lcc-neighbor-container')
      .on('mouseover', '.zoomable-tile', function (evt) {

        $(this).css({'transform': 'scale(1.7)',
                     'z-index':1000});

      });
    $('#lcc-neighbor-container')
      .on('mouseout', '.zoomable-tile', function (evt) {

        $(this).css({'transform': 'scale(1.0)',
                     'z-index':0});

      });

    // mouse zoom/pan effect for phased LC and periodogram panels on the
    // period-search tab
    $('.display-panel')
      .on('mouseover', '.zoomable-tile', function (evt) {

        $(this).css({'transform': 'scale(2.0)',
                     'z-index':1000});

      });
    $('.display-panel')
      .on('mouseout', '.zoomable-tile', function (evt) {

        $(this).css({'transform': 'scale(1.0)',
                     'z-index':0});

      });
    $('.display-panel')
      .on('mousemove', '.zoomable-tile', function (evt) {

        $(this).css({
          'transform-origin': ((evt.pageX - $(this).offset().left) /
                               $(this).width()) * 100 + '% ' +
            ((evt.pageY - $(this).offset().top) / $(this).height())
            * 100 +'%'
        });

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
      if (!cpv.readonlymode) {
        $('#varcheck').val(1);
      }
    });

    // alt+shift+u: object is not variable
    Mousetrap.bind('alt+shift+n', function() {
      if (!cpv.readonlymode) {
        $('#varcheck').val(2);
      }
    });

    // alt+shift+m: object is maybe a variable
    Mousetrap.bind('alt+shift+m', function() {
      if (!cpv.readonlymode) {
        $('#varcheck').val(3);
      }
    });

    // alt+shift+u: unset variability flag
    Mousetrap.bind('alt+shift+u', function() {
      if (!cpv.readonlymode) {
        $('#varcheck').val(0);
      }
    });


    //////////
    // TABS //
    //////////

    // alt+shift+o: overview tab
    Mousetrap.bind('alt+shift+o', function() {
      $('#overview-tab').click();
    });

    // alt+shift+l: phased LCs tab
    Mousetrap.bind('alt+shift+l', function() {
      $('#phasedlcs-tab').click();
    });

    // alt+shift+x: cross-matches tab
    Mousetrap.bind('alt+shift+x', function() {
      $('#xmatches-tab').click();
    });

    // alt+shift+p: period-search tab
    Mousetrap.bind('alt+shift+p', function() {
      $('#periodsearch-tab').click();
    });

    //////////////
    // MOVEMENT //
    //////////////

    // ctrl+right: save this, move to next checkplot
    Mousetrap.bind(['ctrl+right','alt+shift+right'], function() {
      $('.checkplot-next').click();
    });

    // ctrl+left: save this, move to prev checkplot
    Mousetrap.bind(['ctrl+left','alt+shift+left'], function() {
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
      if (!cpv.readonlymode) {
        cpv.save_checkplot(undefined, undefined, true);
      }
    });


    // ctrl+down: move to the next phased LC and set it as the best
    Mousetrap.bind(['ctrl+shift+down'], function() {

      if (!cpv.readonlymode) {

        // check the current phased index, if it's null, then set it to
        // 0
        if (cpv.currphasedind == null) {
          cpv.currphasedind = 0;
        }
        else if (cpv.currphasedind < cpv.maxphasedind) {
          cpv.currphasedind = cpv.currphasedind + 1;
        }

        var targetelem = $('a[data-phasedind="' +
                           cpv.currphasedind + '"]');

        if (targetelem.length > 0) {

          // scroll into view if the bottom of this plot is off the
          // screen
          if ( (targetelem.offset().top + targetelem.height()) >
               $(window).height() ) {
            targetelem[0].scrollIntoView(true);
          }

          // click on the target elem to select it
          targetelem.click();

        }
      }

    });

    // ctrl+up: move to the prev phased LC and set it as the best
    Mousetrap.bind(['ctrl+shift+up'], function() {

      if (!cpv.readonlymode) {

        // check the current phased index, if it's null, then set it to
        // 0
        if (cpv.currphasedind == null) {
          cpv.currphasedind = 0;
        }
        else if (cpv.currphasedind > 0) {
          cpv.currphasedind = cpv.currphasedind - 1;
        }

        var targetelem = $('a[data-phasedind="' +
                           cpv.currphasedind + '"]');

        if (targetelem.length > 0) {

          // scroll into view if the top of this plot is off the
          // screen
          if ( (targetelem.offset().top) > $(window).height() ) {
            targetelem[0].scrollIntoView(true);
          }

          // click on the target elem to select it
          targetelem.click();

        }
      }

    });

    // ctrl+backspace: clear variability tags
    Mousetrap.bind('ctrl+backspace', function() {

      if (!cpv.readonlymode) {

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

      }

    });

    // ctrl+shift+backspace: clear all info
    Mousetrap.bind('ctrl+shift+backspace', function() {

      if (!cpv.readonlymode) {

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

      }
    });


    ///////////////////////
    // TAGGING VARIABLES //
    ///////////////////////

    // ctrl+shift+1: planet candidate
    Mousetrap.bind('ctrl+shift+1', function () {

      if (!cpv.readonlymode) {

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

      }

    });

    // ctrl+shift+2: RRab pulsator
    Mousetrap.bind('ctrl+shift+2', function () {

      if (!cpv.readonlymode) {

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
        if (vartags.indexOf('RR Lyrae pulsator') == -1) {
          vartags.push('RR Lyrae pulsator');
          vartags = vartags.join(', ');
          $('#vartags').val(vartags);
        }

      }

    });

    // ctrl+shift+3: RRc pulsator
    Mousetrap.bind('ctrl+shift+3', function () {

      if (!cpv.readonlymode) {

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
        if (vartags.indexOf('Cepheid pulsator') == -1) {
          vartags.push('Cepheid pulsator');
          vartags = vartags.join(', ');
          $('#vartags').val(vartags);
        }

      }

    });

    // ctrl+shift+4: starspot rotation
    Mousetrap.bind('ctrl+shift+4', function () {

      if (!cpv.readonlymode) {

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

      }

    });

    // ctrl+shift+5: flare star
    Mousetrap.bind('ctrl+shift+5', function () {

      if (!cpv.readonlymode) {

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

      }

    });

    // ctrl+shift+6: contact EB
    Mousetrap.bind('ctrl+shift+6', function () {

      if (!cpv.readonlymode) {

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

      }

    });

    // ctrl+shift+7: semi-detached EB
    Mousetrap.bind('ctrl+shift+7', function () {

      if (!cpv.readonlymode) {

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

      }

    });

    // ctrl+shift+8: detached EB
    Mousetrap.bind('ctrl+shift+8', function () {

      if (!cpv.readonlymode) {

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

      }

    });

    // ctrl+shift+9: weird variability
    Mousetrap.bind('ctrl+shift+9', function () {

      if (!cpv.readonlymode) {

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

      }

    });

    // ctrl+shift+0: period harmonic
    Mousetrap.bind('ctrl+shift+0', function () {

      if (!cpv.readonlymode) {

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

      }

    });


    //////////////////////////
    // TAGGING OBJECT TYPES //
    //////////////////////////

    // alt+shift+1: white dwarf
    Mousetrap.bind(['alt+shift+1','command+shift+1'], function () {

      if (!cpv.readonlymode) {

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

      }

    });

    // alt+shift+2: hot star (OB)
    Mousetrap.bind(['alt+shift+2','command+shift+2'], function () {

      if (!cpv.readonlymode) {

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

      }

    });

    // alt+shift+3: A star
    Mousetrap.bind(['alt+shift+3','command+shift+3'], function () {

      if (!cpv.readonlymode) {

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

      }

    });

    // alt+shift+4: F or G dwarf
    Mousetrap.bind(['alt+shift+4','command+shift+4'], function () {

      if (!cpv.readonlymode) {

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

      }

    });

    // alt+shift+5: red giant
    Mousetrap.bind(['alt+shift+5','command+shift+5'], function () {

      if (!cpv.readonlymode) {

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

      }

    });

    // alt+shift+6: K or M dwarf
    Mousetrap.bind(['alt+shift+6','command+shift+6'], function () {

      if (!cpv.readonlymode) {

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

      }

    });

    // alt+shift+7: giant star
    Mousetrap.bind(['alt+shift+7','command+shift+7'], function () {

      if (!cpv.readonlymode) {

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

      }

    });

    // alt+shift+8: dwarf star
    Mousetrap.bind(['alt+shift+8','command+shift+8'], function () {

      if (!cpv.readonlymode) {

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

      }

    });

    // alt+shift+9: blended with neighbors
    Mousetrap.bind(['alt+shift+9','command+shift+9'], function () {

      if (!cpv.readonlymode) {

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

      }

    });

    // alt+shift+0: weird object
    Mousetrap.bind(['alt+shift+0','command+shift+0'], function () {

      if (!cpv.readonlymode) {

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

    // don't do anything if we're in readonly mode
    if (cpv.readonlymode) {
      return null;
    }

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

    if ( (lspmethod == 'gls') || (lspmethod == 'bls') ||
         (lspmethod == 'pdm') || (lspmethod == 'aov') ||
         (lspmethod == 'mav') || (lspmethod == 'acf') ||
         (lspmethod == 'win') ) {
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


    // get the specified sigma clip
    var sigclip = $('#psearch-sigclip').val();
    if (sigclip.length > 0) {

      // try to see if this is a composite sigma clip for lo, hi
      if (sigclip.indexOf(',') != -1) {

        sigclip = sigclip.trim().split(',');
        sigclip = [sigclip[0].trim(), sigclip[1].trim()];
        sigclip = [parseFloat(sigclip[0]), parseFloat(sigclip[1])];

        if (isNaN(sigclip[0]) || isNaN(sigclip[1]) ||
            (sigclip[0] < 0) || (sigclip[1] < 0)) {
          messages.push("one or both sigclip values is invalid");
          proceed = false;
        }
      }

      // if a single sigma clip, parse it
      else {

        sigclip = parseFloat(sigclip.trim());

        if (isNaN(sigclip) || sigclip < 0) {
          messages.push("sigclip value is invalid");
          proceed = false;
        }

        else {
          sigclip = [sigclip, sigclip];
        }

      }

    }

    else {
      sigclip = null;
    }

    // check the number of peaks to retrieve
    var nbestpeaks = $('#psearch-nbestpeaks').val();
    if (nbestpeaks.length > 0) {

      nbestpeaks = parseInt(nbestpeaks.trim());

      if (isNaN(nbestpeaks) || nbestpeaks < 0) {
        messages.push("nbestpeaks value is invalid");
        proceed = false;
      }

    }

    else {
      nbestpeaks = 10;
    }

    // get the lctimefilters and lcmagfilters
    // these will be processed server-side so nothing is required here
    var lctimefilters = $('#psearch-timefilters').val();
    if (lctimefilters.length == 0) {
      lctimefilters = null;
    }

    var lcmagfilters = $('#psearch-magfilters').val();
    if (lcmagfilters.length == 0) {
      lcmagfilters = null;
    }

    var extraparams = {};

    // finally, get the extra parameters for this periodogram method
    $('.pf-extraparam').each(function (index) {
      extraparams[$(this).attr('name')] = $(this).val();
    });

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
      var ajaxurl = cpv.CPSERVER_BASEURL + 'tools/' +
          encodeURIComponent(cputils.b64_encode(cpv.currfile));

      if (autofreq) {

        if (lctooltocall == 'psearch-acf') {
          var sentdata = {
            objectid: currobjectid,
            lctool: lctooltocall,
            forcereload: true,
            magsarefluxes: magsarefluxes,
            autofreq: autofreq,
            sigclip: sigclip,
            maxacfpeaks: nbestpeaks,
            lctimefilters: lctimefilters,
            lcmagfilters: lcmagfilters
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
            sigclip: sigclip,
            nbestpeaks: nbestpeaks,
            lctimefilters: lctimefilters,
            lcmagfilters: lcmagfilters
          };
        }

      }

      else {

        if (lctooltocall == 'psearch-acf') {
          // this is the data object to send to the backend
          var sentdata = {
            objectid: currobjectid,
            lctool: lctooltocall,
            forcereload: true,
            magsarefluxes: magsarefluxes,
            autofreq: autofreq,
            startp: startp,
            endp: endp,
            stepsize: freqstep,
            sigclip: sigclip,
            maxacfpeaks: nbestpeaks,
            lctimefilters: lctimefilters,
            lcmagfilters: lcmagfilters
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
            stepsize: freqstep,
            sigclip: sigclip,
            nbestpeaks: nbestpeaks,
            lctimefilters: lctimefilters,
            lcmagfilters: lcmagfilters
          };
        }


      }

      // update the sentdata with the extraparams
      var ei = 0;
      var ep = Object.getOwnPropertyNames(extraparams);

      for (ei; ei < ep.length; ei++) {
        sentdata[ep[ei]] = extraparams[ep[ei]];
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
            // with all best peaks
            $('#psearch-pgrampeaks').empty();

            //
            // first, get the first 3 peaks
            //

            $('#psearch-pgrampeaks').append(
              '<option value="0|' + lsp.phasedlc0.period +
                '|' + lsp.phasedlc0.epoch +
                '" selected>peak 1: ' +
                math.format(lsp.phasedlc0.period, 7) +
                '</option>'
            );

            if (lsp.hasOwnProperty('phasedlc1')) {

              $('#psearch-pgrampeaks').append(
                '<option value="1|' + lsp.phasedlc1.period +
                  '|' + lsp.phasedlc1.epoch +
                  '">peak 2: ' +
                  math.format(lsp.phasedlc1.period, 7) +
                  '</option>'
              );
            }

            if (lsp.hasOwnProperty('phasedlc2')) {

              $('#psearch-pgrampeaks').append(
                '<option value="2|' + lsp.phasedlc2.period +
                  '|' + lsp.phasedlc2.epoch +
                  '">peak 3: ' +
                  math.format(lsp.phasedlc2.period, 7) +
                  '</option>'
              );

            }

            //
            // then get any more peaks if remaining
            //
            if (lsp.nbestperiods.length > 3) {

              var peakind = 3;
              for (peakind;
                   peakind < lsp.nbestperiods.length;
                   peakind++) {

                $('#psearch-pgrampeaks').append(
                  '<option value="' + peakind +
                    '|' + lsp.nbestperiods[peakind] +
                    '|' + 'auto-minimumlight' +
                    '">peak ' + (peakind + 1) + ': ' +
                    math.format(
                      lsp.nbestperiods[peakind], 7) +
                    '</option>'
                );

              }

            }

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

    // don't do anything if we're in readonly mode
    if (cpv.readonlymode) {
      return null;
    }

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
    var ajaxurl = cpv.CPSERVER_BASEURL + 'tools/' +
        encodeURIComponent(cputils.b64_encode(cpv.currfile));

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

    // don't do anything if we're in readonly mode
    if (cpv.readonlymode) {
      return null;
    }

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
    if (periodind === null || periodind == 'none') {
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
      messages.push(
        "plot epoch not provided or invalid, set to automatic"
      )
      proceed = true;
      plotepoch = null;
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

    // get the specified sigma clip
    var sigclip = $('#psearch-sigclip').val();
    if (sigclip.length > 0) {

      // try to see if this is a composite sigma clip for lo, hi
      if (sigclip.indexOf(',') != -1) {

        sigclip = sigclip.trim().split(',');
        sigclip = [sigclip[0].trim(), sigclip[1].trim()];
        sigclip = [parseFloat(sigclip[0]), parseFloat(sigclip[1])];

        if (isNaN(sigclip[0]) || isNaN(sigclip[1]) ||
            (sigclip[0] < 0) || (sigclip[1] < 0)) {
          messages.push("one or both sigclip values is invalid");
          proceed = false;
        }
      }

      // if a single sigma clip, parse it
      else {

        sigclip = parseFloat(sigclip.trim());

        if (isNaN(sigclip) || sigclip < 0) {
          messages.push("sigclip value is invalid");
          proceed = false;
        }

        else {
          sigclip = [sigclip, sigclip];
        }

      }

    }

    else {
      sigclip = null;
    }

    // finally, get the lctimefilters and lcmagfilters
    // these will be processed server-side so nothing is required here
    var lctimefilters = $('#psearch-timefilters').val();
    if (lctimefilters.length == 0) {
      lctimefilters = null;
    }

    var lcmagfilters = $('#psearch-magfilters').val();
    if (lcmagfilters.length == 0) {
      lcmagfilters = null;
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
      var ajaxurl = cpv.CPSERVER_BASEURL + 'tools/' +
          encodeURIComponent(cputils.b64_encode(cpv.currfile));

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
        phasebin: phasebin,
        sigclip: sigclip,
        lctimefilters: lctimefilters,
        lcmagfilters: lcmagfilters
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

          var lsp = reqresult[lspmethod];

          // only update if the user is still on the object
          // we started with
          if (cpobjectid == currobjectid) {

            var lckey = 'phasedlc' + periodind;

            // put the phased LC for the best period into
            // #psearch-phasedlc-display
            cputils.b64_to_image(lsp[lckey]['plot'],
                                 '#psearch-phasedlc-display');

            // update the text box for epoch using the returned
            // value from the plotter
            $('#psearch-plotepoch').val(lsp[lckey]['epoch']);

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

    // don't do anything if we're in readonly mode
    if (cpv.readonlymode) {
      return null;
    }

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
    var ajaxurl = cpv.CPSERVER_BASEURL + 'tools/' +
        encodeURIComponent(cputils.b64_encode(cpv.currfile));

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

    // don't do anything if we're in readonly mode
    if (cpv.readonlymode) {
      return null;
    }

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
    var ajaxurl = cpv.CPSERVER_BASEURL + 'tools/' +
        encodeURIComponent(cputils.b64_encode(cpv.currfile));

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

    // don't do anything if we're in readonly mode
    if (cpv.readonlymode) {
      return null;
    }

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
    var ajaxurl = cpv.CPSERVER_BASEURL + 'tools/' +
        encodeURIComponent(cputils.b64_encode(cpv.currfile));

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

    // periodogram method select - change options as needed
    $('#psearch-lspmethod').on('change', function (evt) {

      var newval = $(this).val();

      // FIXME: update the psearch param panel for any special params for
      // this period-finder
      var extraparamelem = $('#psearch-pf-extraparams');
      extraparamelem.empty();

      if (newval == 'acf') {

        extraparamelem.append(
          '<div class="form-group">' +
            '<label for="psearch-smoothacf">' +
            'ACF smoothing parameter' +
            '</label>' +
            '<input type="text" name="smoothacf" ' +
            'class="form-control form-control-sm pf-extraparam" ' +
            'id="psearch-smoothacf" value="721"></div>'
        );

        extraparamelem.append(
          '<div class="form-group">' +
            '<label for="psearch-fillgaps">' +
            'Fill value for time-series gaps' +
            '</label>' +
            '<input type="text" name="fillgaps" ' +
            'class="form-control form-control-sm pf-extraparam" ' +
            'id="psearch-fillgaps" value="0.0"></div>'
        );


      }

      else if (newval == 'aov') {

        extraparamelem.append(
          '<div class="form-group">' +
            '<label for="psearch-phasebinsize">' +
            'Phase bin size' +
            '</label>' +
            '<input type="text" name="phasebinsize" ' +
            'class="form-control form-control-sm pf-extraparam" ' +
            'id="psearch-phasebinsize" value="0.05"></div>'
        );

        extraparamelem.append(
          '<div class="form-group">' +
            '<label for="psearch-mindetperbin">' +
            'Min observations per phase bin' +
            '</label>' +
            '<input type="text" name="mindetperbin" ' +
            'class="form-control form-control-sm pf-extraparam" ' +
            'id="psearch-mindetperbin" value="9"></div>'
        );

      }

      else if (newval == 'bls') {

        extraparamelem.append(
          '<div class="form-group">' +
            '<label for="psearch-mintransitduration">' +
            'Min transit duration [fractional phase]' +
            '</label>' +
            '<input type="text" name="mintransitduration" ' +
            'class="form-control form-control-sm pf-extraparam" ' +
            'id="psearch-mintransitduration" value="0.01"></div>'
        );

        extraparamelem.append(
          '<div class="form-group">' +
            '<label for="psearch-maxtransitduration">' +
            'Max transit duration [fractional phase]' +
            '</label>' +
            '<input type="text" name="maxtransitduration" ' +
            'class="form-control form-control-sm pf-extraparam" ' +
            'id="psearch-maxtransitduration" value="0.8"></div>'
        );

      }

      else if (newval == 'gls') {

      }

      else if (newval == 'mav') {

        extraparamelem.append(
          '<div class="form-group">' +
            '<label for="psearch-nharmonics">' +
            'Number of harmonics' +
            '</label>' +
            '<input type="text" name="nharmonics" ' +
            'class="form-control form-control-sm pf-extraparam" ' +
            'id="psearch-nharmonics" value="6"></div>'
        );

      }

      else if (newval == 'pdm') {

        extraparamelem.append(
          '<div class="form-group">' +
            '<label for="psearch-phasebinsize">' +
            'Phase bin size' +
            '</label>' +
            '<input type="text" name="phasebinsize" ' +
            'class="form-control form-control-sm pf-extraparam" ' +
            'id="psearch-phasebinsize" value="0.05"></div>'
        );

        extraparamelem.append(
          '<div class="form-group">' +
            '<label for="psearch-mindetperbin">' +
            'Min observations per phase bin' +
            '</label>' +
            '<input type="text" name="mindetperbin" ' +
            'class="form-control form-control-sm pf-extraparam" ' +
            'id="psearch-mindetperbin" value="9"></div>'
        );

      }

    });

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


    // periodogram search - add half period to epoch
    $('#psearch-addhalfp-epoch').on('click', function (evt) {

      evt.preventDefault();
      var plotperiod = parseFloat($('#psearch-plotperiod').val());
      var plotepoch = parseFloat($('#psearch-plotepoch').val());
      if (!isNaN(plotperiod) && !isNaN(plotepoch)) {
        $('#psearch-plotepoch').val(plotepoch + plotperiod/2.0);
      }

    });

    // periodogram search - subtract half period to epoch
    $('#psearch-subhalfp-epoch').on('click', function (evt) {

      evt.preventDefault();
      var plotperiod = parseFloat($('#psearch-plotperiod').val());
      var plotepoch = parseFloat($('#psearch-plotepoch').val());
      if (!isNaN(plotperiod) && !isNaN(plotepoch)) {
        $('#psearch-plotepoch').val(plotepoch - plotperiod/2.0);
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
