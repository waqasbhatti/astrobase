// classyweb.js - Waqas Bhatti (wbhatti@astro.princeton.edu) - Nov 2016
// License: MIT. See LICENSE for the full text.

// This contains JS for the classy variable star interface.

// websock message handler functions
var msghandlers = {

    handle_glsp_message: function (msgcontents) {

    },

    handle_lcplot_message: function (msgcontents) {

    },

    handle_objectinfo_message: function (msgcontents) {

    },

    handle_objectstamp_message: function (msgcontents) {

    },

    handle_classyweb_message: function (msgcontents) {

    }

};


// websock message router function
var msgrouter = {

    // this figures out where to send backend messages
    backend_message_router: function (websock_message) {

        var json_msg = JSON.parse(websock_message);

        var msg_sender = json_msg.sender;
        var msg_status = json_msg.status;
        var msg_contents = json_msg.contents;

        switch (msg_sender) {

        case 'worker-glsp':
            msghandlers.handle_glsp_message(msg_contents);
            break;

        case 'worker-lcplot':
            msghandlers.handle_lcplot_message(msg_contents);
            break;

        case 'worker-objectinfo':
            msghandlers.handle_objectinfo_message(msg_contents);
            break;

        case 'worker-objectstamp':
            msghandlers.handle_objectstamp_message(msg_contents);
            break;

        case 'classyweb':
            msghandlers.handle_classyweb_message(msg_contents);
            break;

        }

    }

};

// functions for setting up the UI of the frontend
var classyui = {

    // this sets up all actions
    action_setup: function (websock) {

    },


    // this just returns a Bootstrap v4 alert message
    alert_message: function(alerttext, alerttype) {

        var msgbox =
            '<div class="alert alert-' + alerttype + ' ' +
            'alert-dismissible fade in" ' +
            'role="alert">' +
            '<button type="button" class="close" data-dismiss="alert" ' +
            'aria-label="Close">' +
            '<span aria-hidden="true">&times;</span>' +
            '</button>' + alerttext + '</div>';

        return msgbox

    }

};
