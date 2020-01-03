#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# emailutils.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jun 2013
# License: MIT. See LICENSE.txt for complete text.

'''This is a small utility module to send email using an SMTP server that
requires logins. The email settings are stored in a file called .emailsettings
that should be located in the ~/.astrobase/ directory in your home
directory. This file should have permissions 0600 (so only you can read/write to
it), and should contain the following info in a single row, separated by the |
character::

        <email user>|<email password>|<email server>

Example::

        exampleuser@email.com|correcthorsebatterystaple|mail.example.com

NOTE: This assumes the email server uses STARTTLS encryption and listens on SMTP
port 587. Most email servers support this.

'''

from email.mime.text import MIMEText
from email.utils import formatdate, make_msgid
import smtplib
import socket

import os.path
import os
import stat

import time
from datetime import datetime

## EMAIL SETTINGS ##

# get config from the astrobase.conf file
import configparser

# parse the configuration file to get the default database credentials
CONF_FILE = os.path.abspath(os.path.expanduser('~/.astrobase/astrobase.conf'))

if not os.path.exists(CONF_FILE):
    # make the ~/.astrobase directory and copy over the astrobase.conf file to
    # it.
    import shutil

    # make the ~/.astrobase directory if it doesn't exist
    confpath = os.path.expanduser('~/.astrobase')
    if not os.path.exists(confpath):
        os.makedirs(confpath)
    modpath = os.path.dirname(os.path.abspath(__file__))

    # copy over the astrobase.conf file to ~/.astrobase if it doesn't exist
    if not os.path.exists(os.path.join(confpath,'astrobase.conf')):
        shutil.copy(os.path.join(modpath,'astrobase.conf'),
                    confpath)


CONF_FILE = os.path.abspath(
    os.path.expanduser('~/.astrobase/astrobase.conf')
)
CONF = configparser.ConfigParser()
CONF.read(CONF_FILE)

#
# Now, check the conf file for email credentials
#

# first, check if the .emailsettings file exists and has permissions 0600
SETTINGSFILE = os.path.join(os.path.expanduser('~/.astrobase'),
                            CONF.get('email','credentials'))

if os.path.exists(SETTINGSFILE):

    # check if this file is readable/writeable by user only
    fileperm = oct(os.stat(SETTINGSFILE)[stat.ST_MODE])

    # if we're good, read the settings
    if fileperm == '0100600' or fileperm == '0o100600':
        EMAIL_USER, EMAIL_PASSWORD, EMAIL_SERVER = open(
            SETTINGSFILE
        ).read().strip('\n').split('|')

    else:
        raise OSError('the email settings file %s has bad permissions '
                      '(you need to chmod 600 this file) and '
                      'is insecure, not reading...'
                      % SETTINGSFILE)
else:
    print('the email settings file: %s does not exist!' % SETTINGSFILE)
    EMAIL_USER, EMAIL_PASSWORD, EMAIL_SERVER = None, None, None

EMAIL_TEMPLATE = '''\
This is an automated notification from {sender} on {hostname}.

Time: {activity_time}

Report:

{activity_report}

'''


def send_email(sender,
               subject,
               content,
               email_recipient_list,
               email_address_list,
               email_user=None,
               email_pass=None,
               email_server=None):
    '''This sends an email to addresses, informing them about events.

    The email account settings are retrieved from the settings file as described
    above.

    Parameters
    ----------

    sender : str
        The name of the sender to use in the email header.

    subject : str
        Subject of the email.

    content : str
        Content of the email.

    email_recipient list : list of str
        This is a list of email recipient names of the form:
        `['Example Person 1', 'Example Person 1', ...]`

    email_recipient list : list of str
        This is a list of email recipient addresses of the form:
        `['example1@example.com', 'example2@example.org', ...]`

    email_user : str
        The username of the email server account that will send the emails. If
        this is None, the value of EMAIL_USER from the
        ~/.astrobase/.emailsettings file will be used. If that is None as well,
        this function won't work.

    email_pass : str
        The password of the email server account that will send the emails. If
        this is None, the value of EMAIL_PASS from the
        ~/.astrobase/.emailsettings file will be used. If that is None as well,
        this function won't work.

    email_server : str
        The address of the email server that will send the emails. If this is
        None, the value of EMAIL_USER from the ~/.astrobase/.emailsettings file
        will be used. If that is None as well, this function won't work.

    Returns
    -------

    bool
        True if email sending succeeded. False if email sending failed.

    '''

    if not email_user:
        email_user = EMAIL_USER

    if not email_pass:
        email_pass = EMAIL_PASSWORD

    if not email_server:
        email_server = EMAIL_SERVER

    if not email_server and email_user and email_pass:
        raise ValueError("no email server address and "
                         "credentials available, can't continue")

    msg_text = EMAIL_TEMPLATE.format(
        sender=sender,
        hostname=socket.gethostname(),
        activity_time='%sZ' % datetime.utcnow().isoformat(),
        activity_report=content
    )

    email_sender = '%s <%s>' % (sender, EMAIL_USER)

    # put together the recipient and email lists
    email_recipients = [('%s <%s>' % (x,y))
                        for (x,y) in zip(email_recipient_list,
                                         email_address_list)]

    # put together the rest of the message
    email_msg = MIMEText(msg_text)
    email_msg['From'] = email_sender
    email_msg['To'] = ', '.join(email_recipients)
    email_msg['Message-Id'] = make_msgid()
    email_msg['Subject'] = '[%s on %s] %s' % (
        sender,
        socket.gethostname(),
        subject
    )
    email_msg['Date'] = formatdate(time.time())

    # start the email process

    try:
        server = smtplib.SMTP(EMAIL_SERVER, 587)
        server.ehlo()

        if server.has_extn('STARTTLS'):

            try:

                server.starttls()
                server.ehlo()

                server.login(EMAIL_USER, EMAIL_PASSWORD)

                send_response = (
                    server.sendmail(email_sender,
                                    email_address_list,
                                    email_msg.as_string())
                )

            except Exception as e:

                print('script email sending failed with error: %s'
                      % e)
                send_response = None

            if send_response is not None:

                print('script email sent successfully')
                server.quit()
                return True

            else:

                server.quit()
                return False

        else:

            print('email server does not support STARTTLS,'
                  ' bailing out...')
            server.quit()
            return False

    except Exception as e:
        print('sending email failed with error: %s' % e)
        returnval = False

    server.quit()
    return returnval
