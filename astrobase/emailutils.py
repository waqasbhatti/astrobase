#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''emailutils.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jun 2013

License: MIT. See LICENSE.txt for complete text.

This is a small utility module to send email using an SMTP server that requires
logins. The email settings are stored in a file called .emailsettings that
should be located in the same directory as emailutils.py. This file should have
permissions 600 (so only you can read/write to it), and should contain the
following info in a single row, separated by the | character

<email user>|<email password>|<email server>

Example:

exampleuser@email.com|correcthorsebatterystaple|mail.example.com

NOTE:

This assumes the email server uses STARTTLS encryption and listens on SMTP port
587. Most email servers support this.

'''

from email.mime.text import MIMEText
from email.utils import formatdate, make_msgid
import smtplib
import socket

import os.path
import sys
import os
import stat

import time
from datetime import datetime

import subprocess

## EMAIL SETTINGS ##

# get config from the astrobase.conf file
try:
    import ConfigParser
except:
    import configparser as ConfigParser

modpath = os.path.abspath(os.path.dirname(__file__))
CONF_FILE = os.path.join(modpath,'astrobase.conf')
CONF = ConfigParser.ConfigParser()
CONF.read(CONF_FILE)

# first, check if the .emailsettings file exists and has permissions 0600
SETTINGSFILE = os.path.join(modpath, CONF.get('email','credentials'))

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
    raise IOError('the email settings file (%s does not exist!' % SETTINGSFILE)


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
               email_address_list):
    '''
    This sends an email to addresses, informing them about events.

    email_recipient_list is of the form:

    ['Example Person', ...]

    email_address_list is of the form:

    ['exampleperson@example.com', ...]

    '''
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
        server_ehlo_response = server.ehlo()

        if server.has_extn('STARTTLS'):

            try:

                tls_start_response = server.starttls()
                tls_ehlo_response = server.ehlo()

                login_response = server.login(EMAIL_USER, EMAIL_PASSWORD)

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
                quit_response = server.quit()
                return True

            else:

                quit_response = server.quit()
                return False

        else:

            print('email server does not support STARTTLS,'
                  ' bailing out...')
            quit_response = server.quit()
            return False

    except Exception as e:
        print('sending email failed with error: %s' % e)
        returnval =  False


    quit_response = server.quit()
    return returnval
