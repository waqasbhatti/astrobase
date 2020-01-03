#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# lcdb.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - 05/13
# License: MIT - see LICENSE for the full text.

'''
Serves as a lightweight PostgreSQL DB interface for other modules in this
project.

'''

#############
## LOGGING ##
#############

import logging
from astrobase import log_sub, log_fmt, log_date_fmt

DEBUG = False
if DEBUG:
    level = logging.DEBUG
else:
    level = logging.INFO
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=level,
    style=log_sub,
    format=log_fmt,
    datefmt=log_date_fmt,
)

LOGDEBUG = LOGGER.debug
LOGINFO = LOGGER.info
LOGWARNING = LOGGER.warning
LOGERROR = LOGGER.error
LOGEXCEPTION = LOGGER.exception


#############
## IMPORTS ##
#############

import os.path
import os
import stat
import hashlib
import configparser


#############################
## SEE IF WE HAVE PSYCOPG2 ##
#############################
try:

    import psycopg2 as pg
    import psycopg2.extras

except Exception:

    LOGEXCEPTION('psycopg2 is not available for import. '
                 'Please install it to use this module.\n'
                 'You may have to get development packages for libpq '
                 '(lipgq-dev, postgresql-devel, etc.) to compile '
                 'psycopg2 successfully. '
                 'Alternatively, install psycopg2-binary from PyPI')
    raise


############
## CONFIG ##
############

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

try:

    HAVECONF = False

    CONF = configparser.ConfigParser()
    CONF.read(CONF_FILE)

    LOGINFO('using database config in %s' % os.path.abspath(CONF_FILE))

    # database config
    DBCREDENTIALS = os.path.join(os.path.expanduser('~/.astrobase'),
                                 CONF.get('lcdb','credentials'))

    # see if this file exists, read it in and get credentials
    if os.path.exists(DBCREDENTIALS):

        # check if this file is readable/writeable by user only
        fileperm = oct(os.stat(DBCREDENTIALS)[stat.ST_MODE])

        if fileperm == '0100600' or fileperm == '0o100600':

            with open(DBCREDENTIALS) as infd:
                creds = infd.read().strip('\n')
            DBHOST, DBPORT, DBDATA, DBUSER, DBPASS = creds.split(':')
            HAVECONF = True

        else:
            LOGWARNING('the lcdb settings file %s has bad permissions '
                       '(you need to chmod 600 this file) and is insecure, '
                       'not reading...' % DBCREDENTIALS)
            HAVECONF = False

    else:

        DBHOST = CONF.get('lcdb','host')
        DBPORT = CONF.get('lcdb','port')
        DBDATA = CONF.get('lcdb','database')
        DBUSER = CONF.get('lcdb','user')
        DBPASS = CONF.get('lcdb','password')

    if DBHOST and DBPORT and DBDATA and DBUSER and DBPASS:
        HAVECONF = True
    else:
        HAVECONF = False

except Exception:

    LOGEXCEPTION("no configuration file "
                 "found for this module in %s, "
                 "the LCDB object's open_default() function won't work" %
                 CONF_FILE)
    HAVECONF = False


class LCDB(object):
    '''This is an object serving as an interface to a PostgreSQL DB.

    LCDB's main purpose is to avoid creating new postgres connections for each
    query; these are relatively expensive. Instead, we get new cursors when
    needed, and then pass these around as needed.

    Attributes
    ----------

    database : str
        Name of the database to connect to.

    user : str
        User name of the database server user.

    password : str
        Password for the database server user.

    host : str
        Database hostname or IP address to connect to.

    connection : psycopg2.Connection object
        The underlying connection to the database.

    cursors : dict of psycopg2.Cursor objects
        The keys of this dict are random hash strings, the values of this dict
        are the actual `Cursor` objects.

    '''

    def __init__(self,
                 database=None,
                 user=None,
                 password=None,
                 host=None):
        '''Constructor for this class.

        Parameters
        ----------

        database : str
            Name of the database to connect to.

        user : str
            User name of the database server user.

        password : str
            Password for the database server user.

        host : str
            Database hostname or IP address to connect to.

        Returns
        -------

        `LCDB` object instance

        '''

        self.connection = None
        self.user = None
        self.database = None
        self.host = None
        self.cursors = {}

        if database and user and password and host:
            self.open(database, user, password, host)

    def open(self, database, user, password, host):
        '''This opens a new database connection.

        Parameters
        ----------

        database : str
            Name of the database to connect to.

        user : str
            User name of the database server user.

        password : str
            Password for the database server user.

        host : str
            Database hostname or IP address to connect to.

        '''

        try:

            self.connection = pg.connect(user=user,
                                         password=password,
                                         database=database,
                                         host=host)

            LOGINFO('postgres connection successfully '
                    'created, using DB %s, user %s' % (database,
                                                       user))

            self.database = database
            self.user = user

        except Exception:

            LOGEXCEPTION('postgres connection failed, '
                         'using DB %s, user %s' % (database,
                                                   user))

            self.database = None
            self.user = None

    def open_default(self):
        '''
        This opens the database connection using the default database parameters
        given in the ~/.astrobase/astrobase.conf file.

        '''

        if HAVECONF:
            self.open(DBDATA, DBUSER, DBPASS, DBHOST)
        else:
            LOGERROR("no default DB connection config found in lcdb.conf, "
                     "this function won't work otherwise")

    def autocommit(self):
        '''
        This sets the database connection to autocommit. Must be called before
        any cursors have been instantiated.

        '''

        if len(self.cursors.keys()) == 0:
            self.connection.autocommit = True
        else:
            raise AttributeError('database cursors are already active, '
                                 'cannot switch to autocommit now')

    def cursor(self, handle, dictcursor=False):
        '''This gets or creates a DB cursor for the current DB connection.

        Parameters
        ----------

        handle : str
            The name of the cursor to look up in the existing list or if it
            doesn't exist, the name to be used for a new cursor to be returned.

        dictcursor : bool
            If True, returns a cursor where each returned row can be addressed
            as a dictionary by column name.

        Returns
        -------

        psycopg2.Cursor instance

        '''

        if handle in self.cursors:

            return self.cursors[handle]

        else:
            if dictcursor:
                self.cursors[handle] = self.connection.cursor(
                    cursor_factory=psycopg2.extras.DictCursor
                )
            else:
                self.cursors[handle] = self.connection.cursor()

            return self.cursors[handle]

    def newcursor(self, dictcursor=False):
        '''
        This creates a DB cursor for the current DB connection using a
        randomly generated handle. Returns a tuple with cursor and handle.

        Parameters
        ----------

        dictcursor : bool
            If True, returns a cursor where each returned row can be addressed
            as a dictionary by column name.

        Returns
        -------

        tuple
            The tuple is of the form (handle, psycopg2.Cursor instance).

        '''

        handle = hashlib.sha256(os.urandom(12)).hexdigest()

        if dictcursor:
            self.cursors[handle] = self.connection.cursor(
                cursor_factory=psycopg2.extras.DictCursor
            )
        else:
            self.cursors[handle] = self.connection.cursor()

            return (self.cursors[handle], handle)

    def commit(self):
        '''
        This just calls the connection's commit method.

        '''
        if not self.connection.closed:
            self.connection.commit()
        else:
            raise AttributeError('postgres connection to %s is closed' %
                                 self.database)

    def rollback(self):
        '''
        This just calls the connection's commit method.

        '''
        if not self.connection.closed:
            self.connection.rollback()
        else:
            raise AttributeError('postgres connection to %s is closed' %
                                 self.database)

    def close_cursor(self, handle):
        '''
        Closes the cursor specified and removes it from the `self.cursors`
        dictionary.

        '''

        if handle in self.cursors:
            self.cursors[handle].close()
        else:
            raise KeyError('cursor with handle %s was not found' % handle)

    def close_connection(self):
        '''
        This closes all cursors currently in use, and then closes the DB
        connection.

        '''

        self.connection.close()
        LOGINFO('postgres connection closed for DB %s' % self.database)
