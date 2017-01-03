#!/usr/bin/env python

'''
lcdb.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - 05/13
License: MIT - see LICENSE for the full text.

Serves as a lightweight PostgreSQL DB interface for other modules in this
project.

'''

import logging
try:
    import ConfigParser
except:
    import configparser as ConfigParser
import os
import stat
import os.path
import hashlib
from datetime import datetime
from traceback import format_exc

#############
## LOGGING ##
#############

# setup a logger
LOGGER = None

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.lcdb' % parent_name)

def LOGDEBUG(message):
    if LOGGER:
        LOGGER.debug(message)
    elif DEBUG:
        print('%sZ [DBUG]: %s' % (datetime.utcnow().isoformat(), message))

def LOGINFO(message):
    if LOGGER:
        LOGGER.info(message)
    else:
        print('%sZ [INFO]: %s' % (datetime.utcnow().isoformat(), message))

def LOGERROR(message):
    if LOGGER:
        LOGGER.error(message)
    else:
        print('%sZ [ERR!]: %s' % (datetime.utcnow().isoformat(), message))

def LOGWARNING(message):
    if LOGGER:
        LOGGER.warning(message)
    else:
        print('%sZ [WRN!]: %s' % (datetime.utcnow().isoformat(), message))

def LOGEXCEPTION(message):
    if LOGGER:
        LOGGER.exception(message)
    else:
        print(
            '%sZ [EXC!]: %s\nexception was: %s' % (
                datetime.utcnow().isoformat(),
                message, format_exc()
                )
            )

#############################
## SEE IF WE HAVE PSYCOPG2 ##
#############################
try:

    import psycopg2 as pg
    import psycopg2.extras

except:

    LOGEXCEPTION('psycopg2 is not available for import. '
                 'Please install it to use this module.\n'
                 'You may have to get development packages for libpq '
                 '(lipgq-dev, postgresql-devel, etc.) to compile '
                 'psycopg2 successfully.')
    raise


############
## CONFIG ##
############

# parse the configuration file to get the default database credentials
# NOTE:this is relative to the current module's path
# get the current module's path

modpath = os.path.abspath(os.path.dirname(__file__))
CONF_FILE = os.path.join(modpath,'astrobase.conf')

try:

    HAVECONF = False

    CONF = ConfigParser.ConfigParser()
    CONF.read(CONF_FILE)

    LOGINFO('using database config in %s' % os.path.abspath(CONF_FILE))

    # database config
    DBCREDENTIALS = os.path.join(modpath, CONF.get('lcdb','credentials'))

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

except:

    LOGEXCEPTION("no configuration file "
               "found for this module in %s, "
               "the LCDB object's open_default() function won't work" %
               modpath)
    HAVECONF = False


class LCDB(object):
    '''
    This is an object serving as an interface to the postgres DB. It implements
    the following methods:

    LCDB.open(database,
              user,
              password) -> open a new connection to postgres using the provided
                           credentials

    LCDB.cursor(handle,
                dictcursor=<True/False>) -> return a postgres DB cursor with the
                                            handle specified. if the handle does
                                            not exist, a new one will be created
                                            for the cursor. if dictcursor is
                                            True, will return a cursor allowing
                                            addressing columns as a dictionary

    LCDB.commit() -> commits any pending transactions to the DB

    LCDB.close_cursor(handle) -> closes cursor specified in handle

    LCDB.close_connection() -> close all cursors currently active, and then
                               close the connection to the database

    LCDB's main purpose is to avoid creating new postgres connections throughout
    the course of lcserver's work; these are relatively expensive. Instead, we
    get new cursors when needed, and then pass these around as needed. The
    connection also remains open for the whole lifetime of the lcserver's
    runtime, keeping things simple.

    '''

    def __init__(self,
                 database=None,
                 user=None,
                 password=None,
                 host=None):

        self.connection = None
        self.user = None
        self.database = None
        self.host = None
        self.cursors = {}

        if database and user and password and host:
            self.open(database, user, password, host)



    def open(self, database, user, password, host):
        '''
        This just creates the database connection and stores it in
        self.connection.

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

        except Exception as e:

            LOGEXCEPTION('postgres connection failed, '
                         'using DB %s, user %s' % (database,
                                                   user))

            self.database = None
            self.user = None



    def open_default(self):
        '''
        This opens the database connection using the default database parameters
        given in the lcserver.conf file.

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
        '''
        This gets or creates a DB cursor for the current DB connection.

        dictcursor = True -> use a cursor where each returned row can be
                             addressed as a dictionary by column name

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

        dictcursor = True -> use a cursor where each returned row can be
                             addressed as a dictionary by column name

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
        Closes the cursor specified and removes it from the self.cursors
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
