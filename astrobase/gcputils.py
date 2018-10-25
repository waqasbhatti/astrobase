#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""gcputils.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2018
License: MIT - see the LICENSE file for the full text.

This contains useful functions to set up Google Cloud Platform services for use
with lcproc_gcp.py.

"""

#############
## LOGGING ##
#############

import logging
from datetime import datetime
from traceback import format_exc

# setup a logger
LOGGER = None
LOGMOD = __name__
DEBUG = False

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.%s' % (parent_name, LOGMOD))

def LOGDEBUG(message):
    if LOGGER:
        LOGGER.debug(message)
    elif DEBUG:
        print('[%s - DBUG] %s' % (
            datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            message)
        )

def LOGINFO(message):
    if LOGGER:
        LOGGER.info(message)
    else:
        print('[%s - INFO] %s' % (
            datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            message)
        )

def LOGERROR(message):
    if LOGGER:
        LOGGER.error(message)
    else:
        print('[%s - ERR!] %s' % (
            datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            message)
        )

def LOGWARNING(message):
    if LOGGER:
        LOGGER.warning(message)
    else:
        print('[%s - WRN!] %s' % (
            datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            message)
        )

def LOGEXCEPTION(message):
    if LOGGER:
        LOGGER.exception(message)
    else:
        print(
            '[%s - EXC!] %s\nexception was: %s' % (
                datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
                message, format_exc()
            )
        )


#############
## IMPORTS ##
#############

import os.path
import os
import json
import time
from datetime import timedelta

try:

    from apiclient.discovery import build
    from google.cloud import pubsub
    from google.cloud import storage
    import paramiko

except ImportError as e:
    raise ImportError(
        "This module requires the following packages from PyPI:\n\n "
        "paramiko google-api-python-client "
        "google-cloud-storage google-cloud-pubsub\n\n"
        "You'll also need the gcloud utility to set up service roles "
        "and API keys for Google Cloud Platform before using anything here.\n\n"
        "https://cloud.google.com/sdk/docs/quickstarts"
    )


###################
## GCE INSTANCES ##
###################

# variables:
# - instance_name
# - project_name
# - zone_name
# - instance_type
# - startup_script_text
# - shutdown_script_text
# - service_account_email

GCE_INSTANCE_TEMPLATE = {
    'kind': 'compute#instance',
    'name': '{instance_name}',
    'zone': 'projects/{project_name}/zones/{zone_name}',
    'machineType': ('projects/{project_name}/zones/'
                    '{zone_name}/machineTypes/{instance_type}'),
    'metadata': {
        'kind': 'compute#metadata',
        'items': [
            {'key':'startup-script',
             'value':'{startup_script_text}'},
            {'key':'shutdown-script',
             'value':'{shutdown_script_text}'}
        ]
    },
    'tags': {
        'items': []
    },
    'disks': [
        {'kind': 'compute#attachedDisk',
         'type': 'PERSISTENT',
         'boot': True,
         'mode': 'READ_WRITE',
         'autoDelete': True,
         'deviceName': 'instance-1',
         'initializeParams': {
             'sourceImage': ('projects/debian-cloud/global/'
                             'images/debian-9-stretch-v20181011'),
             'diskType': ('projects/{project_name}/zones/'
                          '{zone_name}/diskTypes/pd-standard'),
             'diskSizeGb': '10'
         },
         'diskEncryptionKey': {}}
    ],
    'canIpForward': False,
    'networkInterfaces': [
        {'kind': 'compute#networkInterface',
         'subnetwork': ('projects/{project_name}/regions'
                        '/{zone_name}/subnetworks/default'),
         'accessConfigs': [{'kind': 'compute#accessConfig',
                            'name': 'External NAT',
                            'type': 'ONE_TO_ONE_NAT',
                            'networkTier': 'PREMIUM'}],
         'aliasIpRanges': []}
    ],
    'description': '',
    'labels': {},
    'scheduling': {
        'preemptible': False,
        'onHostMaintenance': 'TERMINATE',
        'automaticRestart': False,
        'nodeAffinities': []
    },
    'deletionProtection': False,
    'serviceAccounts': [
        {'email': '{service_account_email}',
         'scopes': [
             'https://www.googleapis.com/auth/pubsub',
             'https://www.googleapis.com/auth/servicecontrol',
             'https://www.googleapis.com/auth/service.management.readonly',
             'https://www.googleapis.com/auth/logging.write',
             'https://www.googleapis.com/auth/monitoring.write',
             'https://www.googleapis.com/auth/trace.append',
             'https://www.googleapis.com/auth/devstorage.read_write'
         ]}
    ]
}

def make_gce_instances():
    """This makes new GCE worker nodes.

    Use preemptible instances and startup/shutdown scripts to emulate AWS spot
    fleet behavior and run stuff at cheaper prices.

    """



def delete_gce_instances():
    """
    This deletes GCE worker nodes.

    """


################
## GCS CLIENT ##
################

def gcs_get_file():
    """This gets a single file from a Google Cloud Storage bucket.

    """


def gcs_get_url():
    """This gets a single file from a Google Cloud Storage bucket.

    This uses the gs:// URL instead of a bucket name and key.

    """

def gcs_put_file():
    """This puts a single file into a Google Cloud Storage bucket.

    """



###################
## PUBSUB CLIENT ##
###################

def gps_create_topic():
    """
    This creates a Google Pub/Sub topic.

    """



def gps_delete_topic():
    """
    This deletes a Google Pub/Sub topic.

    """


def gps_topic_pull():
    """
    This pulls a single message from a pubsub topic.

    """


def gps_topic_publish():
    """
    This publishes a JSON message to a topic.

    """
