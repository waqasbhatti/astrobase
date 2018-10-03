#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''lcproc_aws.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2018
License: MIT - see the LICENSE file for the full text.

This contains lcproc worker loops useful for AWS processing of light curves.

The basic workflow is:

LCs from S3 -> SQS -> worker loop -> products back to S3 | result JSON to SQS

All functions here assume AWS credentials have been loaded already using awscli
as described at:

https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html

'''

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

try:

    import boto3
    import paramiko

except ImportError:
    raise ImportError(
        "This module requires the boto3 and paramiko packages from PyPI. "
        "You'll also need the awscli package to set up the "
        "AWS secret key config for this module."
    )

from astrobase import lcproc



########
## S3 ##
########

def s3_get_file(bucket, filename, local_file, client=None):
    '''
    This justs gets a file from S3.

    '''

    if not client:
        client = boto3.client('s3')

    try:
        client.download_file(bucket, filename, local_file)
        return local_file
    except Exception as e:
        LOGEXCEPTION('could not download s3://%s/%s' % (bucket, filename))
        return None



def s3_put_file(local_file, bucket, client=None):
    '''
    This uploads a file to S3.

    '''

    if not client:
        client = boto3.client('s3')

    try:
        client.upload_file(local_file, bucket, os.path.basename(local_file))
        return 's3://%s/%s' % (bucket, os.path.basename(local_file))
    except Exception as e:
        LOGEXCEPTION('could not upload %s to bucket: %s' % (local_file,
                                                            bucket))
        return None



def s3_delete_file(bucket, filename, client=None):
    '''
    This deletes a file from S3.

    '''

    if not client:
        client = boto3.client('s3')

    try:
        resp = client.delete_object(Bucket=bucket, Key=filename)
        if not resp:
            LOGERROR('could not delete file %s from bucket %s' % (filename,
                                                                  bucket))
        else:
            return resp['DeleteMarker']
    except Exception as e:
        LOGEXCEPTION('could not delete file %s from bucket %s' % (filename,
                                                                  bucket))
        return None



#########
## SQS ##
#########

def sqs_create_queue(queue_name, options=None, client=None):
    '''
    This creates a queue.

    '''

    if not client:
        client = boto3.client('sqs')

    try:

        if isinstance(options, dict):
            resp = client.create_queue(QueueName=queue_name, Attributes=options)
        else:
            resp = client.create_queue(QueueName=queue_name)

        if resp is not None:
            return {'url':resp['QueueUrl'],
                    'name':queue_name}
        else:
            LOGERROR('could not create the specified queue: %s with options: %s'
                     % (queue_name, options))
            return None

    except Exception as e:
        LOGEXCEPTION('could not create the specified queue: %s with options: %s'
                     % (queue_name, options))
        return None




def sqs_delete_queue(queue_url, client=None):
    '''
    This deletes a queue.

    '''

    if not client:
        client = boto3.client('sqs')

    try:

        client.delete_queue(QueueUrl=queue_url)
        return True

    except Exception as e:
        LOGEXCEPTION('could not delete the specified queue: %s'
                     % (queue_url,))
        return False



def sqs_put_item(queue_url,
                 item,
                 delay_seconds=0,
                 client=None):
    '''
    This pushes an item to the specified queue name.

    item is a JSON object. It will be serialized to a JSON string.

    '''

    if not client:
        client = boto3.client('sqs')

    try:

        json_msg = json.dumps(item)

        resp = client.send_message(
            QueueUrl=queue_url,
            MessageBody=json_msg,
            DelaySeconds=delay_seconds,
        )
        if not resp:
            LOGERROR('could not send item to queue: %s' % queue_url)
            return None
        else:
            return resp

    except Exception as e:
        LOGEXCEPTION('could not send item to queue: %s' % queue_url)
        return None



def sqs_get_item(queue_url,
                 max_items=1,
                 wait_time_seconds=5,
                 client=None):
    '''This gets a single item from the SQS queue.

    The queue url is composed of some internal SQS junk plus a queue_name. The
    queue name will be something like:

    lcproc_queue_<action> where action is one of:

    runcp
    runpf

    The item is always a JSON object:

    {'target': S3 bucket address of the file to process.

     'action': the action to perform on the file (e.g. 'runpf', 'runcp', etc.)

     'args': the action's args as a tuple (not including filename, which will be
             generated randomly as a temporary local file)

     'kwargs': the action's kwargs as a dict

     'outbucket: S3 bucket to write the result to.

     'outqueue': SQS queue to write the processed item's info to (optional)}

    The action MUST match the <action> in the queue name for this item to be
    processed.

    The wait_time_seconds specifies how long the function should block until a
    message is received on the queue. If the timeout expires, an empty list will
    be returned. If the timeout doesn't expire, the function will return a list
    of items received (up to max_items).

    '''

    if not client:
        client = boto3.client('sqs')

    try:

        resp = client.receive_message(
            QueueUrl=queue_url,
            AttributeNames=['All'],
            MaxNumberOfMessages=max_items,
            WaitTimeSeconds=wait_time_seconds
        )

        if not resp:
            LOGERROR('could not receive messages from queue: %s' %
                     queue_url)

        else:

            messages = []

            for msg in resp.get('Messages',[]):

                try:
                    messages.append({
                        'id':msg['MessageId'],
                        'receipt_handle':msg['ReceiptHandle'],
                        'md5':msg['MD5OfBody'],
                        'attributes':msg['Attributes'],
                        'item':json.loads(msg['Body']),
                    })
                except Exception as e:
                    LOGEXCEPTION(
                        'could not deserialize message ID: %s, body: %s' %
                        (msg['MessageId'], msg['Body'])
                    )
                    continue

            return messages

    except Exception as e:
        LOGEXCEPTION('could not get items from queue: %s' % queue_url)
        return None



def sqs_delete_item(queue_url,
                    receipt_handle,
                    client=None):
    '''This deletes a message from the queue, effectively acknowledging its
    receipt.

    Call this only at the end of processing.

    '''

    if not client:
        client = boto3.client('sqs')

    try:

        client.delete_message(
            QueueUrl=queue_url,
            ReceiptHandle=receipt_handle
        )

    except Exception as e:

        LOGEXCEPTION(
            'could not delete message with receipt handle: '
            '%s from queue: %s' % (receipt_handle, queue_url)
        )



def sqs_enqueue_s3_filelist(bucket,
                            filelist_fname,
                            queue_name,
                            action,
                            outbucket,
                            outqueue=None,
                            client=None):

    '''
    This puts all of the files in the specified filelist into the SQS queue.

    '''



def sqs_enqueue_local_filelist(filelist_fname,
                               bucket,
                               queue_name,
                               action,
                               outbucket,
                               outqueue=None,
                               client=None):
    '''This puts all of the files in the specified filelist into the SQS queue.

    All of the files will be first uploaded to the specified bucket and then put
    into an SQS queue for processing.

    '''


#########
## EC2 ##
#########

def make_ec2_node(ami='ami-04681a1dbd79675a5',
                  instance='c5.2xlarge',
                  security_groupid=None,
                  keypair=None):
    '''
    This makes a new EC2 worker node.

    Installs Python 3.6, a virtualenv, and a git checked out copy of astrobase.

    Returns instance ID.

    '''


def make_ec2_cluster(nodes,
                     ami='ami-04681a1dbd79675a5',
                     instance='c5.2xlarge',
                     security_groupid=None,
                     keypair=None):
    '''
    This makes a full EC2 cluster to work on the light curves.

    '''




######################
## WORKER FUNCTIONS ##
######################


def runpf_worker():
    '''
    This is the basic worker for running period-finding on a light curve.

    '''


def runcp_worker():
    '''
    This is the basic worker for running checkplot making on a light curve.

    '''




##################
## WORKER LOOPS ##
##################

def runpf_loop():
    '''
    This runs period-finding in a loop until interrupted.

    '''


def runcp_loop():
    '''
    This runs period-finding in a loop until interrupted.

    '''



def runpf_loop_on_instance():
    '''
    This starts a runpf worker loop on the given EC2 instance.

    '''


def runcp_loop_on_instance():
    '''
    This starts a runcp worker loop on the given EC2 instance.

    '''


##########
## MAIN ##
##########

def main():
    '''
    This starts the lcproc_aws process.

    The cmdline args are:

    <action>
    <inqueue>
    <outqueue>

    '''



if __name__ == '__main__':
    main()
