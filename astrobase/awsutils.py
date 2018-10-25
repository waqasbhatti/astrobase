#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""awsutils.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2018
License: MIT - see the LICENSE file for the full text.

This contains functions that handle various AWS services for use with
lcproc_aws.py.

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
import base64

try:

    import boto3
    from botocore.exceptions import ClientError
    import paramiko
    import paramiko.client

except ImportError:
    raise ImportError(
        "This module requires the boto3 and paramiko packages from PyPI. "
        "You'll also need the awscli package to set up the "
        "AWS secret key config for this module."
    )



#############################
## SSHING TO EC2 INSTANCES ##
#############################

def ec2_ssh(ip_address,
            keypem_file,
            username='ec2-user',
            raiseonfail=False):
    """This opens an SSH connection to the EC2 instance at ip_address.

    Returns an SSHClient object.

    use SSHClient.exec_command(command, environment=None) to exec a shell
    command.

    use SSHClient.open_sftp() to get a SFTPClient for the server.

    use SFTPClient.get() and .put() to copy files from and to the server.

    """

    c = paramiko.client.SSHClient()
    c.load_system_host_keys()
    c.set_missing_host_key_policy(paramiko.client.AutoAddPolicy)

    # load the private key from the AWS keypair pem
    privatekey = paramiko.RSAKey.from_private_key_file(keypem_file)

    # connect to the server
    try:

        c.connect(ip_address,
                  pkey=privatekey,
                  username='ec2-user')

        return c

    except Exception as e:
        LOGEXCEPTION('could not connect to EC2 instance at %s '
                     'using keyfile: %s and user: %s' %
                     (ip_address, keypem_file, username))
        if raiseonfail:
            raise

        return None




########
## S3 ##
########

def s3_get_file(bucket,
                filename,
                local_file,
                altexts=None,
                client=None):

    """This gets a file from an S3 bucket.

    altexts is a list of alternate extensions to try for the file other than the
    one provided.

    e.g. to get anything that's an .sqlite where .sqlite.gz is expected, use
    altexts=[''] to strip the .gz.

    """

    if not client:
        client = boto3.client('s3')

    try:

        client.download_file(bucket, filename, local_file)
        return local_file

    except Exception as e:

        if altexts is not None:

            for alt_extension in altexts:

                split_ext = os.path.splitext(filename)
                check_file = split_ext[0] + alt_extension
                try:
                    client.download_file(
                        bucket,
                        check_file,
                        local_file.replace(split_ext[-1],
                                           alt_extension)
                    )
                    return local_file.replace(split_ext[-1],
                                              alt_extension)
                except Exception as e:
                    pass

        else:

            LOGEXCEPTION('could not download s3://%s/%s' % (bucket, filename))
            return None



def s3_get_url(url,
               altexts=None,
               client=None):
    """
    This gets a file from an S3 bucket based on its s3:// URL.

    """

    bucket_item = url.replace('s3://','')
    bucket_item = bucket_item.split('/')
    bucket = bucket_item[0]
    filekey = '/'.join(bucket_item[1:])

    return s3_get_file(bucket,
                       filekey,
                       bucket_item[-1],
                       altexts=altexts,
                       client=client)



def s3_put_file(local_file, bucket, client=None):
    """
    This uploads a file to S3.

    """

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
    """
    This deletes a file from S3.

    """

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
    """
    This creates a queue.

    """

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
    """
    This deletes a queue.

    """

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
                 client=None,
                 raiseonfail=False):
    """
    This pushes an item to the specified queue name.

    item is a JSON object. It will be serialized to a JSON string.

    """

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

        if raiseonfail:
            raise

        return None



def sqs_get_item(queue_url,
                 max_items=1,
                 wait_time_seconds=5,
                 client=None,
                 raiseonfail=False):
    """This gets a single item from the SQS queue.

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

    """

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

        if raiseonfail:
            raise

        return None



def sqs_delete_item(queue_url,
                    receipt_handle,
                    client=None,
                    raiseonfail=False):
    """This deletes a message from the queue, effectively acknowledging its
    receipt.

    Call this only at the end of processing.

    """

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

        if raiseonfail:
            raise



#########
## EC2 ##
#########

SUPPORTED_AMIS = [
    # Debian 9
    'ami-03006931f694ea7eb',
    # Amazon Linux 2
    'ami-04681a1dbd79675a5',
]



def make_ec2_nodes(
        security_groupid,
        subnet_id,
        keypair_name,
        iam_instance_profile_arn,
        launch_instances=1,
        ami='ami-04681a1dbd79675a5',
        instance='t3.micro',
        ebs_optimized=True,
        user_data=None,
        wait_until_up=True,
        client=None,
        raiseonfail=False,
):
    """This makes new EC2 worker nodes.

    This requires a security group ID attached to a VPC config and subnet, a
    keypair generated beforehand, and an IAM role ARN for the instance. See:

    https://docs.aws.amazon.com/cli/latest/userguide/tutorial-ec2-ubuntu.html

    Use user_data to launch tasks on instance launch.

    Returns instance info as a dict.

    """

    if not client:
        client = boto3.client('ec2')

    # get the user data from a string or a file
    # note: boto3 will base64 encode this itself
    if isinstance(user_data, str) and os.path.exists(user_data):
        with open(user_data,'r') as infd:
            udata = infd.read()

    elif isinstance(user_data, str):
        udata = user_data

    else:
        udata = (
            '#!/bin/bash\necho "No user data provided. '
            'Launched instance at: %s UTC"' % datetime.utcnow().isoformat()
        )


    # fire the request
    try:
        resp = client.run_instances(
            ImageId=ami,
            InstanceType=instance,
            SecurityGroupIds=[
                security_groupid,
            ],
            SubnetId=subnet_id,
            UserData=udata,
            IamInstanceProfile={'Arn':iam_instance_profile_arn},
            InstanceInitiatedShutdownBehavior='terminate',
            KeyName=keypair_name,
            MaxCount=launch_instances,
            MinCount=launch_instances,
            EbsOptimized=ebs_optimized,
        )

        if not resp:
            LOGERROR('could not launch requested instance')
            return None

        else:

            instance_dict = {}

            instance_list = resp.get('Instances',[])

            if len(instance_list) > 0:

                for instance in instance_list:

                    LOGINFO('launched instance ID: %s of type: %s at: %s. '
                            'current state: %s'
                            % (instance['InstanceId'],
                               instance['InstanceType'],
                               instance['LaunchTime'].isoformat(),
                               instance['State']['Name']))

                    instance_dict[instance['InstanceId']] = {
                        'type':instance['InstanceType'],
                        'launched':instance['LaunchTime'],
                        'state':instance['State']['Name'],
                        'info':instance
                    }

            # if we're waiting until we're up, then do so
            if wait_until_up:

                ready_instances = []

                LOGINFO('waiting until launched instances are up...')

                ntries = 5
                curr_try = 0

                while ( (curr_try < ntries) or
                        ( len(ready_instances) <
                          len(list(instance_dict.keys()))) ):

                    resp = client.describe_instances(
                        InstanceIds=list(instance_dict.keys()),
                    )

                    if len(resp['Reservations']) > 0:
                        for resv in resp['Reservations']:
                            if len(resv['Instances']) > 0:
                                for instance in resv['Instances']:
                                    if instance['State']['Name'] == 'running':

                                        ready_instances.append(
                                            instance['InstanceId']
                                        )

                                        instance_dict[
                                            instance['InstanceId']
                                        ]['state'] = 'running'

                                        instance_dict[
                                            instance['InstanceId']
                                        ]['ip'] = instance['PublicIpAddress']

                                        instance_dict[
                                            instance['InstanceId']
                                        ]['info'] = instance

                    # sleep for a bit so we don't hit the API too often
                    curr_try = curr_try + 1
                    time.sleep(5.0)

                if len(ready_instances) == len(list(instance_dict.keys())):
                    LOGINFO('all instances now up.')
                else:
                    LOGWARNING(
                        'reached maximum number of tries for instance status, '
                        'not all instances may be up.'
                    )


            return instance_dict

    except ClientError as e:

        LOGEXCEPTION('could not launch requested instance')
        if raiseonfail:
            raise

        return None

    except Exception as e:

        LOGEXCEPTION('could not launch requested instance')
        if raiseonfail:
            raise

        return None



def delete_ec2_nodes(
        instance_id_list,
        client=None
):
    """
    This deletes EC2 nodes and terminates the instances.

    """

    if not client:
        client = boto3.client('ec2')

    resp = client.terminate_instances(
        InstanceIds=instance_id_list
    )

    return resp



#########################
## SPOT FLEET CLUSTERS ##
#########################

SPOT_FLEET_CONFIG = {
    "IamFleetRole": "iam-fleet-role-arn",
    "AllocationStrategy": "lowestPrice",
    "TargetCapacity": 20,
    "SpotPrice": "0.4",
    "TerminateInstancesWithExpiration": True,
    'InstanceInterruptionBehavior': 'terminate',
    "LaunchSpecifications": [],
    "Type": "maintain",
    "ReplaceUnhealthyInstances": True,
    "ValidUntil": "datetime-utc"
}


SPOT_INSTANCE_TYPES = [
    "m5.xlarge",
    "m5.2xlarge",
    "c5.xlarge",
    "c5.2xlarge",
    "c5.4xlarge",
]


SPOT_PERINSTANCE_CONFIG = {
    "InstanceType": "instance-type",
    "ImageId": "image-id",
    "SubnetId": "subnet-id",
    "KeyName": "keypair-name",
    "IamInstanceProfile": {
        "Arn": "instance-profile-role-arn"
    },
    "SecurityGroups": [
        {
            "GroupId": "security-group-id"
        }
    ],
    "UserData":"base64-encoded-userdata",
    "EbsOptimized":True,
}



def make_spot_fleet_cluster(
        security_groupid,
        subnet_id,
        keypair_name,
        iam_instance_profile_arn,
        spot_fleet_iam_role,
        target_capacity=20,
        spot_price=0.4,
        expires_days=7,
        allocation_strategy='lowestPrice',
        instance_types=SPOT_INSTANCE_TYPES,
        instance_weights=None,
        instance_ami='ami-04681a1dbd79675a5',
        instance_user_data=None,
        instance_ebs_optimized=True,
        wait_until_up=True,
        client=None,
        raiseonfail=False
):
    """
    This makes an EC2 spot-fleet cluster to work on the light curves.

    Returns the spot fleet request ID if successful.

    """

    fleetconfig = SPOT_FLEET_CONFIG.copy()
    fleetconfig['IamFleetRole'] = spot_fleet_iam_role
    fleetconfig['AllocationStrategy'] = allocation_strategy
    fleetconfig['TargetCapacity'] = target_capacity
    fleetconfig['SpotPrice'] = str(spot_price)
    fleetconfig['ValidUntil'] = (
        datetime.utcnow() + timedelta(days=expires_days)
    ).strftime(
        '%Y-%m-%dT%H:%M:%SZ'
    )

    # get the user data from a string or a file
    # we need to base64 encode it here
    if (isinstance(instance_user_data, str) and
        os.path.exists(instance_user_data)):
        with open(instance_user_data,'rb') as infd:
            udata = base64.b64encode(infd.read()).decode()

    elif isinstance(instance_user_data, str):
        udata = base64.b64encode(instance_user_data.encode()).decode()

    else:
        udata = (
            '#!/bin/bash\necho "No user data provided. '
            'Launched instance at: %s UTC"' % datetime.utcnow().isoformat()
        )
        udata = base64.b64encode(udata.encode()).decode()


    for ind, itype in enumerate(instance_types):

        thisinstance = SPOT_PERINSTANCE_CONFIG.copy()
        thisinstance['InstanceType'] = itype
        thisinstance['ImageId'] = instance_ami
        thisinstance['SubnetId'] = subnet_id
        thisinstance['KeyName'] = keypair_name
        thisinstance['IamInstanceProfile']['Arn'] = iam_instance_profile_arn
        thisinstance['SecurityGroups'][0] = {'GroupId':security_groupid}
        thisinstance['UserData'] = udata
        thisinstance['EbsOptimized'] = instance_ebs_optimized

        # get the instance weights
        if isinstance(instance_weights, list):
            thisinstance['WeightedCapacity'] = instance_weights[ind]

        fleetconfig['LaunchSpecifications'].append(thisinstance)

    #
    # launch the fleet
    #

    if not client:
        client = boto3.client('ec2')

    try:

        resp = client.request_spot_fleet(
            SpotFleetRequestConfig=fleetconfig,
        )

        if not resp:

            LOGERROR('spot fleet request failed.')
            return None

        else:

            spot_fleet_reqid = resp['SpotFleetRequestId']
            LOGINFO('spot fleet requested successfully. request ID: %s' %
                    spot_fleet_reqid)

            if not wait_until_up:
                return spot_fleet_reqid

            else:

                ntries = 10
                curr_try = 0

                while curr_try < ntries:

                    resp = client.describe_spot_fleet_requests(
                        SpotFleetRequestIds=[
                            spot_fleet_reqid
                        ]
                    )

                    curr_state = resp.get('SpotFleetRequestConfigs',[])

                    if len(curr_state) > 0:

                        curr_state = curr_state[0]['SpotFleetRequestState']

                        if curr_state == 'active':
                            LOGINFO('spot fleet with reqid: %s is now active' %
                                    spot_fleet_reqid)
                            break

                    LOGINFO(
                        'spot fleet not yet active, waiting 15 seconds. '
                        'try %s/%s' % (curr_try, ntries)
                    )
                    curr_try = curr_try + 1
                    time.sleep(15.0)

                return spot_fleet_reqid


    except ClientError as e:

        LOGEXCEPTION('could not launch spot fleet')
        if raiseonfail:
            raise

        return None

    except Exception as e:

        LOGEXCEPTION('could not launch spot fleet')
        if raiseonfail:
            raise

        return None



def delete_spot_fleet_cluster(
        spot_fleet_reqid,
        client=None,
):
    """
    This deletes a spot-fleet cluster.

    """

    if not client:
        client = boto3.client('ec2')

    resp = client.cancel_spot_fleet_requests(
        SpotFleetRequestIds=[spot_fleet_reqid],
        TerminateInstances=True
    )

    return resp
