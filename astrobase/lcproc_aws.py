#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""lcproc_aws.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2018
License: MIT - see the LICENSE file for the full text.

This contains lcproc worker loops useful for AWS processing of light curves.

The basic workflow is:

LCs from S3 -> SQS -> worker loop -> products back to S3 | result JSON to SQS

All functions here assume AWS credentials have been loaded already using awscli
as described at:

https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html

Example script for runcp_consumer_loop() to launch one processing loop per CPU
on an EC2 instance (this goes in the instance's run-data [assuming AMZ Linux 2]
to execute when the instance finishes launching):

---

#!/bin/bash

yum install -y python3-devel gcc-gfortran jq htop emacs-nox git

cat << 'EOF' > /home/ec2-user/launch-runcp.sh
#!/bin/bash

python3 -m venv /home/ec2-user/py3

git clone https://github.com/waqasbhatti/astrobase
cd /home/ec2-user/astrobase
pip install pip setuptools numpy -U
pip install -e .[aws]

mkdir /home/ec2-user/work
cd /home/ec2-user/work

for s in seq `lscpu -J | jq ".lscpu[3].data|tonumber"`; do nohup /home/ec2-user/py3/bin/python3 -u -c "from astrobase import lcproc_aws as lcp; lcp.runcp_consumer_loop('https://queue-url','.','s3://path/to/lclist.pkl')" > runcp-loop.out & done
EOF

chown ec2-user /home/ec2-user/launch-runcp.sh

su ec2-user -c 'bash /home/ec2-user/launch-runcp.sh'

---


Example run-data script if we're using a custom AMI with python bits installed
and the directories set up already:

---

#!/bin/bash

yum install -y python3-devel gcc-gfortran jq htop emacs-nox git

cat << 'EOF' > /home/ec2-user/launch-runcp.sh
#!/bin/bash

cd /home/ec2-user/astrobase
git pull origin master

cd /home/ec2-user/work

for s in seq `lscpu -J | jq ".lscpu[3].data|tonumber"`; do nohup /home/ec2-user/py3/bin/python3 -u -c "from astrobase import lcproc_aws as lcp; lcp.runcp_consumer_loop('https://queue-url','.','s3://path/to/lclist.pkl')" > runcp-loop.out & done
EOF

chown ec2-user /home/ec2-user/launch-runcp.sh

su ec2-user -c 'bash /home/ec2-user/launch-runcp.sh'

---

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
import pickle
import time
import signal
import subprocess
from datetime import timedelta
import base64

import requests
from requests.exceptions import HTTPError

try:

    import boto3
    from botocore.exceptions import ClientError
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
    """
    This justs gets a file from S3.

    """

    if not client:
        client = boto3.client('s3')

    try:
        client.download_file(bucket, filename, local_file)
        return local_file
    except Exception as e:
        LOGEXCEPTION('could not download s3://%s/%s' % (bucket, filename))
        return None



def s3_get_url(url, client=None):
    """
    This gets a file from an S3 bucket based on its s3:// URL.

    """

    bucket_item = url.replace('s3://','')
    bucket_item = bucket_item.split('/')
    bucket = bucket_item[0]
    filekey = '/'.join(bucket_item[1:])

    return s3_get_file(bucket, filekey, bucket_item[-1], client=client)



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



def sqs_enqueue_s3_filelist(bucket,
                            filelist_fname,
                            queue_name,
                            action,
                            outbucket,
                            outqueue=None,
                            client=None):

    """
    This puts all of the files in the specified filelist into the SQS queue.

    """



def sqs_enqueue_local_filelist(filelist_fname,
                               bucket,
                               queue_name,
                               action,
                               outbucket,
                               outqueue=None,
                               client=None):
    """This puts all of the files in the specified filelist into the SQS queue.

    All of the files will be first uploaded to the specified bucket and then put
    into an SQS queue for processing.

    """


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
            EbsOptimized=True
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
                while len(ready_instances) < len(list(instance_dict.keys())):

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
                    time.sleep(5.0)


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
    "TargetCapacity": 32,
    "SpotPrice": "0.33",
    "TerminateInstancesWithExpiration": True,
    'InstanceInterruptionBehavior': 'terminate',
    "LaunchSpecifications": [],
    "Type": "maintain",
    "ReplaceUnhealthyInstances": True,
    "ValidUntil": "datetime-utc"
}


SPOT_INSTANCE_TYPES = [
    "t2.large",
    "t2.xlarge",
    "t2.2xlarge",
    "c4.8xlarge",
    "c4.large",
    "c4.xlarge"
    "m5.large"
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
    "WeightedCapacity":1
}



def make_spot_fleet_cluster(
        security_groupid,
        subnet_id,
        keypair_name,
        iam_instance_profile_arn,
        spot_fleet_iam_role,
        target_capacity=32,
        spot_price=0.33,
        expires_days=7,
        allocation_strategy='lowestPrice',
        instance_types=SPOT_INSTANCE_TYPES,
        instance_weights=None,
        instance_ami='ami-04681a1dbd79675a5',
        instance_user_data=None,
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
    fleetconfig['ValidUntil'] = (datetime.utcnow() +
                                 timedelta(days=expires_days))

    # get the user data from a string or a file
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

    # figure out the instance weights
    if instance_weights is None:
        instance_weights = [1 for x in instance_types]

    for itype, iweight in zip(instance_types, instance_weights):

        thisinstance = SPOT_PERINSTANCE_CONFIG.copy()
        thisinstance['InstanceType'] = itype
        thisinstance['ImageId'] = instance_ami
        thisinstance['SubnetId'] = subnet_id
        thisinstance['KeyName'] = keypair_name
        thisinstance['IamInstanceProfile']['Arn'] = iam_instance_profile_arn
        thisinstance['SecurityGroups']['GroupId'] = security_groupid
        thisinstance['WeightedCapacity'] = iweight
        thisinstance['UserData'] = udata

        fleetconfig['LaunchSpecifications'].append(thisinstance)

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
                curr_try = 1

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

                    curr_try = curr_try + 1
                    LOGINFO(
                        'spot fleet not yet active, waiting 15 seconds. '
                        'try %s/%s' % (curr_try, ntries)
                    )
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



##################
## WORKER LOOPS ##
##################

def kill_handler(sig, frame):
    raise KeyboardInterrupt


def cache_clean_handler(min_age_hours=1):
    """This periodically cleans up the ~/.astrobase cache to save us from
    disk-space doom.

    """

    # find the files to delete
    cmd = (
        "find ~ec2-user/.astrobase -type f -mmin +{mmin} -exec rm -v '{{}}' \;"
    )
    mmin = '%.1f' % (min_age_hours*60.0)
    cmd = cmd.format(mmin=mmin)

    try:
        proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
        ndeleted = len(proc.stdout.decode().split('\n'))
        LOGWARNING('cache clean: %s files older than %s hours deleted' %
                   (ndeleted, min_age_hours))
    except Exception as e:
        LOGEXCEPTION('cache clean: could not delete old files')



def shutdown_check_handler():
    """This checks the instance data URL to see if there's a pending
    shutdown for the instance.

    This is useful for spot instances. If there is a pending shutdown posted to
    the instance data URL, we'll break out of the loop.

    """

    url = 'http://169.254.169.254/latest/meta-data/spot/instance-action'

    try:
        resp = requests.get(url)
        resp.raise_for_status()

        stopinfo = resp.json()
        if 'action' in stopinfo and stopinfo['action'] in ('stop',
                                                           'terminate',
                                                           'hibernate'):
            stoptime = stopinfo['time']
            LOGWARNING('instance is going to %s at %s' % (stopinfo['action'],
                                                          stoptime))

            resp.close()
            return True
        else:
            resp.close()
            return False

    except HTTPError as e:
        resp.close()
        return False



def runpf_loop():

    """
    This runs period-finding in a loop until interrupted.

    """



def runcp_producer_loop(
        lightcurve_list,
        input_queue,
        input_bucket,
        result_queue,
        result_bucket,
        pfresult_list=None,
        runcp_kwargs=None,
        process_list_slice=None,
        download_when_done=True,
        purge_queues_when_done=True,
        save_state_when_done=True,
        delete_queues_when_done=False,
        s3_client=None,
        sqs_client=None
):
    """This sends tasks to the input queue and monitors the result queue for
    task completion.

    process_list_slice is highly recommended because SQS can only handle up to
    120k messages per queue (or maybe this is only 120k received messages and
    not 120k messages actually put into the queue? the SQS docs suck, so
    whatever tf).

    """

    if not sqs_client:
        sqs_client = boto3.client('sqs')
    if not s3_client:
        s3_client = boto3.client('s3')

    if isinstance(lightcurve_list, str) and os.path.exists(lightcurve_list):

        # get the LC list
        with open(lightcurve_list, 'r') as infd:
            lclist = infd.readlines()

        lclist = [x.replace('\n','') for x in lclist if len(x) > 0]
        if process_list_slice is not None:
            lclist = lclist[process_list_slice[0]:process_list_slice[1]]
            lclist = [x[1:] for x in lclist if x.startswith('/')]
            lclist = ['s3://%s/%s' % (input_bucket, x) for x in lclist]

    # this handles direct invocation using lists of s3:// urls of light curves
    elif isinstance(lightcurve_list, list):
        lclist = lightcurve_list

    # set up the input and output queues

    # check if the queues by the input and output names given exist already
    # if they do, go ahead and use them
    # if they don't, make new ones.
    try:
        inq = sqs_client.get_queue_url(QueueName=input_queue)
        inq_url = inq['QueueUrl']
        LOGINFO('input queue already exists, skipping creation...')
    except ClientError as e:
        inq = sqs_create_queue(input_queue, client=sqs_client)
        inq_url = inq['url']

    try:
        outq = sqs_client.get_queue_url(QueueName=result_queue)
        outq_url = outq['QueueUrl']
        LOGINFO('result queue already exists, skipping creation...')
    except ClientError as e:
        outq = sqs_create_queue(result_queue, client=sqs_client)
        outq_url = outq['url']

    LOGINFO('input queue: %s' % inq_url)
    LOGINFO('output queue: %s' % outq_url)

    # wait until queues are up
    LOGINFO('waiting for queues to become ready...')
    time.sleep(10.0)

    # for each item in the lightcurve_list, send it to the input queue and wait
    # until it's done to send another one

    if pfresult_list is None:
        pfresult_list = [None for x in lclist]

    for lc, pf in zip(lclist, pfresult_list):

        this_item = {
            'target': lc,
            'action': 'runcp',
            'args': (pf,),
            'kwargs':runcp_kwargs if runcp_kwargs is not None else {},
            'outbucket': result_bucket,
            'outqueue': outq_url
        }

        resp = sqs_put_item(inq_url, this_item, client=sqs_client)
        if resp:
            LOGINFO('sent %s to queue: %s' % (lc,inq_url))

    # now block until all objects are done
    done_objects = {}

    LOGINFO('all items queued, waiting for results...')

    # listen to the kill and term signals and raise KeyboardInterrupt when
    # called
    signal.signal(signal.SIGINT, kill_handler)
    signal.signal(signal.SIGTERM, kill_handler)

    while len(list(done_objects.keys())) < len(lclist):

        try:

            result = sqs_get_item(outq_url, client=sqs_client)

            if result is not None and len(result) > 0:

                recv = result[0]
                processed_object = recv['item']['target']
                cpf = recv['item']['cpf']
                receipt = recv['receipt_handle']

                if processed_object in lclist:

                    if processed_object not in done_objects:
                        done_objects[processed_object] = [cpf]
                    else:
                        done_objects[processed_object].append(cpf)

                    LOGINFO('done with %s -> %s' % (processed_object, cpf))

                    if download_when_done:

                        getobj = s3_get_url(
                            cpf,
                            client=s3_client
                        )
                        LOGINFO('downloaded %s -> %s' % (cpf, getobj))

                sqs_delete_item(outq_url, receipt)

        except KeyboardInterrupt as e:

            LOGWARNING('breaking out of producer wait-loop')
            break


    # delete the input and output queues when we're done
    LOGINFO('done with processing.')
    time.sleep(1.0)

    if purge_queues_when_done:
        LOGWARNING('purging queues at exit, please wait 10 seconds...')
        sqs_client.purge_queue(QueueUrl=inq_url)
        sqs_client.purge_queue(QueueUrl=outq_url)
        time.sleep(10.0)

    if delete_queues_when_done:
        LOGWARNING('deleting queues at exit')
        sqs_delete_queue(inq_url)
        sqs_delete_queue(outq_url)

    work_state = {
        'done': done_objects,
        'in_progress': list(set(lclist) - set(done_objects.keys())),
        'args':(os.path.abspath(lightcurve_list),
                input_queue,
                input_bucket,
                result_queue,
                result_bucket),
        'kwargs':{'pfresult_list':pfresult_list,
                  'runcp_kwargs':runcp_kwargs,
                  'process_list_slice':process_list_slice,
                  'download_when_done':download_when_done,
                  'purge_queues_when_done':purge_queues_when_done,
                  'save_state_when_done':save_state_when_done,
                  'delete_queues_when_done':delete_queues_when_done}
    }

    if save_state_when_done:
        with open('runcp-queue-producer-loop-state.pkl','wb') as outfd:
            pickle.dump(work_state, outfd, pickle.HIGHEST_PROTOCOL)

    # at the end, return the done_objects dict
    # also return the list of unprocessed items if any
    return work_state



def runcp_producer_loop_savedstate(
        use_saved_state=None,
        lightcurve_list=None,
        input_queue=None,
        input_bucket=None,
        result_queue=None,
        result_bucket=None,
        pfresult_list=None,
        runcp_kwargs=None,
        process_list_slice=None,
        download_when_done=True,
        purge_queues_when_done=True,
        save_state_when_done=True,
        delete_queues_when_done=False,
        s3_client=None,
        sqs_client=None
):
    """This wraps the function above to allow for loading previous state from a
    file.

    """

    if use_saved_state is not None and os.path.exists(use_saved_state):

        with open(use_saved_state,'rb') as infd:
            saved_state = pickle.load(infd)

        # run the producer loop using the saved state's todo list
        return runcp_producer_loop(
            saved_state['in_progress'],
            saved_state['args'][1],
            saved_state['args'][2],
            saved_state['args'][3],
            saved_state['args'][4],
            **saved_state['kwargs']
        )

    else:

        return runcp_producer_loop(
            lightcurve_list,
            input_queue,
            input_bucket,
            result_queue,
            result_bucket,
            pfresult_list=pfresult_list,
            runcp_kwargs=runcp_kwargs,
            process_list_slice=process_list_slice,
            download_when_done=download_when_done,
            purge_queues_when_done=purge_queues_when_done,
            save_state_when_done=save_state_when_done,
            delete_queues_when_done=delete_queues_when_done,
            s3_client=s3_client,
            sqs_client=sqs_client
        )



def runcp_consumer_loop(
        in_queue_url,
        workdir,
        lclist_pkl_s3url,
        wait_time_seconds=5,
        cache_clean_timer_seconds=3600.0,
        shutdown_check_timer_seconds=60.0,
        sqs_client=None,
        s3_client=None
):

    """This runs checkplot making in a loop until interrupted.

    For the moment, we don't generate neighbor light curves since this would
    require a lot more S3 calls.

    """

    if not sqs_client:
        sqs_client = boto3.client('sqs')
    if not s3_client:
        s3_client = boto3.client('s3')

    lclist_pklf = lclist_pkl_s3url.split('/')[-1]

    if not os.path.exists(lclist_pklf):

        # get the lclist pickle from S3 to help with neighbor queries
        lclist_pklf = s3_get_url(
            lclist_pkl_s3url,
            client=s3_client
        )

    with open(lclist_pklf,'rb') as infd:
        lclistpkl = pickle.load(infd)

    # listen to the kill and term signals and raise KeyboardInterrupt when
    # called
    signal.signal(signal.SIGINT, kill_handler)
    signal.signal(signal.SIGTERM, kill_handler)

    shutdown_last_time = time.monotonic()
    diskspace_last_time = time.monotonic()

    while True:

        curr_time = time.monotonic()

        if (curr_time - shutdown_last_time) > shutdown_check_timer_seconds:
            shutdown_check = shutdown_check_handler()
            if shutdown_check:
                LOGWARNING('instance will die soon, breaking loop')
                break
            shutdown_last_time = time.monotonic()

        if (curr_time - diskspace_last_time) > cache_clean_timer_seconds:
            cache_clean_handler()
            diskspace_last_time = time.monotonic()

        try:

            # receive a single message from the inqueue
            work = sqs_get_item(in_queue_url,
                                client=sqs_client,
                                raiseonfail=True)

            # JSON deserialize the work item
            if work is not None and len(work) > 0:

                recv = work[0]

                # skip any messages that don't tell us to runcp
                # FIXME: use the MessageAttributes for setting topics instead
                action = recv['item']['action']
                if action != 'runcp':
                    continue

                target = recv['item']['target']
                args = recv['item']['args']
                kwargs = recv['item']['kwargs']
                outbucket = recv['item']['outbucket']

                if 'outqueue' in recv['item']:
                    out_queue_url = recv['item']['outqueue']
                else:
                    out_queue_url = None

                receipt = recv['receipt_handle']

                # download the target from S3 to a file in the work directory
                try:

                    lc_filename = s3_get_url(
                        target,
                        client=s3_client
                    )

                    # get the period-finder pickle if present in args
                    if len(args) > 0 and args[0] is not None:

                        pf_pickle = s3_get_url(
                            args[0],
                            client=s3_client
                        )

                    else:

                        pf_pickle = None

                    # now runcp
                    cpfs = lcproc.runcp(
                        pf_pickle,
                        workdir,
                        workdir,
                        lcfname=lc_filename,
                        lclistpkl=lclistpkl,
                        makeneighborlcs=False,
                        **kwargs
                    )

                    if cpfs and all(os.path.exists(x) for x in cpfs):

                        LOGINFO('runcp OK for LC: %s, PF: %s -> %s' %
                                (lc_filename, pf_pickle, cpfs))

                        # check if the file exists already because it's been
                        # processed somewhere else
                        resp = s3_client.list_objects_v2(
                            Bucket=outbucket,
                            MaxKeys=1,
                            Prefix=cpfs[0]
                        )
                        outbucket_list = resp.get('Contents',[])

                        if outbucket_list and len(outbucket_list) > 0:

                            LOGWARNING(
                                'not uploading runcp results for %s because '
                                'they exist in the output bucket already'
                                % target
                            )
                            sqs_delete_item(in_queue_url, receipt)
                            continue

                        for cpf in cpfs:

                            put_url = s3_put_file(cpf,
                                                  outbucket,
                                                  client=s3_client)

                            if put_url is not None:

                                LOGINFO('result uploaded to %s' % put_url)

                                # put the S3 URL of the output into the output
                                # queue if requested
                                if out_queue_url is not None:

                                    sqs_put_item(
                                        out_queue_url,
                                        {'cpf':put_url,
                                         'target': target,
                                         'lc_filename':lc_filename,
                                         'lclistpkl':lclist_pklf,
                                         'kwargs':kwargs},
                                        raiseonfail=True
                                    )

                                # delete the result from the local directory
                                os.remove(cpf)

                            # if the upload fails, don't acknowledge the
                            # message. might be a temporary S3 failure, so
                            # another worker might succeed later.
                            else:
                                LOGERROR('failed to upload %s to S3' % cpf)

                        # delete the input item from the input queue to
                        # acknowledge its receipt and indicate that
                        # processing is done and successful
                        sqs_delete_item(in_queue_url,
                                        receipt)

                        # delete the light curve file when we're done with it
                        os.remove(lc_filename)

                    # if runcp failed outright, don't requeue. instead, write a
                    # ('failed-checkplot-%s.pkl' % lc_filename) file to the
                    # output S3 bucket.
                    else:

                        LOGWARNING('runcp failed for LC: %s, PF: %s' %
                                   (lc_filename, pf_pickle))

                        with open('failed-checkplot-%s.pkl' %
                                  lc_filename, 'wb') as outfd:
                            pickle.dump(
                                {'in_queue_url':in_queue_url,
                                 'target':target,
                                 'lc_filename':lc_filename,
                                 'lclistpkl':lclist_pklf,
                                 'kwargs':kwargs,
                                 'outbucket':outbucket,
                                 'out_queue_url':out_queue_url},
                                outfd, pickle.HIGHEST_PROTOCOL
                            )

                        put_url = s3_put_file(
                            'failed-checkplot-%s.pkl' % lc_filename,
                            outbucket,
                            client=s3_client
                        )

                        # put the S3 URL of the output into the output
                        # queue if requested
                        if out_queue_url is not None:

                            sqs_put_item(
                                out_queue_url,
                                {'cpf':put_url,
                                 'lc_filename':lc_filename,
                                 'lclistpkl':lclist_pklf,
                                 'kwargs':kwargs},
                                raiseonfail=True
                            )

                        # delete the input item from the input queue to
                        # acknowledge its receipt and indicate that
                        # processing is done
                        sqs_delete_item(in_queue_url,
                                        receipt,
                                        raiseonfail=True)

                        # delete the light curve file when we're done with it
                        os.remove(lc_filename)


                except ClientError as e:

                    LOGWARNING('queues have disappeared. stopping worker loop')
                    break


                # if there's any other exception, put a failed response into the
                # output bucket and queue
                except Exception as e:

                    LOGEXCEPTION('could not process input from queue')

                    with open('failed-checkplot-%s.pkl' %
                              lc_filename,'wb') as outfd:
                        pickle.dump(
                            {'in_queue_url':in_queue_url,
                             'target':target,
                             'lc_filename':lc_filename,
                             'lclistpkl':lclist_pklf,
                             'kwargs':kwargs,
                             'outbucket':outbucket,
                             'out_queue_url':out_queue_url},
                            outfd, pickle.HIGHEST_PROTOCOL
                        )

                    put_url = s3_put_file(
                        'failed-checkplot-%s.pkl' % lc_filename,
                        outbucket,
                        client=s3_client
                    )


                    # put the S3 URL of the output into the output
                    # queue if requested
                    if out_queue_url is not None:

                        sqs_put_item(
                            out_queue_url,
                            {'cpf':put_url,
                             'lc_filename':lc_filename,
                             'lclistpkl':lclistpkl,
                             'kwargs':kwargs},
                            raiseonfail=True
                        )

                    # delete the input item from the input queue to
                    # acknowledge its receipt and indicate that
                    # processing is done
                    sqs_delete_item(in_queue_url,
                                    receipt,
                                    raiseonfail=True)

                    # delete the light curve file when we're done with it
                    os.remove(lc_filename)

        # a keyboard interrupt kills the loop
        except KeyboardInterrupt:

            LOGWARNING('breaking out of the processing loop.')
            break


        # if the queues disappear, then the producer loop is done and we should
        # exit
        except ClientError as e:

            LOGWARNING('queues have disappeared. stopping worker loop')
            break


        # any other exception continues the loop we'll write the output file to
        # the output S3 bucket (and any optional output queue), but add a
        # failed-* prefix to it to indicate that processing failed. FIXME: could
        # use a dead-letter queue for this instead
        except Exception as e:


            LOGEXCEPTION('could not process input from queue')

            with open('failed-checkplot-%s.pkl' %
                      lc_filename,'wb') as outfd:
                pickle.dump(
                    {'in_queue_url':in_queue_url,
                     'target':target,
                     'lclistpkl':lclist_pklf,
                     'kwargs':kwargs,
                     'outbucket':outbucket,
                     'out_queue_url':out_queue_url},
                    outfd, pickle.HIGHEST_PROTOCOL
                )

            put_url = s3_put_file(
                'failed-checkplot-%s.pkl' % lc_filename,
                outbucket,
                client=s3_client
            )


            # put the S3 URL of the output into the output
            # queue if requested
            if out_queue_url is not None:

                sqs_put_item(
                    out_queue_url,
                    {'cpf':put_url,
                     'lclistpkl':lclist_pklf,
                     'kwargs':kwargs},
                    raiseonfail=True
                )

            # delete the input item from the input queue to
            # acknowledge its receipt and indicate that
            # processing is done
            sqs_delete_item(in_queue_url, receipt, raiseonfail=True)



############################################
## STARTING WORKER LOOPS ON EC2 INSTANCES ##
############################################


def runpf_loop_on_instance():
    """
    This starts a runpf worker loop on the given EC2 instance.

    """


def runcp_loop_on_instance():
    """
    This starts a runcp worker loop on the given EC2 instance.

    """


##########
## MAIN ##
##########

def main():
    """
    This starts the lcproc_aws process.

    The cmdline args are:

    <action>
    <inqueue>

    """



if __name__ == '__main__':
    main()
