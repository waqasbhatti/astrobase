#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# awsrun.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2018
# License: MIT - see the LICENSE file for the full text.

"""

This contains lcproc worker loops useful for AWS processing of light curves.

The basic workflow is::

   LCs from S3 -> SQS -> worker loop -> products back to S3 | result JSON to SQS

All functions here assume AWS credentials have been loaded already using awscli
as described at:

https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html

General recommendations:

- use t3.medium or t3.micro instances for runcp_consumer_loop. Checkplot making
  isn't really a CPU intensive activity, so using these will be cheaper.

- use c5.2xlarge or above instances for runpf_consumer_loop. Period-finders
  require a decent number of fast cores, so a spot fleet of these instances
  should be cost-effective.

- you may want a t3.micro instance running in the same region and VPC as your
  worker node instances to serve as a head node driving the producer_loop
  functions. This can be done from a machine outside AWS, but you'll incur
  (probably tiny) charges for network egress from the output queues.

- It's best not to download results from S3 as soon as they're produced. Leave
  them on S3 until everything is done, then use rclone (https://rclone.org) to
  sync them back to your machines using --transfers <large number>.

The user_data and instance_user_data kwargs for the make_ec2_nodes and
make_spot_fleet_cluster functions can be used to start processing loops as soon
as EC2 brings up the VM instance. This is especially useful for spot fleets set
to maintain a target capacity, since worker nodes will be terminated and
automatically replaced. Bringing up the processing loop at instance start up
makes it easy to continue processing light curves exactly where you left off
without having to manually intervene.

Example script for user_data bringing up a checkplot-making loop on instance
creation (assuming we're using Amazon Linux 2)::

    #!/bin/bash

    cat << 'EOF' > /home/ec2-user/launch-runcp.sh
    #!/bin/bash
    sudo yum -y install python3-devel gcc-gfortran jq htop emacs-nox git

    # create the virtualenv
    python3 -m venv /home/ec2-user/py3

    # get astrobase
    cd /home/ec2-user
    git clone https://github.com/waqasbhatti/astrobase

    # install it
    cd /home/ec2-user/astrobase
    /home/ec2-user/py3/bin/pip install pip setuptools numpy -U
    /home/ec2-user/py3/bin/pip install -e .[aws]

    # make the work dir
    mkdir /home/ec2-user/work
    cd /home/ec2-user/work

    # wait a bit for the instance info to be populated
    sleep 5

    # set some environ vars for boto3 and the processing loop
    export AWS_DEFAULT_REGION=`curl --silent http://169.254.169.254/latest/dynamic/instance-identity/document/ | jq '.region' | tr -d '"'`
    export NCPUS=`lscpu -J | jq ".lscpu[3].data|tonumber"`

    # launch the processor loops
    for s in `seq $NCPUS`; do nohup /home/ec2-user/py3/bin/python3 -u -c "from astrobase.lcproc import awsrun as lcp; lcp.runcp_consumer_loop('https://queue-url','.','s3://path/to/lclist.pkl')" > runcp-$s-loop.out & done
    EOF

    # run the script we just created as ec2-user
    chown ec2-user /home/ec2-user/launch-runcp.sh
    su ec2-user -c 'bash /home/ec2-user/launch-runcp.sh'

Here's a similar script for a runpf consumer loop. We launch only a single
instance of the loop because runpf will use all CPUs by default for its
period-finder parallelized functions::

    #!/bin/bash

    cat << 'EOF' > /home/ec2-user/launch-runpf.sh
    #!/bin/bash
    sudo yum -y install python3-devel gcc-gfortran jq htop emacs-nox git

    python3 -m venv /home/ec2-user/py3

    cd /home/ec2-user
    git clone https://github.com/waqasbhatti/astrobase

    cd /home/ec2-user/astrobase
    /home/ec2-user/py3/bin/pip install pip setuptools numpy -U
    /home/ec2-user/py3/bin/pip install -e .[aws]

    mkdir /home/ec2-user/work
    cd /home/ec2-user/work

    # wait a bit for the instance info to be populated
    sleep 5

    export AWS_DEFAULT_REGION=`curl --silent http://169.254.169.254/latest/dynamic/instance-identity/document/ | jq '.region' | tr -d '"'`
    export NCPUS=`lscpu -J | jq ".lscpu[3].data|tonumber"`

    # launch the processes
    nohup /home/ec2-user/py3/bin/python3 -u -c "from astrobase.lcproc import awsrun as lcp; lcp.runpf_consumer_loop('https://input-queue-url','.')" > runpf-loop.out &
    EOF

    chown ec2-user /home/ec2-user/launch-runpf.sh
    su ec2-user -c 'bash /home/ec2-user/launch-runpf.sh'

"""

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
import pickle
import time
import signal
import subprocess

import requests
from requests.exceptions import HTTPError

try:

    import boto3
    from botocore.exceptions import ClientError

except ImportError:
    raise ImportError(
        "This module requires the boto3 package from PyPI. "
        "You'll also need the awscli package to set up the "
        "AWS secret key config for this module."
    )

from .. import awsutils
from .periodsearch import runpf
from .checkplotgen import runcp


####################################
## WORKER LOOPS UTILITY FUNCTIONS ##
####################################

def kill_handler(sig, frame):
    ''' This raises a KeyboardInterrupt when a SIGKILL comes in.

    This is a handle for use with the Python `signal.signal` function.

    '''
    raise KeyboardInterrupt


def cache_clean_handler(min_age_hours=1):
    """This periodically cleans up the ~/.astrobase cache to save us from
    disk-space doom.

    Parameters
    ----------

    min_age_hours : int
        Files older than this number of hours from the current time will be
        deleted.

    Returns
    -------

    Nothing.

    """

    # find the files to delete
    cmd = (
        r"find ~ec2-user/.astrobase -type f -mmin +{mmin} -exec rm -v '{{}}' \;"
    )
    mmin = '%.1f' % (min_age_hours*60.0)
    cmd = cmd.format(mmin=mmin)

    try:
        proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
        ndeleted = len(proc.stdout.decode().split('\n'))
        LOGWARNING('cache clean: %s files older than %s hours deleted' %
                   (ndeleted, min_age_hours))
    except Exception:
        LOGEXCEPTION('cache clean: could not delete old files')


def shutdown_check_handler():
    """This checks the AWS instance data URL to see if there's a pending
    shutdown for the instance.

    This is useful for AWS spot instances. If there is a pending shutdown posted
    to the instance data URL, we'll use the result of this function break out of
    the processing loop and shut everything down ASAP before the instance dies.

    Returns
    -------

    bool
        - True if the instance is going to die soon.
        - False if the instance is still safe.

    """

    url = 'http://169.254.169.254/latest/meta-data/spot/instance-action'

    try:
        resp = requests.get(url, timeout=1.0)
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

    except HTTPError:
        resp.close()
        return False

    except Exception:
        resp.close()
        return False


############################
## CHECKPLOT MAKING LOOPS ##
############################

def runcp_producer_loop(
        lightcurve_list,
        input_queue,
        input_bucket,
        result_queue,
        result_bucket,
        pfresult_list=None,
        runcp_kwargs=None,
        process_list_slice=None,
        purge_queues_when_done=False,
        delete_queues_when_done=False,
        download_when_done=True,
        save_state_when_done=True,
        s3_client=None,
        sqs_client=None
):
    """This sends checkplot making tasks to the input queue and monitors the
    result queue for task completion.

    Parameters
    ----------

    lightcurve_list : str or list of str
        This is either a string pointing to a file containing a list of light
        curves filenames to process or the list itself. The names must
        correspond to the full filenames of files stored on S3, including all
        prefixes, but not include the 's3://<bucket name>/' bit (these will be
        added automatically).

    input_queue : str
        This is the name of the SQS queue which will receive processing tasks
        generated by this function. The queue URL will automatically be obtained
        from AWS.

    input_bucket : str
        The name of the S3 bucket containing the light curve files to process.

    result_queue : str
        This is the name of the SQS queue that this function will listen to for
        messages from the workers as they complete processing on their input
        elements. This function will attempt to match input sent to the
        `input_queue` with results coming into the `result_queue` so it knows
        how many objects have been successfully processed. If this function
        receives task results that aren't in its own input queue, it will
        acknowledge them so they complete successfully, but not download them
        automatically. This handles leftover tasks completing from a previous
        run of this function.

    result_bucket : str
        The name of the S3 bucket which will receive the results from the
        workers.

    pfresult_list : list of str or None
        This is a list of periodfinder result pickle S3 URLs associated with
        each light curve. If provided, this will be used to add in phased light
        curve plots to each checkplot pickle. If this is None, the worker loop
        will produce checkplot pickles that only contain object information,
        neighbor information, and unphased light curves.

    runcp_kwargs : dict
        This is a dict used to pass any extra keyword arguments to the
        `lcproc.checkplotgen.runcp` function that will be run by the worker
        loop.

    process_list_slice : list
        This is used to index into the input light curve list so a subset of the
        full list can be processed in this specific run of this function.

        Use None for a slice index elem to emulate single slice spec behavior:

        process_list_slice = [10, None]  -> lightcurve_list[10:]
        process_list_slice = [None, 500] -> lightcurve_list[:500]

    purge_queues_when_done : bool
        If this is True, and this function exits (either when all done, or when
        it is interrupted with a Ctrl+C), all outstanding elements in the
        input/output queues that have not yet been acknowledged by workers or by
        this function will be purged. This effectively cancels all outstanding
        work.

    delete_queues_when_done : bool
        If this is True, and this function exits (either when all done, or when
        it is interrupted with a Ctrl+C'), all outstanding work items will be
        purged from the input/queues and the queues themselves will be deleted.

    download_when_done : bool
        If this is True, the generated checkplot pickle for each input work item
        will be downloaded immediately to the current working directory when the
        worker functions report they're done with it.

    save_state_when_done : bool
        If this is True, will save the current state of the work item queue and
        the work items acknowledged as completed to a pickle in the current
        working directory. Call the `runcp_producer_loop_savedstate` function
        below to resume processing from this saved state later.

    s3_client : boto3.Client or None
        If None, this function will instantiate a new `boto3.Client` object to
        use in its S3 download operations. Alternatively, pass in an existing
        `boto3.Client` instance to re-use it here.

    sqs_client : boto3.Client or None
        If None, this function will instantiate a new `boto3.Client` object to
        use in its SQS operations. Alternatively, pass in an existing
        `boto3.Client` instance to re-use it here.

    Returns
    -------

    dict or str
        Returns the current work state as a dict or str path to the generated
        work state pickle depending on if `save_state_when_done` is True.

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
    except ClientError:
        inq = awsutils.sqs_create_queue(input_queue, client=sqs_client)
        inq_url = inq['url']

    try:
        outq = sqs_client.get_queue_url(QueueName=result_queue)
        outq_url = outq['QueueUrl']
        LOGINFO('result queue already exists, skipping creation...')
    except ClientError:
        outq = awsutils.sqs_create_queue(result_queue, client=sqs_client)
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

        resp = awsutils.sqs_put_item(inq_url, this_item, client=sqs_client)
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

            result = awsutils.sqs_get_item(outq_url, client=sqs_client)

            if result is not None and len(result) > 0:

                recv = result[0]
                try:
                    processed_object = recv['item']['target']
                except KeyError:
                    LOGWARNING('unknown target in received item: %s' % recv)
                    processed_object = 'unknown-lc'

                cpf = recv['item']['cpf']
                receipt = recv['receipt_handle']

                if processed_object in lclist:

                    if processed_object not in done_objects:
                        done_objects[processed_object] = [cpf]
                    else:
                        done_objects[processed_object].append(cpf)

                    LOGINFO('done with %s -> %s' % (processed_object, cpf))

                    if download_when_done:

                        getobj = awsutils.awsutils.s3_get_url(
                            cpf,
                            client=s3_client
                        )
                        LOGINFO('downloaded %s -> %s' % (cpf, getobj))

                else:
                    LOGWARNING('processed object returned is not in '
                               'queued target list, probably from an '
                               'earlier run. accepting but not downloading.')

                awsutils.sqs_delete_item(outq_url, receipt)

        except KeyboardInterrupt:

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
        awsutils.sqs_delete_queue(inq_url)
        awsutils.sqs_delete_queue(outq_url)

    work_state = {
        'done': done_objects,
        'in_progress': list(set(lclist) - set(done_objects.keys())),
        'args':((os.path.abspath(lightcurve_list) if
                 isinstance(lightcurve_list, str) else lightcurve_list),
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

    Parameters
    ----------

    use_saved_state : str or None
        This is the path to the saved state pickle file produced by a previous
        run of `runcp_producer_loop`. Will get all of the arguments to run
        another instance of the loop from that pickle file. If this is None, you
        MUST provide all of the appropriate arguments to that function.

    lightcurve_list : str or list of str or None
        This is either a string pointing to a file containing a list of light
        curves filenames to process or the list itself. The names must
        correspond to the full filenames of files stored on S3, including all
        prefixes, but not include the 's3://<bucket name>/' bit (these will be
        added automatically).

    input_queue : str or None
        This is the name of the SQS queue which will receive processing tasks
        generated by this function. The queue URL will automatically be obtained
        from AWS.

    input_bucket : str or None
        The name of the S3 bucket containing the light curve files to process.

    result_queue : str or None
        This is the name of the SQS queue that this function will listen to for
        messages from the workers as they complete processing on their input
        elements. This function will attempt to match input sent to the
        `input_queue` with results coming into the `result_queue` so it knows
        how many objects have been successfully processed. If this function
        receives task results that aren't in its own input queue, it will
        acknowledge them so they complete successfully, but not download them
        automatically. This handles leftover tasks completing from a previous
        run of this function.

    result_bucket : str or None
        The name of the S3 bucket which will receive the results from the
        workers.

    pfresult_list : list of str or None
        This is a list of periodfinder result pickle S3 URLs associated with
        each light curve. If provided, this will be used to add in phased light
        curve plots to each checkplot pickle. If this is None, the worker loop
        will produce checkplot pickles that only contain object information,
        neighbor information, and unphased light curves.

    runcp_kwargs : dict or None
        This is a dict used to pass any extra keyword arguments to the
        `lcproc.checkplotgen.runcp` function that will be run by the worker
        loop.

    process_list_slice : list or None
        This is used to index into the input light curve list so a subset of the
        full list can be processed in this specific run of this function.

        Use None for a slice index elem to emulate single slice spec behavior:

        process_list_slice = [10, None]  -> lightcurve_list[10:]
        process_list_slice = [None, 500] -> lightcurve_list[:500]

    purge_queues_when_done : bool or None
        If this is True, and this function exits (either when all done, or when
        it is interrupted with a Ctrl+C), all outstanding elements in the
        input/output queues that have not yet been acknowledged by workers or by
        this function will be purged. This effectively cancels all outstanding
        work.

    delete_queues_when_done : bool or None
        If this is True, and this function exits (either when all done, or when
        it is interrupted with a Ctrl+C'), all outstanding work items will be
        purged from the input/queues and the queues themselves will be deleted.

    download_when_done : bool or None
        If this is True, the generated checkplot pickle for each input work item
        will be downloaded immediately to the current working directory when the
        worker functions report they're done with it.

    save_state_when_done : bool or None
        If this is True, will save the current state of the work item queue and
        the work items acknowledged as completed to a pickle in the current
        working directory. Call the `runcp_producer_loop_savedstate` function
        below to resume processing from this saved state later.

    s3_client : boto3.Client or None
        If None, this function will instantiate a new `boto3.Client` object to
        use in its S3 download operations. Alternatively, pass in an existing
        `boto3.Client` instance to re-use it here.

    sqs_client : boto3.Client or None
        If None, this function will instantiate a new `boto3.Client` object to
        use in its SQS operations. Alternatively, pass in an existing
        `boto3.Client` instance to re-use it here.

    Returns
    -------

    dict or str
        Returns the current work state as a dict or str path to the generated
        work state pickle depending on if `save_state_when_done` is True.

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
        lc_altexts=('',),
        wait_time_seconds=5,
        cache_clean_timer_seconds=3600.0,
        shutdown_check_timer_seconds=60.0,
        sqs_client=None,
        s3_client=None
):

    """This runs checkplot pickle making in a loop until interrupted.

    Consumes work task items from an input queue set up by `runcp_producer_loop`
    above. For the moment, we don't generate neighbor light curves since this
    would require a lot more S3 calls.

    Parameters
    ----------

    in_queue_url : str
        The SQS URL of the input queue to listen to for work assignment
        messages. The task orders will include the input and output S3 bucket
        names, as well as the URL of the output queue to where this function
        will report its work-complete or work-failed status.

    workdir : str
        The directory on the local machine where this worker loop will download
        the input light curves and associated period-finder results (if any),
        process them, and produce its output checkplot pickles. These will then
        be uploaded to the specified S3 output bucket and then deleted from the
        workdir when the upload is confirmed to make it safely to S3.

    lclist_pkl : str
        S3 URL of a catalog pickle generated by `lcproc.catalogs.make_lclist`
        that contains objectids and coordinates, as well as a kdtree for all of
        the objects in the current light curve collection being processed. This
        is used to look up neighbors for each object being processed.

    lc_altexts : sequence of str
        If not None, this is a sequence of alternate extensions to try for the
        input light curve file other than the one provided in the input task
        order. For example, to get anything that's an .sqlite where .sqlite.gz
        is expected, use altexts=[''] to strip the .gz.

    wait_time_seconds : int
        The amount of time to wait in the input SQS queue for an input task
        order. If this timeout expires and no task has been received, this
        function goes back to the top of the work loop.

    cache_clean_timer_seconds : float
        The amount of time in seconds to wait before periodically removing old
        files (such as finder chart FITS, external service result pickles) from
        the astrobase cache directory. These accumulate as the work items are
        processed, and take up significant space, so must be removed
        periodically.

    shutdown_check_timer_seconds : float
        The amount of time to wait before checking for a pending EC2 shutdown
        message for the instance this worker loop is operating on. If a shutdown
        is noticed, the worker loop is cancelled in preparation for instance
        shutdown.

    sqs_client : boto3.Client or None
        If None, this function will instantiate a new `boto3.Client` object to
        use in its SQS operations. Alternatively, pass in an existing
        `boto3.Client` instance to re-use it here.

    s3_client : boto3.Client or None
        If None, this function will instantiate a new `boto3.Client` object to
        use in its S3 operations. Alternatively, pass in an existing
        `boto3.Client` instance to re-use it here.

    Returns
    -------

    Nothing.

    """

    if not sqs_client:
        sqs_client = boto3.client('sqs')
    if not s3_client:
        s3_client = boto3.client('s3')

    lclist_pklf = lclist_pkl_s3url.split('/')[-1]

    if not os.path.exists(lclist_pklf):

        # get the lclist pickle from S3 to help with neighbor queries
        lclist_pklf = awsutils.s3_get_url(
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
            work = awsutils.sqs_get_item(in_queue_url,
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

                    lc_filename = awsutils.s3_get_url(
                        target,
                        altexts=lc_altexts,
                        client=s3_client,
                    )

                    # get the period-finder pickle if present in args
                    if len(args) > 0 and args[0] is not None:

                        pf_pickle = awsutils.s3_get_url(
                            args[0],
                            client=s3_client
                        )

                    else:

                        pf_pickle = None

                    # now runcp
                    cpfs = runcp(
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
                            awsutils.sqs_delete_item(in_queue_url, receipt)
                            continue

                        for cpf in cpfs:

                            put_url = awsutils.s3_put_file(cpf,
                                                           outbucket,
                                                           client=s3_client)

                            if put_url is not None:

                                LOGINFO('result uploaded to %s' % put_url)

                                # put the S3 URL of the output into the output
                                # queue if requested
                                if out_queue_url is not None:

                                    awsutils.sqs_put_item(
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
                        awsutils.sqs_delete_item(in_queue_url,
                                                 receipt)

                        # delete the light curve file when we're done with it
                        if ( (lc_filename is not None) and
                             (os.path.exists(lc_filename)) ):
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

                        put_url = awsutils.s3_put_file(
                            'failed-checkplot-%s.pkl' % lc_filename,
                            outbucket,
                            client=s3_client
                        )

                        # put the S3 URL of the output into the output
                        # queue if requested
                        if out_queue_url is not None:

                            awsutils.sqs_put_item(
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
                        awsutils.sqs_delete_item(in_queue_url,
                                                 receipt,
                                                 raiseonfail=True)

                        # delete the light curve file when we're done with it
                        if ( (lc_filename is not None) and
                             (os.path.exists(lc_filename)) ):
                            os.remove(lc_filename)

                except ClientError:

                    LOGWARNING('queues have disappeared. stopping worker loop')
                    break

                # if there's any other exception, put a failed response into the
                # output bucket and queue
                except Exception:

                    LOGEXCEPTION('could not process input from queue')

                    if 'lc_filename' in locals():

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

                        put_url = awsutils.s3_put_file(
                            'failed-checkplot-%s.pkl' % lc_filename,
                            outbucket,
                            client=s3_client
                        )

                        # put the S3 URL of the output into the output
                        # queue if requested
                        if out_queue_url is not None:

                            awsutils.sqs_put_item(
                                out_queue_url,
                                {'cpf':put_url,
                                 'lc_filename':lc_filename,
                                 'lclistpkl':lclist_pklf,
                                 'kwargs':kwargs},
                                raiseonfail=True
                            )

                        if ( (lc_filename is not None) and
                             (os.path.exists(lc_filename)) ):
                            os.remove(lc_filename)

                    # delete the input item from the input queue to
                    # acknowledge its receipt and indicate that
                    # processing is done
                    awsutils.sqs_delete_item(in_queue_url,
                                             receipt,
                                             raiseonfail=True)

        # a keyboard interrupt kills the loop
        except KeyboardInterrupt:

            LOGWARNING('breaking out of the processing loop.')
            break

        # if the queues disappear, then the producer loop is done and we should
        # exit
        except ClientError:

            LOGWARNING('queues have disappeared. stopping worker loop')
            break

        # any other exception continues the loop we'll write the output file to
        # the output S3 bucket (and any optional output queue), but add a
        # failed-* prefix to it to indicate that processing failed. FIXME: could
        # use a dead-letter queue for this instead
        except Exception:

            LOGEXCEPTION('could not process input from queue')

            if 'lc_filename' in locals():

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

                put_url = awsutils.s3_put_file(
                    'failed-checkplot-%s.pkl' % lc_filename,
                    outbucket,
                    client=s3_client
                )

                # put the S3 URL of the output into the output
                # queue if requested
                if out_queue_url is not None:

                    awsutils.sqs_put_item(
                        out_queue_url,
                        {'cpf':put_url,
                         'lclistpkl':lclist_pklf,
                         'kwargs':kwargs},
                        raiseonfail=True
                    )

                if ( (lc_filename is not None) and
                     (os.path.exists(lc_filename)) ):
                    os.remove(lc_filename)

            # delete the input item from the input queue to
            # acknowledge its receipt and indicate that
            # processing is done
            awsutils.sqs_delete_item(in_queue_url, receipt, raiseonfail=True)


#########################
## PERIOD-FINDER LOOPS ##
#########################

def runpf_producer_loop(
        lightcurve_list,
        input_queue,
        input_bucket,
        result_queue,
        result_bucket,
        pfmethods=('gls','pdm','mav','bls','win'),
        pfkwargs=({}, {}, {}, {}, {}),
        extra_runpf_kwargs={'getblssnr':True},
        process_list_slice=None,
        purge_queues_when_done=False,
        delete_queues_when_done=False,
        download_when_done=True,
        save_state_when_done=True,
        s3_client=None,
        sqs_client=None
):

    """This queues up work for period-finders using SQS.

    Parameters
    ----------

    lightcurve_list : str or list of str
        This is either a string pointing to a file containing a list of light
        curves filenames to process or the list itself. The names must
        correspond to the full filenames of files stored on S3, including all
        prefixes, but not include the 's3://<bucket name>/' bit (these will be
        added automatically).

    input_queue : str
        This is the name of the SQS queue which will receive processing tasks
        generated by this function. The queue URL will automatically be obtained
        from AWS.

    input_bucket : str
        The name of the S3 bucket containing the light curve files to process.

    result_queue : str
        This is the name of the SQS queue that this function will listen to for
        messages from the workers as they complete processing on their input
        elements. This function will attempt to match input sent to the
        `input_queue` with results coming into the `result_queue` so it knows
        how many objects have been successfully processed. If this function
        receives task results that aren't in its own input queue, it will
        acknowledge them so they complete successfully, but not download them
        automatically. This handles leftover tasks completing from a previous
        run of this function.

    result_bucket : str
        The name of the S3 bucket which will receive the results from the
        workers.

    pfmethods : sequence of str
        This is a list of period-finder method short names as listed in the
        `lcproc.periodfinding.PFMETHODS` dict. This is used to tell the worker
        loop which period-finders to run on the input light curve.

    pfkwargs : sequence of dicts
        This contains optional kwargs as dicts to be supplied to all of the
        period-finder functions listed in `pfmethods`. This should be the same
        length as that sequence.

    extra_runpf_kwargs : dict
        This is a dict of kwargs to be supplied to `runpf` driver function
        itself.

    process_list_slice : list
        This is used to index into the input light curve list so a subset of the
        full list can be processed in this specific run of this function.

        Use None for a slice index elem to emulate single slice spec behavior:

        process_list_slice = [10, None]  -> lightcurve_list[10:]
        process_list_slice = [None, 500] -> lightcurve_list[:500]

    purge_queues_when_done : bool
        If this is True, and this function exits (either when all done, or when
        it is interrupted with a Ctrl+C), all outstanding elements in the
        input/output queues that have not yet been acknowledged by workers or by
        this function will be purged. This effectively cancels all outstanding
        work.

    delete_queues_when_done : bool
        If this is True, and this function exits (either when all done, or when
        it is interrupted with a Ctrl+C'), all outstanding work items will be
        purged from the input/queues and the queues themselves will be deleted.

    download_when_done : bool
        If this is True, the generated periodfinding result pickle for each
        input work item will be downloaded immediately to the current working
        directory when the worker functions report they're done with it.

    save_state_when_done : bool
        If this is True, will save the current state of the work item queue and
        the work items acknowledged as completed to a pickle in the current
        working directory. Call the `runcp_producer_loop_savedstate` function
        below to resume processing from this saved state later.

    s3_client : boto3.Client or None
        If None, this function will instantiate a new `boto3.Client` object to
        use in its S3 download operations. Alternatively, pass in an existing
        `boto3.Client` instance to re-use it here.

    sqs_client : boto3.Client or None
        If None, this function will instantiate a new `boto3.Client` object to
        use in its SQS operations. Alternatively, pass in an existing
        `boto3.Client` instance to re-use it here.

    Returns
    -------

    dict or str
        Returns the current work state as a dict or str path to the generated
        work state pickle depending on if `save_state_when_done` is True.

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
    except ClientError:
        inq = awsutils.sqs_create_queue(input_queue, client=sqs_client)
        inq_url = inq['url']

    try:
        outq = sqs_client.get_queue_url(QueueName=result_queue)
        outq_url = outq['QueueUrl']
        LOGINFO('result queue already exists, skipping creation...')
    except ClientError:
        outq = awsutils.sqs_create_queue(result_queue, client=sqs_client)
        outq_url = outq['url']

    LOGINFO('input queue: %s' % inq_url)
    LOGINFO('output queue: %s' % outq_url)

    # wait until queues are up
    LOGINFO('waiting for queues to become ready...')
    time.sleep(10.0)

    all_runpf_kwargs = {'pfmethods':pfmethods,
                        'pfkwargs':pfkwargs}
    if isinstance(extra_runpf_kwargs, dict):
        all_runpf_kwargs.update(extra_runpf_kwargs)

    # enqueue the work items
    for lc in lclist:

        this_item = {
            'target': lc,
            'action': 'runpf',
            'args': ('.',),
            'kwargs':all_runpf_kwargs,
            'outbucket': result_bucket,
            'outqueue': outq_url
        }

        resp = awsutils.sqs_put_item(inq_url, this_item, client=sqs_client)
        if resp:
            LOGINFO('sent %s to queue: %s' % (lc, inq_url))

    # now block until all objects are done
    done_objects = {}

    LOGINFO('all items queued, waiting for results...')

    # listen to the kill and term signals and raise KeyboardInterrupt when
    # called
    signal.signal(signal.SIGINT, kill_handler)
    signal.signal(signal.SIGTERM, kill_handler)

    while len(list(done_objects.keys())) < len(lclist):

        try:

            result = awsutils.sqs_get_item(outq_url, client=sqs_client)

            if result is not None and len(result) > 0:

                recv = result[0]
                try:
                    processed_object = recv['item']['target']
                except KeyError:
                    LOGWARNING('unknown target in received item: %s' % recv)
                    processed_object = 'unknown-lc'

                pfresult = recv['item']['pfresult']
                receipt = recv['receipt_handle']

                if processed_object in lclist:

                    if processed_object not in done_objects:
                        done_objects[processed_object] = [pfresult]
                    else:
                        done_objects[processed_object].append(pfresult)

                    LOGINFO('done with %s -> %s' % (processed_object, pfresult))

                    if download_when_done:

                        getobj = awsutils.s3_get_url(
                            pfresult,
                            client=s3_client
                        )
                        LOGINFO('downloaded %s -> %s' % (pfresult, getobj))

                else:
                    LOGWARNING('processed object returned is not in '
                               'queued target list, probably from an '
                               'earlier run. accepting but not downloading.')

                awsutils.sqs_delete_item(outq_url, receipt)

        except KeyboardInterrupt:

            LOGWARNING('breaking out of runpf producer wait-loop')
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
        awsutils.sqs_delete_queue(inq_url)
        awsutils.sqs_delete_queue(outq_url)

    work_state = {
        'done': done_objects,
        'in_progress': list(set(lclist) - set(done_objects.keys())),
        'args':((os.path.abspath(lightcurve_list) if
                 isinstance(lightcurve_list, str) else lightcurve_list),
                input_queue,
                input_bucket,
                result_queue,
                result_bucket),
        'kwargs':{'pfmethods':pfmethods,
                  'pfkwargs':pfkwargs,
                  'extra_runpf_kwargs':extra_runpf_kwargs,
                  'process_list_slice':process_list_slice,
                  'purge_queues_when_done':purge_queues_when_done,
                  'delete_queues_when_done':delete_queues_when_done,
                  'download_when_done':download_when_done,
                  'save_state_when_done':save_state_when_done}
    }

    if save_state_when_done:
        with open('runpf-queue-producer-loop-state.pkl','wb') as outfd:
            pickle.dump(work_state, outfd, pickle.HIGHEST_PROTOCOL)

    # at the end, return the done_objects dict
    # also return the list of unprocessed items if any
    return work_state


def runpf_consumer_loop(
        in_queue_url,
        workdir,
        lc_altexts=('',),
        wait_time_seconds=5,
        shutdown_check_timer_seconds=60.0,
        sqs_client=None,
        s3_client=None
):
    """This runs period-finding in a loop until interrupted.

    Consumes work task items from an input queue set up by `runpf_producer_loop`
    above.

    Parameters
    ----------

    in_queue_url : str
        The SQS URL of the input queue to listen to for work assignment
        messages. The task orders will include the input and output S3 bucket
        names, as well as the URL of the output queue to where this function
        will report its work-complete or work-failed status.

    workdir : str
        The directory on the local machine where this worker loop will download
        the input light curves, process them, and produce its output
        periodfinding result pickles. These will then be uploaded to the
        specified S3 output bucket, and then deleted from the local disk.

    lc_altexts : sequence of str
        If not None, this is a sequence of alternate extensions to try for the
        input light curve file other than the one provided in the input task
        order. For example, to get anything that's an .sqlite where .sqlite.gz
        is expected, use altexts=[''] to strip the .gz.

    wait_time_seconds : int
        The amount of time to wait in the input SQS queue for an input task
        order. If this timeout expires and no task has been received, this
        function goes back to the top of the work loop.

    shutdown_check_timer_seconds : float
        The amount of time to wait before checking for a pending EC2 shutdown
        message for the instance this worker loop is operating on. If a shutdown
        is noticed, the worker loop is cancelled in preparation for instance
        shutdown.

    sqs_client : boto3.Client or None
        If None, this function will instantiate a new `boto3.Client` object to
        use in its SQS operations. Alternatively, pass in an existing
        `boto3.Client` instance to re-use it here.

    s3_client : boto3.Client or None
        If None, this function will instantiate a new `boto3.Client` object to
        use in its S3 operations. Alternatively, pass in an existing
        `boto3.Client` instance to re-use it here.

    Returns
    -------

    Nothing.

    """

    if not sqs_client:
        sqs_client = boto3.client('sqs')
    if not s3_client:
        s3_client = boto3.client('s3')

    # listen to the kill and term signals and raise KeyboardInterrupt when
    # called
    signal.signal(signal.SIGINT, kill_handler)
    signal.signal(signal.SIGTERM, kill_handler)

    shutdown_last_time = time.monotonic()

    while True:

        curr_time = time.monotonic()

        if (curr_time - shutdown_last_time) > shutdown_check_timer_seconds:
            shutdown_check = shutdown_check_handler()
            if shutdown_check:
                LOGWARNING('instance will die soon, breaking loop')
                break
            shutdown_last_time = time.monotonic()

        try:

            # receive a single message from the inqueue
            work = awsutils.sqs_get_item(in_queue_url,
                                         client=sqs_client,
                                         raiseonfail=True)

            # JSON deserialize the work item
            if work is not None and len(work) > 0:

                recv = work[0]

                # skip any messages that don't tell us to runpf
                action = recv['item']['action']
                if action != 'runpf':
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

                    lc_filename = awsutils.s3_get_url(
                        target,
                        altexts=lc_altexts,
                        client=s3_client
                    )

                    runpf_args = (lc_filename, args[0])

                    # now runpf
                    pfresult = runpf(
                        *runpf_args,
                        **kwargs
                    )

                    if pfresult and os.path.exists(pfresult):

                        LOGINFO('runpf OK for LC: %s -> %s' %
                                (lc_filename, pfresult))

                        # check if the file exists already because it's been
                        # processed somewhere else
                        resp = s3_client.list_objects_v2(
                            Bucket=outbucket,
                            MaxKeys=1,
                            Prefix=pfresult
                        )
                        outbucket_list = resp.get('Contents',[])

                        if outbucket_list and len(outbucket_list) > 0:

                            LOGWARNING(
                                'not uploading pfresult for %s because '
                                'it exists in the output bucket already'
                                % target
                            )
                            awsutils.sqs_delete_item(in_queue_url, receipt)
                            continue

                        put_url = awsutils.s3_put_file(pfresult,
                                                       outbucket,
                                                       client=s3_client)

                        if put_url is not None:

                            LOGINFO('result uploaded to %s' % put_url)

                            # put the S3 URL of the output into the output
                            # queue if requested
                            if out_queue_url is not None:

                                awsutils.sqs_put_item(
                                    out_queue_url,
                                    {'pfresult':put_url,
                                     'target': target,
                                     'lc_filename':lc_filename,
                                     'kwargs':kwargs},
                                    raiseonfail=True
                                )

                            # delete the result from the local directory
                            os.remove(pfresult)

                        # if the upload fails, don't acknowledge the
                        # message. might be a temporary S3 failure, so
                        # another worker might succeed later.
                        # FIXME: add SNS bits to warn us of failures
                        else:
                            LOGERROR('failed to upload %s to S3' % pfresult)
                            os.remove(pfresult)

                        # delete the input item from the input queue to
                        # acknowledge its receipt and indicate that
                        # processing is done and successful
                        awsutils.sqs_delete_item(in_queue_url, receipt)

                        # delete the light curve file when we're done with it
                        if ( (lc_filename is not None) and
                             (os.path.exists(lc_filename)) ):
                            os.remove(lc_filename)

                    # if runcp failed outright, don't requeue. instead, write a
                    # ('failed-checkplot-%s.pkl' % lc_filename) file to the
                    # output S3 bucket.
                    else:

                        LOGWARNING('runpf failed for LC: %s' %
                                   (lc_filename,))

                        with open('failed-periodfinding-%s.pkl' %
                                  lc_filename, 'wb') as outfd:
                            pickle.dump(
                                {'in_queue_url':in_queue_url,
                                 'target':target,
                                 'lc_filename':lc_filename,
                                 'kwargs':kwargs,
                                 'outbucket':outbucket,
                                 'out_queue_url':out_queue_url},
                                outfd, pickle.HIGHEST_PROTOCOL
                            )

                        put_url = awsutils.s3_put_file(
                            'failed-periodfinding-%s.pkl' % lc_filename,
                            outbucket,
                            client=s3_client
                        )

                        # put the S3 URL of the output into the output
                        # queue if requested
                        if out_queue_url is not None:

                            awsutils.sqs_put_item(
                                out_queue_url,
                                {'pfresult':put_url,
                                 'lc_filename':lc_filename,
                                 'kwargs':kwargs},
                                raiseonfail=True
                            )

                        # delete the input item from the input queue to
                        # acknowledge its receipt and indicate that
                        # processing is done
                        awsutils.sqs_delete_item(in_queue_url,
                                                 receipt,
                                                 raiseonfail=True)

                        # delete the light curve file when we're done with it
                        if ( (lc_filename is not None) and
                             (os.path.exists(lc_filename)) ):
                            os.remove(lc_filename)

                except ClientError:

                    LOGWARNING('queues have disappeared. stopping worker loop')
                    break

                # if there's any other exception, put a failed response into the
                # output bucket and queue
                except Exception:

                    LOGEXCEPTION('could not process input from queue')

                    if 'lc_filename' in locals():

                        with open('failed-periodfinding-%s.pkl' %
                                  lc_filename,'wb') as outfd:
                            pickle.dump(
                                {'in_queue_url':in_queue_url,
                                 'target':target,
                                 'lc_filename':lc_filename,
                                 'kwargs':kwargs,
                                 'outbucket':outbucket,
                                 'out_queue_url':out_queue_url},
                                outfd, pickle.HIGHEST_PROTOCOL
                            )

                        put_url = awsutils.s3_put_file(
                            'failed-periodfinding-%s.pkl' % lc_filename,
                            outbucket,
                            client=s3_client
                        )

                        # put the S3 URL of the output into the output
                        # queue if requested
                        if out_queue_url is not None:

                            awsutils.sqs_put_item(
                                out_queue_url,
                                {'pfresult':put_url,
                                 'lc_filename':lc_filename,
                                 'kwargs':kwargs},
                                raiseonfail=True
                            )

                        # delete the light curve file when we're done with it
                        if ( (lc_filename is not None) and
                             (os.path.exists(lc_filename)) ):
                            os.remove(lc_filename)

                    # delete the input item from the input queue to
                    # acknowledge its receipt and indicate that
                    # processing is done
                    awsutils.sqs_delete_item(in_queue_url,
                                             receipt,
                                             raiseonfail=True)

        # a keyboard interrupt kills the loop
        except KeyboardInterrupt:

            LOGWARNING('breaking out of the processing loop.')
            break

        # if the queues disappear, then the producer loop is done and we should
        # exit
        except ClientError:

            LOGWARNING('queues have disappeared. stopping worker loop')
            break

        # any other exception continues the loop we'll write the output file to
        # the output S3 bucket (and any optional output queue), but add a
        # failed-* prefix to it to indicate that processing failed. FIXME: could
        # use a dead-letter queue for this instead
        except Exception:

            LOGEXCEPTION('could not process input from queue')

            if 'lc_filename' in locals():

                with open('failed-periodfinding-%s.pkl' %
                          lc_filename,'wb') as outfd:
                    pickle.dump(
                        {'in_queue_url':in_queue_url,
                         'target':target,
                         'kwargs':kwargs,
                         'outbucket':outbucket,
                         'out_queue_url':out_queue_url},
                        outfd, pickle.HIGHEST_PROTOCOL
                    )

                put_url = awsutils.s3_put_file(
                    'failed-periodfinding-%s.pkl' % lc_filename,
                    outbucket,
                    client=s3_client
                )

                # put the S3 URL of the output into the output
                # queue if requested
                if out_queue_url is not None:

                    awsutils.sqs_put_item(
                        out_queue_url,
                        {'cpf':put_url,
                         'kwargs':kwargs},
                        raiseonfail=True
                    )
                if ( (lc_filename is not None) and
                     (os.path.exists(lc_filename)) ):
                    os.remove(lc_filename)

            # delete the input item from the input queue to
            # acknowledge its receipt and indicate that
            # processing is done
            awsutils.sqs_delete_item(in_queue_url, receipt, raiseonfail=True)
