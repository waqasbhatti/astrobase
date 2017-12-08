#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''rfclass.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Dec 2017
License: MIT. See the LICENSE file for more details.

Does variable classification using random forests. Two types of classification
are supported:

- Variable classification using non-periodic features: this is used to perform a
  binary classification between non-variable and variable. Uses the features in
  varclass/features.py and varclass/starfeatures.py.

- Periodic variable classification using periodic features: this is used to
  perform multi-class classification for periodic variables using the features
  in varclass/periodicfeatures.py and varclass/starfeatures.py. The classes
  recognized are listed in PERIODIC_VARCLASSES below and were generated from
  manual classification run on various HATNet, HATSouth and HATPI fields.

'''

import logging
from datetime import datetime
from traceback import format_exc
from time import time as unixtime
import glob
import os.path
import os
import shutil
try:
    import cPickle as pickle
except:
    import pickle

try:
    from tqdm import tqdm
    TQDM = True
except:
    TQDM = False
    pass

import numpy as np
import numpy.random as npr
# seed the numpy random generator
# we'll use RANDSEED for scipy.stats distribution functions as well
RANDSEED = 0xdecaff
npr.seed(RANDSEED)

from scipy.stats import randint as sp_randint

# scikit imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.model_selection import train_test_split

from operator import itemgetter
from sklearn.metrics import r2_score, median_absolute_error, \
    precision_score, recall_score, confusion_matrix, f1_score

#############
## LOGGING ##
#############

# setup a logger
LOGGER = None

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.rfclass' % parent_name)

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



#######################
## UTILITY FUNCTIONS ##
#######################

# Utility function to report best scores
def gridsearch_report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        LOGINFO("Model with rank: {0}".format(i + 1))
        LOGINFO("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        LOGINFO("Parameters: {0}".format(score.parameters))




###################################
## NON-PERIODIC VAR FEATURE LIST ##
###################################

NONPERIODIC_FEATURES_TO_COLLECT = [
    'stetsonj',
    'stetsonk',
    'amplitude',
    'magnitude_ratio',
    'linear_fit_slope',
    'eta_normal',
    'percentile_difference_flux_percentile',
    'mad',
    'skew',
    'kurtosis',
    'mag_iqr',
    'beyond1std',
    'grcolor',
    'gicolor',
    'ricolor',
    'bvcolor',
    'jhcolor',
    'jkcolor',
    'hkcolor',
    'gkcolor',
    'propermotion',
]



########################
## FEATURE COLLECTION ##
########################

def collect_features(
        featuresdir,
        magcol,
        outfile,
        pklglob='varfeatures-*.pkl',
        featurestouse=NONPERIODIC_FEATURES_TO_COLLECT,
        maxobjects=None,
        labeldict=None,
        labeltype='binary',
):
    '''This collects variability features into arrays.

    featuresdir is the directory where all the varfeatures pickles are. Use
    pklglob to specify the glob to search for. varfeatures pickles contain
    objectids, a light curve magcol, and features as dict key-vals. The lcproc
    module can be used to produce these.


    magcol is the light curve magnitude col key to use when looking inside each
    varfeatures pickle.


    Each varfeature pickle can contain any combination of non-periodic, stellar,
    and periodic features; these must have the same names as elements in the
    list of strings provided in featurestouse.  This tries to get all the
    features listed in NONPERIODIC_FEATURES_TO_COLLECT by default. If
    featurestouse is not None, gets only the features listed in this kwarg
    instead.


    maxobjects controls how many pickles to process.


    If labeldict is not None, it must be a dict with the following key:val
    list:

    '<objectid>':<label value>

    for each objectid collected from the varfeatures pickles. This will turn the
    collected information into a training set for classifiers.

    Example: to carry out non-periodic variable feature collection of fake LCS
    prepared by fakelcs.generation, use the value of the 'isvariable' dict elem
    from fakelcs-info.pkl here, like so:

    labeldict={x:y for x,y in zip(fakelcinfo['objectid'],
                                  fakelcinfo['isvariable'])}

    labeltype is either 'binary' or 'classes' for binary/multi-class
    classification respectively.

    '''

    # list of input pickles generated by varfeatures in lcproc.py
    pklist = glob.glob(os.path.join(featuresdir, pklglob))

    if maxobjects:
        pklist = pklist[:maxobjects]


    # fancy progress bar with tqdm if present
    if TQDM:
        listiterator = tqdm(pklist)
    else:
        listiterator = pklist

    # go through all the varfeatures arrays

    feature_dict = {'objectids':[],'magcol':magcol, 'availablefeatures':[]}

    LOGINFO('collecting features for magcol: %s' % magcol)

    for pkl in listiterator:

        with open(pkl,'rb') as infd:
            varf = pickle.load(infd)

        # update the objectid list
        objectid = varf['objectid']
        if objectid not in feature_dict['objectids']:
            feature_dict['objectids'].append(objectid)

        thisfeatures = varf[magcol]

        if featurestouse and len(featurestouse) > 0:
            featurestoget = featurestouse
        else:
            featurestoget = NONPERIODIC_FEATURES_TO_COLLECT

        # collect all the features for this magcol/objectid combination
        for feature in featurestoget:

            # update the global feature list if necessary
            if ((feature not in feature_dict['availablefeatures']) and
                (feature in thisfeatures)):

                feature_dict['availablefeatures'].append(feature)
                feature_dict[feature] = []

            if feature in thisfeatures:

                feature_dict[feature].append(
                    thisfeatures[feature]
                )

    # now that we've collected all the objects and their features, turn the list
    # into arrays, and then concatenate them
    for feat in feature_dict['availablefeatures']:
        feature_dict[feat] = np.array(feature_dict[feat])

    feature_dict['objectids'] = np.array(feature_dict['objectids'])

    feature_array = np.column_stack([feature_dict[feat] for feat in
                                     feature_dict['availablefeatures']])
    feature_dict['features_array'] = feature_array


    # if there's a labeldict available, use it to generate a label array. this
    # feature collection is now a training set.
    if isinstance(labeldict, dict):

        labelarray = np.zeros(feature_dict['objectids'].size, dtype=np.int64)

        # populate the labels for each object in the training set
        for ind, objectid in enumerate(feature_dict['objectids']):

            if objectid in labeldict:

                # if this is a binary classifier training set, convert bools to
                # ones and zeros
                if labeltype == 'binary':

                    if labeldict[objectid]:
                        labelarray[ind] = 1

                # otherwise, use the actual class label integer
                elif labeltype == 'classes':
                    labelarray[ind] = labeldict[objectid]

        feature_dict['labels_array'] = labelarray


    feature_dict['kwargs'] = {'pklglob':pklglob,
                              'featurestouse':featurestouse,
                              'maxobjects':maxobjects,
                              'labeltype':labeltype}

    # write the info to the output pickle
    with open(outfile,'wb') as outfd:
        pickle.dump(feature_dict, outfd, pickle.HIGHEST_PROTOCOL)

    # return the feature_dict
    return feature_dict



#################################
## TRAINING THE RF CLASSIFIERS ##
#################################

def train_rf_classifier(
        collected_features,
        test_fraction=0.25,
        n_crossval_iterations=20,
        n_kfolds=5,
        crossval_scoring_metric='f1',
        classifier_to_pickle=None,
        nworkers=-1,
):

    '''This gets the best RF classifier after running cross-validation.

    - splits the training set into test/train samples
    - does KFold stratified cross-validation using RandomizedSearchCV
    - gets the randomforest with the best performance after CV
    - gets the confusion matrix for the test set

    Runs on the output dict from functions that produce dicts similar to that
    produced by collect_features.

    By default, this is tuned for binary classification. Change the
    crossval_scoring_metric to another metric (probably 'accuracy') for
    multi-class classification, e.g. for periodic variable classification. See
    the link below to specify the scoring parameter (this can either be a string
    or an actual scorer object):

    http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

    '''

    if isinstance(collected_features,str) and os.path.exists(collected_features):
        with open(collected_features,'rb') as infd:
            fdict = pickle.load(infd)
    elif isinstance(collected_features, dict):
        fdict = collected_features
    else:
        LOGERROR("can't figure out the input collected_features arg")
        return None

    tfeatures = fdict['features_array']
    tlabels = fdict['labels_array']
    tfeaturenames = fdict['availablefeatures']
    tmagcol = fdict['magcol']
    tobjectids = fdict['objectids']


    # split the training set into training/test samples using stratification
    # to keep the same fraction of variable/nonvariables in each
    training_features, testing_features, training_labels, testing_labels = (
        train_test_split(
            tfeatures,
            tlabels,
            test_size=test_fraction,
            random_state=RANDSEED,
            stratify=tlabels
        )
    )

    # get a random forest classifier
    clf = RandomForestClassifier(n_jobs=nworkers,
                                 random_state=RANDSEED)


    # this is the grid def for hyperparam optimization
    rf_hyperparams = {
        "max_depth": [3,4,5,10,20,None],
        "n_estimators":sp_randint(100,2000),
        "max_features": sp_randint(1, 5),
        "min_samples_split": sp_randint(1, 11),
        "min_samples_leaf": sp_randint(1, 11),
    }


    # run the stratified kfold cross-validation on training features using our
    # random forest classifier object
    cvsearch = RandomizedSearchCV(
        clf,
        param_distributions=rf_hyperparams,
        n_iter=n_crossval_iterations,
        scoring=crossval_scoring_metric,
        cv=StratifiedKFold(n_splits=n_kfolds,
                           shuffle=True,
                           random_state=RANDSEED),
        random_state=RANDSEED
    )


    LOGINFO('running grid-search CV to optimize RF hyperparameters...')
    cvsearch_classifiers = cvsearch.fit(training_features,
                                        training_labels)

    # report on the classifiers' performance
    gridsearch_report(cvsearch_classifiers.grid_scores_)

    # get the best classifier after CV is done
    bestclf = cvsearch_classifiers.best_estimator_
    bestclf_score = cvsearch_classifiers.best_score_
    bestclf_hyperparams = cvsearch_classifiers.best_params_

    # test this classifier on the testing set
    test_predicted_labels = bestclf.predict(testing_features)

    recscore = recall_score(testing_labels, test_predicted_labels)
    precscore = precision_score(testing_labels,test_predicted_labels)
    f1score = f1_score(testing_labels, test_predicted_labels)
    confmatrix = confusion_matrix(testing_labels, test_predicted_labels)


    # write the classifier, its training/testing set, and its stats to the
    # pickle if requested
    outdict = {'features':tfeatures,
               'labels':tlabels,
               'feature_names':tfeaturenames,
               'magcol':tmagcol,
               'objectids':tobjectids,
               'kwargs':{'test_fraction':test_fraction,
                         'n_crossval_iterations':n_crossval_iterations,
                         'n_kfolds':n_kfolds,
                         'crossval_scoring_metric':crossval_scoring_metric,
                         'nworkers':nworkers},
               'testing_features':testing_features,
               'testing_labels':testing_labels,
               'training_features':training_features,
               'training_labels':training_labels,
               'best_classifier':bestclf,
               'best_score':bestclf_score,
               'best_hyperparams':bestclf_hyperparams,
               'best_recall':recscore,
               'best_precision':precscore,
               'best_f1':f1score,
               'best_confmatrix':confmatrix}


    if classifier_to_pickle:

        with open(classifier_to_pickle,'wb') as outfd:
            pickle.dump(outdict, outfd, pickle.HIGHEST_PROTOCOL)


    # return this classifier and accompanying info
    return outdict



def apply_rf_classifier(classifier,
                        varfeaturesdir,
                        outpickle):
    '''This applys an RF classifier trained using train_rf_classifier
    to pickles in varfeaturesdir.

    classifier is the output dict or pickle from get_rf_classifier. This will
    contain a feature_names key that will be used to collect the features from
    the varfeatures pickles in varfeaturesdir.

    varfeaturesdir is a directory where varfeatures pickles generated by
    lcproc.parallel_varfeatures, etc. are located.

    outpickle is the pickle of the result dict generated by this function.

    '''

    if isinstance(classifier,str) and os.path.exists(classifier):
        with open(classifier,'rb') as infd:
            clfdict = pickle.load(infd)
    elif isinstance(classifier, dict):
        clfdict = classifier
    else:
        LOGERROR("can't figure out the input classifier arg")
        return None


    # get the features to extract from clfdict
    if 'feature_names' not in clfdict:
        LOGERROR("feature_names not present in classifier input, "
                 "can't figure out which ones to extract from "
                 "varfeature pickles in %s" % varfeaturesdir)
        return None


    # extract the features used by the classifier from the varfeatures pickles
    # in varfeaturesdir
