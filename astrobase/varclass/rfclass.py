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




#####################################
## NON-PERIODIC VAR CLASSIFICATION ##
#####################################

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


def collect_nonperiodic_varfeatures(
        featuresdir,
        magcol,
        outfile,
        maxobjects=None,
        varflags=None,
):
    '''This collects nonperiodic varfeatures into arrays.

    featuresdir is the directory where all the varfeatures pickles are.

    magcol is the light curve magnitude col key to use when looking inside each
    varfeatures pickle.

    maxobjects controls how many pickles to process.

    This returns a dict with the following keys per magcol:

    'feature_array' = nobjects x nfeatures array
    'feature_labels' = labels for each feature

    This tries to get all the features listed in NONPERIODIC_FEATURES_TO_COLLECT.

    If varflags is not None, it must be a dict with the following key:val
    list:

    '<objectid>':True or False

    This will turn the collected information into a training set for
    classifiers. For a collection of fake LCS prepared by fakelcs.generation,
    use the value of the 'isvariable' dict elem from fakelcs-info.pkl here, like
    so:

    varflags={x:y for x,y in zip(fakelcinfo['objectid'],
                                 fakelcinfo['isvariable'])}

    '''

    # list of input pickles generated by varfeatures in lcproc.py
    pklist = glob.glob(os.path.join(featuresdir, 'varfeatures-*.pkl'))

    if maxobjects:
        pklist = pklist[:maxobjects]


    # fancy progress bar with tqdm if present
    if TQDM:
        listiterator = tqdm(pklist)
    else:
        listiterator = pklist

    # go through all the varfeatures arrays

    feature_dict = {'objectids':[],'magcol':magcol, 'featurelist':[]}

    LOGINFO('collecting features for magcol: %s' % magcol)

    for pkl in listiterator:

        with open(pkl,'rb') as infd:
            varf = pickle.load(infd)

        # update the objectid list
        objectid = varf['objectid']
        if objectid not in feature_dict['objectids']:
            feature_dict['objectids'].append(objectid)

        thisfeatures = varf[magcol]

        # collect all the features for this magcol/objectid combination
        for feature in NONPERIODIC_FEATURES_TO_COLLECT:

            # update the global feature list if necessary
            if ((feature not in feature_dict['featurelist']) and
                (feature in thisfeatures)):

                feature_dict['featurelist'].append(feature)
                feature_dict[feature] = []

            if feature in thisfeatures:

                feature_dict[feature].append(
                    thisfeatures[feature]
                )

    # now that we've collected all the objects and their features, turn the list
    # into arrays, and then concatenate them
    for feat in feature_dict['featurelist']:
        feature_dict[feat] = np.array(feature_dict[feat])

    feature_dict['objectids'] = np.array(feature_dict['objectids'])

    feature_array = np.column_stack([feature_dict[feat] for feat in
                                     feature_dict['featurelist']])
    feature_dict['feature_array'] = feature_array

    # if there's a varflag dict available, use it to generate a class array
    if isinstance(varflags, dict):

        flagarray = np.zeros(feature_dict['objectids'].size, dtype=np.int64)

        for ind, objectid in enumerate(feature_dict['objectids']):
            if objectid in varflags:
                if varflags[objectid]:
                    flagarray[ind] = 1

        feature_dict['varflag_array'] = flagarray


    # write the info to the output pickle
    with open(outfile,'wb') as outfd:
        pickle.dump(feature_dict, outfd, pickle.HIGHEST_PROTOCOL)

    # return the feature_dict
    return feature_dict





#################################
## TRAINING THE RF CLASSIFIERS ##
#################################


def get_rf_classifier(
        tfeatures,
        tlabels,
        test_fraction=0.25,
        n_crossval_iterations=20,
        n_kfolds=5,
        crossval_scoring_metric='f1',
        classifier_to_pickle=None,
        nworkers=-1,
):
    '''
    This gets the best RF classifier after running cross-validation.

    - splits the training set into test/train samples
    - does KFold stratified cross-validation using RandomizedSearchCV
    - gets the randomforest with the best performance after CV
    - gets the confusion matrix for the test set

    '''

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


    # run the stratified kfold cross-validation on training features
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


    LOGINFO('running grid-search CV...')
    cvsearch_classifiers = cvsearch.fit(training_features,
                                        training_labels)

    # report on the classifiers' performance
    gridsearch_report(cvsearch_classifiers.grid_scores_)

    # get the best classifier after CV
    bestclf = cvsearch_classifiers.best_estimator_
    bestclf_score = cvsearch_classifiers.best_score_
    bestclf_hyperparams = cvsearch_classifiers.best_params_

    # test this classifier on the testing set
    test_predicted_labels = bestclf.predict(test_features)

    recscore = recall_score(testing_labels, test_predicted_labels)
    precscore = precision_score(testing_labels,test_predicted_labels)
    f1score = f1_score(testing_labels, test_predicted_labels)
    confmatrix = confusion_matrix(testing_labels, test_predicted_labels)


    # write the classifier, its training/testing set, and its stats to the
    # pickle if requested
    outdict = {'all_features':tfeatures,
               'all_labels':tlabels,
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
