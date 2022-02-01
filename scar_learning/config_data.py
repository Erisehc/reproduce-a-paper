"""
Data setup script. Returns partition, which governs which patients are part of the training set
and which patients are part of the test set. This dictates which patients will have an event and which will not.
"""

import numpy as np
import csv
import os
from definitions import ROOT_DIR
import pandas as pd


SCAR_LEARNING_SRC_PATH = os.path.join(ROOT_DIR, 'scar_learning')
HYPERSEARCH_RESULTS_SRC_PATH = os.path.join(SCAR_LEARNING_SRC_PATH, 'apnet', 'results', 'hypersearch')
VAL_RESULTS_SRC_PATH = os.path.join(SCAR_LEARNING_SRC_PATH, 'apnet', 'results', 'validation')
TEST_RESULTS_SRC_PATH = os.path.join(SCAR_LEARNING_SRC_PATH, 'apnet', 'results', 'test')
DATA_PATH_ORIGINAL = os.path.join(SCAR_LEARNING_SRC_PATH, 'data', 'data_original')
LABEL_FILE_NAME = os.path.join(SCAR_LEARNING_SRC_PATH, 'data', 'outcomes_20180801_copy.csv')
DATA_FILE_PATH = os.path.join(SCAR_LEARNING_SRC_PATH, 'data', 'hv_64')
BASELINE_COV_FILE_NAME = os.path.join(SCAR_LEARNING_SRC_PATH, 'data', 'covariates.pickle')
OUTPUT_SPECS = {'output_resolution': (64, 64, 12), 'output_spatial_resolution': (2.25, 2.25, 10)}
OUTPUT_CHANNELS = 2

APNET_ANCILLARY_COVARIATES = [
    'age',
    'sex',
    'ethnicity',
    'smoker',
    'diabetes',
    'hypertension',
    'cholesterol',
    'lvef_noncmr',
    'lvef_cmr',
    'lv_mass_ed',
    'infarct_pct',
    'ekg_hr',
    'ekg_lbbb',
    'ekg_atrial_fib',
    'ekg_qrs_dur',
    'ischemic_etiology',
    'ischemic_etiology_time',
    'betablock',
    'acearb',
    'lipid_lower',
    'diuretic',
    'anti_arrhyth',
    'digoxin',
]
APNET_ANCILLARY_COVARIATES_NO = len(APNET_ANCILLARY_COVARIATES) + 1  # accounting for dummy variables

APNET_COVARIATE_TIME = 'event_time'
APNET_COVARIATE_EVENT = 'event_type'
APNET_MODEL = 'loglogistic'  # 'weibull',  'loglogistic', or 'cox_ph'
APNET_TRAIN_COHORT = 'prose'
APNET_TEST_COHORT = 'predetermine'


def label_data(cohort: tuple = (APNET_TRAIN_COHORT, APNET_TEST_COHORT)) -> dict:
    """
    Reads the csv file with outcome data and puts in in the form of a dictionary.
    Keys are patient ids and values are tuples of min(censor time, event time) and event indicaor
    :param cohort: which cohort to return
    :return: dict of labels.
    """
    with open(LABEL_FILE_NAME, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.DictReader(csvfile)

        # extracting each data row one by one
        labels = {
            r['patient_uniq_id']:
                tuple([float(r['event_time']), int(r['event_type'])]) for r in csvreader if r['cohort'] in cohort
        }

    return labels


def ancillary_data(cohort: tuple = (APNET_TRAIN_COHORT, APNET_TEST_COHORT), include_surv: bool = False) -> dict:
    """
    Reads the csv file with outcome data and puts in in the form of a dictionary.
    Keys are patient ids and values are tuples of min(censor time, event time) and event indicaor
    :param cohort: which cohort to return
    :param include_surv: whether to return survival columns (time and censoring status)
    :return: dict of labels.
    """

    raw_dataset = pd.read_csv(
        LABEL_FILE_NAME,
        na_values="?",
        comment='\t',
        skipinitialspace=True,
        header=0
    )
    dataset = raw_dataset.copy()
    dataset.set_index('patient_uniq_id', inplace=True)

    # Filter by cohort
    dataset = dataset[dataset['cohort'].isin(cohort)]

    # Extract covariates
    all_cov = APNET_ANCILLARY_COVARIATES.copy()
    if include_surv:
        all_cov += [APNET_COVARIATE_EVENT, APNET_COVARIATE_TIME]

    dataset = dataset[all_cov]
    dataset = dataset.replace(-1, np.nan)
    dataset.fillna(dataset.mean(), inplace=True)

    return dataset
