from scar_learning.config_data import APNET_ANCILLARY_COVARIATES
from scar_learning.image_processing import encoding_functions as ef
import scar_learning.config_data as data_config


def adapt_baseline_covariates():
    # 'ethnicity is dummy and ischemic_etiology is all 1s
    covars = [v for v in APNET_ANCILLARY_COVARIATES if v not in ['ethnicity', 'ischemic_etiology']]
    covars.insert(2, 'ethnicity_2')
    covars.insert(2, 'ethnicity_1')

    return covars


def load_dataset_ext(cohort=(data_config.APNET_TRAIN_COHORT,)):
    return ef.encode_ancillary_data(data_config.ancillary_data(include_surv=True, cohort=cohort))
