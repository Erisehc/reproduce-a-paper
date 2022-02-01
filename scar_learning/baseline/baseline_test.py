import sys
sys.path.append('/home/dpopesc2/PycharmProjects/avaeseg')

from lifelines import CoxPHFitter
from scar_learning.baseline.baseline_data import load_dataset_ext, adapt_baseline_covariates
from scar_learning.baseline.baseline_model import _concordance_index, _integrated_brier_score
from scar_learning.config_data import APNET_COVARIATE_TIME, APNET_COVARIATE_EVENT
import argparse
import os
import pickle
import scar_learning.config_data as data_config
import warnings


def test_overall(model_id, validate_test, save_path):
    model_params = pickle.load(open(os.path.join(
        data_config.HYPERSEARCH_RESULTS_SRC_PATH,
        'baseline_cox_ph',
        'trial_%s' % model_id,
        'model_params.pkl'
    ), 'rb'))
    x_train = load_dataset_ext(cohort=(data_config.APNET_TRAIN_COHORT,))
    req_vars = [APNET_COVARIATE_TIME, APNET_COVARIATE_EVENT]
    opt_vars = adapt_baseline_covariates()
    x_train = x_train[req_vars + opt_vars]

    best_model = CoxPHFitter(penalizer=10 ** model_params['penalizer_exp'])
    best_model.fit(x_train, duration_col=APNET_COVARIATE_TIME, event_col=APNET_COVARIATE_EVENT)

    if validate_test == 'test':
        x_test = load_dataset_ext(cohort=(data_config.APNET_TEST_COHORT,))
        req_vars = [APNET_COVARIATE_TIME, APNET_COVARIATE_EVENT]
        opt_vars = adapt_baseline_covariates()
        x_test = x_test[req_vars + opt_vars]
    else:
        x_test = x_train.copy()

    c_idx = _concordance_index(best_model, x_test, 'cox')
    ibs = _integrated_brier_score(best_model, x_test)
    results = {'c_index': c_idx, 'brier_score': ibs}

    print(results)
    if save_path:
        save_fpath = os.path.join(
            save_path,
            validate_test,
            'baseline_cox_ph',
            'trial_%s' % model_id,
        )
        pickle.dump(
            best_model,
            open(os.path.join(save_fpath, 'overall_baseline_model_weights.h5'), 'wb')
        )
        pickle.dump(
            {'train': x_train, 'validation': x_test},
            open(os.path.join(save_fpath, 'overall_data_split.pkl'), 'wb')
        )
        pickle.dump(
            results,
            open(os.path.join(save_fpath, 'overall_trial_summary.pkl'), 'wb')
        )


if __name__ == '__main__':
    warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", help="ID for the model", type=str)
    parser.add_argument("--validate_test", help="whether to validate or test", type=str, default='test')
    parser.add_argument(
        "--save_path",
        help="where to save output",
        type=str,
        default=os.path.join(data_config.SCAR_LEARNING_SRC_PATH, 'apnet', 'results')
    )
    args = parser.parse_args()

    test_overall(args.model_id, args.validate_test, args.save_path)
