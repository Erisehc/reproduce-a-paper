import sys
sys.path.append('/home/dpopesc2/PycharmProjects/avaeseg')

from filelock import FileLock
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, space_eval, STATUS_FAIL, trials_from_docs
from lifelines import WeibullAFTFitter, CoxPHFitter
# from lifelines.utils import concordance_index
from scar_learning.baseline.baseline_data import load_dataset_ext, adapt_baseline_covariates
from scar_learning.config_data import APNET_COVARIATE_TIME, APNET_COVARIATE_EVENT
from scar_learning.model_evaluation import integrated_brier_score, concordance_index
from sklearn.model_selection import RepeatedStratifiedKFold
import argparse
import glob
import numpy as np
import os
import pandas as pd
import pickle
import scar_learning.config_data as data_config
import tqdm
import warnings


def _concordance_index(model, data_frame, model_type, ancillary_x=None):
    tau = 12
    outcomes = data_config.label_data((data_config.APNET_TRAIN_COHORT, ))
    cd = np.array([[x[0] for x in outcomes.values()], [x[1] for x in outcomes.values()]])

    event_times = np.squeeze(data_frame.as_matrix(columns=[APNET_COVARIATE_TIME]))
    event_observed = np.squeeze(data_frame.as_matrix(columns=[APNET_COVARIATE_EVENT]))

    if model_type == 'cox':
        pred_scores = -model.predict_partial_hazard(data_frame)
    elif model_type == 'weibull':
        pred_scores = model.predict_median(data_frame, ancillary_X=ancillary_x)
    else:
        raise ValueError('Unrecognized model_type: %s' % model_type)

    pred_scores = np.squeeze(pred_scores.values)
    c_idx = concordance_index(tau, event_times, pred_scores, event_observed=event_observed, censor_data=cd)

    return c_idx


def _integrated_brier_score(model, data_frame, ancillary_x=None, t_min=None, t_max=None, bins=100):
    event_times = np.squeeze(data_frame.as_matrix(columns=[APNET_COVARIATE_TIME]))
    event_observed = np.squeeze(data_frame.as_matrix(columns=[APNET_COVARIATE_EVENT]))

    outcomes = data_config.label_data((data_config.APNET_TRAIN_COHORT, ))
    cd = np.array([[x[0] for x in outcomes.values()], [x[1] for x in outcomes.values()]])

    if ancillary_x is not None:
        def surv_fn_t(t):
            dataset = model.predict_survival_function(data_frame, times=[t], ancillary=ancillary_x)
            dataset.fillna(1, inplace=True)
            return dataset.values[0]
    else:
        def surv_fn_t(t):
            dataset = model.predict_survival_function(data_frame, times=[t])
            dataset.fillna(1, inplace=True)
            return dataset.values[0]

    ibs = integrated_brier_score(surv_fn_t, event_times, event_observed, t_min, t_max, bins=bins, censor_data=cd)

    return ibs


def hypersearch_summary(trials_path, sort_by='hyper_score'):
    """
    Summarizes results of hyperparameter sweep in a df.
    :param trials_path: path where all the trials folders are located
    :param sort_by: sort the dataframe by this. 'hyper_score', 'brier_score', and 'concordance_index'
    :return: pd.Dataframe
    """
    df = pd.DataFrame()
    b_scores = []
    c_idx = []
    trials = []
    scores = []
    n_folds = []

    fpath = os.path.join(trials_path, 'trial*/trial_summary.pkl')

    # Unpickle the data and summarize
    for f in sorted(glob.glob(fpath)):
        data = pickle.load(open(f, 'rb'))
        trials.append(int(f.split('/')[-2].replace('trial_', '')))
        c_idx.append(np.mean(data['c_index']))
        b_scores.append(np.mean(data['brier_score']))
        scores.append(np.mean(data['log_likelihood']))
        n_folds.append(len(data.index))

    df['brier_score'] = b_scores
    df['concordance_index'] = c_idx
    df['trial'] = trials
    df['hyper_score'] = scores
    df['n_folds'] = n_folds
    df.set_index('trial', inplace=True)

    if sort_by == 'brier_score' or sort_by == 'hyper_score':
        ascending = True
    elif sort_by == 'concordance_index':
        ascending = False
    else:
        raise ValueError('Unrecognized sort_by argument: %s' % sort_by)

    df.sort_values(sort_by, ascending=ascending, inplace=True)

    print(df.head())
    return df


def cox_train_and_validate_kfold(
        data: pd.DataFrame,
        model_parameters: dict,
        nfolds: int,
        nrepeats: int = 1,
        filename: str = None
):
    events = np.squeeze(data.as_matrix(columns=[APNET_COVARIATE_EVENT]))
    kfold_object = RepeatedStratifiedKFold(n_splits=nfolds, n_repeats=nrepeats, random_state=2020)

    scores = []
    for (i, (train_index, test_index)) in enumerate(kfold_object.split(data, events)):
        x_train = data.iloc[train_index]
        x_test = data.iloc[test_index]

        cox_ph = CoxPHFitter(penalizer=10 ** model_parameters['penalizer_exp'])
        cox_ph.fit(x_train, duration_col=APNET_COVARIATE_TIME, event_col=APNET_COVARIATE_EVENT)

        c_idx = _concordance_index(cox_ph, x_test, 'cox')
        ibs = _integrated_brier_score(cox_ph, x_test)
        ll = cox_ph.score(x_test)

        scores.append({'c_index': c_idx, 'brier_score': ibs, 'log_likelihood': ll})

    all_results = pd.DataFrame.from_records(scores)

    if filename:
        all_results.to_pickle(os.path.join(filename, 'trial_summary.pkl'))
        pickle.dump(model_parameters, open(os.path.join(filename, 'model_params.pkl'), 'wb'))

    result = all_results.mean().to_dict()

    return result


def cox_hypersearch_optimal_params_kfold(
        data: pd.DataFrame,
        fixed_space,
        search_space,
        val_nfolds=10,
        val_nrepeats=10,
        hyper_niter=100,
        trials_save_fpath='',
):
    """
    Searches for optimal parameters and returns them
    :param data: list of ids to use as data
    :param fixed_space: dictionary of fixed model parameters
    :param search_space: dictionary of model parameters over which to optimize
    :param val_nfolds: number of folds for validation kfold
    :param val_nrepeats: number of repetitions for validation kfold
    :param hyper_niter: number of iterations of the hyperparameter search
    :param trials_save_fpath: whether to save down the trials object
    :return: dict
    """

    def objective_fn_gen(idx):
        if idx is not None:
            this_trial_save_path = os.path.join(trials_save_fpath, 'trial_%d' % idx)
            os.mkdir(this_trial_save_path)
        else:
            this_trial_save_path = ''

        def objective_fn(search_params):
            model_parameters = {**fixed_space, **search_params}
            result = cox_train_and_validate_kfold(
                data,
                model_parameters,
                nfolds=val_nfolds,
                nrepeats=val_nrepeats,
                filename=this_trial_save_path
            )
            score = -result['log_likelihood']

            if np.isnan(score):
                return {'loss': np.nan, 'status': STATUS_FAIL}
            else:
                return {'loss': score, 'status': STATUS_OK}

        return objective_fn

    trials = Trials()

    if trials_save_fpath:
        trial_idx_path = os.path.join(trials_save_fpath, 'trial_idx.pkl')
        if not os.path.exists(trial_idx_path):
            pickle.dump(list(range(hyper_niter)), open(trial_idx_path, 'wb'))

        while True:
            with FileLock('%s.lock' % trial_idx_path):
                # Read the available indices
                available_indices = pickle.load(open(trial_idx_path, 'rb'))

                # If ran out of indices, stop
                if not available_indices:
                    break

                # Extract index
                this_idx = available_indices[0]

                # Record new indices
                del available_indices[0]
                pickle.dump(available_indices, open(trial_idx_path, 'wb'))

            print('Running trial %d' % this_idx)
            # Get existing trials
            trials_path = os.path.join(trials_save_fpath, 'trial_info.pkl')
            existing_trials = pickle.load(open(trials_path, 'rb')) if os.path.exists(trials_path) else Trials()

            # Perform one trial
            fmin(
                objective_fn_gen(this_idx),
                search_space,
                return_argmin=False,
                algo=tpe.suggest,
                max_evals=len(existing_trials.trials) + 1,
                trials=existing_trials,
                catch_eval_exceptions=True,
                show_progressbar=False,
            )

            # Ensure new trial has the correct index
            new_trial = existing_trials.trials[-1]

            new_trial['tid'] = this_idx
            new_trial['misc']['tid'] = this_idx
            for k in new_trial['misc']['idxs'].keys():
                new_trial['misc']['idxs'][k] = [this_idx]

            with FileLock('%s.lock' % trials_path):
                # Reload existing trials in case they changed due to parallel processes
                existing_trials = pickle.load(open(trials_path, 'rb')) if os.path.exists(trials_path) else Trials()
                # Update trials and save down
                trials = trials_from_docs(list(existing_trials) + [new_trial])
                pickle.dump(trials, open(trials_path, 'wb'))

        best = trials.argmin

    else:
        best = fmin(objective_fn_gen(None), search_space, algo=tpe.suggest, max_evals=hyper_niter, trials=trials)

    opt_params = space_eval(search_space, best)

    return opt_params


def cox_hypersearch_optimal_params_holdout(data: pd.DataFrame, fixed_space: dict, search_space: dict):
    """
    Searches for optimal parameters and returns them
    :param data: list of ids to use as data
    :param fixed_space: dictionary of fixed model parameters
    :param search_space: dictionary of model parameters over which to optimize
    :return: dict
    """

    x_test_idx = np.random.choice(len(data), 40, replace=False)
    x_test_train = np.setdiff1d(range(len(data)), x_test_idx)
    x_test = data.iloc[x_test_idx]
    x_train = data.iloc[x_test_train]

    def objective_fn(search_params):
        model_parameters = {**fixed_space, **search_params}

        # Training + Validation:
        cox_ph_model = CoxPHFitter(penalizer=10 ** model_parameters['penalizer_exp'])
        cox_ph_model.fit(x_train, duration_col=APNET_COVARIATE_TIME, event_col=APNET_COVARIATE_EVENT)

        # Testing:
        c_idx = _concordance_index(cox_ph_model, x_test, 'cox')

        return {'loss': -c_idx, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(objective_fn, search_space, algo=tpe.suggest, max_evals=50, trials=trials, verbose=0)
    opt_params = space_eval(search_space, best)

    return opt_params


def cox_test_kfold_fixed(
        model_id,
        nfolds,
        nrepeats,
        verbose,
        save_path,
        retrain=False,
        val_test='validation'
):
    """
    Evaluate Cox model via k-fold.
    :param model_id: Id to the model's parameters
    :param nfolds: number of folds
    :param nrepeats: number of repetitions
    :param verbose: printouts?
    :param save_path: where to store the results
    :param retrain: whether to re-train based on original data or CV on new data only
    :param val_test: which data to use
    :return:
    """

    if save_path:
        save_path = os.path.join(
            save_path,
            'baseline_cox_ph',
            'trial_%s' % model_id
        )
    if save_path:
        os.mkdir(save_path)

    model_params = pickle.load(open(os.path.join(
        data_config.HYPERSEARCH_RESULTS_SRC_PATH,
        'baseline_cox_ph',
        'trial_%s' % model_id,
        'model_params.pkl'
    ), 'rb'))

    int_dataset = load_dataset_ext(cohort=(data_config.APNET_TRAIN_COHORT,))
    ext_dataset = load_dataset_ext(cohort=(data_config.APNET_TEST_COHORT,))
    req_vars = [APNET_COVARIATE_TIME, APNET_COVARIATE_EVENT]
    opt_vars = adapt_baseline_covariates()

    int_dataset = int_dataset[req_vars + opt_vars]
    ext_dataset = ext_dataset[req_vars + opt_vars]

    if val_test == 'test':
        data_to_fold = ext_dataset.copy()
    else:
        data_to_fold = int_dataset.copy()
        retrain = False

    events = np.squeeze(data_to_fold[APNET_COVARIATE_EVENT].values)
    kfold_object = RepeatedStratifiedKFold(n_splits=nfolds, n_repeats=nrepeats, random_state=2020)

    fold_data = []
    iterator = tqdm.tqdm(enumerate(kfold_object.split(data_to_fold, events)), total=nfolds * nrepeats)
    for (i, (train_index, test_index)) in iterator:
        if retrain:
            x_train = pd.concat([int_dataset, data_to_fold.iloc[train_index]])
        else:
            x_train = data_to_fold.iloc[train_index]
        x_test = data_to_fold.iloc[test_index]

        # Drop columns of zero
        zero_idx = (x_train != 0).any(axis=0)
        x_train = x_train.loc[:, zero_idx]
        x_test = x_test.loc[:, zero_idx]

        # Training:
        cox_ph_model = CoxPHFitter(penalizer=10 ** model_params['penalizer_exp'])
        cox_ph_model.fit(x_train, duration_col=APNET_COVARIATE_TIME, event_col=APNET_COVARIATE_EVENT)

        # Testing:
        c_idx = _concordance_index(cox_ph_model, x_test, 'cox')
        ibs = _integrated_brier_score(cox_ph_model, x_test)
        ll = cox_ph_model.score(x_test)

        fold_data.append({
            'fold': i,
            'c_index': c_idx,
            'brier_score': ibs,
            'log_likelihood': ll
        })
        if save_path:
            pickle.dump(
                cox_ph_model,
                open(os.path.join(save_path, 'baseline_model_weights_%d.h5' % i), 'wb')
            )
            pickle.dump(
                {'train': x_train, 'validation': x_test},
                open(os.path.join(save_path, 'data_split_%d.pkl' % i), 'wb')
            )

    all_results = pd.DataFrame.from_records(fold_data)

    if verbose:
        print(all_results.mean().to_dict())

    if save_path:
        all_results.to_pickle(os.path.join(save_path, 'trial_summary.pkl'))
        pickle.dump(model_params, open(os.path.join(save_path, 'model_params.pkl'), 'wb'))

    return all_results.mean().to_dict()


def wb_train_and_validate_kfold(
        data: pd.DataFrame,
        model_parameters,
        nfolds: int,
        nrepeats: int = 1,
        filename: str = None
):
    events = np.squeeze(data.as_matrix(columns=[APNET_COVARIATE_EVENT]))
    kfold_object = RepeatedStratifiedKFold(n_splits=nfolds, n_repeats=nrepeats, random_state=123456)

    scores = []
    for (i, (train_index, test_index)) in enumerate(kfold_object.split(data, events)):
        x_train = data.iloc[train_index]
        x_test = data.iloc[test_index]

        aft = WeibullAFTFitter(
            penalizer=10 ** model_parameters['penalizer_exp'],
            l1_ratio=model_parameters['l1_ratio']
        )
        aft.fit(x_train, duration_col=APNET_COVARIATE_TIME, event_col=APNET_COVARIATE_EVENT, ancillary_df=True)

        integ_brier = _integrated_brier_score(aft, x_test, ancillary_x=x_test)
        c_idx = _concordance_index(aft, x_test, 'weibull', ancillary_x=x_test)

        scores.append({'brier_score': integ_brier, 'c_index': c_idx})

    all_results = pd.DataFrame.from_records(scores)

    if filename:
        all_results.to_pickle(filename)

    result = all_results.mean().to_dict()

    return result


def wb_hypersearch_optimal_params_kfold(data: pd.DataFrame, nfolds: int, fixed_space: dict, search_space: dict):
    """
    Searches for optimal parameters and returns them
    :param data: list of ids to use as data
    :param nfolds: number of folds
    :param fixed_space: dictionary of fixed model parameters
    :param search_space: dictionary of model parameters over which to optimize
    :return: dict
    """

    def objective_fn(search_params):
        model_parameters = {**fixed_space, **search_params}
        result = wb_train_and_validate_kfold(data, model_parameters, nfolds)

        return {'loss': result['brier_score'], 'status': STATUS_OK, 'c_index': result['c_index']}

    trials = Trials()
    best = fmin(objective_fn, search_space, algo=tpe.suggest, max_evals=50, trials=trials, verbose=0)
    opt_params = space_eval(search_space, best)

    return opt_params


def wb_hypersearch_optimal_params_holdout(data: pd.DataFrame, fixed_space: dict, search_space: dict):
    """
    Searches for optimal parameters and returns them
    :param data: list of ids to use as data
    :param fixed_space: dictionary of fixed model parameters
    :param search_space: dictionary of model parameters over which to optimize
    :return: dict
    """

    x_test_idx = np.random.choice(len(data), 40, replace=False)
    x_test_train = np.setdiff1d(range(len(data)), x_test_idx)
    x_test = data.iloc[x_test_idx]
    x_train = data.iloc[x_test_train]

    def objective_fn(search_params):
        model_parameters = {**fixed_space, **search_params}

        # Training + Validation:
        aft_model = WeibullAFTFitter(
            penalizer=10 ** model_parameters['penalizer_exp'],
            l1_ratio=model_parameters['l1_ratio']
        )
        aft_model.fit(x_train, duration_col=APNET_COVARIATE_TIME, event_col=APNET_COVARIATE_EVENT, ancillary_df=x_train)

        integ_brier = _integrated_brier_score(aft_model, x_test, ancillary_x=x_train)
        c_idx = _concordance_index(aft_model, x_test, 'weibull', ancillary_x=x_test)

        return {'loss': integ_brier, 'status': STATUS_OK, 'c_index': c_idx}

    trials = Trials()
    best = fmin(objective_fn, search_space, algo=tpe.suggest, max_evals=50, trials=trials, verbose=0)
    opt_params = space_eval(search_space, best)

    return opt_params


def wb_test_kfold(data: pd.DataFrame, nfolds: int, fixed_space: dict, search_space: dict):
    """
    Evaluate Weibull model via k-fold.
    :param data: dataframe with all the data used in regression
    :param nfolds: Number of folds
    :param fixed_space: fixed parameters
    :param search_space: parameters derived from hyperparameter optimization
    :return: score for the model
    """

    events = np.squeeze(data.as_matrix(columns=[APNET_COVARIATE_EVENT]))
    kfold_object = RepeatedStratifiedKFold(n_splits=nfolds, n_repeats=1)

    scores = []
    for (i, (train_index, test_index)) in tqdm.tqdm(enumerate(kfold_object.split(data, events)), total=nfolds):
        x_train = data.iloc[train_index]
        x_test = data.iloc[test_index]

        # Training + Validation:
        if search_space:
            model_optim_parameters = wb_hypersearch_optimal_params_kfold(x_train, nfolds, fixed_space, search_space)
        else:
            model_optim_parameters = {}

        model_parameters = {**fixed_space, **model_optim_parameters}
        aft_model = WeibullAFTFitter(
            penalizer=10 ** model_parameters['penalizer_exp'],
            l1_ratio=model_parameters['l1_ratio']
        )
        aft_model.fit(x_train, duration_col=APNET_COVARIATE_TIME, event_col=APNET_COVARIATE_EVENT, ancillary_df=x_train)

        integ_brier = _integrated_brier_score(aft_model, x_test, ancillary_x=x_train)
        c_idx = _concordance_index(aft_model, x_test, 'weibull', ancillary_x=x_test)

        scores.append({'brier_score': integ_brier, 'c_index': c_idx})

    result = pd.DataFrame.from_records(scores).mean().to_dict()

    return result


def main(mode, model_id, save_path_name):
    save_path_name = os.path.join(save_path_name, mode)
    if mode == 'hypersearch':
        dataset = load_dataset_ext()
        req_vars = [APNET_COVARIATE_TIME, APNET_COVARIATE_EVENT]
        opt_vars = adapt_baseline_covariates()
        dataset = dataset[req_vars + opt_vars]

        fixed_params = {}
        search_params = {'penalizer_exp': hp.uniform('penalizer_exp', -2, 3)}

        # Hypersearch
        cox_hypersearch_optimal_params_kfold(
            data=dataset,
            fixed_space=fixed_params,
            search_space=search_params,
            val_nfolds=10,
            val_nrepeats=10,
            hyper_niter=100,
            trials_save_fpath=save_path_name,
        )
    elif mode == 'validation':
        # validate
        cox_test_kfold_fixed(
            model_id=model_id,
            nfolds=10,
            nrepeats=10,
            verbose=1,
            save_path=save_path_name,
            retrain=False,
            val_test=mode
        )
    elif mode == 'test':
        # Test
        cox_test_kfold_fixed(
            model_id=model_id,
            nfolds=10,
            nrepeats=10,
            verbose=1,
            save_path=save_path_name,
            retrain=True,
            val_test=mode
        )
    else:
        raise ValueError('Unrecognized mode: %s' % mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", help="ID for the model", type=str, default='')
    parser.add_argument("--validate_test", help="whether to validate or test", type=str, default='test')
    parser.add_argument(
        "--save_path",
        help="where to save output",
        type=str,
        default=os.path.join(data_config.SCAR_LEARNING_SRC_PATH, 'apnet', 'results')
    )
    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main(args.validate_test, args.model_id, args.save_path)
