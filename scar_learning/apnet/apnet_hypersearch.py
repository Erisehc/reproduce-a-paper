import sys
sys.path.append('/home/dpopesc2/PycharmProjects/avaeseg')

from filelock import FileLock
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials, space_eval, trials_from_docs
from hyperopt import hp
from scar_learning.apnet.apnet_evaluate_fns import apnet_early_stopping_score
from scar_learning.apnet.apnet_train_fns import apnet_train_and_validate_kfold, apnet_train_atom
from sklearn.model_selection import train_test_split
import argparse
import glob
import hyperopt.plotting as hyperplt
import numpy as np
import os
import pandas as pd
import pickle
import scar_learning.config_data as data_config
import shutil


def set_trial_index(trial, index):
    """
    Re-index object
    :param trial: Trials.trial object
    :param index: index to replace
    :return: same trial, but different index
    """
    trial['tid'] = index
    trial['misc']['tid'] = index
    for k in trial['misc']['idxs'].keys():
        trial['misc']['idxs'][k] = [index]

    return trial


def re_index_trials(initial_trials, indices=None):
    """
    Given a Trials object, relabel trials with indices
    :param initial_trials: list of trials
    :param indices: hashable indices to use in trials
    :return: Trials() with new indices
    """
    if indices is None:
        indices = list(range(len(initial_trials)))
    else:
        if len(indices) != len(initial_trials):
            raise ValueError('Cannot have different number of trials and indices')

    new_trials = []
    for (t, idx) in zip(initial_trials, indices):
        new_trials.append(set_trial_index(t, idx))

    return new_trials


def combine_results(source_paths, destination_path):
    """
    Takes results from source paths and combines them into destination_path
    :param source_paths: list of results sources
    :param destination_path: where the combined results will be stored
    :return: None
    """
    all_trials = []
    re_index = 0
    for sp in source_paths:
        trial_file = os.path.join(sp, 'trial_info.pkl')
        all_trials += pickle.load(open(trial_file, 'rb')).trials

        for dir_name in os.listdir(sp):
            src = os.path.join(sp, dir_name)
            if os.path.isdir(src):
                dest = os.path.join(destination_path, 'trial_%d' % re_index)
                re_index += 1
                print('Copying from %s to %s' % (src, dest))
                shutil.copytree(src, dest)

    new_trials = trials_from_docs(re_index_trials(all_trials))
    pickle.dump(new_trials, open(os.path.join(destination_path, 'trial_info.pkl'), 'wb'))


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
        scores.append(np.mean(data['score']))
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


def find_best_trial(model_type, criterion, search_path=data_config.HYPERSEARCH_RESULTS_SRC_PATH):
    """
    Return the model performing best in terms of
    :param model_type: which model to investigate
    :param criterion: sort the dataframe by this. 'hyper_score', 'brier_score', and 'concordance_index'
    :return:
    """
    tp = os.path.join(search_path, model_type)
    df = hypersearch_summary(tp, criterion)
    hyper_model = df.iloc[0].name

    return hyper_model


def find_best_weights_index(model_type, criterion, search_path = data_config.HYPERSEARCH_RESULTS_SRC_PATH):
    """
    Finds the index of with the best performance in terms of crterion
    :param model_type: which model to investigate
    :param criterion: sort the dataframe by this. 'hyper_score', 'brier_score', and 'concordance_index'
    :return:
    """
    arch_type = model_type.split('_')[1]
    if arch_type == 'ens':
        idx_aux = find_best_weights_index(model_type.replace(arch_type, 'aux'), criterion, search_path)
        idx_conv = find_best_weights_index(model_type.replace(arch_type, 'conv'), criterion, search_path)
        return idx_aux, idx_conv

    tp = os.path.join(search_path, model_type)
    hyper_model = find_best_trial(model_type, criterion, search_path)
    hyper_model_path = os.path.join(tp, 'trial_%d' % hyper_model)
    ts = pickle.load(open(os.path.join(hyper_model_path, 'trial_summary.pkl'), 'rb'))
    if criterion == 'brier_score':
        ascending = True
        sort_by = criterion
    elif criterion == 'hyper_score':
        ascending = True
        sort_by = 'score'
    elif criterion == 'concordance_index':
        sort_by = 'c_index'
        ascending = False
    else:
        raise ValueError('Unrecognized sort_by argument: %s' % criterion)

    ts.sort_values(sort_by, ascending=ascending, inplace=True)
    best_weights_idx = int(ts.iloc[0].fold)

    return best_weights_idx


def main_plot_vars(
        trials,
        do_show=True,
        fontsize=10,
        colorize_best=None,
        columns=5,
        arrange_by_loss=False
):
    """
    Adapted from hyperopt for better controls on colors
    :param trials:
    :param do_show:
    :param fontsize:
    :param colorize_best:
    :param columns:
    :param arrange_by_loss:
    :return:
    """
    # -- import here because file-level import is too early
    import matplotlib.pyplot as plt

    idxs, vals = hyperplt.miscs_to_idxs_vals(trials.miscs)
    losses = trials.losses()
    finite_losses = [y for y in losses if y not in (None, float('inf'))]
    asrt = np.argsort(finite_losses)
    if colorize_best is not None:
        colorize_thresh = finite_losses[asrt[colorize_best + 1]]
    else:
        # -- set to lower than best (disabled)
        colorize_thresh = finite_losses[asrt[0]] - 1

    loss_min = min(finite_losses)
    loss_max = max(finite_losses)
    loss_min_highlight = min([f for f in finite_losses if f < colorize_thresh])
    loss_max_highlight = max([f for f in finite_losses if f < colorize_thresh])
    print('finite loss range', loss_min, loss_max, colorize_thresh)

    loss_by_tid = dict(zip(trials.tids, losses))

    def color_fn(lossval):
        if lossval is None:
            return 1, 1, 1
        else:
            t = 4 * (lossval - loss_min) / (loss_max - loss_min + .0001)
            if t < 1:
                return t, 0, 0
            if t < 2:
                return 2 - t, t - 1, 0
            if t < 3:
                return 0, 3 - t, t - 2
            return 0, 0, 4 - t

    def color_fn_bw(lossval):
        if lossval in (None, float('inf')):
            return 1, 1, 1
        else:
            if lossval < colorize_thresh:
                t = (lossval - loss_min_highlight) / (loss_max_highlight - loss_min_highlight + .0001)
                return .5 + (1. - t) / 2, 0., 0.   # -- red best black worst
            else:
                t = (lossval - loss_min) / (loss_max - loss_min + .0001)
                return t, t, t  # -- white=worst, black=best

    all_labels = list(idxs.keys())
    titles = all_labels
    order = np.argsort(titles)

    C = min(columns, len(all_labels))
    R = int(np.ceil(len(all_labels) / float(C)))

    for plotnum, varnum in enumerate(order):
        label = all_labels[varnum]
        plt.subplot(R, C, plotnum + 1)

        # hide x ticks
        ticks_num, ticks_txt = plt.xticks()
        plt.xticks(ticks_num, [''] * len(ticks_num))

        dist_name = label

        if arrange_by_loss:
            x = [loss_by_tid[ii] for ii in idxs[label]]
        else:
            x = idxs[label]
        if 'log' in dist_name:
            y = np.log(vals[label])
        else:
            y = vals[label]
        plt.title(titles[varnum], fontsize=fontsize)
        c = list(map(color_fn_bw, [loss_by_tid[ii] for ii in idxs[label]]))
        if len(y):
            plt.scatter(x, y, c=c)
        if 'log' in dist_name:
            nums, texts = plt.yticks()
            plt.yticks(nums, ['%.2e' % np.exp(t) for t in nums])

    if do_show:
        plt.show()


def analyze_hypersearch_results(
        results_filepath,
        search_space,
        max_loss_threshold=-np.inf,
):
    trials = pickle.load(open(results_filepath, 'rb'))
    filt_trials = [t for t in trials if t['result']['status'] == 'ok' and t['result']['loss'] > max_loss_threshold]
    trials_obj = trials_from_docs(filt_trials)

    # Plot
    main_plot_vars(trials_obj, arrange_by_loss=False, colorize_best=4)

    # Find best and worst performers
    first = 10
    last = 10
    best_trials = [trials_obj.trials[t]['misc']['vals'] for t in np.argsort(trials_obj.losses())][:first]
    worst_trials = [trials_obj.trials[t]['misc']['vals'] for t in np.argsort(trials_obj.losses())][-last:]

    for i, bt in enumerate(best_trials):
        best_trials[i] = space_eval(search_space, {k: v[0] for (k, v) in bt.items()})

    for i, wt in enumerate(worst_trials):
        worst_trials[i] = space_eval(search_space, {k: v[0] for (k, v) in wt.items()})

    for param in search_space.keys():
        best = [b[param] for b in best_trials]
        most_common_best = max(set(best), key=best.count)
        no_best = best.count(most_common_best)

        worst = [w[param] for w in worst_trials]
        most_common_worst = max(set(worst), key=worst.count)
        no_worst = worst.count(most_common_worst)

        print('Parameter %s: best: %s (%d/%d), worst: %s (%d/%d)' %
              (param, most_common_best, no_best, first, most_common_worst, no_worst, last))


def apnet_hypersearch_optimal_params_holdout(
        data: list,
        fixed_space: dict,
        search_space: dict,
        val_split: float,
        hyper_niter: int,
        trials_save_fname: str = '',
        ancillary: bool = False,
):
    """
    Searches for optimal parameters and returns them
    :param data: list of ids to use as data
    :param fixed_space: dictionary of fixed model parameters
    :param search_space: dictionary of model parameters over which to optimize
    :param val_split: percent data to use for validation under holdout
    :param hyper_niter: number of iterations of the hyperparameter search
    :param trials_save_fname: whether to save down the trials object
    :param ancillary: whether to run main model or ancillary
    :return: dict
    """

    events = [
        data_config.label_data((data_config.APNET_TRAIN_COHORT, data_config.APNET_TEST_COHORT))[y][1] for y in data
    ]
    x_train, x_test = train_test_split(data, test_size=val_split, random_state=2020, stratify=events)

    def objective_fn(search_params):
        apnet_model_parameters = {**fixed_space, **search_params}

        # Training + Validation:
        train_rv = apnet_train_atom(
            data_dict={'train_ids': x_train, 'validation_ids': x_test},
            model_params=apnet_model_parameters,
            early_stopping={'monitor': 'survival'},
            verbose=0,
            ancillary=ancillary
        )
        apnet_model = train_rv['model']

        # Testing:
        final_score = apnet_early_stopping_score(apnet_model.history.history, 3)
        return {
            'loss': final_score,
            'status': STATUS_OK,
            'stopped_epoch': int(train_rv['stopped_epoch']),
        }

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    trials = Trials()
    best = fmin(objective_fn, search_space, algo=tpe.suggest, max_evals=hyper_niter, trials=trials, verbose=0)

    opt_params = space_eval(search_space, best)
    opt_params['epochs'] = trials.best_trial['result']['stopped_epoch']

    if trials_save_fname:
        with open(trials_save_fname, 'wb') as handle:
            pickle.dump(trials, handle)

    return opt_params


def apnet_hypersearch_optimal_params_kfold(
        data: list,
        fixed_space: dict,
        search_space: dict,
        val_nfolds: int,
        val_nrepeats: int,
        hyper_niter: int,
        trials_save_fpath: str = '',
        ancillary: bool = False,
        ensemble: bool = False,
):
    """
    Searches for optimal parameters usign k-fold validation and returns them
    :param data: list of ids to use as data
    :param fixed_space: dictionary of fixed model parameters
    :param search_space: dictionary of model parameters over which to optimize
    :param val_nfolds: number of folds for validation kfold
    :param val_nrepeats: number of repetitions for validation kfold
    :param hyper_niter: number of iterations of the hyperparameter search
    :param trials_save_fpath: whether to save down the trials object
    :param ancillary: whether to run main model or ancillary
    :param ensemble: whether to run ensembled model
    :return: dict
    """
    trials = Trials()

    # Define objective function
    def objective_fn_gen(idx):
        def objective_fn(search_params):
            if idx is not None:
                this_trial_save_path = os.path.join(trials_save_fpath, 'trial_%d' % idx)
                os.mkdir(this_trial_save_path)
            else:
                this_trial_save_path = ''

            apnet_model_parameters = {**fixed_space, **search_params}
            score = apnet_train_and_validate_kfold(
                data,
                apnet_model_parameters,
                nfolds=val_nfolds,
                nrepeats=val_nrepeats,
                filepath=this_trial_save_path,
                verbose=0,
                ancillary=ancillary,
                ensemble=ensemble,
            )
            if np.isnan(score):
                return {
                    'loss': np.nan,
                    'status': STATUS_FAIL,
                }
            else:
                return {
                    'loss': score,
                    'status': STATUS_OK,
                }
        return objective_fn

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

    else:  # create a new trials object and start searching
        best = fmin(
            objective_fn_gen(None),
            search_space,
            algo=tpe.suggest,
            max_evals=hyper_niter,
            trials=trials,
            catch_eval_exceptions=True
        )

    opt_params = space_eval(search_space, best)

    return opt_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", help="whether to run conv or dense", type=str, default='convolutional')
    parser.add_argument("--gpu_id", help="which gpu to use", type=str)
    parser.add_argument("--gpu_mem_fraction", help="memory fraction for gpu", type=float)
    parser.add_argument("--save_path", help="where to save output", type=str)
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.gpu_mem_fraction:
        import tensorflow as tf
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        config.gpu_options.per_process_gpu_memory_fraction = args.gpu_mem_fraction
        sess = tf.Session(config=config)

    if args.save_path is None:
        save_path = '/home/dpopesc2/PycharmProjects/DNN/scar_learning/apnet/results/test'
    else:
        save_path = args.save_path

    if args.model_type == 'convolutional':
        ancillary = False
        hyper_trials = 100
        fs = {
            'epochs': 2000,
            'initial_lr': 1e-2,
            'batch_size': 32,
            'steps_per_epoch': 20,
            'activation': 'relu',
            'use_batch_normalization': False,
            'conv_kernel_size': 3,
            'data_aug_params': {
                'rotation': False,  # degrees to rotate the images in the x-y plane
                'width_shift_pct': .9,  # shift images horizontally (fraction of total available space)
                'height_shift_pct': .9,  # shift images vertically (fraction of total available space)
                'depth_shift_pct': .9,  # shift images in the short axis plane (fraction of total available space)
                'gaussian_noise_std': 0,  # standard deviation of gaussian noise applied to input
            },
            'weight_scalar': 0,
        }

        ss = {
            'network_depth': hp.choice('network_depth', [2, 4]),
            'risk_categories': hp.choice('risk_categories', [1, 2, 3]),
            'dropout_value': hp.uniform('dropout_value', 0, .5),
            'l1_reg': hp.loguniform('l1_reg', np.log(10 ** (-7)), np.log(10 ** (-3))),
            'l2_reg': hp.loguniform('l2_reg', np.log(10 ** (-6)), np.log(10 ** (-2))),
            'reconstruction_loss_wt': hp.uniform('reconstruction_loss_wt', 0, 10),
            'no_convolutions': hp.choice('no_convolutions', [1, 2]),
            'conv_filter_no_init': hp.choice('conv_filter_no_init', [8, 12, 16, 20, 24]),
            'latent_representation_dim': hp.quniform('latent_representation_dim', 10, 30, 1),
        }
    elif args.model_type == 'dense':
        ancillary = True
        hyper_trials = 300

        fs = {
            'epochs': 2000,
            'initial_lr': 1e-2,
            'batch_size': 32,
            'steps_per_epoch': 20,
            'activation': 'relu',
            'use_batch_normalization': False,
            'weight_scalar': 0,
        }

        ss = {
            'network_depth': hp.choice('network_depth', [1, 2, 3, 4, 5]),
            'risk_categories': hp.choice('risk_categories', [1, 2, 3]),
            'dropout_value': hp.uniform('dropout_value', 0, .5),
            'l1_reg': hp.loguniform('l1_reg', np.log(10 ** (-7)), np.log(10 ** (-3))),
            'l2_reg': hp.loguniform('l2_reg', np.log(10 ** (-6)), np.log(10 ** (-2))),
            'no_units': hp.quniform('no_units', 4, 24, 1),
        }
    else:
        raise argparse.ArgumentError(args.model_type, 'Unrecognized model type')

    apnet_hypersearch_optimal_params_kfold(
        data=sorted(list(data_config.label_data((data_config.APNET_TRAIN_COHORT,)).keys())),
        fixed_space=fs,
        search_space=ss,
        val_nfolds=10,
        val_nrepeats=10,
        hyper_niter=hyper_trials,
        trials_save_fpath=save_path,
        ancillary=ancillary,
    )
