import sys
sys.path.append('/home/dpopesc2/PycharmProjects/avaeseg')

from scar_learning.apnet.apnet_data import get_validation_data
from scar_learning.apnet.apnet_evaluate_fns import apnet_evaluate_atom
from scar_learning.apnet.apnet_model import compiled_model_from_params
from scar_learning.apnet.apnet_train_fns import apnet_train_atom, apnet_train_and_validate_kfold
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
import argparse
import numpy as np
import os
import pandas as pd
import pickle
import scar_learning.config_data as data_config
import tqdm
import warnings


def test_overall(
        model_id,
        verbose,
        val_split,
        ancillary,
        save_path,
        ensemble,
):
    """
    Test with warmed-up weights.
    :param model_id: model id from hypersearch
    :param verbose: printouts?
    :param val_split: how much to reserve for validation
    :param ancillary: whether model is imaging or ancillary
    :param save_path: where to store the results
    :param ensemble: whether to run in ensemble mode
    :return:
    """
    model_type = 'ens' if ensemble else 'aux' if ancillary else 'conv'
    wt_key = 'ensemble' if ensemble else 'ancillary' if ancillary else 'convolutional'
    trial_path = os.path.join(
        data_config.VAL_RESULTS_SRC_PATH,
        'apnet_%s_%s' % (model_type, data_config.APNET_MODEL),
        'trial_%s' % model_id
    )
    model_params = pickle.load(open(os.path.join(trial_path, 'overall_model_params.pkl'), 'rb'))
    model_weights = os.path.join(trial_path, 'overall_apnet_model_weights.h5')
    model_params['model_weights'] = {wt_key: model_weights}

    if save_path:
        save_path = os.path.join(
            save_path,
            'apnet_%s_%s' % (model_type, data_config.APNET_MODEL),
            'trial_%s' % model_id
        )
        os.mkdir(save_path)

    train_val_data = data_config.label_data((data_config.APNET_TRAIN_COHORT,))
    x_train = sorted(list(train_val_data.keys()))
    x_test = sorted(list(data_config.label_data((data_config.APNET_TEST_COHORT,))))

    apnet_model, _ = compiled_model_from_params(
        model_params,
        ancillary,
        ensemble,
        verbose,
    )
    apnet_model.load_weights(model_weights)
    # Testing:
    test_data = get_validation_data(x_test, verbose=verbose, ancillary=ancillary, ensemble=ensemble)
    results = apnet_evaluate_atom(apnet_model, test_data[0], test_data[1], single_output=ensemble or ancillary)

    print(results)

    if save_path:
        apnet_model.save_weights(os.path.join(save_path, 'overall_apnet_model_weights.h5'))
        pickle.dump(
            {'train': x_train, 'validation': x_test},
            open(os.path.join(save_path, 'overall_data_split.pkl'), 'wb')
        )
        pickle.dump(results, open(os.path.join(save_path, 'overall_trial_summary.pkl'), 'wb'))
        pickle.dump(model_params, open(os.path.join(save_path, 'overall_model_params.pkl'), 'wb'))


def test_kfold_fixed(
        model_id,
        nfolds,
        nrepeats,
        verbose,
        val_split,
        ancillary,
        save_path,
        ensemble,
        retrain
):
    """
    K-fold testing with hold-out validation in inner fold.
    :param model_id: model id from hypersearch
    :param nfolds: number of folds
    :param nrepeats: number of repetitions
    :param verbose: printouts?
    :param val_split: how much to reserve for validation
    :param ancillary: whether model is imaging or ancillary
    :param save_path: where to store the results
    :param ensemble: whether to run in ensemble mode
    :param retrain: whether we fine-tune or retrain from scratch
    :return:
    """
    model_type = 'ens' if ensemble else 'aux' if ancillary else 'conv'
    wt_key = 'ensemble' if ensemble else 'ancillary' if ancillary else 'convolutional'
    trial_path = os.path.join(
        data_config.VAL_RESULTS_SRC_PATH,
        'apnet_%s_%s' % (model_type, data_config.APNET_MODEL),
        'trial_%s' % model_id
    )
    model_params = pickle.load(open(os.path.join(trial_path, 'overall_model_params.pkl'), 'rb'))
    model_params['model_weights'] = {wt_key: os.path.join(trial_path, 'overall_apnet_model_weights.h5')}

    if ensemble:
        early_stopping = {'monitor': 'loss', 'mode': 'min', 'patience': 50, 'delay': 0}
    else:
        model_params['initial_lr'] = 5e-4 if ancillary else 1e-4
        model_params['optimizer'] = 'Adam'
        early_stopping = {'monitor': 'val_c_idx', 'mode': 'max', 'patience': 50, 'delay': 0}

    if save_path:
        save_path = os.path.join(
            save_path,
            'apnet_%s_%s' % (model_type, data_config.APNET_MODEL),
            'trial_%s' % model_id
        )
        os.mkdir(save_path)

    int_outcomes = data_config.label_data(cohort=(data_config.APNET_TRAIN_COHORT,)) if retrain else []
    int_all_ids = sorted(list(int_outcomes.keys())) if retrain else []
    int_all_events = [int_outcomes[y][1] for y in int_all_ids] if retrain else []

    outcomes = data_config.label_data(cohort=(data_config.APNET_TEST_COHORT,))
    all_ids = sorted(list(outcomes.keys()))
    all_events = [outcomes[y][1] for y in all_ids]

    k_fold_obj = RepeatedStratifiedKFold(n_splits=nfolds, n_repeats=nrepeats, random_state=2020)

    fold_data = []
    iterator = tqdm.tqdm(enumerate(k_fold_obj.split(all_ids, all_events)), total=nfolds * nrepeats)
    for (i, (train_index, test_index)) in iterator:
        x_train = [all_ids[idx] for idx in train_index]
        train_events = [outcomes[y][1] for y in x_train]
        if retrain:
            x_train += int_all_ids
            train_events += int_all_events

        if val_split:
            x_train, x_val = train_test_split(x_train, test_size=val_split, random_state=2020, stratify=train_events)
        else:
            x_val = []
        x_test = [all_ids[idx] for idx in test_index]

        # Training + Validation:
        train_rv = apnet_train_atom(
            data_dict={'train_ids': x_train, 'validation_ids': x_val},
            model_params=model_params,
            early_stopping=early_stopping,
            verbose=verbose,
            ancillary=ancillary,
            ensemble=ensemble
        )

        # if training finished prematurely
        m_hist = train_rv['model'].history.history
        if np.isnan(m_hist['loss'][-1]) or np.isinf(m_hist['loss'][-1]):
            if verbose:
                warnings.warn('Model terminated on nan for %s' % x_train)
            continue

        apnet_model = train_rv['model']
        # Testing:
        test_data = get_validation_data(x_test, verbose=verbose, ancillary=ancillary, ensemble=ensemble)
        metrics = apnet_evaluate_atom(apnet_model, test_data[0], test_data[1], single_output=ensemble or ancillary)

        fold_data.append({
            'fold': i,
            'loss': m_hist['loss'][-(1 + early_stopping['patience'])],
            **metrics,
            'score': m_hist[early_stopping['monitor']][-(1 + early_stopping['patience'])],  # aggregate model score
            'stopped_epoch': train_rv['stopped_epoch']
        })

        if save_path:
            apnet_model.save_weights(os.path.join(save_path, 'apnet_model_weights_%d.h5' % i))
            pickle.dump(
                {'train': x_train, 'validation': x_test},
                open(os.path.join(save_path, 'data_split_%d.pkl' % i), 'wb')
            )

    all_results = pd.DataFrame.from_records(fold_data)

    if save_path:
        all_results.to_pickle(os.path.join(save_path, 'trial_summary.pkl'))
        pickle.dump(model_params, open(os.path.join(save_path, 'model_params.pkl'), 'wb'))

    if len(all_results) < .8 * nrepeats * nfolds:
        warnings.warn('Insufficient folds: %d/100' % len(all_results))
        return np.nan
    else:
        return all_results['score'].mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", help="ID for the model", type=str)
    parser.add_argument("--model_type", help="whether to run conv or dense", type=str, default='dense')
    parser.add_argument("--ensemble", help="whether to ensemble models", type=bool, default=False)
    parser.add_argument("--val_split", help="how much to reserve for early stopping", type=float, default=.1)
    parser.add_argument(
        "--save_path",
        help="where to store the results",
        type=str,
        default=data_config.TEST_RESULTS_SRC_PATH
    )
    parser.add_argument("--gpu_id", help="which gpu to use", type=str)
    parser.add_argument("--gpu_mem_fraction", help="memory fraction for gpu", type=float)
    args = parser.parse_args()

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.gpu_mem_fraction:
        import tensorflow as tf
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        config.gpu_options.per_process_gpu_memory_fraction = args.gpu_mem_fraction
        sess = tf.Session(config=config)

    test_overall(
        args.model_id,
        verbose=2,
        val_split=args.val_split,
        ancillary=args.model_type == 'dense',
        save_path=args.save_path,
        ensemble=args.ensemble,
    )

    test_kfold_fixed(
        args.model_id,
        nfolds=10,
        nrepeats=10,
        verbose=2,
        val_split=args.val_split,
        ancillary=args.model_type == 'dense',
        save_path=args.save_path,
        ensemble=args.ensemble,
        retrain=True
    )
