"""
In this script, we define the necessary functions to train the model
"""
from keras.callbacks import TerminateOnNaN
from scar_learning.apnet.apnet_data import get_training_generator
from scar_learning.apnet.apnet_data import get_validation_data
from scar_learning.apnet.apnet_evaluate_fns import apnet_evaluate_atom
from scar_learning.apnet.apnet_model import compiled_model_from_params
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
import numpy as np
import os
import pandas as pd
import pickle
import scar_learning.apnet.apnet_callbacks as apnet_cbks
import scar_learning.config_data as data_config
import warnings


def extract_weights(trained_model) -> dict:
    weights = {
        'encoder': trained_model.get_layer('encoder').get_weights(),
        'decoder': trained_model.get_layer('decoder').get_weights(),
        'latent_parameter': trained_model.get_layer('latent_parameter').get_weights(),
    }

    return weights


def apnet_train_atom(
        data_dict: dict,
        model_params: dict,
        early_stopping: dict = None,
        verbose: int = 0,
        ancillary: bool = False,
        ensemble: bool = False,
):
    """
    Trains a single model returned by get_apnet_model using data in data. Optionally,
    implements early stopping using validation data. See below for list of expected model_params
    :param data_dict: dictionary with train_ids and validation_ids
    :param model_params:
    :param early_stopping: config of early stoppping parameters
    :param verbose:
    :param ancillary: use ancillary model
    :param ensemble: whether to return the ensemble model
    :return: dict
    """

    ###########################
    # 1. MODEL DEFINITION
    ###########################
    epochs = int(model_params.get('epochs', 100))
    steps_per_epoch = int(model_params.get('steps_per_epoch', 100))
    batch_size = int(model_params.get('batch_size', 16))
    data_aug_params = model_params.get('data_aug_params', None)

    apnet_m_template, apnet_m_parallel = compiled_model_from_params(model_params, ancillary, ensemble, verbose)

    callbacks = []

    ###########################
    # 2. GATHER DATA
    ###########################
    train_ids = data_dict['train_ids']
    train_generator = get_training_generator(
        train_ids,
        batch_size,
        data_aug_params,
        verbose,
        ancillary=ancillary,
        ensemble=ensemble
    )
    if 'validation_ids' in data_dict and len(data_dict['validation_ids']):
        validation_data = get_validation_data(
            data_dict['validation_ids'],
            verbose,
            ancillary=ancillary,
            ensemble=ensemble
        )

    else:
        validation_data = None

    early_stopping_monitor = None
    if early_stopping:
        if validation_data:
            metrics = None
            if early_stopping['monitor'] == 'survival':
                metrics = ('brier_score', 'c_index')
            elif early_stopping['monitor'] == 'val_c_idx':
                metrics = ('c_index',)
            elif early_stopping['monitor'] == 'val_ibs':
                metrics = ('brier_score',)

            if metrics:
                callbacks.append(apnet_cbks.ApnetMetricsLogger(ancillary=ancillary, ensemble=ensemble, metrics=metrics))

        early_stopping_monitor = apnet_cbks.ApnetEarlyStopping(
            monitor=early_stopping['monitor'],
            mode=early_stopping.get('mode', 'min'),
            restore_best_weights=True,
            patience=early_stopping.get('patience', 30),  # 200
            verbose=verbose,
            delay=early_stopping.get('delay', 10),  # 10
            window=early_stopping.get('window', 1),  # 1,
        )
        callbacks.append(early_stopping_monitor)

    ###########################
    # 3. Train Model
    ###########################
    callbacks += [
        # apnet_cbks.LossWeightScheduler(
        #     reconstruction_loss_weight=reconstruction_loss_wt,
        #     survival_loss_wts=survival_loss_wts
        # ),
        TerminateOnNaN()
    ]
    apnet_m_parallel.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=verbose,
        validation_data=validation_data,
        callbacks=callbacks,
        workers=1,
        use_multiprocessing=False,
        max_queue_size=10,
    )

    best_epoch = epochs
    if early_stopping_monitor is not None:
        if early_stopping_monitor.stopped_epoch > 0:
            best_epoch = early_stopping_monitor.stopped_epoch

    return {'model': apnet_m_template, 'stopped_epoch': best_epoch}


def apnet_train_and_validate_kfold(
        data,
        model_params,
        nfolds=5,
        verbose=0,
        nrepeats=1,
        filepath=None,
        ancillary=False,
        ensemble=False,
        early_stopping=None,
        val_split=.1
):
    if not early_stopping:
        es_metric = 'val_loss' if ancillary else 'val_latent_parameter_loss'
        early_stopping = {'monitor': es_metric, 'patience': 0}

    label_data = data_config.label_data()
    events = [label_data[y][1] for y in data]

    kfold_obj = RepeatedStratifiedKFold(n_splits=nfolds, n_repeats=nrepeats, random_state=2020)

    scores = []
    for (i, (train_index, test_index)) in enumerate(kfold_obj.split(data, events)):
        x_train = [data[idx] for idx in train_index]
        train_val_events = [events[idx] for idx in train_index]

        if val_split > 0:
            x_train, x_val = train_test_split(x_train, test_size=val_split, random_state=2020, stratify=train_val_events)
        else:
            x_val = []
        x_test = [data[idx] for idx in test_index]

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

        scores.append({
            'fold': i,
            'loss': m_hist['loss'][-(1 + early_stopping['patience'])],
            **metrics,
            'score': m_hist[early_stopping['monitor']][-(1 + early_stopping['patience'])],  # aggregate model score
        })

        if filepath:
            apnet_model.save_weights(os.path.join(filepath, 'apnet_model_weights_%d.h5' % i))
            pickle.dump(
                {'train': x_train, 'validation': x_test}, open(os.path.join(filepath, 'data_split_%d.pkl' % i), 'wb')
            )

    all_results = pd.DataFrame.from_records(scores)

    if filepath:
        all_results.to_pickle(os.path.join(filepath, 'trial_summary.pkl'))
        pickle.dump(model_params, open(os.path.join(filepath, 'model_params.pkl'), 'wb'))

    if len(all_results) < .8 * nrepeats * nfolds:
        return np.nan
    else:
        return all_results['score'].mean()
