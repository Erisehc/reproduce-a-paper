# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ''
import pickle
# from definitions import set_random_seed, set_tf_seed
# set_random_seed()
# import sys
# sys.path.append('/home/dpopesc2/PycharmProjects/avaeseg')
import numpy as np
import scar_learning.config_data as data_config
from scar_learning.apnet.apnet_train_fns import apnet_train_atom
from scar_learning.apnet.apnet_data import get_validation_data
from scar_learning.apnet.apnet_evaluate_fns import apnet_evaluate_atom
from sklearn.model_selection import train_test_split
from scar_learning.apnet.apnet_train_fns import apnet_train_and_validate_kfold
import os


def train_conv():
    model_id = '93'
    es_metric = 'val_c_idx'
    early_stopping = {'monitor': es_metric, 'mode': 'max', 'patience': 100, 'delay': 0}
    model_type = 'conv'
    trial_path = os.path.join(
        data_config.HYPERSEARCH_RESULTS_SRC_PATH,
        'apnet_%s_%s' % (model_type, data_config.APNET_MODEL),
        'trial_%s' % model_id
    )

    i = np.random.randint(0, 99)
    model_params = pickle.load(open(os.path.join(trial_path, 'model_params.pkl'), 'rb'))
    weights_placeholder = os.path.join(trial_path, 'apnet_model_weights_%d.h5')

    model_params['initial_lr'] = 1e-5
    model_params['optimizer'] = 'Adam'
    model_params['model_weights'] = {'convolutional': weights_placeholder % i}

    train_val_test_data = data_config.label_data((data_config.APNET_TEST_COHORT,))
    train_ids = sorted(list(train_val_test_data.keys()))
    events = [train_val_test_data[y][1] for y in train_ids]
    train_val_ids, test_ids = train_test_split(train_ids, test_size=.2, random_state=2020, stratify=events)
    events = [train_val_test_data[y][1] for y in train_val_ids]
    train_ids, val_ids = train_test_split(train_val_ids, test_size=.2, random_state=2020, stratify=events)

    m = apnet_train_atom(
        data_dict={
            'train_ids': train_ids,
            'validation_ids': val_ids
        },
        model_params=model_params,
        verbose=2,
        early_stopping=early_stopping,
        ancillary=False,
        ensemble=False
    )['model']

    test_data = get_validation_data(test_ids, verbose=1, ancillary=False, ensemble=False)
    score = apnet_evaluate_atom(m, test_data[0], test_data[1], single_output=False)
    print('final', score)


def train_aux():
    model_id = '277'
    es_metric = 'val_c_idx'
    early_stopping = {'monitor': es_metric, 'mode': 'max', 'patience': 5, 'delay': 0}
    model_type = 'aux'
    trial_path = os.path.join(
        data_config.HYPERSEARCH_RESULTS_SRC_PATH,
        'apnet_%s_%s' % (model_type, data_config.APNET_MODEL),
        'trial_%s' % model_id
    )

    i = np.random.randint(0, 99)
    model_params = pickle.load(open(os.path.join(trial_path, 'model_params.pkl'), 'rb'))
    weights_placeholder = os.path.join(trial_path, 'apnet_model_weights_%d.h5')

    model_params['initial_lr'] = 1e-4
    model_params['optimizer'] = 'Adam'
    model_params['model_weights'] = {'ancillary': weights_placeholder % i}

    train_val_test_data = data_config.label_data((data_config.APNET_TEST_COHORT,))
    train_ids = sorted(list(train_val_test_data.keys()))
    events = [train_val_test_data[y][1] for y in train_ids]
    train_val_ids, test_ids = train_test_split(train_ids, test_size=.2, random_state=2020, stratify=events)
    events = [train_val_test_data[y][1] for y in train_val_ids]
    train_ids, val_ids = train_test_split(train_val_ids, test_size=.2, random_state=2020, stratify=events)

    m = apnet_train_atom(
        data_dict={
            'train_ids': train_ids,
            'validation_ids': val_ids
        },
        model_params=model_params,
        verbose=2,
        early_stopping=early_stopping,
        ancillary=True,
        ensemble=False
    )['model']

    test_data = get_validation_data(test_ids, verbose=1, ancillary=True, ensemble=False)
    score = apnet_evaluate_atom(m, test_data[0], test_data[1], single_output=True)
    print('final', score)


def train_ensemble():
    train_val_test_data = data_config.label_data((data_config.APNET_TEST_COHORT,))
    train_ids = sorted(list(train_val_test_data.keys()))
    events = [train_val_test_data[y][1] for y in train_ids]
    train_val_ids, test_ids = train_test_split(train_ids, test_size=.2, random_state=2020, stratify=events)
    events = [train_val_test_data[y][1] for y in train_val_ids]
    train_ids, val_ids = train_test_split(train_val_ids, test_size=.2, random_state=2020, stratify=events)

    model_id = '277_90'
    model_id_aux, model_id_conv = model_id.split('_')
    trial_path_aux = os.path.join(
        data_config.TEST_RESULTS_SRC_PATH,  # todo: confirm that these are the right weights
        'apnet_aux_%s' % data_config.APNET_MODEL,
        'model_%s' % model_id_aux
    )
    weights_placeholder_aux = os.path.join(trial_path_aux, 'apnet_model_weights_%d.h5')
    trial_path_conv = os.path.join(
        data_config.TEST_RESULTS_SRC_PATH,
        'apnet_conv_%s' % data_config.APNET_MODEL,
        'model_%s' % model_id_conv
    )
    weights_placeholder_conv = os.path.join(trial_path_conv, 'apnet_model_weights_%d.h5')
    i = 0
    model_params = {
        'epochs': 2000,
        'initial_lr': 1e-3,
        'batch_size': 32,
        'steps_per_epoch': 20,
        'ensemble_no_units': 20,
        'ensemble_depth': 2,
        'l1_reg': 1e-6,
        'l2_reg': 1e-5,
        'ensemble_aux_params': pickle.load(open(os.path.join(trial_path_aux, 'model_params.pkl'), 'rb')),
        'ensemble_conv_params': pickle.load(open(os.path.join(trial_path_conv, 'model_params.pkl'), 'rb')),
        'optimizer': 'Adam',
        'model_weights': {
            'ancillary': weights_placeholder_aux % i,
            'convolutional': weights_placeholder_conv % i
        },
        'data_aug_params': {
            'rotation': False,  # degrees to rotate the images in the x-y plane
            'width_shift_pct': .9,  # shift images horizontally (fraction of total available space)
            'height_shift_pct': .9,  # shift images vertically (fraction of total available space)
            'depth_shift_pct': .9,
            # shift images in the short axis plane (fraction of total available space)
            'gaussian_noise_std': 0,  # standard deviation of gaussian noise applied to input
        },
    }

    m = apnet_train_atom(
        data_dict={
            'train_ids': train_ids,
            'validation_ids': val_ids
        },
        model_params=model_params,
        verbose=2,
        early_stopping={'monitor': 'val_c_idx', 'mode': 'max', 'patience': 30, 'delay': 20},
        ensemble=True
    )['model']

    test_data = get_validation_data(test_ids, verbose=1, ensemble=True)
    score = apnet_evaluate_atom(m, test_data[0], test_data[1], single_output=True)
    print('final', score)


if __name__ == '__main__':
    # import tensorflow as tf
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = .5
    # sess = tf.Session(config=config)

    train_aux()
    train_conv()
    train_ensemble()
