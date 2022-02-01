import argparse
import os
import pickle
import scar_learning.config_data as data_config
from scar_learning.apnet.apnet_hypersearch import find_best_weights_index
from scar_learning.apnet.apnet_train_fns import apnet_train_atom, apnet_train_and_validate_kfold
from scar_learning.apnet.apnet_model import ensemble_model_parameters


def validate_kfold_fixed(
        model_id,
        nfolds,
        nrepeats,
        verbose,
        val_split,
        ancillary,
        save_path,
        ensemble,
):
    """
    Perform k-fold cross-valiation on development set.
    :param model_id: model id from hypersearch
    :param nfolds: number of folds
    :param nrepeats: number of repetitions
    :param verbose: printouts?
    :param val_split: how much to reserve for validation
    :param ancillary: whether model is imaging or ancillary
    :param save_path: where to store the results
    :param ensemble: whether to run in ensemble mode
    :return:
    """

    # Figure out model parameters
    if ensemble:
        model_type = 'ens'
        model_id_aux, model_id_conv = model_id.split('_')
        trial_path_aux = os.path.join(
            data_config.VAL_RESULTS_SRC_PATH,
            'apnet_aux_%s' % data_config.APNET_MODEL,
            'trial_%s' % model_id_aux
        )
        trial_path_conv = os.path.join(
            data_config.VAL_RESULTS_SRC_PATH,
            'apnet_conv_%s' % data_config.APNET_MODEL,
            'trial_%s' % model_id_conv
        )
        model_params = ensemble_model_parameters(
            pickle.load(open(os.path.join(trial_path_aux, 'overall_model_params.pkl'), 'rb')),
            os.path.join(trial_path_aux, 'overall_apnet_model_weights.h5'),
            pickle.load(open(os.path.join(trial_path_conv, 'overall_model_params.pkl'), 'rb')),
            os.path.join(trial_path_conv, 'overall_apnet_model_weights.h5'),
        )
        early_stopping = {'monitor': 'loss', 'mode': 'min', 'patience': 50, 'delay': 0}
    else:
        model_type = 'aux' if ancillary else 'conv'
        wt_key = 'ancillary' if ancillary else 'convolutional'
        trial_path = os.path.join(
            data_config.HYPERSEARCH_RESULTS_SRC_PATH,
            'apnet_%s_%s' % (model_type, data_config.APNET_MODEL),
            'trial_%s' % model_id
        )

        model_params = pickle.load(open(os.path.join(trial_path, 'model_params.pkl'), 'rb'))
        weights_placeholder = os.path.join(trial_path, 'apnet_model_weights_%d.h5')
        best_w_idx = find_best_weights_index('apnet_%s_%s' % (model_type, data_config.APNET_MODEL), 'concordance_index')
        model_params['model_weights'] = {wt_key: weights_placeholder % best_w_idx}
        model_params['initial_lr'] = 5e-4 if ancillary else 1e-4
        model_params['optimizer'] = 'Adam'

        early_stopping = {'monitor': 'val_c_idx', 'mode': 'max', 'patience': 50, 'delay': 0}
    if save_path:
        save_path = os.path.join(
            save_path,
            'apnet_%s_%s' % (model_type, data_config.APNET_MODEL),
            'trial_%s' % model_id
        )
        if not os.path.exists(save_path):
            os.mkdir(save_path)

    train_val_data = data_config.label_data((data_config.APNET_TRAIN_COHORT,))
    train_val_ids = sorted(list(train_val_data.keys()))

    apnet_train_and_validate_kfold(
        data=train_val_ids,
        model_params=model_params,
        nfolds=nfolds,
        verbose=verbose,
        nrepeats=nrepeats,
        filepath=save_path,
        ancillary=ancillary,
        ensemble=ensemble,
        early_stopping=early_stopping,
        val_split=val_split
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", help="ID for the model", type=str)
    parser.add_argument("--model_type", help="whether to run conv or dense", type=str, default='')
    parser.add_argument("--ensemble", help="whether to ensemble models", type=bool, default=False)
    parser.add_argument("--val_split", help="how much to reserve for early stopping", type=float, default=.1)
    parser.add_argument(
        "--save_path",
        help="where to store the results",
        type=str,
        default=data_config.VAL_RESULTS_SRC_PATH
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

    validate_kfold_fixed(
        model_id=args.model_id,
        nfolds=10,
        nrepeats=10,
        verbose=2,
        val_split=args.val_split,
        ancillary=args.model_type == 'dense',
        save_path=args.save_path,
        ensemble=args.ensemble,
    )
