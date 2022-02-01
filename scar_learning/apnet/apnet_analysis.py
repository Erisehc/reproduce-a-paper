import pickle
import os
from scar_learning.apnet.apnet_model import compiled_model_from_params
from scar_learning.apnet.apnet_data import get_validation_data
import numpy as np
import pandas as pd
from scar_learning.model_evaluation import integrated_brier_score, concordance_index
from lifelines import KaplanMeierFitter
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import scar_learning.model_evaluation as m_eval
import scar_learning.config_data as data_config
import matplotlib.pyplot as plt
from scar_learning.apnet.apnet_model import survival_fn_gen
import tqdm
import scipy.interpolate as interp
from tensorflow import keras
from keras.models import Model, Sequential
from scar_learning.image_processing import crop

# Display
import matplotlib.cm as cm
from PIL import Image

# Internal
import keras.backend as K


nfolds = 10
nrepeats = 10
times = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
src = data_config.VAL_RESULTS_SRC_PATH


def _interp_patched(x0, x, y):
    if len(x) == 1:
        return y[0]
    else:
        return interp.interp1d(x, y, kind='next', fill_value='extrapolate')(x0)


def _cache_predictions_available(pred_path, train_val):

    if not os.path.exists(pred_path):
        return False

    preds = pickle.load(open(pred_path, 'rb'))

    if train_val not in preds:
        return False

    return True


def baseline_generate_predictions(model_id, cohort, train_val='validation', refresh_cache=False, is_overall=False):
    """
    Returns a list of lists of np.arrays with gt and predictions
    :param model_id:
    :param cohort: val or test
    :param train_val: whether to use train data or test data. 'train' or 'validation'
    :param refresh_cache: refresh the cache
    :param is_overall: single run or multiple
    :return:
    """
    # check if cache available
    results_path = os.path.join(cohort, 'baseline_cox_ph', 'trial_%s' % model_id)
    p_name = 'overall_predictions.pkl' if is_overall else 'predictions.pkl'
    pred_path = os.path.join(results_path, p_name)
    has_cache = _cache_predictions_available(pred_path, train_val)
    n = 1 if is_overall else nrepeats * nfolds

    if has_cache and not refresh_cache:
        return pickle.load(open(pred_path, 'rb'))[train_val]
    else:
        y = []
        y_h = []
        for i in range(n):
            ds_name = 'overall_data_split.pkl' if is_overall else 'data_split_%d.pkl' % i
            x_test = pickle.load(open(os.path.join(results_path, ds_name), 'rb'))[train_val]
            wt_name = 'overall_baseline_model_weights.h5' if is_overall else 'data_split_%d.pkl' % i
            trained_model = pickle.load(open(os.path.join(results_path, wt_name), 'rb'))
            y_rep = \
                [[float(el[0]), bool(el[1])] for el in zip(x_test['event_time'].values, x_test['event_type'].values)]
            y_h_rep = [[-float(el), 0] for el in trained_model.predict_partial_hazard(x_test).values]  # make cons w/ nn

            y.append(np.asarray(y_rep))
            y_h.append(np.asarray(y_h_rep))

        #  refresh cache
        cache = pickle.load(open(pred_path, 'rb')) if os.path.exists(pred_path) else {}
        cache[train_val] = (y, y_h)
        pickle.dump(cache, open(pred_path, 'wb'))

        return y, y_h


def apnet_generate_predictions(
        model_id,
        model_fit,
        ancillary,
        ensemble,
        cohort,
        train_val='validation',
        refresh_cache=False,
        is_overall=False,

):
    """
    Returns a list of lists of np.arrays with gt and predictions
    :param model_id:
    :param model_fit:
    :param ancillary:
    :param ensemble:
    :param cohort: val or test
    :param train_val: whether to use train data or test data. 'train' or 'validation'
    :param refresh_cache:
    :param is_overall: single run or multiple
    :return:
    """
    verbose = False
    model_type = 'ens' if ensemble else 'aux' if ancillary else 'conv'
    results_path = os.path.join(
        cohort,
        'apnet_%s_%s' % (model_type, model_fit),
        'trial_%s' % model_id
    )

    p_name = 'overall_predictions.pkl' if is_overall else 'predictions.pkl'
    pred_path = os.path.join(results_path, p_name)
    has_cache = _cache_predictions_available(pred_path, train_val)
    n = 1 if is_overall else nrepeats * nfolds

    if has_cache and not refresh_cache:
        return pickle.load(open(pred_path, 'rb'))[train_val]
    else:
        y = []
        y_h = []

        mp_name = 'overall_model_params.pkl' if is_overall else 'model_params.pkl'
        model_params = pickle.load(open(os.path.join(results_path, mp_name), 'rb'))
        model_params.pop('model_weights')  # loading them separately
        if 'ensemble_aux_params' in model_params:
            model_params['ensemble_aux_params'].pop('model_weights')  # loading them separately
        if 'ensemble_conv_params' in model_params:
            model_params['ensemble_conv_params'].pop('model_weights')  # loading them separately
        apnet_m, _ = compiled_model_from_params(model_params, ancillary, ensemble, 0)

        for i in range(n):
            wt_name = 'overall_apnet_model_weights.h5' if is_overall else 'apnet_model_weights_%d.h5' % i
            wt_path = os.path.join(results_path, wt_name)
            if not os.path.exists(wt_path):
                continue
            apnet_m.load_weights(wt_path)
            ds_name = 'overall_data_split.pkl' if is_overall else 'data_split_%d.pkl' % i
            x_ids = pickle.load(open(os.path.join(results_path, ds_name), 'rb'))[train_val]
            x_rep_fold, y_rep_fold, _ = get_validation_data(
                x_ids,
                verbose=verbose,
                ancillary=ancillary,
                ensemble=ensemble,
                shuffle=False
            )
            y_h_rep_fold = apnet_m.predict(x_rep_fold, batch_size=1)

            if not ensemble and not ancillary:
                y_h_rep_fold = y_h_rep_fold[1]
                y_rep_fold = y_rep_fold[1]

            y_rep = np.asarray(y_rep_fold, dtype='float64')  # apnet works in log-time
            y_rep[:, 0] = np.exp(y_rep[:, 0])
            y_h_rep = np.asarray(y_h_rep_fold, dtype='float64')

            y.append(y_rep)
            y_h.append(y_h_rep)

        #  refresh cache
        cache = pickle.load(open(pred_path, 'rb')) if os.path.exists(pred_path) else {}
        cache[train_val] = (y, y_h)
        pickle.dump(cache, open(pred_path, 'wb'))

        return y, y_h


def compute_stats(
        model_id,
        is_baseline,
        model_fit,
        is_overall,
        ancillary=None,
        ensemble=None,
):
    """
    Compute and save comprehensive stats
    :param model_id: identifier for the model
    :param is_baseline: baseline cox_ph or APNet
    :param model_fit: 'cox_ph' or 'loglogistic'
    :param is_overall: wheter stats are computed for single data_split
    :param ancillary: whether dense or convolutional
    :param ensemble: whether model is ensemble
    :return:
    """
    outcomes = data_config.label_data((data_config.APNET_TRAIN_COHORT,))
    cd = np.array([[x[0] for x in outcomes.values()], [x[1] for x in outcomes.values()]])
    kmf = KaplanMeierFitter().fit(cd[0], cd[1])
    baseline = -np.log(kmf.survival_function_)

    #  Baseline model
    if is_baseline:
        y_true, y_pred = baseline_generate_predictions(model_id=model_id, cohort=src, is_overall=is_overall)  # model_id = 21
        y_true_train, y_pred_train = baseline_generate_predictions(model_id=model_id, cohort=src, train_val='train', is_overall=is_overall)
        model_type = ''
    else:
        model_type = 'ens' if ensemble else 'aux' if ancillary else 'conv'
        y_true, y_pred = apnet_generate_predictions(model_id, model_fit, ancillary, ensemble, src, is_overall=is_overall)
        y_true_train, y_pred_train = apnet_generate_predictions(model_id, model_fit, ancillary, ensemble, src, 'train', is_overall=is_overall)

    n = len(y_true)

    # Function used to blank out if insufficient data
    def blank_fn(y):
        return False if is_overall else len(y) < n // 4

    stats_map = {'mean': lambda x: np.mean(x)}

    if not is_overall:
        stats_map = {
            **stats_map,
            'std': lambda x: np.std(x),
            'percentile_5': lambda x: np.percentile(x, 5),
            'percentile_95': lambda x: np.percentile(x, 95),
            'count': lambda x: len(x),
            'min': lambda x: np.min(x),
            'max': lambda x: np.max(x),
        }

    all_stats = []
    for tau in tqdm.tqdm(times):
        measure_map = {
            'c_index': [],
            'brier_score': [],
            'auprc': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'auroc': [],
            'specificity': [],
            'sensitivity': [],
            'balanced_accuracy': []
        }

        for i in range(n):
            # Calibration data
            surv_fn_train = survival_fn_gen(y_pred_train[i], baseline, model_fit)
            y_pred_i_train = 1 - surv_fn_train(tau)
            y_true_t_train, y_prob_t_train = m_eval.y_pred_at_t(y_true_train[i], y_pred_i_train, tau, None)

            # Method 1: Use J-index
            fpr_train, tpr_train, thresholds_roc = roc_curve(y_true_t_train, y_prob_t_train)
            best_thresh_J = thresholds_roc[np.argmax(tpr_train - fpr_train)]  # get the best threshold

            # Method 2: Use F-1
            precision_train, recall_train, thresholds_pr = precision_recall_curve(y_true_t_train, y_prob_t_train)
            best_thresh_f1 = thresholds_pr[np.argmax(
                (precision_train * recall_train) / (precision_train + recall_train)
            )]  # get the best threshold

            # Results
            surv_fn = survival_fn_gen(y_pred[i], baseline, model_fit)
            y_pred_i = 1 - surv_fn(tau)
            y_true_t, y_prob_t = m_eval.y_pred_at_t(y_true[i], y_pred_i, tau, None)
            if not np.sum(y_true_t):
                print('All labels are class 0 at time %d for fold %d. Skipping metrics involving true positives.' % (tau, i))
                continue

            # 1. Time-dependent measures
            measure_map['c_index'].append(concordance_index(tau, y_true[i][:, 0], y_pred[i][:, 0], y_true[i][:, 1], cd))
            measure_map['brier_score'].append(
                integrated_brier_score(surv_fn, y_true[i][:, 0], y_true[i][:, 1], censor_data=cd, t_max=tau)
            )

            if np.all(y_true_t):
                print('All labels are class 1 at time %d for fold %d. Skipping metrics involving true negatives.' % (tau, i))
                continue

            # 2. Precision-recall measures (best model = highest F1)
            precision, recall, thrs_pr = precision_recall_curve(y_true_t, y_prob_t)
            prec = _interp_patched(best_thresh_f1, thrs_pr, precision[:-1])
            rec = _interp_patched(best_thresh_f1, thrs_pr, recall[:-1])
            if np.isclose(prec, 0) and np.isclose(rec, 0):
                f1 = 0
            else:
                f1 = 2 * prec * rec / (prec + rec)

            measure_map['auprc'].append(auc(recall, precision))
            measure_map['precision'].append(prec)
            measure_map['recall'].append(rec)
            measure_map['f1_score'].append(f1)

            # 3. Sensitivity-Specificity measures (best model = highest J)
            fpr, tpr, thrs_roc = roc_curve(y_true_t, y_prob_t)
            spec = 1 - _interp_patched(best_thresh_J, thrs_roc, fpr)
            sens = _interp_patched(best_thresh_J, thrs_roc, tpr)
            measure_map['auroc'].append(auc(fpr, tpr))
            measure_map['sensitivity'].append(sens)
            measure_map['specificity'].append(spec)
            measure_map['balanced_accuracy'].append((sens + spec) / 2)

        for msr_name, msr_vals in measure_map.items():
            msr_vals = np.array(msr_vals)
            msr_vals = msr_vals[np.logical_not(np.isnan(msr_vals))]
            for stat_name, stat_fn in stats_map.items():
                all_stats.append({
                    'measure': msr_name,
                    'period': tau,
                    'stat': stat_name,
                    'value': np.nan if blank_fn(msr_vals) else stat_fn(msr_vals)
                })

    stats_df = pd.DataFrame.from_records(all_stats)
    results_folder = '_'.join(['baseline' if is_baseline else 'apnet', model_type, model_fit]).replace('__', '_')
    results_path = os.path.join(src, results_folder, 'trial_%s' % model_id)
    rr_name = 'overall_raw_results.pkl' if is_overall else 'raw_results.pkl'
    pickle.dump(stats_df, open(os.path.join(results_path, rr_name), 'wb'))

    return stats_df


def viz_layer_ram():
    # Get model
    ancillary = False
    ensemble = False
    verbose = False
    model_path = os.path.join(data_config.VAL_RESULTS_SRC_PATH, 'apnet_conv_loglogistic', 'trial_93')
    model_params = pickle.load(open(os.path.join(model_path, 'overall_model_params.pkl'), 'rb'))
    apnet_m, _ = compiled_model_from_params(model_params, ancillary, ensemble, verbose)
    apnet_m.load_weights(os.path.join(model_path, 'overall_apnet_model_weights.h5'), by_name=True)
    apnet_no_recon = Model(inputs=apnet_m.input, outputs=apnet_m.output[-1])
    apnet_m_flat = Sequential([apnet_no_recon.layers[0]] + apnet_no_recon.layers[1].layers + [apnet_no_recon.layers[2]])

    last_conv_layer_name = 'activation_2'

    # Get data
    img_array_idx = 0
    # train_val_data = data_config.label_data((data_config.APNET_TRAIN_COHORT,))
    # train_ids = sorted(list(train_val_data.keys()))
    # train_id_to_get = train_ids[patient_id]
    train_id_to_get = 'P030'
    test_data = get_validation_data([train_id_to_get], verbose=0, ancillary=False, shuffle=False)
    data_x_i = test_data[0]
    data_y_i = test_data[1]

    img_array = np.expand_dims(data_x_i[img_array_idx], axis=0)

    # Compute regression predictions
    mu_output = apnet_m_flat.output[:, 0]  # first parameter is mode
    last_conv_layer = apnet_m_flat.get_layer(last_conv_layer_name)

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = K.gradients(mu_output, last_conv_layer.get_output_at(-1))[0]

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = K.mean(grads, axis=(0, 1, 2, 3))

    iterate = K.function([apnet_m_flat.input], [pooled_grads, last_conv_layer.get_output_at(-1)[0]])

    pooled_grads_value, conv_layer_output_value = iterate([img_array])

    # # We multiply each channel in the feature map array
    # # by "how important this channel is" with regard to the top predicted class
    #
    for i in range(pooled_grads.shape[-1]):
        conv_layer_output_value[:, :, :, i] *= pooled_grads_value[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(conv_layer_output_value, axis=-1)

    # Overlap the 2 images
    idx = 2
    plt.figure(1)
    plt.imshow(data_x_i[img_array_idx][:, :, :, 0][:, :, idx])

    plt.figure(2)
    plt.imshow(data_x_i[img_array_idx][:, :, :, 1][:, :, idx])

    # get high res version of image
    rv = pickle.load(
        open('/home/dpopesc2/PycharmProjects/DNN/scar_learning/data/archive/hv_gt_128/%s.pickle' % train_id_to_get,
             'rb'))

    plt.figure(3)
    plt.imshow(rv['_segmentation'][:, :, idx])
    plt.figure(4)
    plt.imshow(rv['_pixel_array'][:, :, idx])
    # plt.show()

    original_image = rv['_pixel_array']
    original_seg = rv['_segmentation']

    nx1, ny1, nz1 = original_image.shape
    x1 = np.linspace(0, 1, nx1)
    y1 = np.linspace(0, 1, ny1)
    z1 = np.linspace(0, 1, nz1)
    xv1, yv1, zv1 = np.meshgrid(x1, y1, z1)

    nx2, ny2, nz2 = heatmap.shape
    x2 = np.linspace(0, 1, nx2)
    y2 = np.linspace(0, 1, ny2)
    z2 = np.linspace(0, 1, nz2)
    xv2, yv2, zv2 = np.meshgrid(x2, y2, z2)

    hmp_all = interp.interpn((x2, y2, z2), heatmap, (xv1, yv1, zv1), fill_value=0)
    fov = original_seg
    its = original_image

    hmp_all_grads = hmp_all[np.isclose(fov, 4)]
    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    # hmp_all[np.isclose(fov, 0)] = 0  # 0 out area outside of ROI
    # hmp_all[np.isclose(fov, 1)] = 0  # 0 out area outside of ROI

    # Scale heatmap from - mag to mag
    # mag = np.max(np.abs(heatmap))
    # hmp_all = .5 * hmp_all / mag + .5

    # # Scaling 1: Scale heatmap from min to max
    reverse = False
    # if np.max(hmp_all) <= 0:
    #     reverse = True
    #
    # hmp_all = (hmp_all - np.min(hmp_all)) / (np.max(hmp_all) - np.min(hmp_all))

    # Scaling 2:
    m = np.min(hmp_all_grads)
    M = np.max(hmp_all_grads)

    if M <= 0:
        a = .5 / (M - m)
        b = -.5 * m / (M - m)
    elif m >= 0:
        a = .5 / (M - m)
        b = (M - 2 * m) / (M - m)
    elif np.abs(m) < np.abs(M):
        a = .5 / M
        b = .5
    else:
        a = -.5 / m
        b = .5

    hmp_all = np.clip(a * hmp_all + b, 0, 1)

    its = 255 * its / np.max(its)

    its_disp = its.copy()
    its[np.isclose(fov, 4)] = 0  # 0 out area outside of ROI

    bb = crop.find_bounding_box([its_disp[:, :, i] for i in range(its.shape[-1])], force_square=True)
    for i in range(its.shape[-1]):
        im = crop.crop_image(its[:, :, i], bb)
        im_disp = crop.crop_image(its_disp[:, :, i], bb)
        hmp = crop.crop_image(hmp_all[:, :, i], bb)
        fv = crop.crop_image(fov[:, :, i], bb)

        im = keras.preprocessing.image.array_to_img(im, scale=True)
        im_disp = keras.preprocessing.image.array_to_img(im_disp, scale=True)
        im = keras.preprocessing.image.img_to_array(im)
        im_disp = keras.preprocessing.image.img_to_array(im_disp)

        # We rescale heatmap to a range 0-255
        hmp = np.uint8(255 * hmp)

        # We use jet colormap to colorize heatmap
        jet = cm.get_cmap("RdBu")

        # We use RGB values of the colormap
        if reverse:
            jet_colors = jet(np.arange(256)[::-1])[:, :3]
        else:
            jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = np.squeeze(jet_colors[hmp])

        # We create an image with RGB colorized heatmap
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        # jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]), resample=Image.NEAREST)
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap
        superimposed_img[np.where(np.isclose(np.repeat(fv, 3, axis=-1), 0))] = 0
        superimposed_img[np.where(np.isclose(np.repeat(fv, 3, axis=-1), 1))] = 0
        superimposed_img += im
        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
        # superimposed_img = superimposed_img.resize((1024, 1024), resample=Image.NEAREST)
        superimposed_img = superimposed_img.resize((1024, 1024))
        superimposed_img.save('grad_cam/%d_3hm.png' % i)

        im_disp = keras.preprocessing.image.array_to_img(im_disp)
        # im_disp = im_disp.resize((1024, 1024), resample=Image.NEAREST)
        im_disp = im_disp.resize((1024, 1024))
        im_disp.save('grad_cam/%d_1img.png' % i)

        r_map = fv * 0
        r_map[np.isclose(fv, 1)] = 217
        r_map[np.isclose(fv, 4)] = 27
        g_map = fv * 0
        g_map[np.isclose(fv, 1)] = 95
        g_map[np.isclose(fv, 4)] = 158
        b_map = fv * 0
        b_map[np.isclose(fv, 1)] = 2
        b_map[np.isclose(fv, 4)] = 119

        seg_img = np.concatenate([r_map, g_map, b_map], axis=-1)
        seg_img = keras.preprocessing.image.array_to_img(seg_img)
        # seg_img = seg_img.resize((1024, 1024), resample=Image.NEAREST)
        seg_img = seg_img.resize((1024, 1024))
        seg_img.save('grad_cam/%d_2sg.png' % i)


def viz_dense_grads():
    from scar_learning.apnet.apnet_model import compiled_model_from_params
    import scar_learning.config_data as data_config
    from scar_learning.apnet.apnet_data import get_validation_data
    import keras.backend as K
    import os
    import pickle
    from scar_learning.image_processing import encoding_functions as ef
    import numpy as np
    from keras.models import Model, Input, Sequential

    model_path = os.path.join(data_config.VAL_RESULTS_SRC_PATH, 'apnet_aux_loglogistic', 'trial_289')
    model_params = pickle.load(open(os.path.join(model_path, 'overall_model_params.pkl'), 'rb'))
    apnet_m, _ = compiled_model_from_params(model_params, True, False, False)
    apnet_m.load_weights(os.path.join(model_path, 'overall_apnet_model_weights.h5'), by_name=True)

    mu_output = apnet_m.output[:, 0]  # first parameter is mode
    grads = K.gradients(mu_output, apnet_m.input)[0]
    iterate = K.function([apnet_m.input], [grads, apnet_m.output])

    train_id_to_get = pickle.load(open(os.path.join(model_path, 'overall_data_split.pkl'), 'rb'))['train']
    test_data = get_validation_data(train_id_to_get, verbose=0, ancillary=True, shuffle=False)
    data_x_i = test_data[0]
    data_y_i = test_data[1]

    all_grads = []
    for i in range(len(train_id_to_get)):
        all_grads.append(
            iterate(
                [np.expand_dims(data_x_i[i], axis=0)]
            )[0]
        )

    all_grads = np.squeeze(np.asarray(all_grads))
    g1 = np.mean(all_grads, axis=0)
    g2 = np.mean(all_grads[data_y_i[:, 1] == 1], axis=0)
    g3 = np.mean(all_grads[data_y_i[:, 1] == 0], axis=0)

    s1 = np.std(all_grads, axis=0) / np.sqrt(all_grads.shape[0])
    s2 = np.std(all_grads[data_y_i[:, 1] == 1], axis=0) / np.sqrt(all_grads[data_y_i[:, 1] == 1].shape[0])
    s3 = np.std(all_grads[data_y_i[:, 1] == 0], axis=0) / np.sqrt(all_grads[data_y_i[:, 1] == 0].shape[0])

    import matplotlib.pyplot as plt

    first_n = 3
    covariate_list = list(ef.encode_ancillary_data(data_config.ancillary_data()))

    sort_order = np.argsort(g1)
    sort_order = np.concatenate([sort_order[:first_n], sort_order[-first_n:]])
    c = np.asarray(covariate_list)[sort_order]

    g1 = g1[sort_order]
    s1 = s1[sort_order]

    g2 = g2[sort_order]
    s2 = s2[sort_order]

    g3 = g3[sort_order]
    s3 = s3[sort_order]

    fig, ax = plt.subplots(figsize=(4, 6.4721))

    width = 0.25  # the width of the bars
    eps = .04
    x1 = np.arange(len(c))
    x2 = [x + width for x in x1]
    x3 = [x + width for x in x2]
    ax.barh(x3, g1, width - eps, label='All', xerr=s1, color='tab:blue', ecolor='tab:red')
    ax.barh(x2, g2, width - eps, label='SCDA', xerr=s2, color='tab:blue', hatch='+++', ecolor='tab:red')
    ax.barh(x1, g3, width - eps, label='No SCDA', xerr=s3, color='tab:blue', hatch='///', ecolor='tab:red')
    plt.xlabel("Gradient")
    plt.ylabel("Covariate Name")
    plt.axvline(0, color='black')

    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(left=0.2)
    plt.yticks([r + width for r in range(len(c))], c, fontsize=10)
    plt.legend()
    fig.tight_layout()

    plt.show()

# src = data_config.VAL_RESULTS_SRC_PATH
# apnet_generate_predictions('93', 'loglogistic', False, False, src, is_overall=True, train_val='train', refresh_cache=True)
# apnet_generate_predictions('93', 'loglogistic', False, False, src, is_overall=True, train_val='validation', refresh_cache=True)
# compute_stats(
#     model_id='29',
#     is_baseline=True,
#     model_fit='cox_ph',
#     is_overall=True
# )
# compute_stats(
#     model_id='29',
#     is_baseline=True,
#     model_fit='cox_ph',
#     is_overall=False
# )
# compute_stats(
#     model_id='289',
#     is_baseline=False,
#     model_fit='loglogistic',
#     is_overall=True,
#     ancillary=True,
#     ensemble=False,
# )
# compute_stats(
#     model_id='289',
#     is_baseline=False,
#     model_fit='loglogistic',
#     is_overall=False,
#     ancillary=True,
#     ensemble=False,
# )
# compute_stats(
#     model_id='93',
#     is_baseline=False,
#     model_fit='loglogistic',
#     is_overall=True,
#     ancillary=False,
#     ensemble=False,
# )
# compute_stats(
#     model_id='93',
#     is_baseline=False,
#     model_fit='loglogistic',
#     is_overall=False,
#     ancillary=False,
#     ensemble=False,
# )
# compute_stats(
#     model_id='289_93',
#     is_baseline=False,
#     model_fit='loglogistic',
#     is_overall=True,
#     ensemble=True,
# )
# compute_stats(
#     model_id='289_93',
#     is_baseline=False,
#     model_fit='loglogistic',
#     is_overall=False,
#     ensemble=True,
# )
# src = data_config.TEST_RESULTS_SRC_PATH
# compute_stats(
#     model_id='29',
#     is_baseline=True,
#     model_fit='cox_ph',
#     is_overall=True
# )
# compute_stats(
#     model_id='29',
#     is_baseline=True,
#     model_fit='cox_ph',
#     is_overall=False
# )
# compute_stats(
#     model_id='289',
#     is_baseline=False,
#     model_fit='loglogistic',
#     is_overall=True,
#     ancillary=True,
#     ensemble=False,
# )
# compute_stats(
#     model_id='289',
#     is_baseline=False,
#     model_fit='loglogistic',
#     is_overall=False,
#     ancillary=True,
#     ensemble=False,
# )
# compute_stats(
#     model_id='93',
#     is_baseline=False,
#     model_fit='loglogistic',
#     is_overall=True,
#     ancillary=False,
#     ensemble=False,
# )
# compute_stats(
#     model_id='289_93',
#     is_baseline=False,
#     model_fit='loglogistic',
#     is_overall=True,
#     ensemble=True,
# )
# compute_stats(
#     model_id='93',
#     is_baseline=False,
#     model_fit='loglogistic',
#     is_overall=False,
#     ancillary=False,
#     ensemble=False,
# )
# compute_stats(
#     model_id='289_93',
#     is_baseline=False,
#     model_fit='loglogistic',
#     is_overall=False,
#     ensemble=True,
# )

# src = data_config.VAL_RESULTS_SRC_PATH
# compute_stats(
#     model_id='59',
#     is_baseline=False,
#     model_fit='cox_ph',
#     is_overall=True,
#     ancillary=True,
#     ensemble=False,
# )
# compute_stats(
#     model_id='59',
#     is_baseline=False,
#     model_fit='cox_ph',
#     is_overall=False,
#     ancillary=True,
#     ensemble=False,
# )
# compute_stats(
#     model_id='7',
#     is_baseline=False,
#     model_fit='cox_ph',
#     is_overall=True,
#     ancillary=False,
#     ensemble=False,
# )
# compute_stats(
#     model_id='59_7',
#     is_baseline=False,
#     model_fit='cox_ph',
#     is_overall=True,
#     ensemble=True,
# )
# compute_stats(
#     model_id='7',
#     is_baseline=False,
#     model_fit='cox_ph',
#     is_overall=False,
#     ancillary=False,
#     ensemble=False,
# )
# compute_stats(
#     model_id='59_7',
#     is_baseline=False,
#     model_fit='cox_ph',
#     is_overall=False,
#     ensemble=True,
# )
#
# src = data_config.TEST_RESULTS_SRC_PATH
# compute_stats(
#     model_id='59',
#     is_baseline=False,
#     model_fit='cox_ph',
#     is_overall=True,
#     ancillary=True,
#     ensemble=False,
# )
# compute_stats(
#     model_id='59',
#     is_baseline=False,
#     model_fit='cox_ph',
#     is_overall=False,
#     ancillary=True,
#     ensemble=False,
# )
# compute_stats(
#     model_id='7',
#     is_baseline=False,
#     model_fit='cox_ph',
#     is_overall=True,
#     ancillary=False,
#     ensemble=False,
# )
# compute_stats(
#     model_id='59_7',
#     is_baseline=False,
#     model_fit='cox_ph',
#     is_overall=True,
#     ensemble=True,
# )
# compute_stats(
#     model_id='7',
#     is_baseline=False,
#     model_fit='cox_ph',
#     is_overall=False,
#     ancillary=False,
#     ensemble=False,
# )
# compute_stats(
#     model_id='59_7',
#     is_baseline=False,
#     model_fit='cox_ph',
#     is_overall=False,
#     ensemble=True,
# )

# compute_stats('conv_loglogistic', '93')
# compute_stats('ens_loglogistic', '289_93')
# print(compute_stats('conv_loglogistic', '93').head(200))