from scar_learning.model_evaluation import integrated_brier_score, concordance_index
from lifelines import KaplanMeierFitter
import numpy as np
import scar_learning.config_data as data_config
from scar_learning.apnet.apnet_model import survival_fn_gen


def apnet_evaluate_predictions(y_true, y_pred, metrics=('brier_score', 'c_index')):
    """
    Note that y_true is in log_times as per the input to the NN
    :param y_true:
    :param y_pred:
    :param metrics: which metrics to compute in the evaluation
    :return:
    """
    outcomes = data_config.label_data((data_config.APNET_TRAIN_COHORT, ))
    cd = np.array([[x[0] for x in outcomes.values()], [x[1] for x in outcomes.values()]])
    kmf = KaplanMeierFitter().fit(cd[0], cd[1])
    baseline = -np.log(kmf.survival_function_)

    event_times = np.exp(y_true[:, 0])
    event_observed = y_true[:, 1]

    loc = y_pred[:, 0]
    surv_fn_t = survival_fn_gen(y_pred, baseline=baseline)

    tau = np.max(cd[0])  # in years
    rv = {}
    if 'brier_score' in metrics:
        rv['brier_score'] = integrated_brier_score(surv_fn_t, event_times, event_observed, censor_data=cd)

    if 'c_index' in metrics:
        rv['c_index'] = concordance_index(tau, event_times, loc, event_observed=event_observed, censor_data=cd)

    return rv


def apnet_evaluate_atom(model, data_x, data_y, metrics=('brier_score', 'c_index'), single_output=False):
    """
    Calculates Integrated Brier Score and Concordance Index
    :param model:
    :param data_x:
    :param data_y: Just (t,e) np.array
    :param metrics: which metrics to compute as part of the evaluation
    :param single_output: Whether model returns 2 outputs
    :return:
    """

    y_pred = model.predict(data_x, batch_size=1)

    if not single_output:
        y_pred = y_pred[1]
        y_true = data_y[1]
    else:
        y_true = data_y

    return apnet_evaluate_predictions(y_true, y_pred, metrics=metrics)


def apnet_early_stopping_score(model_history, window):
    score = np.nan

    if 'val_ibs' in model_history and 'val_c_idx' in model_history:
        b_scores = np.array(model_history['val_ibs'])
        c_idxes = np.array(model_history['val_c_idx'])

        if window < b_scores.size and window < c_idxes.size:
            score = np.mean((0 * (1 - c_idxes) + 1 * np.sqrt(b_scores))[-window:])

    return score
