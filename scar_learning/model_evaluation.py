"""
    Library with functions helpful in evaluating models.
"""
from keras import backend as k_backend
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from lifelines import KaplanMeierFitter


def y_pred_at_t(y_true, y_pred, tau, threshold=.5):
    """
    Get point-in-time predictions
    :param y_true: true binary labels and event status
    :param y_pred: probability prediction
    :param tau: time for predictions
    :param threshold: cutoff to consider class 0 or 1
    :return: y_true and y_pred
    """
    event_times = y_true[:, 0]
    event_observed = y_true[:, 1]

    known_event = np.logical_not((event_observed == 0) * (event_times < tau))
    zero_mask = np.greater(event_times, tau)[known_event]

    y_true = np.logical_not(zero_mask) * 1
    y_pred = y_pred[known_event]
    if threshold:
        y_pred = (y_pred > threshold) * 1

    return y_true, y_pred


def clean_binary_y_pred(y_pred: np.ndarray) -> np.ndarray:
    """
    Take the y_predictions, clip them between 0 and 1 and collapse to 0 or 1
    :param y_pred: predicted values for y
    :return: y_pred collapsed into either 0s or 1s
    """
    return k_backend.round(k_backend.clip(y_pred, 0, 1))


def tp(y_true: np.ndarray, y_pred: np.ndarray) -> k_backend.floatx():
    """
    Calculates the number of true positives
    :param y_true: ground truth for y (i.e., labels)
    :param y_pred: predicted values for y
    :return: number of true positives
    """
    y_pred = clean_binary_y_pred(y_pred)
    true_positives = k_backend.sum(k_backend.cast(y_true * y_pred, 'float'), axis=0)
    return true_positives


def tn(y_true: np.ndarray, y_pred: np.ndarray) -> k_backend.floatx():
    """
    Calculates the number of true negatives
    :param y_true: ground truth for y (i.e., labels)
    :param y_pred: predicted values for y
    :return: number of true negatives
    """
    y_pred = clean_binary_y_pred(y_pred)
    true_negatives = k_backend.sum(k_backend.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    return true_negatives


def fp(y_true: np.ndarray, y_pred: np.ndarray) -> k_backend.floatx():
    """
    Calculates the number of false positives
    :param y_true: ground truth for y (i.e., labels)
    :param y_pred: predicted values for y
    :return: number of false positives
    """
    y_pred = clean_binary_y_pred(y_pred)
    false_positives = k_backend.sum(k_backend.cast((1 - y_true) * y_pred, 'float'), axis=0)
    return false_positives


def fn(y_true: np.ndarray, y_pred: np.ndarray) -> k_backend.floatx():
    """
    Calculates the number of false negatives
    :param y_true: ground truth for y (i.e., labels)
    :param y_pred: predicted values for y
    :return: number of false negatives
    """
    y_pred = clean_binary_y_pred(y_pred)
    false_negatives = k_backend.sum(k_backend.cast(y_true * (1 - y_pred), 'float'), axis=0)
    return false_negatives


def sens(y_true: np.ndarray, y_pred: np.ndarray) -> k_backend.floatx():
    """
    Calculates the sensitivity
    :param y_true: ground truth for y (i.e., labels)
    :param y_pred: predicted values for y
    :return: sensitivity/recall/hit rate/true positive rate
    """
    return tp(y_true, y_pred) / (tp(y_true, y_pred) + fn(y_true, y_pred) + k_backend.epsilon())


def spec(y_true: np.ndarray, y_pred: np.ndarray) -> k_backend.floatx():
    """
    Calculates the specificity
    :param y_true: ground truth for y (i.e., labels)
    :param y_pred: predicted values for y
    :return: specificity/selectivity/true negative rate
    """
    return tn(y_true, y_pred) / (fp(y_true, y_pred) + tn(y_true, y_pred) + k_backend.epsilon())


def ppv(y_true: np.ndarray, y_pred: np.ndarray) -> k_backend.floatx():
    """
    Calculates the precision
    :param y_true: ground truth for y (i.e., labels)
    :param y_pred: predicted values for y
    :return: precision/positive predictive value
    """
    return tp(y_true, y_pred) / (tp(y_true, y_pred) + fp(y_true, y_pred) + k_backend.epsilon())


def avg_acc(y_true: np.ndarray, y_pred: np.ndarray) -> k_backend.floatx():
    """
    Calculates the specificity
    :param y_true: ground truth for y (i.e., labels)
    :param y_pred: predicted values for y
    :return: average accuracy
    """
    return (sens(y_true, y_pred) + spec(y_true, y_pred)) / 2


def f1(y_true, y_pred):
    """
    Calculates the F1-score (harmonic mean of
    :param y_true: ground truth for y (i.e., labels)
    :param y_pred: predicted values for y
    :return: f1 score
    """
    p = ppv(y_true, y_pred)
    r = sens(y_true, y_pred)

    f1_score = 2 * p * r / (p + r + k_backend.epsilon())
    f1_score = tf.where(tf.is_nan(f1_score), tf.zeros_like(f1_score), f1_score)
    return k_backend.mean(f1_score)


def f1_loss(y_true, y_pred):
    """
    Loss function based on F1-score
    :param y_true: ground truth for y (i.e., labels)
    :param y_pred: predicted values for y
    :return: float F-1 score
    """

    t_p = k_backend.sum(k_backend.cast(y_true * y_pred, 'float'), axis=0)
    f_p = k_backend.sum(k_backend.cast((1 - y_true) * y_pred, 'float'), axis=0)
    f_n = k_backend.sum(k_backend.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = t_p / (t_p + f_p + k_backend.epsilon())
    r = t_p / (t_p + f_n + k_backend.epsilon())

    f_1 = 2 * p * r / (p + r + k_backend.epsilon())
    f_1 = tf.where(tf.is_nan(f_1), tf.zeros_like(f_1), f_1)
    return 1 - k_backend.mean(f_1)


def plot_metrics(dir_name: str, metrics: list, save: bool = False):
    """
    Function that reads a folder of models and plots the average loss and accuracy among models
    :param dir_name: path with the directory where models are saved
    :param metrics: list of metrics that we want to plt
    :param save: whether to save figures on disk
    :return: None
    """
    histories = []
    for file in glob.glob(os.path.join(dir_name, 'model_hist*')):
        with open(file) as f:
            data = json.load(f)
        histories.append(data)

    for (i, m) in enumerate(metrics):
        metric = np.average(np.stack([h[m] for h in histories]), axis=0)
        val_metric = np.average(np.stack([h['val_%s' % m] for h in histories]), axis=0)

        epochs = np.arange(1, 1 + len(metric))

        plt.figure(1 + i, figsize=(12, 6))
        plt.rc('grid', linestyle=":", color='black')
        plt.grid()
        plt.plot(epochs, metric, 'bo', label='Training %s' % m)
        plt.plot(epochs, val_metric, 'r-', label='Validation %s' % m, drawstyle='steps')
        plt.xlabel('Epochs')
        plt.title('Training and validation %s' % m)
        plt.legend()

        if save:
            plt.savefig(os.path.join(dir_name, 'fig_' + m + '.png'))

    plt.show()


def dice_coef(y_true, y_pred, smooth=1):
    intersection = k_backend.sum(y_true * y_pred, axis=[1, 2, 3])
    union = k_backend.sum(y_true, axis=[1, 2, 3]) + k_backend.sum(y_pred, axis=[1, 2, 3])
    return k_backend.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def weibull_nll_loss(y_true, y_pred):
    """
    Loss function based on the negative loss likelihood of a Weibull distribution with
    right censored data.
    :param y_true: tensor censor times and censor statuses
    :param y_pred: tensor of scale and shape parameters
    :return:
    """
    x = y_true[:, 0]  # x = min(t, c), t is event time and c is censor time
    d = y_true[:, 1]  # d = indicator t < c
    a = y_pred[:, 0]  # a is scale parameter of the Weibull distribution
    b = y_pred[:, 1]  # b is shape parameter of the Weibull distribution

    # a = k_backend.print_tensor(a, message="a is: ")
    # b = k_backend.print_tensor(b, message="b is: ")

    times_sc = x / (a + k_backend.epsilon())

    nll = -1 * k_backend.mean(
        d * (k_backend.log(b) + b * k_backend.log(times_sc)) - k_backend.pow(times_sc, b)
    )

    return nll


def brier_score(t, event_times, events_observed, surv_prob_at_t, censor_data=None):

    if censor_data is not None:
        kmf = KaplanMeierFitter().fit(censor_data[0], 1 - censor_data[1])
    else:
        kmf = KaplanMeierFitter().fit(event_times, 1 - events_observed)

    surv_at_t = surv_prob_at_t(t)
    score = 0
    # Uncensored branch
    uncens_cond = np.logical_and(event_times <= t, events_observed)
    uncens_t = event_times[uncens_cond]

    score += np.sum(
        np.float_power((0 - surv_at_t[uncens_cond]), 2) / kmf.survival_function_at_times(uncens_t).values
    )

    # Censored branch
    cens_cond = event_times > t
    score += np.sum(np.float_power(1 - surv_at_t[cens_cond], 2) / kmf.survival_function_at_times(t).iloc[0])

    score /= len(event_times)

    return score


def integrated_brier_score(
    survival_fn,
    event_times,
    events_observed,
    t_min=None,
    t_max=None,
    bins=100,
    censor_data=None
):
    t_min = 0 if t_min is None else max(t_min, 0)
    t_max = max(event_times) if t_max is None else min(t_max, max(event_times))
    t_min += k_backend.epsilon()  # Corrections for endpoints
    t_max -= k_backend.epsilon()  # Corrections for endpoints
    times = np.linspace(t_min, t_max, bins)

    scores = np.asarray([brier_score(t, event_times, events_observed, survival_fn, censor_data) for t in times])
    ibs = np.trapz(scores, times) / (t_max - t_min)

    return ibs


def concordance_index(tau, times, pred_times, event_observed=None, censor_data=None):
    """
    Caveat: censor_data times and times need to be the same units
    :param tau: time at which to truncate C-index calculation
    :param times: original times of event
    :param pred_times: or scores. the higher, the lower risk
    :param event_observed: event statuses
    :param censor_data: additional data for KM censor weighing
    :return:
    """
    if event_observed is None:
        event_observed = np.ones((len(times),))

    correct_pairs = 0
    all_pairs = 0

    if censor_data is not None:
        kmf = KaplanMeierFitter().fit(censor_data[0], 1 - censor_data[1])
    else:
        kmf = KaplanMeierFitter().fit(times, 1 - event_observed)

    g = 1 / (kmf.survival_function_at_times(times).values ** 2)
    for j in range(len(times)):
        this_correct_pairs = 1 * g * (times < times[j]) * (times < tau) * (pred_times < pred_times[j])
        this_all_pairs = 1 * g * (times < times[j]) * (times < tau)

        correct_pairs += np.sum(this_correct_pairs[event_observed > 0])
        all_pairs += np.sum(this_all_pairs[event_observed > 0])

    c_idx = correct_pairs / all_pairs

    return c_idx
