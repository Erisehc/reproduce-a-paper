from keras.callbacks import EarlyStopping, Callback
from scar_learning.apnet.apnet_evaluate_fns import apnet_evaluate_atom, apnet_early_stopping_score
import keras.backend as kbe
import numpy as np


class ApnetMetricsLogger(Callback):
    """
    Compute additional metrics and store them to logs.
    """
    def __init__(self, ancillary=False, ensemble=False, metrics=('brier_score', 'c_index')):
        self.ancillary = ancillary
        self.ensemble = ensemble
        self.metrics = metrics
        super(ApnetMetricsLogger, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        if self.ensemble:
            batch_x = list(self.validation_data[0:2])
            batch_y = self.validation_data[2]
        else:
            batch_x = self.validation_data[0]  # 1 input
            if self.ancillary:
                batch_y = self.validation_data[1]
            else:
                batch_y = self.validation_data[1:3]

        epoch_metrics = apnet_evaluate_atom(
            self.model,
            batch_x,
            batch_y,
            metrics=self.metrics,
            single_output=self.ancillary or self.ensemble
        )

        if 'brier_score' in epoch_metrics:
            logs['val_ibs'] = epoch_metrics['brier_score']
        if 'c_index' in epoch_metrics:
            logs['val_c_idx'] = epoch_metrics['c_index']

        print('brier:', '%.4f' % logs.get('val_ibs', np.nan), 'c_index:', '%.4f' % logs.get('val_c_idx', np.nan))


class ApnetEarlyStopping(EarlyStopping):
    """Stop training when a monitored quantity has stopped improving.

    # Arguments
        delay: only start monitor after these many epochs of training
        window: how many epochs to average
    """

    def __init__(
        self,
        monitor='val_loss',
        min_delta=0,
        patience=0,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=False,
        delay=0,
        window=1,
    ):
        self.delay = delay
        self.window = window

        super(ApnetEarlyStopping, self).__init__(
            monitor,
            min_delta,
            patience,
            verbose,
            mode,
            baseline,
            restore_best_weights
        )

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.delay:
            pass
        else:
            super(ApnetEarlyStopping, self).on_epoch_end(epoch, logs=logs)

    def get_monitor_value(self, logs):
        if self.monitor == 'survival':
            monitor_value = apnet_early_stopping_score(self.model.history.history, self.window)
            return monitor_value
        else:
            return super(ApnetEarlyStopping, self).get_monitor_value(logs)


class LossWeightScheduler(Callback):
    """
    Class for adjusting loss weights
    """
    def __init__(self, reconstruction_loss_weight, survival_loss_wts):
        super().__init__()
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.survival_loss_wts = survival_loss_wts

    def on_epoch_end(self, epoch, logs=None):
        # First 10 epochs, dial down reconstruction loss
        if epoch < 10:
            # Handle reconstruction loss
            reconstruction_loss = kbe.get_value(self.reconstruction_loss_weight)
            reconstruction_loss /= 2
            kbe.set_value(self.reconstruction_loss_weight, reconstruction_loss)
        # Next 10  epochs, dial up location and scale regularizers
        elif epoch < 20:
            w1 = kbe.get_value(self.survival_loss_wts)[0]
            w2 = kbe.get_value(self.survival_loss_wts)[1]
            w2 *= 1.25
            w3 = kbe.get_value(self.survival_loss_wts)[2]
            w3 *= 1.25

            kbe.set_value(self.survival_loss_wts, np.array([w1, w2, w3], dtype=kbe.floatx()))

