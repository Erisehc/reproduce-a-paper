"""
ApNetDataGenerator is a generator that returns mini-batches of volumes with their corresponding labels
"""

from keras.preprocessing.image import Iterator
import scar_learning.image_processing.constants_mri_image_proc as const
import heart_volume.heart_classes as hc
import keras.backend as k_backend
import numpy as np
import os
import pickle
import scar_learning.config_data as data_config
from scar_learning.image_processing import encoding_functions as ef


SRC_INDEX = 0
MSK_INDEX = 1


class ApNetDataGenerator(Iterator):
    """
    Iterator yielding data from IDs
    """
    def __init__(
            self,
            x_ids,
            batch_size=32,
            target_size=(64, 64, 12),
            data_augmentation_params=None,
            encode_function=lambda images: images,
            decode_function=lambda encoded: encoded,
            shuffle=False,
            seed=None,
            include_sample_weights=False,
    ):
        """
        Initialization
        :param x_ids: list of IDs corresponding to the X dataset
        :param data_augmentation_params: dictionary of parameters for augmenting data
            Currently handles 'rotation angle' and 'height/width/depth_shift_pct'
        :param batch_size: Integer, size of a batch.
        :param target_size: tuple of integers, dimensions to resize input images to (i, j, k)
        :param shuffle: Boolean, whether to shuffle the data between epochs.
        :param seed: Seed for shuffling data between epochs
        # :param save_to_dir: Optional directory where to save the pictures being yielded, in a viewable format.
        #     This is useful for visualizing the random transformations being applied, for debugging purposes.
        # :param save_format: Format to use for saving sample images (if `save_to_dir` is set).
        :param include_sample_weights: Whether to return sample weights
        """
        self.x_ids = np.asarray(x_ids)
        self.encode_function = encode_function
        self.decode_function = decode_function

        if data_augmentation_params is None:
            self.data_augmentation_params = {}
        else:
            self.data_augmentation_params = data_augmentation_params

        target_size = tuple(target_size)
        if len(target_size) is not 3:
            raise ValueError('The generator expects 3 dimensions, found %d in target_size' % len(target_size))
        else:
            self.target_size = target_size

        # self.save_to_dir = save_to_dir
        # self.save_format = save_format

        # Load all data upfront
        self._x_data = np.asarray([self.encode_function(self.read_image_tensor_for_id(x)) for x in x_ids])
        self._buffers = self._calculate_buffers()

        # Set the corresponding y_labels for x_ids
        all_labels = data_config.label_data()
        t_d_table = np.asarray([all_labels[x] for x in x_ids], dtype=k_backend.floatx())
        self._y_labels = np.array([[np.log(a[0]), a[1]] for a in t_d_table], dtype=k_backend.floatx())

        self.include_sample_weights = include_sample_weights

        super(ApNetDataGenerator, self).__init__(self.x_ids.shape[0], batch_size, shuffle, seed)

    @staticmethod
    def read_image_tensor_for_id(unique_id: str) -> np.ndarray:
        """
        Reads the images corresponding to patient unique_id. Then, if adjust shape to
        self.target_size if needed
        :param unique_id: unique ID for patient
        :return: 5d np.ndarray, first index is image/mask, then target_size, then channels
        """
        fname = os.path.join(data_config.DATA_FILE_PATH, '%s.pickle' % unique_id)
        with open(fname, 'rb') as f:
            hv_dict = pickle.load(f)

        hv_obj = hc.HeartVolume(**hv_dict)
        img_vol = hv_obj.pixel_array
        msk_vol = hv_obj.segmentation
        for r in {'rvbp', 'rvmyo'}:
            msk_vol[msk_vol == const.GT_LABELS[r]] = 0

        volumes = np.stack([img_vol, msk_vol], axis=0)
        volumes = np.expand_dims(volumes, axis=-1)

        return volumes

    def _calculate_buffers(self):
        buffers = np.zeros(shape=(len(self.x_ids), 3, 2))
        for pidx in range(len(self.x_ids)):
            for axis in [0, 1, 2]:
                x = self._x_data[pidx].copy() > 0
                x = np.moveaxis(x, axis, 0)

                buffer = [0, 0]
                for i, j in enumerate([-1, 1]):
                    for k in range(1, x.shape[0]):
                        if np.isclose(np.sum(np.abs(x[k * (-j)])), 0):
                            buffer[i] += j
                        else:
                            break

                buffers[pidx, axis] = buffer

        return buffers

    def _random_3d_transform(self, x: np.ndarray, buffers) -> np.ndarray:
        """
        Random transformations of the whole volume. Currently handles translations only
        :param x: 3D tensor representing the image
        :param buffers: how much space to translate
        :return: transformed 3d volume
        """

        aug_params = self.data_augmentation_params

        # Pan/shift
        axis_to_key = {
            0: 'height_shift_pct',
            1: 'width_shift_pct',
            2: 'depth_shift_pct'
        }
        for ax in range(x.ndim):
            key = axis_to_key.get(ax, None)

            if key is None or key not in aug_params:
                continue

            # buffer is computed using the mask
            buffer = buffers[ax] * aug_params[key]
            shift = round(np.random.uniform(low=buffer[0], high=buffer[1]))
            x = np.roll(x, shift=shift, axis=ax)

        return x

    def _get_batches_of_transformed_samples(self, index_array):
        """
        Gets a batch of transformed samples.
        :param index_array: array of sample indices to include in batch.
        :return: A batch of transformed samples.
        """
        mini_batches = [self._random_3d_transform(self._x_data[idx], self._buffers[idx]) for idx in index_array]

        batch_time_events = self._y_labels[index_array]
        batch_x = np.stack(mini_batches, axis=0)

        batch_y = [
            np.stack(mini_batches, axis=0),
            batch_time_events,  # (t, e)
        ]
        if self.include_sample_weights:
            sample_weights_reconst = np.asarray(
                1 / np.linalg.norm(batch_x.reshape((batch_x.shape[0], -1)), axis=1), dtype=k_backend.floatx()
            )
            w1 = 1 / np.maximum(1, np.sum(self._y_labels[:, 1]))
            w2 = 1 / np.maximum(1, np.sum(np.logical_not(self._y_labels[:, 1])))
            sample_weights_surv = np.array([w1 if i else w2 for i in batch_time_events[:, 1]], dtype=k_backend.floatx())

            return batch_x, batch_y, [sample_weights_reconst, sample_weights_surv]
        else:
            return batch_x, batch_y


class ApNetDataGeneratorAux(Iterator):
    """
    Iterator yielding data from IDs
    """
    def __init__(
            self,
            x_ids,
            batch_size=32,
            shuffle=False,
            seed=None,
            include_sample_weights=False,
    ):
        """
        Initialization
        :param x_ids: list of IDs corresponding to the X dataset
            Currently handles 'rotation angle' and 'height/width/depth_shift_pct'
        :param batch_size: Integer, size of a batch.
        :param shuffle: Boolean, whether to shuffle the data between epochs.
        :param seed: Seed for shuffling data between epochs
        :param include_sample_weights: Whether to return sample weights
        """
        self.x_ids = np.asarray(x_ids)

        # Load the x-values matrix
        self._x_values = ef.encode_ancillary_data(data_config.ancillary_data())

        # Set the corresponding y_labels for x_ids
        all_labels = data_config.label_data()
        t_d_table = np.asarray([all_labels[x] for x in x_ids], dtype=k_backend.floatx())
        self._y_labels = np.array([[np.log(a[0]), a[1]] for a in t_d_table], dtype=k_backend.floatx())

        self.include_sample_weights = include_sample_weights

        super(ApNetDataGeneratorAux, self).__init__(self.x_ids.shape[0], batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        """
        Gets a batch of transformed samples.
        :param index_array: array of sample indices to include in batch.
        :return: A batch of transformed samples.
        """
        batch_x = np.asarray(self._x_values.loc[self.x_ids[index_array]].values, dtype=k_backend.floatx())
        batch_y = self._y_labels[index_array]

        if self.include_sample_weights:
            w1 = 1 / np.maximum(1, np.sum(self._y_labels[:, 1]))
            w2 = 1 / np.maximum(1, np.sum(np.logical_not(self._y_labels[:, 1])))
            sample_weights_surv = np.array([w1 if i else w2 for i in batch_y[:, 1]], dtype=k_backend.floatx())
            return batch_x, batch_y, sample_weights_surv
        else:
            return batch_x, batch_y
