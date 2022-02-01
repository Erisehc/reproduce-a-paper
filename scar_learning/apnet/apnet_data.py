"""
Generate the data needed for ApNet
"""
# import numpy as np
from functools import partial
import scar_learning.config_data as data_config
from scar_learning.apnet.apnet_datagen import ApNetDataGenerator, ApNetDataGeneratorAux
from scar_learning.image_processing import encoding_functions as ef


def get_training_generator(ids, batch_size, data_aug_params=None, verbose=True, ancillary=False, ensemble=False):
    # Ancillary Generator
    train_data_gen_aux = ApNetDataGeneratorAux(
        x_ids=ids,
        batch_size=batch_size,
        shuffle=True,
        include_sample_weights=True
    )

    # Conv Generator
    data_gen_params = {
        'decode_function': lambda _: _,
        'encode_function': partial(ef.encode_2_stage_normalization, include_seg=data_config.OUTPUT_CHANNELS == 2),
        'seed': 2020,  # seed for deterministic shuffling
        'shuffle': True,  # whether to randomly shuffle data in between epochs
        'target_size': data_config.OUTPUT_SPECS['output_resolution'],  # the shape of the input
        'batch_size': batch_size,
        'include_sample_weights': True,
    }

    if data_aug_params is not None:
        data_gen_params['data_augmentation_params'] = data_aug_params
    else:
        data_gen_params['data_augmentation_params'] = {
            'rotation_range': 0,  # degrees to rotate the images in the x-y plane
            'width_shift_pct': 0,  # shift images horizontally (fraction of total available space)
            'height_shift_pct': 0,  # shift images vertically (fraction of total available space)
            'depth_shift_pct': 0,  # shift images in the short axis plane (fraction of total available space)
        }

    train_data_generator_conv = ApNetDataGenerator(ids, **data_gen_params)

    if ensemble:
        train_data_generator = (
            ([a[0], b[0]], a[1], a[-1]) for (a, b) in zip(train_data_gen_aux, train_data_generator_conv)
        )
    else:
        if ancillary:
            train_data_generator = train_data_gen_aux
        else:
            train_data_generator = train_data_generator_conv

    if verbose:
        print('Constructing training generator using: %s' % ids)

    return train_data_generator


def get_validation_data(ids, verbose=True, ancillary=False, ensemble=False, shuffle=True):
    data_gen_params = {
        'batch_size': 999,  # how many samples (training and validation) should we include in each batch
        'seed': 2020,  # seed for deterministic shuffling
        'shuffle': shuffle,  # whether to randomly shuffle data in between epochs
        'include_sample_weights': True,
    }

    # Ancillary Generator
    val_data_gen_aux = ApNetDataGeneratorAux(ids, **data_gen_params)

    # Conv Generator
    addtl_params = {
        'decode_function': lambda _: _,
        'encode_function': partial(ef.encode_2_stage_normalization, include_seg=data_config.OUTPUT_CHANNELS == 2),
        'target_size': data_config.OUTPUT_SPECS['output_resolution'],  # the shape of the input
    }

    val_data_gen_conv = ApNetDataGenerator(ids, **data_gen_params, **addtl_params)

    if ensemble:
        val_data_generator = (
            ([a[0], b[0]], a[1], a[-1]) for (a, b) in zip(val_data_gen_aux, val_data_gen_conv)
        )
    else:
        if ancillary:
            val_data_generator = val_data_gen_aux
        else:
            val_data_generator = val_data_gen_conv

    if verbose:
        print('Constructing validation generator using: %s' % ids)

    return next(val_data_generator)
