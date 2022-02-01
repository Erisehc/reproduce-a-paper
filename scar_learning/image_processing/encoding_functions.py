"""
Library with functions which convert to and from cardiac images and encoded forms used in model training
"""
import keras.backend as k_backend
import numpy as np
import scar_learning.image_processing.constants_mri_image_proc as const

SRC_INDEX = 0
MSK_INDEX = 1

PARAMETERS_3D_INTENSITY = {
    'scar_scalar': 1 / 255,
    'outside_value': -1,
}

PARAMETERS_3D_4REGION_ONE_HOT = {
    'scar_gray_threshold': .5,
}

PARAMETERS_3D_3REGION_ONE_HOT = {
    'scar_gray_threshold': .5,
}

PARAMETERS_3D_OTSU = {
    'n_max_sigma': 6,  # number of standard deviations away from healthy avg., where there is no conductivity
    'minimum_level': .1,  # level of signal for signal less than or equal to mu
    'maximum_level': 1,  # maximum level of signal for non-conducting tissue
    'outside_value': 0,  # value for region outside of mask
}


def encode_3d_intensity(output_shape: tuple, images: np.ndarray) -> np.ndarray:
    """
    Everything that is not heart is OUTSIDE_VALUE, healthy
    myocardial tissue is HEALTHY and scar information is re-scaled by SCAR_SCALAR. This has the follwing benefits:
        - we provide information regarding myocardium vs non-myocardium
        - scar values are at most 1
        - preserves heterogeneity across slices and patients
        - can easily be applied on augmented/transformed images
    :param output_shape: the desired shape of the output tensor
    :param images: 5-d tensor of images for a patient as returned from self.get_image_tensor_for_id
    :return: 3-d tensor encoding the scar information and its position relative to the myocardium
    """

    if output_shape[-1] is not 1:
        raise ValueError('Invalid number of channels for encoding type (needs 1, found %d)' % output_shape[-1])

    scar_scalar = PARAMETERS_3D_INTENSITY['scar_scalar']
    outside_value = PARAMETERS_3D_INTENSITY['outside_value']

    encoded_x = np.zeros(output_shape, dtype=k_backend.floatx())

    for z_slice in range(output_shape[0]):  # z-stack loop
        myo_image = images[SRC_INDEX, z_slice, :, :, :]  # get myocardium
        reverse_myo_mask = np.logical_not(myo_image)  # get the non-heart region

        scar_image = images[MSK_INDEX, z_slice, :, :, :]  # get scar
        scar_image_scaled = scar_image * scar_scalar  # re-scale the values
        scar_image_scaled[reverse_myo_mask] = outside_value  # highlight region outside of ROI

        encoded_x[z_slice] = scar_image_scaled

    return encoded_x


def decode_3d_intensity(encoded_image: np.ndarray) -> np.ndarray:
    """
    Takes in encoded image as returned by encode_3d_intensity and returns a human readable version with dark on
    the outside of the myocardium, green for healthy myocardial tissue and actual intensity for the scar.
    :param encoded_image: output of encode_3d_intensity
    :return: numpy.ndarray of size (*image_tensor.shape, 3)
    """
    scar_scalar = PARAMETERS_3D_INTENSITY['scar_scalar']
    outside_value = PARAMETERS_3D_INTENSITY['outside_value']

    decoded_images = np.zeros((*encoded_image.shape[0:3], 3))

    for z_slice in range(encoded_image.shape[0]):  # z-stack loop
        this_slice = encoded_image[z_slice, :, :]

        outside_mask = np.isclose(this_slice, outside_value)
        healthy_mask = np.isclose(this_slice, 0)

        scar_info = this_slice.copy()
        scar_info[np.logical_or(outside_mask, healthy_mask)] = 0
        scar_info *= (1/scar_scalar)

        decoded_images[z_slice, :, :, :] = scar_info
        decoded_images[z_slice, :, :, 1] += 255 * np.squeeze(healthy_mask)

    return decoded_images


def encode_3d_4region_one_hot(output_shape: tuple, images: np.ndarray) -> np.ndarray:
    """
    Encode as 4 channels, corresponding to 4 regions: outside ROI, healthy/remote myocardium, gray zone, core.
    The distinction between gray zone and core is based on half max width method( >50% max intensity is core)
    :param output_shape: the desired shape of the output tensor
    :param images: 5-d tensor of images for a patient as returned from self.get_image_tensor_for_id
    :return: 4-d tensor encoding the scar information and its position relative to the myocardium
    """

    if output_shape[-1] is not 4:
        raise ValueError('Invalid number of channels for encoding type (needs 4, found %d)' % output_shape[-1])

    gray_core_cutoff = PARAMETERS_3D_4REGION_ONE_HOT['scar_gray_threshold']

    encoded_x = np.zeros(output_shape, dtype=k_backend.floatx())

    for z_slice in range(output_shape[0]):  # z-stack loop
        myo_image = images[SRC_INDEX, z_slice, :, :, :]  # get myocardium
        scar_image = images[MSK_INDEX, z_slice, :, :, :]  # get scar

        encoded_x[z_slice, :, :, 0] = np.squeeze(np.logical_not(np.logical_or(myo_image, scar_image)))  # outside
        encoded_x[z_slice, :, :, 1] = np.squeeze(np.logical_and(myo_image, np.logical_not(scar_image)))  # healthy

        if not np.isclose(np.sum(scar_image), 0):
            encoded_x[z_slice, :, :, 2] = np.squeeze(np.logical_and(
                scar_image >= np.min(scar_image[np.nonzero(scar_image)]),
                scar_image < gray_core_cutoff * np.max(scar_image)
            ))  # gray zone
            encoded_x[z_slice, :, :, 3] = np.squeeze(scar_image >= gray_core_cutoff * np.max(scar_image))  # core
        else:
            encoded_x[z_slice, :, :, 2] = np.zeros(np.squeeze(scar_image).shape, dtype=k_backend.floatx())
            encoded_x[z_slice, :, :, 3] = np.zeros(np.squeeze(scar_image).shape, dtype=k_backend.floatx())

    return encoded_x


def decode_3d_4region_one_hot(encoded_image: np.ndarray) -> np.ndarray:
    """
    Transform the encoded image returned by encode_3d_4region_one_hot into a human-readable format. Resulting image
    is dark outside the myocardium, green for healthy myocardium and red/yellow depending on severity of the scar.
    :param encoded_image: output of encode_3d_4region_one_hot
    :return: numpy.ndarray of same width and height with 4 channels
    """
    decoded_images = np.zeros(encoded_image.shape[0:3] + (3,))

    for z_slice in range(encoded_image.shape[0]):  # z-stack loop
        this_slice = encoded_image[z_slice, :, :, :]
        decoded_images[z_slice, :, :, 0] = 255 * np.logical_or(this_slice[:, :, 2], this_slice[:, :, 3])
        decoded_images[z_slice, :, :, 1] = 255 * np.logical_or(this_slice[:, :, 2], this_slice[:, :, 1])

    return decoded_images


def encode_3d_3region_one_hot(images: np.ndarray) -> np.ndarray:
    """
    Encode as 3 channels, corresponding to 3 regions: healthy/remote myocardium, gray zone, core.
    The distinction between gray zone and core is based on half max width method( >50% max intensity is core)
    :param images: 5-d tensor of images for a patient as returned from self.get_image_tensor_for_id
    :return: 3-d tensor encoding the scar information and its position relative to the myocardium
    """

    gray_core_cutoff = PARAMETERS_3D_3REGION_ONE_HOT['scar_gray_threshold']

    encoded_x = np.zeros(images.shape[0:3] + (3,), dtype=k_backend.floatx())

    for z_slice in range(images.shape[-1]):  # z-stack loop
        myo_image = images[SRC_INDEX, z_slice, :, :, :]  # get myocardium
        scar_image = images[MSK_INDEX, z_slice, :, :, :]  # get scar

        encoded_x[z_slice, :, :, 0] = np.squeeze(np.logical_and(myo_image, np.logical_not(scar_image)))  # healthy

        if not np.isclose(np.sum(scar_image), 0):
            encoded_x[z_slice, :, :, 1] = np.squeeze(np.logical_and(
                scar_image >= np.min(scar_image[np.nonzero(scar_image)]),
                scar_image < gray_core_cutoff * np.max(scar_image)
            ))  # gray zone
            encoded_x[z_slice, :, :, 2] = np.squeeze(scar_image >= gray_core_cutoff * np.max(scar_image))  # core
        else:
            encoded_x[z_slice, :, :, 1] = np.zeros(np.squeeze(scar_image).shape, dtype=k_backend.floatx())
            encoded_x[z_slice, :, :, 2] = np.zeros(np.squeeze(scar_image).shape, dtype=k_backend.floatx())

    return encoded_x


def decode_3d_3region_one_hot(encoded_image: np.ndarray) -> np.ndarray:
    """
    Transform the encoded image returned by encode_3d_4region_one_hot into a human-readable format. Resulting image
    is dark outside the myocardium, green for healthy myocardium and red/yellow depending on severity of the scar.
    :param encoded_image: output of encode_3d_3region_one_hot
    :return: numpy.ndarray of same width and height with 3 channels
    """

    decoded_images = np.zeros(encoded_image.shape[0:3] + (3,))

    for z_slice in range(encoded_image.shape[0]):  # z-stack loop
        this_slice = encoded_image[z_slice, :, :, :]
        decoded_images[z_slice, :, :, 0] = 255 * np.logical_or(this_slice[:, :, 1], this_slice[:, :, 2])
        decoded_images[z_slice, :, :, 1] = 255 * np.logical_or(this_slice[:, :, 1], this_slice[:, :, 0])

    return decoded_images


def encode_z_score(images: np.ndarray) -> np.ndarray:
    """
    Return the raw images after scaling by mean and stddev.
    :param images: Input to be encoded
    :return: Encoded input as numpy.ndarray
    """

    data_source = images[SRC_INDEX]
    data_encoded = np.zeros(data_source.shape)

    # TODO: Review if this should handle 0s differently - mean is artificially low is many 0s due to border effects
    for z in range(data_source.shape[-1]):
        this_slice = data_source[:, :, z]
        this_slice_normalized = (this_slice - np.mean(this_slice)) / np.std(this_slice)
        data_encoded[:, :, z] = this_slice_normalized

    return data_encoded


def encode_apply_mask(images: np.ndarray, include_mask: bool = True) -> np.ndarray:
    """
    Apply the mask and return the images
    :param images: Input to be encoded
    :param include_mask: whether to return both img and mask in the last axis
    :return: Encoded input as numpy.ndarray
    """
    # TODO: Consider adding anisotropic diffusion filter from medpy
    data_src = images[SRC_INDEX]
    data_msk = images[MSK_INDEX]

    data_encoded = data_src.copy()
    data_encoded[np.logical_not(data_msk > 0)] = 0  # blank out non-mask voxels

    max_allowable_intensity = np.percentile(data_src[data_msk > 0], 99, interpolation='nearest')
    data_encoded[data_encoded > max_allowable_intensity] = max_allowable_intensity
    data_encoded = data_encoded / max_allowable_intensity  # scale

    if include_mask:
        return np.concatenate([data_encoded, data_msk / 255], axis=-1)
    else:
        return data_encoded


def encode_2_stage_normalization(images: np.ndarray, include_seg: bool = False) -> np.ndarray:
    """
    Apply the mask and return the images
    :param images: Input to be encoded
    :param include_seg: whether to include
    :return: Encoded input as numpy.ndarray
    """
    data_src = np.squeeze(images[SRC_INDEX])
    data_msk = np.squeeze(images[MSK_INDEX])
    no_slices = data_src.shape[-1]
    # scale each slice by respective bloodpool. If bloodpool unavailabe, use closest one
    bp_medians = np.array([])
    for z in range(no_slices):
        bp_intensities = data_src[:, :, z][np.isclose(data_msk[:, :, z], const.GT_LABELS['lvbp'])]
        if np.sum(bp_intensities):
            bp_medians = np.append(bp_medians, np.median(bp_intensities))
        else:
            bp_medians = np.append(bp_medians, np.nan)

    # Linearly interpolate to get median bp intensity for slices with no bp
    nan_bp = np.isnan(bp_medians)
    bp_medians[nan_bp] = np.interp(
        np.where(nan_bp)[0],
        np.where(np.logical_not(nan_bp))[0],
        bp_medians[np.logical_not(nan_bp)]
    )

    # Divide by twice the median bp to make that region ~.5
    data_src_scaled = np.stack([data_src[:, :, z] / (2 * bp_medians[z]) for z in range(no_slices)], axis=-1)

    # Zero out low intensities
    min_viable = np.percentile(data_src_scaled[np.isclose(data_msk, const.GT_LABELS['lvmyo'])], 1)
    data_src_scaled -= min_viable
    data_src_scaled = np.clip(data_src_scaled, 0, np.inf)
    data_src_scaled[np.logical_not(np.isclose(data_msk, const.GT_LABELS['lvmyo']))] = 0

    if include_seg:
        data_src_scaled = np.stack([data_src_scaled, data_msk / const.GT_LABELS['lvmyo']], axis=-1)
    else:
        data_src_scaled = np.expand_dims(data_src_scaled, axis=-1)

    return data_src_scaled


def encode_ancillary_data(df):
    """
    Data normalization. Fill it explicit normalization for every covariate in config_data.APNET_ANCILLARY_COVARIATES
    :param df: Dataframe of covariates
    :return: scaled dataframe
    """
    df = df.copy()
    to_scl = [
        'age',
        'lvef_noncmr',
        'lvef_cmr',
        'infarct_pct',
        'lv_mass_ed',
        'ekg_hr',
        'ekg_qrs_dur',
        'ischemic_etiology_time'
    ]
    for covariate in to_scl:
        if covariate in df:
            df[covariate] = (df[covariate] - df[covariate].mean()) / df[covariate].std()

    to_dummify = [
        'ethnicity'
    ]

    for covariate in to_dummify:
        if covariate in df:
            distinct_values = np.unique(df.loc[:, [covariate]].values)
            for dv in distinct_values[:-1]:
                df['%s_%s' % (covariate, dv)] = df[covariate] == dv
            df.drop(labels=covariate, axis='columns', inplace=True)

    return df
