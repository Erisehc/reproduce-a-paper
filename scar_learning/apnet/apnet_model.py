from keras.layers import (
    Input,
    Conv3D,
    Flatten,
    Dense,
    Reshape,
    BatchNormalization,
    Dropout,
    Concatenate,
    Dot,
    Activation,
    LeakyReLU,
    UpSampling3D,
    GaussianNoise,
    MaxPooling3D,
    Lambda
)
from definitions import MULTI_GPU
from keras import backend as kbe
from keras import losses
from keras import optimizers
from keras import Model
from keras.layers import Layer
from keras.layers.merge import _Merge
from keras.regularizers import l1_l2
from keras.utils import multi_gpu_model
import keras.initializers
import numpy as np
import scar_learning.config_data as data_config
import tensorflow as tf

KERNEL_INITIALIZER = keras.initializers.glorot_uniform()
MODEL_FIT = data_config.APNET_MODEL


class WeightedSum(Layer):
    def __init__(self, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)

    def build(self, input_shape=1):
        self.w = self.add_weight(
            name='alpha',
            shape=(1,),
            initializer='ones',
            trainable=True,
            constraint=tf.keras.constraints.MinMaxNorm()
        )
        super(WeightedSum, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return self.w * inputs[0] + (1 - self.w) * inputs[1]

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class LinearCombination(_Merge):
    def __init__(self, no_components, **kwargs):
        self.no_components = no_components
        super(LinearCombination, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(
            name='loadings',
            shape=(self.no_components,),
            initializer=keras.initializers.constant(value=1/self.no_components),
            trainable=True,
        )

        self.b = self.add_weight(
            name='bias',
            shape=(self.no_components,),
            initializer='zeros',
            trainable=True,
        )
        super(LinearCombination, self).build(input_shape)

    def _merge_function(self, inputs):
        output = self.b
        for i in range(len(inputs)):
            output += self.w[i] * inputs[i]
        return output


def filters_fn(depth, filt_0): return min((2 ** depth) * filt_0, 512)


def downsample_fn(depth):
    if depth == 4:
        return (2, 2, 1), (2, 2, 1), (2, 2, 2), (2, 2, 2)
    elif depth == 2:
        return (4, 4, 1), (4, 4, 4)
    else:
        raise ValueError('Unsupported depth: %d' % depth)


def gumbel_survival_fn_gen(location, scale):
    def gumbel_survival_fn(t):
        return np.exp(-np.exp((np.log(t) - location) / scale))

    return gumbel_survival_fn


def logistic_survival_fn_gen(location, scale):
    def logistic_survival_fn(t):
        return 1 / (1 + np.exp((np.log(t) - location) / scale))

    return logistic_survival_fn


def cox_ph_survival_fn_gen(location, scale, baseline):
    def cox_survival_fn(t):
        cum_haz_at_t = np.clip(np.interp(t, baseline.index.values, baseline['KM_estimate'].values), 0, 1)
        return np.exp(-cum_haz_at_t * np.exp(-location))

    return cox_survival_fn


def survival_fn_gen(y_pred, baseline=None, model_fit=MODEL_FIT):
    loc = y_pred[:, 0]
    scl = np.exp(y_pred[:, 1])

    if model_fit == 'weibull':
        return gumbel_survival_fn_gen(loc, scl)
    elif model_fit == 'loglogistic':
        return logistic_survival_fn_gen(loc, scl)
    elif model_fit == 'cox_ph':
        return cox_ph_survival_fn_gen(loc, scl, baseline)
    else:
        raise ValueError('Unrecognized model fit: %s' % model_fit)


def gumbel_nll_loss(mu, log_sigma, x, d):
    """
    Loss function based on the negative log likelihood of a Gumbel distribution with
    right censored data.
    :param mu: tensor of the location latent parameter
    :param log_sigma: tensor of the scale latent parameter
    :param x: tensor of min(log(t), log(c)), t is event time and c is censor time
    :param d: tensor of indicator t < c
    :return:
    """
    x_scaled = (x - mu) / kbe.exp(log_sigma)
    nll = -(d * (x_scaled - log_sigma) - kbe.exp(x_scaled))

    return nll


def logistic_nll_loss(mu, log_sigma, x, d):
    """
    Loss function based on the negative log likelihood of a logistic distribution with
    right censored data.
    :param mu: tensor of the location latent parameter
    :param log_sigma: tensor of the scale latent parameter
    :param x: tensor of min(log(t), log(c)), t is event time and c is censor time
    :param d: tensor of indicator t < c
    :return:
    """
    x_scaled = (x - mu) / kbe.exp(log_sigma)
    nll = x_scaled + d * log_sigma + (1 + d) * kbe.log(1 + kbe.exp(-x_scaled))

    return nll


def cox_nll_loss(mu, log_sigma, x, d):
    """
    Loss function based on the negative log likelihood of the Cox PH model
    right censored data.
    :param mu: negative of the log hazard ratio
    :param log_sigma: not used
    :param x: tensor of min(log(t), log(c)), t is event time and c is censor time
    :param d: tensor of indicator t < c
    :return:
    """
    nu = -mu
    risk_set = kbe.cast(
        kbe.less_equal(kbe.expand_dims(x, axis=-1), kbe.transpose(kbe.expand_dims(x, axis=-1))),
        'float32'
    )
    risk_set_contrib = kbe.flatten(kbe.log(kbe.dot(risk_set, kbe.exp(kbe.expand_dims(nu, axis=-1)))))

    nll = -d * (nu - risk_set_contrib)

    return nll


def nll_loss_gen(model_fit=MODEL_FIT):
    if model_fit == 'weibull':
        return gumbel_nll_loss
    elif model_fit == 'loglogistic':
        return logistic_nll_loss
    elif model_fit == 'cox_ph':
        return cox_nll_loss
    else:
        raise ValueError('Unrecognized model fit: %s' % model_fit)


def snr_loss(mu, sigma, max_val=2):
    """
    Loss function for signal to noise ratio
    :param mu: predicted mean
    :param sigma: predicted sigma
    :param max_val: max value for mu/sigma
    :return:
    """
    snr = mu / sigma
    return kbe.square(kbe.maximum(snr - max_val, 0))


def l1l2_hinge_loss(mu, x, d, l1l2=1):
    """
    Loss function based on l1 loss between median time and real times taking into account
    right censored data.
    :param mu: tensor of the location latent parameter
    :param x: tensor of min(log(t), log(c)), t is event time and c is censor time
    :param d: tensor of indicator t < c
    :param l1l2: whether to use l1 or l2
    :return:
    """
    if l1l2 == 1:
        return d * kbe.abs(x - mu) + (1 - d) * kbe.maximum(x - mu, 0.)
    elif l1l2 == 2:
        return d * kbe.square(x - mu) + (1 - d) * kbe.square(kbe.maximum(x - mu, 0.))
    else:
        raise ValueError('Unrecognized l12 type: %s' % l1l2)


def survival_loss_gen(weights):
    def survival_loss(y_true, y_pred):
        x = y_true[:, 0]
        d = y_true[:, 1]
        mu = y_pred[:, 0]
        log_sigma = y_pred[:, 1]

        loss1 = nll_loss_gen()(mu, log_sigma, x, d)
        # loss2 = l1l2_hinge_loss(mu, x, d)
        # loss3 = kbe.square(kbe.exp(log_sigma))
        # loss4 = snr_loss(mu, kbe.exp(log_sigma))
        loss2 = 0
        loss3 = 0
        loss4 = 0
        loss = (weights[0] * loss1 + weights[1] * loss2 + weights[2] * loss3 + weights[3] * loss4) / kbe.sum(weights)

        return loss

    return survival_loss


def reconstruction_loss(y_true, y_pred):
    return kbe.mean(losses.mse(y_true, y_pred), axis=[1, 2, 3])


def exp_softplus_activation(ab):
    """
    Activation where the first element is exp and second is softplus
    :param ab: neuron
    :return: activated neuron
    """
    a = kbe.exp(ab[:, 0])
    b = kbe.softplus(ab[:, 1])

    a = kbe.reshape(a, (kbe.shape(a)[0], 1))
    b = kbe.reshape(b, (kbe.shape(b)[0], 1))

    return kbe.concatenate((a, b), axis=1)


def loglogistic_activation(mu_logsig):
    """
    Activation which ensures mu is between -3 and 3 and sigma is such that
    prediction is not more precise than 1 / n of a year.
    :param mu_logsig:
    :return:
    """
    n = 12  # 1 / n is the fraction of the year in which at least p quantile of the distribution lies
    p = .95  # quantile
    mu = kbe.clip(mu_logsig[:, 0], -3, 3)
    sig = kbe.exp(mu_logsig[:, 1])
    thrs = kbe.log((1 / (2 * n)) * (kbe.exp(-mu) + kbe.sqrt((2 * n) ** 2 + kbe.exp(-2 * mu)))) / \
        kbe.log(kbe.cast_to_floatx(p / (1 - p)))

    logsig = kbe.log(thrs + kbe.relu(sig - thrs))

    mu = kbe.reshape(mu, (kbe.shape(mu)[0], 1))
    logsig = kbe.reshape(logsig, (kbe.shape(logsig)[0], 1))

    new = kbe.concatenate((mu, logsig), axis=1)
    return new


def cox_ph_activation(haz):
    """
    Activation which ensures the second prediction is 0
    :param haz: hazard rate
    :return:
    """
    h = haz[:, 0]
    dummy = 0 * haz[:, 1]
    h = kbe.reshape(h, (kbe.shape(h)[0], 1))
    dummy = kbe.reshape(dummy, (kbe.shape(dummy)[0], 1))

    return kbe.concatenate((h, dummy), axis=1)


def apnet_conv_encode_model(
        input_shape,
        network_depth,
        no_convolutions,
        conv_filter_no_init,
        conv_kernel_size,
        latent_representation_dim,
        l1,
        l2,
        dropout_value,
        use_batch_normalization,
        activation,
        gaussian_noise_std,
        verbose,
        trainable=True,
        weights=None,
) -> (Model, tuple):
    """
    This model encodes a 3D volume into a tensor
    :param input_shape: input shape of the tensor as a tuple
    :param network_depth: the number of times data is downsampled/upsampled after convolutions (2 or 4)
    :param no_convolutions: number of convolutions per level
    :param conv_filter_no_init: beginning convolution number of filters
    :param conv_kernel_size: kernel window for convolutions
    :param latent_representation_dim: vector length in the encoded space
    :param l1: l1 regularization parameter
    :param l2: l2 regularization parameter
    :param dropout_value: value for the dropout layer
    :param use_batch_normalization: boolean of whether to use batch normalization
    :param activation: type of activation layer. e.g., 'relu'. Also supports 'leakyrelu'
    :param gaussian_noise_std: standard deviation of gaussian noise for images
    :param verbose: whether we should print out a summary of the model
    :param trainable: whether the model encoder-decoder is trainable
    :param weights: dictionary of weights to initialize the model
    :return:
    """

    input_vol = Input(shape=input_shape, name='encoder_input')

    x = input_vol

    if gaussian_noise_std:
        x = GaussianNoise(gaussian_noise_std)(x)

    for i in range(network_depth):
        for j in range(no_convolutions):
            x = Conv3D(
                filters=filters_fn(i, conv_filter_no_init),
                kernel_size=conv_kernel_size,
                kernel_initializer=KERNEL_INITIALIZER,
                padding='same'
            )(x)

            if activation == 'leakyrelu':
                x = LeakyReLU()(x)
            else:
                x = Activation(activation)(x)
            x = BatchNormalization()(x) if use_batch_normalization else x

        x = MaxPooling3D(pool_size=downsample_fn(network_depth)[i])(x)
        x = Dropout(dropout_value)(x) if dropout_value else x

    conv_shape = kbe.int_shape(x)[1:]

    x = Flatten()(x)
    x = Dense(latent_representation_dim, kernel_initializer=KERNEL_INITIALIZER, kernel_regularizer=l1_l2(l1, l2))(x)

    if activation == 'leakyrelu':
        x = LeakyReLU()(x)
    else:
        x = Activation(activation)(x)

    encode_model = Model(input_vol, x, name='encoder')

    if not trainable:
        encode_model.trainable = False
        for lyr in encode_model.layers:
            lyr.trainable = False

    if weights is not None:
        encode_model.set_weights(weights)

    if verbose:
        encode_model.summary(line_length=150)

    return encode_model, conv_shape


def apnet_conv_decode_model(
        conv_shape,
        network_depth,
        no_convolutions,
        conv_filter_no_init,
        conv_kernel_size,
        latent_representation_dim,
        l1,
        l2,
        dropout_value,
        use_batch_normalization,
        activation,
        verbose,
        trainable=True,
        weights=None,
) -> Model:
    """
    This model decodes a tensor into a 3D volume
    :param conv_shape: input shape of the convolution result
    :param network_depth: the number of times data is downsampled/upsampled after convolutions
    :param no_convolutions: number of convolutions per level
    :param conv_filter_no_init: beginning convolution number of filters
    :param conv_kernel_size: kernel window for convolutions
    :param latent_representation_dim: vector length in the encoded space
    :param l1: l1 regularization parameter
    :param l2: l2 regularization parameter
    :param dropout_value: value for the dropout layer
    :param use_batch_normalization: boolean of whether to use batch normalization
    :param activation: type of activation layer. e.g., 'relu'. Also supports 'leakyrelu'
    :param verbose: whether we should print out a summary of the model
    :param trainable: whether the model encoder-decoder is trainable
    :param weights: dictionary of weights to initialize the model
    :return:
    """

    encoded = Input(shape=(latent_representation_dim,), name='decoder_input')
    x = Dense(np.prod(conv_shape), kernel_initializer=KERNEL_INITIALIZER, kernel_regularizer=l1_l2(l1, l2))(encoded)
    if activation == 'leakyrelu':
        x = LeakyReLU()(x)
    else:
        x = Activation(activation)(x)
    x = Reshape(conv_shape)(x)

    for i in reversed(range(network_depth)):
        x = UpSampling3D(size=downsample_fn(network_depth)[i])(x)
        for j in range(no_convolutions):
            x = Conv3D(
                filters=filters_fn(i, conv_filter_no_init),
                kernel_size=conv_kernel_size,
                kernel_initializer=KERNEL_INITIALIZER,
                padding='same'
            )(x)

            if activation == 'leakyrelu':
                x = LeakyReLU()(x)
            else:
                x = Activation(activation)(x)

            x = BatchNormalization()(x) if use_batch_normalization else x

        x = Dropout(dropout_value)(x) if dropout_value else x

    x = Conv3D(
        filters=data_config.OUTPUT_CHANNELS,
        kernel_size=conv_kernel_size,
        kernel_initializer=KERNEL_INITIALIZER,
        activation='relu',
        name='decoder_output',
        padding='same'
    )(x)

    decode_model = Model(encoded, x, name='decoder')

    if not trainable:
        decode_model.trainable = False
        for lyr in decode_model.layers:
            lyr.trainable = False

    if weights is not None:
        decode_model.set_weights(weights)

    if verbose:
        decode_model.summary(line_length=150)

    return decode_model


def apnet_latent_parameters_model(
        risk_categories,
        latent_representation_dim,
        l1,
        l2,
        verbose,
        trainable=True,
        weights=None,
        model_fit=MODEL_FIT,
) -> Model:
    """
    This function transforms a flat tensor into 2 parameters based on a classification
    :param risk_categories: number of risk categories in which to split the data
    :param latent_representation_dim: vector length in the encoded space
    :param l1: l1 regularization parameter
    :param l2: l2 regularization parameter
    :param verbose: whether we should print out a summary of the model
    :param trainable: whether the model encoder-decoder is trainable
    :param weights: dictionary of weights to initialize the model
    :param model_fit: survival model used (choice of cox_ph, weibull, loglogistic)
    :return:
    """
    encoded = Input(shape=(latent_representation_dim,))
    # Distribution Fit by risk category
    if risk_categories == 1:  # do not split cohort by risk category
        l_loc_scl = Dense(
            units=2,
            name="l_loc_scl",
            kernel_initializer=KERNEL_INITIALIZER,
            kernel_regularizer=l1_l2(l1, l2),
            bias_regularizer=l1_l2(l1, l2),
        )(encoded)

    else:  # split by risk category
        category_weights = Dense(
            units=risk_categories,
            use_bias=False,
            activation='softmax',
            kernel_initializer='ones',
            name="categ_weights",
            kernel_regularizer=l1_l2(l1, l2),
            bias_regularizer=l1_l2(l1, l2),
        )(encoded)

        # handle location
        all_l_loc_scl = [Dense(
            units=2,
            kernel_initializer=KERNEL_INITIALIZER,
            kernel_regularizer=l1_l2(l1, l2),
            bias_regularizer=l1_l2(l1, l2),
        )(encoded) for _ in range(risk_categories)]
        all_l_loc_scl = [Lambda(lambda t: kbe.expand_dims(t, axis=-1))(x) for x in all_l_loc_scl]
        l_loc_scl_by_categ = Concatenate(axis=-1)(all_l_loc_scl)
        l_loc_scl = Dot(
            axes=-1,
            normalize=False,
            name='l_loc',
        )([l_loc_scl_by_categ, category_weights])

    if model_fit == 'loglogistic':
        l_loc_scl = Activation(loglogistic_activation)(l_loc_scl)
    elif model_fit == 'cox_ph':
        l_loc_scl = Activation(cox_ph_activation)(l_loc_scl)

    latent_param_model = Model(encoded, l_loc_scl, name='latent_parameter')

    if not trainable:
        latent_param_model.trainable = False
        for lyr in latent_param_model.layers:
            lyr.trainable = False

    if weights is not None:
        latent_param_model.set_weights(weights)

    if verbose:
        latent_param_model.summary(line_length=150)

    return latent_param_model


def get_apnet_model(
        risk_categories,
        network_depth,
        no_convolutions,
        conv_filter_no_init,
        conv_kernel_size,
        latent_representation_dim,
        dropout_value,
        use_batch_normalization,
        activation,
        gaussian_noise_std,
        l1,
        l2,
        verbose=2,
        trainable=True,
        weights=None
) -> tuple:
    """
    This function generates an uncompiled keras model. The architecture is flexible and
    can be changed depending on parameters passed into the function.
    :param risk_categories: number of risk categories in which to split the data
    :param network_depth: the number of times data is downsampled/upsampled after convolutions
    :param no_convolutions: number of convolutions per level
    :param conv_filter_no_init: beginning convolution number of filters
    :param conv_kernel_size: kernel window for convolutions
    :param latent_representation_dim: vector length in the encoded space
    :param dropout_value: value for the dropout layer
    :param use_batch_normalization: boolean of whether to use batch normalization
    :param activation: type of activation layer. e.g., 'relu'. Also supports 'leakyrelu'
    :param gaussian_noise_std: standard deviation of gaussian noise for images
    :param l1: l1 regularization parameter
    :param l2: l2 regularization parameter
    :param verbose: whether we should print out a summary of the model
    :param trainable: whether the convolutional encoder-decoder is trainable
    :param weights: dictionary of weights to initialize the model
    :return:
    """
    input_shape = (*data_config.OUTPUT_SPECS['output_resolution'], data_config.OUTPUT_CHANNELS)
    encode_model, output_conv_shape = apnet_conv_encode_model(
        input_shape,
        network_depth,
        no_convolutions,
        conv_filter_no_init,
        conv_kernel_size,
        latent_representation_dim,
        l1,
        l2,
        dropout_value,
        use_batch_normalization,
        activation,
        gaussian_noise_std,
        verbose,
        trainable,
        weights.get('encoder', None) if weights else None,
    )
    decode_model = apnet_conv_decode_model(
        output_conv_shape,
        network_depth,
        no_convolutions,
        conv_filter_no_init,
        conv_kernel_size,
        latent_representation_dim,
        l1,
        l2,
        dropout_value,
        use_batch_normalization,
        activation,
        verbose,
        trainable,
        weights.get('decoder', None) if weights else None,
    )
    latent_param_model = apnet_latent_parameters_model(
        risk_categories,
        latent_representation_dim,
        l1,
        l2,
        verbose,
        trainable,
        weights.get('latent_parameter', None) if weights else None,
    )

    input_vol = Input(shape=input_shape, name='input_volume')
    encoded = encode_model(input_vol)
    decoded = decode_model(encoded)
    l_loc_logscalesq = latent_param_model(encoded)
    template_model = Model(input_vol, [decoded, l_loc_logscalesq])

    if weights is not None:
        if 'convolutional' in weights:
            template_model.load_weights(weights['convolutional'])

    if verbose:
        template_model.summary(line_length=150)

    if MULTI_GPU > 1:
        parallel_model = multi_gpu_model(template_model, gpus=MULTI_GPU, cpu_merge=False)
    else:
        parallel_model = template_model

    return template_model, parallel_model

    # from keras.utils import plot_model
    # plot_model(
    #   complete_model,
    #   to_file='/home/dpopesc2/Desktop/model.png',
    #   show_shapes=True,
    #   rankdir='LR',
    #   show_layer_names=True
    # )


def get_apnet_ancillary_model(
        network_depth,
        activation,
        no_units,
        risk_categories,
        dropout_value,
        use_batch_normalization,
        l1,
        l2,
        verbose=0,
        trainable=True,
        weights=None
):
    input_shape = (data_config.APNET_ANCILLARY_COVARIATES_NO,)
    input_cov = Input(shape=input_shape, name='ancillary_covariates')

    x = input_cov
    for i in range(network_depth):
        x = Dense(
            units=no_units,
            activation=activation,
            kernel_regularizer=l1_l2(l1, l2),
            bias_regularizer=l1_l2(l1, l2),
        )(x)
        if use_batch_normalization:
            x = BatchNormalization(momentum=.5)(x)

    x = Dropout(dropout_value)(x) if dropout_value else x

    if network_depth:
        latent_param_model = apnet_latent_parameters_model(risk_categories, no_units, l1, l2, verbose, trainable)
    else:
        latent_param_model = apnet_latent_parameters_model(risk_categories, input_shape[0], l1, l2, verbose, trainable)
    l_loc_logscalesq = latent_param_model(x)

    template_model = Model(input_cov, l_loc_logscalesq)

    if not trainable:
        template_model.trainable = False
        for lyr in template_model.layers:
            lyr.trainable = False

    if weights is not None and weights:
        if 'ancillary' in weights:
            template_model.load_weights(weights['ancillary'])

    if verbose:
        template_model.summary(line_length=150)

    if MULTI_GPU > 1:
        parallel_model = multi_gpu_model(template_model, gpus=MULTI_GPU, cpu_merge=False)
    else:
        parallel_model = template_model

    return template_model, parallel_model


def get_apnet_ensemble_model_dense(
        ensemble_aux_params,
        ensemble_conv_params,
        ensemble_units,
        ensemble_depth,
        l1,
        l2,
        verbose=0,
        weights=None,
        trainable=True
):
    apnet_model_aux, _ = compiled_model_from_params(
        model_params=ensemble_aux_params,
        ancillary=True,
        ensemble=False,
        verbose=verbose,
        compile_model=False,
        trainable=False
    )

    ensemble_aux_weights = weights.get('ancillary', '')
    if ensemble_aux_weights:
        apnet_model_aux.trainable = False
        apnet_model_aux.load_weights(ensemble_aux_weights)
    apnet_model_aux.name = 'ancillary'

    apnet_model_conv, _ = compiled_model_from_params(
        model_params=ensemble_conv_params,
        ancillary=False,
        ensemble=False,
        verbose=verbose,
        compile_model=False,
        trainable=False
    )

    ensemble_conv_weights = weights.get('convolutional', '')
    if ensemble_conv_weights:
        apnet_model_conv.trainable = False
        apnet_model_conv.load_weights(ensemble_conv_weights)
    apnet_model_conv.name = 'convolutional'

    # Ensemble the models
    aux_model = Model(inputs=apnet_model_aux.input, outputs=apnet_model_aux.get_layer(index=-2).output)
    conv_model = Model(inputs=apnet_model_conv.input, outputs=apnet_model_conv.get_layer('encoder').get_output_at(1))

    x = Concatenate(axis=-1)([aux_model.output, conv_model.output])

    for i in range(ensemble_depth):
        x = Dense(
            ensemble_units,
            kernel_initializer=KERNEL_INITIALIZER,
            kernel_regularizer=l1_l2(l1, l2),
            bias_regularizer=l1_l2(l1, l2),
            activation='relu'
        )(x)

    output = apnet_latent_parameters_model(
        np.max([ensemble_aux_params['risk_categories'], ensemble_conv_params['risk_categories']]),
        x.get_shape().as_list()[-1],
        l1,
        l2,
        verbose
    )(x)
    # output = Lambda(lambda x: x[0] * .5 + (1-.5) * x[1])([apnet_output_aux, apnet_output_main])

    apnet_model_ens = Model(inputs=[aux_model.input, conv_model.input], outputs=output)

    if weights is not None and weights:
        if 'ensemble' in weights:
            apnet_model_ens.load_weights(weights['ensemble'])

    if not trainable:
        apnet_model_ens.trainable = False
        for lyr in apnet_model_ens.layers:
            lyr.trainable = False

    if verbose:
        apnet_model_ens.summary(line_length=150)

    if MULTI_GPU > 1:
        parallel_model = multi_gpu_model(apnet_model_ens, gpus=MULTI_GPU, cpu_merge=False)
    else:
        parallel_model = apnet_model_ens

    return apnet_model_ens, parallel_model


def get_apnet_ensemble_model_wtsum(
        ensemble_aux_params,
        ensemble_conv_params,
        ensemble_units,
        ensemble_depth,
        l1,
        l2,
        verbose=0,
        weights=None,
        trainable=True
):
    apnet_model_aux, _ = compiled_model_from_params(
        model_params=ensemble_aux_params,
        ancillary=True,
        ensemble=False,
        verbose=verbose,
        compile_model=False,
        trainable=False
    )

    ensemble_aux_weights = weights.get('ancillary', '')
    if ensemble_aux_weights:
        apnet_model_aux.trainable = False
        apnet_model_aux.load_weights(ensemble_aux_weights)
    apnet_model_aux.name = 'ancillary'

    apnet_model_conv, _ = compiled_model_from_params(
        model_params=ensemble_conv_params,
        ancillary=False,
        ensemble=False,
        verbose=verbose,
        compile_model=False,
        trainable=False
    )

    ensemble_conv_weights = weights.get('convolutional', '')
    if ensemble_conv_weights:
        apnet_model_conv.trainable = False
        apnet_model_conv.load_weights(ensemble_conv_weights)
    apnet_model_conv.name = 'convolutional'

    # Ensemble the models
    aux_model = Model(inputs=apnet_model_aux.input, outputs=apnet_model_aux.get_layer(index=-2).output)
    conv_model = Model(inputs=apnet_model_conv.input, outputs=apnet_model_conv.get_layer('encoder').get_output_at(1))

    conv_model_proj = Dense(
        aux_model.output.get_shape().as_list()[-1],
        kernel_initializer=keras.initializers.zeros(),
        bias_initializer=keras.initializers.zeros(),
        kernel_regularizer=l1_l2(l1, l2),
        bias_regularizer=l1_l2(l1, l2),
        activation='relu'
    )(conv_model.output)

    x = WeightedSum()([aux_model.output, conv_model_proj])

    output = apnet_latent_parameters_model(
        ensemble_aux_params['risk_categories'],
        x.get_shape().as_list()[-1],
        ensemble_aux_params['l1_reg'],
        ensemble_aux_params['l2_reg'],
        verbose,
        weights=apnet_model_aux.layers[-1].get_weights()
    )(x)

    apnet_model_ens = Model(inputs=[aux_model.input, conv_model.input], outputs=output)

    if weights is not None and weights:
        if 'ensemble' in weights:
            apnet_model_ens.load_weights(weights['ensemble'])

    if not trainable:
        apnet_model_ens.trainable = False
        for lyr in apnet_model_ens.layers:
            lyr.trainable = False

    if verbose:
        apnet_model_ens.summary(line_length=150)

    if MULTI_GPU > 1:
        parallel_model = multi_gpu_model(apnet_model_ens, gpus=MULTI_GPU, cpu_merge=False)
    else:
        parallel_model = apnet_model_ens

    return apnet_model_ens, parallel_model


def get_apnet_ensemble_model(
        ensemble_aux_params,
        ensemble_conv_params,
        ensemble_units,
        ensemble_depth,
        l1,
        l2,
        verbose=0,
        weights=None,
        trainable=True
):
    apnet_model_aux, _ = compiled_model_from_params(
        model_params=ensemble_aux_params,
        ancillary=True,
        ensemble=False,
        verbose=verbose,
        compile_model=False,
        trainable=False
    )

    ensemble_aux_weights = weights.get('ancillary', '')
    if ensemble_aux_weights:
        apnet_model_aux.trainable = False
        apnet_model_aux.load_weights(ensemble_aux_weights)
    apnet_model_aux.name = 'ancillary'

    apnet_model_conv, _ = compiled_model_from_params(
        model_params=ensemble_conv_params,
        ancillary=False,
        ensemble=False,
        verbose=verbose,
        compile_model=False,
        trainable=False
    )

    ensemble_conv_weights = weights.get('convolutional', '')
    if ensemble_conv_weights:
        apnet_model_conv.trainable = False
        apnet_model_conv.load_weights(ensemble_conv_weights)
    apnet_model_conv.name = 'convolutional'

    # Ensemble the models
    apnet_model_aux.layers[-1].name += '_ens1'
    apnet_model_conv.layers[-1].name += '_ens2'
    aux_model = Model(inputs=apnet_model_aux.input, outputs=apnet_model_aux.output)
    conv_model = Model(inputs=apnet_model_conv.input, outputs=apnet_model_conv.output[-1])

    output = LinearCombination(no_components=2)([aux_model.output, conv_model.output])

    apnet_model_ens = Model(inputs=[aux_model.input, conv_model.input], outputs=output)

    if weights is not None and weights:
        if 'ensemble' in weights:
            apnet_model_ens.load_weights(weights['ensemble'])

    if not trainable:
        apnet_model_ens.trainable = False
        for lyr in apnet_model_ens.layers:
            lyr.trainable = False

    if verbose:
        apnet_model_ens.summary(line_length=150)

    if MULTI_GPU > 1:
        parallel_model = multi_gpu_model(apnet_model_ens, gpus=MULTI_GPU, cpu_merge=False)
    else:
        parallel_model = apnet_model_ens

    return apnet_model_ens, parallel_model


def compiled_model_from_params(model_params, ancillary, ensemble, verbose=1, compile_model=True, trainable=True):
    """
    Parse model parameters and return compiled model. Can return either convolutional or ancillary
    or an ensemble of the 2 if paths to trained versions of them exist.
    :param model_params: dictionary of model parameters
    :param ancillary: whether to return convolutional or ancillary model
    :param ensemble: whether to return ensembled model
    :param verbose: verbosity level
    :param compile_model: whether to call compile on model
    :param trainable: whether resulting model is trainable
    :return:
    """
    # Extract parameters
    risk_categories = int(model_params.get('risk_categories', 1))
    network_depth = int(model_params.get('network_depth', 4))
    dropout_value = model_params.get('dropout_value', 0)
    use_batch_normalization = model_params.get('use_batch_normalization', False)
    activation = model_params.get('activation', 'relu')
    l1 = model_params.get('l1_reg', 0)
    l2 = model_params.get('l2_reg', 0)
    lr_init = model_params.get('initial_lr', 1e-3)
    no_convolutions = int(model_params.get('no_convolutions', 1))
    conv_filter_no_init = int(model_params.get('conv_filter_no_init', 16))
    latent_representation_dim = int(model_params.get('latent_representation_dim', 32))
    conv_kernel_size = int(model_params.get('conv_kernel_size', 3))
    data_aug_params = model_params.get('data_aug_params', None)
    no_units = int(model_params.get('no_units', 64))
    weight_scalar = model_params.get('weight_scalar', 0)
    ensemble_units = int(model_params.get('ensemble_no_units', 5))
    ensemble_depth = int(model_params.get('ensemble_depth', 0))
    ensemble_aux_params = model_params.get('ensemble_aux_params', {})
    ensemble_conv_params = model_params.get('ensemble_conv_params', {})
    model_weights = model_params.get('model_weights', {})
    optimizer = model_params.get('optimizer', 'SGD')

    if ensemble:
        apnet_m_template, apnet_m_parallel = get_apnet_ensemble_model(
            ensemble_aux_params,
            ensemble_conv_params,
            ensemble_units,
            ensemble_depth,
            l1,
            l2,
            verbose=verbose,
            weights=model_weights,
            trainable=trainable
        )

        survival_loss_wts = kbe.variable(
            np.array([1, 0, 0, weight_scalar], dtype=kbe.floatx()))  # magnitudes about 1:10:100
        loss = survival_loss_gen(survival_loss_wts)
        loss_weights = None
    else:
        if ancillary:
            apnet_m_template, apnet_m_parallel = get_apnet_ancillary_model(
                network_depth=network_depth,
                activation=activation,
                no_units=no_units,
                risk_categories=risk_categories,
                dropout_value=dropout_value,
                use_batch_normalization=use_batch_normalization,
                l1=l1,
                l2=l2,
                verbose=verbose,
                weights=model_weights,
                trainable=trainable
            )
            survival_loss_wts = kbe.variable(
                np.array([1, 0, 0, weight_scalar], dtype=kbe.floatx()))  # magnitudes about 1:10:100
            loss = survival_loss_gen(survival_loss_wts)
            loss_weights = None
        else:
            apnet_m_template, apnet_m_parallel = get_apnet_model(
                risk_categories=risk_categories,
                network_depth=network_depth,
                no_convolutions=no_convolutions,
                conv_filter_no_init=conv_filter_no_init,
                conv_kernel_size=conv_kernel_size,
                latent_representation_dim=latent_representation_dim,
                dropout_value=dropout_value,
                use_batch_normalization=use_batch_normalization,
                activation=activation,
                gaussian_noise_std=0 if data_aug_params is None else data_aug_params.get('gaussian_noise_std', 0),
                l1=l1,
                l2=l2,
                verbose=verbose,
                weights=model_weights,
                trainable=trainable
            )
            survival_loss_wts = kbe.variable(
                np.array([1, 0, 0, weight_scalar], dtype=kbe.floatx()))  # magnitudes about 1:10:100
            loss = [reconstruction_loss, survival_loss_gen(survival_loss_wts)]
            reconstruction_loss_wt = kbe.variable(model_params.get('reconstruction_loss_wt', 0), dtype=kbe.floatx())
            loss_weights = [reconstruction_loss_wt, kbe.variable(1, dtype=kbe.floatx())]

    if compile_model:
        apnet_m_parallel.compile(
            optimizer=getattr(optimizers, optimizer)(lr=lr_init),
            loss=loss,
            loss_weights=loss_weights
        )

    return apnet_m_template, apnet_m_parallel


def ensemble_model_parameters(
        aux_model_params,
        aux_model_weights,
        conv_model_params,
        conv_model_weights
):
    """
    Dummy function that returns the parameters for the ensemble
    to avoid duplication

    :param aux_model_params: auxiliary model parameters
    :param aux_model_weights: path to auxiliary model weights
    :param conv_model_params: convolutional model parameters
    :param conv_model_weights: path to convolutional model weights
    :return: dict
    """

    model_params = {
        # Ensembling part
        'epochs': 2000,
        'initial_lr': 1e-2,
        'batch_size': 32,
        'steps_per_epoch': 20,
        'ensemble_no_units': 25, # 25
        'ensemble_depth': 0, # 3
        'l1_reg': 0,
        'l2_reg': 0,
        'optimizer': 'Adam',
        'ensemble_aux_params': aux_model_params,
        'ensemble_conv_params': conv_model_params,
        'data_aug_params': {
            'rotation': False,  # degrees to rotate the images in the x-y plane
            'width_shift_pct': .9,  # shift images horizontally (fraction of total available space)
            'height_shift_pct': .9,  # shift images vertically (fraction of total available space)
            'depth_shift_pct': .9,  # shift images in the short axis plane (fraction of total available space)
            'gaussian_noise_std': 0,  # standard deviation of gaussian noise applied to input
        },
        'model_weights': {
            'ancillary': aux_model_weights,
            'convolutional': conv_model_weights,
        },
    }

    return model_params
