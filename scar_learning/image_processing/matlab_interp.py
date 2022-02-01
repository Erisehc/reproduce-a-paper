"""
Library for matlab-based 3D interporation.
"""

import matlab.engine as mat
import matlab
import numpy as np


def __np_to_mat(np_arr: np.ndarray):
    # noinspection PyUnresolvedReferences
    return matlab.double(np_arr.tolist())


def matlab_interp3(old_grid, values, new_grid, method):
    """
    Wrapper around Matlab's interp3 function
    :param old_grid: old meshgrid
    :param values: values to interpolate
    :param new_grid: new meshgrid
    :param method: either 'variational_implicit' or 'spline'
    :return: interpolated values as np.ndarray
    """
    x, y, z = old_grid
    xq, yq, zq = new_grid

    eng = mat.start_matlab()

    try:
        if method == 'source':
            values_interpolated = eng.interp3(
                __np_to_mat(x), __np_to_mat(y), __np_to_mat(z),
                __np_to_mat(values),
                __np_to_mat(xq), __np_to_mat(yq), __np_to_mat(zq),
                'spline',  # interpolation method
                0  # extrapolation values
            )

        elif method == 'mask':
            #  using nearest neighbor
            values_interpolated = eng.interp3(
                __np_to_mat(x), __np_to_mat(y), __np_to_mat(z),
                __np_to_mat(values),
                __np_to_mat(xq), __np_to_mat(yq), __np_to_mat(zq),
                'nearest',  # interpolation method
                0  # extrapolation values
            )
        else:
            raise ValueError('Unrecognized interpolation method: %s' % method)
    finally:
        eng.quit()

    rv = np.asarray(values_interpolated)

    return rv
