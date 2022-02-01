# A library containing mask-related image processing helper functions

import numpy
import cv2
from scar_learning.image_processing import constants_mri_image_proc as const


def make_border_square(border: numpy.ndarray) -> numpy.ndarray:
    """
    Take a rectangular border and make it square
    :param border: numpy array with 4 elements, rmin, rmax, cmin, cmax
    :return: extend the border to make it a square with side equal to the largest side of the rectangle
    """
    height = border[1] - border[0]
    width = border[3] - border[2]
    delta = height - width  # height - width

    if delta != 0:
        ref_idx = 2 if delta > 0 else 0
        delta_abs = abs(delta)
        border[ref_idx] = border[ref_idx] - delta_abs // 2
        border[ref_idx + 1] = border[ref_idx + 1] + (delta_abs - delta_abs // 2)

    return border


def get_mask_borders(mask: numpy.ndarray, force_square=False) -> tuple:
    """
    Given a binary array mask extract borders as (x_min, x_max, y_min, y_max)
    :param mask: binary array mask
    :param force_square: whether to return a square
    :return: a list of coordinates for the box
    """

    [rows, cols] = numpy.where(mask)
    row_min = numpy.min(rows)
    row_max = numpy.max(rows)
    col_min = numpy.min(cols)
    col_max = numpy.max(cols)

    if force_square:
        row_min, row_max, col_min, col_max = make_border_square(numpy.asarray([row_min, row_max, col_min, col_max]))

    return row_min, row_max, col_min, col_max  # top, bottom, left, right


