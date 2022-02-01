# A library containing cropping-related image processing helper functions

import numpy
from scar_learning.image_processing import mask


def crop_image(img: numpy.ndarray, border: numpy.array, buffer: int = 0) -> numpy.ndarray:
    """
    Crop image based on border with a buffer
    :param img: Image to crop
    :param border: an array of row/column min/max
    :param buffer: expand the bus by buffer in every direction
    :return: cropped image
    """
    pic_copy = img.copy()

    if pic_copy.ndim == 2:
        pic_copy = numpy.expand_dims(pic_copy, axis=-1)

    [min_r, max_r, min_c, max_c] = border
    new_min_r = min_r - buffer
    new_max_r = max_r + buffer
    new_min_c = min_c - buffer
    new_max_c = max_c + buffer

    if new_min_r < 0 or new_min_c < 0 or new_max_r > img.shape[0] or new_max_c > img.shape[1]:
        raise ValueError('Buffered image is larger than original image, which may lead to unexpected results.')

    cropped = pic_copy[new_min_r:new_max_r, new_min_c:new_max_c, :]

    return cropped


def find_bounding_box(images: list, force_square: bool = False) -> numpy.ndarray:
    """
    Given a list of ndarray images, find the bounding box encompassing all regions of interest (myocardiums).
    :param images: list of ndarrays representing images
    :param force_square: whether to return a square
    :return: a list of cropped images
    """

    # Initialize biggest border variable:
    super_border = [numpy.inf, -numpy.inf, numpy.inf, -numpy.inf]

    for img in images:
        if numpy.isclose(img.max(), 0):  # no contour in this image
            continue

        this_border = mask.get_mask_borders(img)  # get borders

        if this_border[0] < super_border[0]:  # handle minimum row
            super_border[0] = this_border[0]

        if this_border[1] > super_border[1]:  # handle maximum row
            super_border[1] = this_border[1]

        if this_border[2] < super_border[2]:  # handle minimum column
            super_border[2] = this_border[2]

        if this_border[3] > super_border[3]:  # handle maximum column
            super_border[3] = this_border[3]

    if not numpy.all(numpy.isfinite(super_border)):
        raise (Exception('Edge of bounding box out of range'))

    super_border = numpy.array(super_border, dtype=int)

    if force_square:
        super_border = mask.make_border_square(super_border)

    return super_border
