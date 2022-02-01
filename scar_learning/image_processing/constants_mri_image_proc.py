# Constants used in cardiac image processing.

import numpy as np

MAX_BRIGHTNESS = np.uint8(255)
MIN_BRIGHTNESS = np.uint8(0)

# BGR is the default in opencv, which is used extensively
RGB_INDEX_MAP = {
    'B': 0,
    'G': 1,
    'R': 2,
}

#  The following constants are for 'contours' mode only
CHANNEL_SCAR = 'R'
CHANNEL_ENDOCARDIUM = 'R'
CHANNEL_EPICARDIUM = 'B'
CHANNEL_FREE = 'G'

#  The following constants are for 'patches' mode only
COLOR_GRAY_ZONE = np.array([MIN_BRIGHTNESS, MAX_BRIGHTNESS, MAX_BRIGHTNESS])

EPSILON = 1e-10

SORT_ORDER_APEX_TO_BASE = 'apex to base'
SORT_ORDER_BASE_TO_APEX = 'base to apex'

# GT-labels
GT_LABELS = {
    'nonroi': 0,
    'lvbp': 2 ** 0,
    'rvbp': 2 ** 1,
    'lvmyo': 2 ** 2,
    'rvmyo': 2 ** 3,
}

GT_INTENSITIES = {
    'nonroi': 0,
    'lvbp': np.uint8(np.clip((np.log2(GT_LABELS['lvbp']) + 1) * 64, 0, 255)),
    'rvbp': np.uint8(np.clip((np.log2(GT_LABELS['rvbp']) + 1) * 64, 0, 255)),
    'lvmyo': np.uint8(np.clip((np.log2(GT_LABELS['lvmyo']) + 1) * 64, 0, 255)),
    'rvmyo': np.uint8(np.clip((np.log2(GT_LABELS['rvmyo']) + 1) * 64, 0, 255)),
}