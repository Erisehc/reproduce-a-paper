# These patients do not have mat segmentation, use tiff instead
from scar_learning.config_data import DATA_PATH_ORIGINAL, SCAR_LEARNING_SRC_PATH

data_directory = DATA_PATH_ORIGINAL
code_directory = SCAR_LEARNING_SRC_PATH

dicom_window_override = {
    'P003': {
        'center': 50,
        'width': 150
    },
    'P083': {
        'center': 2710,
        'width': 1375,
    },
    'P195': {
        'center': 2000,
        'width': 4800
    },
    'P210': {
        'center': 600,
        'width': 1200
    },
}
