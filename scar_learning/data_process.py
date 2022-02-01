"""
This is the script which parses through the raw DICOM files and their segmentations and performs
any processing steps to get the data in a consistent format. Steps here can include: resizing, adjusting spatial
resolution, making the number of slices consistent, applying rotations, etc.
"""

import csv
import glob
import heart_volume.heart_classes as hc
import heart_volume.heart_utils as hu
import os
import os.path as path
import pickle
import scar_learning.config_data as data_config
import scar_learning.image_processing.config as config
import scar_learning.image_processing.config as img_proc_config

# The following import require the segmentation project described at https://arxiv.org/abs/2010.11081
import segmentation.predict as avaeseg_predict


def data_processing_hv_dict(
    statistics_override: tuple = (),
    cohorts: tuple = (data_config.APNET_TRAIN_COHORT, data_config.APNET_TEST_COHORT),
    write: bool = False,
):
    """
    Perform all the data processing and store data as an hv_dict object.
    :param statistics_override: optionally run a subset of patients
    :param cohorts: optionally run a subset of patients
    :param write: whether to write patient to file
    :return:
    """

    # Read all patients who are relevant, create file
    statistics_file = data_config.LABEL_FILE_NAME
    patient_ids = {}
    with open(statistics_file, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.DictReader(csvfile)

        # extracting each data row one by one
        for row in csvreader:
            patient_ids[row['patient_uniq_id']] = row['cohort']

    # Only run patients provided in the override
    if len(statistics_override):
        patient_ids = {k: v for (k, v) in patient_ids.items() if k in statistics_override}

    # Filter patients by cohort
    patient_ids = {k: v for (k, v) in patient_ids.items() if v in cohorts}

    for (patient_id, cohort) in patient_ids.items():
        try:
            print('Now processing %s...' % patient_id)
            s_files = glob.glob(path.join(config.data_directory, '%s*' % patient_id, 'DICOM', '*'))

            hvdict = hu.dcm_to_hvdict(
                patient_id,
                s_files,
                patient_name='%s_%s' % (cohort, patient_id),
                visualization_parameters_override=img_proc_config.dicom_window_override.get(patient_id, None))

            hv_obj = hc.HeartVolume(**hvdict)
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            hv_obj = avaeseg_predict.segment_hv_object(hv_obj)

            # from utils.image_slices_viewer import launch_3d_slice_viewer
            # launch_3d_slice_viewer((hv_obj.pixel_array, hv_obj.segmentation))

            # Adjust meshgrid
            mesh_grid_old = hv_obj.meshgrid
            specs = data_config.OUTPUT_SPECS
            mesh_grid_new = hu.modify_meshgrid(
                mesh_grid_old,
                specs['output_resolution'],
                specs['output_spatial_resolution']
            )
            hv_obj.meshgrid = mesh_grid_new

            # from utils.image_slices_viewer import launch_3d_slice_viewer
            # launch_3d_slice_viewer((hv_obj.pixel_array, hv_obj.segmentation))

            hvdict = hv_obj.to_stored_dict()

            if write:
                file_name = path.join(data_config.DATA_FILE_PATH, patient_id + '.pickle')
                with open(file_name, 'wb') as pickle_file:
                    pickle.dump(hvdict, pickle_file)

        except Exception as e:
            print('Something went wrong with patient %s' % patient_id)
            print(e)
            continue


if __name__ == '__main__':
    data_processing_hv_dict(
        cohorts=(data_config.APNET_TRAIN_COHORT, data_config.APNET_TEST_COHORT),
        write=False
    )
