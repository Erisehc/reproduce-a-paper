import glob
import shutil
import os
import time


def copy_directory(src, dest):
    try:
        shutil.copytree(src, dest)
    # Directories are the same
    except shutil.Error as e:
        print('Directory not copied. Error: %s' % e)
    # Any error saying that the directory doesn't exist
    except OSError as e:
        print('Directory not copied. Error: %s' % e)


src_root = ''  # ''/home/dpopesc2/PycharmProjects/DNN/scar_learning/data/data_original_DCM/data_new/'
dst_root = ''  # ''/home/dpopesc2/PycharmProjects/DNN/scar_learning/data/data_original'

fname_src = 'Transmatrix'
fname_dst = 'TransformMatrix'

for i in range(1, 110):
    patient_id = 'P%.3d' % i
    print('Processing patient id: %s' % patient_id)

    source_dirs = glob.glob(src_root + '/%s_*' % patient_id)
    if source_dirs:
        source_dir = source_dirs[0]
    else:
        print('Patient folder does not exist: %s' % patient_id)
        continue
    dest_dir = glob.glob(dst_root+ '/%s_*' % patient_id)[0]

    dcm_src_dir = os.path.join(source_dir, fname_src)
    dcm_dst_dir = os.path.join(dest_dir, fname_dst)

    if os.path.isdir(dcm_src_dir):  # Check that source directory exists
        if os.path.isdir(dcm_dst_dir):  # Check that destination directory exists and delete if it does
            shutil.rmtree(dcm_dst_dir)
        copy_directory(dcm_src_dir, dcm_dst_dir)
    else:
        raise(Exception('Source directory does not exists: %s' % dcm_src_dir))

    time.sleep(.5)

print('Done!')