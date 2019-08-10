import sys
sys.path.extend([r"E:\data\github_packages\CaImAn"])

import caiman as cm
import numpy as np
import os
from caiman.motion_correction import MotionCorrect, tile_and_correct, motion_correction_piecewise
import tifffile as tf
import h5py
import warnings

base_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data_rabies_project" \
              r"\180427-M386507-2p\FOV1_zstack\FOV1_zstack"

identifier = 'FOV1_zstack'

n_processes = 3


def correct_single_movie(data_folder, identifier, dview):

    #=======================================setup parameters==============================================
    # number of iterations for rigid motion correction
    niter_rig = 5

    # maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
    max_shifts = (30, 30)

    # for parallelization split the movies in  num_splits chuncks across time
    # if none all the splits are processed and the movie is saved
    splits_rig = 56

    # intervals at which patches are laid out for motion correction
    # num_splits_to_process_rig = None

    # create a new patch every x pixels for pw-rigid correction
    strides = (48, 48)

    # overlap between pathes (size of patch strides+overlaps)
    overlaps = (24, 24)

    # for parallelization split the movies in  num_splits chuncks across time
    splits_els = 56

    # num_splits_to_process_els = [28, None]

    # upsample factor to avoid smearing when merging patches
    upsample_factor_grid = 4

    # maximum deviation allowed for patch with respect to rigid shifts
    max_deviation_rigid = 3

    # if True, apply shifts fast way (but smoothing results) by using opencv
    shifts_opencv = True

    # if True, make the SAVED movie and template mostly nonnegative by removing min_mov from movie
    nonneg_movie = False
    # =======================================setup parameters==============================================


    fname = [f for f in os.listdir(data_folder) if f[-4:] == '.tif' and identifier in f]

    if len(fname) == 0:
        print('\ndid not find movie file in directory: {}.'.format(data_folder))
        print('Do nothing.')
        return
    elif len(fname) > 1:
        fname.sort()
        print('\n')
        print('\n'.join(fname))
        warnings.warn('more than one movie file in directory: {}. skip ...'.format(data_folder))
        return
    else:
        fname = fname[0]
        print('\ncorrecting {} in directory {}.'.format(fname, data_folder))

        # m_orig = cm.load(os.path.join(data_folder, fname))
        # offset_mov = np.min(m_orig)  # if the data has very negative values compute an offset value

        offset_mov = 0.

        # create a motion correction object# creat
        mc = MotionCorrect(os.path.join(data_folder, fname), offset_mov,
                           dview=dview, max_shifts=max_shifts, niter_rig=niter_rig,
                           splits_rig=splits_rig, strides=strides, overlaps=overlaps,
                           splits_els=splits_els, upsample_factor_grid=upsample_factor_grid,
                           max_deviation_rigid=max_deviation_rigid,
                           shifts_opencv=shifts_opencv, nonneg_movie=nonneg_movie)

        mc.motion_correct_rigid(save_movie=True)
        # load motion corrected movie
        m_rig = cm.load(mc.fname_tot_rig)
        m_rig = m_rig.astype(np.int16)
        save_name = os.path.splitext(fname)[0] + '_corrected.tif'
        tf.imsave(os.path.join(data_folder, save_name), m_rig)
        tf.imsave(os.path.join(data_folder, 'corrected_mean_projection.tif'),
                  np.mean(m_rig, axis=0).astype(np.float32))
        tf.imsave(os.path.join(data_folder, 'corrected_max_projection.tif'),
                  np.max(m_rig, axis=0).astype(np.float32))

        offset_f = h5py.File(os.path.join(data_folder, 'correction_offsets.hdf5'))
        offsets = mc.shifts_rig
        offsets = np.array([np.array(o) for o in offsets]).astype(np.float32)
        offset_dset = offset_f.create_dataset(name='file_0000', data=offsets)
        offset_dset.attrs['format'] = 'height, width'
        offset_dset.attrs['path'] = os.path.join(data_folder, fname)

        os.remove(mc.fname_tot_rig[0])


if __name__ == '__main__':
    subfolder_ns = [f for f in os.listdir(base_folder) if identifier in f]
    subfolder_ns.sort()
    print('\n'.join(subfolder_ns))

    # %% start the cluster (if a cluster already exists terminate it)
    if 'dview' in locals():
        dview.terminate()

    c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                     n_processes=n_processes,
                                                     single_thread=False)

    for subfolder_n in subfolder_ns:

        correct_single_movie(data_folder=os.path.join(base_folder, subfolder_n),
                                 identifier=identifier,
                                 dview=dview)