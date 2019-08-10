import os
import h5py
import numpy as np
import skimage.external.tifffile as tf

file_prefix = '180309_M360495_03'
data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data_rabies_project" \
              r"\180309-M360495-deepscope\03\03_"

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

plane_fns = [f for f in os.listdir(data_folder) if f[:5] == 'plane']
plane_fns.sort()
print('\n'.join(plane_fns))

data_f = h5py.File(file_prefix + '_2p_movies.hdf5')

for plane_fn in plane_fns:
    print('\nprocessing {} ...'.format(plane_fn))
    plane_folder = os.path.join(data_folder, plane_fn, 'green', 'corrected')
    mov_fns = [f for f in os.listdir(plane_folder) if f[-14:] == '_corrected.tif']
    mov_fns.sort()
    print('\n'.join(mov_fns))

    frame_num_tot = 0
    x = None
    y = None
    z = 0
    for mov_fn in mov_fns:
        print('reading {} ...'.format(mov_fn))
        curr_z, curr_y, curr_x = tf.imread(os.path.join(plane_folder, mov_fn)).shape

        if y is None:
            y = curr_y
        else:
            if y != curr_y:
                raise ValueError('y dimension ({}) of file "{}" does not agree with previous file(s) ({}).'
                                 .format(curr_y, mov_fn, y))

        if x is None:
            x = curr_x
        else:
            if x != curr_x:
                raise ValueError('x dimension ({}) of file "{}" does not agree with previous file(s) ({}).'
                                 .format(curr_x, mov_fn, x))

        z = z + curr_z

    dset = data_f.create_dataset(plane_fn, (z, y, x), dtype=np.int16, compression='lzf')

    start_frame = 0
    end_frame = 0
    for mov_fn in mov_fns:
        print('reading {} ...'.format(mov_fn))
        curr_mov = tf.imread(os.path.join(plane_folder, mov_fn))
        end_frame = start_frame + curr_mov.shape[0]
        dset[start_frame : end_frame] = curr_mov
        start_frame = end_frame

    dset.attrs['conversion'] = 1.
    dset.attrs['resolution'] = 1.
    dset.attrs['unit'] = 'arbiturary_unit'

data_f.close()