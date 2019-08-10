import os
import h5py
import numpy as np
import tifffile as tf
from toolbox.misc import BinarySlicer
import corticalmapping.core.FileTools as ft

save_fn = '180404_M360495_110_2p_movies.hdf5'
data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data_rabies_project\180404-M360495-2p\2p_movie" \
              r"\reorged\corrected"
identifier = '_00110_'
frame_num_tot = 16000
resolution = [1024, 1024]  # rows, columns of each frame

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

data_shape = (frame_num_tot, resolution[0], resolution[1])

curr_flist = [f for f in os.listdir(data_folder) if identifier in f and f[-14:] == '_corrected.tif']
curr_flist.sort()
print('\n'.join(curr_flist))

print ('\nWriting file: ' + save_fn)
save_f = h5py.File(save_fn)
save_dset = save_f.create_dataset('2p_movie', data_shape, dtype=np.int16, compression='lzf')

start_frame = 0
for curr_f in curr_flist:
    print curr_f
    curr_mov = tf.imread(os.path.join(data_folder, curr_f))
    end_frame = start_frame + curr_mov.shape[0]
    save_dset[start_frame : end_frame, :, :] = curr_mov
    start_frame = end_frame

save_dset.attrs['conversion'] = 1.
save_dset.attrs['resolution'] = 1.
save_dset.attrs['unit'] = 'arbiturary_unit'

save_f.close()