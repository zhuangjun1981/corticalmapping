import os
import numpy as np
import tifffile as tf
import h5py
import corticalmapping.NwbTools as nt

corrected_file_path = '180404_M360495_110_2p_movies.hdf5'

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

input_parameters = []

offsets_path = 'correction_offsets.hdf5'
offsets_f = h5py.File(offsets_path)
offsets_keys = offsets_f.keys()
if 'path_list' in offsets_keys:
    offsets_keys.remove('path_list')

offsets_keys.sort()
offsets = []
for offsets_key in offsets_keys:
    offsets.append(offsets_f[offsets_key].value)
offsets = np.concatenate(offsets, axis=0)
offsets = np.array(zip(offsets[:, 1], offsets[:, 0]))
offsets_f.close()

mean_projection = tf.imread('corrected_mean_projection.tif')
max_projection = tf.imread('corrected_max_projection.tif')

nwb_fn = [f for f in os.listdir(curr_folder) if f[-4:] == '.nwb'][0]
nwb_f = nt.RecordedFile(nwb_fn)

nwb_f.add_motion_correction_module(module_name='motion_correction',
                                   original_timeseries_path='/acquisition/timeseries/2p_movie',
                                   corrected_file_path=corrected_file_path, corrected_dataset_path='2p_movie',
                                   xy_translation_offsets=offsets, mean_projection=mean_projection,
                                   max_projection=max_projection)
nwb_f.close()



