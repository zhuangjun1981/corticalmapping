import os
import numpy as np
import tifffile as tf
import h5py
import corticalmapping.NwbTools as nt

movie_2p_fn = '180328_M360495_04_2p_movies.hdf5'

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

input_parameters = []

for i in range(5):

    plane_n = 'plane{}'.format(i)

    offsets_path = os.path.join(plane_n, 'correction_offsets.hdf5')
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

    mean_projection = tf.imread(os.path.join(plane_n, 'corrected_mean_projection.tif'))
    max_projection = tf.imread(os.path.join(plane_n, 'corrected_max_projection.tif'))

    input_dict = {'field_name': plane_n,
                  'original_timeseries_path': '/acquisition/timeseries/2p_movie_plane' + str(i),
                  'corrected_file_path': movie_2p_fn,
                  'corrected_dataset_path': plane_n,
                  'xy_translation_offsets': offsets,
                  'mean_projection': mean_projection,
                  'max_projection': max_projection,
                  'description': '',
                  'comments': '',
                  'source': ''}

    input_parameters.append(input_dict)

nwb_fn = [f for f in os.listdir(curr_folder) if f[-4:] == '.nwb'][0]
nwb_f = nt.RecordedFile(nwb_fn)

nwb_f.add_muliple_dataset_to_motion_correction_module(input_parameters=input_parameters,
                                                      module_name='motion_correction')
nwb_f.close()



