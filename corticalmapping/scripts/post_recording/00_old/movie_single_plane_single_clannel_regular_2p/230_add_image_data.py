import os
import h5py
import corticalmapping.NwbTools as nt

dset_n = '2p_movie'
temporal_downsample_rate = 2
pixel_size = 0.00000035  # meter, 0.2 micron, deepscope 8K Hz scanner, zoom 2, 1024 x 1024

description = '2-photon imaging data'

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

nwb_fn = [f for f in os.listdir(curr_folder) if f[-4:] == '.nwb'][0]
nwb_f = nt.RecordedFile(nwb_fn)
mov_ts = nwb_f.file_pointer['/acquisition/timeseries/digital_vsync_2p_rise/timestamps'].value
print('\ntotal 2p timestamps count: {}'.format(len(mov_ts)))

mov_ts_d = mov_ts[::temporal_downsample_rate]
print('downsampled 2p timestamps count: {}'.format(len(mov_ts_d)))

mov_fn = os.path.splitext(nwb_fn)[0] + '_2p_movies.hdf5'
mov_f = h5py.File(mov_fn, 'r')
mov_dset = mov_f[dset_n]
print('downsampled 2p movie frame num: {}'.format(mov_dset.shape[0]))

mov_ts_d = mov_ts_d[0: mov_dset.shape[0]]

# if len(mov_ts) == mov_dset.shape[0]:
#     pass
# elif len(mov_ts) == mov_dset.shape[0] + 1:
#     mov_ts = mov_ts[0: -1]
# else:
#     raise ValueError('the number of timestamps of {} movie ({}) does not equal (or is not greater by one) '
#                     'the number of frames in the movie ({})'.format(mov_dn, len(mov_ts), curr_dset.shape[0]))

nwb_f.add_acquired_image_series_as_remote_link('2p_movie', image_file_path=mov_fn, dataset_path=dset_n,
                                               timestamps=mov_ts_d, description=description, comments='',
                                               data_format='zyx', pixel_size=[pixel_size, pixel_size],
                                               pixel_size_unit='meter')

mov_f.close()
nwb_f.close()