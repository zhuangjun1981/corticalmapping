import os
import numpy as np
import tifffile as tf
import corticalmapping.core.ImageAnalysis as ia
import h5py


file_range = None # [0, 37] # None

data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data_rabies_project" \
              r"\180404-M360495-2p\2p_movie\reorged\corrected"
base_name = '180404_M360495_110'
t_downsample_rate = 10

f_ns = [f for f in os.listdir(data_folder) if f[-14:] == '_corrected.tif']
f_ns.sort()
if file_range is not None:
    f_ns = f_ns[file_range[0] : file_range[1]]
print('\n'.join(f_ns))

mov_join = []
for f_n in f_ns:
    curr_mov = tf.imread(os.path.join(data_folder, f_n))

    if curr_mov.shape[0] % t_downsample_rate !=0:
        raise ValueError('the frame number of {} ({}) is not divisible by t_downsample_rate ({}).'
                         .format(f_n, curr_mov.shape[0], t_downsample_rate))

    curr_mov_d = ia.z_downsample(curr_mov, downSampleRate=t_downsample_rate)
    mov_join.append(curr_mov_d)

mov_join = np.concatenate(mov_join, axis=0)
add_to_mov = 10 - np.amin(mov_join)

save_name = '{}_d1_{}_d2_{}_d3_1_order_C_frames_{}_.mmap'\
    .format(base_name, mov_join.shape[2], mov_join.shape[1], mov_join.shape[0])

mov_join = mov_join.reshape((mov_join.shape[0], mov_join.shape[1] * mov_join.shape[2]), order='F').transpose()

mov_join_mmap = np.memmap(os.path.join(data_folder, save_name), shape=mov_join.shape, order='C', dtype=np.float32,
                             mode='w+')
mov_join_mmap[:] = mov_join + add_to_mov
mov_join_mmap.flush()
del mov_join_mmap

save_file = h5py.File(os.path.join(data_folder, 'caiman_segmentation_results.hdf5'))
save_file['bias_added_to_movie'] = add_to_mov
save_file.close()