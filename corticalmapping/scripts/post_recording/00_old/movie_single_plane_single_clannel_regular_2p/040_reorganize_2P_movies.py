import os
import numpy as np
import corticalmapping.core.FileTools as ft
import corticalmapping.core.ImageAnalysis as ia
import tifffile as tf

data_folder = r'\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data_rabies_project\180404-M360495-2p\2p_movie'
frames_per_file = 500
temporal_downsample_rate = 2

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

file_list = [f for f in os.listdir(data_folder) if f[-4:] == '.tif']
file_list.sort()
print '\n'.join(file_list)

file_paths = [os.path.join(data_folder, f) for f in file_list]

file_id_save = 0
total_mov = None
base_name = '_'.join(file_list[0].split('_')[:-1])
save_folder = os.path.join(data_folder, 'reorged')
if not os.path.isdir(save_folder):
    os.makedirs(save_folder)

for file_path in file_paths:
    print('\nprocessing {} ...'.format(os.path.split(file_path)[1]))

    curr_mov = tf.imread(file_path)
    curr_mov = curr_mov.transpose((0, 2, 1))[:, ::-1, :]

    if temporal_downsample_rate != 1:
        curr_mov = ia.z_downsample(curr_mov, downSampleRate=temporal_downsample_rate)

    if total_mov is None:
        total_mov = curr_mov
    else:
        total_mov = np.concatenate((total_mov, curr_mov), axis=0)

    while (total_mov is not None) and (total_mov.shape[0] >= frames_per_file):

        num_file_to_save = total_mov.shape[0] // frames_per_file

        for save_file_id in range(num_file_to_save):
            save_chunk = total_mov[save_file_id * frames_per_file : (save_file_id + 1) * frames_per_file]
            save_path = os.path.join(save_folder, '{}_{:05d}_reorged.tif'.format(base_name, file_id_save))
            print('saving {} ...'.format(os.path.split(save_path)[1]))
            tf.imsave(save_path, save_chunk)
            file_id_save = file_id_save + 1

        if total_mov.shape[0] % frames_per_file == 0:
            total_mov = None
        else:
            frame_num_left = total_mov.shape[0] % frames_per_file
            total_mov = total_mov[-frame_num_left:]

if total_mov is not None:
    save_path = os.path.join(save_folder, '{}_{:05d}_reorged.tif'.format(base_name, file_id_save))
    print('saving {} ...'.format(os.path.split(save_path)[1]))
    tf.imsave(save_path, total_mov)