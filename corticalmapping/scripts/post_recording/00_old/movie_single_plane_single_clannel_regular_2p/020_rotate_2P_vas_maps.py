import os
import numpy as np
import corticalmapping.core.FileTools as ft
import corticalmapping.core.ImageAnalysis as ia
import tifffile as tf



data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data_rabies_project\180404-M360495-2p\vasmap_2p"

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

data_folder = os.path.join(curr_folder, data_folder)

file_list = [f for f in os.listdir(data_folder) if f[-4:] == '.tif']
file_list.sort()
print '\n'.join(file_list)

file_paths = [os.path.join(data_folder, f) for f in file_list]

for file_path in file_paths:
    fn, ext = os.path.splitext(os.path.split(file_path)[1])
    save_path = os.path.join(data_folder, fn + '_rotated.tif')
    print save_path

    curr_mov = tf.imread(file_path)
    curr_mov = curr_mov.transpose((0, 2, 1))[:, ::-1, :]
    tf.imsave(save_path, curr_mov)


# correction = mc.align_multiple_files_iterate(file_paths, output_folder=data_folder, is_output_mov=True, iteration=10,
#                                              max_offset=(10., 10.), align_func=mc.phase_correlation, fill_value=0.,
#                                              verbose=True, offset_file_name=offset_file_name,
#                                              mean_projection_file_name=mean_projection_file_name)


# print correction