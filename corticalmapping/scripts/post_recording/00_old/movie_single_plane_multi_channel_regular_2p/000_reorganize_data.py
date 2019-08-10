import os
import numpy as np
import tifffile as tf


data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data_rabies_project\180215-M371139-2p"
file_identifier = 'Posterior_FOV_00001'
ch_ns = ['green', 'red']

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

f_ns = [f for f in os.listdir(data_folder) if file_identifier in f and f[-4:] == '.tif']
f_ns.sort()
print('\n'.join(f_ns))

save_folders = []
for ch_n in ch_ns:
    curr_save_folder = os.path.join(data_folder, file_identifier, ch_n)
    if not os.path.isdir(curr_save_folder):
        os.makedirs(curr_save_folder)
    save_folders.append(curr_save_folder)

for f_n in f_ns:
    print('procesing: {} ...'.format(f_n))
    curr_mov = tf.imread(os.path.join(data_folder, f_n))
    for ch_num, ch_n in enumerate(ch_ns):
        curr_mov_ch = curr_mov[ch_num::len(ch_ns)].transpose((0, 2, 1))[:, ::-1, :]
        curr_save_name = os.path.splitext(f_n)[0] + '_' + ch_n + '.tif'
        tf.imsave(os.path.join(save_folders[ch_num], curr_save_name), curr_mov_ch)
