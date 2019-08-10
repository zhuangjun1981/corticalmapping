import os
import tifffile as tf
import numpy as np

file_identifier = 'cell1_zoom4'
data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data_rabies_project\180419-388189-2p"
frame_per_step = 500

save_folder = os.path.join(data_folder, file_identifier)
if not os.path.isdir(save_folder):
    os.makedirs(save_folder)

fns = [f for f in os.listdir(data_folder) if file_identifier in f and f[-4:] == '.tif']
fns.sort()
print('\n'.join(fns))

curr_step = 0

print('\n')
for fn in fns:
    print(fn)
    curr_mov = tf.imread(os.path.join(data_folder, fn))

    # reorient movie
    curr_mov = curr_mov.transpose((0, 2, 1))[:, ::-1, :]

    steps = curr_mov.shape[0] / frame_per_step

    for step in range(steps):
        # print(curr_step)
        curr_step_mov = curr_mov[step * frame_per_step : (step + 1) * frame_per_step]
        curr_fn = '{}_step_{:03d}'.format(file_identifier, curr_step)
        curr_save_folder = os.path.join(save_folder, curr_fn)
        if not os.path.isdir(curr_save_folder):
            os.makedirs(curr_save_folder)
        tf.imsave(os.path.join(curr_save_folder, curr_fn + '.tif'), curr_step_mov)
        curr_step += 1