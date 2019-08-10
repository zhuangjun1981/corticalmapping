import os
import numpy as np
import tifffile as tf

data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data_rabies_project" \
              r"\180328-M360495-deepscope\04"

identifier = '04_'
plane_num = 5
temporal_downsample_rate = 1
frame_each_file = 2000

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

fns = np.array([f for f in os.listdir(data_folder) if f[-4:] == '.tif' and identifier in f])
f_nums = [int(os.path.splitext(fn)[0].split('_')[1]) for fn in fns]
fns = fns[np.argsort(f_nums)]
print('total file number: {}'.format(len(fns)))

# print('\n'.join(fns))

save_folders = []
for i in range(plane_num):
    curr_save_folder = os.path.join(data_folder, identifier, 'plane{}'.format(i))
    if not os.path.isdir(curr_save_folder):
        os.makedirs(curr_save_folder)
    save_folders.append(curr_save_folder)

# frame_per_plane = len(fns) // plane_num
for plane_ind in range(plane_num):
    print('\nprocessing plane: {}'.format(plane_ind))
    curr_fns = fns[plane_ind::plane_num]

    total_frames_down = len(curr_fns) // temporal_downsample_rate
    curr_fns = curr_fns[: total_frames_down * temporal_downsample_rate].reshape((total_frames_down, temporal_downsample_rate))

    # print curr_fns

    print('current file ind: 000')
    curr_file_ind = 0
    curr_frame_ind = 0
    curr_mov = []

    for fgs in curr_fns:

        curr_frame = np.mean([tf.imread(os.path.join(data_folder, fn)) for fn in fgs], axis=0).astype(np.int16)
        curr_frame = curr_frame.transpose()[::-1, ::-1]

        if curr_frame_ind < frame_each_file:
            curr_mov.append(curr_frame)
            curr_frame_ind = curr_frame_ind + 1
        else:
            curr_mov = np.array(curr_mov, dtype=np.int16)
            save_name = 'plane{}_{:03d}.tif'.format(plane_ind, curr_file_ind)
            tf.imsave(os.path.join(save_folders[plane_ind], save_name), curr_mov)
            curr_file_ind += 1
            curr_frame_ind = 1
            curr_mov = [curr_frame]
            print('current file ind: {:03d}'.format(curr_file_ind))

    curr_mov = np.array(curr_mov, dtype=np.int16)
    save_name = 'plane{}_{:03d}.tif'.format(plane_ind, curr_file_ind)
    tf.imsave(os.path.join(save_folders[plane_ind], save_name), curr_mov)

