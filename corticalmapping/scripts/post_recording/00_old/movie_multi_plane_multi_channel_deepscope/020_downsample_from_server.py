import os
import numpy as np
import tifffile as tf

import os
import numpy as np
import tifffile as tf
import corticalmapping.core.ImageAnalysis as ia

data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data_rabies_project" \
              r"\180309-M360495-deepscope\03\03_"
xy_downsample_rate = 2
t_downsample_rate = 10

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

plane_ns = [f for f in os.listdir(data_folder) if os.path.isdir(f) and f[:5] == 'plane']
plane_ns.sort()
print('planes:')
print('\n'.join(plane_ns))

for plane_n in plane_ns:
    print('\nprocessing plane: {}'.format(plane_n))
    plane_folder = os.path.join(data_folder, plane_n, 'green', 'corrected')

    f_ns = [f for f in os.listdir(plane_folder) if f[-14:] == '_corrected.tif']
    f_ns.sort()
    print('\n'.join(f_ns))

    mov_d = []

    for f_n in f_ns:
        print('processing {} ...'.format(f_n))
        curr_mov = tf.imread(os.path.join(plane_folder, f_n))
        curr_mov_d = ia.rigid_transform_cv2(img=curr_mov, zoom=(1. / xy_downsample_rate))
        curr_mov_d = ia.z_downsample(curr_mov_d, downSampleRate=t_downsample_rate)
        mov_d.append(curr_mov_d)

    mov_d = np.concatenate(mov_d, axis=0)
    save_n = os.path.split(data_folder)[1] + '_' + plane_n + '_downsampled.tif'
    tf.imsave(os.path.join(plane_n, save_n), mov_d)