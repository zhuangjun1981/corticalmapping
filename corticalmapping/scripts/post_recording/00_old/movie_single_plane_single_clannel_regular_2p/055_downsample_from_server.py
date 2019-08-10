import os
import numpy as np
import tifffile as tf

import os
import numpy as np
import tifffile as tf
import corticalmapping.core.ImageAnalysis as ia

data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data_rabies_project" \
              r"\180404-M360495-2p\2p_movie\reorged"
xy_downsample_rate = 2
t_downsample_rate = 10

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

corr_folder = os.path.join(data_folder, 'corrected')

f_ns = [f for f in os.listdir(corr_folder) if f[-14:] == '_corrected.tif']
f_ns.sort()
print('\n'.join(f_ns))

mov_d = []

for f_n in f_ns:
    print('processing {} ...'.format(f_n))
    curr_mov = tf.imread(os.path.join(corr_folder, f_n))
    curr_mov_d = ia.rigid_transform_cv2(img=curr_mov, zoom=(1. / xy_downsample_rate))
    curr_mov_d = ia.z_downsample(curr_mov_d, downSampleRate=t_downsample_rate)
    mov_d.append(curr_mov_d)

mov_d = np.concatenate(mov_d, axis=0).astype(np.int16)
tf.imsave('2p_movie_downsampled.tif', mov_d)