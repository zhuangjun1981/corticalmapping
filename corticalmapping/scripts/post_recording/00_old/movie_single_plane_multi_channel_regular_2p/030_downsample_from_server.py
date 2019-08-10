import os
import numpy as np
import tifffile as tf
import corticalmapping.core.ImageAnalysis as ia

data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data_rabies_project" \
              r"\180104-M361012-2p\FOV1_injection_site_00001"
chn = 'red'
xy_downsample_rate = 2
t_downsample_rate = 30

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

f_ns = [f for f in os.listdir(os.path.join(data_folder, chn, 'corrected')) if f[-14:] == '_corrected.tif']
f_ns.sort()
print('\n'.join(f_ns))

mov_d = []

for f_n in f_ns:
    print('processing {} ...'.format(f_n))
    curr_mov = tf.imread(os.path.join(data_folder, chn, 'corrected', f_n))
    curr_mov_d = ia.rigid_transform_cv2(img=curr_mov, zoom=(1. / xy_downsample_rate))
    curr_mov_d = ia.z_downsample(curr_mov_d, downSampleRate=t_downsample_rate)
    mov_d.append(curr_mov_d)

mov_d = np.concatenate(mov_d, axis=0)
save_n = os.path.split(data_folder)[1] + '_' + chn + '_downsampled.tif'
tf.imsave(save_n, mov_d)