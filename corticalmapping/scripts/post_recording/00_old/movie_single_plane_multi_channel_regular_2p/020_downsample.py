import os
import tifffile as tf
import corticalmapping.core.ImageAnalysis as ia

xy_downsample_rate = 2
t_downsample_rate = 10

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

f_ns = [f for f in os.listdir(curr_folder) if f[-4:] == '.tif' and 'downsampled' not in f]
f_ns.sort()
print('\n'.join(f_ns))

for f_n in f_ns:
    print('processing {} ...'.format(f_n))
    mov = tf.imread(f_n)
    mov_d = ia.rigid_transform_cv2(img=mov, zoom=(1. / xy_downsample_rate))
    mov_d = ia.z_downsample(mov_d, downSampleRate=t_downsample_rate)
    save_n = os.path.splitext(f_n)[0] + '_downsampled.tif'
    tf.imsave(save_n, mov_d)