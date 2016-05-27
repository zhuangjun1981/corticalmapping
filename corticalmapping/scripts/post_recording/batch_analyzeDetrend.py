import tifffile as tf
import corticalmapping.core.ImageAnalysis as ia
import corticalmapping.core.PlottingTools as pt
import corticalmapping.HighLevel as hl
import matplotlib.pyplot as plt
import numpy as np
import os


mov_name = "160525_M235516_aveMov_102.tif"

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)
mov_file_name, mov_file_ext = os.path.splitext(mov_name)
save_name = mov_file_name + '_detrend' + mov_file_ext

mov = tf.imread(mov_name).astype(np.float32)

height = mov.shape[1]
width = mov.shape[2]

roi = ia.generate_oval_mask((height, width), (height/2, width/2), int(height*0.6), int(width*0.6))

f = plt.figure(figsize=(10,10))
ax = f.add_subplot(111)
ax.imshow(mov[0,:,:],cmap='gray',interpolation='nearest')
pt.plot_mask_borders(roi,plotAxis=ax)

plt.show()

mov_detrend, trend, amp, rvalue = hl.regression_detrend(mov, roi)

f = plt.figure(figsize=(15, 4))
ax = f.add_subplot(111)
ax.plot(trend)
ax.set_title('trend')

f = plt.figure(figsize=(15,5))
ax1 = f.add_subplot(121)
fig1 = ax1.imshow(amp)
ax1.set_title('contribution')
f.colorbar(fig1)
ax2 = f.add_subplot(122)
fig2 = ax2.imshow(rvalue)
ax2.set_title('r value')
f.colorbar(fig2)

plt.show()

tf.imsave(save_name, mov_detrend.astype(np.float32))
