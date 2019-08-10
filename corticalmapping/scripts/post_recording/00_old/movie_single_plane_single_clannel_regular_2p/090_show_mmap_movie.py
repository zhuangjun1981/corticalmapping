import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend([r"E:\data\github_packages\CaImAn"])

import os
import numpy as np
import caiman as cm
import matplotlib.pyplot as plt

data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data_rabies_project" \
              r"\180209-M360495-2p\2p_movie\rotated\corrected"
fn = '180209_M360495_110_d1_512_d2_512_d3_1_order_C_frames_400_.mmap'

fn_parts = fn.split('_')
d1 = int(fn_parts[fn_parts.index('d1') + 1]) # column, x
d2 = int(fn_parts[fn_parts.index('d2') + 1]) # row, y
d3 = int(fn_parts[fn_parts.index('d3') + 1]) # channel
d4 = int(fn_parts[fn_parts.index('frames') + 1]) # frame, T
order = fn_parts[fn_parts.index('order') + 1]

mov = np.memmap(filename=os.path.join(data_folder, fn), shape=(d1, d2, d4), order=order, dtype=np.float32, mode='r')
mov = mov.transpose((2, 1, 0))

print('movie shape: {}'.format(mov.shape))

f = plt.figure(figsize=(8, 5))
ax = f.add_subplot(111)
fig = ax.imshow(np.mean(mov, axis=0), vmin=300, vmax=1500, cmap='inferno', interpolation='nearest')
f.colorbar(fig)
plt.show()

input("Press enter to continue ...")

print('playing {} ...'.format(fn))
cm.movie(mov).play(fr=30,magnification=1,gain=2.)

