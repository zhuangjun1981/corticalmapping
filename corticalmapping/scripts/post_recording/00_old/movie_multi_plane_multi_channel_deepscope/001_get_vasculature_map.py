import os
import numpy as np
import tifffile as tf
import skimage.io as io
import matplotlib.pyplot as plt
import corticalmapping.core.ImageAnalysis as ia

vasmap_wf_path = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data_rabies_project" \
                 r"\180502-M376019-deepscope\Widefield.tif"

vasmap_2p_zoom1_path = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data_rabies_project" \
                       r"\180502-M376019-deepscope\01\01_00001.tif"

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

vasmap_wf = io.imread(vasmap_wf_path, as_grey=True)
vasmap_wf = vasmap_wf.transpose()[::-1, ::-1]

vasmap_2p_zoom1 = tf.imread(vasmap_2p_zoom1_path).astype(np.float32)
vasmap_2p_zoom1 = np.mean(vasmap_2p_zoom1, axis=0)
vasmap_2p_zoom1 = vasmap_2p_zoom1.transpose()[::-1, ::-1]

f = plt.figure(figsize=(12, 5))
ax_wf = f.add_subplot(121)
ax_wf.imshow(ia.array_nor(vasmap_wf), vmin=0., vmax=1., cmap='gray', interpolation='nearest')
ax_wf.set_title('vasmap wide field')
ax_wf.set_axis_off()
ax_2p = f.add_subplot(122)
ax_2p.imshow(ia.array_nor(vasmap_2p_zoom1), vmin=0., vmax=0.15, cmap='gray', interpolation='nearest')
ax_2p.set_title('vasmap 2p zoom1')
ax_2p.set_axis_off()

plt.show()

tf.imsave('vasmap_wf.tif', vasmap_wf)
tf.imsave('vasmap_2p_zoom1.tif', vasmap_2p_zoom1)