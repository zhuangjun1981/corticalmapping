import os
import corticalmapping.NwbTools as nt
import matplotlib.pyplot as plt
import tifffile as tf


vasmap_name_wf = 'vasmap_wf.tif'
vasmap_name_2p_zoom1 = 'vasmap_2p_zoom1.tif'
# vasmap_name_2p_zoom4 = 'vasmap_2p_zoom4.tif'

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

vasmap_wf = tf.imread(vasmap_name_wf)
vasmap_2p_zoom1 = tf.imread(vasmap_name_2p_zoom1)
# vasmap_2p_zoom4 = tf.imread(vasmap_name_2p_zoom4)

# f = plt.figure(figsize=(15, 7))
# ax1 = f.add_subplot(121)
# ax1.imshow(vasmap_wf, cmap='gray', interpolation='nearest')
# ax1.set_title('wide field surface vasculature')
# ax2 = f.add_subplot(122)
# ax2.imshow(vasmap_2p_zoom1, cmap='gray', interpolation='nearest')
# ax2.set_title('two photon surface vasculature')
# plt.show()

nwb_fn = [f for f in os.listdir(curr_folder) if f[-4:] == '.nwb'][0]

nwb_f = nt.RecordedFile(nwb_fn)
nwb_f.add_acquisition_image('surface_vas_map_wf', vasmap_wf,
                            description='wide field surface vasculature map through cranial window')
nwb_f.add_acquisition_image('surface_vas_map_2p_zoom1', vasmap_2p_zoom1,
                            description='2-photon surface vasculature map through cranial window, zoom 1')
# nwb_f.add_acquisition_image('surface_vas_map_2p_zoom2', vasmap_2p_zoom4,
#                             description='2-photon surface vasculature map through cranial window, zoom 4')
nwb_f.close()