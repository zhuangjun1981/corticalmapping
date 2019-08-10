import os
import corticalmapping.NwbTools as nt
import matplotlib.pyplot as plt
import tifffile as tf


vasmap_name_wf = 'vasmap_wf.tif'
# vasmap_name_2p_zoom1_g = 'vasmap_2p_zoom1_green.tif'
vasmap_name_2p_zoom1_r = 'vasmap_2p_zoom1.tif'

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

vasmap_wf = tf.imread(vasmap_name_wf)
# vasmap_2p_zoom1_g = tf.imread(vasmap_name_2p_zoom1_g)
vasmap_2p_zoom1_r = tf.imread(vasmap_name_2p_zoom1_r)

nwb_fn = [f for f in os.listdir(curr_folder) if f[-4:] == '.nwb'][0]

nwb_f = nt.RecordedFile(nwb_fn)
nwb_f.add_acquisition_image('surface_vas_map_wf', vasmap_wf,
                            description='wide field surface vasculature map through cranial window')
# nwb_f.add_acquisition_image('surface_vas_map_2p_zoom1_green', vasmap_2p_zoom1_g,
#                             description='2-photon surface vasculature map through cranial window, zoom 1, green')
nwb_f.add_acquisition_image('surface_vas_map_2p_zoom1_red', vasmap_2p_zoom1_r,
                            description='2-photon surface vasculature map through cranial window, zoom 1, red')
nwb_f.close()