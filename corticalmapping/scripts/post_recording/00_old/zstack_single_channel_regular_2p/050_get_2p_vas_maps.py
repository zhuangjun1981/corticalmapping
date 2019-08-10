import os
import numpy as np
import tifffile as tf
import corticalmapping.core.ImageAnalysis as ia
import matplotlib.pyplot as plt


data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data_rabies_project\180322-M360495-2p"
file_name = "vasmap_zoom1_00001_00001.tif"
save_name = 'vasmap_2p_zoom1.tif'

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

vasmap = tf.imread(os.path.join(data_folder, file_name))
vasmap = np.mean(vasmap.transpose((0, 2, 1))[:, ::-1, :], axis=0)
vasmap = ia.array_nor(vasmap)

tf.imsave(save_name, vasmap.astype(np.float32))
