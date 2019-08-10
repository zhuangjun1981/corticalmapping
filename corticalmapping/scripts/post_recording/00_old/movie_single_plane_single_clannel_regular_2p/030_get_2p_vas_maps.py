import os
import numpy as np
import tifffile as tf
import corticalmapping.core.ImageAnalysis as ia
import matplotlib.pyplot as plt


data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data_rabies_project\180404-M360495-2p\vasmap_2p"

zoom1_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder)
               if f[-12:] == '_rotated.tif' and '_zoom1_' in f]

# zoom2_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder)
#                if f[-12:] == '_rotated.tif' and '_zoom2_' in f]

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

vas_map_zoom1 = []
# vas_map_zoom2 = []

for zoom1_path in zoom1_paths:
    curr_vasmap = np.mean(tf.imread(zoom1_path), axis=0)
    vas_map_zoom1.append(curr_vasmap)

# for zoom2_path in zoom2_paths:
#     curr_vasmap = np.mean(tf.imread(zoom2_path), axis=0)
#     vas_map_zoom2.append(curr_vasmap)

vas_map_zoom1 = ia.array_nor(np.mean(vas_map_zoom1, axis=0))
# vas_map_zoom2 = ia.array_nor(np.mean(vas_map_zoom2, axis=0))

tf.imsave('vas_map_focused_2p_zoom1.tif', vas_map_zoom1.astype(np.float32))
# tf.imsave('vas_map_focused_2p_zoom2.tif', vas_map_zoom2.astype(np.float32))
