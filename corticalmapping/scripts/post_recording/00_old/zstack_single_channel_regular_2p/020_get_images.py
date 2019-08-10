import os
import tifffile as tf
import numpy as np

identifier = 'cell1_zoom4'
data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data_rabies_project\180419-M388189-2p"

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

# save_folder = os.path.join(curr_folder, identifier)
# if not os.path.isdir(save_folder):
#     os.makedirs(save_folder)

source_folder = os.path.join(data_folder, identifier)
folder_ns = [f for f in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, f))]
folder_ns.sort()
print('\n'.join(folder_ns))

stack = []
for folder_n in folder_ns:
    curr_source_folder = os.path.join(source_folder, folder_n)
    stack.append(tf.imread(os.path.join(curr_source_folder, 'corrected_mean_projection.tif')))

stack = np.array(stack)
tf.imsave(os.path.join(curr_folder, '{}_zstack.tif'.format(identifier)), stack)
# tf.imsave(os.path.join(save_folder, '{}_zstack.tif'.format(identifier)), stack)