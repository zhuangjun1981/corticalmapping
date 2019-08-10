import os
import h5py
import tifffile as tf
import numpy as np
import matplotlib.pyplot as plt
import corticalmapping.core.PlottingTools as pt
import corticalmapping.core.ImageAnalysis as ia


cor_thr = 0.8

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

def merger_pairs(pairs):

    total_set = set([])
    for pair in pairs:
        total_set.update(set(pair))

    all_nodes = list(total_set)
    node_grps = [{n} for n in all_nodes]

    for pair in pairs:

        node0 = pair[0]
        node1 = pair[1]

        for node_grp in node_grps:
            if node0 in node_grp:
                node_grp0 = node_grp
            if node1 in node_grp:
                node_grp1 = node_grp

        if node_grp0 != node_grp1:
            node_grp0.update(node_grp1)
            node_grps.remove(node_grp1)

    return node_grps


save_plot_dir = os.path.join(curr_folder, 'figures', 'dff_extraction')
if not os.path.isdir(save_plot_dir):
    os.makedirs(save_plot_dir)

bg = ia.array_nor(tf.imread('corrected_mean_projection.tif'))

data_f = h5py.File('rois_and_traces.hdf5')
traces_subtracted = data_f['traces_center_subtracted'].value
masks = data_f['masks_center'].value

f, axs = plt.subplots(1, 2, figsize=(16, 5))

cor_mat = np.corrcoef(traces_subtracted)
fig = axs[0].imshow(cor_mat, vmin=-1, vmax=1, cmap='jet', interpolation='nearest')
axs[0].set_title('coriance matrix')
f.colorbar(fig, ax=axs[0])

cors = cor_mat[np.tril_indices(cor_mat.shape[0], k=-1)]
cor_dist = axs[1].hist(cors, range=[-1., 1.], bins=40)
axs[1].set_title('coriance distribution')

# cors = np.sort(cors)
# cor_thr = cors[int(cors.shape[0] * 0.99)]
# print('Cutoff threshold for coriance: {}'.format(cor_thr))

pos_cor_loc = np.where(cor_mat > cor_thr)

roi_pairs = []
for ind in range(len(pos_cor_loc[0])):
    if pos_cor_loc[0][ind] < pos_cor_loc[1][ind]:
        roi_pairs.append([pos_cor_loc[0][ind], pos_cor_loc[1][ind]])
print(roi_pairs)

roi_grps = merger_pairs(roi_pairs)
print roi_grps

cor_grps = []
for roi_grp in roi_grps:
    grp_traces = traces_subtracted[list(roi_grp)]
    grp_cors = np.corrcoef(grp_traces)[np.tril_indices(len(roi_grp), k=-1)]
    cor_grps.append(np.mean(grp_cors))

cor_grps = np.array(cor_grps)
cor_scalars = [(c + 1) / 2 for c in cor_grps]
print cor_scalars
cor_colors = [pt.value_2_rgb(c, cmap='inferno') for c in cor_scalars]

f_roi = plt.figure()
ax_roi = f_roi.add_subplot(111)
ax_roi.imshow(bg, vmin=0, vmax=0.5, cmap='gray', interpolation='nearest')
for grp_ind, roi_grp in enumerate(roi_grps):
    for roi_ind in roi_grp:
        print roi_ind, cor_colors[grp_ind]
        pt.plot_mask_borders(masks[roi_ind], plotAxis=ax_roi, color=cor_colors[grp_ind])

plt.show()

data_f.close()