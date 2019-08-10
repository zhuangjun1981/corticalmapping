import os
import h5py
import allensdk.brain_observatory.dff as dff
import numpy as np
import corticalmapping.HighLevel as hl
import corticalmapping.core.FileTools as ft


curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

save_plot_dir = os.path.join(curr_folder, 'figures', 'dff_extraction')
if not os.path.isdir(save_plot_dir):
    os.makedirs(save_plot_dir)

data_f = h5py.File('rois_and_traces.hdf5')
traces_subtracted = data_f['traces_center_subtracted'].value

traces_dff = dff.compute_dff(traces_subtracted, save_plot_dir=save_plot_dir,
                             mode_kernelsize=100, mean_kernelsize=100)
data_f['traces_center_dff'] = traces_dff
data_f.close()