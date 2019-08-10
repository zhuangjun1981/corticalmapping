import os
import h5py
import corticalmapping.DatabaseTools as dt
import matplotlib.pyplot as plt

nwb_folder = 'nwbs'
nwb_fn = "190326_M441626_110.nwb"

plane_n = 'plane2'
roi_n = 'roi_0001'

analysis_params = dt.ANALYSIS_PARAMS
plot_params = dt.PLOTTING_PARAMS

analysis_params['trace_type'] = 'f_center_raw'
analysis_params['response_window_dgc'] = [0.5, 1.5]
analysis_params['baseline_window_dgc'] = [-0.5, 0.5]
plot_params['sftf_vmax'] = 6
plot_params['sftf_vmin'] = -6

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

nwb_f = h5py.File(os.path.join(nwb_folder, nwb_fn), 'r')
f = dt.roi_page_report(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n, params=analysis_params, plot_params=plot_params)
nwb_f.close()

plt.show()

f.savefig('{}_{}_{}.pdf'.format(os.path.splitext(nwb_fn)[0], plane_n, roi_n))