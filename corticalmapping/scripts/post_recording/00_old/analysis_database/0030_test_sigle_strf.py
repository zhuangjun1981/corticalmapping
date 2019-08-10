import os
import h5py
import numpy as np
import corticalmapping.DatabaseTools as dt
import corticalmapping.SingleCellAnalysis as sca

fn_original = '180813_M386444_110.nwb'
fn_repacked = '180813_M386444_110_repacked.nwb'

roi_ind = 0
params = dt.ANALYSIS_PARAMS
params['gaussian_filter_sigma_rf'] = 1.
params['interpolate_rate_rf'] = 10.
params['rf_z_threshold'] = 1.3

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

roi_n = 'roi_{:04d}'.format(roi_ind)

def print_peak_z(strf):
    strf_dff = strf.get_local_dff_strf(is_collaps_before_normalize=True, add_to_trace=1)

    # positive spatial receptive fields
    srf_on, srf_off = strf_dff.get_zscore_receptive_field(timeWindow=[0., 0.5])

    srf = srf_on.gaussian_filter(sigma=1.)
    srf = srf.interpolate(ratio=10.)

    print(np.max(srf.weights))


f_o = h5py.File(fn_original, 'r')
strf_o = sca.SpatialTemporalReceptiveField.from_h5_group(f_o['analysis/STRFs/plane0/strf_roi_{:04d}'.format(roi_ind)])
print_peak_z(strf_o)
# print(strf_o.data['traces'][0][5])
# print(strf_o.time)
f_o.close()

f_r = h5py.File(os.path.join('repacked', fn_repacked), 'r')
strf_r = dt.get_strf(f_r, plane_n='plane0', roi_ind=0, trace_type='sta_f_center_subtracted')
print_peak_z(strf_r)
# print(strf_r.data['traces'][0][5])
# print(strf_r.time)

roi_properties, _, _, _, _, _, _, _, _, _, _, _, _, _ = \
                        dt.get_everything_from_roi(nwb_f=f_r, plane_n='plane0', roi_n=roi_n, params=params)
print(roi_properties['rf_pos_on_peak_z'])

f_r.close()