import sys
sys.path.extend(['/home/junz/PycharmProjects/corticalmapping'])
import os
import numpy as np
import h5py
# import time
import pandas as pd
import corticalmapping.DatabaseTools as dt
import corticalmapping.core.ImageAnalysis as ia

table_name = 'big_roi_table_190423135026.xlsx'
sheet_name = 'f_center_subtracted'
nwb_folder = 'nwbs'
save_folder = "temp_xlsx"

response_dir = 'pos'
skew_thr = 0.6
trace_bias = 1.
rf_peak_z_thr = 1.3
gaussian_filter_sigma = 1. # float, in pixels, filtering sigma for z-score receptive fields
interpolate_rate = 10. # float, interpolate rate of filtered z-score maps
response_window = [0., 0.5]

rf_map_fn = 'rf_maps.hdf5'
notes = '''
   zscore receptive field maps of all significant rois. Spatial temporal receptive fields
   are first converted to df/f. Then 2-d zscore maps are generated. Then the zscore maps are
   2d filtered to smooth and interpolated in to high resolution. After preprocessing, if the
   peak value of zscore is larger than the threshold, the receptive field will be considered 
   as sigificant.
        '''

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

rf_map_f = h5py.File(os.path.join(curr_folder, rf_map_fn))
if 'notes' not in rf_map_f.keys():
    rf_map_f['notes'] = notes

table_path = os.path.join(curr_folder, 'intermediate_results', table_name)
df = pd.read_excel(table_path, sheetname=sheet_name)
subdf = df[np.logical_not(df['rf_pos_on_peak_z'].isnull())]
subdf = subdf[subdf['skew_fil'] >= skew_thr]

df_nwb = subdf[['date', 'mouse_id']].drop_duplicates()
nwb_fns_total = ['{}_{}_110_repacked'.format(r['date'], r['mouse_id']) for r_i, r in df_nwb.iterrows()]
print('all nwb files:')
print('\n'.join(nwb_fns_total))

# nwb_fns_saved = [gn[0:14] + '_110_repacked' for gn in rf_map_f.keys() if len(gn) >= 14]
# print('\nsaved nwb files:')
# print('\n'.join(nwb_fns_saved))
#
# nwb_fns_todo = []
# for nwb_fn in nwb_fns_total:
#     if nwb_fn not in nwb_fns_saved:
#         nwb_fns_todo.append(nwb_fn)
# print('\ntodo nwb files:')
# print('\n'.join(nwb_fns_todo))

for nwb_fn in nwb_fns_total:
    print('\nprocessing {} ...'.format(nwb_fn))
    nwb_f = h5py.File(os.path.join(nwb_folder, nwb_fn + '.nwb'))
    localdf = subdf[subdf['date'] == int(nwb_fn[0:6])]
    localdf = localdf[localdf['mouse_id'] == nwb_fn[7:14]]

    # ====================================ON=============================================================
    localdfon = localdf[localdf['rf_{}_on_peak_z'.format(response_dir)] >= rf_peak_z_thr].reset_index()
    print('\tnumber of rois with significant ON RFs: {}'.format(len(localdfon)))

    for roi_i, roi_row in localdfon.iterrows():

        # get the saving group
        group_n = '{}_{}_{}_ON'.format(roi_row['date'], roi_row['mouse_id'], roi_row['plane_n'])
        if group_n in rf_map_f.keys():
            local_grp = rf_map_f[group_n]
        else:
            local_grp = rf_map_f.create_group(name=group_n)
            local_grp.attrs['trace_type'] = sheet_name
            local_grp.attrs['response_dir'] = response_dir
            local_grp.attrs['skew_thr'] = 0.6
            local_grp.attrs['trace_bias'] = trace_bias
            local_grp.attrs['rf_peak_z_thr'] = rf_peak_z_thr
            local_grp.attrs['gaussian_filter_sigma'] = gaussian_filter_sigma
            local_grp.attrs['interpolation_rate'] = interpolate_rate
            local_grp.attrs['response_window'] = response_window

        print('\t\tprocessing roi {}/{} ...'.format(roi_i + 1, len(localdfon)))

        if roi_row['roi_n'] not in local_grp.keys():

            # get constant to add to trace
            trace, _ = dt.get_single_trace(nwb_f=nwb_f,
                                           plane_n=roi_row['plane_n'],
                                           roi_n=roi_row['roi_n'],
                                           trace_type=sheet_name)
            if np.min(trace) < trace_bias:
                add_to_trace = -np.min(trace) + trace_bias
            else:
                add_to_trace = 0.

            # get strf
            curr_strf = dt.get_strf(nwb_f=nwb_f,
                                    plane_n=roi_row['plane_n'],
                                    roi_ind=int(roi_row['roi_n'][-4:]),
                                    trace_type='sta_'+sheet_name)
            curr_strf_dff = curr_strf.get_local_dff_strf(is_collaps_before_normalize=True,
                                                         add_to_trace=add_to_trace)
            curr_srf, _ = curr_strf_dff.get_zscore_receptive_field(timeWindow=response_window)
            if response_dir == 'pos':
                polarity = 'positive'
            elif response_dir == 'neg':
                polarity = 'negative'
            else:
                raise ValueError('polarity ({}) should be either "pos" or "neg".')
            _, _, _, rf_mask = dt.get_rf_properties(srf=curr_srf,
                                                     polarity=polarity,
                                                     sigma=gaussian_filter_sigma,
                                                     interpolate_rate=interpolate_rate,
                                                     z_thr=rf_peak_z_thr)
            roi_grp = local_grp.create_group(name=roi_row['roi_n'])
            ia.WeightedROI(mask=rf_mask).to_h5_group(roi_grp)
            if 'alts_deg' not in local_grp.attrs:
                local_grp.attrs['alts_deg'] = curr_srf.altPos
            if 'azis_deg' not in local_grp.attrs:
                local_grp.attrs['alts_deg'] = curr_srf.aziPos
        else:
            print('\t\t\tAlready exists. Skip.')


    # ====================================OFF========================================================
    localdfoff = localdf[localdf['rf_{}_off_peak_z'.format(response_dir)] >= rf_peak_z_thr].reset_index()
    print('\n\tnumber of rois with significant OFF RFs: {}'.format(len(localdfoff)))

    for roi_i, roi_row in localdfoff.iterrows():

        # get the saving group
        group_n = '{}_{}_{}_OFF'.format(roi_row['date'], roi_row['mouse_id'], roi_row['plane_n'])
        if group_n in rf_map_f.keys():
            local_grp = rf_map_f[group_n]
        else:
            local_grp = rf_map_f.create_group(name=group_n)
            local_grp.attrs['trace_type'] = sheet_name
            local_grp.attrs['response_dir'] = response_dir
            local_grp.attrs['skew_thr'] = 0.6
            local_grp.attrs['trace_bias'] = trace_bias
            local_grp.attrs['rf_peak_z_thr'] = rf_peak_z_thr
            local_grp.attrs['gaussian_filter_sigma'] = gaussian_filter_sigma
            local_grp.attrs['interpolation_rate'] = interpolate_rate
            local_grp.attrs['response_window'] = response_window

        print('\t\tprocessing roi {}/{} ...'.format(roi_i + 1, len(localdfoff)))

        if roi_row['roi_n'] not in local_grp.keys():

            # get constant to add to trace
            trace, _ = dt.get_single_trace(nwb_f=nwb_f,
                                           plane_n=roi_row['plane_n'],
                                           roi_n=roi_row['roi_n'],
                                           trace_type=sheet_name)
            if np.min(trace) < trace_bias:
                add_to_trace = -np.min(trace) + trace_bias
            else:
                add_to_trace = 0.

            # get strf
            curr_strf = dt.get_strf(nwb_f=nwb_f,
                                    plane_n=roi_row['plane_n'],
                                    roi_ind=int(roi_row['roi_n'][-4:]),
                                    trace_type='sta_' + sheet_name)
            curr_strf_dff = curr_strf.get_local_dff_strf(is_collaps_before_normalize=True,
                                                         add_to_trace=add_to_trace)
            _, curr_srf = curr_strf_dff.get_zscore_receptive_field(timeWindow=response_window)
            if response_dir == 'pos':
                polarity = 'positive'
            elif response_dir == 'neg':
                polarity = 'negative'
            else:
                raise ValueError('polarity ({}) should be either "pos" or "neg".')
            _, _, _, rf_mask = dt.get_rf_properties(srf=curr_srf,
                                                     polarity=polarity,
                                                     sigma=gaussian_filter_sigma,
                                                     interpolate_rate=interpolate_rate,
                                                     z_thr=rf_peak_z_thr)
            roi_grp = local_grp.create_group(name=roi_row['roi_n'])
            ia.WeightedROI(mask=rf_mask).to_h5_group(roi_grp)
            if 'alts_deg' not in local_grp.attrs:
                local_grp.attrs['alts_deg'] = curr_srf.altPos
            if 'azis_deg' not in local_grp.attrs:
                local_grp.attrs['alts_deg'] = curr_srf.aziPos
        else:
            print('\t\t\tAlready exists. Skip.')

            