"""
these are the functions that deal with the .nwb database of GCaMP labelled LGN boutons.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.ndimage as ni
import scipy.interpolate as ip
import corticalmapping.SingleCellAnalysis as sca
import corticalmapping.core.ImageAnalysis as ia

ANALYSIS_PARAMS = {
    'trace_type': 'f_center_subtracted',
    'filter_length_skew_sec': 5., # float, second, the length to filter input trace to get slow trend
    'response_window_positive_rf': [0., 0.5], # list of 2 floats, temporal window to get upwards calcium response for receptive field
    'response_window_negative_rf': [0., 1.], # list of 2 floats, temporal window to get downward calcium response for receptive field
    'gaussian_filter_sigma_rf': 1., # float, filtering sigma for z-score receptive fields
    'interpolate_rate_rf': 10., # float, interpolate rate of filtered z-score maps
    'response_window_dgc': [0., 1.], # list of two floats, temporal window for getting response value for drifting grating
    'baseline_window_dgc': [-0.5, 0.] # list of two floats, temporal window for getting baseline value for drifting grating
                   }

def get_peak_z_rf(strf_grp, add_to_trace,
                  pos_win=ANALYSIS_PARAMS['response_window_positive_rf'],
                  neg_win=ANALYSIS_PARAMS['response_window_negative_rf'],
                  filter_sigma=ANALYSIS_PARAMS['gaussian_filter_sigma_rf'],
                  interpolate_rate=ANALYSIS_PARAMS['interpolate_rate_rf']):
    """
    return the maximum z-score from a spatial temporal receptive field

    the z-score is from four different z-score maps:
    on (bright probe) positive (upward calcium signal)
    off (dark probe) positive (upward calcium signal)
    on (bright probe) negative (downward calcium signal)
    off (dark probe) negative (downward calcium signal)

    z-score maps are first 2d gaussian filtered and interpolated and negative z-score for downward calcium signal are
    flipped to get the positive value.

    :param strf_grp: h5py group containing spatial temporal receptive field
    :param add_to_trace: float, a number to add to responses for avoiding negative baseline
    :param pos_win: list of 2 floats, temporal window to get upwards calcium response
    :param neg_win: list of 2 floats, temporal window to get downward calcium response
    :param filter_sigma: float, filtering sigma
    :param interpolate_rate: float, interpolate rate of filtered z-score maps
    :return peak_z_score: float, the maximum z_score of all four z-score maps
    """

    strf = sca.SpatialTemporalReceptiveField.from_h5_group(strf_grp)
    strf_dff = strf.get_local_dff_strf(is_collaps_before_normalize=True, add_to_trace=add_to_trace)

    peak_z = []

    zscore_ON_pos, zscore_OFF_pos, allAltPos, allAziPos = strf_dff.get_zscore_map(timeWindow=pos_win)
    zscore_ON_pos[np.isnan(zscore_ON_pos)] = 0.
    zscore_OFF_pos[np.isnan(zscore_OFF_pos)] = 0.
    zscore_ON_neg, zscore_OFF_neg, allAltPos, allAziPos = strf_dff.get_zscore_map(timeWindow=neg_win)
    zscore_ON_neg[np.isnan(zscore_ON_neg)] = 0.
    zscore_OFF_neg[np.isnan(zscore_OFF_neg)] = 0.
    zscore_ON_neg = -zscore_ON_neg
    zscore_OFF_neg = -zscore_OFF_neg

    altStep = np.mean(np.diff(allAltPos))
    aziStep = np.mean(np.diff(allAziPos))
    newAltPos = np.arange(allAltPos[0], allAltPos[-1], altStep / float(interpolate_rate))
    newAziPos = np.arange(allAziPos[0], allAziPos[-1], aziStep / float(interpolate_rate))

    zscore_ON_pos = ni.gaussian_filter(zscore_ON_pos, sigma=filter_sigma)
    ip_ON_pos = ip.interp2d(allAziPos, allAltPos, zscore_ON_pos, kind='cubic', fill_value=0.)
    zscore_ON_pos = ip_ON_pos(newAziPos, newAltPos)
    peak_z.append(np.amax(zscore_ON_pos))

    zscore_OFF_pos = ni.gaussian_filter(zscore_OFF_pos, sigma=filter_sigma)
    ip_OFF_pos = ip.interp2d(allAziPos, allAltPos, zscore_OFF_pos, kind='cubic', fill_value=0.)
    zscore_OFF_pos = ip_OFF_pos(newAziPos, newAltPos)
    peak_z.append(np.amax(zscore_OFF_pos))

    zscore_ON_neg = ni.gaussian_filter(zscore_ON_neg, sigma=filter_sigma)
    ip_ON_neg = ip.interp2d(allAziPos, allAltPos, zscore_ON_neg, kind='cubic', fill_value=0.)
    zscore_ON_neg = ip_ON_neg(newAziPos, newAltPos)
    peak_z.append(np.amax(zscore_ON_neg))

    zscore_OFF_neg = ni.gaussian_filter(zscore_OFF_neg, sigma=filter_sigma)
    ip_OFF_neg = ip.interp2d(allAziPos, allAltPos, zscore_OFF_neg, kind='cubic', fill_value=0.)
    zscore_OFF_neg = ip_OFF_neg(newAziPos, newAltPos)
    peak_z.append(np.amax(zscore_OFF_neg))

    peak_z = np.max(peak_z)

    return peak_z


def get_dgc_responsiveness(dgc_grp, add_to_trace,
                           trace_type=ANALYSIS_PARAMS['trace_type'],
                           response_window=ANALYSIS_PARAMS['response_window_dgc'],
                           baseline_window=ANALYSIS_PARAMS['baseline_window_dgc']):
    """
    calculate p valuses of all rois from their responses to drifting grating for determining their responsiveness

    :param dgc_grp: hdf5 group containing response table, from nwb file
    :param add_to_trace: float, a number to add to responses for avoiding negative baseline
    :param trace_type: str, type of traces for calculation
    :param response_window: list of two floats, temporal window for getting response value
    :param baseline_window: list of two floats, temporal window for getting baseline
    :return p_ttest_lst: list of min p-values for each roi across all conditions against blank trial. length is number of rois
    :return p_anova_lst: list of p-values of one-way ANOVA across all conditions for each roi
    :return peak_dff_lst: list of maximum dff across all conditions
    """

    t_axis = dgc_grp.attrs['sta_timestamps']
    grating_ns = dgc_grp.keys()

    dffs_table_trial = [] # 3d array, roi x trial x condition
    dffs_table_mean = [] # 2d array, condition x roi

    for grating_n in grating_ns:
        traces_cond = dgc_grp[grating_n]['sta_' + trace_type].value + add_to_trace
        dffs_cond_trial, dffs_cond_mean = sca.get_dff(traces=traces_cond, t_axis=t_axis,
                                                      response_window=response_window,
                                                      baseline_window=baseline_window)
        dffs_table_trial.append(dffs_cond_trial)
        dffs_table_mean.append(dffs_cond_mean)

        if grating_n[22:26] == '0.00': # this check is vulnerable
            dffs_blank_trial = dffs_cond_trial # roi x trial x 1

    # this is response table of dffs
    # dimension: roi x trial x condition
    dffs_table_trial = np.concatenate(dffs_table_trial, axis=2)

    # this is mean response table of dffs
    # dimension: condition x roi
    dffs_table_mean = np.array(dffs_table_mean)
    peak_dff_lst = np.max(dffs_table_mean, axis=0)

    p_ttest_lst = []
    p_anova_lst = []

    for roi_i in range(dffs_table_trial.shape[0]):

        _, p_anova = stats.f_oneway(*dffs_table_trial[roi_i].transpose())

        p_ttest = []

        for cond_i in range(dffs_table_trial.shape[2]):
            dff_blank = dffs_blank_trial[roi_i, :, 0]
            dff_cond = dffs_table_trial[roi_i, :, cond_i]
            _, curr_p = stats.ttest_ind(dff_cond, dff_blank)
            p_ttest.append(curr_p)

        p_ttest = np.min(p_ttest)

        p_ttest_lst.append(p_ttest)
        p_anova_lst.append(p_anova)

    return p_ttest_lst, p_anova_lst, peak_dff_lst


def roi_page_report(nwb_path, plane_n, roi_n, params=ANALYSIS_PARAMS):
    """
    generate a page of description of an roi

    :param nwb_path:
    :param plane_n:
    :param roi_n:
    :param params:
    :return:
    """

    nwb_f = h5py.File(nwb_path)
    segmentation_grp = nwb_f['processing/rois_and_traces_{}/ImageSegmentation/imaging_plane'.format(plane_n)]
    rf_img_grp = segmentation_grp['reference_images']
    if 'mean_projection' in rf_img_grp.keys():
        rf_img = rf_img_grp['mean_projection/data'].value
    else:
        rf_img = rf_img_grp['max_projection/data'].value




    f = plt.figure(figsize=(8.5, 11))
    f.subplots_adjust(0, 0, 1, 1)
    ax_rf_img = f.add_axes([0.01, 0.75, 0.3, 0.25])
    ax_rf_img.imshow(ia.array_nor(rf_img), cmap='gray', vmin=0, vmax=0.5, interpolation='nearest')
    ax_rf_img.set_axis_off()
    plt.show()

    pass

if __name__ == '__main__':
    nwb_path = r"F:\data2\chandelier_cell_project\database\190208_M421761_110.nwb"
    plane_n = 'plane0'
    roi_n = 'roi_0000'
    roi_page_report(nwb_path=nwb_path, plane_n=plane_n, roi_n=roi_n)




