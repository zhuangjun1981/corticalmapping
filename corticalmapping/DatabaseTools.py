"""
these are the functions that deal with the .nwb database of GCaMP labelled LGN boutons.
"""
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.ndimage as ni
import scipy.interpolate as ip
import corticalmapping.SingleCellAnalysis as sca
import corticalmapping.core.ImageAnalysis as ia
import corticalmapping.core.PlottingTools as pt

ANALYSIS_PARAMS = {
    'trace_type': 'f_center_subtracted',
    'add_to_trace_bias': 1.,
    'filter_length_skew_sec': 5., # float, second, the length to filter input trace to get slow trend
    'response_window_positive_rf': [0., 0.5], # list of 2 floats, temporal window to get upwards calcium response for receptive field
    'response_window_negative_rf': [0., 1.], # list of 2 floats, temporal window to get downward calcium response for receptive field
    'gaussian_filter_sigma_rf': 1., # float, filtering sigma for z-score receptive fields
    'interpolate_rate_rf': 10., # float, interpolate rate of filtered z-score maps
    # 'peak_z_threshold_rf': 1.5, # float, threshold for significant receptive field of z score after filtering.
    'rf_z_threshold': 1.6, # float, threshold for significant zscore receptive field
    'response_window_dgc': [0., 1.], # list of two floats, temporal window for getting response value for drifting grating
    'baseline_window_dgc': [-0.5, 0.], # list of two floats, temporal window for getting baseline value for drifting grating
    'is_collapse_sf': True, # bool, average across sf or not for direction/tf tuning curve
    'is_collapse_tf': False, # bool, average across tf or not for direction/sf tuning curve
    'is_collapse_dire': False, # bool, average across direction or not for tf/sf tuning curve
    'is_rectify_dgc_tuning': True, # bool, if True, responses below zero (in defined polarity) will be set as zero
                   }

PLOTTING_PARAMS = {
    'response_type_for_plot': 'zscore', # str, 'df', 'dff', or 'zscore'
    'fig_size': (8.5, 11),
    'fig_facecolor': "#ffffff",
    'ax_roi_img_coord': [0.01, 0.75, 0.3, 0.24], # coordinates of roi image
    'rf_img_vmin': 0., # reference image min
    'rf_img_vmax': 0.5, # reference image max
    'roi_border_color': '#ff0000',
    'roi_border_width': 2,
    'field_traces_coord': [0.32, 0.75, 0.67, 0.24], # field coordinates of trace plot
    'traces_panels': 4, # number of panels to plot traces
    'traces_color': '#888888',
    'traces_line_width': 0.5,
    'ax_rf_pos_coord': [0.01, 0.535, 0.3, 0.24],
    'ax_rf_neg_coord': [0.32, 0.535, 0.3, 0.24],
    'rf_zscore_vmax': 4.,
    'ax_peak_traces_pos_coord': [0.01, 0.39, 0.3, 0.17],
    'ax_peak_traces_neg_coord': [0.32, 0.39, 0.3, 0.17],
    'blank_traces_color': '#888888',
    'peak_traces_pos_color': '#ff0000',
    'peak_traces_neg_color': '#0000ff',
    'response_window_color': '#ff00ff',
    'baseline_window_color': '#888888',
    'block_face_color': '#cccccc',
    'single_traces_lw': 0.5,
    'mean_traces_lw': 2.,
    'ax_text_coord': [0.63, 0.01, 0.36, 0.73],
    'ax_sftf_pos_coord': [0.01, 0.21, 0.3, 0.17],
    'ax_sftf_neg_coord': [0.32, 0.21, 0.3, 0.17],
    'sftf_cmap': 'RdBu_r',
    'sftf_vmax': 4,
    'sftf_vmin': -4,
    'ax_dire_pos_coord': [0.01, 0.01, 0.28, 0.18],
    'ax_dire_neg_coord': [0.32, 0.01, 0.28, 0.18],
    'dire_color_pos': '#ff0000',
    'dire_color_neg': '#0000ff',
    'dire_line_width': 2,

}


def get_strf_grp_key(nwb_f):
    analysis_grp = nwb_f['analysis']
    strf_key = [k for k in analysis_grp.keys() if k[0:4] == 'strf' and 'SparseNoise' in k]
    if len(strf_key) == 0:
        return None
    elif len(strf_key) == 1:
        return strf_key[0]
    else:
        raise LookupError('more than one drifting grating response table found.')


def get_strf(nwb_f, plane_n, roi_ind, trace_type):
    strf_key = get_strf_grp_key(nwb_f=nwb_f)

    if strf_key is not None:
        strf_grp = nwb_f['analysis/{}/{}'.format(strf_key, plane_n)]
        strf = sca.get_strf_from_nwb(h5_grp=strf_grp, roi_ind=roi_ind, trace_type=trace_type)
        return strf
        # try:
        #     strf_grp = nwb_f['analysis/{}/{}'.format(strf_key, plane_n)]
        #     strf = sca.get_strf_from_nwb(h5_grp=strf_grp, roi_ind=roi_ind, trace_type=trace_type)
        #     return strf
        # except Exception:
        #     return None
    else:
        return None


def get_dgcrm_grp_key(nwb_f):
    analysis_grp = nwb_f['analysis']
    dgcrt_key = [k for k in analysis_grp.keys() if k[0:14] == 'response_table' and 'DriftingGrating' in k]
    if len(dgcrt_key) == 0:
        return None
    elif len(dgcrt_key) == 1:
        return dgcrt_key[0]
    else:
        raise LookupError('more than one drifting grating response table found.')


def get_dgcrm(nwb_f, plane_n, roi_ind, trace_type):

    dgcrm_key = get_dgcrm_grp_key(nwb_f=nwb_f)

    if dgcrm_key is not None:
        dgcrm_grp = nwb_f['analysis/{}/{}'.format(dgcrm_key, plane_n)]
        dgcrm = sca.get_dgc_response_matrix_from_nwb(h5_grp=dgcrm_grp,
                                                     roi_ind=roi_ind,
                                                     trace_type=trace_type)
        return dgcrm
        # try:
        #     dgcrm_grp = nwb_f['analysis/{}/{}'.format(dgcrm_key, plane_n)]
        #     dgcrm = sca.get_dgc_response_matrix_from_nwb(h5_grp=dgcrm_grp,
        #                                                  roi_ind=roi_ind,
        #                                                  trace_type=trace_type)
        #     return dgcrm
        # except Exception as e:
        #     # print(e)
        #     return None
    else:
        return None


def get_rf_properties(srf,
                      polarity,
                      sigma=ANALYSIS_PARAMS['gaussian_filter_sigma_rf'],
                      interpolate_rate=ANALYSIS_PARAMS['interpolate_rate_rf'],
                      z_thr=ANALYSIS_PARAMS['rf_z_threshold']):
    """
    return receptive field properties from a SpatialReceptiveField

    :param srf: SingleCellAnalysis.SpatialReceptiveField object
    :param polarity: str, 'positive' or 'negative', the direction to apply threshold
    :param probe_size: list of two floats, height and width of square size
    :param simgma: float, 2d gaussian filter size, in pixel
    :param interpolate_rate: int, interpolation upsample rate
    :param peak_z_thr:
    :return rf_z: peak absolute zscore after filtering and interpolation
    :return rf_center: list of two floats, (alt, azi) in visual degrees
    :return rf_area: float, unit: visual degree squares
    """

    srf = srf.gaussian_filter(sigma=sigma)
    srf = srf.interpolate(ratio=interpolate_rate)

    if polarity == 'positive':
        rf_z = np.max(srf.weights)
    elif polarity == 'negative':
        srf.weight = -srf.weights
        rf_z = np.max(srf.weights)
    else:
        raise LookupError('Do not understand "polarity" ({}), should be "positive" or "negative".'.format(polarity))

    srf = srf.threshold(thr=z_thr)
    rf_center = srf.get_weighted_rf_center()
    rf_area = srf.get_binary_rf_area()
    rf_mask = srf.get_weighted_mask()

    return  rf_z, rf_center, rf_area, rf_mask


def get_roi(nwb_f, plane_n, roi_n):
    """

    :param nwb_f: h5py File object of the nwb file
    :param plane_n:
    :param roi_n:
    :return: core.ImageAnalysis.WeightedROI object of the specified roi
    """

    try:
        pixel_size = nwb_f['acquisition/timeseries/2p_movie_{}/pixel_size'.format(plane_n)].value
        pixel_size_unit = nwb_f['acquisition/timeseries/2p_movie_{}/pixel_size_unit'.format(plane_n)].value
    except Exception as e:
        pixel_size = None
        pixel_size_unit = None

    roi_grp = nwb_f['processing/rois_and_traces_{}/ImageSegmentation/imaging_plane/{}'.format(plane_n, roi_n)]
    mask = roi_grp['img_mask'].value
    return ia.WeightedROI(mask=mask, pixelSize=pixel_size, pixelSizeUnit=pixel_size_unit)


def get_single_trace(nwb_f, plane_n, roi_n, trace_type=ANALYSIS_PARAMS['trace_type']):
    roi_i = int(roi_n[-4:])
    trace = nwb_f['processing/rois_and_traces_{}/Fluorescence/{}/data'.format(plane_n, trace_type)][roi_i, :]
    trace_ts = nwb_f['processing/rois_and_traces_{}/Fluorescence/{}/timestamps'.format(plane_n, trace_type)].value
    return trace, trace_ts


def render_rb(rf_on, rf_off, vmax=PLOTTING_PARAMS['rf_zscore_vmax']):

    rf_on = (rf_on / vmax)
    rf_on[rf_on < 0] = 0
    rf_on[rf_on > 1] = 1
    rf_on = np.array(rf_on * 255, dtype=np.uint8)

    rf_off = (rf_off / vmax)
    rf_off[rf_off < 0] = 0
    rf_off[rf_off > 1] = 1
    rf_off = np.array(rf_off * 255, dtype=np.uint8)

    g_channel = np.zeros(rf_on.shape, dtype=np.uint8)
    rf_rgb = np.array([rf_on, g_channel, rf_off]).transpose([1, 2, 0])
    return rf_rgb


def get_everything_from_roi(nwb_f, plane_n, roi_n, params=ANALYSIS_PARAMS):
    """

    :param nwbf: h5py.File object
    :param plane_n:
    :param roi_n:
    :return:
    """

    roi_ind = int(roi_n[-4:])

    roi_properties = {'date': nwb_f['identifier'].value[0:6],
                      'mouse_id': nwb_f['identifier'].value[7:14],
                      'plane_n': plane_n,
                      'roi_n': roi_n,
                      'depth': nwb_f['processing/rois_and_traces_{}/imaging_depth_micron'.format(plane_n)].value}

    # get roi properties
    roi = get_roi(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n)
    pixel_size = nwb_f['acquisition/timeseries/2p_movie_{}/pixel_size'.format(plane_n)].value * 1000000.
    roi_area = roi.get_binary_area() * pixel_size[0] * pixel_size[1]
    roi_center_row, roi_center_col = roi.get_weighted_center()
    roi_properties.update({'roi_area': roi_area,
                           'roi_center_row': roi_center_row,
                           'roi_center_col': roi_center_col})

    # get skewness
    trace, trace_ts = get_single_trace(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n,
                                       trace_type=params['trace_type'])
    skew_raw, skew_fil = sca.get_skewness(trace=trace, ts=trace_ts,
                                          filter_length=params['filter_length_skew_sec'])
    roi_properties.update({'skew_raw': skew_raw,
                           'skew_fil': skew_fil})

    if np.min(trace) <= 0.:
        add_to_trace = -np.min(trace) + params['add_to_trace_bias']
    else:
        add_to_trace = 0.

    strf = get_strf(nwb_f=nwb_f, plane_n=plane_n, roi_ind=roi_ind, trace_type='sta_' + params['trace_type'])
    if strf is not None:
        # get strf properties
        strf_dff = strf.get_local_dff_strf(is_collaps_before_normalize=True, add_to_trace=add_to_trace)

        # positive spatial receptive fields
        srf_pos_on, srf_pos_off = strf_dff.get_zscore_receptive_field(timeWindow=params['response_window_positive_rf'])

        # ON positive spatial receptive field
        rf_pos_on_z, rf_pos_on_center, rf_pos_on_area, rf_pos_on_mask = get_rf_properties(srf= srf_pos_on,
                                                                          polarity='positive',
                                                                          sigma=params['gaussian_filter_sigma_rf'],
                                                                          interpolate_rate=params['interpolate_rate_rf'],
                                                                          z_thr=params['rf_z_threshold'])
        roi_properties.update({'rf_pos_on_peak_z': rf_pos_on_z,
                               'rf_pos_on_area': rf_pos_on_area,
                               'rf_pos_on_center_alt': rf_pos_on_center[0],
                               'rf_pos_on_center_azi': rf_pos_on_center[1]})

        # OFF positive spatial receptive field
        rf_pos_off_z, rf_pos_off_center, rf_pos_off_area, rf_pos_off_mask = get_rf_properties(srf=srf_pos_off,
                                                                             polarity='positive',
                                                                             sigma=params['gaussian_filter_sigma_rf'],
                                                                             interpolate_rate=params[
                                                                              'interpolate_rate_rf'],
                                                                             z_thr=params['rf_z_threshold'])

        # on off overlapping
        rf_pos_lsi = sca.get_local_similarity_index(rf_pos_on_mask, rf_pos_off_mask)

        roi_properties.update({'rf_pos_off_peak_z': rf_pos_off_z,
                               'rf_pos_off_area': rf_pos_off_area,
                               'rf_pos_off_center_alt': rf_pos_off_center[0],
                               'rf_pos_off_center_azi': rf_pos_off_center[1],
                               'rf_pos_lsi': rf_pos_lsi})


        # negative spatial receptive fields
        srf_neg_on, srf_neg_off = strf_dff.get_zscore_receptive_field(timeWindow=params['response_window_negative_rf'])

        # ON negative spatial receptive field
        rf_neg_on_z, rf_neg_on_center, rf_neg_on_area, rf_neg_on_mask = get_rf_properties(srf=srf_neg_on,
                                                                          polarity='negative',
                                                                          sigma=params['gaussian_filter_sigma_rf'],
                                                                          interpolate_rate=params[
                                                                              'interpolate_rate_rf'],
                                                                          z_thr=params['rf_z_threshold'])
        roi_properties.update({'rf_neg_on_peak_z': rf_neg_on_z,
                               'rf_neg_on_area': rf_neg_on_area,
                               'rf_neg_on_center_alt': rf_neg_on_center[0],
                               'rf_neg_on_center_azi': rf_neg_on_center[1]})

        # OFF negative spatial receptive field
        rf_neg_off_z, rf_neg_off_center, rf_neg_off_area, rf_neg_off_mask = get_rf_properties(srf=srf_pos_off,
                                                                             polarity='positive',
                                                                             sigma=params[
                                                                                 'gaussian_filter_sigma_rf'],
                                                                             interpolate_rate=params[
                                                                                 'interpolate_rate_rf'],
                                                                             z_thr=params['rf_z_threshold'])

        # on off overlapping
        rf_neg_lsi = sca.get_local_similarity_index(rf_neg_on_mask, rf_neg_off_mask)

        roi_properties.update({'rf_neg_off_peak_z': rf_neg_off_z,
                               'rf_neg_off_area': rf_neg_off_area,
                               'rf_neg_off_center_alt': rf_neg_off_center[0],
                               'rf_neg_off_center_azi': rf_neg_off_center[1],
                               'rf_neg_lsi': rf_neg_lsi})
    else:
        srf_pos_on = None
        srf_pos_off = None
        srf_neg_on = None
        srf_neg_off = None

        roi_properties.update({'rf_pos_on_peak_z': np.nan,
                               'rf_pos_on_area': np.nan,
                               'rf_pos_on_center_alt': np.nan,
                               'rf_pos_on_center_azi': np.nan,
                               'rf_pos_off_peak_z': np.nan,
                               'rf_pos_off_area': np.nan,
                               'rf_pos_off_center_alt': np.nan,
                               'rf_pos_off_center_azi': np.nan,
                               'rf_pos_lsi': np.nan,
                               'rf_neg_on_peak_z': np.nan,
                               'rf_neg_on_area': np.nan,
                               'rf_neg_on_center_alt': np.nan,
                               'rf_neg_on_center_azi': np.nan,
                               'rf_neg_off_peak_z': np.nan,
                               'rf_neg_off_area': np.nan,
                               'rf_neg_off_center_alt': np.nan,
                               'rf_neg_off_center_azi': np.nan,
                               'rf_neg_lsi': np.nan,
                               })


    # analyze response to drifring grating
    dgcrm = get_dgcrm(nwb_f=nwb_f, plane_n=plane_n, roi_ind=roi_ind, trace_type='sta_' + params['trace_type'])
    if dgcrm is not None:
        dgcrm_grp_key = get_dgcrm_grp_key(nwb_f=nwb_f)
        dgc_block_dur = nwb_f['stimulus/presentation/{}/block_dur'.format(dgcrm_grp_key[15:])].value
        # print('block duration: {}'.format(block_dur))

        # get df statistics ============================================================================================
        _ = dgcrm.get_df_response_table(baseline_win=params['baseline_window_dgc'],
                                        response_win=params['response_window_dgc'])
        dgcrt_df, dgc_p_anova_df, dgc_pos_p_ttest_df, dgc_neg_p_ttest_df = _
        roi_properties.update({'dgc_pos_peak_df': dgcrt_df.peak_response_pos,
                               'dgc_neg_peak_df': dgcrt_df.peak_response_neg,
                               'dgc_pos_p_ttest_df': dgc_pos_p_ttest_df,
                               'dgc_neg_p_ttest_df': dgc_neg_p_ttest_df,
                               'dgc_p_anova_df': dgc_p_anova_df})

        # get dff statics ==============================================================================================
        _ = dgcrm.get_dff_response_table(baseline_win=params['baseline_window_dgc'],
                                         response_win=params['response_window_dgc'],
                                         bias=add_to_trace)
        dgcrt_dff, dgc_p_anova_dff, dgc_pos_p_ttest_dff, dgc_neg_p_ttest_dff = _
        roi_properties.update({'dgc_pos_peak_dff': dgcrt_dff.peak_response_pos,
                               'dgc_neg_peak_dff': dgcrt_dff.peak_response_neg,
                               'dgc_pos_p_ttest_dff': dgc_pos_p_ttest_dff,
                               'dgc_neg_p_ttest_dff': dgc_neg_p_ttest_dff,
                               'dgc_p_anova_dff': dgc_p_anova_dff})

        # get zscore statistics ========================================================================================
        _ = dgcrm.get_zscore_response_table(baseline_win=params['baseline_window_dgc'],
                                            response_win=params['response_window_dgc'])
        dgcrt_z, dgc_p_anova_z, dgc_pos_p_ttest_z, dgc_neg_p_ttest_z = _
        roi_properties.update({'dgc_pos_peak_z': dgcrt_z.peak_response_pos,
                               'dgc_neg_peak_z': dgcrt_z.peak_response_neg,
                               'dgc_pos_p_ttest_z': dgc_pos_p_ttest_z,
                               'dgc_neg_p_ttest_z': dgc_neg_p_ttest_z,
                               'dgc_p_anova_z': dgc_p_anova_z})

        # get dgc response matrices ====================================================================================
        dgcrm_df = dgcrm.get_df_response_matrix(baseline_win=params['baseline_window_dgc'])
        dgcrm_dff = dgcrm.get_dff_response_matrix(baseline_win=params['baseline_window_dgc'],
                                                  bias=params['add_to_trace_bias'])
        dgcrm_z = dgcrm.get_zscore_response_matrix(baseline_win=params['baseline_window_dgc'])


        # direction/orientation tuning of df responses in positive direction ===========================================
        dire_tuning_df_pos = dgcrt_df.get_dire_tuning(response_dir='pos',
                                                      is_collapse_sf=params['is_collapse_sf'],
                                                      is_collapse_tf=params['is_collapse_tf'])
        osi_df_pos, gosi_df_pos, dsi_df_pos, gdsi_df_pos, peak_dire_raw_df_pos, peak_dire_vs_df_pos, \
        peak_orie_vs_df_pos = dgcrt_df.get_dire_tuning_properties(dire_tuning_df_pos,
                                                                  response_dir='pos',
                                                                  is_rectify=params['is_rectify_dgc_tuning'])
        roi_properties.update({'dgc_pos_osi_df': osi_df_pos,
                               'dgc_pos_gosi_df': gosi_df_pos,
                               'dgc_pos_dsi_df': dsi_df_pos,
                               'dgc_pos_gdsi_df': gdsi_df_pos,
                               'dgc_pos_peak_dire_raw_df': peak_dire_raw_df_pos,
                               'dgc_pos_peak_dire_vs_df': peak_dire_vs_df_pos,
                               'dgc_pos_peak_orie_vs_df': peak_orie_vs_df_pos})


        # direction/orientation tuning of df responses in negative direction ===========================================
        dire_tuning_df_neg = dgcrt_df.get_dire_tuning(response_dir='neg',
                                                      is_collapse_sf=params['is_collapse_sf'],
                                                      is_collapse_tf=params['is_collapse_tf'])
        osi_df_neg, gosi_df_neg, dsi_df_neg, gdsi_df_neg, peak_dire_raw_df_neg, peak_dire_vs_df_neg, \
        peak_orie_vs_df_neg = dgcrt_df.get_dire_tuning_properties(dire_tuning_df_neg,
                                                                  response_dir='neg',
                                                                  is_rectify=params['is_rectify_dgc_tuning'])
        roi_properties.update({'dgc_neg_osi_df': osi_df_neg,
                               'dgc_neg_gosi_df': gosi_df_neg,
                               'dgc_neg_dsi_df': dsi_df_neg,
                               'dgc_neg_gdsi_df': gdsi_df_neg,
                               'dgc_neg_peak_dire_raw_df': peak_dire_raw_df_neg,
                               'dgc_neg_peak_dire_vs_df': peak_dire_vs_df_neg,
                               'dgc_neg_peak_orie_vs_df': peak_orie_vs_df_neg})


        # sf tuning of df responses in positive direction ==============================================================
        sf_tuning_df_pos = dgcrt_df.get_sf_tuning(response_dir='pos', is_collapse_tf=params['is_collapse_tf'],
                                                  is_collapse_dire=params['is_collapse_dire'])
        peak_sf_raw_df_pos, peak_sf_linear_df_pos, peak_sf_log_df_pos = \
            dgcrt_df.get_sf_tuning_properties(sf_tuning_df_pos, response_dir='pos',
                                              is_rectify=params['is_rectify_dgc_tuning'])
        roi_properties.update({'dgc_pos_peak_sf_raw_df': peak_sf_raw_df_pos,
                               'dgc_pos_peak_sf_linear_df': peak_sf_linear_df_pos,
                               'dgc_pos_peak_sf_log_df': peak_sf_log_df_pos})


        # sf tuning of df responses in negative direction ==============================================================
        sf_tuning_df_neg = dgcrt_df.get_sf_tuning(response_dir='neg', is_collapse_tf=params['is_collapse_tf'],
                                                  is_collapse_dire=params['is_collapse_dire'])
        peak_sf_raw_df_neg, peak_sf_linear_df_neg, peak_sf_log_df_neg = \
            dgcrt_df.get_sf_tuning_properties(sf_tuning_df_neg, response_dir='neg',
                                              is_rectify=params['is_rectify_dgc_tuning'])
        roi_properties.update({'dgc_neg_peak_sf_raw_df': peak_sf_raw_df_neg,
                               'dgc_neg_peak_sf_linear_df': peak_sf_linear_df_neg,
                               'dgc_neg_peak_sf_log_df': peak_sf_log_df_neg})


        # tf tuning of df responses in positive direction ==============================================================
        tf_tuning_df_pos = dgcrt_df.get_tf_tuning(response_dir='pos', is_collapse_sf=params['is_collapse_sf'],
                                                  is_collapse_dire=params['is_collapse_dire'])
        peak_tf_raw_df_pos, peak_tf_linear_df_pos, peak_tf_log_df_pos = \
            dgcrt_df.get_tf_tuning_properties(tf_tuning_df_pos, response_dir='pos',
                                              is_rectify=params['is_rectify_dgc_tuning'])
        roi_properties.update({'dgc_pos_peak_tf_raw_df': peak_tf_raw_df_pos,
                               'dgc_pos_peak_tf_linear_df': peak_tf_linear_df_pos,
                               'dgc_pos_peak_tf_log_df': peak_tf_log_df_pos})


        # tf tuning of df responses in negative direction ==============================================================
        tf_tuning_df_neg = dgcrt_df.get_tf_tuning(response_dir='neg', is_collapse_sf=params['is_collapse_sf'],
                                                  is_collapse_dire=params['is_collapse_dire'])
        peak_tf_raw_df_neg, peak_tf_linear_df_neg, peak_tf_log_df_neg = \
            dgcrt_df.get_tf_tuning_properties(tf_tuning_df_neg, response_dir='neg',
                                              is_rectify=params['is_rectify_dgc_tuning'])
        roi_properties.update({'dgc_neg_peak_tf_raw_df': peak_tf_raw_df_neg,
                               'dgc_neg_peak_tf_linear_df': peak_tf_linear_df_neg,
                               'dgc_neg_peak_tf_log_df': peak_tf_log_df_neg})

        # direction/orientation tuning of dff responses in positive direction ==========================================
        dire_tuning_dff_pos = dgcrt_dff.get_dire_tuning(response_dir='pos',
                                                      is_collapse_sf=params['is_collapse_sf'],
                                                      is_collapse_tf=params['is_collapse_tf'])
        osi_dff_pos, gosi_dff_pos, dsi_dff_pos, gdsi_dff_pos, peak_dire_raw_dff_pos, peak_dire_vs_dff_pos, \
        peak_orie_vs_dff_pos = dgcrt_dff.get_dire_tuning_properties(dire_tuning_dff_pos,
                                                                  response_dir='pos',
                                                                  is_rectify=params['is_rectify_dgc_tuning'])
        roi_properties.update({'dgc_pos_osi_dff': osi_dff_pos,
                               'dgc_pos_gosi_dff': gosi_dff_pos,
                               'dgc_pos_dsi_dff': dsi_dff_pos,
                               'dgc_pos_gdsi_dff': gdsi_dff_pos,
                               'dgc_pos_peak_dire_raw_dff': peak_dire_raw_dff_pos,
                               'dgc_pos_peak_dire_vs_dff': peak_dire_vs_dff_pos,
                               'dgc_pos_peak_orie_vs_dff': peak_orie_vs_dff_pos})

        # direction/orientation tuning of dff responses in negative direction ==========================================
        dire_tuning_dff_neg = dgcrt_dff.get_dire_tuning(response_dir='neg',
                                                      is_collapse_sf=params['is_collapse_sf'],
                                                      is_collapse_tf=params['is_collapse_tf'])
        osi_dff_neg, gosi_dff_neg, dsi_dff_neg, gdsi_dff_neg, peak_dire_raw_dff_neg, peak_dire_vs_dff_neg, \
        peak_orie_vs_dff_neg = dgcrt_dff.get_dire_tuning_properties(dire_tuning_dff_neg,
                                                                  response_dir='neg',
                                                                  is_rectify=params['is_rectify_dgc_tuning'])
        roi_properties.update({'dgc_neg_osi_dff': osi_dff_neg,
                               'dgc_neg_gosi_dff': gosi_dff_neg,
                               'dgc_neg_dsi_dff': dsi_dff_neg,
                               'dgc_neg_gdsi_dff': gdsi_dff_neg,
                               'dgc_neg_peak_dire_raw_dff': peak_dire_raw_dff_neg,
                               'dgc_neg_peak_dire_vs_dff': peak_dire_vs_dff_neg,
                               'dgc_neg_peak_orie_vs_dff': peak_orie_vs_dff_neg})

        # sf tuning of dff responses in positive direction =============================================================
        sf_tuning_dff_pos = dgcrt_dff.get_sf_tuning(response_dir='pos', is_collapse_tf=params['is_collapse_tf'],
                                                  is_collapse_dire=params['is_collapse_dire'])
        peak_sf_raw_dff_pos, peak_sf_linear_dff_pos, peak_sf_log_dff_pos = \
            dgcrt_dff.get_sf_tuning_properties(sf_tuning_dff_pos, response_dir='pos',
                                              is_rectify=params['is_rectify_dgc_tuning'])
        roi_properties.update({'dgc_pos_peak_sf_raw_dff': peak_sf_raw_dff_pos,
                               'dgc_pos_peak_sf_linear_dff': peak_sf_linear_dff_pos,
                               'dgc_pos_peak_sf_log_dff': peak_sf_log_dff_pos})

        # sf tuning of dff responses in negative direction =============================================================
        sf_tuning_dff_neg = dgcrt_dff.get_sf_tuning(response_dir='neg', is_collapse_tf=params['is_collapse_tf'],
                                                  is_collapse_dire=params['is_collapse_dire'])
        peak_sf_raw_dff_neg, peak_sf_linear_dff_neg, peak_sf_log_dff_neg = \
            dgcrt_dff.get_sf_tuning_properties(sf_tuning_dff_neg, response_dir='neg',
                                              is_rectify=params['is_rectify_dgc_tuning'])
        roi_properties.update({'dgc_neg_peak_sf_raw_dff': peak_sf_raw_dff_neg,
                               'dgc_neg_peak_sf_linear_dff': peak_sf_linear_dff_neg,
                               'dgc_neg_peak_sf_log_dff': peak_sf_log_dff_neg})

        # tf tuning of dff responses in positive direction =============================================================
        tf_tuning_dff_pos = dgcrt_dff.get_tf_tuning(response_dir='pos', is_collapse_sf=params['is_collapse_sf'],
                                                  is_collapse_dire=params['is_collapse_dire'])
        peak_tf_raw_dff_pos, peak_tf_linear_dff_pos, peak_tf_log_dff_pos = \
            dgcrt_dff.get_tf_tuning_properties(tf_tuning_dff_pos, response_dir='pos',
                                              is_rectify=params['is_rectify_dgc_tuning'])
        roi_properties.update({'dgc_pos_peak_tf_raw_dff': peak_tf_raw_dff_pos,
                               'dgc_pos_peak_tf_linear_dff': peak_tf_linear_dff_pos,
                               'dgc_pos_peak_tf_log_dff': peak_tf_log_dff_pos})

        # tf tuning of dff responses in negative direction =============================================================
        tf_tuning_dff_neg = dgcrt_dff.get_tf_tuning(response_dir='neg', is_collapse_sf=params['is_collapse_sf'],
                                                  is_collapse_dire=params['is_collapse_dire'])
        peak_tf_raw_dff_neg, peak_tf_linear_dff_neg, peak_tf_log_dff_neg = \
            dgcrt_dff.get_tf_tuning_properties(tf_tuning_dff_neg, response_dir='neg',
                                              is_rectify=params['is_rectify_dgc_tuning'])
        roi_properties.update({'dgc_neg_peak_tf_raw_dff': peak_tf_raw_dff_neg,
                               'dgc_neg_peak_tf_linear_dff': peak_tf_linear_dff_neg,
                               'dgc_neg_peak_tf_log_dff': peak_tf_log_dff_neg})

        # direction/orientation tuning of zscore responses in positive direction =======================================
        dire_tuning_z_pos = dgcrt_z.get_dire_tuning(response_dir='pos',
                                                        is_collapse_sf=params['is_collapse_sf'],
                                                        is_collapse_tf=params['is_collapse_tf'])
        osi_z_pos, gosi_z_pos, dsi_z_pos, gdsi_z_pos, peak_dire_raw_z_pos, peak_dire_vs_z_pos, \
        peak_orie_vs_z_pos = dgcrt_z.get_dire_tuning_properties(dire_tuning_z_pos,
                                                                    response_dir='pos',
                                                                    is_rectify=params['is_rectify_dgc_tuning'])
        roi_properties.update({'dgc_pos_osi_z': osi_z_pos,
                               'dgc_pos_gosi_z': gosi_z_pos,
                               'dgc_pos_dsi_z': dsi_z_pos,
                               'dgc_pos_gdsi_z': gdsi_z_pos,
                               'dgc_pos_peak_dire_raw_z': peak_dire_raw_z_pos,
                               'dgc_pos_peak_dire_vs_z': peak_dire_vs_z_pos,
                               'dgc_pos_peak_orie_vs_z': peak_orie_vs_z_pos})

        # direction/orientation tuning of zscore responses in negative direction =======================================
        dire_tuning_z_neg = dgcrt_z.get_dire_tuning(response_dir='neg',
                                                        is_collapse_sf=params['is_collapse_sf'],
                                                        is_collapse_tf=params['is_collapse_tf'])
        osi_z_neg, gosi_z_neg, dsi_z_neg, gdsi_z_neg, peak_dire_raw_z_neg, peak_dire_vs_z_neg, \
        peak_orie_vs_z_neg = dgcrt_z.get_dire_tuning_properties(dire_tuning_z_neg,
                                                                    response_dir='neg',
                                                                    is_rectify=params['is_rectify_dgc_tuning'])
        roi_properties.update({'dgc_neg_osi_z': osi_z_neg,
                               'dgc_neg_gosi_z': gosi_z_neg,
                               'dgc_neg_dsi_z': dsi_z_neg,
                               'dgc_neg_gdsi_z': gdsi_z_neg,
                               'dgc_neg_peak_dire_raw_z': peak_dire_raw_z_neg,
                               'dgc_neg_peak_dire_vs_z': peak_dire_vs_z_neg,
                               'dgc_neg_peak_orie_vs_z': peak_orie_vs_z_neg})

        # sf tuning of zscore responses in positive direction ==========================================================
        sf_tuning_z_pos = dgcrt_z.get_sf_tuning(response_dir='pos', is_collapse_tf=params['is_collapse_tf'],
                                                    is_collapse_dire=params['is_collapse_dire'])
        peak_sf_raw_z_pos, peak_sf_linear_z_pos, peak_sf_log_z_pos = \
            dgcrt_z.get_sf_tuning_properties(sf_tuning_z_pos, response_dir='pos',
                                               is_rectify=params['is_rectify_dgc_tuning'])
        roi_properties.update({'dgc_pos_peak_sf_raw_z': peak_sf_raw_z_pos,
                               'dgc_pos_peak_sf_linear_z': peak_sf_linear_z_pos,
                               'dgc_pos_peak_sf_log_z': peak_sf_log_z_pos})

        # sf tuning of zscore responses in negative direction ==========================================================
        sf_tuning_z_neg = dgcrt_z.get_sf_tuning(response_dir='neg', is_collapse_tf=params['is_collapse_tf'],
                                                    is_collapse_dire=params['is_collapse_dire'])
        peak_sf_raw_z_neg, peak_sf_linear_z_neg, peak_sf_log_z_neg = \
            dgcrt_z.get_sf_tuning_properties(sf_tuning_z_neg, response_dir='neg',
                                               is_rectify=params['is_rectify_dgc_tuning'])
        roi_properties.update({'dgc_neg_peak_sf_raw_z': peak_sf_raw_z_neg,
                               'dgc_neg_peak_sf_linear_z': peak_sf_linear_z_neg,
                               'dgc_neg_peak_sf_log_z': peak_sf_log_z_neg})

        # tf tuning of zscore responses in positive direction ==========================================================
        tf_tuning_z_pos = dgcrt_z.get_tf_tuning(response_dir='pos', is_collapse_sf=params['is_collapse_sf'],
                                                    is_collapse_dire=params['is_collapse_dire'])
        peak_tf_raw_z_pos, peak_tf_linear_z_pos, peak_tf_log_z_pos = \
            dgcrt_z.get_tf_tuning_properties(tf_tuning_z_pos, response_dir='pos',
                                               is_rectify=params['is_rectify_dgc_tuning'])
        roi_properties.update({'dgc_pos_peak_tf_raw_z': peak_tf_raw_z_pos,
                               'dgc_pos_peak_tf_linear_z': peak_tf_linear_z_pos,
                               'dgc_pos_peak_tf_log_z': peak_tf_log_z_pos})

        # tf tuning of zscore responses in negative direction ==========================================================
        tf_tuning_z_neg = dgcrt_z.get_tf_tuning(response_dir='neg', is_collapse_sf=params['is_collapse_sf'],
                                                    is_collapse_dire=params['is_collapse_dire'])
        peak_tf_raw_z_neg, peak_tf_linear_z_neg, peak_tf_log_z_neg = \
            dgcrt_z.get_tf_tuning_properties(tf_tuning_z_neg, response_dir='neg',
                                               is_rectify=params['is_rectify_dgc_tuning'])
        roi_properties.update({'dgc_neg_peak_tf_raw_z': peak_tf_raw_z_neg,
                               'dgc_neg_peak_tf_linear_z': peak_tf_linear_z_neg,
                               'dgc_neg_peak_tf_log_z': peak_tf_log_z_neg})

    else:
        dgcrm_df = None
        dgcrm_dff = None
        dgcrm_z = None
        dgcrt_df = None
        dgcrt_dff = None
        dgcrt_z = None
        dgc_block_dur = None

        roi_properties.update({'dgc_pos_peak_df': np.nan,
                               'dgc_neg_peak_df': np.nan,
                               'dgc_pos_p_ttest_df': np.nan,
                               'dgc_neg_p_ttest_df': np.nan,
                               'dgc_p_anova_df': np.nan,
                               'dgc_pos_peak_dff': np.nan,
                               'dgc_neg_peak_dff': np.nan,
                               'dgc_pos_p_ttest_dff': np.nan,
                               'dgc_neg_p_ttest_dff': np.nan,
                               'dgc_p_anova_dff': np.nan,
                               'dgc_pos_peak_z': np.nan,
                               'dgc_neg_peak_z': np.nan,
                               'dgc_pos_p_ttest_z': np.nan,
                               'dgc_neg_p_ttest_z': np.nan,
                               'dgc_p_anova_z': np.nan,
                               'dgc_pos_osi_df': np.nan,
                               'dgc_pos_gosi_df': np.nan,
                               'dgc_pos_dsi_df': np.nan,
                               'dgc_pos_gdsi_df': np.nan,
                               'dgc_pos_peak_dire_raw_df': np.nan,
                               'dgc_pos_peak_dire_vs_df': np.nan,
                               'dgc_pos_peak_orie_vs_df': np.nan,
                               'dgc_neg_osi_df': np.nan,
                               'dgc_neg_gosi_df': np.nan,
                               'dgc_neg_dsi_df': np.nan,
                               'dgc_neg_gdsi_df': np.nan,
                               'dgc_neg_peak_dire_raw_df': np.nan,
                               'dgc_neg_peak_dire_vs_df': np.nan,
                               'dgc_neg_peak_orie_vs_df': np.nan,
                               'dgc_pos_peak_sf_raw_df': np.nan,
                               'dgc_pos_peak_sf_linear_df': np.nan,
                               'dgc_pos_peak_sf_log_df': np.nan,
                               'dgc_neg_peak_sf_raw_df': np.nan,
                               'dgc_neg_peak_sf_linear_df': np.nan,
                               'dgc_neg_peak_sf_log_df': np.nan,
                               'dgc_pos_peak_tf_raw_df': np.nan,
                               'dgc_pos_peak_tf_linear_df': np.nan,
                               'dgc_pos_peak_tf_log_df': np.nan,
                               'dgc_neg_peak_tf_raw_df': np.nan,
                               'dgc_neg_peak_tf_linear_df': np.nan,
                               'dgc_neg_peak_tf_log_df': np.nan,
                               'dgc_pos_osi_dff': np.nan,
                               'dgc_pos_gosi_dff': np.nan,
                               'dgc_pos_dsi_dff': np.nan,
                               'dgc_pos_gdsi_dff': np.nan,
                               'dgc_pos_peak_dire_raw_dff': np.nan,
                               'dgc_pos_peak_dire_vs_dff': np.nan,
                               'dgc_pos_peak_orie_vs_dff': np.nan,
                               'dgc_neg_osi_dff': np.nan,
                               'dgc_neg_gosi_dff': np.nan,
                               'dgc_neg_dsi_dff': np.nan,
                               'dgc_neg_gdsi_dff': np.nan,
                               'dgc_neg_peak_dire_raw_dff': np.nan,
                               'dgc_neg_peak_dire_vs_dff': np.nan,
                               'dgc_neg_peak_orie_vs_dff': np.nan,
                               'dgc_pos_peak_sf_raw_dff': np.nan,
                               'dgc_pos_peak_sf_linear_dff': np.nan,
                               'dgc_pos_peak_sf_log_dff': np.nan,
                               'dgc_neg_peak_sf_raw_dff': np.nan,
                               'dgc_neg_peak_sf_linear_dff': np.nan,
                               'dgc_neg_peak_sf_log_dff': np.nan,
                               'dgc_pos_peak_tf_raw_dff': np.nan,
                               'dgc_pos_peak_tf_linear_dff': np.nan,
                               'dgc_pos_peak_tf_log_dff': np.nan,
                               'dgc_neg_peak_tf_raw_dff': np.nan,
                               'dgc_neg_peak_tf_linear_dff': np.nan,
                               'dgc_neg_peak_tf_log_dff': np.nan,
                               'dgc_pos_osi_z': np.nan,
                               'dgc_pos_gosi_z': np.nan,
                               'dgc_pos_dsi_z': np.nan,
                               'dgc_pos_gdsi_z': np.nan,
                               'dgc_pos_peak_dire_raw_z': np.nan,
                               'dgc_pos_peak_dire_vs_z': np.nan,
                               'dgc_pos_peak_orie_vs_z': np.nan,
                               'dgc_neg_osi_z': np.nan,
                               'dgc_neg_gosi_z': np.nan,
                               'dgc_neg_dsi_z': np.nan,
                               'dgc_neg_gdsi_z': np.nan,
                               'dgc_neg_peak_dire_raw_z': np.nan,
                               'dgc_neg_peak_dire_vs_z': np.nan,
                               'dgc_neg_peak_orie_vs_z': np.nan,
                               'dgc_pos_peak_sf_raw_z': np.nan,
                               'dgc_pos_peak_sf_linear_z': np.nan,
                               'dgc_pos_peak_sf_log_z': np.nan,
                               'dgc_neg_peak_sf_raw_z': np.nan,
                               'dgc_neg_peak_sf_linear_z': np.nan,
                               'dgc_neg_peak_sf_log_z': np.nan,
                               'dgc_pos_peak_tf_raw_z': np.nan,
                               'dgc_pos_peak_tf_linear_z': np.nan,
                               'dgc_pos_peak_tf_log_z': np.nan,
                               'dgc_neg_peak_tf_raw_z': np.nan,
                               'dgc_neg_peak_tf_linear_z': np.nan,
                               'dgc_neg_peak_tf_log_z': np.nan,
                               })


    return roi_properties, roi, trace, srf_pos_on, srf_pos_off, srf_neg_on, srf_neg_off, dgcrm_df, dgcrm_dff, \
           dgcrm_z, dgcrt_df, dgcrt_dff, dgcrt_z, dgc_block_dur


def roi_page_report(nwb_f, plane_n, roi_n, params=ANALYSIS_PARAMS, plot_params=PLOTTING_PARAMS):
    """
    generate a page of description of an roi

    :param nwb_f: h5py.File object
    :param plane_n:
    :param roi_n:
    :param params:
    :return:
    """

    roi_ind = int(roi_n[-4:])

    roi_properties, roi, trace, srf_pos_on, srf_pos_off, srf_neg_on, srf_neg_off, dgcrm_df, dgcrm_dff, \
    dgcrm_z, dgcrt_df, dgcrt_dff, dgcrt_z, dgc_block_dur = get_everything_from_roi(nwb_f=nwb_f,
                                                                                   plane_n=plane_n,
                                                                                   roi_n=roi_n,
                                                                                   params=params)

    segmentation_grp = nwb_f['processing/rois_and_traces_{}/ImageSegmentation/imaging_plane'.format(plane_n)]
    rf_img_grp = segmentation_grp['reference_images']
    if 'mean_projection' in rf_img_grp.keys():
        rf_img = rf_img_grp['mean_projection/data'].value
    else:
        rf_img = rf_img_grp['max_projection/data'].value

    f = plt.figure(figsize=plot_params['fig_size'], facecolor=plot_params['fig_facecolor'])

    # plot roi mask
    f.subplots_adjust(0, 0, 1, 1)
    ax_roi_img = f.add_axes(plot_params['ax_roi_img_coord'])
    ax_roi_img.imshow(ia.array_nor(rf_img), cmap='gray', vmin=plot_params['rf_img_vmin'],
                      vmax=plot_params['rf_img_vmax'], interpolation='nearest')
    pt.plot_mask_borders(mask=roi.get_binary_mask(), plotAxis=ax_roi_img, color=plot_params['roi_border_color'],
                         borderWidth=plot_params['roi_border_width'])
    ax_roi_img.set_axis_off()

    # plot traces
    trace_chunk_length = trace.shape[0] // plot_params['traces_panels']
    trace_max = np.max(trace)
    trace_min = np.min(trace)

    trace_axis_height = (plot_params['field_traces_coord'][3] - (0.01 * (plot_params['traces_panels'] - 1))) \
                        / plot_params['traces_panels']
    for trace_i in range(plot_params['traces_panels']):
        curr_trace_axis = f.add_axes([
            plot_params['field_traces_coord'][0],
            plot_params['field_traces_coord'][1] + trace_i * (0.01 + trace_axis_height),
            plot_params['field_traces_coord'][2],
            trace_axis_height
        ])
        curr_trace_chunk = trace[trace_i * trace_chunk_length: (trace_i + 1) * trace_chunk_length]
        curr_trace_axis.plot(curr_trace_chunk, color=plot_params['traces_color'],
                             lw=plot_params['traces_line_width'])
        curr_trace_axis.set_xlim([0, trace_chunk_length])
        curr_trace_axis.set_ylim([trace_min, trace_max])
        curr_trace_axis.set_axis_off()

    # plot receptive field
    if srf_pos_on is not None:
        ax_rf_pos = f.add_axes(plot_params['ax_rf_pos_coord'])
        zscore_pos = render_rb(rf_on=srf_pos_on.get_weighted_mask(),
                               rf_off=srf_pos_off.get_weighted_mask(), vmax=plot_params['rf_zscore_vmax'])
        ax_rf_pos.imshow(zscore_pos, interpolation='nearest')
        ax_rf_pos.set_axis_off()

        # plotting negative ON and OFF receptive fields
        ax_rf_neg = f.add_axes(plot_params['ax_rf_neg_coord'])
        zscore_neg = render_rb(rf_on=-srf_neg_on.get_weighted_mask(),
                               rf_off=-srf_neg_off.get_weighted_mask(), vmax=plot_params['rf_zscore_vmax'])
        ax_rf_neg.imshow(zscore_neg, interpolation='nearest')
        ax_rf_neg.set_axis_off()

    # select dgc response matrix and response table for plotting
    if plot_params['response_type_for_plot'] == 'df':
        dgcrm_plot = dgcrm_df
        dgcrt_plot = dgcrt_df
    elif plot_params['response_type_for_plot'] == 'dff':
        dgcrm_plot = dgcrm_dff
        dgcrt_plot = dgcrt_dff
    elif plot_params['response_type_for_plot'] == 'zscore':
        dgcrm_plot = dgcrm_z
        dgcrt_plot = dgcrt_z
    else:
        raise LookupError("Do not understand 'response_type_for_plot': {}. Should be "
                          "'df', 'dff' or 'zscore'.".format(params['response_type_for_plot']))

    if dgcrm_plot is not None:

        # plot peak condition traces
        ax_peak_traces_pos = f.add_axes(plot_params['ax_peak_traces_pos_coord'])
        ax_peak_traces_neg = f.add_axes(plot_params['ax_peak_traces_neg_coord'])

        ymin_pos, ymax_pos = dgcrm_plot.plot_traces(condi_ind=dgcrt_plot.peak_condi_ind_pos,
                                                    axis=ax_peak_traces_pos,
                                                    blank_ind=dgcrt_plot.blank_condi_ind,
                                                    block_dur=dgc_block_dur,
                                                    response_window=params['response_window_dgc'],
                                                    baseline_window=params['baseline_window_dgc'],
                                                    trace_color=plot_params['peak_traces_pos_color'],
                                                    block_face_color=plot_params['block_face_color'],
                                                    response_window_color=plot_params['response_window_color'],
                                                    baseline_window_color=plot_params['baseline_window_color'],
                                                    blank_trace_color=plot_params['blank_traces_color'],
                                                    lw_single=plot_params['single_traces_lw'],
                                                    lw_mean=plot_params['mean_traces_lw'])

        ymin_neg, ymax_neg = dgcrm_plot.plot_traces(condi_ind=dgcrt_plot.peak_condi_ind_neg,
                                                    axis=ax_peak_traces_neg,
                                                    blank_ind=dgcrt_plot.blank_condi_ind,
                                                    block_dur=dgc_block_dur,
                                                    response_window=params['response_window_dgc'],
                                                    baseline_window=params['baseline_window_dgc'],
                                                    trace_color=plot_params['peak_traces_neg_color'],
                                                    block_face_color=plot_params['block_face_color'],
                                                    response_window_color=plot_params['response_window_color'],
                                                    baseline_window_color=plot_params['baseline_window_color'],
                                                    blank_trace_color=plot_params['blank_traces_color'],
                                                    lw_single=plot_params['single_traces_lw'],
                                                    lw_mean=plot_params['mean_traces_lw'])

        ax_peak_traces_pos.set_ylim(min([ymin_pos, ymin_neg]), max([ymax_pos, ymax_neg]))
        ax_peak_traces_neg.set_ylim(min([ymin_pos, ymin_neg]), max([ymax_pos, ymax_neg]))
        ax_peak_traces_pos.set_xticks([])
        ax_peak_traces_pos.set_yticks([])
        ax_peak_traces_neg.set_xticks([])
        ax_peak_traces_neg.set_yticks([])

        # plot sf-tf matrix
        ax_sftf_pos = f.add_axes(plot_params['ax_sftf_pos_coord'])
        ax_sftf_neg = f.add_axes(plot_params['ax_sftf_neg_coord'])

        dgcrt_plot.plot_sf_tf_matrix(response_dir='pos',
                                     axis=ax_sftf_pos,
                                     cmap=plot_params['sftf_cmap'],
                                     vmax=plot_params['sftf_vmax'],
                                     vmin=plot_params['sftf_vmin'])
        dgcrt_plot.plot_sf_tf_matrix(response_dir='neg',
                                     axis=ax_sftf_neg,
                                     cmap=plot_params['sftf_cmap'],
                                     vmax=plot_params['sftf_vmax'],
                                     vmin=plot_params['sftf_vmin'])

        # plot direction tuning curve
        ax_dire_pos = f.add_axes(plot_params['ax_dire_pos_coord'], projection='polar')
        ax_dire_neg = f.add_axes(plot_params['ax_dire_neg_coord'], projection='polar')

        r_max_pos = dgcrt_plot.plot_dire_tuning(response_dir='pos', axis=ax_dire_pos,
                                                is_collapse_sf=params['is_collapse_sf'],
                                                is_collapse_tf=params['is_collapse_tf'],
                                                trace_color=plot_params['dire_color_pos'],
                                                lw=plot_params['dire_line_width'],
                                                is_rectify=params['is_rectify_dgc_tuning'])

        r_max_neg = dgcrt_plot.plot_dire_tuning(response_dir='neg', axis=ax_dire_neg,
                                                is_collapse_sf=params['is_collapse_sf'],
                                                is_collapse_tf=params['is_collapse_tf'],
                                                trace_color=plot_params['dire_color_neg'],
                                                lw=plot_params['dire_line_width'],
                                                is_rectify=params['is_rectify_dgc_tuning'])

        rmax = max([r_max_pos, r_max_neg])

        ax_dire_pos.set_rlim([0, rmax])
        ax_dire_pos.set_rticks([rmax])
        ax_dire_neg.set_rlim([0, rmax])
        ax_dire_neg.set_rticks([rmax])

    # print text
    ax_text = f.add_axes(plot_params['ax_text_coord'])
    ax_text.set_xticks([])
    ax_text.set_yticks([])

    file_n = os.path.splitext(os.path.split(nwb_f.filename)[1])[0]

    txt = '\n'
    txt += 'nwb: {}\n'.format(file_n)
    txt += '\n'
    txt += 'plane name:          {}\n'.format(plane_n)
    txt += 'roi name:            {}\n'.format(roi_n)
    txt += '\n'
    txt += 'depth (um):          {}\n'.format(roi_properties['depth'])
    txt += 'roi area (um^2):     {:.2f}\n'.format(roi_properties['roi_area'])
    txt += '\n'
    txt += 'trace type:{:>19}\n'.format(params['trace_type'])
    txt += 'response type:{:>14}\n'.format(plot_params['response_type_for_plot'])
    txt += '\n'
    txt += 'skewness raw:        {:.2f}\n'.format(roi_properties['skew_raw'])
    txt += 'skewness fil:        {:.2f}\n'.format(roi_properties['skew_fil'])
    txt += '\n'

    rf_pos_peak_z = max([roi_properties['rf_pos_on_peak_z'],
                         roi_properties['rf_pos_off_peak_z']])

    txt += 'rf_pos_peak_z:       {:.2f}\n'.format(rf_pos_peak_z)
    txt += 'rf_pos_lsi:          {:.2f}\n'.format(roi_properties['rf_pos_lsi'])
    txt += '\n'

    rf_neg_peak_z = max([roi_properties['rf_neg_on_peak_z'],
                         roi_properties['rf_neg_off_peak_z']])

    txt += 'rf_neg_peak_z:       {:.2f}\n'.format(rf_neg_peak_z)
    txt += 'rf_neg_lsi:          {:.2f}\n'.format(roi_properties['rf_neg_lsi'])
    txt += '\n'

    if plot_params['response_type_for_plot'] == 'df':
        txt += 'dgc_p_anova:         {:.2f}\n'.format(roi_properties['dgc_p_anova_df'])
        txt += '\n'
        txt += 'dgc_pos_p_ttest:     {:.2f}\n'.format(roi_properties['dgc_pos_p_ttest_df'])
        txt += 'dgc_pos_peak_resp:   {:.2f}\n'.format(roi_properties['dgc_pos_peak_df'])
        txt += 'dgc_pos_OSI:         {:.2f}\n'.format(roi_properties['dgc_pos_osi_df'])
        txt += 'dgc_pos_gOSI:        {:.2f}\n'.format(roi_properties['dgc_pos_gosi_df'])
        txt += 'dgc_pos_DSI:         {:.2f}\n'.format(roi_properties['dgc_pos_dsi_df'])
        txt += 'dgc_pos_gDSI:        {:.2f}\n'.format(roi_properties['dgc_pos_gdsi_df'])
        txt += 'dgc_pos_peak_sf:     {:.2f}\n'.format(roi_properties['dgc_pos_peak_sf_log_df'])
        txt += 'dgc_pos_peak_tf:     {:.2f}\n'.format(roi_properties['dgc_pos_peak_tf_log_df'])
        txt += '\n'
        txt += 'dgc_neg_p_ttest:     {:.2f}\n'.format(roi_properties['dgc_neg_p_ttest_df'])
        txt += 'dgc_neg_peak_resp:   {:.2f}\n'.format(roi_properties['dgc_neg_peak_df'])
        txt += 'dgc_neg_OSI:         {:.2f}\n'.format(roi_properties['dgc_neg_osi_df'])
        txt += 'dgc_neg_gOSI:        {:.2f}\n'.format(roi_properties['dgc_neg_gosi_df'])
        txt += 'dgc_neg_DSI:         {:.2f}\n'.format(roi_properties['dgc_neg_dsi_df'])
        txt += 'dgc_neg_gDSI:        {:.2f}\n'.format(roi_properties['dgc_neg_gdsi_df'])
        txt += 'dgc_neg_peak_sf:     {:.2f}\n'.format(roi_properties['dgc_neg_peak_sf_log_df'])
        txt += 'dgc_neg_peak_tf:     {:.2f}\n'.format(roi_properties['dgc_neg_peak_tf_log_df'])

    elif plot_params['response_type_for_plot'] == 'dff':
        txt += 'dgc_p_anova:         {:.2f}\n'.format(roi_properties['dgc_p_anova_dff'])
        txt += '\n'
        txt += 'dgc_pos_p_ttest:     {:.2f}\n'.format(roi_properties['dgc_pos_p_ttest_dff'])
        txt += 'dgc_pos_peak_resp:   {:.2f}\n'.format(roi_properties['dgc_pos_peak_dff'])
        txt += 'dgc_pos_OSI:         {:.2f}\n'.format(roi_properties['dgc_pos_osi_dff'])
        txt += 'dgc_pos_gOSI:        {:.2f}\n'.format(roi_properties['dgc_pos_gosi_dff'])
        txt += 'dgc_pos_DSI:         {:.2f}\n'.format(roi_properties['dgc_pos_dsi_dff'])
        txt += 'dgc_pos_gDSI:        {:.2f}\n'.format(roi_properties['dgc_pos_gdsi_dff'])
        txt += 'dgc_pos_peak_sf:     {:.2f}\n'.format(roi_properties['dgc_pos_peak_sf_log_dff'])
        txt += 'dgc_pos_peak_tf:     {:.2f}\n'.format(roi_properties['dgc_pos_peak_tf_log_dff'])
        txt += '\n'
        txt += 'dgc_neg_p_ttest:     {:.2f}\n'.format(roi_properties['dgc_neg_p_ttest_dff'])
        txt += 'dgc_neg_peak_resp:   {:.2f}\n'.format(roi_properties['dgc_neg_peak_dff'])
        txt += 'dgc_neg_OSI:         {:.2f}\n'.format(roi_properties['dgc_neg_osi_dff'])
        txt += 'dgc_neg_gOSI:        {:.2f}\n'.format(roi_properties['dgc_neg_gosi_dff'])
        txt += 'dgc_neg_DSI:         {:.2f}\n'.format(roi_properties['dgc_neg_dsi_dff'])
        txt += 'dgc_neg_gDSI:        {:.2f}\n'.format(roi_properties['dgc_neg_gdsi_dff'])
        txt += 'dgc_neg_peak_sf:     {:.2f}\n'.format(roi_properties['dgc_neg_peak_sf_log_dff'])
        txt += 'dgc_neg_peak_tf:     {:.2f}\n'.format(roi_properties['dgc_neg_peak_tf_log_dff'])

    elif plot_params['response_type_for_plot'] == 'zscore':
        txt += 'dgc_p_anova:         {:.2f}\n'.format(roi_properties['dgc_p_anova_z'])
        txt += '\n'
        txt += 'dgc_pos_p_ttest:     {:.2f}\n'.format(roi_properties['dgc_pos_p_ttest_z'])
        txt += 'dgc_pos_peak_resp:   {:.2f}\n'.format(roi_properties['dgc_pos_peak_z'])
        txt += 'dgc_pos_OSI:         {:.2f}\n'.format(roi_properties['dgc_pos_osi_z'])
        txt += 'dgc_pos_gOSI:        {:.2f}\n'.format(roi_properties['dgc_pos_gosi_z'])
        txt += 'dgc_pos_DSI:         {:.2f}\n'.format(roi_properties['dgc_pos_dsi_z'])
        txt += 'dgc_pos_gDSI:        {:.2f}\n'.format(roi_properties['dgc_pos_gdsi_z'])
        txt += 'dgc_pos_peak_sf:     {:.2f}\n'.format(roi_properties['dgc_pos_peak_sf_log_z'])
        txt += 'dgc_pos_peak_tf:     {:.2f}\n'.format(roi_properties['dgc_pos_peak_tf_log_z'])
        txt += '\n'
        txt += 'dgc_neg_p_ttest:     {:.2f}\n'.format(roi_properties['dgc_neg_p_ttest_z'])
        txt += 'dgc_neg_peak_resp:   {:.2f}\n'.format(roi_properties['dgc_neg_peak_z'])
        txt += 'dgc_neg_OSI:         {:.2f}\n'.format(roi_properties['dgc_neg_osi_z'])
        txt += 'dgc_neg_gOSI:        {:.2f}\n'.format(roi_properties['dgc_neg_gosi_z'])
        txt += 'dgc_neg_DSI:         {:.2f}\n'.format(roi_properties['dgc_neg_dsi_z'])
        txt += 'dgc_neg_gDSI:        {:.2f}\n'.format(roi_properties['dgc_neg_gdsi_z'])
        txt += 'dgc_neg_peak_sf:     {:.2f}\n'.format(roi_properties['dgc_neg_peak_sf_log_z'])
        txt += 'dgc_neg_peak_tf:     {:.2f}\n'.format(roi_properties['dgc_neg_peak_tf_log_z'])
    else:
        raise LookupError("Do not understand 'response_type_for_plot': {}. Should be "
                          "'df', 'dff' or 'zscore'.".format(params['response_type_for_plot']))

    ax_text.text(0.01, 0.99, txt, horizontalalignment='left', verticalalignment='top', family='monospace')

    # plt.show()
    return f


if __name__ == '__main__':

    nwb_path = r"F:\data2\chandelier_cell_project\database\nwbs\190326_M441626_110.nwb"
    plane_n = 'plane0'
    roi_n = 'roi_0000'
    nwb_f = h5py.File(nwb_path, 'r')

    roi_properties, _, _, _, _, _, _, _, _, _, _, _, _, _ = \
        get_everything_from_roi(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n)

    keys = roi_properties.keys()
    keys.sort()
    for key in keys:
        print('{}: {}'.format(key, roi_properties[key]))

    roi_page_report(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n)

    nwb_f.close()
    plt.show()
