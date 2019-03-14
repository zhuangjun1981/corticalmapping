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
    'response_window_dgc': [0., 1.], # list of two floats, temporal window for getting response value for drifting grating
    'baseline_window_dgc': [-0.5, 0.] # list of two floats, temporal window for getting baseline value for drifting grating
                   }

PLOTTING_PARAMS = {
    'fig_size': (8.5, 11),
    'fig_facecolor': "#ffffff",
    'ax_roi_img_coord': [0.01, 0.75, 0.3, 0.24], # coordinates of roi image
    'rf_img_vmin': 0., # reference image min
    'rf_img_vmax': 0.5, # reference image max
    'roi_border_color': '#ff0000',
    'roi_border_width': 1,
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
    'single_traces_lw': 0.5,
    'mean_traces_mean_lw': 2.,
    'ax_text_coord': [0.63, 0.01, 0.36, 0.73]

}


def has_strf(nwb_f, plane_n):
    return 'analysis/STRFs/{}'.format(plane_n) in nwb_f


def get_dgcrt_grp_key(nwb_f):
    analysis_grp = nwb_f['analysis']
    dgcrt_key = [k for k in analysis_grp.keys() if k[0:14] == 'response_table' and 'DriftingGrating' in k]
    if len(dgcrt_key) == 0:
        return None
    elif len(dgcrt_key) == 1:
        return dgcrt_key[0]
    else:
        raise LookupError('more than one drifting grating response table found.')


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


def get_strf(nwb_f, plane_n, roi_n):
    strf_grp = nwb_f['analysis/STRFs/{}/strf_{}'.format(plane_n, roi_n)]
    strf = sca.SpatialTemporalReceptiveField.from_h5_group(strf_grp)
    return strf


def get_pos_neg_zscore_maps(strf, pos_window=ANALYSIS_PARAMS['response_window_positive_rf'],
                            neg_window=ANALYSIS_PARAMS['response_window_negative_rf']):
    """

    :param strf:
    :param pos_window:
    :param neg_window:
    :return zscore_pos: ON-OFF combined (max projection) zscore for positivie receptive field
    :return zscore_neg: ON-OFF combined (min projection) zscore for negative receptive field
    """

    zscore_on_pos, zscore_off_pos, _, _ = strf.get_zscore_map(timeWindow=pos_window)
    zscore_on_pos[np.isnan(zscore_on_pos)] = 0.
    zscore_off_pos[np.isnan(zscore_off_pos)] = 0.

    zscore_on_neg, zscore_off_neg, _, _ = strf.get_zscore_map(timeWindow=neg_window)
    zscore_on_neg[np.isnan(zscore_on_neg)] = 0.
    zscore_off_neg[np.isnan(zscore_off_neg)] = 0.

    return zscore_on_pos, zscore_off_pos, zscore_on_neg, zscore_off_neg


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


def get_dgc_peak_traces(nwb_f, plane_n, roi_n, trace_type=ANALYSIS_PARAMS['trace_type'],
                        response_window=ANALYSIS_PARAMS['response_window_dgc'],
                        baseline_window=ANALYSIS_PARAMS['baseline_window_dgc']):
    """
    find peak conditions based on averaged delta f (not df/f)

    :param nwb_f:
    :param plane_n:
    :param roi_n:
    :param trace_type:
    :param response_window:
    :param baseline_window:
    :return peak_cond_n_pos:
    :return peak_cond_n_neg:
    :return peak_traces_pos:
    :return peak_traces_neg:
    :return ts: timestamps
    """

    dgcrt_grp_key = get_dgcrt_grp_key(nwb_f=nwb_f)

    if dgcrt_grp_key:
        dgcrt_grp = nwb_f['analysis'][dgcrt_grp_key][plane_n]

        ts = dgcrt_grp.attrs['sta_timestamps']
        baseline_ind = np.logical_and(ts > baseline_window[0], ts <= baseline_window[1])
        response_ind = np.logical_and(ts > response_window[0], ts <= response_window[1])

        # remove blank condition
        cond_ns = [k for k in dgcrt_grp.keys() if 'tf00.0' not in k]
        cond_ns.sort()

        peak_cond_n_pos = None
        peak_cond_n_neg = None
        peak_traces_pos = None
        peak_traces_neg = None
        max_df = None
        min_df = None

        roi_i = int(roi_n[-4:])

        for cond_n in cond_ns:
            curr_traces = dgcrt_grp[cond_n]['sta_' + trace_type][roi_i, :, :]
            curr_trace_mean = np.mean(curr_traces, axis=0)
            curr_baseline = np.mean(curr_trace_mean[baseline_ind])
            curr_response = np.mean(curr_trace_mean[response_ind])
            curr_df = curr_response - curr_baseline

            if max_df is None:
                max_df = curr_df
                peak_cond_n_pos = cond_n
                peak_traces_pos = curr_traces
            else:
                if curr_df > max_df:
                    max_df = curr_df
                    peak_cond_n_pos = cond_n
                    peak_traces_pos = curr_traces

            if min_df is None:
                min_df = curr_df
                peak_cond_n_neg = cond_n
                peak_traces_neg = curr_traces
            else:
                if curr_df < min_df:
                    min_df = curr_df
                    peak_cond_n_neg = cond_n
                    peak_traces_neg = curr_traces

        return peak_cond_n_pos, peak_cond_n_neg, peak_traces_pos, peak_traces_neg, ts

    else:
        print('the nwb file contains no response table to drifting grating.')


def get_dgc_blank_traces(nwb_f, plane_n, roi_n, trace_type=ANALYSIS_PARAMS['trace_type']):
    """

    :param nwb_f:
    :param plane_n:
    :param roi_n:
    :return:
    """

    roi_i = int(roi_n[-4:])

    dgcrt_grp_key = get_dgcrt_grp_key(nwb_f=nwb_f)

    if dgcrt_grp_key:
        dgcrt_grp = nwb_f['analysis'][dgcrt_grp_key][plane_n]

        ts = dgcrt_grp.attrs['sta_timestamps']

        cond_n = [k for k in dgcrt_grp.keys() if 'sf0.00_tf00.0_dire000_con0.00_rad000' in k]
        if len(cond_n) == 0:
            print('no blank trial found.')
            return None, ts
        elif len(cond_n) == 1:
            cond_n = cond_n[0]
            blank_traces = dgcrt_grp[cond_n]['sta_' + trace_type][roi_i, :, :]
            return blank_traces, ts
        else:
            raise LookupError('more than one blank trials found. Combine them all.')
    else:
        print('the nwb file contains no response table to drifting grating.')


def roi_page_report(nwb_path, plane_n, roi_n, params=ANALYSIS_PARAMS, plot_params=PLOTTING_PARAMS):
    """
    generate a page of description of an roi

    :param nwb_path:
    :param plane_n:
    :param roi_n:
    :param params:
    :return:
    """

    nwb_f = h5py.File(nwb_path, 'r')

    # plotting roi
    # get background image
    segmentation_grp = nwb_f['processing/rois_and_traces_{}/ImageSegmentation/imaging_plane'.format(plane_n)]
    rf_img_grp = segmentation_grp['reference_images']
    if 'mean_projection' in rf_img_grp.keys():
        rf_img = rf_img_grp['mean_projection/data'].value
    else:
        rf_img = rf_img_grp['max_projection/data'].value
    # getting roi mask
    roi = get_roi(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n)
    pixel_size = nwb_f['acquisition/timeseries/2p_movie_{}/pixel_size'.format(plane_n)].value * 1000000.
    roi_area = roi.get_binary_area() * pixel_size[0] * pixel_size[1]

    # get depth
    depth = nwb_f['processing/rois_and_traces_{}/imaging_depth_micron'.format(plane_n)].value

    f = plt.figure(figsize=plot_params['fig_size'], facecolor=plot_params['fig_facecolor'])
    f.subplots_adjust(0, 0, 1, 1)
    ax_roi_img = f.add_axes(plot_params['ax_roi_img_coord'])
    ax_roi_img.imshow(ia.array_nor(rf_img), cmap='gray', vmin=plot_params['rf_img_vmin'],
                      vmax=plot_params['rf_img_vmax'], interpolation='nearest')
    pt.plot_mask_borders(mask=roi.get_binary_mask(), plotAxis=ax_roi_img, color=plot_params['roi_border_color'],
                         borderWidth=plot_params['roi_border_width'])
    ax_roi_img.set_axis_off()

    # plotting traces
    trace, trace_ts = get_single_trace(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n,
                                       trace_type=params['trace_type'])
    skew_raw, skew_fil = sca.get_skewness(trace=trace, ts=trace_ts,
                                          filter_length=ANALYSIS_PARAMS['filter_length_skew_sec'])
    trace_chunk_length = trace.shape[0] // plot_params['traces_panels']
    trace_min = np.min(trace)

    if trace_min <=0.:
        add_to_trace = -trace_min + params['add_to_trace_bias']
    else:
        add_to_trace = 0.

    trace = trace + add_to_trace
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
        curr_trace_chunk = trace[trace_i * trace_chunk_length : (trace_i + 1) * trace_chunk_length]
        curr_trace_axis.plot(curr_trace_chunk, color=plot_params['traces_color'],
                             lw=plot_params['traces_line_width'])
        curr_trace_axis.set_xlim([0, trace_chunk_length])
        curr_trace_axis.set_ylim([trace_min, trace_max])
        curr_trace_axis.set_axis_off()

    # plotting receptive fields
    if has_strf(nwb_f=nwb_f, plane_n=plane_n):
        strf = get_strf(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n)
        strf_dff = strf.get_local_dff_strf(is_collaps_before_normalize=True, add_to_trace=add_to_trace)
        zscore_rfs = get_pos_neg_zscore_maps(strf=strf_dff,
                                             pos_window=params['response_window_positive_rf'],
                                             neg_window=params['response_window_negative_rf'])

        zscore_on_pos, zscore_off_pos, zscore_on_neg, zscore_off_neg = zscore_rfs

        ax_rf_pos = f.add_axes(plot_params['ax_rf_pos_coord'])

        # render ON and OFF zscore maps in to red and blue channels
        zscore_pos = render_rb(rf_on=zscore_on_pos, rf_off=zscore_off_pos, vmax=plot_params['rf_zscore_vmax'])
        ax_rf_pos.imshow(zscore_pos, interpolation='nearest')
        ax_rf_pos.set_axis_off()

        ax_rf_neg = f.add_axes(plot_params['ax_rf_neg_coord'])
        zscore_neg = render_rb(rf_on=-zscore_on_neg, rf_off=-zscore_off_neg, vmax=plot_params['rf_zscore_vmax'])
        ax_rf_neg.imshow(zscore_neg, interpolation='nearest')
        ax_rf_neg.set_axis_off()

        peak_z_rf_on_pos = np.amax(ni.gaussian_filter(zscore_on_pos,
                                                      sigma=params['gaussian_filter_sigma_rf']))
        peak_z_rf_off_pos = np.amax(ni.gaussian_filter(zscore_off_pos,
                                                       sigma=params['gaussian_filter_sigma_rf']))
        peak_z_rf_on_neg = -np.amin(ni.gaussian_filter(zscore_on_neg,
                                                       sigma=params['gaussian_filter_sigma_rf']))
        peak_z_rf_off_neg = -np.amin(ni.gaussian_filter(zscore_off_neg,
                                                       sigma=params['gaussian_filter_sigma_rf']))
    else:
        peak_z_rf_on_pos = np.nan
        peak_z_rf_off_pos = np.nan
        peak_z_rf_on_neg = np.nan
        peak_z_rf_off_neg = np.nan

    # plotting drifting grating peak response
    dgcrt_grp_key = get_dgcrt_grp_key(nwb_f=nwb_f)
    if dgcrt_grp_key:

        block_dur = nwb_f['stimulus/presentation/{}/block_dur'.format(dgcrt_grp_key[15:])].value
        # print('block duration: {}'.format(block_dur))

        # dgcrt_grp = nwb_f['analysis'][dgcrt_grp_key]

        _ = get_dgc_peak_traces(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n, trace_type=ANALYSIS_PARAMS['trace_type'],
                                response_window=ANALYSIS_PARAMS['response_window_dgc'],
                                baseline_window=ANALYSIS_PARAMS['baseline_window_dgc'])
        peak_cond_n_pos, peak_cond_n_neg, peak_traces_pos, peak_traces_neg, ts_dgc = _

        blank_traces, _ = get_dgc_blank_traces(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n,
                                               trace_type=ANALYSIS_PARAMS['trace_type'])

        ax_peak_traces_pos = f.add_axes(PLOTTING_PARAMS['ax_peak_traces_pos_coord'])
        ax_peak_traces_pos.axhline(y=0, linestyle='--', color='#000000', lw=2)
        ax_peak_traces_pos.axvline(x=0, linestyle='--', color='#000000', lw=2)
        ax_peak_traces_pos.axvline(x=block_dur, linestyle='--', color='#000000', lw=2)
        ax_peak_traces_pos.axvline(x=ANALYSIS_PARAMS['response_window_dgc'][0], linestyle='--', color='#ff00ff', lw=2)
        ax_peak_traces_pos.axvline(x=ANALYSIS_PARAMS['response_window_dgc'][1], linestyle='--', color='#ff00ff', lw=2)
        ax_peak_traces_pos.set_xticks([])
        ax_peak_traces_pos.set_yticks([])
        ax_peak_traces_neg = f.add_axes(PLOTTING_PARAMS['ax_peak_traces_neg_coord'])
        ax_peak_traces_neg.axhline(y=0, linestyle='--', color='#000000', lw=2)
        ax_peak_traces_neg.axvline(x=0, linestyle='--', color='#000000', lw=2)
        ax_peak_traces_neg.axvline(x=block_dur, linestyle='--', color='#000000', lw=2)
        ax_peak_traces_neg.axvline(x=ANALYSIS_PARAMS['response_window_dgc'][0], linestyle='--', color='#ff00ff', lw=2)
        ax_peak_traces_neg.axvline(x=ANALYSIS_PARAMS['response_window_dgc'][1], linestyle='--', color='#ff00ff', lw=2)
        ax_peak_traces_neg.set_xticks([])
        ax_peak_traces_neg.set_yticks([])


        # plot blank traces
        for t in blank_traces:
            trace_df, trace_dff = sca.get_df_dff_trace(trace=t, t_axis=ts_dgc,
                                                       baseline_window=ANALYSIS_PARAMS['baseline_window_dgc'])
            ax_peak_traces_pos.plot(ts_dgc, trace_dff, color=PLOTTING_PARAMS['blank_traces_color'],
                                    lw=PLOTTING_PARAMS['single_traces_lw'])
            ax_peak_traces_neg.plot(ts_dgc, trace_dff, color=PLOTTING_PARAMS['blank_traces_color'],
                                    lw=PLOTTING_PARAMS['single_traces_lw'])

        blank_mean_df, blank_mean_dff = sca.get_df_dff_trace(trace=np.mean(blank_traces, axis=0),
                                                             t_axis=ts_dgc,
                                                             baseline_window=ANALYSIS_PARAMS['baseline_window_dgc'])
        ax_peak_traces_pos.plot(ts_dgc, blank_mean_dff, color=PLOTTING_PARAMS['blank_traces_color'],
                                lw=PLOTTING_PARAMS['mean_traces_mean_lw'])
        ax_peak_traces_neg.plot(ts_dgc, blank_mean_dff, color=PLOTTING_PARAMS['blank_traces_color'],
                                lw=PLOTTING_PARAMS['mean_traces_mean_lw'])

        # plot peak traces for positive response
        for t in peak_traces_pos:
            trace_df, trace_dff = sca.get_df_dff_trace(trace=t, t_axis=ts_dgc,
                                                       baseline_window=ANALYSIS_PARAMS['baseline_window_dgc'])
            ax_peak_traces_pos.plot(ts_dgc, trace_dff, color=PLOTTING_PARAMS['peak_traces_pos_color'],
                                    lw=PLOTTING_PARAMS['single_traces_lw'])

        peak_mean_df_pos, peak_mean_dff_pos = sca.get_df_dff_trace(trace=np.mean(peak_traces_pos, axis=0),
                                                                   t_axis=ts_dgc,
                                                                   baseline_window=ANALYSIS_PARAMS['baseline_window_dgc'])
        ax_peak_traces_pos.plot(ts_dgc, peak_mean_dff_pos, color=PLOTTING_PARAMS['peak_traces_pos_color'],
                                lw=PLOTTING_PARAMS['mean_traces_mean_lw'])

        # plot peak traces for positive response
        for t in peak_traces_neg:
            trace_df, trace_dff = sca.get_df_dff_trace(trace=t, t_axis=ts_dgc,
                                                       baseline_window=ANALYSIS_PARAMS['baseline_window_dgc'])
            ax_peak_traces_neg.plot(ts_dgc, trace_dff, color=PLOTTING_PARAMS['peak_traces_neg_color'],
                                    lw=PLOTTING_PARAMS['single_traces_lw'])

        peak_mean_df_neg, peak_mean_dff_neg = sca.get_df_dff_trace(trace=np.mean(peak_traces_neg, axis=0),
                                                                   t_axis=ts_dgc,
                                                                   baseline_window=ANALYSIS_PARAMS['baseline_window_dgc'])
        ax_peak_traces_neg.plot(ts_dgc, peak_mean_dff_neg, color=PLOTTING_PARAMS['peak_traces_neg_color'],
                                lw=PLOTTING_PARAMS['mean_traces_mean_lw'])

        ylim_pos = ax_peak_traces_pos.get_ylim()
        ylim_neg = ax_peak_traces_neg.get_ylim()
        ax_peak_traces_pos.set_ylim(min([ylim_pos[0], ylim_neg[0]]), max([ylim_pos[1], ylim_neg[1]]))
        ax_peak_traces_neg.set_ylim(min([ylim_pos[0], ylim_neg[0]]), max([ylim_pos[1], ylim_neg[1]]))



    # print text
    ax_text = f.add_axes(plot_params['ax_text_coord'])
    ax_text.set_xticks([])
    ax_text.set_yticks([])

    txt = '\n'
    txt += 'nwb: {}\n'.format(os.path.split(nwb_path)[1])
    txt += '\n'
    txt += 'plane name:        {}\n'.format(plane_n)
    txt += 'roi name:          {}\n'.format(roi_n)
    txt += '\n'
    txt += 'depth (um):        {}\n'.format(depth)
    txt += 'roi area (um^2):   {:.2f}\n'.format(roi_area)
    txt += '\n'
    txt += 'skewness raw:      {:4.2f}\n'.format(skew_raw)
    txt += 'skewness fil:      {:4.2f}\n'.format(skew_fil)
    txt += '\n'
    txt += 'rf_zscore_on_pos:  {:4.2f}\n'.format(peak_z_rf_on_pos)
    txt += 'rf_zscore_off_pos: {:4.2f}\n'.format(peak_z_rf_off_pos)
    txt += 'rf_zscore_on_neg:  {:4.2f}\n'.format(peak_z_rf_on_neg)
    txt += 'rf_zscore_off_neg: {:4.2f}\n'.format(peak_z_rf_off_neg)
    txt += '\n'
    txt += 'dgc_ttest_p:       \n'
    txt += 'dgc_anova_p:       \n'
    txt += '\n'
    txt += 'dgc_peak_df_pos:   \n'
    txt += 'dgc_peak_dff_pos:  \n'
    txt += 'dgc_peak_df_neg:   \n'
    txt += 'dgc_peak_dff_neg:  \n'
    txt += '\n'
    txt += 'dgc_peak_sf_pos:   \n'
    txt += 'dgc_peak_tf_pos:   \n'
    txt += 'dgc_peak_ori_pos:  \n'
    txt += 'dgc_peak_dir_pos:  \n'
    txt += '\n'
    txt += 'dgc_peak_sf_neg:   \n'
    txt += 'dgc_peak_tf_neg:   \n'
    txt += 'dgc_peak_ori_neg:  \n'
    txt += 'dgc_peak_dir_neg:  \n'

    ax_text.text(0.01, 0.99, txt, horizontalalignment='left', verticalalignment='top', family='monospace')

    plt.show()


if __name__ == '__main__':
    nwb_path = r"F:\data2\chandelier_cell_project\database\190208_M421761_110.nwb"
    plane_n = 'plane2'
    roi_n = 'roi_0001'
    roi_page_report(nwb_path=nwb_path, plane_n=plane_n, roi_n=roi_n)