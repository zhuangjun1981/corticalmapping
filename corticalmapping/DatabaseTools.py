"""
these are the functions that deal with the .nwb database of GCaMP labelled LGN boutons.
"""
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from numbers import Number
import scipy.stats as stats
import scipy.ndimage as ni
import scipy.interpolate as ip
import scipy.spatial as spatial
import scipy.cluster as cluster
import corticalmapping.SingleCellAnalysis as sca
import corticalmapping.core.ImageAnalysis as ia
import corticalmapping.core.PlottingTools as pt
import corticalmapping.core.DataAnalysis as da
import corticalmapping.core.TimingAnalysis as ta

ANALYSIS_PARAMS = {
    'trace_type': 'f_center_subtracted',
    'trace_abs_minimum': 1., # float, trace absolute minimum, if the roi trace minimum is lower than this value
                             # it will be added to with a bias to ensure the absolute minimum is no less than
                             # this value for robustness of df/f calculation
    'filter_length_skew_sec': 5., # float, second, the length to filter input trace to get slow trend
    'response_window_positive_rf': [0., 0.5], # list of 2 floats, temporal window to get upwards calcium response for receptive field
    'response_window_negative_rf': [0., 1.], # list of 2 floats, temporal window to get downward calcium response for receptive field
    'gaussian_filter_sigma_rf': 1., # float, in pixels, filtering sigma for z-score receptive fields
    'interpolate_rate_rf': 10., # float, interpolate rate of filtered z-score maps
    # 'peak_z_threshold_rf': 1.5, # float, threshold for significant receptive field of z score after filtering.
    'rf_z_thr_abs': 1.6, # float, absolute threshold for significant zscore receptive field
    'rf_z_thr_rel': 0.4, # float, relative threshold for significant zscore receptive field
    'response_window_dgc': [0., 1.], # list of two floats, temporal window for getting response value for drifting grating
    'baseline_window_dgc': [-0.5, 0.], # list of two floats, temporal window for getting baseline value for drifting grating
    'is_collapse_sf': True, # bool, average across sf or not for direction/tf tuning curve
    'is_collapse_tf': False, # bool, average across tf or not for direction/sf tuning curve
    'is_collapse_dire': False, # bool, average across direction or not for tf/sf tuning curve
    'dgc_elevation_bias': 0., # float, the bias to lift the dgc tuning curves if postprocess is 'elevate'
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
    'dgc_postprocess': 'elevate',
    'ax_text_coord': [0.63, 0.005, 0.36, 0.74],
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


def get_roi_triplets(nwb_f, overlap_ratio=0.9):
    """
    for deepscope imaging session with 3 planes, get overlapping roi triplets
    each triplets contain one roi for each plane and they are highly overlapping
    this is to find cells appear in multiple plane, and the results can be passed
    to HighLevel.plot_roi_traces_three_planes to plot and decided if they represent
    same cell.
    :param nwb_f:
    :param overlap_ratio:
    :return: list of triplets (tuple of three strings)
    """

    roi_grp0 = nwb_f['processing/rois_and_traces_plane0/ImageSegmentation/imaging_plane']
    roi_lst0 = roi_grp0['roi_list'].value
    roi_lst0 = [r for r in roi_lst0 if r[0:4] == 'roi_']

    roi_grp1 = nwb_f['processing/rois_and_traces_plane1/ImageSegmentation/imaging_plane']
    roi_lst1 = roi_grp1['roi_list'].value
    roi_lst1 = [r for r in roi_lst1 if r[0:4] == 'roi_']

    roi_grp2 = nwb_f['processing/rois_and_traces_plane2/ImageSegmentation/imaging_plane']
    roi_lst2 = roi_grp2['roi_list'].value
    roi_lst2 = [r for r in roi_lst2 if r[0:4] == 'roi_']

    triplets = []

    while roi_lst1: # start from middle plane

        curr_roi1_n = roi_lst1.pop(0)
        curr_triplet = [None, curr_roi1_n, None]

        curr_roi1 = get_roi(nwb_f=nwb_f, plane_n='plane1', roi_n=curr_roi1_n)
        curr_roi1_area = curr_roi1.get_binary_area()

        for curr_roi0_ind, curr_roi0_n in enumerate(roi_lst0):
            # look through rois in plane0, pick the one overlaps with curr_roi1
            curr_roi0 = get_roi(nwb_f=nwb_f, plane_n='plane0', roi_n=curr_roi0_n)
            curr_roi0_area = curr_roi0.get_binary_area()
            curr_overlap = curr_roi1.binary_overlap(curr_roi0)
            if float(curr_overlap) / min([curr_roi1_area, curr_roi0_area]) >= overlap_ratio:
                curr_triplet[0] = roi_lst0.pop(curr_roi0_ind)
                break

        for curr_roi2_ind, curr_roi2_n in enumerate(roi_lst2):
            # look through rois in plane0, pick the one overlaps with curr_roi1
            curr_roi2 = get_roi(nwb_f=nwb_f, plane_n='plane2', roi_n=curr_roi2_n)
            curr_roi2_area = curr_roi2.get_binary_area()
            curr_overlap = curr_roi1.binary_overlap(curr_roi2)
            if float(curr_overlap) / min([curr_roi1_area, curr_roi2_area]) >= overlap_ratio:
                curr_triplet[2] = roi_lst2.pop(curr_roi2_ind)
                break

        print(curr_triplet)
        triplets.append(tuple(curr_triplet))

    while roi_lst2: # next, more superficial plane

        curr_roi2_n = roi_lst2.pop(0)
        curr_triplet = [None, None, curr_roi2_n]

        curr_roi2 = get_roi(nwb_f=nwb_f, plane_n='plane2', roi_n=curr_roi2_n)
        curr_roi2_area = curr_roi2.get_binary_area()

        for curr_roi0_ind, curr_roi0_n in enumerate(roi_lst0):
            # look through rois in plane0, pick the one overlaps with curr_roi2
            curr_roi0 = get_roi(nwb_f=nwb_f, plane_n='plane0', roi_n=curr_roi0_n)
            curr_roi0_area = curr_roi0.get_binary_area()
            curr_overlap = curr_roi2.binary_overlap(curr_roi0)
            if float(curr_overlap) / min([curr_roi2_area, curr_roi0_area]) >= overlap_ratio:
                curr_triplet[0] = roi_lst0.pop(curr_roi0_ind)
                break

        triplets.append(tuple(curr_triplet))

    triplets = triplets + [(rn, None, None) for rn in roi_lst0] # finally add the rest rois in deep plane

    return triplets


def get_plane_ns(nwb_f):
    keys = [k[-6:] for k in nwb_f['processing'].keys() if 'rois_and_traces_' in k]
    return keys


def get_roi_ns(nwb_f, plane_n):
    roi_lst = nwb_f['processing/rois_and_traces_{}/ImageSegmentation/imaging_plane/roi_list'.format(plane_n)].value
    roi_ns = [r for r in roi_lst if r[0:4] == 'roi_']
    return roi_ns


def get_sampling_rate(nwb_f, ts_name):
    grp = nwb_f['acquisition/timeseries/{}'.format(ts_name)]

    if 'starting_time' in grp.keys():
        return grp['starting_time'].attrs['rate']
    else:
        ts = grp['timestamps'].value
        return 1. / np.mean(np.diff(ts))


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
                      z_thr_abs=ANALYSIS_PARAMS['rf_z_thr_abs'],
                      z_thr_rel=ANALYSIS_PARAMS['rf_z_thr_rel']):
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

    srf_new = srf.gaussian_filter(sigma=sigma)
    srf_new = srf_new.interpolate(ratio=interpolate_rate)

    if polarity == 'positive':
        rf_z = np.max(srf_new.weights)
    elif polarity == 'negative':
        srf_new.weights = -srf_new.weights
        rf_z = np.max(srf_new.weights)
    else:
        raise LookupError('Do not understand "polarity" ({}), should be "positive" or "negative".'.format(polarity))

    if rf_z > (z_thr_abs / z_thr_rel):
        srf_new = srf_new.threshold(thr=(rf_z * z_thr_rel))
    else:
        srf_new = srf_new.threshold(thr=z_thr_abs)
    # rf_center = srf_new.get_weighted_rf_center()
    # rf_area = srf_new.get_binary_rf_area()
    # rf_mask = srf_new.get_weighted_mask()
    return  rf_z, srf_new


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


def get_traces(nwb_f, plane_n, trace_type=ANALYSIS_PARAMS['trace_type']):

    traces = nwb_f['processing/rois_and_traces_{}/Fluorescence/{}/data'.format(plane_n, trace_type)].value
    trace_ts = nwb_f['processing/rois_and_traces_{}/Fluorescence/{}/timestamps'.format(plane_n, trace_type)].value
    return traces, trace_ts


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


def get_UC_ts_mask(nwb_f, plane_n='plane0'):
    """
    return a 1d boolean array, same size as imaging timestamps of the traces in plane_n.
    These index masks represent the time period of all UniformContrast stimuli.

    :return mask: 1d boolean array.
    :return has_uc: bool, False: has no UniformContrast stimulus
                          True: has UniformContrast stimulus
    """

    ts = nwb_f['processing/rois_and_traces_{}/Fluorescence/f_center_raw/timestamps'.format(plane_n)].value
    mask = np.zeros(ts.shape, dtype=np.bool)

    stim_ns = [n for n in nwb_f['stimulus/presentation'].keys() if 'UniformContrast' in n]

    if len(stim_ns) == 0:
        return mask, False

    else:
        for stim_n in stim_ns:

            stim_dur = nwb_f['stimulus/presentation/{}/duration'.format(stim_n)].value

            pd_grp = nwb_f['analysis/photodiode_onsets/{}'.format(stim_n)]
            pd_key = pd_grp.keys()[0]
            stim_onset = pd_grp[pd_key]['pd_onset_ts_sec'][0]

            curr_inds = np.logical_and(ts >= stim_onset, ts <= (stim_onset + stim_dur))
            mask = np.logical_or(mask, curr_inds)

        return mask, True


def get_DGC_spont_ts_mask(nwb_f, plane_n='plane0'):
    """
    return a 1d boolean array, same size as imaging timestamps of the traces in plane_n.
    These index masks represent the time period of blank sweep and second half of intersweep
    intervals. This representing the "spontaneous" period during DriftingGratingCircle stimuli

    :return mask: 1d boolean array
    :return has_dgc: bool, False: has no DriftingGratingCircle stimulus
                           True: has DriftingGratingCircle stimulus
    """

    ts = nwb_f['processing/rois_and_traces_{}/Fluorescence/f_center_raw/timestamps'.format(plane_n)].value
    mask = np.zeros(ts.shape, dtype=np.bool)

    stim_ns = [n for n in nwb_f['stimulus/presentation'].keys() if 'DriftingGratingCircle' in n]

    if len(stim_ns) == 0:
        return mask, False

    else:
        for stim_n in stim_ns:
            midgap_dur = nwb_f['stimulus/presentation/{}/midgap_dur'.format(stim_n)].value
            block_dur = nwb_f['stimulus/presentation/{}/block_dur'.format(stim_n)].value

            pd_grp = nwb_f['analysis/photodiode_onsets/{}'.format(stim_n)]
            pd_keys = pd_grp.keys()

            for pd_key in pd_keys:

                stim_onsets = pd_grp[pd_key]['pd_onset_ts_sec']

                for stim_onset in stim_onsets:

                    if pd_key[-36:] == 'sf0.00_tf00.0_dire000_con0.00_rad000':  # blank sweeps

                        curr_inds = np.logical_and(ts >= (stim_onset - 0.5 * midgap_dur),
                                                   ts <= (stim_onset + block_dur + midgap_dur))
                        mask = np.logical_or(mask, curr_inds)

                    else: # other sweeps
                        curr_inds = np.logical_and(ts >= (stim_onset - 0.5 * midgap_dur), ts <= stim_onset)
                        mask = np.logical_or(mask, curr_inds)

        return mask, True


def get_LSN_ts_mask(nwb_f, plane_n='plane0'):
    """
    return a 1d boolean array, same size as imaging timestamps of the traces in plane_n.
    These index masks represent the time period of all LocallySparseNoise stimuli.

    :return mask: 1d boolean array.
    :return has_uc: bool, False: has no LocallySparseNoise stimulus
                          True: has LocallySparseNoise stimulus
    """

    ts = nwb_f['processing/rois_and_traces_{}/Fluorescence/f_center_raw/timestamps'.format(plane_n)].value
    mask = np.zeros(ts.shape, dtype=np.bool)

    stim_ns = [n for n in nwb_f['stimulus/presentation'].keys() if 'LocallySparseNoise' in n]

    if len(stim_ns) == 0:
        return mask, False

    else:
        for stim_n in stim_ns:

            probe_frame_num = nwb_f['stimulus/presentation/{}/probe_frame_num'.format(stim_n)].value
            probe_dur = probe_frame_num / 60.

            pd_grp = nwb_f['analysis/photodiode_onsets/{}'.format(stim_n)]
            pd_keys = pd_grp.keys()

            stim_onset = None
            stim_offset = None

            for pd_key in pd_keys:
                curr_onsets = pd_grp[pd_key]['pd_onset_ts_sec'].value

                if stim_onset is None:
                    stim_onset = np.min(curr_onsets)
                else:
                    stim_onset = min([stim_onset, np.min(curr_onsets)])

                if stim_offset is None:
                    stim_offset = np.max(curr_onsets)
                else:
                    stim_offset = max([stim_offset, np.max(curr_onsets)])

            stim_offset = stim_offset + probe_dur

            curr_inds = np.logical_and(ts >= stim_onset, ts <= stim_offset)

            mask = np.logical_or(mask, curr_inds)

        return mask, True


# def group_boutons(traces, corr_std_thr=1.5, is_show=False):
#     """
#     given traces of a population of boutons, classify them into a tree based on their activity correlations.
#
#     method modified from: Liang et al., Cell, 2018, 173(6):1343
#
#     :param traces: 2d array, row: roi, col: time point
#     :param corr_std_thr: float, used to determine the threshold of correlation coefficients. for each roi, the
#                          coefficients lower than (mean + corr_std_thr * std) will be set zero.
#     :param is_show: bool
#     :return:
#     """
#
#     mat_corr = np.corrcoef(traces, rowvar=True)
#     mat_corr[np.isnan(mat_corr)] = 0.
#
#     # threshold correlation coefficient matrix
#     mask = np.ones(mat_corr.shape)
#     for row_i, row in enumerate(mat_corr):
#         curr_std = np.std(row)
#         curr_mean = np.mean(row)
#         curr_thr = curr_mean + corr_std_thr * curr_std
#         mask[row_i, :][row < curr_thr] = 0.
#         mask[:, row_i][row < curr_thr] = 0.
#     mat_corr_thr = mat_corr * mask
#
#     # calculated distance matrix based on cosine similarity
#     mat_dis = np.zeros(mat_corr_thr.shape)
#     roi_num = mat_dis.shape[0]
#     # print('total roi number: {}'.format(roi_num))
#     for i in range(roi_num):
#         for j in range(i + 1, roi_num, 1):
#
#             ind = np.ones(roi_num, dtype=np.bool)
#             ind[i] = 0
#             ind[j] = 0
#
#             row_i = mat_corr_thr[i][ind]
#             row_j = mat_corr_thr[j][ind]
#
#             if max(row_i) == 0 or max(row_j) == 0:
#                 mat_dis[i, j] = 1
#             else:
#                 mat_dis[i, j] = 1 - spatial.distance.cosine(row_i, row_j)
#
#
#     # calculate linkage Z matrix using WPGMA algorithm
#     z_linkage = cluster.hierarchy.linkage(mat_dis, method='weighted')
#     # if is_plot:
#     #     _ = cluster.hierarchy.dendrogram(z_linkage)
#     #     plt.title('dendrogram')
#     #     plt.show()
#
#
#     # reorganize thresholded correlation coefficient matrix
#     clu = cluster.hierarchy.fcluster(z_linkage, t=0, criterion='distance')
#     mat_0 = np.zeros(mat_corr_thr.shape)
#     for l_i, l in enumerate(clu):
#         mat_0[l - 1, :] = mat_corr_thr[l_i, :]
#
#     mat_corr_thr_reorg = np.zeros(mat_0.shape)
#     for l_i, l in enumerate(clu):
#         mat_corr_thr_reorg[:, l - 1] = mat_0[:, l_i]
#
#     # plotting
#     f = plt.figure(figsize=(13, 10))
#
#     ax00 = f.add_subplot(221)
#     f00 = ax00.imshow(mat_corr, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
#     ax00.set_title('corr coef matrix')
#     f.colorbar(f00)
#
#     ax01 = f.add_subplot(222)
#     f01 = ax01.imshow(mat_corr_thr, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
#     ax01.set_title('thresholded corr coef matrix')
#     f.colorbar(f01)
#
#     ax10 = f.add_subplot(223)
#     f10 = ax10.imshow(mat_dis, cmap='plasma', vmin=0, vmax=1, interpolation='nearest')
#     ax10.set_title('distance matrix')
#     f.colorbar(f10)
#
#     ax11 = f.add_subplot(224)
#     f11 = ax11.imshow(mat_corr_thr_reorg, cmap='plasma', vmin=0, vmax=1, interpolation='nearest')
#     ax11.set_title('reorganized thresholded corr coef matrix')
#     f.colorbar(f11)
#
#     if is_show:
#         plt.show()
#
#     return mat_corr, mat_corr_thr, mat_dis, z_linkage, mat_corr_thr_reorg, f


def plot_roi_retinotopy(coords_roi, coords_rf, ax_alt, ax_azi, alt_range=None, azi_range=None, cmap='viridis',
                        canvas_shape=(512, 512), nan_color='#cccccc', **kwargs):
    """
    plot color coded retinotopy on roi locations
    :param coords_roi: 2d array with shape (n, 2), row and col of roi location
    :param coords_rf: 2d array with same shape of coords_roi, alt and azi locations for each roi
    :param ax_alt: plotting axis for altitude
    :param ax_azi: plotting axis for azimuth
    :param alt_range:
        if None, the range to decide color is [minimum of altitudes of all rois, maximum of altitude of all rois]
        if float, the range to decide color is [median altitude - alt_range, median altitude + alt_range]
        if list of two floats, the range to decide color is [alt_range[0], alt_range[1]]
    :param azi_range: same as alt_range but for azimuth
    :param cmap: matplotlib color map
    :param canvas_shape: plotting shape (height, width)
    :param nan_color: color string, for nan data point, if None, do not plot nan data points
    :param kwargs: inputs to plotting functions
    :return:
    """

    if len(coords_roi.shape) != 2:
        raise ValueError('input coords_roi should be 2d array.')

    if coords_roi.shape[1] != 2:
        raise ValueError('input coords_roi should have 2 columns.')

    if coords_roi.shape != coords_rf.shape:
        raise ValueError('coords_roi and coords_rf should have same shape.')

    if alt_range is None:
        alt_ratio = ia.array_nor(coords_rf[:, 0])
    elif isinstance(alt_range, Number):
        if alt_range > 0:
            alt_median = np.nanmedian(coords_rf[:, 0])
            alt_min = alt_median - float(alt_range)
            alt_max = alt_median + float(alt_range)
            alt_ratio = (coords_rf[:, 0] - alt_min) / (alt_max - alt_min)
        else:
            raise ValueError('if "alt_range" is a number, it should be larger than 0.')
    elif len(alt_range) == 2:
        if alt_range[0] < alt_range[1]:
            alt_ratio = (coords_rf[:, 0] - alt_range[0]) / (alt_range[1] - alt_range[0])
        else:
            raise ValueError('if "alt_range" is a list or a tuple or a array, the first element should be '
                             'smaller than the second element.')
    else:
        raise ValueError('Do not understand input "alt_range", should be None or a single positive number or a '
                         'list or a tuple or a array with two elements with the first element smaller than the '
                         'second.')

    if azi_range is None:
        azi_ratio = ia.array_nor(coords_rf[:, 1])
    elif isinstance(azi_range, Number):
        if azi_range > 0:
            azi_median = np.nanmedian(coords_rf[:, 1])
            azi_min = azi_median - float(azi_range)
            azi_max = azi_median + float(azi_range)
            azi_ratio = (coords_rf[:, 1] - azi_min) / (azi_max - azi_min)
        else:
            raise ValueError('if "azi_range" is a number, it should be larger than 0.')
    elif len(azi_range) == 2:
        if azi_range[0] < azi_range[1]:
            azi_ratio = (coords_rf[:, 1] - azi_range[0]) / (azi_range[1] - azi_range[0])
        else:
            raise ValueError('if "azi_range" is a list or a tuple or a array, the first element should be '
                             'smaller than the second element.')
    else:
        raise ValueError('Do not understand input "azi_range", should be None or a single positive number or a '
                         'list or a tuple or a array with two elements with the first element smaller than the '
                         'second.')

    xs = coords_roi[:, 1]
    ys = coords_roi[:, 0]

    ax_alt.set_xlim([0, canvas_shape[1]])
    ax_alt.set_ylim([0, canvas_shape[0]])
    ax_alt.set_aspect('equal')
    ax_alt.invert_yaxis()
    ax_alt.set_xticks([])
    ax_alt.set_yticks([])

    ax_azi.set_xlim([0, canvas_shape[1]])
    ax_azi.set_ylim([0, canvas_shape[0]])
    ax_azi.set_aspect('equal')
    ax_azi.invert_yaxis()
    ax_azi.set_xticks([])
    ax_azi.set_yticks([])

    for roi_i in range(coords_roi.shape[0]):

        curr_alt_ratio = alt_ratio[roi_i]
        if np.isnan(curr_alt_ratio):
            if nan_color is not None:
                ax_alt.scatter([xs[roi_i]], [ys[roi_i]], marker='o', color=nan_color, **kwargs)
        else:
            alt_c = pt.cmap_2_rgb(curr_alt_ratio, cmap_string=cmap)
            ax_alt.scatter([xs[roi_i]], [ys[roi_i]], marker='o', color=alt_c, **kwargs)

        curr_azi_ratio = azi_ratio[roi_i]
        if np.isnan(curr_azi_ratio):
            if nan_color is not None:
                ax_azi.scatter([xs[roi_i]], [ys[roi_i]], marker='o', color=nan_color, **kwargs)
        else:
            azi_c = pt.cmap_2_rgb(azi_ratio[roi_i], cmap_string=cmap)
            ax_azi.scatter([xs[roi_i]], [ys[roi_i]], marker='o', color=azi_c, **kwargs)


def get_pupil_area(nwb_f, module_name, ell_thr=0.5, median_win=3.):

    pupil_shape = nwb_f['processing/{}/PupilTracking/eyetracking/pupil_shape'.format(module_name)].value
    pupil_ts = nwb_f['processing/{}/PupilTracking/eyetracking/timestamps'.format(module_name)].value

    fs = 1. / np.mean(np.diff(pupil_ts))
    # print(fs)

    pupil_area = da.get_pupil_area(pupil_shapes=pupil_shape, fs=fs, ell_thr=ell_thr, median_win=median_win)
    return pupil_area, pupil_ts


def get_running_speed(nwb_f, disk_radius=8., fs_final=30., speed_thr_pos=100., speed_thr_neg=-20.,
                      gauss_sig=1.):

    ref = nwb_f['acquisition/timeseries/analog_running_ref/data'].value
    sig = nwb_f['acquisition/timeseries/analog_running_sig/data'].value
    starting_time = nwb_f['acquisition/timeseries/analog_running_ref/starting_time'].value
    ts_rate = nwb_f['acquisition/timeseries/analog_running_ref/starting_time'].attrs['rate']
    num_sample = nwb_f['acquisition/timeseries/analog_running_ref/num_samples'].value

    ts = starting_time + np.arange(num_sample) / ts_rate

    speed, speed_ts = da.get_running_speed(sig=sig, ts=ts, ref=ref, disk_radius=disk_radius, fs_final=fs_final,
                                           speed_thr_pos=speed_thr_pos, speed_thr_neg=speed_thr_neg,
                                           gauss_sig=gauss_sig)

    return speed, speed_ts


def plot_roi_contour_on_background(nwb_f, plane_n, plot_ax, **kwargs):
    """
    :param nwb_f:
    :param plane_n:
    :param plot_ax:
    :param kwargs: input variable to corticalmapping.core.PlottingTools.plot_mask_borders
    :return:
    """

    seg_grp = nwb_f['processing/rois_and_traces_{}/ImageSegmentation/imaging_plane'.format(plane_n)]

    if 'max_projection' in seg_grp['reference_images']:
        bg = seg_grp['reference_images/max_projection/data'].value
        bg = ia.array_nor(bg)
        plot_ax.imshow(bg, vmin=0, vmax=0.8, cmap='gray', interpolation='nearest')
    elif 'max_projection' in seg_grp['reference_images']:
        bg = seg_grp['reference_images/mean_projection/data'].value
        bg = ia.array_nor(bg)
        plot_ax.imshow(bg, vmin=0, vmax=0.8, cmap='gray', interpolation='nearest')
    else:
        print('cannot find reference image, set background to black')
        # plot_ax.set_facecolor('#000000') # for matplotlib >= v2.0
        plot_ax.set_axis_bgcolor('#000000') # for matplotlib < v2.0

    roi_ns = [r for r in seg_grp['roi_list'] if r[0:4] == 'roi_']
    for roi_n in roi_ns:
        roi_mask = seg_grp[roi_n]['img_mask'].value
        pt.plot_mask_borders(mask=roi_mask, plotAxis=plot_ax, **kwargs)


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

    if np.min(trace) < params['trace_abs_minimum']:
        add_to_trace = -np.min(trace) + params['trace_abs_minimum']
    else:
        add_to_trace = 0.

    strf = get_strf(nwb_f=nwb_f, plane_n=plane_n, roi_ind=roi_ind, trace_type='sta_' + params['trace_type'])
    if strf is not None:

        # get strf properties
        strf_dff = strf.get_local_dff_strf(is_collaps_before_normalize=True, add_to_trace=add_to_trace)

        # positive spatial receptive fields
        srf_pos_on, srf_pos_off = strf_dff.get_zscore_receptive_field(timeWindow=params['response_window_positive_rf'])

        # # get filter sigma in pixels
        # mean_probe_size = (np.abs(np.mean(np.diff(srf_pos_on.altPos))) +
        #                   np.abs(np.mean(np.diff(srf_pos_on.aziPos)))) / 2.
        # print(mean_probe_size)
        # sigma = params['gaussian_filter_sigma_rf'] / mean_probe_size
        # print(sigma)

        # ON positive spatial receptive field
        rf_pos_on_z, rf_pos_on_new = get_rf_properties(srf= srf_pos_on,
                                                       polarity='positive',
                                                       sigma=params['gaussian_filter_sigma_rf'],
                                                       interpolate_rate=params['interpolate_rate_rf'],
                                                       z_thr_abs=params['rf_z_thr_abs'],
                                                       z_thr_rel=params['rf_z_thr_rel'])
        rf_pos_on_area = rf_pos_on_new.get_binary_rf_area()
        rf_pos_on_center = rf_pos_on_new.get_weighted_rf_center()

        roi_properties.update({'rf_pos_on_peak_z': rf_pos_on_z,
                               'rf_pos_on_area': rf_pos_on_area,
                               'rf_pos_on_center_alt': rf_pos_on_center[0],
                               'rf_pos_on_center_azi': rf_pos_on_center[1]})

        # OFF positive spatial receptive field
        rf_pos_off_z, rf_pos_off_new = get_rf_properties(srf=srf_pos_off,
                                                         polarity='positive',
                                                         sigma=params['gaussian_filter_sigma_rf'],
                                                         interpolate_rate=params['interpolate_rate_rf'],
                                                         z_thr_abs=params['rf_z_thr_abs'],
                                                         z_thr_rel=params['rf_z_thr_rel'])

        rf_pos_off_area = rf_pos_off_new.get_binary_rf_area()
        rf_pos_off_center = rf_pos_off_new.get_weighted_rf_center()

        roi_properties.update({'rf_pos_off_peak_z': rf_pos_off_z,
                               'rf_pos_off_area': rf_pos_off_area,
                               'rf_pos_off_center_alt': rf_pos_off_center[0],
                               'rf_pos_off_center_azi': rf_pos_off_center[1]})

        # on off overlapping
        rf_pos_on_mask = rf_pos_on_new.get_weighted_mask()
        rf_pos_off_mask = rf_pos_off_new.get_weighted_mask()
        rf_pos_lsi = sca.get_local_similarity_index(rf_pos_on_mask, rf_pos_off_mask)

        rf_pos_onoff_new = sca.SpatialReceptiveField(mask=np.max([rf_pos_on_mask, rf_pos_off_mask], axis=0),
                                                     altPos=rf_pos_on_new.altPos,
                                                     aziPos=rf_pos_on_new.aziPos,
                                                     sign='ON_OFF',
                                                     thr=params['rf_z_thr_abs'])
        if len(rf_pos_onoff_new.weights) == 0:
            rf_pos_onoff_z = np.nan
        else:
            rf_pos_onoff_z = np.max(rf_pos_onoff_new.weights)
        rf_pos_onoff_area = rf_pos_onoff_new.get_binary_rf_area()
        rf_pos_onoff_center = rf_pos_onoff_new.get_weighted_rf_center()
        roi_properties.update({'rf_pos_lsi': rf_pos_lsi,
                               'rf_pos_onoff_peak_z':rf_pos_onoff_z,
                               'rf_pos_onoff_area': rf_pos_onoff_area,
                               'rf_pos_onoff_center_alt': rf_pos_onoff_center[0],
                               'rf_pos_onoff_center_azi': rf_pos_onoff_center[1]})


        # negative spatial receptive fields
        srf_neg_on, srf_neg_off = strf_dff.get_zscore_receptive_field(timeWindow=params['response_window_negative_rf'])

        # ON negative spatial receptive field
        rf_neg_on_z, rf_neg_on_new = get_rf_properties(srf=srf_neg_on,
                                                       polarity='negative',
                                                       sigma=params['gaussian_filter_sigma_rf'],
                                                       interpolate_rate=params['interpolate_rate_rf'],
                                                       z_thr_abs=params['rf_z_thr_abs'],
                                                       z_thr_rel=params['rf_z_thr_rel'])
        rf_neg_on_area = rf_neg_on_new.get_binary_rf_area()
        rf_neg_on_center = rf_neg_on_new.get_weighted_rf_center()
        roi_properties.update({'rf_neg_on_peak_z': rf_neg_on_z,
                               'rf_neg_on_area': rf_neg_on_area,
                               'rf_neg_on_center_alt': rf_neg_on_center[0],
                               'rf_neg_on_center_azi': rf_neg_on_center[1]})

        # OFF negative spatial receptive field
        rf_neg_off_z, rf_neg_off_new = get_rf_properties(srf=srf_neg_off,
                                                         polarity='negative',
                                                         sigma=params['gaussian_filter_sigma_rf'],
                                                         interpolate_rate=params['interpolate_rate_rf'],
                                                         z_thr_abs=params['rf_z_thr_abs'],
                                                         z_thr_rel=params['rf_z_thr_rel'])
        rf_neg_off_area = rf_neg_off_new.get_binary_rf_area()
        rf_neg_off_center = rf_neg_off_new.get_weighted_rf_center()
        roi_properties.update({'rf_neg_off_peak_z': rf_neg_off_z,
                               'rf_neg_off_area': rf_neg_off_area,
                               'rf_neg_off_center_alt': rf_neg_off_center[0],
                               'rf_neg_off_center_azi': rf_neg_off_center[1]})

        # on off overlapping
        rf_neg_on_mask = rf_neg_on_new.get_weighted_mask()
        rf_neg_off_mask = rf_neg_off_new.get_weighted_mask()
        rf_neg_lsi = sca.get_local_similarity_index(rf_neg_on_mask, rf_neg_off_mask)

        rf_neg_onoff_new = sca.SpatialReceptiveField(mask=np.max([rf_neg_on_mask, rf_neg_off_mask], axis=0),
                                                     altPos=rf_neg_on_new.altPos,
                                                     aziPos=rf_neg_on_new.aziPos,
                                                     sign='ON_OFF',
                                                     thr=params['rf_z_thr_abs'])
        if len(rf_neg_onoff_new.weights) == 0:
            rf_neg_onoff_z = np.nan
        else:
            rf_neg_onoff_z = np.max(rf_neg_onoff_new.weights)
        rf_neg_onoff_area = rf_neg_onoff_new.get_binary_rf_area()
        rf_neg_onoff_center = rf_neg_onoff_new.get_weighted_rf_center()
        roi_properties.update({'rf_neg_onoff_peak_z': rf_neg_onoff_z,
                               'rf_neg_onoff_area': rf_neg_onoff_area,
                               'rf_neg_onoff_center_alt': rf_neg_onoff_center[0],
                               'rf_neg_onoff_center_azi': rf_neg_onoff_center[1],
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
                               'rf_pos_onoff_peak_z': np.nan,
                               'rf_pos_onoff_area': np.nan,
                               'rf_pos_onoff_center_alt': np.nan,
                               'rf_pos_onoff_center_azi': np.nan,
                               'rf_pos_lsi': np.nan,
                               'rf_neg_on_peak_z': np.nan,
                               'rf_neg_on_area': np.nan,
                               'rf_neg_on_center_alt': np.nan,
                               'rf_neg_on_center_azi': np.nan,
                               'rf_neg_off_peak_z': np.nan,
                               'rf_neg_off_area': np.nan,
                               'rf_neg_off_center_alt': np.nan,
                               'rf_neg_off_center_azi': np.nan,
                               'rf_neg_onoff_peak_z': np.nan,
                               'rf_neg_onoff_area': np.nan,
                               'rf_neg_onoff_center_alt': np.nan,
                               'rf_neg_onoff_center_azi': np.nan,
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
                                                  bias=add_to_trace)
        dgcrm_z = dgcrm.get_zscore_response_matrix(baseline_win=params['baseline_window_dgc'])


        # direction/orientation tuning of df responses in positive direction ===========================================
        dire_tuning_df_pos = dgcrt_df.get_dire_tuning(response_dir='pos',
                                                      is_collapse_sf=params['is_collapse_sf'],
                                                      is_collapse_tf=params['is_collapse_tf'])
        osi_df_pos_raw, dsi_df_pos_raw, gosi_df_pos_raw, gdsi_df_pos_raw, \
        osi_df_pos_ele, dsi_df_pos_ele, gosi_df_pos_ele, gdsi_df_pos_ele, \
        osi_df_pos_rec, dsi_df_pos_rec, gosi_df_pos_rec, gdsi_df_pos_rec, \
        peak_dire_raw_df_pos, vs_dire_raw_df_pos, vs_dire_ele_df_pos, vs_dire_rec_df_pos\
            = dgcrt_df.get_dire_tuning_properties(dire_tuning_df_pos,
                                                  response_dir='pos',
                                                  elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_pos_osi_raw_df': osi_df_pos_raw,
                               'dgc_pos_dsi_raw_df': dsi_df_pos_raw,
                               'dgc_pos_gosi_raw_df': gosi_df_pos_raw,
                               'dgc_pos_gdsi_raw_df': gdsi_df_pos_raw,
                               'dgc_pos_osi_ele_df': osi_df_pos_ele,
                               'dgc_pos_dsi_ele_df': dsi_df_pos_ele,
                               'dgc_pos_gosi_ele_df': gosi_df_pos_ele,
                               'dgc_pos_gdsi_ele_df': gdsi_df_pos_ele,
                               'dgc_pos_osi_rec_df': osi_df_pos_rec,
                               'dgc_pos_dsi_rec_df': dsi_df_pos_rec,
                               'dgc_pos_gosi_rec_df': gosi_df_pos_rec,
                               'dgc_pos_gdsi_rec_df': gdsi_df_pos_rec,
                               'dgc_pos_peak_dire_raw_df': peak_dire_raw_df_pos,
                               'dgc_pos_vs_dire_raw_df': vs_dire_raw_df_pos,
                               'dgc_pos_vs_dire_ele_df': vs_dire_ele_df_pos,
                               'dgc_pos_vs_dire_rec_df': vs_dire_rec_df_pos})


        # direction/orientation tuning of df responses in negative direction ===========================================
        dire_tuning_df_neg = dgcrt_df.get_dire_tuning(response_dir='neg',
                                                      is_collapse_sf=params['is_collapse_sf'],
                                                      is_collapse_tf=params['is_collapse_tf'])
        osi_df_neg_raw, dsi_df_neg_raw, gosi_df_neg_raw, gdsi_df_neg_raw, \
        osi_df_neg_ele, dsi_df_neg_ele, gosi_df_neg_ele, gdsi_df_neg_ele, \
        osi_df_neg_rec, dsi_df_neg_rec, gosi_df_neg_rec, gdsi_df_neg_rec, \
        peak_dire_raw_df_neg, vs_dire_raw_df_neg, vs_dire_ele_df_neg, vs_dire_rec_df_neg \
            = dgcrt_df.get_dire_tuning_properties(dire_tuning_df_neg,
                                                  response_dir='neg',
                                                  elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_neg_osi_raw_df': osi_df_neg_raw,
                               'dgc_neg_dsi_raw_df': dsi_df_neg_raw,
                               'dgc_neg_gosi_raw_df': gosi_df_neg_raw,
                               'dgc_neg_gdsi_raw_df': gdsi_df_neg_raw,
                               'dgc_neg_osi_ele_df': osi_df_neg_ele,
                               'dgc_neg_dsi_ele_df': dsi_df_neg_ele,
                               'dgc_neg_gosi_ele_df': gosi_df_neg_ele,
                               'dgc_neg_gdsi_ele_df': gdsi_df_neg_ele,
                               'dgc_neg_osi_rec_df': osi_df_neg_rec,
                               'dgc_neg_dsi_rec_df': dsi_df_neg_rec,
                               'dgc_neg_gosi_rec_df': gosi_df_neg_rec,
                               'dgc_neg_gdsi_rec_df': gdsi_df_neg_rec,
                               'dgc_neg_peak_dire_raw_df': peak_dire_raw_df_neg,
                               'dgc_neg_vs_dire_raw_df': vs_dire_raw_df_neg,
                               'dgc_neg_vs_dire_ele_df': vs_dire_ele_df_neg,
                               'dgc_neg_vs_dire_rec_df': vs_dire_rec_df_neg})


        # sf tuning of df responses in positive direction ==============================================================
        sf_tuning_df_pos = dgcrt_df.get_sf_tuning(response_dir='pos', is_collapse_tf=params['is_collapse_tf'],
                                                  is_collapse_dire=params['is_collapse_dire'])
        peak_sf_raw_df_pos, weighted_sf_raw_df_pos, weighted_sf_log_raw_df_pos, \
                            weighted_sf_ele_df_pos, weighted_sf_log_ele_df_pos, \
                            weighted_sf_rec_df_pos, weighted_sf_log_rec_df_pos= \
            dgcrt_df.get_sf_tuning_properties(sf_tuning_df_pos, response_dir='pos',
                                              elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_pos_peak_sf_raw_df': peak_sf_raw_df_pos,
                               'dgc_pos_weighted_sf_raw_df': weighted_sf_raw_df_pos,
                               'dgc_pos_weighted_sf_log_raw_df': weighted_sf_log_raw_df_pos,
                               'dgc_pos_weighted_sf_ele_df': weighted_sf_ele_df_pos,
                               'dgc_pos_weighted_sf_log_ele_df': weighted_sf_log_ele_df_pos,
                               'dgc_pos_weighted_sf_rec_df': weighted_sf_rec_df_pos,
                               'dgc_pos_weighted_sf_log_rec_df': weighted_sf_log_rec_df_pos})


        # sf tuning of df responses in negative direction ==============================================================
        sf_tuning_df_neg = dgcrt_df.get_sf_tuning(response_dir='neg', is_collapse_tf=params['is_collapse_tf'],
                                                  is_collapse_dire=params['is_collapse_dire'])
        peak_sf_raw_df_neg, weighted_sf_raw_df_neg, weighted_sf_log_raw_df_neg, \
        weighted_sf_ele_df_neg, weighted_sf_log_ele_df_neg, \
        weighted_sf_rec_df_neg, weighted_sf_log_rec_df_neg = \
            dgcrt_df.get_sf_tuning_properties(sf_tuning_df_neg, response_dir='neg',
                                              elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_neg_peak_sf_raw_df': peak_sf_raw_df_neg,
                               'dgc_neg_weighted_sf_raw_df': weighted_sf_raw_df_neg,
                               'dgc_neg_weighted_sf_log_raw_df': weighted_sf_log_raw_df_neg,
                               'dgc_neg_weighted_sf_ele_df': weighted_sf_ele_df_neg,
                               'dgc_neg_weighted_sf_log_ele_df': weighted_sf_log_ele_df_neg,
                               'dgc_neg_weighted_sf_rec_df': weighted_sf_rec_df_neg,
                               'dgc_neg_weighted_sf_log_rec_df': weighted_sf_log_rec_df_neg})


        # tf tuning of df responses in positive direction ==============================================================
        tf_tuning_df_pos = dgcrt_df.get_tf_tuning(response_dir='pos', is_collapse_sf=params['is_collapse_sf'],
                                                  is_collapse_dire=params['is_collapse_dire'])
        peak_tf_raw_df_pos, weighted_tf_raw_df_pos, weighted_tf_log_raw_df_pos, \
        weighted_tf_ele_df_pos, weighted_tf_log_ele_df_pos, \
        weighted_tf_rec_df_pos, weighted_tf_log_rec_df_pos = \
            dgcrt_df.get_tf_tuning_properties(tf_tuning_df_pos, response_dir='pos',
                                              elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_pos_peak_tf_raw_df': peak_tf_raw_df_pos,
                               'dgc_pos_weighted_tf_raw_df': weighted_tf_raw_df_pos,
                               'dgc_pos_weighted_tf_log_raw_df': weighted_tf_log_raw_df_pos,
                               'dgc_pos_weighted_tf_ele_df': weighted_tf_ele_df_pos,
                               'dgc_pos_weighted_tf_log_ele_df': weighted_tf_log_ele_df_pos,
                               'dgc_pos_weighted_tf_rec_df': weighted_tf_rec_df_pos,
                               'dgc_pos_weighted_tf_log_rec_df': weighted_tf_log_rec_df_pos})

        # tf tuning of df responses in negative direction ==============================================================
        tf_tuning_df_neg = dgcrt_df.get_tf_tuning(response_dir='neg', is_collapse_sf=params['is_collapse_sf'],
                                                  is_collapse_dire=params['is_collapse_dire'])
        peak_tf_raw_df_neg, weighted_tf_raw_df_neg, weighted_tf_log_raw_df_neg, \
        weighted_tf_ele_df_neg, weighted_tf_log_ele_df_neg, \
        weighted_tf_rec_df_neg, weighted_tf_log_rec_df_neg = \
            dgcrt_df.get_tf_tuning_properties(tf_tuning_df_neg, response_dir='neg',
                                              elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_neg_peak_tf_raw_df': peak_tf_raw_df_neg,
                               'dgc_neg_weighted_tf_raw_df': weighted_tf_raw_df_neg,
                               'dgc_neg_weighted_tf_log_raw_df': weighted_tf_log_raw_df_neg,
                               'dgc_neg_weighted_tf_ele_df': weighted_tf_ele_df_neg,
                               'dgc_neg_weighted_tf_log_ele_df': weighted_tf_log_ele_df_neg,
                               'dgc_neg_weighted_tf_rec_df': weighted_tf_rec_df_neg,
                               'dgc_neg_weighted_tf_log_rec_df': weighted_tf_log_rec_df_neg})

        # direction/orientation tuning of dff responses in positive direction ===========================================
        dire_tuning_dff_pos = dgcrt_dff.get_dire_tuning(response_dir='pos',
                                                      is_collapse_sf=params['is_collapse_sf'],
                                                      is_collapse_tf=params['is_collapse_tf'])
        osi_dff_pos_raw, dsi_dff_pos_raw, gosi_dff_pos_raw, gdsi_dff_pos_raw, \
        osi_dff_pos_ele, dsi_dff_pos_ele, gosi_dff_pos_ele, gdsi_dff_pos_ele, \
        osi_dff_pos_rec, dsi_dff_pos_rec, gosi_dff_pos_rec, gdsi_dff_pos_rec, \
        peak_dire_raw_dff_pos, vs_dire_raw_dff_pos, vs_dire_ele_dff_pos, vs_dire_rec_dff_pos \
            = dgcrt_dff.get_dire_tuning_properties(dire_tuning_dff_pos,
                                                  response_dir='pos',
                                                  elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_pos_osi_raw_dff': osi_dff_pos_raw,
                               'dgc_pos_dsi_raw_dff': dsi_dff_pos_raw,
                               'dgc_pos_gosi_raw_dff': gosi_dff_pos_raw,
                               'dgc_pos_gdsi_raw_dff': gdsi_dff_pos_raw,
                               'dgc_pos_osi_ele_dff': osi_dff_pos_ele,
                               'dgc_pos_dsi_ele_dff': dsi_dff_pos_ele,
                               'dgc_pos_gosi_ele_dff': gosi_dff_pos_ele,
                               'dgc_pos_gdsi_ele_dff': gdsi_dff_pos_ele,
                               'dgc_pos_osi_rec_dff': osi_dff_pos_rec,
                               'dgc_pos_dsi_rec_dff': dsi_dff_pos_rec,
                               'dgc_pos_gosi_rec_dff': gosi_dff_pos_rec,
                               'dgc_pos_gdsi_rec_dff': gdsi_dff_pos_rec,
                               'dgc_pos_peak_dire_raw_dff': peak_dire_raw_dff_pos,
                               'dgc_pos_vs_dire_raw_dff': vs_dire_raw_dff_pos,
                               'dgc_pos_vs_dire_ele_dff': vs_dire_ele_dff_pos,
                               'dgc_pos_vs_dire_rec_dff': vs_dire_rec_dff_pos})

        # direction/orientation tuning of dff responses in negative direction ===========================================
        dire_tuning_dff_neg = dgcrt_dff.get_dire_tuning(response_dir='neg',
                                                      is_collapse_sf=params['is_collapse_sf'],
                                                      is_collapse_tf=params['is_collapse_tf'])
        osi_dff_neg_raw, dsi_dff_neg_raw, gosi_dff_neg_raw, gdsi_dff_neg_raw, \
        osi_dff_neg_ele, dsi_dff_neg_ele, gosi_dff_neg_ele, gdsi_dff_neg_ele, \
        osi_dff_neg_rec, dsi_dff_neg_rec, gosi_dff_neg_rec, gdsi_dff_neg_rec, \
        peak_dire_raw_dff_neg, vs_dire_raw_dff_neg, vs_dire_ele_dff_neg, vs_dire_rec_dff_neg \
            = dgcrt_dff.get_dire_tuning_properties(dire_tuning_dff_neg,
                                                  response_dir='neg',
                                                  elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_neg_osi_raw_dff': osi_dff_neg_raw,
                               'dgc_neg_dsi_raw_dff': dsi_dff_neg_raw,
                               'dgc_neg_gosi_raw_dff': gosi_dff_neg_raw,
                               'dgc_neg_gdsi_raw_dff': gdsi_dff_neg_raw,
                               'dgc_neg_osi_ele_dff': osi_dff_neg_ele,
                               'dgc_neg_dsi_ele_dff': dsi_dff_neg_ele,
                               'dgc_neg_gosi_ele_dff': gosi_dff_neg_ele,
                               'dgc_neg_gdsi_ele_dff': gdsi_dff_neg_ele,
                               'dgc_neg_osi_rec_dff': osi_dff_neg_rec,
                               'dgc_neg_dsi_rec_dff': dsi_dff_neg_rec,
                               'dgc_neg_gosi_rec_dff': gosi_dff_neg_rec,
                               'dgc_neg_gdsi_rec_dff': gdsi_dff_neg_rec,
                               'dgc_neg_peak_dire_raw_dff': peak_dire_raw_dff_neg,
                               'dgc_neg_vs_dire_raw_dff': vs_dire_raw_dff_neg,
                               'dgc_neg_vs_dire_ele_dff': vs_dire_ele_dff_neg,
                               'dgc_neg_vs_dire_rec_dff': vs_dire_rec_dff_neg})

        # sf tuning of dff responses in positive direction ==============================================================
        sf_tuning_dff_pos = dgcrt_dff.get_sf_tuning(response_dir='pos', is_collapse_tf=params['is_collapse_tf'],
                                                  is_collapse_dire=params['is_collapse_dire'])
        peak_sf_raw_dff_pos, weighted_sf_raw_dff_pos, weighted_sf_log_raw_dff_pos, \
        weighted_sf_ele_dff_pos, weighted_sf_log_ele_dff_pos, \
        weighted_sf_rec_dff_pos, weighted_sf_log_rec_dff_pos = \
            dgcrt_dff.get_sf_tuning_properties(sf_tuning_dff_pos, response_dir='pos',
                                              elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_pos_peak_sf_raw_dff': peak_sf_raw_dff_pos,
                               'dgc_pos_weighted_sf_raw_dff': weighted_sf_raw_dff_pos,
                               'dgc_pos_weighted_sf_log_raw_dff': weighted_sf_log_raw_dff_pos,
                               'dgc_pos_weighted_sf_ele_dff': weighted_sf_ele_dff_pos,
                               'dgc_pos_weighted_sf_log_ele_dff': weighted_sf_log_ele_dff_pos,
                               'dgc_pos_weighted_sf_rec_dff': weighted_sf_rec_dff_pos,
                               'dgc_pos_weighted_sf_log_rec_dff': weighted_sf_log_rec_dff_pos})

        # sf tuning of dff responses in negative direction ==============================================================
        sf_tuning_dff_neg = dgcrt_dff.get_sf_tuning(response_dir='neg', is_collapse_tf=params['is_collapse_tf'],
                                                  is_collapse_dire=params['is_collapse_dire'])
        peak_sf_raw_dff_neg, weighted_sf_raw_dff_neg, weighted_sf_log_raw_dff_neg, \
        weighted_sf_ele_dff_neg, weighted_sf_log_ele_dff_neg, \
        weighted_sf_rec_dff_neg, weighted_sf_log_rec_dff_neg = \
            dgcrt_dff.get_sf_tuning_properties(sf_tuning_dff_neg, response_dir='neg',
                                              elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_neg_peak_sf_raw_dff': peak_sf_raw_dff_neg,
                               'dgc_neg_weighted_sf_raw_dff': weighted_sf_raw_dff_neg,
                               'dgc_neg_weighted_sf_log_raw_dff': weighted_sf_log_raw_dff_neg,
                               'dgc_neg_weighted_sf_ele_dff': weighted_sf_ele_dff_neg,
                               'dgc_neg_weighted_sf_log_ele_dff': weighted_sf_log_ele_dff_neg,
                               'dgc_neg_weighted_sf_rec_dff': weighted_sf_rec_dff_neg,
                               'dgc_neg_weighted_sf_log_rec_dff': weighted_sf_log_rec_dff_neg})

        # tf tuning of dff responses in positive direction ==============================================================
        tf_tuning_dff_pos = dgcrt_dff.get_tf_tuning(response_dir='pos', is_collapse_sf=params['is_collapse_sf'],
                                                  is_collapse_dire=params['is_collapse_dire'])
        peak_tf_raw_dff_pos, weighted_tf_raw_dff_pos, weighted_tf_log_raw_dff_pos, \
        weighted_tf_ele_dff_pos, weighted_tf_log_ele_dff_pos, \
        weighted_tf_rec_dff_pos, weighted_tf_log_rec_dff_pos = \
            dgcrt_dff.get_tf_tuning_properties(tf_tuning_dff_pos, response_dir='pos',
                                              elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_pos_peak_tf_raw_dff': peak_tf_raw_dff_pos,
                               'dgc_pos_weighted_tf_raw_dff': weighted_tf_raw_dff_pos,
                               'dgc_pos_weighted_tf_log_raw_dff': weighted_tf_log_raw_dff_pos,
                               'dgc_pos_weighted_tf_ele_dff': weighted_tf_ele_dff_pos,
                               'dgc_pos_weighted_tf_log_ele_dff': weighted_tf_log_ele_dff_pos,
                               'dgc_pos_weighted_tf_rec_dff': weighted_tf_rec_dff_pos,
                               'dgc_pos_weighted_tf_log_rec_dff': weighted_tf_log_rec_dff_pos})

        # tf tuning of dff responses in negative direction ==============================================================
        tf_tuning_dff_neg = dgcrt_dff.get_tf_tuning(response_dir='neg', is_collapse_sf=params['is_collapse_sf'],
                                                  is_collapse_dire=params['is_collapse_dire'])
        peak_tf_raw_dff_neg, weighted_tf_raw_dff_neg, weighted_tf_log_raw_dff_neg, \
        weighted_tf_ele_dff_neg, weighted_tf_log_ele_dff_neg, \
        weighted_tf_rec_dff_neg, weighted_tf_log_rec_dff_neg = \
            dgcrt_dff.get_tf_tuning_properties(tf_tuning_dff_neg, response_dir='neg',
                                              elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_neg_peak_tf_raw_dff': peak_tf_raw_dff_neg,
                               'dgc_neg_weighted_tf_raw_dff': weighted_tf_raw_dff_neg,
                               'dgc_neg_weighted_tf_log_raw_dff': weighted_tf_log_raw_dff_neg,
                               'dgc_neg_weighted_tf_ele_dff': weighted_tf_ele_dff_neg,
                               'dgc_neg_weighted_tf_log_ele_dff': weighted_tf_log_ele_dff_neg,
                               'dgc_neg_weighted_tf_rec_dff': weighted_tf_rec_dff_neg,
                               'dgc_neg_weighted_tf_log_rec_dff': weighted_tf_log_rec_dff_neg})


        # direction/orientation tuning of zscore responses in positive direction ===========================================
        dire_tuning_z_pos = dgcrt_z.get_dire_tuning(response_dir='pos',
                                                      is_collapse_sf=params['is_collapse_sf'],
                                                      is_collapse_tf=params['is_collapse_tf'])
        osi_z_pos_raw, dsi_z_pos_raw, gosi_z_pos_raw, gdsi_z_pos_raw, \
        osi_z_pos_ele, dsi_z_pos_ele, gosi_z_pos_ele, gdsi_z_pos_ele, \
        osi_z_pos_rec, dsi_z_pos_rec, gosi_z_pos_rec, gdsi_z_pos_rec, \
        peak_dire_raw_z_pos, vs_dire_raw_z_pos, vs_dire_ele_z_pos, vs_dire_rec_z_pos \
            = dgcrt_z.get_dire_tuning_properties(dire_tuning_z_pos,
                                                  response_dir='pos',
                                                  elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_pos_osi_raw_z': osi_z_pos_raw,
                               'dgc_pos_dsi_raw_z': dsi_z_pos_raw,
                               'dgc_pos_gosi_raw_z': gosi_z_pos_raw,
                               'dgc_pos_gdsi_raw_z': gdsi_z_pos_raw,
                               'dgc_pos_osi_ele_z': osi_z_pos_ele,
                               'dgc_pos_dsi_ele_z': dsi_z_pos_ele,
                               'dgc_pos_gosi_ele_z': gosi_z_pos_ele,
                               'dgc_pos_gdsi_ele_z': gdsi_z_pos_ele,
                               'dgc_pos_osi_rec_z': osi_z_pos_rec,
                               'dgc_pos_dsi_rec_z': dsi_z_pos_rec,
                               'dgc_pos_gosi_rec_z': gosi_z_pos_rec,
                               'dgc_pos_gdsi_rec_z': gdsi_z_pos_rec,
                               'dgc_pos_peak_dire_raw_z': peak_dire_raw_z_pos,
                               'dgc_pos_vs_dire_raw_z': vs_dire_raw_z_pos,
                               'dgc_pos_vs_dire_ele_z': vs_dire_ele_z_pos,
                               'dgc_pos_vs_dire_rec_z': vs_dire_rec_z_pos})

        # direction/orientation tuning of zscore responses in negative direction ===========================================
        dire_tuning_z_neg = dgcrt_z.get_dire_tuning(response_dir='neg',
                                                      is_collapse_sf=params['is_collapse_sf'],
                                                      is_collapse_tf=params['is_collapse_tf'])
        osi_z_neg_raw, dsi_z_neg_raw, gosi_z_neg_raw, gdsi_z_neg_raw, \
        osi_z_neg_ele, dsi_z_neg_ele, gosi_z_neg_ele, gdsi_z_neg_ele, \
        osi_z_neg_rec, dsi_z_neg_rec, gosi_z_neg_rec, gdsi_z_neg_rec, \
        peak_dire_raw_z_neg, vs_dire_raw_z_neg, vs_dire_ele_z_neg, vs_dire_rec_z_neg \
            = dgcrt_z.get_dire_tuning_properties(dire_tuning_z_neg,
                                                  response_dir='neg',
                                                  elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_neg_osi_raw_z': osi_z_neg_raw,
                               'dgc_neg_dsi_raw_z': dsi_z_neg_raw,
                               'dgc_neg_gosi_raw_z': gosi_z_neg_raw,
                               'dgc_neg_gdsi_raw_z': gdsi_z_neg_raw,
                               'dgc_neg_osi_ele_z': osi_z_neg_ele,
                               'dgc_neg_dsi_ele_z': dsi_z_neg_ele,
                               'dgc_neg_gosi_ele_z': gosi_z_neg_ele,
                               'dgc_neg_gdsi_ele_z': gdsi_z_neg_ele,
                               'dgc_neg_osi_rec_z': osi_z_neg_rec,
                               'dgc_neg_dsi_rec_z': dsi_z_neg_rec,
                               'dgc_neg_gosi_rec_z': gosi_z_neg_rec,
                               'dgc_neg_gdsi_rec_z': gdsi_z_neg_rec,
                               'dgc_neg_peak_dire_raw_z': peak_dire_raw_z_neg,
                               'dgc_neg_vs_dire_raw_z': vs_dire_raw_z_neg,
                               'dgc_neg_vs_dire_ele_z': vs_dire_ele_z_neg,
                               'dgc_neg_vs_dire_rec_z': vs_dire_rec_z_neg})

        # sf tuning of zscore responses in positive direction ==============================================================
        sf_tuning_z_pos = dgcrt_z.get_sf_tuning(response_dir='pos', is_collapse_tf=params['is_collapse_tf'],
                                                  is_collapse_dire=params['is_collapse_dire'])
        peak_sf_raw_z_pos, weighted_sf_raw_z_pos, weighted_sf_log_raw_z_pos, \
        weighted_sf_ele_z_pos, weighted_sf_log_ele_z_pos, \
        weighted_sf_rec_z_pos, weighted_sf_log_rec_z_pos = \
            dgcrt_z.get_sf_tuning_properties(sf_tuning_z_pos, response_dir='pos',
                                              elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_pos_peak_sf_raw_z': peak_sf_raw_z_pos,
                               'dgc_pos_weighted_sf_raw_z': weighted_sf_raw_z_pos,
                               'dgc_pos_weighted_sf_log_raw_z': weighted_sf_log_raw_z_pos,
                               'dgc_pos_weighted_sf_ele_z': weighted_sf_ele_z_pos,
                               'dgc_pos_weighted_sf_log_ele_z': weighted_sf_log_ele_z_pos,
                               'dgc_pos_weighted_sf_rec_z': weighted_sf_rec_z_pos,
                               'dgc_pos_weighted_sf_log_rec_z': weighted_sf_log_rec_z_pos})

        # sf tuning of zscore responses in negative direction ==============================================================
        sf_tuning_z_neg = dgcrt_z.get_sf_tuning(response_dir='neg', is_collapse_tf=params['is_collapse_tf'],
                                                  is_collapse_dire=params['is_collapse_dire'])
        peak_sf_raw_z_neg, weighted_sf_raw_z_neg, weighted_sf_log_raw_z_neg, \
        weighted_sf_ele_z_neg, weighted_sf_log_ele_z_neg, \
        weighted_sf_rec_z_neg, weighted_sf_log_rec_z_neg = \
            dgcrt_z.get_sf_tuning_properties(sf_tuning_z_neg, response_dir='neg',
                                              elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_neg_peak_sf_raw_z': peak_sf_raw_z_neg,
                               'dgc_neg_weighted_sf_raw_z': weighted_sf_raw_z_neg,
                               'dgc_neg_weighted_sf_log_raw_z': weighted_sf_log_raw_z_neg,
                               'dgc_neg_weighted_sf_ele_z': weighted_sf_ele_z_neg,
                               'dgc_neg_weighted_sf_log_ele_z': weighted_sf_log_ele_z_neg,
                               'dgc_neg_weighted_sf_rec_z': weighted_sf_rec_z_neg,
                               'dgc_neg_weighted_sf_log_rec_z': weighted_sf_log_rec_z_neg})

        # tf tuning of zcore responses in positive direction ==============================================================
        tf_tuning_z_pos = dgcrt_z.get_tf_tuning(response_dir='pos', is_collapse_sf=params['is_collapse_sf'],
                                                  is_collapse_dire=params['is_collapse_dire'])
        peak_tf_raw_z_pos, weighted_tf_raw_z_pos, weighted_tf_log_raw_z_pos, \
        weighted_tf_ele_z_pos, weighted_tf_log_ele_z_pos, \
        weighted_tf_rec_z_pos, weighted_tf_log_rec_z_pos = \
            dgcrt_z.get_tf_tuning_properties(tf_tuning_z_pos, response_dir='pos',
                                              elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_pos_peak_tf_raw_z': peak_tf_raw_z_pos,
                               'dgc_pos_weighted_tf_raw_z': weighted_tf_raw_z_pos,
                               'dgc_pos_weighted_tf_log_raw_z': weighted_tf_log_raw_z_pos,
                               'dgc_pos_weighted_tf_ele_z': weighted_tf_ele_z_pos,
                               'dgc_pos_weighted_tf_log_ele_z': weighted_tf_log_ele_z_pos,
                               'dgc_pos_weighted_tf_rec_z': weighted_tf_rec_z_pos,
                               'dgc_pos_weighted_tf_log_rec_z': weighted_tf_log_rec_z_pos})

        # tf tuning of zscore responses in negative direction ==============================================================
        tf_tuning_z_neg = dgcrt_z.get_tf_tuning(response_dir='neg', is_collapse_sf=params['is_collapse_sf'],
                                                  is_collapse_dire=params['is_collapse_dire'])
        peak_tf_raw_z_neg, weighted_tf_raw_z_neg, weighted_tf_log_raw_z_neg, \
        weighted_tf_ele_z_neg, weighted_tf_log_ele_z_neg, \
        weighted_tf_rec_z_neg, weighted_tf_log_rec_z_neg = \
            dgcrt_z.get_tf_tuning_properties(tf_tuning_z_neg, response_dir='neg',
                                              elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_neg_peak_tf_raw_z': peak_tf_raw_z_neg,
                               'dgc_neg_weighted_tf_raw_z': weighted_tf_raw_z_neg,
                               'dgc_neg_weighted_tf_log_raw_z': weighted_tf_log_raw_z_neg,
                               'dgc_neg_weighted_tf_ele_z': weighted_tf_ele_z_neg,
                               'dgc_neg_weighted_tf_log_ele_z': weighted_tf_log_ele_z_neg,
                               'dgc_neg_weighted_tf_rec_z': weighted_tf_rec_z_neg,
                               'dgc_neg_weighted_tf_log_rec_z': weighted_tf_log_rec_z_neg})

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

                               'dgc_pos_osi_raw_df': np.nan,
                               'dgc_pos_dsi_raw_df': np.nan,
                               'dgc_pos_gosi_raw_df': np.nan,
                               'dgc_pos_gdsi_raw_df': np.nan,
                               'dgc_pos_osi_ele_df': np.nan,
                               'dgc_pos_dsi_ele_df': np.nan,
                               'dgc_pos_gosi_ele_df': np.nan,
                               'dgc_pos_gdsi_ele_df': np.nan,
                               'dgc_pos_osi_rec_df': np.nan,
                               'dgc_pos_dsi_rec_df': np.nan,
                               'dgc_pos_gosi_rec_df': np.nan,
                               'dgc_pos_gdsi_rec_df': np.nan,
                               'dgc_pos_peak_dire_raw_df': np.nan,
                               'dgc_pos_vs_dire_raw_df': np.nan,
                               'dgc_pos_vs_dire_ele_df': np.nan,
                               'dgc_pos_vs_dire_rec_df': np.nan,
                               'dgc_neg_osi_raw_df': np.nan,
                               'dgc_neg_dsi_raw_df': np.nan,
                               'dgc_neg_gosi_raw_df': np.nan,
                               'dgc_neg_gdsi_raw_df': np.nan,
                               'dgc_neg_osi_ele_df': np.nan,
                               'dgc_neg_dsi_ele_df': np.nan,
                               'dgc_neg_gosi_ele_df': np.nan,
                               'dgc_neg_gdsi_ele_df': np.nan,
                               'dgc_neg_osi_rec_df': np.nan,
                               'dgc_neg_dsi_rec_df': np.nan,
                               'dgc_neg_gosi_rec_df': np.nan,
                               'dgc_neg_gdsi_rec_df': np.nan,
                               'dgc_neg_peak_dire_raw_df': np.nan,
                               'dgc_neg_vs_dire_raw_df': np.nan,
                               'dgc_neg_vs_dire_ele_df': np.nan,
                               'dgc_neg_vs_dire_rec_df': np.nan,
                               'dgc_pos_peak_sf_raw_df': np.nan,
                               'dgc_pos_weighted_sf_raw_df': np.nan,
                               'dgc_pos_weighted_sf_log_raw_df': np.nan,
                               'dgc_pos_weighted_sf_ele_df': np.nan,
                               'dgc_pos_weighted_sf_log_ele_df': np.nan,
                               'dgc_pos_weighted_sf_rec_df': np.nan,
                               'dgc_pos_weighted_sf_log_rec_df': np.nan,
                               'dgc_neg_peak_sf_raw_df': np.nan,
                               'dgc_neg_weighted_sf_raw_df': np.nan,
                               'dgc_neg_weighted_sf_log_raw_df': np.nan,
                               'dgc_neg_weighted_sf_ele_df': np.nan,
                               'dgc_neg_weighted_sf_log_ele_df': np.nan,
                               'dgc_neg_weighted_sf_rec_df': np.nan,
                               'dgc_neg_weighted_sf_log_rec_df': np.nan,
                               'dgc_pos_peak_tf_raw_df': np.nan,
                               'dgc_pos_weighted_tf_raw_df': np.nan,
                               'dgc_pos_weighted_tf_log_raw_df': np.nan,
                               'dgc_pos_weighted_tf_ele_df': np.nan,
                               'dgc_pos_weighted_tf_log_ele_df': np.nan,
                               'dgc_pos_weighted_tf_rec_df': np.nan,
                               'dgc_pos_weighted_tf_log_rec_df': np.nan,
                               'dgc_neg_peak_tf_raw_df': np.nan,
                               'dgc_neg_weighted_tf_raw_df': np.nan,
                               'dgc_neg_weighted_tf_log_raw_df': np.nan,
                               'dgc_neg_weighted_tf_ele_df': np.nan,
                               'dgc_neg_weighted_tf_log_ele_df': np.nan,
                               'dgc_neg_weighted_tf_rec_df': np.nan,
                               'dgc_neg_weighted_tf_log_rec_df': np.nan,

                               'dgc_pos_osi_raw_dff': np.nan,
                               'dgc_pos_dsi_raw_dff': np.nan,
                               'dgc_pos_gosi_raw_dff': np.nan,
                               'dgc_pos_gdsi_raw_dff': np.nan,
                               'dgc_pos_osi_ele_dff': np.nan,
                               'dgc_pos_dsi_ele_dff': np.nan,
                               'dgc_pos_gosi_ele_dff': np.nan,
                               'dgc_pos_gdsi_ele_dff': np.nan,
                               'dgc_pos_osi_rec_dff': np.nan,
                               'dgc_pos_dsi_rec_dff': np.nan,
                               'dgc_pos_gosi_rec_dff': np.nan,
                               'dgc_pos_gdsi_rec_dff': np.nan,
                               'dgc_pos_peak_dire_raw_dff': np.nan,
                               'dgc_pos_vs_dire_raw_dff': np.nan,
                               'dgc_pos_vs_dire_ele_dff': np.nan,
                               'dgc_pos_vs_dire_rec_dff': np.nan,
                               'dgc_neg_osi_raw_dff': np.nan,
                               'dgc_neg_dsi_raw_dff': np.nan,
                               'dgc_neg_gosi_raw_dff': np.nan,
                               'dgc_neg_gdsi_raw_dff': np.nan,
                               'dgc_neg_osi_ele_dff': np.nan,
                               'dgc_neg_dsi_ele_dff': np.nan,
                               'dgc_neg_gosi_ele_dff': np.nan,
                               'dgc_neg_gdsi_ele_dff': np.nan,
                               'dgc_neg_osi_rec_dff': np.nan,
                               'dgc_neg_dsi_rec_dff': np.nan,
                               'dgc_neg_gosi_rec_dff': np.nan,
                               'dgc_neg_gdsi_rec_dff': np.nan,
                               'dgc_neg_peak_dire_raw_dff': np.nan,
                               'dgc_neg_vs_dire_raw_dff': np.nan,
                               'dgc_neg_vs_dire_ele_dff': np.nan,
                               'dgc_neg_vs_dire_rec_dff': np.nan,
                               'dgc_pos_peak_sf_raw_dff': np.nan,
                               'dgc_pos_weighted_sf_raw_dff': np.nan,
                               'dgc_pos_weighted_sf_log_raw_dff': np.nan,
                               'dgc_pos_weighted_sf_ele_dff': np.nan,
                               'dgc_pos_weighted_sf_log_ele_dff': np.nan,
                               'dgc_pos_weighted_sf_rec_dff': np.nan,
                               'dgc_pos_weighted_sf_log_rec_dff': np.nan,
                               'dgc_neg_peak_sf_raw_dff': np.nan,
                               'dgc_neg_weighted_sf_raw_dff': np.nan,
                               'dgc_neg_weighted_sf_log_raw_dff': np.nan,
                               'dgc_neg_weighted_sf_ele_dff': np.nan,
                               'dgc_neg_weighted_sf_log_ele_dff': np.nan,
                               'dgc_neg_weighted_sf_rec_dff': np.nan,
                               'dgc_neg_weighted_sf_log_rec_dff': np.nan,
                               'dgc_pos_peak_tf_raw_dff': np.nan,
                               'dgc_pos_weighted_tf_raw_dff': np.nan,
                               'dgc_pos_weighted_tf_log_raw_dff': np.nan,
                               'dgc_pos_weighted_tf_ele_dff': np.nan,
                               'dgc_pos_weighted_tf_log_ele_dff': np.nan,
                               'dgc_pos_weighted_tf_rec_dff': np.nan,
                               'dgc_pos_weighted_tf_log_rec_dff': np.nan,
                               'dgc_neg_peak_tf_raw_dff': np.nan,
                               'dgc_neg_weighted_tf_raw_dff': np.nan,
                               'dgc_neg_weighted_tf_log_raw_dff': np.nan,
                               'dgc_neg_weighted_tf_ele_dff': np.nan,
                               'dgc_neg_weighted_tf_log_ele_dff': np.nan,
                               'dgc_neg_weighted_tf_rec_dff': np.nan,
                               'dgc_neg_weighted_tf_log_rec_dff': np.nan,

                               'dgc_pos_osi_raw_z': np.nan,
                               'dgc_pos_dsi_raw_z': np.nan,
                               'dgc_pos_gosi_raw_z': np.nan,
                               'dgc_pos_gdsi_raw_z': np.nan,
                               'dgc_pos_osi_ele_z': np.nan,
                               'dgc_pos_dsi_ele_z': np.nan,
                               'dgc_pos_gosi_ele_z': np.nan,
                               'dgc_pos_gdsi_ele_z': np.nan,
                               'dgc_pos_osi_rec_z': np.nan,
                               'dgc_pos_dsi_rec_z': np.nan,
                               'dgc_pos_gosi_rec_z': np.nan,
                               'dgc_pos_gdsi_rec_z': np.nan,
                               'dgc_pos_peak_dire_raw_z': np.nan,
                               'dgc_pos_vs_dire_raw_z': np.nan,
                               'dgc_pos_vs_dire_ele_z': np.nan,
                               'dgc_pos_vs_dire_rec_z': np.nan,
                               'dgc_neg_osi_raw_z': np.nan,
                               'dgc_neg_dsi_raw_z': np.nan,
                               'dgc_neg_gosi_raw_z': np.nan,
                               'dgc_neg_gdsi_raw_z': np.nan,
                               'dgc_neg_osi_ele_z': np.nan,
                               'dgc_neg_dsi_ele_z': np.nan,
                               'dgc_neg_gosi_ele_z': np.nan,
                               'dgc_neg_gdsi_ele_z': np.nan,
                               'dgc_neg_osi_rec_z': np.nan,
                               'dgc_neg_dsi_rec_z': np.nan,
                               'dgc_neg_gosi_rec_z': np.nan,
                               'dgc_neg_gdsi_rec_z': np.nan,
                               'dgc_neg_peak_dire_raw_z': np.nan,
                               'dgc_neg_vs_dire_raw_z': np.nan,
                               'dgc_neg_vs_dire_ele_z': np.nan,
                               'dgc_neg_vs_dire_rec_z': np.nan,
                               'dgc_pos_peak_sf_raw_z': np.nan,
                               'dgc_pos_weighted_sf_raw_z': np.nan,
                               'dgc_pos_weighted_sf_log_raw_z': np.nan,
                               'dgc_pos_weighted_sf_ele_z': np.nan,
                               'dgc_pos_weighted_sf_log_ele_z': np.nan,
                               'dgc_pos_weighted_sf_rec_z': np.nan,
                               'dgc_pos_weighted_sf_log_rec_z': np.nan,
                               'dgc_neg_peak_sf_raw_z': np.nan,
                               'dgc_neg_weighted_sf_raw_z': np.nan,
                               'dgc_neg_weighted_sf_log_raw_z': np.nan,
                               'dgc_neg_weighted_sf_ele_z': np.nan,
                               'dgc_neg_weighted_sf_log_ele_z': np.nan,
                               'dgc_neg_weighted_sf_rec_z': np.nan,
                               'dgc_neg_weighted_sf_log_rec_z': np.nan,
                               'dgc_pos_peak_tf_raw_z': np.nan,
                               'dgc_pos_weighted_tf_raw_z': np.nan,
                               'dgc_pos_weighted_tf_log_raw_z': np.nan,
                               'dgc_pos_weighted_tf_ele_z': np.nan,
                               'dgc_pos_weighted_tf_log_ele_z': np.nan,
                               'dgc_pos_weighted_tf_rec_z': np.nan,
                               'dgc_pos_weighted_tf_log_rec_z': np.nan,
                               'dgc_neg_peak_tf_raw_z': np.nan,
                               'dgc_neg_weighted_tf_raw_z': np.nan,
                               'dgc_neg_weighted_tf_log_raw_z': np.nan,
                               'dgc_neg_weighted_tf_ele_z': np.nan,
                               'dgc_neg_weighted_tf_log_ele_z': np.nan,
                               'dgc_neg_weighted_tf_rec_z': np.nan,
                               'dgc_neg_weighted_tf_log_rec_z': np.nan,
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
                                                postprocess=plot_params['dgc_postprocess'])

        r_max_neg = dgcrt_plot.plot_dire_tuning(response_dir='neg', axis=ax_dire_neg,
                                                is_collapse_sf=params['is_collapse_sf'],
                                                is_collapse_tf=params['is_collapse_tf'],
                                                trace_color=plot_params['dire_color_neg'],
                                                lw=plot_params['dire_line_width'],
                                                postprocess=plot_params['dgc_postprocess'])

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

    txt = '{}\n'.format(file_n)
    txt += '\n'
    txt += 'plane name:          {}\n'.format(plane_n)
    txt += 'roi name:            {}\n'.format(roi_n)
    txt += 'depth (um):          {}\n'.format(roi_properties['depth'])
    txt += 'roi area (um^2):     {:.2f}\n'.format(roi_properties['roi_area'])
    # txt += '\n'
    txt += 'trace type:{:>19}\n'.format(params['trace_type'])
    txt += 'response type:{:>14}\n'.format(plot_params['response_type_for_plot'])
    txt += 'dgc postprocess:{:>13}\n'.format(plot_params['dgc_postprocess'])
    txt += '\n'
    txt += 'skewness raw:        {:.2f}\n'.format(roi_properties['skew_raw'])
    txt += 'skewness fil:        {:.2f}\n'.format(roi_properties['skew_fil'])
    # txt += '\n'

    rf_pos_peak_z = max([roi_properties['rf_pos_on_peak_z'],
                         roi_properties['rf_pos_off_peak_z']])
    rf_neg_peak_z = max([roi_properties['rf_neg_on_peak_z'],
                         roi_properties['rf_neg_off_peak_z']])

    if plot_params['response_type_for_plot'] == 'df':
        surfix1 = 'df'
    elif plot_params['response_type_for_plot'] == 'dff':
        surfix1 = 'dff'
    elif plot_params['response_type_for_plot'] == 'zscore':
        surfix1 = 'z'
    else:
        raise LookupError("Do not ',understand 'response_type_for_plot': {}. Should be "
                          "'df 'dff' or 'zscore'.".format(plot_params['response_type_for_plot']))

    if plot_params['dgc_postprocess'] == 'raw':
        surfix2 = 'raw'
    elif plot_params['dgc_postprocess'] == 'elevate':
        surfix2 = 'ele'
    elif plot_params['dgc_postprocess'] == 'rectify':
        surfix2 = 'rec'
    else:
        raise LookupError("Do not ',understand 'response_type_for_plot': {}. Should be "
                          "'raw', 'elevate' or 'rectify'.".format(plot_params['dgc_postprocess']))

    txt += 'dgc_p_anova:         {:.2f}\n'.format(roi_properties['dgc_p_anova_{}'.format(surfix1)])
    txt += '\n'
    txt += 'positive response:\n'
    txt += 'rf_peak_z:           {:.2f}\n'.format(rf_pos_peak_z)
    txt += 'rf_lsi:              {:.2f}\n'.format(roi_properties['rf_pos_lsi'])
    txt += 'dgc_p_ttest:         {:.2f}\n'.format(roi_properties['dgc_pos_p_ttest_{}'.format(surfix1)])
    txt += 'dgc_peak_resp:       {:.2f}\n'.format(roi_properties['dgc_pos_peak_{}'.format(surfix1)])
    txt += 'dgc_OSI:             {:.2f}\n'.format(roi_properties['dgc_pos_osi_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_gOSI:            {:.2f}\n'.format(roi_properties['dgc_pos_gosi_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_DSI:             {:.2f}\n'.format(roi_properties['dgc_pos_dsi_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_gDSI:            {:.2f}\n'.format(roi_properties['dgc_pos_gdsi_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_vs_dire:         {:.2f}\n'.format(roi_properties['dgc_pos_vs_dire_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_weighted_sf:     {:.2f}\n'.format(roi_properties['dgc_pos_weighted_sf'
                                                                     '_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_weighted_sf_log: {:.2f}\n'.format(roi_properties['dgc_pos_weighted_sf_log'
                                                                     '_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_weighted_tf:     {:.2f}\n'.format(roi_properties['dgc_pos_weighted_tf'
                                                                     '_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_weighted_tf_log: {:.2f}\n'.format(roi_properties['dgc_pos_weighted_tf_log'
                                                                     '_{}_{}'.format(surfix2, surfix1)])
    txt += '\nnegative response:\n'
    txt += 'rf_peak_z:           {:.2f}\n'.format(rf_neg_peak_z)
    txt += 'rf_lsi:              {:.2f}\n'.format(roi_properties['rf_neg_lsi'])
    txt += 'dgc_p_ttest:         {:.2f}\n'.format(roi_properties['dgc_neg_p_ttest_{}'.format(surfix1)])
    txt += 'dgc_peak_resp:       {:.2f}\n'.format(roi_properties['dgc_neg_peak_{}'.format(surfix1)])
    txt += 'dgc_OSI:             {:.2f}\n'.format(roi_properties['dgc_neg_osi_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_gOSI:            {:.2f}\n'.format(roi_properties['dgc_neg_gosi_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_DSI:             {:.2f}\n'.format(roi_properties['dgc_neg_dsi_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_gDSI:            {:.2f}\n'.format(roi_properties['dgc_neg_gdsi_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_vs_dire:         {:.2f}\n'.format(roi_properties['dgc_neg_vs_dire_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_weighted_sf:     {:.2f}\n'.format(roi_properties['dgc_neg_weighted_sf'
                                                                     '_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_weighted_sf_log: {:.2f}\n'.format(roi_properties['dgc_neg_weighted_sf_log'
                                                                     '_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_weighted_tf:     {:.2f}\n'.format(roi_properties['dgc_neg_weighted_tf'
                                                                     '_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_weighted_tf_log: {:.2f}\n'.format(roi_properties['dgc_neg_weighted_tf_log'
                                                                     '_{}_{}'.format(surfix2, surfix1)])

    ax_text.text(0.01, 0.99, txt, horizontalalignment='left', verticalalignment='top', family='monospace')

    # plt.show()
    return f


class BoutonClassifier(object):

    def __init___(self, skew_filter_sigma=5., skew_thr=0.6, lowpass_sigma=0.1, detrend_sigma=3.,
                  event_std_thr=3., peri_event_dur=(-3, 3), corr_len_thr=300., corr_abs_thr=0.5,
                  corr_std_thr=3., is_cosine_similarity=False, distance_metric='cosine',
                  linkage_method='weighted', distance_thr=1.0):
        """
        initiate the object. setup a bunch of analysis parameters.

        for detailed the bouton classification method, please see: Liang et al, Cell, 2018, 173:1343

        There are a few simplifications that I found fitted better to my data.
        1. not necessary to run cosine similarity to get distance matrix.
        2. not necessary to use x std above mean to threshold correlation coefficient matrix, one absolute value is
           enough
        3. the application of distance matrix is somewhat different from the scipy documentation. The documentation
           says feed y matrix to scipy.cluster.hierarchy.linkage() will generate the corrected linkage. But it is
           not the clear what

        :param skew_filter_sigma: float, in second, sigma for gaussian filter for skewness
        :param skew_thr: float, threshold of skewness of filtered trace to pickup responsive traces
        :param lowpass_sigma: float, in second, sigma for gaussian filter to highpass single trace
        :param detrend_sigma: float, in second, sigma for gaussian filter to remove slow trend
        :param event_std_thr: float, how many standard deviation above mean to detect events
        :param peri_event_dur: list of two floats, in seconds, pre- and post- duration to be included into detected
                               events
        :param corr_len_thr: float, in seconds, length threshold to calculate correlation between a pair
                                       of two traces. if the length is too short (detected events are too few),
                                       their correlation coefficient will be set to 0.
        :param corr_abs_thr: float, [0, 1], absolute threshold to treat correlation coefficient matrix
        :param corr_std_thr: float, how many standard deviation above the mean to threshold correlation
                                    coefficient matrix for each roi (currently not implemented)
        :param is_cosine_similarity: bool, if True: use cosine similarity to calculate distance matrix
                                           if False: use 1 - thresholded correlation coefficient matrix as distance
                                           matrix
        :param distance_metric: str, metric for scipy to get distance. "metric" input to scipy.spatial.distance.pdist()
                                method and scipy.cluster.hierarchy.linkage method.
        :param linkage_method: str, method argument to the scipy.cluster.hierarchy.linkage() method
        :param distance_thr: float, positive, the distance threshold to classify boutons into axons from the linkage
                             array
        """

        self.skew_filter_sigma = float(skew_filter_sigma)
        self.skew_thr = float(skew_thr)
        self.lowpass_sigma = float(lowpass_sigma)
        self.detrend_sigma = float(detrend_sigma)
        self.event_std_thr = float(event_std_thr)
        self.peri_event_dur = tuple(peri_event_dur)
        self.corr_len_thr = float(corr_len_thr)
        self.corr_abs_thr = float(corr_abs_thr)
        self.corr_std_thr = float(corr_std_thr)
        self.is_cosine_similarity = bool(is_cosine_similarity)
        self.distance_metric = str(distance_metric)
        self.linkage_method = str(linkage_method)
        self.distance_thr = float(distance_thr)

    def filter_traces(self, traces, roi_ns, sample_dur):
        """
        filter traces by filtered skewness, also detect events for each traces
        :param traces: n x m array, n: roi numbers, m: time points
        :param roi_ns: list of strings, length = n, name of all rois
        :param sample_dur: float, duration of each sample in second.
        :return traces_res: l x m array, l number of rois that pass the skewness thresold, self.skew_thr
        :return roi_ns_res: list of strings, length = l, name of these rois
        :return event_masks: l x m array, dtype: np.bool, event masks for each roi. events are deteced by
                             larger than trace_mean + self.event_std_thr * trace_std
        """

        if traces.shape[0] != len(roi_ns):
            raise ValueError('traces.shape[0] ({}) should be the same as len(roi_ns) ({})'.format(traces.shape[0],
                                                                                                  len(roi_ns)))

        lowpass_sig_pt = self.lowpass_sigma / sample_dur
        detrend_sig_pt = self.detrend_sigma / sample_dur

        trace_ts = np.arange(traces.shape[1]) * sample_dur

        event_start_pt = int(np.floor(self.peri_event_dur[0] / sample_dur))
        event_end_pt = int(np.ceil(self.peri_event_dur[1] / sample_dur))

        roi_ns_res = []
        traces_res = []
        event_masks = []

        for trace_i, trace in enumerate(traces):
            _, skew_fil = sca.get_skewness(trace=trace, ts=trace_ts,
                                           filter_length=self.skew_filter_sigma)

            if skew_fil >= self.skew_thr:

                trace_l = ni.gaussian_filter1d(trace, sigma=lowpass_sig_pt) # lowpass
                trace_d = trace_l - ni.gaussian_filter1d(trace_l, sigma=detrend_sig_pt) # detrend

                # get event masks
                event_mask = np.zeros(trace_ts.shape, dtype=np.bool)

                trace_mean = np.mean(trace_d)
                trace_std = np.std(trace_d)
                event_intervals = ta.threshold_to_intervals(trace=trace_d,
                                                            thr=trace_mean + self.event_std_thr * trace_std,
                                                            comparison='>=')

                for inte in event_intervals:
                    start_ind = max([0, inte[0] + event_start_pt])
                    end_ind = min([inte[1] + event_end_pt, traces.shape[1]])
                    event_mask[start_ind : end_ind] = True

                roi_ns_res.append(roi_ns[trace_i])
                traces_res.append(trace_d)
                event_masks.append(event_mask)

        return traces_res, roi_ns_res, event_masks

    def get_correlation_coefficient_matrix(self, traces, event_masks, sample_dur, is_plot=False):
        """
        calculate event based correcation coefficient matrix of a set of rois.
        ideally, the traces and event_masks will be the output of self.filter_traces() method.

        :param traces: l x m array, l: number of rois, m: number of time points
        :param event_masks: array same size of traces, dtype=np.bool, masks of event for each trace.
        :param sample_dur:  float, duration of each sample in second.
        :param is_plot: bool
        :return mat_corr: l x l array, correlation coefficient matrix
        """

        roi_num_res = traces.shape[0]
        mat_corr = np.zeros((roi_num_res, roi_num_res))
        np.fill_diagonal(mat_corr, 1.)

        for i in range(0, roi_num_res - 1):
            for j in range(i + 1, roi_num_res):
                trace_i = traces[i]
                event_mask_i = event_masks[i]

                trace_j = traces[j]
                event_mask_j = event_masks[j]

                res_ind_merge = np.logical_or(event_mask_i, event_mask_j)
                #         print("({}, {}), trace_length: {}".format(i, j, np.sum(res_ind_merge)))

                if np.sum(res_ind_merge) < self.corr_len_thr // sample_dur:
                    mat_corr[i, j] = 0
                    mat_corr[j, i] = 0
                else:
                    trace_i = trace_i[res_ind_merge]
                    trace_j = trace_j[res_ind_merge]
                    coeff = np.corrcoef(np.array([trace_i, trace_j]), rowvar=True)
                    mat_corr[i, j] = coeff[1, 0]
                    mat_corr[j, i] = coeff[1, 0]

        if is_plot:
            f = plt.figure(figsize=(8, 6))
            ax = f.add_subplot(111)
            ax.set_title('corr coef matrix')
            fig = ax.imshow(mat_corr, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
            f.colorbar(fig)
            plt.show()

        return mat_corr

    def threshold_correlation_coefficient_matrix(self, mat_corr, is_plot=False):
        """
        threshold correlation coefficient matrix based on each roi

        for each roi, the corr coeff smaller than min([self.corr_abs_thr, mean + self.corr_std_thr * std])
        will be set zero

        :param mat_corr:
        :param is_plot:
        :return:
        """

        mask = np.ones(mat_corr.shape)

        for row_i, row in enumerate(mat_corr):
            curr_std = np.std(row)
            curr_mean = np.mean(row)
            curr_thr = min([self.corr_abs_thr, curr_mean + self.corr_std_thr * curr_std])

            mask[row_i, :][row < curr_thr] = 0.
            mask[:, row_i][row < curr_thr] = 0.

        mat_corr_thr = mat_corr * mask

        if is_plot:
            f = plt.figure(figsize=(8, 6))
            ax = f.add_subplot(111)
            ax.set_title('thresholded corr coef matrix')
            fig = ax.imshow(mat_corr, cmap='plasma', vmin=0, vmax=1, interpolation='nearest')
            f.colorbar(fig)
            plt.show()

        return mat_corr_thr

    def get_distance_matrix(self, mat_corr, is_plot=False):
        """
        calculated the distance matrix from correlation coefficient matrix.
        if self.is_cosine_similarity is True, use the cosine similarity method described in the paper
        if self.is_cosine_similarity is False, simply use 1 - corr coeff as distance

        :param mat_corr:
        :param is_plot:
        :return:
        """

        if self.is_cosine_similarity:

            mat_dis = np.zeros(mat_corr.shape)
            roi_num = mat_dis.shape[0]
            # print('total roi number: {}'.format(roi_num))
            for i in range(roi_num):
                for j in range(i+1, roi_num, 1):

                    ind = np.ones(roi_num, dtype=np.bool)
                    ind[i] = 0
                    ind[j] = 0

                    row_i = mat_corr[i][ind]
                    row_j = mat_corr[j][ind]

                    if max(row_i) == 0 or max(row_j) == 0:
                        mat_dis[i, j] = 1
                        mat_dis[j, i] = 1
                    else:
                        cos = spatial.distance.cosine(row_i, row_j)
                        mat_dis[i, j] = 1 - cos
                        mat_dis[j, i] = 1 - cos
        else:
            mat_dis = 1 - mat_corr

        if is_plot:
            f = plt.figure(figsize=(8, 6))
            ax = f.add_subplot(111)
            ax.set_title('distance matrix')
            fig = ax.imshow(mat_dis, cmap='plasma', vmin=0, vmax=1, interpolation='nearest')
            f.colorbar(fig)
            plt.show()

        return mat_dis

    def hierarchy_clustering(self, mat_dis, is_plot=False, **kwargs):
        """

        cluster the boutons based on the distance matrix using scipy.cluster.hierarchy.linkage function
        the "method" argument of this function is defined by self.linkage_method

        :param mat_dis: 2d array, distance matrix
        :param is_plot: bool
        :param kwargs: other inputs to scipy.cluster.hierarchy.dendrogram function
        :return linkage_z: 2d array, the linkage array Z from scipy.cluster.hierarchy.linkage method
        :return mat_dis_reorg: 2d array, reorganized the distance matrix based on the clustering
        :return c: float, the cophentic correlation distance of the clustering. Value range: [0, 1].
                   Better if it is more close to 1.
        """

        linkage_z = cluster.hierarchy.linkage(mat_dis, method=self.linkage_method, metric=self.distance_metric)

        c = cluster.hierarchy.cophenet(linkage_z, spatial.distance.pdist(mat_dis, metric=self.distance_metric))

        print('Cophentic correlation distance of clustering: {}'.format(c))

        # reorganize distance matrix
        clu = cluster.hierarchy.fcluster(linkage_z, t=0, criterion='distance')
        mat_0 = np.zeros(mat_dis.shape)

        for l_i, l in enumerate(clu):
            mat_0[l - 1, :] = mat_dis[l_i, :]

        mat_dis_reorg = np.zeros(mat_dis.shape)
        for l_i, l in enumerate(clu):
            mat_dis_reorg[:, l - 1] = mat_0[:, l_i]

        if is_plot:
            f_den = plt.figure(figsize=(20, 8))
            ax_den = f_den.add_subplot(111)
            _ = cluster.hierarchy.dendrogram(linkage_z, ax=ax_den, **kwargs)
            ax_den.axhline(y=self.distance_thr)

            f_mat = plt.figure(figsize=(8, 6))
            ax_mat = f_mat.add_subplot(111)
            fig = ax_mat.imshow(mat_dis_reorg, cmap='plasma_r', vmin=0, vmax=1, interpolation='nearest')
            ax_mat.set_title('clustered distance matrix')
            f_mat.colorbar(fig)

            plt.show()

        return linkage_z, mat_dis_reorg, c

    def get_axon_dict(self, linkage_z, roi_ns):
        """
        generate a dictionary of clustered axons.

        :param linkage_z: 2d array, the linkage array genearted by scipy.cluster.hierarchy.linkage method
        :param roi_ns: list of strings, roi names for the rois used for clustering
        :return axon_dict: dictionary of axons, each entry is {<axon_name> : list of roi names belong to that axon}
        :return clu_axon: list of integers, cluster index list generated by scipy.cluster.hierarchy.fcluster method
        """

        clu_axon = cluster.hierarchy.fcluster(linkage_z, t=self.distance_thr, criterion='distance')
        clu_axon = np.array(clu_axon)
        clu_axon = clu_axon - 1 # change to zero based indexing

        axon_num = max(clu_axon) + 1

        roi_ns = np.array(roi_ns)

        # get axon dictionary
        axon_dict = {}
        axon_num_multi_roi = 0

        for axon_i in range(axon_num):
            axon_n = 'axon_{:04d}'.format(axon_i)
            axon_lst = roi_ns[clu_axon == axon_i]
            axon_dict.update({axon_n: axon_lst})

            if len(axon_lst) > 1:
                axon_num_multi_roi += 1

        print('total number of axons: {}; '
              'number of axons with multiple rois: {}'.format(axon_num, axon_num_multi_roi))

        return axon_dict, clu_axon

    @staticmethod
    def plot_chunked_traces_with_intervals(traces, event_masks=None, chunk_num=4, fig_obj=None, **kwarg):
        """
        plot traces in defined number of chunks. Also mark the defined period indicated by the marked_inds

        :param traces: 2d array, float, each row is a single trace
        :param event_masks: 2d array, bool, same size as traces, masks of detected events for each roi
        :param chunk_num: int, number of chunks for plotting
        :param fig_obj: matplotlib.figure object
        :param **kwarg: inputs to matplotlib.axes.plot() function
        """

        if len(traces.shape) == 1:
            traces = np.array([traces])

        if event_masks is not None and len(event_masks.shape) == 1:
            event_masks = np.array([event_masks])

        if event_masks is not None and traces.shape != event_masks.shape:
            raise ValueError(
                'the shape of input "traces" ({}) and "event_masks" ({}) are not the same.'.format(traces.shape,
                                                                                                   event_masks.shape))

        if fig_obj is None:
            fig_obj = plt.figure(figsize=(15, 10))

        colors = pt.random_color(traces.shape[0])
        len_tot = traces.shape[1]
        len_chunk = len_tot // chunk_num

        for chunk_i in range(chunk_num):
            chunk_traces = traces[:, chunk_i * len_chunk: (chunk_i + 1) * len_chunk]
            chunk_ax = fig_obj.add_subplot(chunk_num, 1, chunk_i + 1)
            chunk_ax.set_xlim([0, len_chunk])

            for trace_i in range(traces.shape[0]):
                chunk_ax.plot(chunk_traces[trace_i], color=colors[trace_i], **kwarg)

            if event_masks is not None:
                chunk_inds = event_masks[:, chunk_i * len_chunk: (chunk_i + 1) * len_chunk]
                chunk_ind = np.any(chunk_inds, axis=0)
                chunk_intes = ta.threshold_to_intervals(trace=chunk_ind.astype(np.float32), thr=0.5, comparison='>=')
                for chunk_int in chunk_intes:
                    chunk_ax.axvspan(chunk_int[0], chunk_int[1], color='#ff0000', alpha=0.2)


if __name__ == '__main__':

    # ===================================================================================================
    nwb_f = h5py.File(r"G:\bulk_LGN_database\nwbs\190404_M439939_110_repacked.nwb")
    uc_inds, _ = get_UC_ts_mask(nwb_f=nwb_f, plane_n='plane0')
    plt.plot(uc_inds)
    plt.show()

    dgc_spont_inds, _ = get_DGC_spont_ts_mask(nwb_f=nwb_f, plane_n='plane0')
    plt.plot(dgc_spont_inds)
    plt.show()
    # ===================================================================================================

    # ===================================================================================================
    # nwb_f = h5py.File(r"Z:\chandelier_cell_project\M447219\2019-06-25-deepscope\190625_M447219_110.nwb", 'r')
    # triplets = get_roi_triplets(nwb_f=nwb_f, overlap_ratio=0.9)
    # print('\n'.join([str(t) for t in triplets]))
    # nwb_f.close()
    # ===================================================================================================

    # ===================================================================================================
    # nwb_path = r"F:\data2\chandelier_cell_project\M455115\2019-06-06-deepscope\190606_M455115_110.nwb"
    # nwb_f = h5py.File(nwb_path, 'r')
    # pupil_area, pupil_ts = get_pupil_area(nwb_f=nwb_f,
    #                                       module_name='eye_tracking_right',
    #                                       ell_thr=0.5,
    #                                       median_win=3.)
    # plt.figure(figsize=(20, 5))
    # plt.plot(pupil_ts, pupil_area)
    # plt.show()
    # ===================================================================================================

    # ===================================================================================================
    # nwb_path = r"F:\data2\chandelier_cell_project\M441626\2019-03-26-deepscope\190326_M441626_110.nwb"
    # nwb_path = r"G:\repacked\190326_M439939_110_repacked.nwb"
    # nwb_path = r"F:\data2\rabies_tracing_project\M439939\2019-04-03-2p\190403_M439939_110.nwb"
    # nwb_path = r"/media/nc-ophys/Jun/bulk_LGN_database/nwbs/190508_M439939_110_repacked.nwb"
    # plane_n = 'plane0'
    # roi_n = 'roi_0000'
    # nwb_f = h5py.File(nwb_path, 'r')
    #
    # roi_properties, _, _, _, _, _, _, _, _, _, _, _, _, _ = \
    #     get_everything_from_roi(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n)
    #
    # keys = roi_properties.keys()
    # keys.sort()
    # for key in keys:
    #     print('{}: {}'.format(key, roi_properties[key]))
    #
    # roi_page_report(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n)
    #
    # nwb_f.close()
    # plt.show()
    # ===================================================================================================

    #===================================================================================================
    # coords_roi = np.array([[50, 60], [100, 200], [300, 400]])
    # coords_rf = np.array([[0., 35.], [10., 70.], [0., 70.]])
    # f = plt.figure()
    # ax_alt = f.add_subplot(121)
    # ax_azi = f.add_subplot(122)
    # plot_roi_retinotopy(coords_roi=coords_roi, coords_rf=coords_rf, ax_alt=ax_alt, ax_azi=ax_azi,
    #                     cmap='viridis', canvas_shape=(512, 512), edgecolors='#000000', linewidths=0.5)
    # plt.show()
    # ===================================================================================================