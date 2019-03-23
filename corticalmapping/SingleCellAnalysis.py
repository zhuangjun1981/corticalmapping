import warnings
import numpy as np
import matplotlib.pyplot as plt
import core.PlottingTools as pt
import core.ImageAnalysis as ia
import scipy.ndimage as ni
import scipy.interpolate as ip
import scipy.stats as stats
import math
import h5py
from pandas import DataFrame
from corticalmapping.core.ImageAnalysis import ROI, WeightedROI

warnings.simplefilter('always', RuntimeWarning)

def get_sparse_noise_onset_index(sparseNoiseDisplayLog):
    """
    return the indices of visual display frames for each square in a sparse noise display

    return:
    allOnsetInd: the indices of frames for each square, list
    onsetIndWithLocationSign: indices of frames for each white square, list with element structure [np.array([alt, azi]),sign,[list of indices]]
    """

    frames = sparseNoiseDisplayLog['presentation']['displayFrames']
    frames = [tuple([np.array([x[1][1], x[1][0]]), x[2], x[3], i]) for i, x in enumerate(frames)]
    dtype = [('location', np.ndarray), ('sign', int), ('isOnset', int), ('index', int)]
    frames = np.array(frames, dtype=dtype)

    allOnsetInd = []
    for i in range(len(frames)):
        if frames[i]['isOnset'] == 1 and (i == 0 or frames[i - 1]['isOnset'] == -1):
            allOnsetInd.append(i)

    onsetFrames = frames[allOnsetInd]

    allSquares = list(set([tuple([x[0][0], x[0][1], x[1]]) for x in onsetFrames]))

    onsetIndWithLocationSign = []

    for square in allSquares:
        indices = []
        for onsetFrame in onsetFrames:
            if onsetFrame['location'][0] == square[0] and onsetFrame['location'][1] == square[1] and onsetFrame[
                'sign'] == square[2]:
                indices.append(onsetFrame['index'])

        onsetIndWithLocationSign.append([np.array([square[0], square[1]]), square[2], indices])

    return allOnsetInd, onsetIndWithLocationSign


def get_peak_weighted_roi(arr, thr):
    """
    return: a WeightROI object representing the mask which contains the peak of arr and cut by the thr (thr)
    """
    nanLabel = np.isnan(arr)
    arr2 = arr.copy()
    arr2[nanLabel] = np.nanmin(arr)
    labeled, _ = ni.label(arr2 >= thr)
    peakCoor = np.array(np.where(arr2 == np.amax(arr2))).transpose()[0]
    peakMask = ia.get_marked_masks(labeled, peakCoor)
    if peakMask is None:
        'Threshold too high! No ROI found. Returning None'; return None
    else:
        return WeightedROI(arr2 * peakMask)


def plot_2d_receptive_field(mapArray, altPos, aziPos, plot_axis=None, **kwargs):
    """
    plot a 2-d receptive field in a given axis

    :param mapArray: 2-d array, should be in the same coordinate system as meshgrid(aziPos,altPos)
    :param altPos: 1-d array, list of sample altitude positions, sorted from high to low
    :param aziPos: 1-d array, list of sample azimuth position, sorted from low to high
    :param plot_axis:
    :param kwargs: input to matplotlib.pyplot.imshow() function
    :return: plot_axis
    """

    if plot_axis == None:
        f = plt.figure(figsize=(10, 10))
        plot_axis = f.add_subplot(111)

    fig = plot_axis.imshow(mapArray, **kwargs)
    plot_axis.set_yticks(np.arange(len(altPos)))
    plot_axis.set_xticks(np.arange(len(aziPos)))
    plot_axis.set_yticklabels(altPos.astype(np.int))
    plot_axis.set_xticklabels(aziPos.astype(np.int))
    return fig


def merge_weighted_rois(roi1, roi2):
    """
    merge two WeightedROI objects, most useful for merge ON and OFF subfields
    """
    if (roi1.pixelSizeX != roi2.pixelSizeX) or (roi1.pixelSizeY != roi2.pixelSizeY):
        raise ValueError, 'The pixel sizes of the two WeightedROI objects should match!'

    if roi1.pixelSizeUnit != roi2.pixelSizeUnit:
        raise ValueError, 'The pixel size units of the two WeightedROI objects should match!'

    mask1 = roi1.get_weighted_mask()
    mask2 = roi2.get_weighted_mask()

    return WeightedROI(mask1 + mask2, pixelSize=[roi1.pixelSizeY, roi1.pixelSizeX], pixelSizeUnit=roi1.pixelSizeUnit)


def merge_binary_rois(roi1, roi2):
    """
    merge two ROI objects, most useful for merge ON and OFF subfields
    """
    if (roi1.pixelSizeX != roi2.pixelSizeX) or (roi1.pixelSizeY != roi2.pixelSizeY):
        raise ValueError, 'The pixel sizes of the two WeightedROI objects should match!'

    if roi1.pixelSizeUnit != roi2.pixelSizeUnit:
        raise ValueError, 'The pixel size units of the two WeightedROI objects should match!'

    mask1 = roi1.get_binary_mask()
    mask2 = roi2.get_binary_mask()
    mask3 = np.logical_or(mask1, mask2).astype(np.int8)

    return ROI(mask3, pixelSize=[roi1.pixelSizeY, roi1.pixelSizeX], pixelSizeUnit=roi1.pixelSizeUnit)


def get_dff(traces, t_axis, response_window, baseline_window):
    """

    :param traces: 3d array, roi x trial x time points
    :param t_axis: local timestamps of sta responses
    :param response_window:
    :param baseline_window:
    :return dffs_trial: 3d array, roi x trial x 1, list of dffs for each roi, each_trial
    :return dffs_mean: 1d array, mean dff of each roi, collapsed across trials before dff calculation
    """

    baseline_ind = np.logical_and(t_axis > baseline_window[0], t_axis <= baseline_window[1])
    response_ind = np.logical_and(t_axis > response_window[0], t_axis <= response_window[1])

    baselines = np.mean(traces[:, :, baseline_ind], axis=2, keepdims=True)
    responses = np.mean(traces[:, :, response_ind], axis=2, keepdims=True)

    dffs_trial = (responses - baselines) /  baselines

    traces_mean = np.mean(traces, axis=1) # roi x time points
    baselines_mean = np.mean(traces_mean[:, baseline_ind], axis=1)
    responses_mean = np.mean(traces_mean[:, response_ind], axis=1)
    dffs_mean = (responses_mean - baselines_mean) / baselines_mean

    return dffs_trial, dffs_mean.squeeze()


def get_df(traces, t_axis, response_window, baseline_window):
    """

    :param traces: 3d array, roi x trial x time points
    :param t_axis: local timestamps of sta responses
    :param response_window:
    :param baseline_window:
    :return dfs_trial: 3d array, roi x trial x 1, list of dffs for each roi, each_trial
    :return dfs_mean: 1d array, mean df of each roi
    """

    baseline_ind = np.logical_and(t_axis > baseline_window[0], t_axis <= baseline_window[1])
    response_ind = np.logical_and(t_axis > response_window[0], t_axis <= response_window[1])

    baselines = np.mean(traces[:, :, baseline_ind], axis=2, keepdims=True)
    responses = np.mean(traces[:, :, response_ind], axis=2, keepdims=True)

    dfs_trial = responses - baselines

    dfs_mean = np.mean(dfs_trial, axis=1).squeeze()

    return dfs_trial, dfs_mean


def get_df_dff_trace(trace, t_axis, baseline_window):

    baseline_ind = np.logical_and(t_axis > baseline_window[0], t_axis <= baseline_window[1])

    baseline = np.mean(trace[baseline_ind])

    trace_df = trace - baseline
    trace_dff = trace_df / baseline

    return trace_df, trace_dff


def get_skewness(trace, ts, filter_length=5.):
    """
    calculate skewness of a calcium trace, returns the skewness of input trace and the skewness of the trace after
    removing slow trend. Because slow drifting trend creates artificial and confounding skewness other than calcium
    signal.

    :param trace: 1d array
    :param ts: 1d array, timestamps of the input trace in seconds
    :param filter_length: float, second, the length to filter input trace to get slow trend
    :return skew_o: skewness of original input trace
    :return skew_d: skewness of detrended trace
    """

    fs = 1. / np.mean(np.diff(ts))
    sigma = float(filter_length) * fs
    skew_o = stats.skew(trace)

    trend = ni.gaussian_filter1d(trace, sigma=sigma)
    trace_d = trace - trend
    skew_d = stats.skew(trace_d)

    return skew_o, skew_d


def get_dgc_condition_name(alt, azi, sf, tf, dire, con, rad):
    return 'alt{:06.1f}_azi{:06.1f}_sf{:04.2f}_tf{:04.1f}_dire{:03d}_con{:04.2f}_rad{:03d}'.format(alt,
                                                                                                   azi,
                                                                                                   sf,
                                                                                                   tf,
                                                                                                   dire,
                                                                                                   con,
                                                                                                   rad)


def get_dgc_condition_params(condi_name):
    alt = float(condi_name[3:9])
    azi = float(condi_name[13:19])
    sf = float(condi_name[22:26])
    tf = float(condi_name[29:33])
    dire = int(condi_name[38:41])
    con = float(condi_name[45:49])
    rad = int(condi_name[53:56])
    return alt, azi, sf, tf, dire, con, rad


def get_dgc_response_matrix_from_h5(h5_grp, roi_ind, trace_type='sta_f_center_subtracted'):

    sta_ts = h5_grp.attrs['sta_timestamps']

    dgcrt = DataFrame([], columns=['alt', 'azi', 'sf', 'tf', 'dire', 'con', 'rad', 'onset_ts', 'matrix'])

    condi_ns = h5_grp.keys()
    condi_ns.sort()

    for condi_i, condi_n in enumerate(condi_ns):

        condi_grp = h5_grp[condi_n]

        alt, azi, sf, tf, dire, con, rad = get_dgc_condition_params(condi_name=condi_n)

        if 'global_trigger_timestamps' in condi_grp.attrs:
            onset_ts = condi_grp.attrs['global_trigger_timestamps']
        else:
            onset_ts = []

        matrix = condi_grp[trace_type][roi_ind, :, :]

        dgcrt.loc[condi_i, 'alt'] = alt
        dgcrt.loc[condi_i, 'azi'] = azi
        dgcrt.loc[condi_i, 'sf'] = sf
        dgcrt.loc[condi_i, 'tf'] = tf
        dgcrt.loc[condi_i, 'dire'] = dire
        dgcrt.loc[condi_i, 'con'] = con
        dgcrt.loc[condi_i, 'rad'] = rad
        dgcrt.loc[condi_i, 'onset_ts'] = onset_ts
        dgcrt.loc[condi_i, 'matrix'] = matrix

    return DriftingGratingResponseMatrix(sta_ts=sta_ts, trace_type=trace_type, data=dgcrt)


def get_local_similarity_index(mask1, mask2):
    """
    calculate local similarity index between two receptive field maps

    LSI = sum(mask1 x mask2) / sqrt( sum(mask1 x mask1) * sum(mask2 x mask2))

    DOI: https://doi.org/10.1523/JNEUROSCI.0863-13.2013

    :param mask1: 2d array
    :param mask2: 2d array
    :return:
    """

    if not len(mask1.shape) == len(mask2.shape) == 2:
        raise ValueError('mask1 and mask2 should both be 2d array with same shape.')


    value1 = np.sum((mask1 * mask2).flat)
    value2 = np.sqrt(np.sum((mask1 * mask1).flat) * np.sum((mask2 * mask2).flat))

    return value1 / value2


def dire2ori(dire):
    """
    convert grating drifting direction to grating orientation, unit: degrees
    direction: right is 0 degree, increase counterclockwise
    orientation: horizontal 0 degree, increase counterclockwise
    """
    return (dire + 90) % 180


class SpatialReceptiveField(WeightedROI):
    """
    Object for spatial receptive field, a subclass of WeightedROI object
    """

    def __init__(self, mask, altPos, aziPos, sign=None, temporalWindow=None, pixelSizeUnit=None, dataType=None,
                 thr=None, filter_sigma=None, interpolate_rate=None):
        """
        subclass of WeightedROI object, because the pixel coordinates are defined by np.meshgrid(aziPos, altPos),
        the old WeightedROI attribute: pixelSize does not make sense, so set it to be None.

        sign: sign of the receptive, stf, 'ON', 'OFF', 'ON_OFF', None if not defined
        dataType: type of data stored, str, example can be 'df/f', 'zscore', or 'firing_rate' ...

        thr: None, float, if not applied
        filter_sigma: gaussian filter sigma in pixel, float, None if not applied
        interpolate_rate: rate for interpolation, int, None if not applied

        the correct way to process RF: gaussian filter first, interpolation second, and thr third
        """
        super(SpatialReceptiveField, self).__init__(mask, pixelSize=None, pixelSizeUnit=pixelSizeUnit)
        self.altPos = altPos
        self.aziPos = aziPos
        self.dataType = dataType

        if (sign is None or sign == 'ON' or sign == 'OFF' or sign == 'ON_OFF'):
            self.sign = sign
        elif sign == 1:
            self.sign = 'ON'
        elif sign == -1:
            self.sign = 'OFF'
        else:
            raise ValueError('sign should be 1, -1, "ON", "OFF", "ON_OFF" or None!')
        self.temporalWindow = temporalWindow
        self.thr = thr
        self.filter_sigma = filter_sigma

        if interpolate_rate is None:
            self.interpolate_rate = interpolate_rate
        else:
            if interpolate_rate > 1:
                self.interpolate_rate = interpolate_rate
            else:
                raise ValueError('interpolate_rate should be larger than 1!')

    def get_name(self):

        name = []

        if self.sign is not None:
            name.append(str(self.sign))
        if self.dataType is not None:
            name.append(self.dataType)

        name.append('RF')

        if self.thr is not None:
            name.append('thr:' + str(self.thr)[0:3])
        else:
            name.append('thr:None')

        if self.filter_sigma is not None:
            name.append('sigma:' + str(self.filter_sigma))
        else:
            name.append('sigma:None')

        if self.interpolate_rate is not None:
            name.append('interp:' + str(self.interpolate_rate))
        else:
            name.append('interp:None')

        return ' '.join(name)

    def plot_rf(self, plot_axis=None, is_colorbar=False, cmap='Reds', interpolation='nearest', **kwargs):
        '''
        return display image (RGBA uint8 format) which can be plotted by plt.imshow
        '''
        mask = self.get_weighted_mask()

        if plot_axis is None:
            f = plt.figure();
            plot_axis = f.add_subplot(111)

        curr_plot = plot_axis.imshow(mask, cmap=cmap, interpolation=interpolation, **kwargs)
        plot_axis.set_title(self.get_name())

        if self.interpolate_rate is None:
            interpolate_rate = 1
        else:
            interpolate_rate = self.interpolate_rate

        plot_axis.set_yticks(range(len(self.altPos))[::interpolate_rate])
        plot_axis.set_xticks(range(len(self.aziPos))[::interpolate_rate])
        plot_axis.set_yticklabels(self.altPos[::interpolate_rate])
        plot_axis.set_xticklabels(self.aziPos[::interpolate_rate])

        if is_colorbar:
            plot_axis.get_figure().colorbar(curr_plot)

        return plot_axis.get_figure()

    def plot_contour(self, plot_axis=None, peak_amplitude=None, level_num=10, **kwargs):
        '''
        return display image (RGBA uint8 format) which can be plotted by plt.imshow
        '''
        mask = self.get_weighted_mask()

        if plot_axis is None:
            f = plt.figure()
            plot_axis = f.add_subplot(111)

        if peak_amplitude is None:
            peak_amplitude = np.amax(self.get_weighted_mask())

        if self.sign == 'ON':
            colors = 'r'
        elif self.sign == 'OFF':
            colors = 'b'
        else:
            colors = 'k'

        contour_levels = list(np.arange(level_num) * (float(peak_amplitude) / (level_num)))

        if self.thr is not None:
            contour_levels = [l for l in contour_levels if l >= self.thr]
            if len(contour_levels) == 0:
                contour_levels = [self.thr]

        X, Y = np.meshgrid(np.arange(len(self.aziPos)),
                           np.arange(len(self.altPos)))

        if len(self.weights) > 0:
            plot_axis.contour(X, Y, self.get_weighted_mask(), levels=contour_levels, colors=colors, **kwargs)

        name = self.get_name()
        name = name.split(' ')
        name = ' '.join(name[1:])
        plot_axis.set_title(name)
        ylim = list(plot_axis.get_ylim())
        ylim.sort(reverse=True)
        plot_axis.set_ylim(ylim)
        plot_axis.set_aspect('equal')

        if self.interpolate_rate is not None:
            plot_axis.set_yticks(range(len(self.altPos))[::self.interpolate_rate])
            plot_axis.set_xticks(range(len(self.aziPos))[::self.interpolate_rate])
            plot_axis.set_yticklabels(self.altPos[::self.interpolate_rate])
            plot_axis.set_xticklabels(self.aziPos[::self.interpolate_rate])
        else:
            plot_axis.set_yticks(range(len(self.altPos)))
            plot_axis.set_xticks(range(len(self.aziPos)))
            plot_axis.set_yticklabels(self.altPos)
            plot_axis.set_xticklabels(self.aziPos)

        return plot_axis.get_figure()

    def threshold(self, thr):

        """
        threshold the current receptive field, return a new SpatialReceptiveField object after thresholding
        """

        if (self.thr is not None) and (thr < self.thr):
            raise ValueError, 'Can not cut a thresholded receptive field with a lower thresold!'
        cutRF = get_peak_weighted_roi(self.get_weighted_mask(), thr)
        if cutRF is None:
            print 'No ROI found. Threshold too high!'
            cutRF = ia.WeightedROI(np.zeros(self.dimension))

        return SpatialReceptiveField(cutRF.get_weighted_mask(), self.altPos, self.aziPos, sign=self.sign,
                                     temporalWindow=self.temporalWindow, pixelSizeUnit=self.pixelSizeUnit,
                                     dataType=self.dataType, thr=thr, filter_sigma=self.filter_sigma,
                                     interpolate_rate=self.interpolate_rate)

    def interpolate(self, ratio, method='cubic', fill_value=0.):

        ratio = float(ratio)

        if ratio <= 1:
            raise ValueError('interpolate_rate should be an integer larger than 1!')

        # altInterpolation = ip.interp1d(np.arange(len(self.altPos)),self.altPos)
        # aziInterpolation = ip.interp1d(np.arange(len(self.aziPos)),self.aziPos)
        altStep = np.mean(np.diff(self.altPos))
        aziStep = np.mean(np.diff(self.aziPos))
        newAltPos = np.arange(self.altPos[0], self.altPos[-1], altStep / ratio)
        newAziPos = np.arange(self.aziPos[0], self.aziPos[-1], aziStep / ratio)
        mask = self.get_weighted_mask()
        mask_ip = ip.interp2d(self.aziPos, self.altPos, mask, kind=method, fill_value=fill_value)
        newMask = mask_ip(newAziPos, newAltPos)

        return SpatialReceptiveField(newMask, newAltPos, newAziPos, sign=self.sign, temporalWindow=self.temporalWindow,
                                     pixelSizeUnit=self.pixelSizeUnit, dataType=self.dataType, thr=self.thr,
                                     filter_sigma=self.filter_sigma, interpolate_rate=ratio)

    def gaussian_filter(self, sigma):
        """
        return a new SpatialReceptiveField object, with mask filtered by a gaussian filter with width sigma pixels
        """

        mask = self.get_weighted_mask()
        mask_f = ni.gaussian_filter(mask, sigma=sigma)

        return SpatialReceptiveField(mask_f, self.altPos, self.aziPos, sign=self.sign,
                                     temporalWindow=self.temporalWindow, pixelSizeUnit=self.pixelSizeUnit,
                                     dataType=self.dataType, thr=self.thr, filter_sigma=sigma,
                                     interpolate_rate=self.interpolate_rate)

    def get_weighted_rf_center(self):
        """
        return weighted center of the receptive field in the coordinate system defined by self.altPos and self.aziPos
        """
        return self.get_weighted_center_in_coordinate(self.altPos, self.aziPos)

    def get_binary_rf_area(self):
        """
        return the thresholded binary receptive field area in the coordinate system defined by self.altPos and
        self.aziPos
        """

        if self.thr is None:
            raise LookupError('To get the area, the receptive field should be thresholded!!')

        alt_step = abs(np.mean(np.diff(self.altPos).astype(np.float)))
        azi_step = abs(np.mean(np.diff(self.aziPos).astype(np.float)))

        return len(self.weights) * alt_step * azi_step


class SpatialTemporalReceptiveField(object):
    """
    class of spatial temporal receptive field represented by traces for each specified retinotopic location
    """

    def __init__(self,
                 locations,
                 signs,
                 traces,
                 time,
                 trigger_ts=None,
                 name=None,
                 locationUnit='degree',
                 trace_data_type='dF_over_F',
                 verbose=False):
        """
        locations : list, tuple or 2-d array of retinotopic locations mapped
            each element has two float numbers: [altitude, azimuth]

        signs: list, tuple or 1d array of signs for each location

        traces: list of traces for each location
            list of 2-d array, each row: a single trace, each column: a single time point

        time: time axis for trace

        trigger_ts: list of lists or tuples or 1-d arrays. The outside list should have same length as locations,
            signs and traces. Each element of the outside list correspond to one probe, and the inside lists save
            the global trigger timestamps of each trace for that particular probe. This is used for filtering
            the STRF for global epochs.
        """

        if verbose:
            print('\ngenerating spatial temporal receptive field ...')

        locations = np.array([np.array(l, dtype=np.float32) for l in locations])
        signs = np.array(signs, dtype=np.float32)
        self.time = np.array(time, dtype=np.float32)
        traces = [np.array([np.array(t, dtype=np.float32) for t in trace]) for trace in traces]

        for trace_i, trace in enumerate(traces):
            if trace.shape[1] != len(time):
                raise ValueError('the shape of {:d}th trace: {} is not consistent with length of time axis: {:d}.'
                                 .format(trace_i, trace.shape, len(self.time)))

        if not (len(locations) == len(signs) == len(traces)):
            raise ValueError('length of "locations", "signs", "traces" should be the same!')

        if verbose:
            print('number of probes: {:d}'.format(len(locations)))

        if trigger_ts is None:
            trigger_ts = [[]] * len(traces)

            # trigger_ts = []
            # for trace in traces:
            #     curr_trigger_ts = np.zeros(trace.shape[0], dtype=np.float32)
            #     curr_trigger_ts[:] = np.nan
            #     trigger_ts.append(curr_trigger_ts)
        else:
            trigger_ts = [np.array(ts, dtype=np.float32) for ts in trigger_ts]

        if len(trigger_ts) != len(locations):
            raise ValueError('length of trigger_ts: {:d} is not consistent with number of probes: {:d}.'
                             .format(len(trigger_ts), len(locations)))

        values = [(location[0], location[1], signs[i], traces[i], trigger_ts[i]) for i, location in
                  enumerate(locations)]
        if len(values) == 0:
            raise ValueError, 'Can not find input traces!'
        self.data = DataFrame(values, columns=['altitude', 'azimuth', 'sign', 'traces', 'trigger_ts'])

        self.name = str(name)
        self.locationUnit = str(locationUnit)
        self.trace_data_type = str(trace_data_type)

        # self.merge_duplication()
        self.sort_probes()
        # print(self.data)
        # print(self.get_data_types())
        # print(self.get_probes())

    def merge_duplication(self):
        probe_grps = self.data.groupby(by=['altitude', 'azimuth', 'sign'])

        values = []
        for probe_n, probe_grp in probe_grps:
            # print('\n')
            # print(probe_grp.iloc[0]['altitude'])
            # print(probe_grp.iloc[0]['azimuth'])
            # print(probe_grp.iloc[0]['sign'])
            # print(np.concatenate(list(probe_grp['traces']), axis=0))
            # print(np.concatenate(list(probe_grp['trigger_ts']), axis=0))
            values.append((probe_grp.iloc[0]['altitude'],
                           probe_grp.iloc[0]['azimuth'],
                           probe_grp.iloc[0]['sign'],
                           np.concatenate(list(probe_grp['traces'])),
                           np.concatenate(list(probe_grp['trigger_ts']))))

        self.data = DataFrame(values, columns=['altitude', 'azimuth', 'sign', 'traces', 'trigger_ts'])

    def sort_probes(self):
        self.data.sort_values(by=['altitude', 'azimuth', 'sign'], inplace=True, ascending=True)

    def get_data_types(self):
        return self.data.dtypes

    def get_probes(self):
        return list(np.array([self.data['altitude'], self.data['azimuth'], self.data['sign']]).transpose())

    def set_trigger_ts(self, trigger_ts=None):
        """

        :param trigger_ts: list of trigger_ts for each probe or None
        :return:
        """

        if trigger_ts is None:
            trigger_ts = [[]] * self.data.shape[0]

        self.data['trigger_ts'] = trigger_ts

    def add_traces(self, locations, signs, traces, trigger_ts=None, verbose=False):

        """
        add traces to existing receptive field
        """

        if verbose:
            print('adding traces to existing STRF ...')

        locations = np.array([np.array(l, dtype=np.float32) for l in locations])
        signs = np.array(signs, dtype=np.float32)
        traces = [np.array([np.array(t, dtype=np.float32) for t in trace]) for trace in traces]

        for trace_i, trace in enumerate(traces):
            if trace.shape[1] != len(self.time):
                raise ValueError('the shape of {:d}th trace: {} is not consistent with length of time axis: {:d}.'
                                 .format(trace_i, trace.shape, len(self.time)))

        if not (len(locations) == len(signs) == len(traces)):
            raise ValueError('length of "locations", "signs", "traces" should be the same!')

        if verbose:
            print('number of probes to add: {:d}'.format(len(locations)))

        if trigger_ts is None:
            trigger_ts = []
            for trace in traces:
                curr_trigger_ts = np.zeros(trace.shape[0], dtype=np.float32)
                curr_trigger_ts[:] = np.nan
                trigger_ts.append(curr_trigger_ts)
        else:
            trigger_ts = [np.array(ts, dtype=np.float32) for ts in trigger_ts]

        if len(trigger_ts) != len(locations):
            raise ValueError('length of trigger_ts: {:d} is not consistent with number of probes: {:d}.'
                             .format(len(trigger_ts), len(locations)))

        values = [(location[0], location[1], signs[i], traces[i], trigger_ts[i]) for i, location in
                  enumerate(locations)]
        if not values: raise ValueError, 'Can not find input traces!'

        df_to_add = DataFrame(values, columns=['altitude', 'azimuth', 'sign', 'traces', 'trigger_ts'])

        self.data = self.data.append(df_to_add, ignore_index=True)

        self.merge_duplication()

    def to_h5_group(self, h5Group):

        h5Group.attrs['time'] = self.time
        h5Group.attrs['time_unit'] = 'second'
        h5Group.attrs['retinotopic_location_unit'] = self.locationUnit
        h5Group.attrs['trace_data_type'] = self.trace_data_type
        h5Group.attrs['trace_shape'] = '(trial, time_point)'
        if self.name is not None:
            h5Group.attrs['name'] = self.name
        else:
            h5Group.attrs['name'] = ''

        for probe_i, probe in self.data.iterrows():
            locationName = 'trace{:04d}'.format(probe_i)
            trace = h5Group.create_dataset(locationName, data=probe['traces'], dtype='f')
            trace.attrs['altitude'] = probe['altitude']
            trace.attrs['azimuth'] = probe['azimuth']
            trace.attrs['sign'] = probe['sign']
            if probe['trigger_ts']:
                trace.attrs['trigger_ts_sec'] = probe['trigger_ts']

    @staticmethod
    def from_h5_group(h5Group):
        """
        load SpatialTemporalReceptiveField object from a hdf5 data group
        """

        time = h5Group.attrs['time']
        # try:
        #     name = h5Group.parent.name[1:] + '.' + h5Group.parent.attrs['name']
        # except KeyError:
        #     name = None
        try:
            name = h5Group.attrs['name']
        except KeyError:
            name = None
        locationUnit = h5Group.attrs['retinotopic_location_unit']
        trace_data_type = h5Group.attrs['trace_data_type']
        locations = []
        signs = []
        traces = []
        trigger_ts = []
        for key, traceItem in h5Group.iteritems():
            locations.append(np.array([traceItem.attrs['altitude'], traceItem.attrs['azimuth']]))
            signs.append(traceItem.attrs['sign'])
            if 'trigger_ts_sec' in traceItem.attrs:
                trigger_ts.append(traceItem.attrs['trigger_ts_sec'])
            else:
                trigger_ts.append([])
            traces.append(traceItem.value)

        # no available or corrupted trigger timestamp info, set it to None
        # if len(trigger_ts) < len(signs):
        #     trigger_ts = None

        return SpatialTemporalReceptiveField(locations=locations, signs=signs, traces=traces, time=time,
                                             trigger_ts=trigger_ts, name=name, locationUnit=locationUnit,
                                             trace_data_type=trace_data_type)

    def plot_traces(self, f=None, figSize=(10, 10), yRange=(0, 20), altRange=None, aziRange=None, **kwargs):

        indexLists, axisLists = self._get_axis_layout(f, figSize, yRange, altRange, aziRange, **kwargs)

        for i, axisList in enumerate(axisLists):
            for j, axis in enumerate(axisList):
                indexList = indexLists[i][j]
                axis.set_axis_off()
                axis.set_xticks([]);
                axis.set_yticks([])
                for pos in ['top', 'bottom', 'left', 'right']:
                    axis.spines[pos].set_linewidth(0.5)
                    axis.spines[pos].set_color('#888888')
                axis.axvline(x=0, ls='--', color='#888888', lw=0.5)
                axis.plot([self.time[0], self.time[-1]], [0., 0.], color='#888888', lw=0.5)

                for index in indexList:
                    traces = self.data.iloc[index]['traces']
                    traces = [t for t in traces if not math.isnan(t[0])]
                    meanTrace = np.mean(np.array(traces, dtype=np.float32), axis=0)

                    if self.data.iloc[index]['sign'] == 1:
                        color = '#ff0000'
                    elif self.data.iloc[index]['sign'] == -1:
                        color = '#0000ff'
                    else:
                        color = '#000000'

                    if len(traces) > 1:
                        stdTrace = np.std(np.array(traces, dtype=np.float32), axis=0)
                        semTrace = stdTrace / np.sqrt(float(len(traces)))
                        axis.fill_between(self.time, meanTrace - semTrace, meanTrace + semTrace, facecolor=color,
                                          linewidth=0, alpha=0.5)
                    axis.plot(self.time, meanTrace, '-', color=color, lw=1)

        return axisLists[0][0].figure

    def _get_axis_layout(self, f=None, figSize=(10, 10), yRange=(0, 20), altRange=None, aziRange=None, **kwargs):

        locations = np.array(self.get_probes())

        altPositions = np.sort(np.unique(locations[:, 0]))[::-1]
        if altRange is not None:
            altPositions = np.array([x for x in altPositions if (x >= altRange[0] and x <= altRange[1])])

        aziPositions = np.sort(np.unique(locations[:, 1]))
        if aziRange is not None:
            aziPositions = np.array([x for x in aziPositions if (x >= aziRange[0] and x <= aziRange[1])])

        indexLists = [[[] for aziPosition in aziPositions] for altPosition in altPositions]

        if f is None:
            f = plt.figure(figsize=figSize)

        f.suptitle('cell:{}; xrange:[{:6.3f}, {:6.3f}]; yrange: [{:.3f}, {:.3f}]'.
                   format(self.name, self.time[0], self.time[-1], yRange[0], yRange[1]))

        axisLists = pt.tile_axis(f, len(altPositions), len(aziPositions), **kwargs)

        for i, altPosition in enumerate(altPositions):
            for j, aziPosition in enumerate(aziPositions):
                axisLists[i][j].text(0, yRange[1], str(int(altPosition)) + ';' + str(int(aziPosition)),
                                     ha='left', va='top', fontsize=10)
                axisLists[i][j].set_xlim([self.time[0], self.time[-1]])
                axisLists[i][j].set_ylim(yRange)

                for k, location in enumerate(locations):
                    if location[0] == altPosition and location[1] == aziPosition:
                        indexLists[i][j].append(k)

        return indexLists, axisLists

    def get_amplitude_map(self, timeWindow=(0, 0.5)):
        """
        return 2d receptive field map and altitude and azimuth coordinates
        each pixel in the map represent mean amplitute of traces within the window defined by timeWindow, and the
        coordinate of each pixel is defined by np.meshgrid(allAziPos, allAltPos)
        """

        windowIndex = np.logical_and(self.time >= timeWindow[0], self.time <= timeWindow[1])

        indON, indOFF, allAltPos, allAziPos = self._sort_index()

        ampON = np.zeros(indON.shape)
        ampON[:] = np.nan
        ampOFF = ampON.copy()

        for i in np.ndindex(indON.shape):
            traceIndON = indON[i]
            traceIndOFF = indOFF[i]
            if traceIndON is not None:
                ampON[i] = np.mean(np.mean(self.data.iloc[traceIndON]['traces'], axis=0)[windowIndex])
            if traceIndOFF is not None:
                ampOFF[i] = np.mean(np.mean(self.data.iloc[traceIndOFF]['traces'], axis=0)[windowIndex])

        return ampON, ampOFF, allAltPos, allAziPos

    def get_amplitude_receptive_field(self, timeWindow=(0, 0.5)):
        """
        very similar to get_amplitude_map(), only difference is that, it is returning spatial temporal receptive fields
        instead of 2d matrix
        each pixel in the map represent mean amplitute of traces within the window defined by timeWindow, and the
        coordinate of each pixel is defined by np.meshgrid(allAziPos, allAltPos)
        """

        ampON, ampOFF, allAltPos, allAziPos = self.get_amplitude_map(timeWindow)

        ampRFON = SpatialReceptiveField(mask=ampON, altPos=allAltPos, aziPos=allAziPos, sign=1,
                                        temporalWindow=timeWindow, pixelSizeUnit=self.locationUnit,
                                        dataType='amplitude')
        ampRFOFF = SpatialReceptiveField(mask=ampOFF, altPos=allAltPos, aziPos=allAziPos, sign=-1,
                                         temporalWindow=timeWindow, pixelSizeUnit=self.locationUnit,
                                         dataType='amplitude')

        return ampRFON, ampRFOFF

    def get_delta_amplitude_map(self, timeWindow=(0, 0.5)):
        """
        return 2d receptive field map and altitude and azimuth coordinates
        each pixel in the map represent mean delta amplitute (raw amplitude minus the mean amplitude before trigger
        onset) of traces within the window defined by timeWindow, and the
        coordinate of each pixel is defined by np.meshgrid(allAziPos, allAltPos)
        """

        windowIndex = np.logical_and(self.time >= timeWindow[0], self.time <= timeWindow[1])

        baseline_index = self.time < 0

        indON, indOFF, allAltPos, allAziPos = self._sort_index()

        ampON = np.zeros(indON.shape)
        ampON[:] = np.nan
        ampOFF = ampON.copy()

        for i in np.ndindex(indON.shape):
            traceIndON = indON[i]
            traceIndOFF = indOFF[i]
            if traceIndON is not None:
                curr_trace_ON = np.mean(self.data.iloc[traceIndON]['traces'], axis=0)
                curr_baseline_ON = np.mean(curr_trace_ON[baseline_index])
                curr_delta_trace_ON = curr_trace_ON - curr_baseline_ON
                ampON[i] = np.mean(curr_delta_trace_ON[windowIndex])
            if traceIndOFF is not None:
                curr_trace_OFF = np.mean(self.data.iloc[traceIndOFF]['traces'], axis=0)
                curr_baseline_OFF = np.mean(curr_trace_OFF[baseline_index])
                curr_delta_trace_OFF = curr_trace_OFF - curr_baseline_OFF
                ampOFF[i] = np.mean(curr_delta_trace_OFF[windowIndex])

        return ampON, ampOFF, allAltPos, allAziPos

    def get_delta_amplitude_receptive_field(self, timeWindow=(0, 0.5)):
        """
        very similar to get_delta_amplitude_map(), only difference is that, it is returning SpatialReceptiveFields
        instead of 2d matrix
        each pixel in the map represent mean delta amplitute of traces within the window defined by timeWindow, and the
        coordinate of each pixel is defined by np.meshgrid(allAziPos, allAltPos)
        """

        ampON, ampOFF, allAltPos, allAziPos = self.get_delta_amplitude_map(timeWindow)

        ampRFON = SpatialReceptiveField(mask=ampON, altPos=allAltPos, aziPos=allAziPos, sign=1,
                                        temporalWindow=timeWindow, pixelSizeUnit=self.locationUnit,
                                        dataType='delta_amplitude')
        ampRFOFF = SpatialReceptiveField(mask=ampOFF, altPos=allAltPos, aziPos=allAziPos, sign=-1,
                                         temporalWindow=timeWindow, pixelSizeUnit=self.locationUnit,
                                         dataType='delta_amplitude')
        return ampRFON, ampRFOFF

    def get_zscore_map(self, timeWindow=(0, 0.5)):
        """
        return 2d receptive field and altitude and azimuth coordinates
        each pixel in the map represent Z score of mean amplitute of traces within the window defined by timeWindow
        """

        ampON, ampOFF, allAltPos, allAziPos = self.get_amplitude_map(timeWindow)

        return ia.zscore(ampON), ia.zscore(ampOFF), allAltPos, allAziPos

    def get_zscore_receptive_field(self, timeWindow=(0, 0.5)):
        """
        outdated


        very similar to get_zscore_map(), only difference is that, it is returning spatial temporal receptive fields
        instead of 2d matrix
        each pixel in the map represent mean amplitute of traces within the window defined by timeWindow, and the
        coordinate of each pixel is defined by np.meshgrid(allAziPos, allAltPos)
        """

        ampON, ampOFF, allAltPos, allAziPos = self.get_amplitude_map(timeWindow)

        zscoreRFON = SpatialReceptiveField(mask=ia.zscore(ampON), altPos=allAltPos, aziPos=allAziPos, sign='ON',
                                           temporalWindow=timeWindow, pixelSizeUnit=self.locationUnit,
                                           dataType='zscore')
        zscoreRFOFF = SpatialReceptiveField(mask=ia.zscore(ampOFF), altPos=allAltPos, aziPos=allAziPos, sign='OFF',
                                            temporalWindow=timeWindow, pixelSizeUnit=self.locationUnit,
                                            dataType='zscore')

        return zscoreRFON, zscoreRFOFF

    def get_zscore_rois(self, timeWindow=(0, 0.5), zscoreThr=2):
        """
        outdated


        return ON, OFF and combined receptive field rois in the format of WeightedROI object

        Amplitude for each pixel was calculated as mean dF over F signal trace within the timeWindow
        mask of ON and OFF receptive field was generated by cutting zscore map by zscoreThr
        Tombined mask is the sum of ON and OFF weighted mask

        The sampled altitude positions and azimuth positions are also returned. The receptive field space coordinates
        were defined as np.meshgrid(allAziPos, allAltPos)
        """
        zscoreON, zscoreOFF, allAltPos, allAziPos = self.get_zscore_map(timeWindow)
        zscoreROION = get_peak_weighted_roi(zscoreON, zscoreThr)
        zscoreROIOFF = get_peak_weighted_roi(zscoreOFF, zscoreThr)
        if zscoreROION is not None and zscoreROIOFF is not None:
            zscoreROIALL = WeightedROI(zscoreROION.get_weighted_mask() + zscoreROIOFF.get_weighted_mask())
        elif zscoreROION is None and zscoreROIOFF is not None:
            print 'No zscore receptive field found for ON channel. Threshold too high.'
            zscoreROIALL = zscoreROIOFF
        elif zscoreROION is not None and zscoreROIOFF is None:
            print 'No zscore receptive field found for OFF channel. Threshold too high.'
            zscoreROIALL = zscoreROION
        else:
            zscoreROIALL = None

        return zscoreROION, zscoreROIOFF, zscoreROIALL, allAltPos, allAziPos

    def get_zscore_thresholded_receptive_fields(self, timeWindow=(0, 0.3), thr_ratio=0.3, filter_sigma=None,
                                                interpolate_rate=None, absolute_thr=None):
        """
        return ON, OFF and combined receptive fields in the format of SpatialReceptiveField

        both ON and OFF RF mask will be filtered, interpolated as defined by the filter_sigma (in pixels) and
        interpolate_ratio respectively.

        Then the max value will be defined as maximum of ON RF peak and OFF RF peak. this max value times the thr_ratio
        (default 0.3, meaning 30% of the maximum) will be applied as a uniform cutting threshold to get thresholded RF
        mask for both ON and OFF RF. If calculated threshold is lower than absolute_thr, then absolute_thr will be used

        Combined receptive is the sum of ON and OFF thresholded zscore receptive field

        """

        zscoreON, zscoreOFF, allAltPos, allAziPos = self.get_zscore_map(timeWindow)

        zscoreRFON = SpatialReceptiveField(zscoreON, allAltPos, allAziPos, sign='ON', temporalWindow=timeWindow,
                                           pixelSizeUnit=self.locationUnit, dataType='zscore')

        zscoreRFOFF = SpatialReceptiveField(zscoreOFF, allAltPos, allAziPos, sign='OFF', temporalWindow=timeWindow,
                                            pixelSizeUnit=self.locationUnit, dataType='zscore')

        if filter_sigma is not None:
            zscoreRFON = zscoreRFON.gaussian_filter(filter_sigma)
            zscoreRFOFF = zscoreRFOFF.gaussian_filter(filter_sigma)

        if interpolate_rate is not None:
            zscoreRFON = zscoreRFON.interpolate(interpolate_rate)
            zscoreRFOFF = zscoreRFOFF.interpolate(interpolate_rate)

        max_value = max([np.amax(zscoreRFON.get_weighted_mask()), np.amax(zscoreRFOFF.get_weighted_mask())])

        thr = max_value * thr_ratio

        if absolute_thr is not None:
            thr = max([thr, absolute_thr])

        zscoreRFON = zscoreRFON.threshold(thr)
        zscoreRFOFF = zscoreRFOFF.threshold(thr)

        zscoreRFALL = SpatialReceptiveField(zscoreRFON.get_weighted_mask() + zscoreRFOFF.get_weighted_mask(),
                                            zscoreRFON.altPos, zscoreRFON.aziPos, sign='ON_OFF',
                                            temporalWindow=timeWindow, pixelSizeUnit=self.locationUnit,
                                            dataType='zscore', thr=thr, filter_sigma=filter_sigma,
                                            interpolate_rate=interpolate_rate)

        return zscoreRFON, zscoreRFOFF, zscoreRFALL

    def get_zscore_roi_centers(self, timeWindow=(0, 0.5), zscoreThr=2):
        """
        outdated


        return retinotopic location of ON subfield, OFF subfield and combined receptive field

        zscore ROIs was generated by the method get_zscore_rois()
        """
        zscoreROION, zscoreROIOFF, zscoreROIALL, allAltPos, allAziPos = self.get_zscore_rois(timeWindow, zscoreThr)
        if zscoreROION is not None:
            centerON = zscoreROION.get_weighted_center_in_coordinate(allAltPos, allAziPos)
        else:
            centerON = None

        if zscoreROIOFF is not None:
            centerOFF = zscoreROIOFF.get_weighted_center_in_coordinate(allAltPos, allAziPos)
        else:
            centerOFF = None

        if zscoreROIALL is not None:
            centerALL = zscoreROIALL.get_weighted_center_in_coordinate(allAltPos, allAziPos)
        else:
            centerALL = None
        return centerON, centerOFF, centerALL

    def _sort_index(self):
        """
        return ON and OFF index matrices for all combination of sampled retinotopic locations along with retinotopic
        coordinates, the retinotopic visual space was defined by np.meshgrid(allAziPos, allAltPos)
        """

        allAltPos = np.array(sorted(list(set(list(self.data['altitude'])))))[::-1]
        allAziPos = np.array(sorted(list(set(list(self.data['azimuth'])))))

        indON = [[None for azi in allAziPos] for alt in allAltPos]
        indOFF = [[None for azi in allAziPos] for alt in allAltPos]

        for i, traceItem in self.data.iterrows():
            alt = traceItem['altitude']
            azi = traceItem['azimuth']
            sign = traceItem['sign']
            for j, altPos in enumerate(allAltPos):
                for k, aziPos in enumerate(allAziPos):
                    if alt == altPos and azi == aziPos:

                        if sign == 1:
                            if indON[j][k] is not None:
                                raise LookupError, 'Duplication of trace items found at location: ' + str(
                                    [alt, azi]) + '; sign: 1!'
                            else:
                                indON[j][k] = i

                        if sign == -1:
                            if indOFF[j][k] is not None:
                                raise LookupError, 'Duplication of trace items found at location: ' + str(
                                    [alt, azi]) + '; sign:-1!'
                            else:
                                indOFF[j][k] = i

        indON = np.array([np.array(x) for x in indON]);
        indOFF = np.array([np.array(x) for x in indOFF])

        return indON, indOFF, allAltPos, allAziPos

    def shrink(self, altRange=None, aziRange=None, is_reset_index=True):
        """
        shrink the current spatial temporal receptive field into the defined altitude and/or azimuth range
        """

        if altRange is None and aziRange is None:
            raise LookupError, 'At least one of altRange and aziRange should be defined!'

        if altRange is not None:
            indAlt = np.logical_and(self.data['altitude'] >= altRange[0],
                                    self.data['altitude'] <= altRange[1])
        else:
            indAlt = np.ones(len(self.data), dtype=np.bool)

        if aziRange is not None:
            indAzi = np.logical_and(self.data['azimuth'] >= aziRange[0],
                                    self.data['azimuth'] <= aziRange[1])
        else:
            indAzi = np.ones(len(self.data), dtype=np.bool)

        ind = np.logical_and(indAlt, indAzi)

        if np.sum(ind) == 0:
            raise ValueError('No probes were sampled within the given altitude and azimuth range.')

        if is_reset_index:
            self.data = self.data[ind].reset_index(drop=True)
        else:
            self.data = self.data[ind]

    def get_local_dff_strf(self, is_collaps_before_normalize=True, add_to_trace=0.):
        """

        :param is_collaps_before_normalize: if True, for each location, the traces across multiple trials will be
                                            averaged before calculating df/f
        :return:
        """

        bl_inds = self.time <= 0
        # print(bl_inds)

        dff_traces = []
        for probe_i, probe in self.data.iterrows():
            curr_traces = np.array(probe['traces']) + add_to_trace

            if is_collaps_before_normalize:
                curr_traces = np.mean(curr_traces, axis=0, keepdims=True)

            curr_bl = np.mean(curr_traces[:, bl_inds], axis=1, keepdims=True)
            curr_dff = (curr_traces - curr_bl) / curr_bl

            dff_traces.append(list(curr_dff))

        locations = zip(list(self.data['altitude']), list(self.data['azimuth']))

        if is_collaps_before_normalize:

            strf_dff = SpatialTemporalReceptiveField(locations=locations,
                                                     signs=list(self.data['sign']),
                                                     traces=dff_traces,
                                                     trigger_ts=None,
                                                     time=self.time,
                                                     name=self.name,
                                                     locationUnit=self.locationUnit,
                                                     trace_data_type=self.trace_data_type + '_local_dff')

        else:

            strf_dff = SpatialTemporalReceptiveField(locations=locations,
                                                     signs=list(self.data['sign']),
                                                     traces=dff_traces,
                                                     trigger_ts=list(self.data['trigger_ts']),
                                                     time=self.time,
                                                     name=self.name,
                                                     locationUnit=self.locationUnit,
                                                     trace_data_type=self.trace_data_type + '_local_dff')
        return strf_dff

    def get_data_range(self):

        v_min = None
        v_max = None

        for probe_i, probe in self.data.iterrows():

            curr_trace = np.array(probe['traces'])

            if curr_trace.shape[0] > 1:
                curr_std_trace = np.std(np.array(curr_trace, dtype=np.float32), axis=0, keepdims=True)
                curr_sem_trace = curr_std_trace / np.sqrt(float(len(curr_trace)))
                curr_trace_high = curr_trace + curr_sem_trace
                curr_trace_low = curr_trace - curr_sem_trace
            else:
                curr_trace_low = curr_trace
                curr_trace_high = curr_trace

            if v_min is None:
                v_min = np.amin(curr_trace_low)
            else:
                v_min = min([v_min, np.amin(curr_trace_low)])

            if v_max is None:
                v_max = np.max(curr_trace_high)
            else:
                v_max = max([v_max, np.amax(curr_trace_high)])

        return v_min, v_max

    def temporal_subset(self):
        # todo: finish this. But need to do TimeIntervals containing TimeIntervals first in TimingAnalysis
        pass


class DriftingGratingResponseMatrix(DataFrame):
    """
    class for response matrix to drifting grating circle
    contains event triggered traces for all traces of one roi

    subclassed from pandas.DataFrame with more attribute:
    sta_ts: 1d array, local time stamps for event triggered traces
    trace_type: str, type of traces

    columns:
    alt  - altitute of circle center
    azi  - azimuth of circle center
    sf   - spatial frequency, cpd
    tf   - temporal frequency, Hz
    dire - drifting direction, deg, 0 is to right, increase counter-clockwise
    con  - contrast, [0, 1]
    rad  - radius, deg
    onset_ts - 1d array, global onset time stamps for each trial
    matrix - 2d array, trial x time point
    """

    def __init__(self, sta_ts, trace_type='', *args, **kwargs):

        super(DriftingGratingResponseMatrix, self).__init__(*args, **kwargs)

        self.sta_ts = sta_ts
        self.trace_type = trace_type

        self.check_integrity()

    def get_condition_name(self, row_index):

        condition_name = get_dgc_condition_name(alt=self.loc[row_index, 'alt'],
                                                azi=self.loc[row_index, 'azi'],
                                                sf=self.loc[row_index, 'sf'],
                                                tf=self.loc[row_index, 'tf'],
                                                dire=self.loc[row_index, 'dire'],
                                                con=self.loc[row_index, 'con'],
                                                rad=self.loc[row_index, 'rad'])
        return condition_name

    def check_integrity(self):

        if len(self.sta_ts.shape) != 1:
            raise ValueError('self.sta_ts should be 1d array.')

        if self.duplicated(subset=['alt', 'azi', 'sf', 'tf', 'dire', 'con', 'rad']).any():
            raise ValueError('there is duplicated conditions.')

        sta_ts_len = self.sta_ts.shape[0]

        for row_i, row in self.iterrows():

            if len(row['onset_ts']) == 0:
                pass
                # print('condition: {}. No onset timestamps available.'.format(self.get_condition_name(row_i)))
            else:
                if (len(row['onset_ts'].shape) != 1):
                    raise ValueError(
                        'condition: {}, onset_ts should be 1-d array.'.format(self.get_condition_name(row_i)))

                if row['matrix'].shape[0] != row['onset_ts'].shape[0]:
                    raise ValueError('condition: {}, mismatched trial number ({}) and onset number ({}).'
                                     .format(self.get_condition_name(row_i), row['matrix'].shape[0],
                                             row['onset_ts'].shape[0]))

            if len(row['matrix'].shape) != 2:
                raise ValueError('condition: {}, onset_ts should be 2-d array.'.format(self.get_condition_name(row_i)))

            if row['matrix'].shape[1] != sta_ts_len:
                raise ValueError('condition: {}, mismatched trace length ({}) and sta ts length ({}).'
                                 .format(self.get_condition_name(row_i), row['matrix'].shape[1], sta_ts_len))

    def get_df_response_matrix(self, baseline_win=(-0.5, 0.)):
        """
        return df response matrix

        :param baseline_win:
        :return:
        """

        baseline_ind = np.logical_and(self.sta_ts > baseline_win[0], self.sta_ts <= baseline_win[1])

        dgcrm_df = self.copy()

        for row_i, row in self.iterrows():
            curr_matrix = row['matrix'].astype(np.float64)
            curr_baseline = np.mean(curr_matrix[:, baseline_ind], axis=1, keepdims=True)
            dgcrm_df.loc[row_i, 'matrix'] = curr_matrix - curr_baseline

        return DriftingGratingResponseMatrix(sta_ts=self.sta_ts, trace_type='{}_df'.format(self.trace_type),
                                             data=dgcrm_df)

    def get_zscore_response_matrix(self, baseline_win=(-0.5, 0.)):
        """

        return zscore response matrix, zscore is calculated as (trace - baseline_mean) / baseline_std

        :param baseline_win:
        :return:
        """

        baseline_ind = np.logical_and(self.sta_ts > baseline_win[0], self.sta_ts <= baseline_win[1])

        dgcrm_zscore = self.copy()

        for row_i, row in self.iterrows():
            curr_matrix = row['matrix'].astype(np.float64)
            curr_baseline_mean = np.mean(curr_matrix[:, baseline_ind], axis=1, keepdims=True)
            curr_baseline_std = np.std(curr_matrix[:, baseline_ind].flat)
            dgcrm_zscore.loc[row_i, 'matrix'] = (curr_matrix - curr_baseline_mean) / curr_baseline_std

        return DriftingGratingResponseMatrix(sta_ts=self.sta_ts, trace_type='{}_zscore'.format(self.trace_type),
                                             data=dgcrm_zscore)

    def get_dff_response_matrix(self, baseline_win=(-0.5, 0.), bias=0., warning_level=1.):
        """

        return df over f response matrix

        :param baseline_win:
        :param bias: float, a number added to all matrices before calculating df over f
        :param warning_level: float, if the absolute value of the baseline of a given condition and a given trial is
                              smaller than this value, print a waring
        :return:
        """

        baseline_ind = np.logical_and(self.sta_ts > baseline_win[0], self.sta_ts <= baseline_win[1])

        dgcrm_dff = self.copy()

        for row_i, row in self.iterrows():
            curr_matrix = row['matrix'].astype(np.float64) + bias
            dff_matrix = np.empty(curr_matrix.shape, dtype=np.float32)
            for trial_i in range(curr_matrix.shape[0]):
                curr_trial = curr_matrix[trial_i, :]
                curr_baseline = np.mean(curr_trial[baseline_ind])

                # print(curr_baseline)
                if curr_baseline <= warning_level:
                    msg = '\ncondition:{}, trial:{}, baseline too low: {}'.format(self.get_condition_name(row_i),
                                                                                  trial_i,
                                                                                  curr_baseline)
                    warnings.warn(msg, RuntimeWarning)

                curr_trial_dff = (curr_trial - curr_baseline) / curr_baseline
                dff_matrix[trial_i, :] = curr_trial_dff

            dgcrm_dff.loc[row_i, 'matrix'] = dff_matrix

        return DriftingGratingResponseMatrix(sta_ts=self.sta_ts, trace_type='{}_dff'.format(self.trace_type),
                                             data=dgcrm_dff)

    def get_condition_trial_responses(self, condi_i, response_win=(0., 1.)):
        """
        for a given condition specified by df index: condi_i, return responses for each trial
        :param condi_i: int
        :param response_win: list of two floats, time window to calculate responses
        :return: 1d array, response of each trial for the specified condition
        """

        response_ind = np.logical_and(self.sta_ts > response_win[0], self.sta_ts <= response_win[1])

        traces = self.loc[condi_i, 'matrix']

        responses = np.mean(traces[:, response_ind], axis=1)

        return responses.squeeze()

    def collapse_trials(self):

        """
        calculate mean response for each condition across all trials

        :return: DriftingGratingResponseMatrix object
        """

        dgcrm_collapsed = self.copy()

        for row_i, row in self.iterrows():
            curr_matrix = row['matrix']
            dgcrm_collapsed.loc[row_i, 'matrix'] = np.mean(curr_matrix, axis=0, keepdims=True)
            dgcrm_collapsed.loc[row_i, 'onset_ts'] = []

        return DriftingGratingResponseMatrix(sta_ts=self.sta_ts, trace_type='{}_collapsed'.format(self.trace_type),
                                             data=dgcrm_collapsed)

    def get_response_table(self, response_win=(0., 1.)):

        response_ind = np.logical_and(self.sta_ts > response_win[0], self.sta_ts <= response_win[1])

        dgcrt = self.loc[:, ['alt', 'azi', 'sf', 'tf', 'dire', 'con', 'rad']]
        dgcrt['resp_mean'] = np.nan
        dgcrt['resp_max'] = np.nan
        dgcrt['resp_min'] = np.nan
        dgcrt['resp_std'] = np.nan
        dgcrt['resp_stdev'] = np.nan

        for row_i, row in self.iterrows():

            responses = np.mean(row['matrix'][:, response_ind], axis=1)

            dgcrt.loc[row_i, 'resp_mean'] = np.mean(responses)
            dgcrt.loc[row_i, 'resp_max'] = np.max(responses)
            dgcrt.loc[row_i, 'resp_min'] = np.min(responses)

            if len(responses) > 1:
                dgcrt.loc[row_i, 'resp_std'] = np.std(responses)
                dgcrt.loc[row_i, 'resp_stdev'] = np.std(responses) / np.sqrt(len(responses))

        return DriftingGratingResponseTable(trace_type=self.trace_type, data=dgcrt)

    def get_df_response_table(self, baseline_win=(-0.5, 0.), response_win=(0., 1.)):
        """
        this is suppose to give the most robust measurement of df response table.

        for each condition:
        1. mean_baseline is calculated by averaging across all trials and all data points in the baseline_win
        2. mean_response is calculated by averaging across all trials and all data points in the response_win
        3. df for every condition is defined by (mean_response - mean_baseline) and response table is generated

        # separate operation
        4. for each trial of each condition, df is calculated by (mean_response - mean_baseline)
        5. one-way anova is performed from these trial responses
        6. peak positive condition and peak negative condition is selected from previously generated response table
        7. ttest is performed for these two conditions against blank trial responses


        :param baseline_win:
        :param response_win:
        :return df_response_table:
        :return p_anova:
        :return p_ttest_pos:
        :return p_ttest_neg:
        """

        baseline_ind = np.logical_and(self.sta_ts > baseline_win[0], self.sta_ts <= baseline_win[1])
        response_ind = np.logical_and(self.sta_ts > response_win[0], self.sta_ts <= response_win[1])

        dgcrt = self.loc[:, ['alt', 'azi', 'sf', 'tf', 'dire', 'con', 'rad']]
        dgcrt['resp_mean'] = np.nan
        dgcrt['resp_max'] = np.nan
        dgcrt['resp_min'] = np.nan
        dgcrt['resp_std'] = np.nan
        dgcrt['resp_stdev'] = np.nan

        trial_responses = []

        for row_i, row in self.iterrows():
            curr_matrix = row['matrix']
            baseline_mean = np.mean(curr_matrix[:, baseline_ind].astype(np.float64).flat)
            response_mean = np.mean(curr_matrix[:, response_ind].astype(np.float64).flat)
            dgcrt.loc[row_i, 'resp_mean'] = response_mean - baseline_mean

            baseline_trial = np.mean(curr_matrix[:, baseline_ind].astype(np.float64), axis=1)
            response_trial = np.mean(curr_matrix[:, response_ind].astype(np.float64), axis=1)
            curr_trial_responses = response_trial - baseline_trial
            trial_responses.append(curr_trial_responses)

            dgcrt.loc[row_i, 'resp_max'] = np.max(curr_trial_responses)
            dgcrt.loc[row_i, 'resp_min'] = np.min(curr_trial_responses)
            dgcrt.loc[row_i, 'resp_std'] = np.std(curr_trial_responses)
            dgcrt.loc[row_i, 'resp_stdev'] = np.std(curr_trial_responses) / np.sqrt(len(curr_trial_responses))

        _, p_anova = stats.f_oneway(*trial_responses)

        df_response_table = DriftingGratingResponseTable(data=dgcrt, trace_type='{}_df'.format(self.trace_type))
        responses_blank = trial_responses[df_response_table.blank_condi_ind]
        responses_peak_pos = trial_responses[df_response_table.peak_condi_ind_pos]
        responses_peak_neg = trial_responses[df_response_table.peak_condi_ind_neg]

        _, p_ttest_pos = stats.ttest_rel(responses_blank, responses_peak_pos)
        _, p_ttest_neg = stats.ttest_rel(responses_blank, responses_peak_neg)

        return df_response_table, p_anova, p_ttest_pos, p_ttest_neg

    def get_dff_response_table(self, baseline_win=(-0.5, 0.), response_win=(0., 1.), bias=0, warning_level=0.1):
        """
        this is suppose to give the most robust measurement of df/f response table.

        for each condition:
        1. mean_baseline is calculated by averaging across all trials and all data points in the baseline_win
        2. mean_response is calculated by averaging across all trials and all data points in the response_win
        3. df/f for each condition is defined by
            (mean_response - mean_baseline) / mean_baseline and response table is generated

        # separate operation
        4. for each trial of each condition, df is calculated by (mean_response - mean_baseline) / mean_baseline
        5. one-way anova is performed from these trial responses
        6. peak positive condition and peak negative condition is selected from previously generated response table
        7. ttest is performed for these two conditions against blank trial responses


        :param baseline_win:
        :param response_win:
        :param bias: float, a constant added to all matrices
        :param warning_level: float, warning level of low baseline
        :return dff_response_table:
        :return p_anova:
        :return p_ttest_pos:
        :return p_ttest_neg:
        """

        baseline_ind = np.logical_and(self.sta_ts > baseline_win[0], self.sta_ts <= baseline_win[1])
        response_ind = np.logical_and(self.sta_ts > response_win[0], self.sta_ts <= response_win[1])

        dgcrt = self.loc[:, ['alt', 'azi', 'sf', 'tf', 'dire', 'con', 'rad']]
        dgcrt['resp_mean'] = np.nan
        dgcrt['resp_max'] = np.nan
        dgcrt['resp_min'] = np.nan
        dgcrt['resp_std'] = np.nan
        dgcrt['resp_stdev'] = np.nan

        trial_responses = []

        for row_i, row in self.iterrows():
            curr_matrix = row['matrix'] + bias
            baseline_mean = np.mean(curr_matrix[:, baseline_ind].astype(np.float64).flat)
            response_mean = np.mean(curr_matrix[:, response_ind].astype(np.float64).flat)
            dgcrt.loc[row_i, 'resp_mean'] = (response_mean - baseline_mean) / baseline_mean

            if baseline_mean <= warning_level:
                msg = '\ncondition:{}, mean baseline too low: {}'.format(self.get_condition_name(row_i), baseline_mean)
                warnings.warn(msg, RuntimeWarning)


            baseline_trial = np.mean(curr_matrix[:, baseline_ind].astype(np.float64), axis=1)
            response_trial = np.mean(curr_matrix[:, response_ind].astype(np.float64), axis=1)
            curr_trial_responses = (response_trial - baseline_trial) / baseline_trial

            if np.min(baseline_trial) <= warning_level:
                msg = '\ncondition:{}, trial baseline too low: {}'.format(self.get_condition_name(row_i),
                                                                          np.min(baseline_trial))
                warnings.warn(msg, RuntimeWarning)


            trial_responses.append(curr_trial_responses)
            dgcrt.loc[row_i, 'resp_max'] = np.max(curr_trial_responses)
            dgcrt.loc[row_i, 'resp_min'] = np.min(curr_trial_responses)
            dgcrt.loc[row_i, 'resp_std'] = np.std(curr_trial_responses)
            dgcrt.loc[row_i, 'resp_stdev'] = np.std(curr_trial_responses) / np.sqrt(len(curr_trial_responses))

        _, p_anova = stats.f_oneway(*trial_responses)

        dff_response_table = DriftingGratingResponseTable(data=dgcrt, trace_type='{}_df'.format(self.trace_type))
        responses_blank = trial_responses[dff_response_table.blank_condi_ind]
        responses_peak_pos = trial_responses[dff_response_table.peak_condi_ind_pos]
        responses_peak_neg = trial_responses[dff_response_table.peak_condi_ind_neg]

        _, p_ttest_pos = stats.ttest_rel(responses_blank, responses_peak_pos)
        _, p_ttest_neg = stats.ttest_rel(responses_blank, responses_peak_neg)

        return dff_response_table, p_anova, p_ttest_pos, p_ttest_neg

    def get_zscore_response_table(self, baseline_win=(-0.5, 0.), response_win=(0., 1.)):
        """
        this is suppose to give the most robust measurement of zscore response table.

        for each condition:
        1. mean_baseline is calculated by averaging across all trials and all data points in the baseline_win
        2. mean_response is calculated by averaging across all trials and all data points in the response_win
        3. mean_standard_deviation is calculated as following:
            i.   the baseline of each trial is normalized with zero mean
            ii.  normalized baselines are concatenated to a 1d array
            iii. mean_standard_deviation is calculated from the concatenated baseline
        4. zscore for each condition is defined by
            (mean_response - mean_baseline) / mean_standard_deviation and response table is generated

        # separate operation
        4. for each trial of each condition, zscore is calculated by (mean_response - mean_baseline) / mean_standard_deviation
        5. one-way anova is performed from these trial responses
        6. peak positive condition and peak negative condition is selected from previously generated response table
        7. ttest is performed for these two conditions against blank trial responses


        :param baseline_win:
        :param response_win:
        :return zscore_response_table:
        :return p_anova:
        :return p_ttest_pos:
        :return p_ttest_neg:
        """

        baseline_ind = np.logical_and(self.sta_ts > baseline_win[0], self.sta_ts <= baseline_win[1])
        response_ind = np.logical_and(self.sta_ts > response_win[0], self.sta_ts <= response_win[1])

        dgcrt = self.loc[:, ['alt', 'azi', 'sf', 'tf', 'dire', 'con', 'rad']]
        dgcrt['resp_mean'] = np.nan
        dgcrt['resp_max'] = np.nan
        dgcrt['resp_min'] = np.nan
        dgcrt['resp_std'] = np.nan
        dgcrt['resp_stdev'] = np.nan

        trial_responses = []

        for row_i, row in self.iterrows():
            curr_matrix = row['matrix']

            baseline = curr_matrix[:, baseline_ind].astype(np.float64)
            baseline_trial = np.mean(baseline, axis=1, keepdims=True)
            baseline_norm = baseline - baseline_trial
            std_mean = np.std(baseline_norm.flat)
            baseline_mean = np.mean(baseline_trial.flat)
            response_mean = np.mean(curr_matrix[:, response_ind].astype(np.float64).flat)
            dgcrt.loc[row_i, 'resp_mean'] = (response_mean - baseline_mean) / std_mean

            baseline_trial = np.mean(curr_matrix[:, baseline_ind].astype(np.float64), axis=1)
            # std_trial = np.std(curr_matrix[:, baseline_ind].astype(np.float64), axis=1)
            response_trial = np.mean(curr_matrix[:, response_ind].astype(np.float64), axis=1)
            curr_trial_responses = (response_trial - baseline_trial) / std_mean

            trial_responses.append(curr_trial_responses)
            dgcrt.loc[row_i, 'resp_max'] = np.max(curr_trial_responses)
            dgcrt.loc[row_i, 'resp_min'] = np.min(curr_trial_responses)
            dgcrt.loc[row_i, 'resp_std'] = np.std(curr_trial_responses)
            dgcrt.loc[row_i, 'resp_stdev'] = np.std(curr_trial_responses) / np.sqrt(len(curr_trial_responses))

        _, p_anova = stats.f_oneway(*trial_responses)

        zscore_response_table = DriftingGratingResponseTable(data=dgcrt, trace_type='{}_df'.format(self.trace_type))
        responses_blank = trial_responses[zscore_response_table.blank_condi_ind]
        responses_peak_pos = trial_responses[zscore_response_table.peak_condi_ind_pos]
        responses_peak_neg = trial_responses[zscore_response_table.peak_condi_ind_neg]

        _, p_ttest_pos = stats.ttest_rel(responses_blank, responses_peak_pos)
        _, p_ttest_neg = stats.ttest_rel(responses_blank, responses_peak_neg)

        return zscore_response_table, p_anova, p_ttest_pos, p_ttest_neg


class DriftingGratingResponseTable(DataFrame):
    """
    class for response table to drifting grating circle
    contains responses to all conditions of one roi

    subclassed from pandas.DataFrame with more attribute:
    trace_type: str, type of traces

    columns:
    alt  - float, altitute of circle center
    azi  - float, azimuth of circle center
    sf   - float, spatial frequency, cpd
    tf   - float, temporal frequency, Hz
    dire - int, drifting direction, deg, 0 is to right, increase counter-clockwise
    con  - float, contrast, [0, 1]
    rad  - float, radius, deg
    resp_mean - float, mean response to the condition
    resp_max - float, max response to the condition
    resp_min - float, min response to the condition
    resp_std - float, standard deviation across trials
    resp_stdev - float, standard error of mean across trials
    """

    def __init__(self, trace_type='', *args, **kwargs):

        super(DriftingGratingResponseTable, self).__init__(*args, **kwargs)

        self.trace_type = trace_type

    def get_peak_condi_params_pos(self):
        ind = self.peak_condi_ind_pos
        return self.loc[ind, ['sf', 'tf', 'dire', 'con', 'rad']]

    @property
    def sfs(self):
        return self['sf'].unique()

    @property
    def tfs(self):
        return self['tf'].unique()

    @property
    def dires(self):
        return self['dire'].unique()

    @property
    def cons(self):
        return self['con'].unique()

    @property
    def rads(self):
        return self['rad'].unique()

    @property
    def blank_condi_ind(self):
        """
        if more than one blank conditions found, raise error
        :return: int, blank condition index. None if no blank condition found
        """
        inds = []

        for row_i, row in self.iterrows():
            if row['sf'] == 0.:
                inds.append(row_i)
            if row['tf'] == 0.:
                inds.append(row_i)
            if row['con'] == 0.:
                inds.append(row_i)
            if row['rad'] == 0.:
                inds.append(row_i)

        inds = list(set(inds))

        if len(inds) == 0: # no blank condition
            return None
        elif len(inds) == 1: # 1 blank condition
            return inds[0]
        else:
            raise LookupError('more than one blank conditions found ({}).'.format(len(inds)))

    @property
    def peak_condi_ind_pos(self):
        """return the index of the condition with biggest postitive response (exclude the blank condition)"""
        if self.blank_condi_ind is None:
            return self['resp_mean'].argmax()
        else:
            return self.drop(self.blank_condi_ind)['resp_mean'].idxmax()

    @property
    def peak_condi_ind_neg(self):
        """return the index of the condition with biggest negative response (exclude the blank condition)"""
        if self.blank_condi_ind is None:
            return self['resp_mean'].argmin()
        else:
            return self.drop(self.blank_condi_ind)['resp_mean'].idxmin()

    @property
    def peak_response_pos(self):
        return self.loc[self.peak_condi_ind_pos, 'resp_mean']

    @property
    def peak_response_neg(self):
        return self.loc[self.peak_condi_ind_neg, 'resp_mean']

    @property
    def peak_response_abs(self):
        return np.max([abs(self.peak_response_pos),
                       abs(self.peak_response_neg)])

    def get_sf_tf_matrix(self, response_dir='pos'):
        """
        rerurn 2d array of sf/tf responses, rows: sf; cols: tf, other conditions are at peak in positive or negative
        direction
        :param response_dir: 'pos' or 'neg', response type to select peak condition
        :return responses: 2d array of 'resp_mean'
        :return sf_lst: 1d array, sf conditions
        :return tf_lst: 1d array, tf conditions
        """

        if response_dir == 'pos':
            ind_p = self.peak_condi_ind_pos
        elif response_dir == 'neg':
            ind_p = self.peak_condi_ind_neg
        else:
            raise LookupError('Do not understand response_dir ({}). Should be "pos" or "neg"'.format(response_dir))

        alt_p = self.loc[ind_p, 'alt']
        azi_p = self.loc[ind_p, 'azi']
        dire_p = self.loc[ind_p, 'dire']
        con_p = self.loc[ind_p, 'con']
        rad_p = self.loc[ind_p, 'rad']

        df_sub = self.loc[(self['alt'] == alt_p) & (self['azi'] == azi_p) & (self['dire'] == dire_p) &
                          (self['con'] == con_p) & (self['rad'] == rad_p)]

        sfs = list(df_sub['sf'].unique())
        sfs.sort()
        tfs = list(df_sub['tf'].unique())
        tfs.sort()

        resps = np.zeros((len(sfs), len(tfs)))
        resps[:] = np.nan

        for sf_i, sf in enumerate(sfs):
            for tf_i, tf in enumerate(tfs):

                curr_condi = df_sub[(df_sub['sf'] == sf) & (df_sub['tf'] == tf)]
                # print(curr_condi['resp_mean'])

                if not curr_condi.empty:
                    resps[sf_i, tf_i] = curr_condi['resp_mean']

        return resps, sfs, tfs

    def get_dire_tuning(self, response_dir='pos', is_collapse_sf=True, is_collapse_tf=False):
        """
        dataframe of direction responses, other conditions are at peak in positive or negative direction, if not
        specified by is_collapse
        :param is_collapse_sf: bool,
        :param is_collapse_tf: bool,
        :param response_dir: 'pos' or 'neg', response type to select peak condition
        :return dire_tuning: dataframe with two columns: 'dire','resp_mean', 'resp_max', 'resp_min', 'resp_std',
                             'resp_stdev'
        """

        if response_dir == 'pos':
            ind_p = self.peak_condi_ind_pos
        elif response_dir == 'neg':
            ind_p = self.peak_condi_ind_neg
        else:
            raise LookupError('Do not understand response_dir ({}). Should be "pos" or "neg"'.format(response_dir))

        alt_p = self.loc[ind_p, 'alt']
        azi_p = self.loc[ind_p, 'azi']
        sf_p = self.loc[ind_p, 'sf']
        tf_p = self.loc[ind_p, 'tf']
        con_p = self.loc[ind_p, 'con']
        rad_p = self.loc[ind_p, 'rad']

        # print('sf_p: {}'.format(sf_p))
        # print('tf_p: {}'.format(tf_p))

        df_sub = self.loc[(self['alt'] == alt_p) & (self['azi'] == azi_p) & (self['con'] == con_p) &
                          (self['rad'] == rad_p)]

        df_sub = df_sub[['sf', 'tf', 'dire', 'resp_mean', 'resp_max', 'resp_min', 'resp_std', 'resp_stdev']]
        # print(df_sub)

        if is_collapse_sf:
            df_sub = df_sub.groupby(['tf', 'dire']).mean().reset_index()
        else:
            df_sub = df_sub.loc[df_sub['sf'] == sf_p].drop('sf', axis=1)

        if is_collapse_tf:
            df_sub = df_sub.groupby(['dire']).mean().reset_index()
        else:
            df_sub = df_sub.loc[df_sub['tf'] == tf_p].drop('tf', axis=1)

        # print(df_sub)

        return df_sub[['dire', 'resp_mean', 'resp_max', 'resp_min', 'resp_std', 'resp_stdev']]

    def get_sf_tuning(self, response_dir='pos', is_collapse_tf=False, is_collapse_dire=False):
        """
        dataframe of sf responses, other conditions are at peak in positive or negative direction, if not
        specified by is_collapse
        :param is_collapse_tf: bool,
        :param is_collapse_dire: bool,
        :param response_dir: 'pos' or 'neg', response type to select peak condition
        :return sf_tuning: dataframe with two columns: 'sf','resp_mean', 'resp_max', 'resp_min', 'resp_std',
                             'resp_stdev'
        """

        if response_dir == 'pos':
            ind_p = self.peak_condi_ind_pos
        elif response_dir == 'neg':
            ind_p = self.peak_condi_ind_neg
        else:
            raise LookupError('Do not understand response_dir ({}). Should be "pos" or "neg"'.format(response_dir))

        alt_p = self.loc[ind_p, 'alt']
        azi_p = self.loc[ind_p, 'azi']
        dire_p = self.loc[ind_p, 'dire']
        tf_p = self.loc[ind_p, 'tf']
        con_p = self.loc[ind_p, 'con']
        rad_p = self.loc[ind_p, 'rad']

        df_sub = self.loc[(self['alt'] == alt_p) & (self['azi'] == azi_p) & (self['con'] == con_p) &
                          (self['rad'] == rad_p)]

        df_sub = df_sub[['sf', 'tf', 'dire', 'resp_mean', 'resp_max', 'resp_min', 'resp_std', 'resp_stdev']]
        # print(df_sub)

        if is_collapse_tf:
            df_sub = df_sub.groupby(['sf', 'dire']).mean().reset_index()
        else:
            df_sub = df_sub.loc[df_sub['tf'] == tf_p].drop('tf', axis=1)

        if is_collapse_dire:
            df_sub = df_sub.groupby(['sf']).mean().reset_index()
        else:
            df_sub = df_sub.loc[df_sub['dire'] == dire_p].drop('dire', axis=1)

        # print(df_sub)
        return df_sub[['sf', 'resp_mean', 'resp_max', 'resp_min', 'resp_std', 'resp_stdev']]

    def get_tf_tuning(self, response_dir='pos', is_collapse_sf=False, is_collapse_dire=False):
        """
        dataframe of tf responses, other conditions are at peak in positive or negative direction, if not
        specified by is_collapse
        :param is_collapse_sf: bool,
        :param is_collapse_dire: bool,
        :param response_dir: 'pos' or 'neg', response type to select peak condition
        :return tf_tuning: dataframe with two columns: 'tf','resp_mean', 'resp_max', 'resp_min', 'resp_std',
                             'resp_stdev'
        """

        if response_dir == 'pos':
            ind_p = self.peak_condi_ind_pos
        elif response_dir == 'neg':
            ind_p = self.peak_condi_ind_neg
        else:
            raise LookupError('Do not understand response_dir ({}). Should be "pos" or "neg"'.format(response_dir))

        alt_p = self.loc[ind_p, 'alt']
        azi_p = self.loc[ind_p, 'azi']
        dire_p = self.loc[ind_p, 'dire']
        sf_p = self.loc[ind_p, 'sf']
        con_p = self.loc[ind_p, 'con']
        rad_p = self.loc[ind_p, 'rad']

        df_sub = self.loc[(self['alt'] == alt_p) & (self['azi'] == azi_p) & (self['con'] == con_p) &
                          (self['rad'] == rad_p)]

        df_sub = df_sub[['sf', 'tf', 'dire', 'resp_mean', 'resp_max', 'resp_min', 'resp_std', 'resp_stdev']]
        # print(df_sub)

        if is_collapse_sf:
            df_sub = df_sub.groupby(['tf', 'dire']).mean().reset_index()
        else:
            df_sub = df_sub.loc[df_sub['sf'] == sf_p].drop('sf', axis=1)

        if is_collapse_dire:
            df_sub = df_sub.groupby(['tf']).mean().reset_index()
        else:
            df_sub = df_sub.loc[df_sub['dire'] == dire_p].drop('dire', axis=1)

        # print(df_sub)
        return df_sub[['tf', 'resp_mean', 'resp_max', 'resp_min', 'resp_std', 'resp_stdev']]

    @staticmethod
    def get_dire_tuning_properties(dire_tuning, response_dir='pos', is_rectify=True):
        """

        :param dire_tuning:
        :param response_dir:
        :param is_rectify:
        :return OSI:
        :return DSI:
        :return gOSI:
        :return gDSI:
        :return peak_dire_raw: optimal direction in tested conditions
        :return peak_dire_vs: optimal direction based on vector sum
        :return peak_orie_vs: optimal orientation based on vector sum
        """

        if response_dir == 'pos':
            pass
        elif response_dir == 'neg':
            dire_tuning['resp_mean'] = -dire_tuning['resp_mean']
        else:
            raise LookupError('Do not understand response_dir ({}). Should be "pos" or "neg"'.format(response_dir))

        if is_rectify:
            dire_tuning.loc[dire_tuning['resp_mean'] < 0., 'resp_mean'] = 0.

        dire_tuning['dire'] = dire_tuning['dire'] % 360

        if is_rectify:
            dire_tuning.loc[dire_tuning['resp_mean'] < 0., 'resp_mean'] = 0.

        if np.max(dire_tuning['resp_mean']) <= 0.:
            return tuple([np.nan] * 7)
        else:
            peak_dire_raw_ind = dire_tuning['resp_mean'].argmax()
            peak_dire_raw = dire_tuning.loc[peak_dire_raw_ind, 'dire']
            peak_resp = dire_tuning.loc[peak_dire_raw_ind, 'resp_mean']

            oppo_dire_ind = (dire_tuning['dire'] == ((peak_dire_raw + 180) % 360)).argmax()
            oppo_resp = dire_tuning.loc[oppo_dire_ind, 'resp_mean']

            othr_dire_ind_1 = (dire_tuning['dire'] == ((peak_dire_raw + 90) % 360)).argmax()
            othr_resp_1 = dire_tuning.loc[othr_dire_ind_1, 'resp_mean']

            othr_dire_ind_2 = (dire_tuning['dire'] == ((peak_dire_raw - 90) % 360)).argmax()
            othr_resp_2 = dire_tuning.loc[othr_dire_ind_2, 'resp_mean']

            othr_resp = np.mean([othr_resp_1, othr_resp_2])

            OSI = (peak_resp - othr_resp) / (peak_resp + othr_resp)
            DSI = (peak_resp - oppo_resp) / (peak_resp + oppo_resp)

            arcs = np.array(list(dire_tuning['dire'] * np.pi / 180))
            resp = np.array(list(dire_tuning['resp_mean']))

            vector_sum = np.sum(resp * np.exp(1j * arcs)) / np.sum(resp)
            peak_dire_vs = (np.angle(vector_sum) * 180 / np.pi) % 360
            gDSI = np.abs(vector_sum)

            vector_sum2 = np.sum(resp * np.exp(1j * 2 *arcs)) / np.sum(resp)
            peak_orie_vs = dire2ori(np.angle(vector_sum2) * 180 / np.pi)
            gOSI = np.abs(vector_sum2)

            return OSI, gOSI, DSI, gDSI, peak_dire_raw, peak_dire_vs, peak_orie_vs

    @staticmethod
    def get_tf_tuning_properties(tf_tuning, response_dir='pos', is_rectify=True):

        if response_dir == 'pos':
            pass
        elif response_dir == 'neg':
            tf_tuning['resp_mean'] = -tf_tuning['resp_mean']
        else:
            raise LookupError('Do not understand response_dir ({}). Should be "pos" or "neg"'.format(response_dir))

        if is_rectify:
            tf_tuning.loc[tf_tuning['resp_mean'] < 0., 'resp_mean'] = 0.

        if np.max(tf_tuning['resp_mean']) <= 0.:
            return tuple([np.nan] * 3)
        else:
            peak_tf_raw_ind = tf_tuning['resp_mean'].argmax()
            peak_tf_raw = tf_tuning.loc[peak_tf_raw_ind, 'tf']


            tfs = tf_tuning['tf'].astype(np.float)
            tfs_log = np.log(tfs) / np.log(2)
            resp = tf_tuning['resp_mean'].astype(np.float)

            peak_tf_linear = np.sum(tfs * resp) / np.sum(resp)

            peak_tf_log = np.sum(tfs_log * resp) / np.sum(resp)
            peak_tf_log = 2 ** peak_tf_log

            return peak_tf_raw, peak_tf_linear, peak_tf_log

    @staticmethod
    def get_sf_tuning_properties(sf_tuning, response_dir='pos', is_rectify=True):

        if response_dir == 'pos':
            pass
        elif response_dir == 'neg':
            sf_tuning['resp_mean'] = -sf_tuning['resp_mean']
        else:
            raise LookupError('Do not understand response_dir ({}). Should be "pos" or "neg"'.format(response_dir))

        if is_rectify:
            sf_tuning.loc[sf_tuning['resp_mean'] < 0., 'resp_mean'] = 0.

        if np.max(sf_tuning['resp_mean']) <= 0.:
            return tuple([np.nan] * 3)
        else:
            peak_sf_raw_ind = sf_tuning['resp_mean'].argmax()
            peak_sf_raw = sf_tuning.loc[peak_sf_raw_ind, 'sf']


            sfs = sf_tuning['sf'].astype(np.float)
            sfs_log = np.log(sfs / 0.01) / np.log(2)
            resp = sf_tuning['resp_mean'].astype(np.float)

            peak_sf_linear = np.sum(sfs * resp) / np.sum(resp)

            peak_sf_log = np.sum(sfs_log * resp) / np.sum(resp)
            peak_sf_log = 2 ** peak_sf_log * 0.01

            return peak_sf_raw, peak_sf_linear, peak_sf_log


if __name__ == '__main__':
    plt.ioff()
    # =====================================================================
    f = h5py.File(r"F:\data2\chandelier_cell_project\database\190208_M421761_110.nwb", 'r')
    dgcrm = get_dgc_response_matrix_from_h5(f['analysis/response_table_003_DriftingGratingCircleRetinotopicMapping/plane0'],
                                            roi_ind=0,
                                            trace_type='sta_f_center_subtracted')

    dgcrm_zscore = dgcrm.get_zscore_response_matrix(baseline_win=[-0.5, 0])
    dgcrt_zscore = dgcrm_zscore.get_response_table(response_win=[0., 1.])
    # print(dgcrt_zscore['resp_mean'])

    # sftf, sfs, tfs = dgcrt_zscore.get_sf_tf_matrix()
    # print(sftf)
    #

    dire_tuning = dgcrt_zscore.get_dire_tuning(response_dir='pos', is_collapse_sf=False, is_collapse_tf=False)
    print(dire_tuning)
    _ = DriftingGratingResponseTable.get_dire_tuning_properties(dire_tuning=dire_tuning,
                                                                response_dir='pos',
                                                                is_rectify=True)
    OSI, gOSI, DSI, gDSI, peak_dire_raw, peak_dire_vs, peak_orie_vs = _
    print('\nOSI: {}'.format(OSI))
    print('gOSI: {}'.format(gOSI))
    print('DSI: {}'.format(DSI))
    print('gDSI: {}'.format(gDSI))
    print('peak_dire_raw: {}'.format(peak_dire_raw))
    print('peak_dire_vs: {}'.format(peak_dire_vs))
    print('peak_orie_vs: {}\n'.format(peak_orie_vs))

    sf_tuning = dgcrt_zscore.get_sf_tuning(response_dir='pos', is_collapse_tf=False, is_collapse_dire=False)
    print(sf_tuning)
    _ = DriftingGratingResponseTable.get_sf_tuning_properties(sf_tuning=sf_tuning, response_dir='pos',
                                                              is_rectify=True)
    peak_sf_raw, peak_sf_linear, peak_sf_log = _
    print('\npeak_sf_raw: {}'.format(peak_sf_raw))
    print('peak_sf_linear: {}'.format(peak_sf_linear))
    print('peak_sf_log: {}\n'.format(peak_sf_log))

    tf_tuning = dgcrt_zscore.get_tf_tuning(response_dir='pos', is_collapse_sf=False, is_collapse_dire=False)
    print(tf_tuning)
    _ = DriftingGratingResponseTable.get_tf_tuning_properties(tf_tuning=tf_tuning, response_dir='pos',
                                                              is_rectify=True)
    peak_tf_raw, peak_tf_linear, peak_tf_log = _
    print('\npeak_tf_raw: {}'.format(peak_tf_raw))
    print('peak_tf_linear: {}'.format(peak_tf_linear))
    print('peak_tf_log: {}\n'.format(peak_tf_log))

    # =====================================================================

    # =====================================================================
    # f = h5py.File(r"E:\data2\2015-07-02-150610-M160809-2P_analysis\cells_test.hdf5")
    # STRF = load_STRF_FromH5(f['cell0003']['spatial_temporal_receptive_field'])
    # ampRFON, ampRFOFF = STRF.get_amplitude_receptive_field()
    #
    # print ampRFON.sign
    # print ampRFOFF.get_weighted_mask()[7,9]
    #
    # plt.imshow(ampRFON.get_weighted_mask(),interpolation='nearest')
    # plt.show()
    # =====================================================================

    # =====================================================================
    # f = h5py.File(r"E:\data2\2015-07-02-150610-M160809-2P_analysis\cells_test.hdf5")
    # STRF = load_STRF_FromH5(f['cell0003']['spatial_temporal_receptive_field'])
    # zscoreRFON, zscoreRFOFF = STRF.get_zscore_receptive_field()
    #
    # print zscoreRFON.sign
    # print zscoreRFOFF.get_weighted_mask()[7,9]
    #
    # plt.imshow(zscoreRFON.get_weighted_mask(),interpolation='nearest')
    # plt.show()
    # =====================================================================

    # =====================================================================
    # f = h5py.File(r"E:\data2\2015-07-02-150610-M160809-2P_analysis\cells_test.hdf5")
    # STRF = load_STRF_FromH5(f['cell0003']['spatial_temporal_receptive_field'])
    # zscoreRFON, zscoreRFOFF = STRF.get_amplitude_receptive_field()
    #
    # zscoreRFON.interpolate(10)
    #
    # plt.imshow(zscoreRFON.get_weighted_mask(),interpolation='nearest')
    # plt.show()
    # =====================================================================

    # =====================================================================
    # f = h5py.File(r"E:\data2\2015-07-02-150610-M160809-2P_analysis\cells_test.hdf5")
    # STRF = load_STRF_FromH5(f['cell0003']['spatial_temporal_receptive_field'])
    # STRF.shrink([-10,10],None)
    # print np.unique(np.array(STRF.get_locations())[:,0])
    # STRF.shrink(None,[0,20])
    # print np.unique(np.array(STRF.get_locations())[:,1])
    # =====================================================================

    # =====================================================================
    # dfile = h5py.File(r"G:\2016-08-15-160815-M238599-wf2p-Retinotopy\sparse_noise_2p\cells_refined.hdf5", 'r')
    # strf = SpatialTemporalReceptiveField.from_h5_group(dfile['cell0519']['spatial_temporal_receptive_field'])
    #
    # rf_on, rf_off, rf_all = strf.get_zscore_thresholded_receptive_fields(timeWindow=(0., 0.3), thr_ratio=0.4,
    #                                                                      filter_sigma=1., interpolate_rate=10,
    #                                                                      absolute_thr=0.8)
    #
    # peak_amplitude = max([np.amax(rf_on.get_weighted_mask()), np.amax(rf_off.get_weighted_mask())])
    #
    # f = plt.figure(figsize=(6, 8))
    # ax = f.add_subplot(111)
    # rf_on.plot_contour(ax, peak_amplitude=peak_amplitude, level_num=10, linewidths=1.5)
    # rf_off.plot_contour(ax, peak_amplitude=peak_amplitude, level_num=10, linewidths=1.5)
    # plt.show()

    # =====================================================================

    print 'for debug...'
