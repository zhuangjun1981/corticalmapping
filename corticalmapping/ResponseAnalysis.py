import h5py
import math
import numpy as np
import warnings
import scipy.ndimage as ni
import scipy.interpolate as ip
import matplotlib.pyplot as plt
from pandas import DataFrame

import core.ImageAnalysis as ia
import core.TimingAnalysis as ta
import core.PlottingTools as pt

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


def get_dff_traces(traces, t_axis, response_window, baseline_window, add_to_trace=0.):
    """

    :param traces: dimension, trial x timepoint
    :param t_axis:
    :param response_window: [response start time, response end time]
    :param baseline_window: [baseline start time, baseline end time]
    :param add_to_trace: float
    :return dff_traces: 2d array, dff traces for every single row in traces
    :return dff_trace_mean: 1d array, dff trace for the mean trace of traces averaged across trials
    :return dff_mean: float, mean dff within response window from dff_trace_mean
    """

    if len(traces.shape) != 2:
        raise ValueError('input traces should be 2d array.')

    traces = traces.astype(np.float32) + float(add_to_trace)

    baseline_ind = np.logical_and(t_axis > baseline_window[0], t_axis <= baseline_window[1])
    response_ind = np.logical_and(t_axis > response_window[0], t_axis <= response_window[1])
    baseline = np.mean(traces[:, baseline_ind], axis=1, keepdims=True)
    dff_traces = (traces - baseline) / baseline

    trace_mean = np.mean(traces, axis=0)
    baseline_mean = np.mean(trace_mean[baseline_ind])
    dff_trace_mean = (trace_mean - baseline_mean) / baseline_mean
    dff_mean = np.mean(dff_trace_mean[response_ind])

    return dff_traces, dff_trace_mean, dff_mean


class SpatialReceptiveField(ia.WeightedROI):
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
        thr the current receptive field, return a new SpatialReceptiveField object after thresholding
        """

        if (self.thr is not None) and (thr < self.thr):
            raise ValueError, 'Can not cut a thresholded receptive field with a lower thresold!'
        cutRF = ia.get_peak_weighted_roi(self.get_weighted_mask(), thr)
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
            raise LookupError('To th area, the receptive field should be thresholded!!')

        alt_step = abs(np.mean(np.diff(self.altPos).astype(np.float)))
        azi_step = abs(np.mean(np.diff(self.aziPos).astype(np.float)))

        return len(self.weights) * alt_step * azi_step


class ResponseMatrix(object):
    """
    Object to store event triggered single trial traces. Base class for SpatialTemporalReceptiveField and
    DriftingGratingResponseMatrix

    Attributes
    ----------
    data : pandas.DataFrame
        each row represents a event condition. The last column should be 'traces' storing the event triggered single
        traces (2d array, trial x time point). The second last column should be 'trigger_ts', storing the global event
        trigger timestamps in seconds (1d) array. If global trigger timestamps not provided, an array of np.nan will
        be saved in this column. trigger_ts.shape[0] should equal traces.shape[0].
    time : 1d array, np.float64
        the local timestamps of event triggered average, should be monotonically increasing. for every condition,
        len(time) should equal traces.shape[1].
    trace_data_type : str
        signal type of traces, i.e. 'df/f', 'raw' etc. default 'unknown'.
    trace_data_unit : str
        unit of trace values, i.e. '%'. default 'unknown'.
    """

    def __init__(self, data, time, trace_data_type='unknown', trace_data_unit='unknown'):

        self.data = data
        self.time = np.array(time, dtype=np.float64)
        self.trace_data_type = str(trace_data_type)
        self.trace_data_unit = str(trace_data_unit)

        self.check_integrity()
        self.sort_conditions(is_ascending=True)
        self.merge_duplicates()

    def check_integrity(self):

        if not isinstance(self.data, DataFrame):
            raise ValueError('"data" should be pandas.DataFrame instance.')
        elif 'traces' not in self.data.columns.values.tolist():
            raise LookupError('"data" should have a column called "traces".')

        t_point_num = None
        for cond_i, cond in self.data.iterrows():
            curr_t_point_num = cond['traces'].shape[1]
            if t_point_num is None:
                t_point_num = curr_t_point_num
            else:
                if curr_t_point_num != t_point_num:
                    raise ValueError('number of trace timepoints in the {}th condition ({}) does not match'
                                     'those in other conditions ({})'.format(cond_i, curr_t_point_num, t_point_num))

        if len(self.time.shape) != 1:
            raise ValueError('"time" should be 1d array.')
        elif not ta.check_monotonicity(self.time, direction='increasing'):
            raise ValueError('"time" should be monotonically increasing.')
        elif len(self.time) != t_point_num:
            raise ValueError('the number of time points in "time" ({}) does not match number of time '
                             'points in traces ({}).'.format(len(self.time), t_point_num))

        if 'trigger_ts' in self.data.columns.values.tolist():
            for cond_i, cond in self.data.iterrows():
                curr_tts = cond['trigger_ts']
                curr_traces = cond['traces']
                if len(curr_tts.shape) != 1:
                    raise ValueError('the "trigger_ts" of {}th condition (shape: {}) is not a 1d array!'
                                     .format(cond_i, curr_tts.shape))
                elif len(curr_tts) != curr_traces.shape[0]:
                    raise ValueError('the number of trigger_ts of {}th condition ({}) does not match number of traces '
                                     'in that condition ({}).'.format(cond_i, len(curr_tts), curr_traces.shape[0]))
        else:
            trigger_tses = []
            for cond_i, cond in self.data.iterrows():
                trigger_num = cond['traces'].shape[0]
                trigger_ts = np.zeros(trigger_num, dtype=np.float64)
                trigger_ts[:] = np.nan
                trigger_tses.append(trigger_ts)
            self.data['trigger_ts'] = trigger_tses

        # rearrange columns
        cols = self.data.columns.tolist()
        cols.remove('trigger_ts')
        cols.remove('traces')
        cols.sort()
        cols = cols + ['trigger_ts', 'traces']
        self.data = self.data[cols]

        # can add more

    def get_param_names(self):
        params = self.data.columns.tolist()
        params.remove('trigger_ts')
        params.remove('traces')
        return(params)

    def get_conditions(self):
        params = self.get_param_names()
        return self.data[params]

    def sort_conditions(self, is_ascending=True):
        params = self.get_param_names()
        self.data.sort_values(by=params, inplace=True, ascending=is_ascending)

    def merge_duplicates(self):
        params = self.get_param_names()
        cond_grps = self.data.groupby(by=params)
        values = []
        for cond_n, cond_grp in cond_grps:
            curr_value = []
            for param in params:
                curr_value.append(cond_grp.iloc[0][param])
            curr_value.append(np.concatenate(list(cond_grp['trigger_ts'])))
            curr_value.append(np.concatenate(list(cond_grp['traces'])))

            values.append(curr_value)

        self.data = DataFrame(values, columns=params + ['trigger_ts', 'traces'])

    def merge(self, response_matrix):

        if len(self.time) != len(response_matrix.time):
            raise ValueError('the time axis length of input response matrix ({}) does not equal that of current '
                             'response matrix ({}). Cannot merge.'.format(len(response_matrix.time), len(self.time)))
        elif not np.array_equal(self.time, response_matrix.time):
            warnings.warn('the time axises of response matrices are not identical. Merging may not be appropriate.')

        attr_ns = self.__dict__.keys()
        attr_ns.remove('time')
        attr_ns.remove('data')
        attr_ns.sort()

        input_attr_ns = response_matrix.__dict__.keys()
        input_attr_ns.remove('time')
        input_attr_ns.remove('data')
        input_attr_ns.sort()

        if attr_ns != input_attr_ns:
            print('\nattributes of input response matrix:')
            print(input_attr_ns)
            print('\nattributes of current response matrix:')
            print(attr_ns)
            warnings.warn('The attributes of input response matrix does not match those of current response matrix. '
                          'using attributes of current response matrix.')

        if self.get_param_names() != response_matrix.get_param_names():
            print('\ncondition parameters of input response matrix:')
            print(response_matrix.get_param_names())
            print('\ncondition parameters of current response matrix:')
            print(self.get_param_names())

            raise ValueError('the condition parameters of input response matrix do not match those of current response '
                             'matrix. Cannot merge.')

        # print(self.data)
        # print(response_matrix.data)

        self.data = self.data.append(response_matrix.data, ignore_index=True)
        self.merge_duplicates()

        # print(self.data)

        self.sort_conditions(is_ascending=True)

        # print(self.data)

    def to_h5_group(self, h5_grp):

        attr_dict = dict(self.__dict__)
        attr_dict.pop('data')

        for attr_key, attr_value in attr_dict.items():
            try:
                h5_grp.attrs[attr_key] = attr_value
            except Exception as e:
                warnings.warn('cannot save attribute "{}" to hdf5 group as an attribute.'.format(attr_key))
                print(e)

        h5_grp.attrs['time_unit'] = 'second'
        h5_grp.attrs['trace_shape'] = '(trial, time_point)'

        params = self.get_param_names()

        for cond_i, cond in self.data.iterrows():
            cond_n = 'condition{:04d}'.format(cond_i)
            trace = h5_grp.create_dataset(cond_n, data=cond['traces'], dtype='f')
            trace.attrs['trigger_ts_sec'] = cond['trigger_ts']
            for param in params:
                trace.attrs[param] = cond[param]

    @staticmethod
    def from_h5_group(h5_grp):

        params = h5_grp[h5_grp.keys()[0]].attrs.keys()
        params.remove('trigger_ts_sec')

        value = []
        for cond_n, cond_dset in h5_grp.items():
            curr_value = []
            for param in params:
                curr_value.append(cond_dset.attrs[param])

            curr_value.append(cond_dset.attrs['trigger_ts_sec'])
            curr_value.append(cond_dset.value)
            value.append(curr_value)

        data = DataFrame(data=value, columns=params + ['trigger_ts', 'traces'])

        rm = ResponseMatrix(data=data, time=h5_grp.attrs['time'], trace_data_type=h5_grp.attrs['trace_data_type'],
                            trace_data_unit=h5_grp.attrs['trace_data_unit'])

        rm_attr_ns = h5_grp.attrs.keys()
        rm_attr_ns.remove('time')
        rm_attr_ns.remove('trace_data_type')
        rm_attr_ns.remove('trace_data_unit')
        rm_attr_ns.remove('time_unit')
        rm_attr_ns.remove('trace_shape')
        for rm_attr_n in rm_attr_ns:
            setattr(rm, rm_attr_n, h5_grp.attrs[rm_attr_n])

        return rm

    def get_trace_value_range(self, type='mean +/- sem'):
        """
        :param type: str, what type of trace the min and max will be applied upon
            should be one of: 'mean +/- sem', 'mean +/- std', 'raw'
        :return: max and min values of traces. this is mostly designed for plotting purposes
        """
        v_min = None
        v_max = None

        for cond_i, cond in self.data.iterrows():

            curr_trace = np.array(cond['traces'], dtype=np.float32)

            if type == 'raw':
                curr_trace_low = curr_trace
                curr_trace_high = curr_trace
            else:
                if curr_trace.shape[0] > 1:
                    curr_mean_trace = np.mean(curr_trace, axis=0, keepdims=True)
                    curr_std_trace = np.std(curr_trace, axis=0, keepdims=True)
                    if type == 'mean +/- std':
                        curr_trace_high = curr_mean_trace + curr_std_trace
                        curr_trace_low = curr_mean_trace - curr_std_trace
                    elif type == 'mean +/- sem':
                        curr_sem_trace = curr_std_trace / np.sqrt(float(curr_trace.shape[0]))
                        curr_trace_high = curr_mean_trace + curr_sem_trace
                        curr_trace_low = curr_mean_trace - curr_sem_trace
                    else:
                        raise NotImplementedError('"type" should be "mean +/- sem", "mean +/- std" or "raw".')
                else:
                    if type != 'mean +/- sem' and type != 'mean +/- std':
                        raise NotImplementedError('"type" should be "mean +/- sem", "mean +/- std" or "raw".')

                    warnings.warn('There is only one trace in {}th condition. Using raw trace instead of {}.'
                                  .format(cond_i, type))
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

    def get_dff_response_matrix(self, response_window, baseline_window):
        pass

    def temporal_filter(self, epochs):
        # todo: finish this. But need to do TimeIntervals containing TimeIntervals first in TimingAnalysis
        pass


class ResponseTable(object):
    pass


class SpatialTemporalReceptiveField(ResponseMatrix):

    def __init__(self, name='unknown', location_unit='degree', **kwargs):
        super(SpatialTemporalReceptiveField, self).__init__(**kwargs)
        self.name = str(name)
        self.location_unit = str(location_unit)

    def check_integrity(self):

        super(SpatialTemporalReceptiveField, self).check_integrity()

        if 'altitude' not in self.get_param_names():
            raise LookupError('Cannot find "altitude" as condition parameter.')
        if 'azimuth' not in self.get_param_names():
            raise LookupError('Cannot find "azimuth" as condition parameter.')
        if 'sign' not in self.get_param_names():
            raise LookupError('Cannot find "sign" as condition parameter.')

    def get_probes(self):
        return list(np.array([self.data['altitude'], self.data['azimuth'], self.data['sign']]).transpose())

    @staticmethod
    def from_components(locations,
                        signs,
                        traces,
                        time,
                        trigger_ts=None,
                        name='unknow',
                        location_unit='degree',
                        trace_data_type='unknow',
                        trace_data_unit='unknow'):
        """
        this is an adapter to make the current object sort of compatible with old class:
        SingleCellAnalysis.SpatialTemporalReceptiveField2

        :param locations: list, tuple or 2-d array of retinotopic locations mapped
            each element has two float numbers: [altitude, azimuth]
        :param signs: list, tuple or 1d array of signs for each location
        :param traces: list of traces for each location
            list of 2-d array, each row: a single trace, each column: a single time point
        :param time: time axis for trace
        :param trigger_ts: list of lists or tuples or 1-d arrays. The outside list should have same length as locations,
            signs and traces. Each element of the outside list correspond to one probe, and the inside lists save
            the global trigger timestamps of each trace for that particular probe. This is used for filtering
            the STRF for global epochs.
        :param name: str
        :param location_unit: str
        :param trace_data_type:  str
        :param trace_data_unit:  str
        :return:
        """

        print('\ngenerating spatial temporal receptive field ...')

        locations = np.array([np.array(l, dtype=np.float32) for l in locations])
        signs = np.array(signs, dtype=np.float32)
        time = np.array(time, dtype=np.float32)
        traces = [np.array([np.array(t, dtype=np.float32) for t in trace]) for trace in traces]

        for trace_i, trace in enumerate(traces):
            if trace.shape[1] != len(time):
                raise ValueError('the shape of {:d}th trace: {} is not consistent with length of time axis: {:d}.'
                                 .format(trace_i, trace.shape, len(time)))

        if not (len(locations) == len(signs) == len(traces)):
            raise ValueError('length of "locations", "signs", "traces" should be the same!')

        print('number of probes: {:d}'.format(len(locations)))

        if trigger_ts is None:
            trigger_ts = []
            for trace in traces:
                curr_trigger_ts = np.zeros(trace.shape[0], dtype=np.float64)
                curr_trigger_ts[:] = np.nan
                trigger_ts.append(curr_trigger_ts)
        else:
            trigger_ts = [np.array(ts, dtype=np.float64) for ts in trigger_ts]

        if len(trigger_ts) != len(locations):
            raise ValueError('length of trigger_ts: {:d} is not consistent with number of probes: {:d}.'
                             .format(len(trigger_ts), len(locations)))

        values = [(location[0], location[1], signs[i], trigger_ts[i], traces[i]) for i, location in
                  enumerate(locations)]
        if len(values) == 0:
            raise ValueError, 'Can not find input traces!'

        data = DataFrame(values, columns=['altitude', 'azimuth', 'sign', 'trigger_ts', 'traces'])

        return SpatialTemporalReceptiveField(data=data,
                                             time=time,
                                             trace_data_type=trace_data_type,
                                             trace_data_unit=trace_data_unit,
                                             name=name,
                                             location_unit=location_unit)

    def add_traces(self, locations, signs, traces, trigger_ts=None, verbose=False):
        """
        this is an adapter to make the current object sort of compatible with old class:
        SingleCellAnalysis.SpatialTemporalReceptiveField2

        :param locations: list, tuple or 2-d array of retinotopic locations mapped
            each element has two float numbers: [altitude, azimuth]
        :param signs: list, tuple or 1d array of signs for each location
        :param traces: list of traces for each location
            list of 2-d array, each row: a single trace, each column: a single time point
        :param trigger_ts: list of lists or tuples or 1-d arrays. The outside list should have same length as locations,
            signs and traces. Each element of the outside list correspond to one probe, and the inside lists save
            the global trigger timestamps of each trace for that particular probe. This is used for filtering
            the STRF for global epochs.
        """

        if verbose:
            print('adding traces to existing STRF ...')

        strf_to_add = SpatialTemporalReceptiveField.from_components(locations=locations,
                                                                    signs=signs,
                                                                    traces=traces,
                                                                    trigger_ts=trigger_ts,
                                                                    time=self.time,
                                                                    name=self.name,
                                                                    location_unit=self.location_unit,
                                                                    trace_data_type=self.trace_data_type,
                                                                    trace_data_unit=self.trace_data_unit)

        self.merge(strf_to_add)

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

        indON = np.array([np.array(x) for x in indON])
        indOFF = np.array([np.array(x) for x in indOFF])

        return indON, indOFF, allAltPos, allAziPos

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
        zscoreROION = ia.get_peak_weighted_roi(zscoreON, zscoreThr)
        zscoreROIOFF = ia.get_peak_weighted_roi(zscoreOFF, zscoreThr)
        if zscoreROION is not None and zscoreROIOFF is not None:
            zscoreROIALL = ia.WeightedROI(zscoreROION.get_weighted_mask() + zscoreROIOFF.get_weighted_mask())
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

            strf_dff = SpatialTemporalReceptiveField.from_components(
                locations=locations,
                signs=list(self.data['sign']),
                traces=dff_traces,
                trigger_ts=None,
                time=self.time,
                name=self.name,
                location_unit=self.location_unit,
                trace_data_type=self.trace_data_type + '_local_dff',
                trace_data_unit='%')

        else:
            strf_dff = SpatialTemporalReceptiveField.from_components(
                locations=locations,
                signs=list(self.data['sign']),
                traces=dff_traces,
                trigger_ts=list(self.data['trigger_ts']),
                time=self.time,
                name=self.name,
                location_unit=self.location_unit,
                trace_data_type=self.trace_data_type + '_local_dff',
                trace_data_unit='%')

        return strf_dff


class DriftingGratingResponseMatrix(ResponseMatrix):

    def __init__(self, name='unknown', altitude=np.nan, azimuth=np.nan, **kwargs):

        super(DriftingGratingResponseMatrix, self).__init__(**kwargs)
        self.name = str(name)
        self.altitude = altitude
        self.azimuth = azimuth

    def check_integrity(self):

        super(DriftingGratingResponseMatrix, self).check_integrity()

        if 'sf' not in self.get_param_names():
            raise LookupError('Cannot find "sf" as condition parameter.')
        if 'tf' not in self.get_param_names():
            raise LookupError('Cannot find "tf" as condition parameter.')
        if 'dire' not in self.get_param_names():
            raise LookupError('Cannot find "dire" as condition parameter.')
        if 'con' not in self.get_param_names():
            raise LookupError('Cannot find "con" as condition parameter.')
        if 'rad' not in self.get_param_names():
            raise LookupError('Cannot find "rad" as condition parameter.')


class DriftingGratingResponseTable(ResponseTable):
    pass