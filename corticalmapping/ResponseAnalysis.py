import h5py
import numpy as np
import warnings
import scipy.ndimage as ni
import scipy.interpolate as ip
import matplotlib.pyplot as plt
from pandas import DataFrame

import core.ImageAnalysis as ia
import core.TimingAnalysis as ta

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


class ResponseTable(object):
    pass


class SpatialTemporalReceptiveField(ResponseMatrix):

    def __init__(self, name='unknown', location_unit='degree', **kwargs):
        super(SpatialTemporalReceptiveField, self).__init__(**kwargs)
        self.name = str(name)
        self.location_unit = str(location_unit)

    def check_integrity(self):

        super(SpatialTemporalReceptiveField, self).check_integrity()
        #todo add more

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


class DriftingGratingResponseMatrix(ResponseMatrix):
    pass


class DriftingGratingResponseTable(ResponseTable):
    pass