from corticalmapping.core.ImageAnalysis import ROI, WeightedROI

__author__ = 'junz'

import numpy as np
import matplotlib.pyplot as plt
import core.PlottingTools as pt
import core.ImageAnalysis as ia
import core.FileTools as ft
import scipy.ndimage as ni
import scipy.interpolate as ip
import math
import h5py


def get_sparse_noise_onset_index(sparseNoiseDisplayLog):
    """
    return the indices of visual display frames for each square in a sparse noise display

    return:
    allOnsetInd: the indices of frames for each square, list
    onsetIndWithLocationSign: indices of frames for each white square, list with element structure [np.array([alt, azi]),sign,[list of indices]]
    """


    frames = sparseNoiseDisplayLog['presentation']['displayFrames']
    frames = [tuple([np.array([x[1][1],x[1][0]]),x[2],x[3],i]) for i, x in enumerate(frames)]
    dtype = [('location',np.ndarray),('sign',int),('isOnset',int),('index',int)]
    frames = np.array(frames, dtype = dtype)

    allOnsetInd = []
    for i in range(len(frames)):
        if frames[i]['isOnset'] == 1 and (i == 0 or frames[i-1]['isOnset'] == -1):
            allOnsetInd.append(i)

    onsetFrames = frames[allOnsetInd]

    allSquares = list(set([tuple([x[0][0],x[0][1],x[1]]) for x in onsetFrames]))

    onsetIndWithLocationSign = []

    for square in allSquares:
        indices = []
        for onsetFrame in onsetFrames:
            if onsetFrame['location'][0]==square[0] and onsetFrame['location'][1]==square[1] and onsetFrame['sign']==square[2]:
                indices.append(onsetFrame['index'])

        onsetIndWithLocationSign.append([np.array([square[0],square[1]]),square[2],indices])

    return allOnsetInd, onsetIndWithLocationSign


def get_peak_weighted_roi(arr, thr):
    """
    return: a WeightROI object representing the mask which contains the peak of arr and cut by the thr (thr)
    """
    nanLabel = np.isnan(arr)
    arr2=arr.copy();arr2[nanLabel]=np.nanmin(arr)
    labeled,_=ni.label(arr2>=thr)
    peakCoor = np.array(np.where(arr2==np.amax(arr2))).transpose()[0]
    peakMask = ia.get_marked_masks(labeled, peakCoor)
    if peakMask is None: 'Threshold too high! No ROI found. Returning None'; return None
    else: return WeightedROI(arr2 * peakMask)


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

    if plot_axis == None: f=plt.figure(figsize=(10,10)); plot_axis=f.add_subplot(111)
    fig = plot_axis.imshow(mapArray,**kwargs)
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

    mask1 = roi1.get_weighted_mask(); mask2 = roi2.get_weighted_mask()

    return WeightedROI(mask1 + mask2, pixelSize=[roi1.pixelSizeY, roi1.pixelSizeX], pixelSizeUnit=roi1.pixelSizeUnit)


def merge_binary_rois(roi1, roi2):
    """
    merge two ROI objects, most useful for merge ON and OFF subfields
    """
    if (roi1.pixelSizeX != roi2.pixelSizeX) or (roi1.pixelSizeY != roi2.pixelSizeY):
        raise ValueError, 'The pixel sizes of the two WeightedROI objects should match!'

    if roi1.pixelSizeUnit != roi2.pixelSizeUnit:
        raise ValueError, 'The pixel size units of the two WeightedROI objects should match!'

    mask1 = roi1.get_binary_mask(); mask2 = roi2.get_binary_mask(); mask3 = np.logical_or(mask1, mask2).astype(np.int8)

    return ROI(mask3, pixelSize=[roi1.pixelSizeY, roi1.pixelSizeX], pixelSizeUnit=roi1.pixelSizeUnit)


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
        super(SpatialReceptiveField,self).__init__(mask, pixelSize = None, pixelSizeUnit = pixelSizeUnit)
        self.altPos = altPos
        self.aziPos = aziPos
        self.dataType = dataType

        if (sign is None or sign=='ON' or sign=='OFF' or sign=='ON_OFF'):
            self.sign=sign
        elif sign==1:
            self.sign='ON'
        elif sign==-1:
            self.sign='OFF'
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
            name.append('sigma:'+str(self.filter_sigma))
        else:
            name.append('sigma:None')

        if self.interpolate_rate is not None:
            name.append('interp:'+str(self.interpolate_rate))
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
            colors ='k'

        contour_levels = list(np.arange(level_num) *  (float(peak_amplitude) / (level_num)))

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

        if (self.thr is not None) and (thr<self.thr):
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
            raise LookupError('To th area, the receptive field should be thresholded!!')

        alt_step = abs(np.mean(np.diff(self.altPos).astype(np.float)))
        azi_step = abs(np.mean(np.diff(self.aziPos).astype(np.float)))

        return len(self.weights) * alt_step * azi_step


class SpatialTemporalReceptiveField(object):
    """
    class of spatial temporal receptive field represented by traces for each specified retinotopic location
    """

    def __init__(self,locations,signs,traces,time,name=None,locationUnit='degree', trace_data_type='dF_over_F'):
        """
        locations: list of retinotopic locations mapped, array([altitude, azimuth])
        signs: list of signs for each location
        traces: list of traces for each location
                list of 2-d array, each row: a single trace, each column: a single time point
        time: time axis for trace
        """

        # if not (len(locations) == len(signs) == len(traces)):
        #     raise ValueError('length of "locations", "signs", "traces" should be the same!')
        #
        # trace_dur = traces[0].shape[1]
        #
        # for i, trace in enumerate(traces):
        #     if trace.shape[1] != trace_dur:
        #         error_msg = "the " + str(i) + "th trace does not have required duration!"
        #         raise ValueError(error_msg)
        #
        # if time.shape[0] != trace_dur:
        #     raise ValueError("The number of sample points of each trace does not equal to number of data points in time "
        #                      "axis!")


        self.time = time
        self.name = name
        self.locationUnit = locationUnit
        self.trace_data_type = trace_data_type
        dtype = [('altitude', float),('azimuth', float),('sign', int),('traces', list)]
        values = [ (location[0], location[1], signs[i], traces[i]) for i, location in enumerate(locations)]
        if len(values) == 0:
            raise ValueError, 'Can not find input traces!'

        self.data = np.array(values, dtype=dtype)
        self.sort_data()

    def merge_duplication(self):
        #todo: merge traces with same retinotopic loacation
        pass

    def sort_data(self):
        self.data = np.sort(self.data, order=['sign','altitude','azimuth'])

    def get_data_type(self):
        return self.data.dtype

    def get_locations(self):
        return list(np.array([self.data['altitude'],self.data['azimuth'],self.data['sign']]).transpose())

    def add_traces(self, locations, signs, traces):

        """
        add traces to existing receptive field
        """

        if not (len(locations) == len(signs) == len(traces)):
            raise ValueError('length of "locations", "signs", "traces" should be the same!')

        for i, trace in enumerate(traces):
            if trace.shape[1] != self.time.shape[0]:
                error_msg = "The number of sample points of " + str(i) + "th trace does not equal to number of data " \
                                                                         "points in time axis!"
                raise ValueError(error_msg)

        dtype = [('altitude',float),('azimuth',float),('sign',int),('traces',np.ndarray)]

        values = [ (location[0], location[1], signs[i], traces[i]) for i, location in enumerate(locations)]
        if not values: raise ValueError, 'Can not find input traces!'

        locations = [np.array([x[0],x[1],x[2]]) for x in values]

        objLocations = self.get_locations()

        traceTuplesNeedToBeAdded = []

        for i, location in enumerate(locations):
            newTraceTuple = values[i]
            findSameLocation = False

            for j, objLocation in enumerate(objLocations):

                if np.array_equal(location,objLocation):
                    findSameLocation = True
                    objTraceItem = self.data[j]
                    objTraceItem['traces'] = np.vstack((objTraceItem['traces'], newTraceTuple[3]))

            if findSameLocation == False:
                traceTuplesNeedToBeAdded.append(tuple(newTraceTuple))

        if traceTuplesNeedToBeAdded:
            self.data = np.concatenate((self.data,np.array(traceTuplesNeedToBeAdded,dtype=dtype)),axis=0)

        self.sort_data()

    def to_h5_group(self, h5Group):

        h5Group.attrs['time'] = self.time
        h5Group.attrs['time_unit'] = 'second'
        h5Group.attrs['retinotopic_location_unit'] = self.locationUnit
        h5Group.attrs['trace_data_type'] = self.trace_data_type
        h5Group.attrs['trace_representation_axis'] = 0
        h5Group.attrs['trace_time_point_axis'] = 1
        if self.name is not None:
            h5Group.attrs['name'] = self.name
        else:
            h5Group.attrs['name'] = ''

        for i in range(len(self.data)):
            locationName = 'trace'+ft.int2str(i,4)
            trace = h5Group.create_dataset(locationName,data=self.data[i]['traces'], dtype='f')
            trace.attrs['altitude'] = self.data[i]['altitude']
            trace.attrs['azimuth'] = self.data[i]['azimuth']
            trace.attrs['sign'] = self.data[i]['sign']

    def plot_traces(self, f=None, figSize=(10, 10), yRange=(0, 20), altRange=None, aziRange=None, **kwargs):

        indexLists, axisLists = self._get_axis_layout(f, figSize, yRange, altRange, aziRange, **kwargs)

        for i, axisList in enumerate(axisLists):
            for j, axis in enumerate(axisList):
                indexList = indexLists[i][j]
                axis.set_axis_off()
                axis.set_xticks([]);axis.set_yticks([])
                for pos in ['top','bottom','left','right']:
                    axis.spines[pos].set_linewidth(0.5)
                    axis.spines[pos].set_color('#888888')
                axis.plot([0,0],[yRange[0],yRange[1]*0.5],'--',color='#888888',lw=0.5)
                axis.plot([self.time[0], self.time[-1]], [0., 0.], color='#888888', lw=0.5)

                for index in indexList:
                    traces = self.data[index]['traces']
                    traces = [t for t in traces if not math.isnan(t[0])]
                    meanTrace = np.mean(np.array(traces, dtype=np.float32),axis=0)

                    if self.data[index]['sign'] == 1:
                        color = '#ff0000'
                    elif self.data[index]['sign'] == -1:
                        color = '#0000ff'
                    else:
                        color = '#000000'

                    if len(traces) > 1:
                        stdTrace = np.std(np.array(traces, dtype=np.float32),axis=0)
                        semTrace = stdTrace/np.sqrt(float(len(traces)))
                        axis.fill_between(self.time,meanTrace - semTrace, meanTrace + semTrace, facecolor=color,
                                          linewidth=0, alpha=0.5)
                    axis.plot(self.time, meanTrace, '-', color=color, lw=1)

        return axisLists[0][0].figure

    def _get_axis_layout(self, f=None, figSize=(10, 10), yRange=(0, 20), altRange=None, aziRange=None, **kwargs):

        locations = np.array(self.get_locations())

        altPositions = np.sort(np.unique(locations[:,0]))[::-1]
        if altRange is not None: altPositions = np.array([x for x in altPositions if (x>=altRange[0] and x<=altRange[1])])

        aziPositions = np.sort(np.unique(locations[:,1]))
        if aziRange is not None: aziPositions = np.array([x for x in aziPositions if (x>=aziRange[0] and x<=aziRange[1])])

        indexLists = [ [[] for aziPosition in aziPositions] for altPosition in altPositions]

        if f is None: f=plt.figure(figsize=figSize)
        f.suptitle('cell:'+str(self.name)+'; xrange:['+str(self.time[0])[0:6]+','+str(self.time[-1])[0:6]+']; yrange:'+str(yRange))

        axisLists = pt.tile_axis(f, len(altPositions), len(aziPositions), **kwargs)

        for i, altPosition in enumerate(altPositions):
            for j, aziPosition in enumerate(aziPositions):
                axisLists[i][j].text(0,yRange[1],str(int(altPosition))+';'+str(int(aziPosition)),ha='left',va='top',fontsize=10)
                axisLists[i][j].set_xlim([self.time[0],self.time[-1]])
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

        windowIndex = np.logical_and(self.time>=timeWindow[0], self.time<=timeWindow[1])

        indON,indOFF,allAltPos,allAziPos = self._sort_index()

        ampON = np.zeros(indON.shape); ampON[:]=np.nan; ampOFF = ampON.copy()

        for i in np.ndindex(indON.shape):
            traceIndON = indON[i]; traceIndOFF = indOFF[i]
            if traceIndON is not None: ampON[i] = np.mean(np.mean(self.data[traceIndON]['traces'],axis=0)[windowIndex])
            if traceIndOFF is not None: ampOFF[i] = np.mean(np.mean(self.data[traceIndOFF]['traces'],axis=0)[windowIndex])

        return ampON, ampOFF, allAltPos, allAziPos

    def get_amplitude_receptive_field(self, timeWindow=(0, 0.5)):
        """
        very similar to get_amplitude_map(), only difference is that, it is returning spatial temporal receptive fields
        instead of 2d matrix
        each pixel in the map represent mean amplitute of traces within the window defined by timeWindow, and the
        coordinate of each pixel is defined by np.meshgrid(allAziPos, allAltPos)
        """

        ampON, ampOFF, allAltPos, allAziPos = self.get_amplitude_map(timeWindow)

        ampRFON = SpatialReceptiveField(ampON,allAltPos,allAziPos,sign=1,temporalWindow=timeWindow,pixelSizeUnit=self.locationUnit,dataType='amplitude')
        ampRFOFF = SpatialReceptiveField(ampOFF,allAltPos,allAziPos,sign=-1,temporalWindow=timeWindow,pixelSizeUnit=self.locationUnit,dataType='amplitude')

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

        ampON = np.zeros(indON.shape);
        ampON[:] = np.nan;
        ampOFF = ampON.copy()

        for i in np.ndindex(indON.shape):
            traceIndON = indON[i];
            traceIndOFF = indOFF[i]
            if traceIndON is not None:
                curr_trace_ON = np.mean(self.data[traceIndON]['traces'], axis=0)
                curr_baseline_ON = np.mean(curr_trace_ON[baseline_index])
                curr_delta_trace_ON = curr_trace_ON - curr_baseline_ON
                ampON[i] = np.mean(curr_delta_trace_ON[windowIndex])
            if traceIndOFF is not None:
                curr_trace_OFF = np.mean(self.data[traceIndOFF]['traces'], axis=0)
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

        ampRFON = SpatialReceptiveField(ampON, allAltPos, allAziPos, sign=1, temporalWindow=timeWindow,
                                        pixelSizeUnit=self.locationUnit, dataType='delta_amplitude')
        ampRFOFF = SpatialReceptiveField(ampOFF, allAltPos, allAziPos, sign=-1,temporalWindow=timeWindow,
                                         pixelSizeUnit=self.locationUnit, dataType='delta_amplitude')

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

        zscoreRFON = SpatialReceptiveField(ia.zscore(ampON),allAltPos,allAziPos,sign='ON',temporalWindow=timeWindow,
                                           pixelSizeUnit=self.locationUnit,dataType='zscore')
        zscoreRFOFF = SpatialReceptiveField(ia.zscore(ampOFF),allAltPos,allAziPos,sign='OFF',temporalWindow=timeWindow,
                                            pixelSizeUnit=self.locationUnit,dataType='zscore')

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

        return zscoreROION,zscoreROIOFF,zscoreROIALL,allAltPos,allAziPos

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

        zscoreRFON = SpatialReceptiveField(zscoreON, allAltPos, allAziPos, sign='ON',temporalWindow=timeWindow,
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

        thr =  max_value * thr_ratio

        if absolute_thr is not None:
            thr = max([thr, absolute_thr])

        zscoreRFON = zscoreRFON.threshold(thr)
        zscoreRFOFF = zscoreRFOFF.threshold(thr)

        zscoreRFALL = SpatialReceptiveField(zscoreRFON.get_weighted_mask()+zscoreRFOFF.get_weighted_mask(),
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
        zscoreROION,zscoreROIOFF,zscoreROIALL,allAltPos,allAziPos = self.get_zscore_rois(timeWindow, zscoreThr)
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

        indON = [[None for azi in allAziPos] for alt in allAltPos]; indOFF = [[None for azi in allAziPos] for alt in allAltPos]

        for i, traceItem in enumerate(self.data):
            alt = traceItem['altitude'];azi = traceItem['azimuth'];sign = traceItem['sign']
            for j, altPos in enumerate(allAltPos):
                for k, aziPos in enumerate(allAziPos):
                    if alt==altPos and azi==aziPos:
                        if sign==1:
                            if indON[j][k] is not None: raise LookupError, 'Duplication of trace items found at location:'+str([alt, azi])+'; sign: 1!'
                            else: indON[j][k]=i

                        if sign==-1:
                            if indOFF[j][k] is not None: raise LookupError, 'Duplication of trace items found at location:'+str([alt, azi])+'; sign:-1!'
                            else: indOFF[j][k]=i

        indON = np.array([np.array(x) for x in indON]); indOFF = np.array([np.array(x) for x in indOFF])

        return indON,indOFF,allAltPos,allAziPos

    def shrink(self,altRange=None,aziRange=None):
        """
        shrink the current spatial temporal receptive field into the defined altitude and/or azimuth range
        """

        if altRange is None and aziRange is None: raise LookupError, 'At least one of altRange and aziRange should be defined!'

        if altRange is not None: indAlt = np.logical_and(self.data['altitude']>=altRange[0],self.data['altitude']<=altRange[1])
        else: indAlt = np.ones(len(self.data),dtype=np.bool)
        if aziRange is not None: indAzi = np.logical_and(self.data['azimuth']>=aziRange[0],self.data['azimuth']<=aziRange[1])
        else: indAzi = np.ones(len(self.data),dtype=np.bool)
        ind = np.logical_and(indAlt,indAzi)
        self.data = self.data[ind]

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
        for key, traceItem in h5Group.iteritems():
            locations.append(np.array([traceItem.attrs['altitude'], traceItem.attrs['azimuth']]))
            signs.append((traceItem.attrs['sign']))
            traces.append(traceItem.value)

        return SpatialTemporalReceptiveField(locations, signs, traces, time, name, locationUnit, trace_data_type)

    def get_local_dff_strf(self, is_collaps_before_normalize=True, add_to_trace=0.):
        """

        :param is_collaps_before_normalize: if True, for each location, the traces across multiple trials will be
                                            averaged before calculating df/f
        :return:
        """

        bl_inds = self.time <= 0
        # print(bl_inds)

        dff_traces = []
        for roi_ind, roi_row in enumerate(self.data):
            curr_traces = np.array(roi_row['traces']) + add_to_trace

            if is_collaps_before_normalize:
                curr_traces = np.mean(curr_traces, axis=0, keepdims=True)

            curr_bl = np.mean(curr_traces[:, bl_inds], axis=1, keepdims=True)
            curr_dff = (curr_traces - curr_bl) / abs(curr_bl)

            dff_traces.append(list(curr_dff))

        locations = zip(self.data['altitude'], self.data['azimuth'])
        strf_dff = SpatialTemporalReceptiveField(locations=locations, signs=self.data['sign'], traces=dff_traces,
                                                 time=self.time, name=self.name, locationUnit=self.locationUnit,
                                                 trace_data_type=self.trace_data_type + '_local_dff')
        return strf_dff

    def get_data_range(self):

        v_min = None
        v_max = None

        for roi_ind, roi_row in enumerate(self.data):

            curr_trace = np.array(roi_row['traces'])

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


if __name__=='__main__':

    plt.ioff()

    #=====================================================================
    # f = h5py.File(r"E:\data2\2015-07-02-150610-M160809-2P_analysis\cells_test.hdf5")
    # STRF = load_STRF_FromH5(f['cell0003']['spatial_temporal_receptive_field'])
    # ampRFON, ampRFOFF = STRF.get_amplitude_receptive_field()
    #
    # print ampRFON.sign
    # print ampRFOFF.get_weighted_mask()[7,9]
    #
    # plt.imshow(ampRFON.get_weighted_mask(),interpolation='nearest')
    # plt.show()
    #=====================================================================

    #=====================================================================
    # f = h5py.File(r"E:\data2\2015-07-02-150610-M160809-2P_analysis\cells_test.hdf5")
    # STRF = load_STRF_FromH5(f['cell0003']['spatial_temporal_receptive_field'])
    # zscoreRFON, zscoreRFOFF = STRF.get_zscore_receptive_field()
    #
    # print zscoreRFON.sign
    # print zscoreRFOFF.get_weighted_mask()[7,9]
    #
    # plt.imshow(zscoreRFON.get_weighted_mask(),interpolation='nearest')
    # plt.show()
    #=====================================================================

    #=====================================================================
    # f = h5py.File(r"E:\data2\2015-07-02-150610-M160809-2P_analysis\cells_test.hdf5")
    # STRF = load_STRF_FromH5(f['cell0003']['spatial_temporal_receptive_field'])
    # zscoreRFON, zscoreRFOFF = STRF.get_amplitude_receptive_field()
    #
    # zscoreRFON.interpolate(10)
    #
    # plt.imshow(zscoreRFON.get_weighted_mask(),interpolation='nearest')
    # plt.show()
    #=====================================================================

    #=====================================================================
    # f = h5py.File(r"E:\data2\2015-07-02-150610-M160809-2P_analysis\cells_test.hdf5")
    # STRF = load_STRF_FromH5(f['cell0003']['spatial_temporal_receptive_field'])
    # STRF.shrink([-10,10],None)
    # print np.unique(np.array(STRF.get_locations())[:,0])
    # STRF.shrink(None,[0,20])
    # print np.unique(np.array(STRF.get_locations())[:,1])
    #=====================================================================

    # =====================================================================
    dfile = h5py.File(r"G:\2016-08-15-160815-M238599-wf2p-Retinotopy\sparse_noise_2p\cells_refined.hdf5", 'r')
    strf = SpatialTemporalReceptiveField.from_h5_group(dfile['cell0519']['spatial_temporal_receptive_field'])

    rf_on, rf_off, rf_all = strf.get_zscore_thresholded_receptive_fields(timeWindow=(0., 0.3), thr_ratio=0.4,
                                                                         filter_sigma=1., interpolate_rate=10,
                                                                         absolute_thr=0.8)

    peak_amplitude = max([np.amax(rf_on.get_weighted_mask()), np.amax(rf_off.get_weighted_mask())])

    f = plt.figure(figsize=(6, 8))
    ax = f.add_subplot(111)
    rf_on.plot_contour(ax, peak_amplitude=peak_amplitude, level_num=10, linewidths=1.5)
    rf_off.plot_contour(ax, peak_amplitude=peak_amplitude, level_num=10, linewidths=1.5)
    plt.show()


    # =====================================================================


    print 'for debug...'