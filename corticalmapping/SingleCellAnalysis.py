from corticalmapping.core.ImageAnalysis import ROI, WeightedROI

__author__ = 'junz'

import numpy as np
import matplotlib.pyplot as plt
import core.PlottingTools as pt
import core.ImageAnalysis as ia
import core.FileTools as ft
import scipy.ndimage as ni
import scipy.interpolate as ip




def get_sparse_noise_onset_index(sparseNoiseDisplayLog):
    '''
    return the indices of visual display frames for each square in a sparse noise display

    return:
    allOnsetInd: the indices of frames for each square, list
    onsetIndWithLocationSign: indices of frames for each white square, list with element structure [np.array([alt, azi]),sign,[list of indices]]
    '''

    framesSingleIter = sparseNoiseDisplayLog['stimulation']['frames']

    frames = framesSingleIter * sparseNoiseDisplayLog['presentation']['displayIteration']
    frames = [tuple([np.array([x[1][1],x[1][0]]),x[2],x[3],i]) for i, x in enumerate(frames)]
    dtype = [('location',np.ndarray),('sign',int),('isOnset',int),('index',int)]
    frames = np.array(frames, dtype = dtype)

    allOnsetInd = np.where(frames['isOnset']==1)[0]

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
    '''
    return: a WeightROI object representing the mask which contains the peak of arr and cut by the threshold (thr)
    '''
    nanLabel = np.isnan(arr)
    arr2=arr.copy();arr2[nanLabel]=np.nanmin(arr)
    labeled,_=ni.label(arr2>=thr)
    peakCoor = np.array(np.where(arr2==np.amax(arr2))).transpose()[0]
    peakMask = ia.get_marked_masks(labeled, peakCoor)
    if peakMask is None: 'Threshold too high! No ROI found. Returning None'; return None
    else: return WeightedROI(arr2 * peakMask)


def plot_2d_receptive_field(mapArray, altPos, aziPos, plotAxis=None, **kwargs):
    '''
    plot a 2-d receptive field in a given axis

    :param mapArray: 2-d array, should be in the same coordinate system as meshgrid(aziPos,altPos)
    :param altPos: 1-d array, list of sample altitude positions, sorted from high to low
    :param aziPos: 1-d array, list of sample azimuth position, sorted from low to high
    :param plotAxis:
    :param kwargs: input to matplotlib.pyplot.imshow() function
    :return: plotAxis
    '''

    if plotAxis == None: f=plt.figure(figsize=(10,10)); plotAxis=f.add_subplot(111)
    fig = plotAxis.imshow(mapArray,**kwargs)
    plotAxis.set_yticks(np.arange(len(altPos)))
    plotAxis.set_xticks(np.arange(len(aziPos)))
    plotAxis.set_yticklabels(altPos.astype(np.int))
    plotAxis.set_xticklabels(aziPos.astype(np.int))
    return fig


def merge_weighted_rois(roi1, roi2):
    '''
    merge two WeightedROI objects, most useful for merge ON and OFF subfields
    '''
    if (roi1.pixelSizeX != roi2.pixelSizeX) or (roi1.pixelSizeY != roi2.pixelSizeY):
        raise ValueError, 'The pixel sizes of the two WeightedROI objects should match!'

    if roi1.pixelSizeUnit != roi2.pixelSizeUnit:
        raise ValueError, 'The pixel size units of the two WeightedROI objects should match!'

    mask1 = roi1.get_weighted_mask(); mask2 = roi2.get_weighted_mask()

    return WeightedROI(mask1 + mask2, pixelSize=[roi1.pixelSizeY, roi1.pixelSizeX], pixelSizeUnit=roi1.pixelSizeUnit)


def merge_binary_rois(roi1, roi2):
    '''
    merge two ROI objects, most useful for merge ON and OFF subfields
    '''
    if (roi1.pixelSizeX != roi2.pixelSizeX) or (roi1.pixelSizeY != roi2.pixelSizeY):
        raise ValueError, 'The pixel sizes of the two WeightedROI objects should match!'

    if roi1.pixelSizeUnit != roi2.pixelSizeUnit:
        raise ValueError, 'The pixel size units of the two WeightedROI objects should match!'

    mask1 = roi1.get_binary_mask(); mask2 = roi2.get_binary_mask(); mask3 = np.logical_or(mask1, mask2).astype(np.int8)

    return ROI(mask3, pixelSize=[roi1.pixelSizeY, roi1.pixelSizeX], pixelSizeUnit=roi1.pixelSizeUnit)


class SpatialReceptiveField(WeightedROI):
    '''
    Object for spatial receptive field, a subclass of WeightedROI object
    '''

    def __init__(self, mask, altPos, aziPos, sign=None,temporalWindow=None, pixelSizeUnit=None, dataType=None, isThresholded=False, threshold=None):
        '''
        subclass of WeightedROI object, because the pixel coordinates are defined by np.meshgrid(aziPos, altPos),
        the old WeightedROI attribute: pixelSize does not make sense, so set it to be None.
        '''
        super(SpatialReceptiveField,self).__init__(mask, pixelSize = None, pixelSizeUnit = pixelSizeUnit)
        self.altPos = altPos
        self.aziPos = aziPos
        self.dataType = dataType
        self.sign=sign
        self.temporalWindow = temporalWindow
        self.isThresholded = isThresholded
        if isThresholded: self.threshold = threshold
        else: self.threshold = None


    def threshold_receptive_field(self, thr):

        '''
        threshold the current receptive field, return a new SpatialReceptiveField object after thresholding
        '''

        if (self.threshold is not None) and (thr<self.threshold):
            raise ValueError, 'Can not cut a thresholded receptive field with a lower thresold!'
        cutRF = get_peak_weighted_roi(self.get_weighted_mask(), thr)
        if cutRF is None: raise LookupError, 'No ROI found. Threshold too high!'

        return SpatialReceptiveField(cutRF.get_weighted_mask(), self.sign, self.temporalWindow, self.pixelSizeUnit, self.dataType, isThresholded=True, threshold=thr)


    def interpolate(self, ratio, method='cubic',fill_value=0.):

        altInterpolation = ip.interp1d(np.arange(len(self.altPos)),self.altPos); aziInterpolation = ip.interp1d(np.arange(len(self.aziPos)),self.aziPos)
        newAltPos = altInterpolation(np.arange(0,len(self.altPos)-1,1./int(ratio)))
        newAziPos = aziInterpolation(np.arange(0,len(self.aziPos)-1,1./int(ratio)))
        mask = self.get_weighted_mask(); aziGrid, altGrid=np.meshgrid(self.aziPos, self.altPos)
        newAziGrid, newAltGrid = np.meshgrid(newAziPos,newAltPos)
        newMask = ip.griddata(np.array([aziGrid.flatten(),altGrid.flatten()]).transpose(),
                              mask.flatten(),
                              (newAziGrid,newAltGrid),
                              method=method,fill_value=fill_value)

        self.__init__(newMask,newAltPos,newAziPos,self.sign,self.temporalWindow,self.pixelSizeUnit,self.dataType,self.isThresholded,self.threshold)



class SpatialTemporalReceptiveField(object):
    '''
    class of spatial temporal receptive field represented by traces for each specified retinotopic location
    '''

    def __init__(self,locations,signs,traces,time,name=None,locationUnit='degree'):
        '''
        locations: list of retinotopic locations mapped, array([altitude, azimuth])
        signs: list of signs for each location
        tracesON: list of traces for each location
                  list of 2-d array, each row: a single trace, each column: a single time point
        time: time axis for trace
        '''

        self.time = time
        self.name = name
        self.locationUnit = locationUnit
        dtype = [('altitude',float),('azimuth',float),('sign',int),('traces',list)]
        values = [ (location[0], location[1], signs[i], traces[i]) for i, location in enumerate(locations)]
        if not values: raise ValueError, 'Can not find input traces!'

        self.data = np.array(values,dtype=dtype)
        self.sort_data()


    def merge_duplication(self):
        #todo: merge traces with same retinotopic loacation
        pass


    def sort_data(self):
        self.data = np.sort(self.data,order=['sign','altitude','azimuth'])


    def get_data_type(self):
        return self.data.dtype


    def get_locations(self):
        return list(np.array([self.data['altitude'],self.data['azimuth'],self.data['sign']]).transpose())


    def add_traces(self, locations, signs, traces):

        '''
        add traces to existing receptive field
        '''

        dtype = [('altitude',float),('azimuth',float),('sign',int),('traces',list)]

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
                    objTraceItem['traces'] = objTraceItem['traces'] + newTraceTuple[3]

            if findSameLocation == False:
                traceTuplesNeedToBeAdded.append(tuple(newTraceTuple))

        if traceTuplesNeedToBeAdded:
            self.data = np.concatenate((self.data,np.array(traceTuplesNeedToBeAdded,dtype=dtype)),axis=0)

        self.sort_data()


    def to_h5_Group(self, h5Group):

        h5Group.attrs['time'] = self.time
        h5Group.attrs['time_unit'] = 'second'
        h5Group.attrs['retinotopic_location_unit'] = self.locationUnit
        h5Group.attrs['trace_data_type'] = 'dF_over_F'
        h5Group.attrs['trace_data_unit'] = '%'
        h5Group.attrs['trace_representation_axis'] = 0
        h5Group.attrs['trace_time_point_axis'] = 1

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

                for index in indexList:
                    traces = self.data[index]['traces']
                    meanTrace = np.mean(traces,axis=0)
                    stdTrace = np.std(traces,axis=0)
                    semTrace = stdTrace/np.sqrt(float(len(traces)))
                    if self.data[index]['sign'] == 1: color = '#ff0000'
                    if self.data[index]['sign'] == -1: color = '#0000ff'
                    axis.fill_between(self.time,meanTrace-semTrace,meanTrace+semTrace,facecolor=color,linewidth=0,alpha=0.5)
                    axis.plot(self.time,meanTrace,'-',color=color,lw=1)

        return f


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
        '''
        return 2d receptive field map and altitude and azimuth coordinates
        each pixel in the map represent mean amplitute of traces within the window defined by timeWindow, and the
        coordinate of each pixel is defined by np.meshgrid(allAziPos, allAltPos)
        '''

        windowIndex = np.logical_and(self.time>=timeWindow[0], self.time<=timeWindow[1])

        indON,indOFF,allAltPos,allAziPos = self._sort_index()

        ampON = np.zeros(indON.shape); ampON[:]=np.nan; ampOFF = ampON.copy()

        for i in np.ndindex(indON.shape):
            traceIndON = indON[i]; traceIndOFF = indOFF[i]
            if traceIndON is not None: ampON[i] = np.mean(np.mean(self.data[traceIndON]['traces'],axis=0)[windowIndex])
            if traceIndOFF is not None: ampOFF[i] = np.mean(np.mean(self.data[traceIndOFF]['traces'],axis=0)[windowIndex])

        return ampON, ampOFF, allAltPos, allAziPos


    def get_amplitude_receptive_field(self, timeWindow=(0, 0.5)):
        '''
        very similar to get_amplitude_map(), only difference is that, it is returning spatial temporal receptive fields
        instead of 2d matrix
        each pixel in the map represent mean amplitute of traces within the window defined by timeWindow, and the
        coordinate of each pixel is defined by np.meshgrid(allAziPos, allAltPos)
        '''

        ampON, ampOFF, allAltPos, allAziPos = self.get_amplitude_map(timeWindow)

        ampRFON = SpatialReceptiveField(ampON,allAltPos,allAziPos,sign=1,temporalWindow=timeWindow,pixelSizeUnit=self.locationUnit,dataType='amplitude')
        ampRFOFF = SpatialReceptiveField(ampOFF,allAltPos,allAziPos,sign=-1,temporalWindow=timeWindow,pixelSizeUnit=self.locationUnit,dataType='amplitude')

        return ampRFON, ampRFOFF


    def get_zscore_map(self, timeWindow=(0, 0.5)):
        '''
        return 2d receptive field and altitude and azimuth coordinates
        each pixel in the map represent Z score of mean amplitute of traces within the window defined by timeWindow
        '''

        ampON, ampOFF, allAltPos, allAziPos = self.get_amplitude_map(timeWindow)

        return ia.zscore(ampON), ia.zscore(ampOFF), allAltPos, allAziPos


    def get_zscore_receptive_field(self, timeWindow=(0, 0.5)):
        '''
        very similar to get_zscore_map(), only difference is that, it is returning spatial temporal receptive fields
        instead of 2d matrix
        each pixel in the map represent mean amplitute of traces within the window defined by timeWindow, and the
        coordinate of each pixel is defined by np.meshgrid(allAziPos, allAltPos)
        '''

        ampON, ampOFF, allAltPos, allAziPos = self.get_amplitude_map(timeWindow)

        zscoreRFON = SpatialReceptiveField(ia.zscore(ampON),allAltPos,allAziPos,sign=1,temporalWindow=timeWindow,pixelSizeUnit=self.locationUnit,dataType='zscore')
        zscoreRFOFF = SpatialReceptiveField(ia.zscore(ampOFF),allAltPos,allAziPos,sign=-1,temporalWindow=timeWindow,pixelSizeUnit=self.locationUnit,dataType='zscore')

        return zscoreRFON, zscoreRFOFF


    def get_zscore_rois(self, timeWindow=(0, 0.5), zscoreThr=2):
        '''
        return ON, OFF and combined receptive field rois in the format of WeightedROI object

        Amplitude for each pixel was calculated as mean dF over F signal trace within the timeWindow
        mask of ON and OFF receptive field was generated by cutting zscore map by zscoreThr
        Tombined mask is the sum of ON and OFF weighted mask

        The sampled altitude positions and azimuth positions are also returned. The receptive field space coordinates
        were defined as np.meshgrid(allAziPos, allAltPos)
        '''
        zscoreON, zscoreOFF, allAltPos, allAziPos = self.get_zscore_map(timeWindow)
        zscoreROION = get_peak_weighted_roi(zscoreON, zscoreThr)
        zscoreROIOFF = get_peak_weighted_roi(zscoreOFF, zscoreThr)
        if zscoreON is not None and zscoreROIOFF is not None:
            zscoreROIALL = WeightedROI(zscoreROION.get_weighted_mask() + zscoreROIOFF.get_weighted_mask())
        elif zscoreROION is None and zscoreROIOFF is not None:
            zscoreROIALL = zscoreROIOFF
        elif zscoreROION is not None and zscoreROIOFF is None:
            zscoreROIALL = zscoreROION
        else: raise LookupError, 'No ROI found for both ON and OFF channel. Threshold too high!'

        return zscoreROION,zscoreROIOFF,zscoreROIALL,allAltPos,allAziPos


    def get_zscore_roi_centers(self, timeWindow=(0, 0.5), zscoreThr=2):
        '''
        return retinotopic location of ON subfield, OFF subfield and combined receptive field

        zscore ROIs was generated by the method get_zscore_rois()
        '''
        zscoreROION,zscoreROIOFF,zscoreROIALL,allAltPos,allAziPos = self.get_zscore_rois(timeWindow, zscoreThr)
        if zscoreROION is not None: centerON = zscoreROION.get_weighted_center_in_coordinate(allAltPos, allAziPos)
        else: centerON = None

        if zscoreROIOFF is not None: centerOFF = zscoreROIOFF.get_weighted_center_in_coordinate(allAltPos, allAziPos)
        else: centerOFF = None

        centerALL = zscoreROIALL.get_weighted_center_in_coordinate(allAltPos, allAziPos)
        return centerON, centerOFF, centerALL


    def _sort_index(self):
        '''
        return ON and OFF index matrices for all combination of sampled retinotopic locations along with retinotopic
        coordinates, the retinotopic visual was defined by np.meshgrid(allAziPos, allAltPos)
        '''

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
        '''
        shrink the current spatial temporal receptive field into the
        '''

        if altRange is None and aziRange is None: raise LookupError, 'At least one of altRange and aziRange should be defined!'

        if altRange is not None: indAlt = np.logical_and(self.data['altitude']>=altRange[0],self.data['altitude']<=altRange[1])
        else: indAlt = np.ones(len(self.data),dtype=np.bool)
        if aziRange is not None: indAzi = np.logical_and(self.data['azimuth']>=aziRange[0],self.data['azimuth']<=aziRange[1])
        else: indAzi = np.ones(len(self.data),dtype=np.bool)
        ind = np.logical_and(indAlt,indAzi)
        self.data = self.data[ind]

    @staticmethod
    def from_h5_group(h5Group):
        '''
        load SpatialTemporalReceptiveField object from a hdf5 data group
        '''

        time = h5Group.attrs['time']
        try:name = h5Group.parent.name[1:]+'.'+h5Group.parent.attrs['name']
        except KeyError: name=None
        locationUnit = h5Group.attrs['retinotopic_location_unit']
        locations = []
        signs = []
        traces = []
        for key, traceItem in h5Group.iteritems():
            locations.append(np.array([traceItem.attrs['altitude'],traceItem.attrs['azimuth']]))
            signs.append((traceItem.attrs['sign']))
            traces.append(list(traceItem.value))

        return SpatialTemporalReceptiveField(locations,signs,traces,time,name,locationUnit)
















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


    print 'for debug...'