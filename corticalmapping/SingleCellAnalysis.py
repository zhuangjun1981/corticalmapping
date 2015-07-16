__author__ = 'junz'


import h5py
import numpy as np
import matplotlib.pyplot as plt
import core.PlottingTools as pt
import core.FileTools as ft

def load_ROI_FromH5(h5Group):
    '''
    load ROI (either ROI or WeightedROI) object from a hdf5 data group
    '''

    dimension = h5Group.attrs['dimension']
    pixelSize = h5Group.attrs['pixelSize']
    if pixelSize == 'None': pixelSize = None
    pixelSizeUnit = h5Group.attrs['pixelSizeUnit']
    if pixelSizeUnit == 'None': pixelSizeUnit = None
    pixels = h5Group['pixels'].value

    if 'weights' in h5Group.keys():
        weights = h5Group['weights'].value
        mask = np.zeros(dimension,dtype=np.float32); mask[tuple(pixels)]=weights
        return WeightedROI(mask,pixelSize=pixelSize,pixelSizeUnit=pixelSizeUnit)
    else:
        mask = np.zeros(dimension,dtype=np.uint8); mask[tuple(pixels)]=1
        return ROI(mask,pixelSize=pixelSize,pixelSizeUnit=pixelSizeUnit)


def load_STRF_FromH5(h5Group):
    '''
    load SpatialTemporalReceptiveField object from a hdf5 data group
    '''

    time = h5Group.attrs['time']
    locations = []
    signs = []
    traces = []
    for key, traceItem in h5Group.iteritems():
        locations.append(np.array([traceItem.attrs['altitude'],traceItem.attrs['azimuth']]))
        signs.append((traceItem.attrs['sign']))
        traces.append(traceItem.value)

    return SpatialTemporalReceptiveField(locations,signs,traces,time)


def getSparseNoiseOnsetIndex(sparseNoiseDisplayLog):
    '''
    return the indices of visual display frames for each square in a sparse noise display

    return:
    allOnsetInd: the indices of frames for each square, list
    onsetIndWithLocationSign: indices of frames for each white square, list with element structure [np.array([alt, azi]),sign,[list of indices]]
    '''

    framesSingleIter = sparseNoiseDisplayLog['stimulation']['frames']

    frames = framesSingleIter * sparseNoiseDisplayLog['presentation']['displayIteration']
    frames = [tuple([np.array([x[1][0],x[1][1]]),x[2],x[3],i]) for i, x in enumerate(frames)]
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



class ROI(object):
    '''
    class of binary ROI
    '''

    def __init__(self, mask, pixelSize = None, pixelSizeUnit = None):
        '''
        :param mask: 2-d array, if not binary, non-zero pixel will be included in mask,
                     zero-pixel will be considered as background
        :param pixelSize: float, can be None, one value (square pixel) or (width, height) for non-square pixel
        :param pixelSizeUnit: str, the unit of pixel size
        '''

        if len(mask.shape)!=2: raise ValueError, 'Input mask should be 2d.'

        self.dimension = mask.shape
        self.pixels = np.where(mask!=0)

        self.pixelSize = pixelSize
        if pixelSize is None: self.pixelSizeUnit=None
        else: self.pixelSizeUnit = pixelSizeUnit


    def getBinaryMask(self):
        '''
        generate binary mask of the ROI, return 2d array, with 0s and 1s, dtype np.uint8
        '''
        mask = np.zeros(self.dimension,dtype=np.uint8)
        mask[self.pixels] = 1
        return mask


    def getNanMask(self):
        '''
        generate float mask of the ROI, return 2d array, with nans and 1s, dtype np.float32
        '''
        mask = np.zeros(self.dimension,dtype=np.float32)
        mask[:] = np.nan
        mask[self.pixels] = 1
        return mask


    def getPixelArea(self):
        '''
        return the area coverage of the ROI
        '''
        return len(self.pixels[0])


    def getCenter(self):
        '''
        return the center coordinates of the centroid of the mask
        '''
        return np.mean(np.array(self.pixels,dtype=np.float).transpose(),axis=0)


    def getTrace(self,mov):
        '''
        return trace of this ROI in a given movie
        '''
        binaryMask = self.getBinaryMask()
        trace = np.multiply(mov,np.array([binaryMask])).sum(axis=1).sum(axis=1)
        return trace


    def plotBinaryMask(self,plotAxis=None,color='#ff0000',alpha=1):
        '''
        return display image (RGBA uint8 format) which can be plotted by plt.imshow, alpha: transparency 0-1
        '''
        mask = self.getBinaryMask()
        displayImg = pt.binary2RGBA(mask,foregroundColor=color,backgroundColor='#000000',foregroundAlpha=int(alpha*255),backgroundAlpha=0)
        if plotAxis is None: f=plt.figure();plotAxis=f.add_subplot(111);plotAxis.imshow(displayImg,interpolation='nearest')
        return displayImg


    def plotBinaryMaskBorder(self,**kwargs):
        pt.plotMask(self.getNanMask(),**kwargs)


    def toH5Group(self, h5Group):
        '''
        add attributes and dataset to a h5 data group
        '''
        h5Group.attrs['dimension'] = self.dimension
        if self.pixelSize is None: h5Group.attrs['pixelSize'] = 'None'
        else: h5Group.attrs['pixelSize'] = self.pixelSize
        if self.pixelSizeUnit is None: h5Group.attrs['pixelSizeUnit'] = 'None'
        else: h5Group.attrs['pixelSizeUnit'] = self.pixelSizeUnit

        dataDict = dict(self.__dict__)
        _ = dataDict.pop('dimension');_ = dataDict.pop('pixelSize');_ = dataDict.pop('pixelSizeUnit')
        for key, value in dataDict.iteritems():
            if value is None: h5Group.create_dataset(key,data='None')
            else: h5Group.create_dataset(key,data=value)



class WeightedROI(ROI):

    def __init__(self, mask, pixelSize = None, pixelSizeUnit = None):
        super(WeightedROI,self).__init__(mask, pixelSize = pixelSize, pixelSizeUnit = pixelSizeUnit)
        self.weights = mask[self.pixels]


    def getWeightedMask(self):
        mask = np.zeros(self.dimension,dtype=np.float32)
        mask[self.pixels] = self.weights
        return mask


    def getWeightedNanMask(self):
        mask = np.zeros(self.dimension,dtype=np.float32)
        mask[:]=np.nan
        mask[self.pixels] = self.weights
        return mask


    def getWeightedCenter(self):
        pixelCor = np.array(self.pixels,dtype=np.float)
        center = np.sum(np.multiply(pixelCor,np.array(self.weights)),axis=1)/np.sum(self.weights)
        return center


    def getWeightedTrace(self, mov):
        mask = self.generateWeightedMask()
        trace = np.multiply(mov,np.array([mask])).sum(axis=1).sum(axis=1)
        return trace


    def plotWeightedMask(self,plotAxis=None,color='#ff0000'):
        '''
        return display image (RGBA uint8 format) which can be plotted by plt.imshow
        '''
        mask = self.getWeightedMask()
        displayImg = pt.scalar2RGBA(mask,color=color)
        if plotAxis is None: f=plt.figure(); plotAxis=f.add_subplot(111); plotAxis.imshow(displayImg,interpolation='nearest')
        return displayImg


    def getTrace(self,mov):
        '''
        return trace of this ROI in a given movie
        '''
        weightedMask = self.getWeightedMask()
        trace = np.multiply(mov,np.array([weightedMask])).sum(axis=1).sum(axis=1)
        return trace



class SpatialTemporalReceptiveField(object):
    '''
    class of spatial temporal receptive field represented by traces for each specified retinotopic location
    '''

    def __init__(self,locations,signs,traces,time):
        '''
        locations: list of retinotopic locations mapped, array([altitude, azimuth])
        signs: list of signs for each location
        tracesON: list of traces for each location
                  list of 2-d array, each row: a single trace, each column: a single time point
        time: time axis for trace
        '''

        self.time = time
        dtype = [('altitude',float),('azimuth',float),('sign',int),('traces',list)]
        values = [ (location[0], location[1], signs[i], traces[i]) for i, location in enumerate(locations)]
        if not values: raise ValueError, 'Can not find input traces!'

        self.data = np.array(values,dtype=dtype)
        self.sortData()


    def mergeDuplication(self):
        #todo: merge traces with same retinotopic loacation
        pass


    def sortData(self):
        self.data = np.sort(self.data,order=['sign','altitude','azimuth'])


    def getDataType(self):
        return self.data.dtype


    def getLocations(self):
        return list(np.array([self.data['altitude'],self.data['azimuth'],self.data['sign']]).transpose())


    def addTraces(self,locations,signs,traces):

        '''
        add traces to existing receptive field
        '''

        dtype = [('altitude',float),('azimuth',float),('sign',int),('traces',list)]

        values = [ (location[0], location[1], signs[i], traces[i]) for i, location in enumerate(locations)]
        if not values: raise ValueError, 'Can not find input traces!'

        locations = [np.array([x[0],x[1],x[2]]) for x in values]

        objLocations = self.getLocations()

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

        self.sortData()


    def toH5Group(self, h5Group):

        h5Group.attrs['time'] = self.time
        h5Group.attrs['time_unit'] = 'second'
        h5Group.attrs['retinotopic_location_unit'] = 'degree'
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











if __name__=='__main__':

    plt.ioff()

    #=====================================================================
    # a = np.zeros((5,5))
    # a[0,4]=1
    # a[2,3]=1
    # roi = ROI(a)
    # print roi.getCenter()
    #=====================================================================

    #=====================================================================
    # mov = np.random.rand(5,4,4)
    # mask = np.zeros((4,4))
    # mask[2,3]=1
    # trace1 = mov[:,2,3]
    # roi = ROI(mask)
    # trace2 = roi.getTrace(mov)
    # assert(np.array_equal(trace1,trace2))
    #=====================================================================

    #=====================================================================
    # aa = np.random.rand(5,5)
    # mask = np.zeros((5,5))
    # mask[2,3]=aa[2,3]
    # mask[1,4]=aa[1,4]
    # mask[3,4]=aa[3,4]
    # roi = WeightedROI(mask)
    # center = roi.getCenter()
    # assert roi.getCenter()[0] == (2*aa[2,3]+1*aa[1,4]+3*aa[3,4])/(aa[2,3]+aa[1,4]+aa[3,4])
    #=====================================================================

    #=====================================================================
    # aa = np.zeros((50,50))
    # aa[15:20,30:35] = np.random.rand(5,5)
    # roi1 = ROI(aa)
    # _ = roi1.plotBinaryMaskBorder()
    # _ = roi1.plotBinaryMask()
    # roi2 = WeightedROI(aa)
    # _ = roi2.plotBinaryMaskBorder()
    # _ = roi2.plotBinaryMask()
    # _ = roi2.plotWeightedMask()
    # plt.show()
    #=====================================================================

    #=====================================================================
    pklPath = r"Z:\Jun\150610-M160809\SparseNoise_5x5_003\150610174646-SparseNoise-mouse160809-Jun-notTriggered.pkl"
    allOnsetInd, onsetIndWithLocationSign = getSparseNoiseOnsetIndex(ft.loadFile(pklPath))
    print allOnsetInd[0:10]
    print onsetIndWithLocationSign[0:3]
    #=====================================================================

    #=====================================================================
    # locations = [[3.0, 4.0], [3.0, 5.0], [2.0, 4.0], [2.0, 5.0],[3.0, 4.0], [3.0, 5.0], [2.0, 4.0], [2.0, 5.0]]
    # signs = [1,1,1,1,-1,-1,-1,-1]
    # traces=[[np.arange(4)],[np.arange(1,5)],[np.arange(2,6)],[np.arange(3,7)],[np.arange(5,9)],[np.arange(6,10)],[np.arange(7,11)],[np.arange(8,12)]]
    # time = np.arange(4,8)
    #
    # STRF = SpatialTemporalReceptiveField(locations,signs,traces,time)
    #
    # print STRF.data
    # print STRF.getLocations()
    #
    # newLocations = [[location[0]+1,location[1]+1] for location in locations[0:4]]
    # newSigns = [1,1,1,1]
    # STRF.addTraces(newLocations,newSigns,traces[0:4])
    #
    # print STRF.data
    #
    # testFile = h5py.File(r"C:\JunZhuang\labwork\data\python_temp_folder\test.hdf5")
    # STRFGroup = testFile.create_group('spatial_temporal_receptive_field')
    # STRF.toH5Group(STRFGroup)
    # testFile.close()
    #=====================================================================

    #=====================================================================
    # filePath = r"C:\JunZhuang\labwork\data\python_temp_folder\test.hdf5"
    # h5File = h5py.File(filePath)
    # STRF = load_STRF_FromH5(h5File['spatial_temporal_receptive_field'])
    # h5File.close()
    # print STRF.data
    #=====================================================================


    print 'for debug...'