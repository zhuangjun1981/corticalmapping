__author__ = 'junz'


import numpy as np
import matplotlib.pyplot as plt
import core.PlottingTools as pt

def loadROIFromH5(h5Group):
    '''
    load ROI (either ROI or WeightedROI) class from a hdf5 data group
    '''

    dimension = h5Group.attrs['dimension']
    pixelSize = h5Group.attrs['pixelSize']
    if pixelSize == 'None': pixelSize = None
    pixelSizeUnit = h5Group.attrs['pixelSizeUnit']
    if pixelSizeUnit == 'None': pixelSizeUnit = None
    pixels = h5Group['pixels'].value

    if 'weights' in h5Group.keys():
        weights = h5Group['weights'].value
        mask = np.zeros(dimension,dtype=np.float32); mask[pixels]=weights
        return WeightedROI(mask,pixelSize=pixelSize,pixelSizeUnit=pixelSizeUnit)
    else:
        mask = np.zeros(dimension,dtype=np.uint8); mask[pixels]=1
        return ROI(mask,pixelSize=pixelSize,pixelSizeUnit=pixelSizeUnit)


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


    def getH5Group(self, h5Group):
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


    print 'for debug...'