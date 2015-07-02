__author__ = 'junz'


import numpy as np
import matplotlib.pyplot as plt
import core.PlottingTools as pt

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
        binMask = self.generateBinaryMask()
        trace = np.multiply(mov,np.array([binMask])).sum(axis=1).sum(axis=1)
        return trace


    def getDisplayImg(self,color='#ff0000',isPlot=False):
        '''
        return display image (RGBA uint8 format) which can be plotted by plt.imshow
        '''
        mask = self.getBinaryMask()
        displayImg = pt.binary2RGBA2(mask,color=color)
        if isPlot: plt.imshow(displayImg,interpolation='nearest'); plt.show()
        return displayImg






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


    def getDisplayImg(self,color='#ff0000',isPlot=False):
        '''
        return display image (RGBA uint8 format) which can be plotted by plt.imshow
        '''
        mask = self.getWeightedMask()
        displayImg = pt.binary2RGBA2(mask,color=color)
        if isPlot: plt.imshow(displayImg,interpolation='nearest'); plt.show()
        return displayImg




if __name__=='__main__':

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
    # _ = roi1.getDisplayImg(isPlot=True)
    # roi2 = WeightedROI(aa)
    # _ = roi2.getDisplayImg(isPlot=True)
    #=====================================================================


    print 'for debug...'