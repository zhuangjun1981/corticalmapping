__author__ = 'junz'

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import scipy.ndimage as ni
from matplotlib import cm
import matplotlib.colors as col
import skimage.morphology as sm
import FileTools as ft
try: import cv2
except ImportError as e: print e


def resample(t1,y1,interval,kind='linear', isPlot = False):

    '''
    :param t1: time stamps of original data
    :param y1: value of original data (relative to t1)
    :param interval: the intervals of resampled time stamps, second
    :param kind: interpolation type, same as 'scipy.interpolate.interp1d'
    :return: t2, y2
    '''

    f = interpolate.interp1d(t1,y1,kind=kind)

    t2 = np.arange(t1[0],t1[-1], interval)

    y2 = f(t2)

    if isPlot:
        plt.figure()
        plt.plot(t1,y1)
        plt.plot(t2,y2)
        plt.legend('original','resampled')

    return t2, y2


def resample2(t1, y1, t2, kind = 'linear', isPlot=False, bounds_error=False):

    '''
    :param t1: time stamps of original data
    :param y1: value of original data (relative to t1)
    :param t2: time stamps for resample
    :param kind: interpolation type, same as 'scipy.interpolate.interp1d'
    :return: y2
    '''

    f = interpolate.interp1d(t1,y1,kind=kind, bounds_error=bounds_error)

    y2 = f(t2)

    if isPlot:
        plt.figure()
        plt.plot(t1,y1)
        plt.plot(t2,y2)
        plt.legend('original','resampled')

    return y2


def arrayNor(A):
    '''
    normalize a np.array to the scale [0, 1]
    '''

    B=A.astype(np.float)
    B = (B-np.amin(B))/(np.amax(B)-np.amin(B))
    return B.astype(A.dtype)


def arrayNorMedian(A):
    '''
    normalize array by minus median, data type will be switch to np.float
    '''
    A=A.astype(np.float)
    B=A-np.median(A.flatten())
    return B


def  arrayNorMean(A):
    '''
    normalize array by minus mean, data type will be switch to np.float
    '''
    A=A.astype(np.float)
    B=A-np.mean(A.flatten())
    return B


def arrayNorMeanStd(A):
    '''
    normalize array by minus mean and then devided by standard deviation, data type will be switch to np.float
    '''
    A=A.astype(np.float)
    B=(A-np.mean(A.flatten()))/np.std(A.flatten())
    return B


def distance(p0, p1):
    '''
    calculate distance between two points, can be multi-dimensinal

    p0 and p1 should be a 1d array, with each element for each dimension
    '''


    #old code======================================================================
    # if (len(p0.shape) > 1) or (len(p1.shape) > 1):
    #     raise LookupError, 'Both input arrays should be 1d array!!'
    #
    # if p0.shape != p1.shape:
    #     raise LookupError, 'The two input arrays should have same dimensions.'
    #
    # distance = math.sqrt(np.sum(((p0.astype(np.float)-p1.astype(np.float))**2)))
    #===============================================================================

    if not isinstance(p0, np.ndarray):p0 = np.array(p0)
    if not isinstance(p1, np.ndarray):p1 = np.array(p1)
    return np.sqrt(np.mean(np.square(p0-p1).flatten()))


def arrayDiff(a0,a1):
    '''
    calculate the sum of pixel-wise difference between two arrays
    '''
    if not isinstance(a0, np.ndarray):a0 = np.array(a0)
    if not isinstance(a1, np.ndarray):a1 = np.array(a1)
    return np.mean(np.abs(a0-a1).flatten())


def binarize(array, threshold):
    '''
    binarize array to 0s and 1s, by cutting at threshold
    '''
    
    newArray = np.array(array)
    
    newArray[array>=threshold] = 1.
    
    newArray[array<threshold] = 0.
    
    newArray = newArray.astype(array.dtype)
    
    return newArray


def centerImage(img, # original image, 2d ndarray
                centerPixel, # the coordinates of center pixel in original image, [col, row]
                newSize = 512, #the size of output image
                borderValue = 0
                ):
    '''
    center a certain image in a new canvas

    the pixel defined by 'centerPixel' in the original image will be at the center of the output image
    the size of output image is defined by 'newSize'

    empty pixels will be filled with zeros
    '''

    x = newSize/2 - centerPixel[1]
    y = newSize/2 - centerPixel[0]

    M = np.float32([[1,0,x],[0,1,y]])

    newImg = cv2.warpAffine(img,M,(newSize,newSize),borderValue=borderValue)

    return newImg


def resizeImage(img, outputShape, fillValue = 0.):
    '''
    resize every frame of a 3-d matrix to defined output shape
    if the original image is too big it will be truncated
    if the original image is too small, value defined as fillValue will filled in. default: 0
    '''
    
    width = outputShape[1]
    height = outputShape[0]
    
    if width < 1:
        raise ValueError, 'width should be bigger than 0!!'
        
    if height < 1:
        raise ValueError, 'height should be bigger than 0!!'
        
    if len(img.shape) !=2 and len(img.shape) !=3 :
        raise ValueError, 'input image should be a 2-d or 3-d array!!'

    if len(img.shape) == 2: # 2-d image
        startWidth = img.shape[-1]
        startHeight = img.shape[-2]
        newImg = np.array(img)
        if startWidth > width:
            newImg = newImg[:,0:width]
        elif startWidth < width:
            attachRight = np.zeros((startHeight,width-startWidth))
            attachRight[:] = fillValue
            attachRight.astype(img.dtype)
            newImg = np.hstack((newImg,attachRight))

        if startHeight > height:
            newImg = newImg[0:height,:]
        elif startHeight < height:
            attachBottom = np.zeros((height - startHeight,width))
            attachBottom[:] = fillValue
            attachBottom.astype(img.dtype)
            newImg = np.vstack((newImg,attachBottom))

    if len(img.shape) == 3: # 3-d matrix
        startDepth = img.shape[0]
        startWidth = img.shape[-1]
        startHeight = img.shape[-2]
        newImg = np.array(img)
        if startWidth > width:
            newImg = newImg[:,:,0:width]
        elif startWidth < width:
            attachRight = np.zeros((startDepth,startHeight,width-startWidth))
            attachRight[:] = fillValue
            attachRight.astype(img.dtype)
            newImg = np.concatenate((img,attachRight),axis=2)

        if startHeight > height:
            newImg = newImg[:,0:height,:]
        elif startHeight < height:
            attachBottom = np.zeros((startDepth,height-startHeight,width))
            attachBottom[:] = fillValue
            attachBottom.astype(img.dtype)
            newImg = np.concatenate((newImg,attachBottom),axis=1)
        
    return newImg


def expandImage_cv2(img):

    if len(img.shape) != 2:
        raise ValueError, 'Input image should be 2d!'

    dtype = img.dtype
    img = img.astype(np.float32)
    rows,cols = img.shape
    diagonal = int(np.sqrt(rows**2+cols**2))
    M = np.float32([[1,0,(diagonal-cols)/2],[0,1,(diagonal-rows)/2]])
    newImg = cv2.warpAffine(img,M,(diagonal,diagonal))
    return newImg.astype(dtype)


def expandImage(img):

    if len(img.shape) == 2:
        rows,cols = img.shape
        diagonal = int(np.sqrt(rows**2+cols**2))
        top = np.zeros(((diagonal-rows)/2,cols),dtype=img.dtype)
        down = np.zeros((diagonal-img.shape[0]-top.shape[0],cols),dtype=img.dtype)
        tall = np.vstack((top,img,down))
        left = np.zeros((tall.shape[0],(diagonal-cols)/2),dtype=img.dtype)
        right = np.zeros((tall.shape[0],diagonal-img.shape[1]-left.shape[1]),dtype=img.dtype)
        newImg = np.hstack((left,tall,right))
        return newImg
    elif len(img.shape) == 3:
        frames,rows,cols = img.shape
        diagonal = int(np.sqrt(rows**2+cols**2))
        top = np.zeros((frames,(diagonal-rows)/2,cols),dtype=img.dtype)
        down = np.zeros((frames,diagonal-img.shape[1]-top.shape[1],cols),dtype=img.dtype)
        tall = np.concatenate((top,img,down),axis=1)
        left = np.zeros((frames,tall.shape[1],(diagonal-cols)/2),dtype=img.dtype)
        right = np.zeros((frames,tall.shape[1],diagonal-img.shape[2]-left.shape[2]),dtype=img.dtype)
        newImg = np.concatenate((left,tall,right),axis=2)
        return newImg
    else:
        raise ValueError, 'Input image should be 2d or 3d!'


def zoomImage(img,zoom,interpolation = cv2.INTER_CUBIC):
    '''
    zoom a 2d image. if zoom is a single value, it will apply to both axes, if zoom has two values it will be applied to
    height and width respectively
    zoom[0]: height
    zoom[1]: width
    '''
    if len(img.shape) != 2:
        raise ValueError, 'Input image should be 2d!'

    try: zoomH = float(zoom[0]); zoomW = float(zoom[1])
    except TypeError: zoomH = float(zoom); zoomW = float(zoom)

    newImg= cv2.resize(img.astype(np.float),dsize=(int(img.shape[1]*zoomW),int(img.shape[0]*zoomH)),interpolation=interpolation)
    return newImg


def moveImage(img,Xoffset,Yoffset,width,height,borderValue=0.0):
    '''
    move image defined by Xoffset and Yoffset

    new canvas size is defined by width and height

    empty pixels will be filled with zeros
    '''
    if len(img.shape) != 2:
        raise ValueError, 'Input image should be 2d!'

    M = np.float32([[1,0,Xoffset],[0,1,Yoffset]])

    newImg = cv2.warpAffine(img,M,(width,height),borderValue=borderValue)

    return newImg


def rotateImage(img,angle,borderValue=0.0):
    '''
    rotate an image conterclock wise by an angle defined by 'angle' in degree

    pixels go out side will be tropped
    pixels with no value will be filled as zeros
    '''

    if len(img.shape) != 2:
        raise ValueError, 'Input image should be 2d!'

    rows,cols = img.shape

    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    newImg = cv2.warpAffine(img,M,(cols,rows),borderValue=borderValue)

    return newImg


def rigidTransform(img, zoom=None, rotation=None, offset=None, outputShape=None, mode='constant',cval=0.0):

    '''
    rigid transformation of a 2d-image or 3d-matrix by using scipy
    :param img: input image/matrix
    :param zoom:
    :param rotation: in degree, counterclock wise
    :param offset: tuple (xoffset, yoffset) pixel value of starting point of output image
    :param outputShape: the shape of output image, (height, width)
    :return: new image or matrix after transformation
    '''

    if len(img.shape) != 2 and len(img.shape) != 3:
        raise LookupError, 'Input image is not a 2d or 3d array!'

    newImg = img.astype(np.float32)

    if zoom:
        if len(img.shape) == 2:
            newZoom = (zoom,zoom)
        elif len(img.shape) == 3:
            newZoom = (1,zoom,zoom)
        newImg = ni.zoom(newImg,zoom=newZoom,mode=mode,cval=cval)

    if rotation:
        newImg = expandImage(newImg)
        if len(img.shape) == 2:
            newImg = ni.rotate(newImg,angle=rotation,reshape=False,mode=mode,cval=cval)
        elif len(img.shape) == 3:
            newImg = ni.rotate(newImg,angle=rotation,axes=(1,2),reshape=False,mode=mode,cval=cval)

    if offset:
        if len(img.shape) == 2:
            newImg = ni.shift(newImg,(offset[1],offset[0]),mode=mode,cval=cval)
        if len(img.shape) == 3:
            newImg = ni.shift(newImg,(0,offset[1],offset[0]),mode=mode,cval=cval)

    if outputShape:
        newImg = resizeImage(newImg,outputShape)

    return newImg.astype(img.dtype)


def rigidTransform_cv2_2d(img, zoom=None, rotation=None, offset=None, outputShape=None):

    '''
    rigid transformation of a 2d-image by using opencv
    :param img: input image/matrix
    :param zoom:
    :param rotation: in degree, counterclock wise
    :param offset: tuple (xoffset, yoffset) pixel value of starting point of output image
    :param outputShape: the shape of output image, (height, width)
    :return: new image or matrix after transformation
    '''

    if len(img.shape) != 2:
        raise LookupError, 'Input image is not a 2d or 3d array!'

    newImg = np.array(img).astype(np.float)
    minValue = np.amin(newImg)

    if zoom:
        newImg = zoomImage(img,zoom=zoom)

    if rotation:
        newImg = expandImage_cv2(newImg)
        newImg = rotateImage(newImg, rotation,borderValue=minValue)

    if (outputShape is None) and (offset is None):
        return newImg
    else:
        if outputShape is None:
            outputShape = newImg.shape
        if offset is None:
            offset = (0,0)
        newImg = moveImage(newImg, offset[0], offset[1], outputShape[1],outputShape[0],borderValue=minValue)

        return newImg.astype(img.dtype)


def rigidTransform_cv2_3d(img, zoom=None, rotation=None, offset=None, outputShape=None):
    
    if len(img.shape) != 3:
        raise LookupError, 'Input image is not a 3d array!'

    if not outputShape:
        if zoom:
            newHeight = int(img.shape[1]*zoom)
            newWidth = int(img.shape[2]*zoom)
        else:
            newHeight = img.shape[1]
            newWidth = img.shape[2]
    else:
        newHeight = outputShape[0]
        newWidth = outputShape[1]
    newImg = np.empty((img.shape[0],newHeight,newWidth),dtype=img.dtype)
    
    for i in range(img.shape[0]):
        newImg[i,:,:] = rigidTransform_cv2_2d(img[i,:,:], zoom=zoom, rotation=rotation, offset=offset, outputShape=outputShape)
    
    return newImg


def rigidTransform_cv2(img, zoom=None, rotation=None, offset=None, outputShape=None):

    '''
    rigid transformation of a 2d-image or 3d-matrix by using opencv
    :param img: input image/matrix
    :param zoom:
    :param rotation: in degree, counterclock wise
    :param offset: tuple (xoffset, yoffset) pixel value of starting point of output image
    :param outputShape: the shape of output image, (height, width)
    :return: new image or matrix after transformation
    '''

    if len(img.shape) == 2:
        return rigidTransform_cv2_2d(img, zoom=zoom, rotation=rotation, offset=offset, outputShape=outputShape)
    elif len(img.shape) == 3:
        return rigidTransform_cv2_3d(img, zoom=zoom, rotation=rotation, offset=offset, outputShape=outputShape)
    else:
        raise ValueError, 'Input image is not a 2d or 3d array!'


def boxcartime_dff(data,
                   window,# boxcar size in seconds
                   fs # sample rate in ms
                   ):
    """
    Created on Mon Nov 24 14:37:02 2014
    
    [dff] = boxcartime_dff(data[t,y,x], rollingwindow[in s], samplerate[in ms])
    boxcar average uses scipy.signal package, imported locally
    
    @author: mattv
    """
    import scipy.signal as sig
    
    if data.ndim != 3:
        raise LookupError, 'input images must be a 3-dim array format [t,y,x]'   

    exposure = np.float(fs/1000) #convert exposure from ms to s
    win = np.float(window)
    win = win/exposure
    win = np.ceil(win)
    
    # rolling average
    kernal = np.ones(win, dtype=('float')) 
    padsize = data.shape[0] + win*2
    mov_ave = np.zeros([padsize+win-1], dtype=('float'))
    mov_pad = np.zeros([padsize], dtype=('float'))
    mov_dff = np.zeros([data.shape[0]-win, data.shape[1], data.shape[2]])
    for y in range(data.shape[1]):
        for x in range(data.shape[2]):      
            # put data within padded array
            mov_pad[win:(padsize-win)] = data[:,y,x]
            # moving average by convolution
            mov_ave = sig.fftconvolve(mov_pad, kernal)/win
            # cut off pad
            mov_ave = mov_ave[win*2:1+mov_ave.shape[0]-win*2]
            # use moving average as f0 for df/f
            mov_dff[:,y,x] = (data[(win/2):data.shape[0]-(win/2),y,x] - mov_ave)/mov_ave
            
    return mov_dff


def normalizeMovie(movie,
                   baselinePic = None, # picture for baseline
                   baselineType = 'mean' # 'mean' or 'median'
                   ):
    '''
    return average image, movie minus avearage, and dF over F for each pixel
    '''
    movie = np.array(movie, dtype = np.float32)

    if baselinePic is not None:

      if movie.shape[1:] != baselinePic.shape:
          raise LookupError, 'The shape of "baselinePic" should match the shape of the frame shape of "movie"!'

      averageImage = baselinePic

    elif baselineType == 'mean':
        averageImage = np.mean(movie, axis = 0)

    elif baselineType == 'median':
        averageImage = np.median(movie, axis = 0)

    else:
        raise LookupError, 'The "baselineType" should be "mean" or "median"!!'

    normalizedMovie = np.subtract(movie,averageImage)
    dFoverFMovie = np.divide(normalizedMovie,averageImage)

    return averageImage, normalizedMovie, dFoverFMovie


def temporalFilterMovie(mov, # array of movie
                        Fs, # sampling rate
                        Flow, # low cutoff frequency
                        Fhigh, # high cutoff frequency
                        mode = 'box'): # filter mode, '1/f' or 'box'):

    if len(mov.shape) != 3:
        raise LookupError, 'The "mov" array should have 3 dimensions!'

    frameNum = mov.shape[0]
    freqs = np.fft.fftfreq(frameNum, d = (1./float(Fs)))

    filterArray = np.ones(frameNum)

    for i in xrange(frameNum):
        if ((freqs[i] > 0) and (freqs[i] < Flow) or (freqs[i] > Fhigh)) or \
           ((freqs[i] < 0) and (freqs[i] > -Flow) or (freqs[i] < -Fhigh)):
            filterArray[i] = 0

    if mode == '1/f':
        filterArray[1:] = filterArray[1:] / abs(freqs[1:])
        filterArray[0] = 0
        filterArray = (filterArray - np.amin(filterArray)) / (np.amax(filterArray) - np.amin(filterArray))
    elif mode == 'box':
        filterArray[0] = 0
    else: raise NameError, 'Variable "mode" should be either "1/f" or "box"!'

    if Flow == 0:
        filterArray[0] = 1

    movFFT = np.fft.fft(mov, axis = 0)

    for i in xrange(mov.shape[1]):
        for j in xrange(mov.shape[2]):
            movFFT[:,i,j] = movFFT[:,i,j] * filterArray

    movF = np.real(np.fft.ifft(movFFT, axis = 0))

    return movF


def generateRectangleMask(movie,center,width,height,isplot = False):

    mask = np.zeros((np.size(movie, -2), np.size(movie, -1)))

    mask[:] = np.nan

    mask[int(round(center[0]-height/2)):int(round(center[0]+height/2)),int(round(center[1]-width/2)):int(round(center[1]+width/2))] = 1

    if np.isnan(np.nansum(mask[:])):
        raise ArithmeticError, 'No element in mask!'

    if isplot == True:
        if len(movie.shape)==3:
            aveMovie = np.mean(movie,axis=0)
        else:
            aveMovie = movie

        f = plt.figure()
        f1 = f.add_subplot(111)
        f1.imshow(aveMovie, cmap = 'gray', interpolation = 'nearest')
        _ = plotMask(mask, plotAxis=f1, color='#ff0000')

    return mask


def generateOvalMask(movie,center,width,height,isplot = False):

    mask = np.zeros((np.size(movie, -2), np.size(movie, -1)))

    mask[:] = np.nan
    
    width = float(width)
    height = float(height)

    for i in range(movie.shape[-2]):
        for j in range(movie.shape[-1]):
            if ((i-center[0])/(height/2))**2 + ((j-center[1])/(width/2))**2 <= 1:
                mask[i,j]=1

    if np.isnan(np.nansum(mask[:])):
        raise ArithmeticError, 'No element in mask!'

    if isplot == True:
        if len(movie.shape)==3:
            aveMovie = np.mean(movie,axis=0)
        else:
            aveMovie = movie

        f = plt.figure()
        f1 = f.add_subplot(111)
        f1.imshow(aveMovie, cmap = 'gray', interpolation = 'nearest')
        _ = plotMask(mask, plotAxis=f1, color='#ff0000')

    return mask


def getTrace(movie, mask):
    '''
    get a trace across a movie with averaged value in a mask
    '''

    pixelInMask = np.nansum(mask)

    trace = [np.nansum(np.multiply(frame, mask)) for frame in movie]

    trace = trace / pixelInMask

    return np.array(trace)


def getTrace2(movie,center,width,height,maskType = 'rect',isplot = False):

    if maskType == 'rect':
        mask = generateRectangleMask(movie,center,width,height)
    elif maskType == 'oval':
        mask = generateOvalMask(movie,center,width,height)
    else:
        raise TypeError, 'maskType should be "rect" or "oval"!'

    trace = getTrace(movie, mask)

    if isplot == True:
        #plt.figure()
        plt.plot(trace)

    return trace


def hitOrMiss(cor, mask):
    mask = mask.astype(np.int8)
    corMask = np.zeros(mask.shape, dtype = np.int8)
    corMask[np.round(cor[0]),np.round(cor[1])] = 1
    if np.sum(np.multiply(corMask, mask)) > 0:
        return True
    if np.sum(np.multiply(corMask, mask)) == 0:
        return False


def harAmp(f, # function value
           period, # how many fundamental harmonic periods inside the function
           n): # return the n-th harmonic
    '''
    calculate the amplitude and phase of the n-th harmonic components of a
    function. the input function should have whole number times of period of
    the fundamental harmonic.
    '''

    if (type(period) != int) | (period <= 0):
        raise ArithmeticError, '"period" should be a positive integer!'

    if (type(n) != int) | (n < 0):
        raise ArithmeticError, '"n" should be a non-negative positive integer!'

    L = len(f)
    x = np.arange(L)

    if n == 0:
        har = np.sum(np.multiply(f, np.exp(-1j*2*np.pi*period*n*x/L)))/L
        har_meg = np.abs(har)
        har_phase = np.nan

    else:
        har = 2 * np.sum(np.multiply(f, np.exp(-1j*2*np.pi*period*n*x/L)))/L
        har_meg = np.abs(har)
        har_phase = np.angle(har)

    return har_meg, har_phase


def discretize(array, binSize):
    '''
    discretize the array by binSize
    '''

    bins = np.arange(np.floor(np.nanmin(array)) - (0.5 * binSize),
                     np.ceil(np.nanmax(array)) + (1.5 * binSize),
                     binSize)

    flatArray = np.ndarray.flatten(array)

    indArray = np.digitize(flatArray, bins)

    newArray = np.zeros(flatArray.shape)
    newArray[:] = np.nan

    for i in xrange(len(indArray)):
        if np.isnan(flatArray[i]) == False:
            newArray[i] = bins[indArray[i]]

    newArray = np.array(newArray).reshape(array.shape)

    return newArray


def seedPixel(markers):
    '''
    marker centroid of every marked local minimum
    '''


    newMarkers = np.zeros(markers.shape).astype(np.int32)
    intMarkers = markers.astype(np.int32)

    for i in range(1,np.amax(intMarkers)+1):
        aa = np.zeros(markers.shape).astype(np.int32)
        aa[intMarkers == i] = 1
        aapixels = np.argwhere(aa)
        center = np.mean(aapixels.astype(np.float32), axis = 0)

        newMarkers[int(np.round(center[0])),int(np.round(center[1]))] = i

    return newMarkers


def isAdjacent(array1, array2, borderWidth = 2):
    '''
    decide if two patches are adjacent within border width
    '''

    p1d = ni.binary_dilation(array1, iterations = borderWidth-1).astype(np.int8)
    p2d = ni.binary_dilation(array2, iterations = borderWidth-1).astype(np.int8)

    if np.amax(p1d + p2d) > 1:
        return True
    else:
        return False


def plotMask(mask,plotAxis=None,color='#ff0000',zoom=1,borderWidth = None,closingIteration = None):
    '''
    plot mask borders in a given color
    '''

    if not plotAxis:
        f = plt.figure()
        plotAxis = f.add_subplot(111)

    cmap1 = col.ListedColormap(color, 'temp')
    cm.register_cmap(cmap=cmap1)

    if zoom != 1:
        mask = ni.interpolation.zoom(mask,zoom,order=0)

    mask2 = mask.astype(np.float32)
    mask2[np.invert(np.isnan(mask2))]= 1.
    mask2[np.isnan(mask2)] = 0.

    struc = ni.generate_binary_structure(2, 2)
    if borderWidth:
        border=mask2 - ni.binary_erosion(mask2,struc,iterations=borderWidth).astype(np.float32)
    else:
        border=mask2 - ni.binary_erosion(mask2,struc).astype(np.float32)

    if closingIteration:
        border = ni.binary_closing(border,iterations=closingIteration).astype(np.float32)

    border[border==0] = np.nan

    currfig = plotAxis.imshow(border, cmap = 'temp', interpolation='nearest')

    return currfig


def removeSmallPatches(mask,areaThr=100,structure=[[1,1,1],[1,1,1],[1,1,1]]):
    '''
    remove small isolated patches
    '''

    if mask.dtype == np.bool:pass
    elif issubclass(mask.dtype.type, np.integer):
        if np.amin(mask)<0 or np.amax(mask)>1:raise ValueError, 'Values of input image should be either 0 or 1.'
    else: raise TypeError, 'Data type of input image should be either np.bool or integer.'

    patches, n = ni.label(mask,structure)
    newMask = np.zeros(mask.shape,dtype=np.uint8)

    if n==0: return newMask
    else:
        for i in range(1,n+1):
            currPatch = np.zeros(mask.shape,dtype=np.uint8)
            currPatch[patches==i]=1
            if np.sum(currPatch.flatten())>=areaThr:newMask += currPatch

    return newMask.astype(np.bool)


def getAreaEdges(img,
                 firstGaussianSigma=50.,
                 medianFilterWidth=100.,
                 areaThr=(0.1,0.9),
                 edgeThrRange=(5,16),
                 secondGaussianSigma=10.,
                 thr=0.2,
                 borderWidth=2,
                 lengthThr=20,
                 isPlot=True):
    '''
    get binary edge of areas
    '''

    img=img.astype(np.float)
    imgFlat = img - ni.filters.gaussian_filter(img,firstGaussianSigma)
    imgMedianFiltered = arrayNor(ni.filters.median_filter(imgFlat,medianFilterWidth))
    imgPatch=np.array(imgMedianFiltered)
    imgPatch[imgMedianFiltered<areaThr[0]]=areaThr[0];imgPatch[imgMedianFiltered>areaThr[1]]=areaThr[1]
    imgPatch=(arrayNor(imgPatch)*255).astype(np.uint8)


    # plt.imshow(imgPatch,vmin=0,vmax=255,cmap='gray')
    # plt.show()

#    import tifffile as tf
#    tf.imsave('Rorb_example_vasMap_filtered.tif',arrayNor(imgPatch.astype(np.float32)))


    cuttingStep = np.arange(edgeThrRange[0],edgeThrRange[1])
    cuttingStep = np.array([cuttingStep[0:-1],cuttingStep[1:]]).transpose()
    edge_cv2 = np.zeros(img.shape).astype(np.uint8)
    for i in range(cuttingStep.shape[0]):
        currEdge = cv2.Canny(imgPatch,cuttingStep[i,0],cuttingStep[i,1])/255
        edge_cv2 += currEdge
        if isPlot:
            if i==1: firstEdgeSet=currEdge
            if i==cuttingStep.shape[0]-1: lastEdgeSet=currEdge

    edgesF=ni.filters.gaussian_filter(edge_cv2.astype(np.float),secondGaussianSigma)

    edgesThr = np.zeros(edgesF.shape).astype(np.uint8)
    edgesThr[edgesF<thr]=0;edgesThr[edgesF>=thr]=1

    edgesThin = sm.skeletonize(edgesThr)

    edgesThin = removeSmallPatches(edgesThin,lengthThr)
    if borderWidth>1: edgesThick=ni.binary_dilation(edgesThin,iterations=borderWidth-1)
    else: edgesThick=edgesThin

    if isPlot:
        displayEdges = np.zeros((edgesThick.shape[0],edgesThick.shape[1],4)).astype(np.uint8)
        displayEdges[edgesThick==1]=np.array([255,0,0,255]).astype(np.uint8)
        displayEdges[edgesThick==0]=np.array([0,0,0,0]).astype(np.uint8)

        f,ax=plt.subplots(2,5,figsize=(15,5))
        ax[0,0].imshow(img,cmap='gray');ax[0,0].set_title('original image');ax[0,0].axis('off')
        ax[0,1].imshow(imgFlat,cmap='gray');ax[0,1].set_title('flattened image');ax[0,1].axis('off')
        ax[0,2].imshow(imgPatch,cmap='gray');ax[0,2].set_title('image for edge detection');ax[0,2].axis('off')
        ax[0,3].imshow(firstEdgeSet,cmap='gray');ax[0,3].set_title('first edge set');ax[0,3].axis('off')
        ax[0,4].imshow(lastEdgeSet,cmap='gray');ax[0,4].set_title('last edge set');ax[0,4].axis('off')
        ax[1,0].imshow(edgesF,cmap='hot');ax[1,0].set_title('filtered edges sum');ax[1,0].axis('off')
        ax[1,1].imshow(edgesThr,cmap='gray');ax[1,1].set_title('binary thresholded edges');ax[1,1].axis('off')
        ax[1,2].imshow(edgesThick,cmap='gray');ax[1,2].set_title('binary edges');ax[1,2].axis('off')
        ax[1,3].imshow(img,cmap='gray');ax[1,3].imshow(displayEdges);ax[1,3].set_title('original image with edges');ax[1,3].axis('off')
        ax[1,4].imshow(imgPatch,cmap='gray');ax[1,4].imshow(displayEdges);ax[1,4].set_title('blurred image with edges');ax[1,4].axis('off')
        plt.tight_layout()
        return edgesThick.astype(np.bool), f

    else: return edgesThick.astype(np.bool)


def zDownsample(img,downSampleRate):
    '''
    downsample input image in z direction
    '''

    if len(img.shape) != 3:
        raise ValueError, 'Input array shoud be 3D!'


    newFrameNum = (img.shape[0] - (img.shape[0]%downSampleRate))/downSampleRate
    newImg = np.empty((newFrameNum,img.shape[1],img.shape[2]),dtype=img.dtype)

    print 'Start downsampling...'
    for i in range(newFrameNum):
#            print (float(i)*100/newFrameNum),'%'
        currChunk = img[i*downSampleRate:(i+1)*downSampleRate,:,:].astype(np.float)
        currFrame = np.mean(currChunk,axis=0)
        newImg[i,:,:]=currFrame.astype(img.dtype)
    print 'End of downsampling.'
    return newImg


def getMasks(labeled,minArea=None,maxArea=None,isSort=True,keyPrefix = None,labelLength=3):
    '''
    get mask dictionary from labeled maps (labeled by scipy.ndimage.label function)

    area range of each mask was defined by minArea and maxArea

    isSort: if True, sort masks by areas, big to small

    keyPrefix: the prefix of key

    labelLength: the number of characters of key
    '''

    maskNum = np.max(labeled.flatten())
    masks = {}
    for i in range(1,maskNum+1):
        currMask = np.zeros(labeled.shape,dtype=np.uint8)
        currMask[labeled==i]=1

        if minArea is not None and np.sum(currMask.flatten()) < minArea: pass
        elif maxArea is not None and np.sum(currMask.flatten()) > maxArea: pass
        else:
            if keyPrefix is not None: currKey = keyPrefix+'.'+ft.int2str(i,labelLength)
            else: currKey = ft.int2str(i,labelLength)
            masks.update({currKey:currMask})

    if isSort:
        masks = sortMasks(masks,keyPrefix = keyPrefix, labelLength=labelLength)

    return masks


def sortMasks(masks,keyPrefix='',labelLength=3):
    '''
    sort a dictionary of binary masks, big to small
    '''

    maskNum = len(masks.keys())
    order = []
    for key, mask in masks.iteritems():
        order.append([key,np.sum(mask.flatten())])

    order = sorted(order, key=lambda a:a[1], reverse=True)

    newMasks = {}
    for i in range(len(order)):
        if keyPrefix is not None: currKey = keyPrefix+'.'+ft.int2str(i,labelLength)
        else: currKey = ft.int2str(i,labelLength)
        newMasks.update({currKey:masks[order[i][0]]})
    return newMasks


def tempDownSample(A, rate, verbose=False):
    '''
    down sample a 3-d array in 0 direction
    '''

    if len(A.shape) != 3: raise ValueError, 'input array should be 3-d.'
    rate = int(rate)
    dataType = A.dtype
    newZDepth = (A.shape[0] - (A.shape[0]%rate))/rate
    newA = np.empty((newZDepth,A.shape[1],A.shape[2]),dtype=dataType)

    for i in range(newZDepth):
        if verbose:
            print (float(i)*100/newZDepth),'%'
            currChunk = A[i*3:(i+1)*3,:,:].astype(np.float)
            currFrame = np.mean(currChunk,axis=0)
            newA[i,:,:]=currFrame.astype(dataType)
    return newA


if __name__ == '__main__':

    #============================================================
    # a = np.random.rand(100,100)
    # mask = generateOvalMask(a,[45,58],20,30,isplot=True)
    # plt.show()

    #============================================================
    # a = np.arange(400).reshape((20,20))
    # b = rigidTransform(a,2,30,(1,5),(30,25))
    # f,ax=plt.subplots(1,2)
    # ax[0].imshow(a,interpolation='nearest')
    # ax[1].imshow(b,interpolation='nearest')
    # plt.show()
    #============================================================

    #============================================================
    # import tifffile as tf
    # imgPath = r"E:\data2\2015-05-28-Average-Ai93-Rorb-Scnn1a-map\AverageVasMap_Ai93.tif"
    # img = tf.imread(imgPath)
    # edges = getAreaEdges(img)
    # plt.show()
    #============================================================

    #============================================================
    # aa=np.zeros((15,15),dtype=np.uint8)
    # aa[4,5]=1
    # aa[5,6]=1
    # aa[12:15,8:13]=1
    # bb=removeSmallPatches(aa,5)
    # f,ax=plt.subplots(1,2)
    # ax[0].imshow(aa,interpolation='nearest');ax[1].imshow(bb,interpolation='nearest')
    # plt.show()
    #============================================================

    #============================================================
    a=5; b=7
    print distance(a,b)

    c=[5,6]; d=[8,2]
    print distance(c,d)

    e=np.random.rand(5,6); f=np.random.rand(5,6)
    print distance(e,f)
    #============================================================

    print 'for debug'





