__author__ = 'junz'


import numpy as np
import core.tifffile as tf
import core.ImageAnalysis as ia
import core.FileTools as ft
import matplotlib.pyplot as plt


plt.ioff()

def iamstupid(imgMat,imgRef,maxDisplacement=10,normFunc=ia.arrayDiff):
    '''

    align two images with rigid transformation

    :param imgRef: reference image
    :param imgMat: matching image
    :param maxDisplacement: maximum displacement, if single value, it will be apply to both rows and columns. if two
    values, it will be [rowMaxDisplacement, columnMaxDisplacement]
    :param normFunc: the function to calculate the distance between two images
    :return:
    : offSet: final offSet
    : hitLimitFlag: the Flag to mark if the maxDisplacement limit was hit for row and column
    '''

    try:
        rowMaxDisplacement = int(abs(maxDisplacement[0]))
        columnMaxDisplacement = int(abs(maxDisplacement[1]))
    except TypeError: rowMaxDisplacement = columnMaxDisplacement = int(abs(maxDisplacement))

    prevDis = normFunc(imgRef,imgMat)
    prevOffset = [0, 0]
    hitLimitFlag = [0, 0]
    tryList = [[-1, 0], [0, -1], [1, 0], [0, 1]]
    currDisList = np.array([normFunc(imgRef, ia.rigidTransform(imgMat,offset=o,outputShape=imgRef.shape)) for o in tryList])
    currDis = np.min(currDisList)
    currOffset = tryList[np.where(currDisList == currDis)[0]]

    while currDis < prevDis:
        prevDis = currDis; prevOffset = currOffset

        tryList = []
        if abs(prevOffset[0]) != rowMaxDisplacement:
            if prevOffset[0] < 0: tryList.append([prevOffset[0]-1, prevOffset[1]])
            elif prevOffset[0] > 0: tryList.append([prevOffset[0]+1, prevOffset[1]])
            else: tryList += [[1, prevOffset[1]],[-1, prevOffset[1]]]
        else: hitLimitFlag[0] = 1

        if abs(prevOffset[1]) != columnMaxDisplacement:
            if prevOffset[1] < 0: tryList.append([prevOffset[0], prevOffset[1]-1])
            elif prevOffset[1] > 0: tryList.append([prevOffset[0], prevOffset[1]+1])
            else: tryList += [[prevOffset[0], 1],[prevOffset[0], -1]]
        else: hitLimitFlag[1] = 1

        if len(tryList)>0:
            currDisList = np.array([normFunc(imgRef, ia.rigidTransform(imgMat,offset=o,outputShape=imgRef.shape)) for o in tryList])
            currDis = np.min(currDisList)
            currOffset = tryList[np.where(currDisList == currDis)[0]]
        else:break

    return np.array(prevOffset,dtype=np.int), hitLimitFlag


def getDistanceList(img, imgRef, normFunc=ia.arrayDiff, isPlot = False):
    '''
    get the list of distances from each frame in img to the reference image, imgRef
    normFunc is the function to calculate distance between two frames
    '''

    distanceList = np.zeros(img.shape[0])
    for i in range(img.shape[0]):
        distanceList[i] = normFunc(img[i,:,:], imgRef)
    if isPlot:
        f = plt.figure(figsize=(15,8))
        ax1 = f.add_subplot(211); ax1.plot(distanceList), ax1.set_ylim([0,200]);ax1.set_title('distance from mean for each frame')
        ax2 = f.add_subplot(212); _ = ax2.hist(distanceList, bins=50, range=(40,200)); ax2.set_title('distribution of distances')
        plt.show()
    return distanceList


def alignSingleMovie(mov, imgRef, badFramdDistanceThr=100, maxDisplacement=10, normFunc=ia.arrayDiff):
    '''
    align the frames in a single movie to the imgRef

    the frame with distance from mean projection larger than badFramdDistanceThr will not be used to update current
    offset, nor will be included in calculation of new mean projection

    return: offsetList, alignedMov, meanFrame
    '''

    dataType = mov.dtype
    currOffset = np.array([0,0]).astype(np.int)
    offsetList = []
    alignedMov = np.empty(mov.shape,dtype=dataType)
    validFrameNum = []

    for i in range(mov.shape[0]):

        if normFunc(mov[i,:,:],imgRef)<=badFramdDistanceThr:
            initCurrFrame = ia.rigidTransform_cv2(mov[i,:,:],offset=currOffset,outputShape=imgRef.shape)
            additionalOffset, hitFlag = iamstupid(initCurrFrame,imgRef,maxDisplacement=maxDisplacement,normFunc=normFunc)
            currOffset = currOffset+additionalOffset
            alignedMov[i,:,:] = ia.rigidTransform_cv2(mov[i,:,:],offset=currOffset,outputShape=imgRef.shape)
            offsetList.append(currOffset)
            validFrameNum.append(i)
            print 'Frame'+ft.int2str(i,5)+'\tgood Frame'+'\tOffset:'+str(currOffset)
        else:
            alignedMov[i,:,:] = ia.rigidTransform_cv2(mov[i,:,:],offset=currOffset,outputShape=imgRef.shape)
            offsetList.append(currOffset)
            print 'Frame'+ft.int2str(i,5)+'\tbad  Frame'+'\tOffset:'+str(currOffset)

    meanFrame = np.mean(alignedMov[np.array(validFrameNum),:,:],axis=0)
    return offsetList, alignedMov, meanFrame











if __name__=='__main__':

    #======================================================================================================
    # img_orig = tf.imread(r"C:\JunZhuang\labwork\data\python_temp_folder\motion_correction\original.tif")
    # img_move = tf.imread(r"C:\JunZhuang\labwork\data\python_temp_folder\motion_correction\moved.tif")
    #
    # offset, _ = iamstupid(img_orig,img_move)
    # assert(offset == [-4,-7])
    #======================================================================================================


    #======================================================================================================
    # imgPath = r'Z:\Jun\150610-M160809\KSStim_B2U_10Sweeps\KSStim_B2U_10sweeps_001_001.tif'
    imgPath = r"E:\data2\2015-06-11-python-2P-analysis-test\motion_correction_test\for_Jun\test_001.tif"
    img = tf.imread(imgPath)
    distanceList = getDistanceList(img,img[0,:,:],isPlot=True)
    #======================================================================================================

    #======================================================================================================
    # imgPath = r"E:\data2\2015-06-11-python-2P-analysis-test\motion_correction_test\for_Jun\test_001.tif"
    # img = tf.imread(imgPath)
    # offset, _ = iamstupid(img[1,:,:],img[2,:,:])
    # print offset
    #======================================================================================================

    #======================================================================================================
    # imgPath = r"E:\data2\2015-06-11-python-2P-analysis-test\motion_correction_test\for_Jun\test_001.tif"
    # img = tf.imread(imgPath)
    # offsetList, alignedMov, meanFrame = alignSingleMovie(img,img[0,:,:],badFramdDistanceThr=110)
    #
    # img2Path = r"E:\data2\2015-06-11-python-2P-analysis-test\motion_correction_test\for_Jun\test_002.tif"
    # img2 = tf.imread(img2Path)
    # offsetList2, alignedMov2, meanFrame2 = alignSingleMovie(img2,img2[0,:,:],badFramdDistanceThr=110)
    #
    # print np.hstack((offsetList,offsetList2))
    #======================================================================================================

    print 'for debug...'