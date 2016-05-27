__author__ = 'junz'

import os
import numpy as np
import tifffile as tf
import core.ImageAnalysis as ia
import core.FileTools as ft
import matplotlib.pyplot as plt


try: import cv2; from core.ImageAnalysis import rigid_transform_cv2 as rigid_transform
except ImportError as e: print e; from core.ImageAnalysis import rigid_transform as rigid_transform


plt.ioff()

def iamstupid(imgMat, imgRef, maxDisplacement=10, normFunc=ia.array_diff):
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

    #temporary code
    # imgMat[imgMat<150]=0
    # imgRef[imgRef<150]=0
    # imgMat[imgMat>400]=400
    # imgRef[imgRef>400]=400

    prevDis = normFunc(imgRef,imgMat)
    prevOffset = [0, 0]
    hitLimitFlag = [0, 0]
    tryList = [[-1, 0], [0, -1], [1, 0], [0, 1]]
    currDisList = np.array([normFunc(imgRef, rigid_transform(imgMat, offset=o, outputShape=imgRef.shape)) for o in tryList])
    currDis = np.min(currDisList)
    minInd = np.where(currDisList == currDis)[0]
    if len(minInd)>0: minInd = minInd[0]
    currOffset = tryList[minInd]

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
            currDisList = np.array([normFunc(imgRef, rigid_transform(imgMat, offset=o, outputShape=imgRef.shape)) for o in tryList])
            currDis = np.min(currDisList)
            currOffset = tryList[np.where(currDisList == currDis)[0]]
        else:break

    return np.array(prevOffset,dtype=np.int), hitLimitFlag


def getDistanceList(img, imgRef, normFunc=ia.array_diff, isPlot = False):
    '''
    get the list of distances from each frame in img to the reference image, imgRef
    normFunc is the function to calculate distance between two frames
    '''

    distanceList = np.zeros(img.shape[0])
    for i in range(img.shape[0]):
        distanceList[i] = normFunc(img[i,:,:], imgRef)
    if isPlot:
        disMedian = np.median(distanceList)
        disStd = np.std(distanceList)
        f = plt.figure(figsize=(15,8))
        ax1 = f.add_subplot(211); ax1.plot(distanceList), ax1.set_ylim([0,200]);ax1.set_title('distance from mean for each frame')
        ax2 = f.add_subplot(212); _ = ax2.hist(distanceList, bins=50, range=(disMedian-3*disStd,disMedian+3*disStd)); ax2.set_title('distribution of distances')
        return distanceList, f
    else: return distanceList


def alignSingleMovie(mov, imgRef, badFrameDistanceThr=100, maxDisplacement=10, normFunc=ia.array_diff, verbose=False, alignOrder=1):
    '''
    align the frames in a single movie to the imgRef

    the frame with distance from mean projection larger than badFramdDistanceThr will not be used to update current
    offset, nor will be included in calculation of new mean projection

    if order is 1: alignment goes from the first frame to the last
    if order is -1: alignment goes from the last frame to the first, this is faster in the alignSingleMovieLoop function

    return: offsetList, alignedMov, meanFrame
    '''

    dataType = mov.dtype
    currOffset = np.array([0,0]).astype(np.int)
    offsetList = []
    alignedMov = np.empty(mov.shape,dtype=dataType)
    validFrameNum = []

    if alignOrder == 1: iterFrames = range(mov.shape[0])
    if alignOrder == -1: iterFrames = range(mov.shape[0])[::-1]

    for i in iterFrames:
        if normFunc(mov[i,:,:],imgRef)<=badFrameDistanceThr:
            if np.array_equal(currOffset,np.array([0,0])):initCurrFrame = mov[i,:,:]
            else: initCurrFrame = rigid_transform(mov[i, :, :], offset=currOffset, outputShape=imgRef.shape)
            additionalOffset, hitFlag = iamstupid(initCurrFrame,imgRef,maxDisplacement=maxDisplacement,normFunc=normFunc)
            currOffset = currOffset+additionalOffset
            alignedMov[i,:,:] = rigid_transform(mov[i, :, :], offset=currOffset, outputShape=imgRef.shape)
            offsetList.append(currOffset)
            validFrameNum.append(i)
            if verbose:
                print 'Frame'+ft.int2str(i,5)+'\tdistance:'+str(normFunc(mov[i,:,:],imgRef))+'\tgood Frame'+'\tOffset:'+str(currOffset)
        else:
            alignedMov[i,:,:] = rigid_transform(mov[i, :, :], offset=currOffset, outputShape=imgRef.shape)
            offsetList.append(currOffset)
            if verbose:
                print 'Frame'+ft.int2str(i,5)+'\tdistance:'+str(normFunc(mov[i,:,:],imgRef))+'\tbad  Frame'+'\tOffset:'+str(currOffset)

    meanFrame = np.mean(alignedMov[np.array(validFrameNum),:,:],axis=0)
    if alignOrder == -1: offsetList = offsetList[::-1]
    return offsetList, alignedMov, meanFrame


def alignSingleMovieLoop(mov, iterations=2, badFrameDistanceThr=100, maxDisplacement=10, normFunc=ia.array_diff, verbose=False):
    '''
    align a single movie with iterations, every time it will use mean frame from last iteration as imgRef

    For every iteration it calls MotionCorrection.alignSingleMovie function

    the imgRef for first iteration is the last frame of the movie
    '''

    if iterations < 1: raise ValueError, 'Iterations should be an integer larger than 0!'

    else:
        offsetList, alignedMov, meanFrame = alignSingleMovie(mov,mov[-1,:,:],badFrameDistanceThr=badFrameDistanceThr,maxDisplacement=maxDisplacement,normFunc=normFunc,verbose=verbose,alignOrder=-1)

        if iterations == 1:
            return offsetList, alignedMov, meanFrame
        else:
            allOffsetList = np.array(offsetList)
            for i in range(iterations-1):
                offsetList, alignedMov, meanFrame = alignSingleMovie(alignedMov,meanFrame,badFrameDistanceThr=badFrameDistanceThr,maxDisplacement=maxDisplacement,normFunc=normFunc,verbose=verbose)
                allOffsetList += offsetList
            return allOffsetList, alignedMov,meanFrame


def alignMultipleTiffs(paths,
                       iterations=2,
                       badFrameDistanceThr=100,
                       maxDisplacement=10,
                       normFunc=ia.array_diff,
                       verbose=True,
                       output=False,
                       saveFolder=None,
                       fileNameSurfix='corrected',
                       cameraBias=0):
    '''
    motion correction of mulitiple tif file by using rigid plane transformation. Motion correction will be applied both
    within and across tif files

    paths: paths of input tif files
    iterations: number of iterations to perform motion correction8
    badFrameDistanceThr: the threshold of distance to define a good or bad frame, if a frame has distance from reference
                         framebigger than this value, it will be defined as bad frame, it will not be included in mean
                         frame calculation
    normFunc: function to calculate distance between two frames.
              options: corticalmapping.core.ImageAnalysis.array_diff (mean of absolute difference across all pixels)
                       corticalmapping.core.ImageAnalysis.distance (Frobenius distance or Euclidean norm)

    verbose: if True, print alignment information for each frame
    output: if True, generate and save motion corrected tif files
    saveFolder: if None, corrected files will be saved in the same folder of original data
    fileNameSurfix: surfix of corrected file names
    '''

    if saveFolder is not None:
        fileNameList = [os.path.split(p)[1] for p in paths]
        if len(set(fileNameList))<len(fileNameList):
            raise ValueError, 'If a save folder is declared, file names in paths should be unique!'

    offsets = []
    meanFrames = []
    for path in paths:
        print '\nStart alignment of file:', path,'...'
        currMov = tf.imread(path)
        currOffset, _, currMeanFrame = alignSingleMovieLoop(currMov,iterations=iterations,badFrameDistanceThr=badFrameDistanceThr,maxDisplacement=maxDisplacement,normFunc=normFunc,verbose=verbose)
        offsets.append(currOffset)
        meanFrames.append(currMeanFrame)

        # temporally save results
        print 'Saving temporary motion correction results for file:', path
        fileFolder, fileName = os.path.split(path)
        newFileName = os.path.splitext(fileName)[0]+'_correction_results.pkl'
        if saveFolder is not None: savePath = os.path.join(saveFolder,newFileName)
        else: savePath = os.path.join(fileFolder,newFileName)
        ft.saveFile(savePath,{'offset':currOffset,'meanFrame':currMeanFrame.astype(np.float32),'path':path,'status':'single_file'})
        print 'End of alignment for file:',path

    meanFrames = np.array(meanFrames)
    if len(paths) > 1:
        print '\nPlotting distance distribution across mean frames of each file ...'
        _, f = getDistanceList(meanFrames,meanFrames[0,:,:],normFunc=normFunc,isPlot=True)
        f.suptitle('Distances across files'); plt.show()
        print 'Start alignment across files...'
        fileOffset, allMeanFrames, aveMeanFrame = alignSingleMovieLoop(meanFrames,iterations=5,badFrameDistanceThr=65535,maxDisplacement=maxDisplacement,normFunc=normFunc,verbose=verbose)
        print 'Plotting mean frame of each file before and after cross file alignment ...'
        tf.imshow(np.dstack((np.array(meanFrames), np.array(allMeanFrames))),photometric='miniswhite', cmap='gray'); plt.show()

        for i, path in enumerate(paths):
            offsets[i] = offsets[i] + fileOffset[i,:]
            print 'Saving motion correction results for file:', path
            fileFolder, fileName = os.path.split(path)
            newFileName = os.path.splitext(fileName)[0]+'_correction_results.pkl'
            if saveFolder is not None: savePath = os.path.join(saveFolder,newFileName)
            else: savePath = os.path.join(fileFolder,newFileName)
            ft.saveFile(savePath,{'offset':offsets[i],'meanFrame':allMeanFrames[i,:,:].astype(np.float32),'path':path,'status':'cross_files'})
        print 'End of cross file alignment.\n'
    else: print '\nThere is only one file in the list. No need to align across files\n'; aveMeanFrame = meanFrames[0]

    if output:
        for i, path in enumerate(paths):
            print 'Generating output image file for '+path
            fileFolder, fileName = os.path.split(path)
            newFileName = ('_'+fileNameSurfix).join(os.path.splitext(fileName))
            if saveFolder is None: newPath = os.path.join(fileFolder,newFileName)
            else: newPath = os.path.join(saveFolder,newFileName)
            mov = tf.imread(path)
            for j in range(mov.shape[0]):
                if not np.array_equal(offsets[i][j,:], np.array([0,0])):
                    mov[j,:,:] = rigid_transform(mov[j, :, :], offset=offsets[i][j, :])
            tf.imsave(newPath, mov-cameraBias)

    return offsets, aveMeanFrame




#todo: added open cv phase correlation method and make motion correction class.












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
    # imgPath = r"E:\data2\2015-06-11-python-2P-analysis-test\motion_correction_test\for_Jun\test_001.tif"
    # img = tf.imread(imgPath)
    # distanceList = getDistanceList(img,img[0,:,:],isPlot=True)
    # #======================================================================================================

    #======================================================================================================
    # imgPath = r"E:\data2\2015-06-11-python-2P-analysis-test\motion_correction_test\for_Jun\test_001.tif"
    # img = tf.imread(imgPath)
    # offset, _ = iamstupid(img[1,:,:],img[2,:,:])
    # print offset
    #======================================================================================================

    #======================================================================================================
    # imgPath = r"E:\data2\2015-06-11-python-2P-analysis-test\motion_correction_test\for_Jun\test_001.tif"
    # img = tf.imread(imgPath)
    # offsetList, alignedMov, meanFrame = alignSingleMovie(img,img[0,:,:],badFrameDistanceThr=110)
    #
    # tf.imshow(np.dstack((img,alignedMov)),cmap='gray')
    # plt.show()
    #======================================================================================================

    #======================================================================================================
    # imgPath = r"E:\data2\2015-06-11-python-2P-analysis-test\motion_correction_test\for_Jun\test_001.tif"
    # img = tf.imread(imgPath)
    # offsetList, alignedMov, meanFrame = alignSingleMovie(img,img[0,:,:],badFrameDistanceThr=110)
    #
    # img2Path = r"E:\data2\2015-06-11-python-2P-analysis-test\motion_correction_test\for_Jun\test_002.tif"
    # img2 = tf.imread(img2Path)
    # offsetList2, alignedMov2, meanFrame2 = alignSingleMovie(img2,img2[0,:,:],badFrameDistanceThr=110)
    #
    # print np.hstack((offsetList,offsetList2))
    #======================================================================================================

    #======================================================================================================
    # imgPath = r"E:\data2\2015-06-11-python-2P-analysis-test\motion_correction_test\for_Jun\test_001.tif"
    # img = tf.imread(imgPath)
    # offsetList, alignedMov, meanFrame = alignSingleMovieLoop(img,badFrameDistanceThr=110)
    # print offsetList
    #======================================================================================================

    #======================================================================================================
    # imgPath = r"E:\data2\2015-06-11-python-2P-analysis-test\motion_correction_test\for_Jun\test_001.tif"
    # img = tf.imread(imgPath)
    # offsetList, alignedMov, meanFrame = alignSingleMovie(img,img[-1,:,:],badFrameDistanceThr=110,alignOrder=1)
    # offsetList2, alignedMov2, meanFrame2 = alignSingleMovie(img,img[-1,:,:],badFrameDistanceThr=110,alignOrder=-1)
    # print np.hstack((offsetList,offsetList2))
    # tf.imshow(np.dstack((img,alignedMov,alignedMov2)),cmap='gray')
    # plt.show()
    #======================================================================================================

    #======================================================================================================
    # paths=[
    #        r"E:\data2\2015-06-11-python-2P-analysis-test\motion_correction_test\for_Jun\test_001.tif",
    #        r"E:\data2\2015-06-11-python-2P-analysis-test\motion_correction_test\for_Jun\test_002.tif"
    #        ]
    # offsets, meanFrame = alignMultipleTiffs(paths,
    #                                            iterations=2,
    #                                            badFrameDistanceThr=100,
    #                                            maxDisplacement=10,
    #                                            normFunc=ia.array_diff,
    #                                            verbose=False,
    #                                            output=True,
    #                                            saveFolder=None,
    #                                            fileNameSurfix='corrected',
    #                                            cameraBias=0)
    #
    # print offsets[0]-offsets[1]
    #======================================================================================================

    print 'for debug...'