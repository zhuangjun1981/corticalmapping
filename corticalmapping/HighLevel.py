__author__ = 'junz'

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import core.ImageAnalysis as ia
import core.TimingAnalysis as ta
import core.tifffile as tf
import RetinotopicMapping as rm
import core.FileTools as ft
import scipy.ndimage as ni
from toolbox.misc import BinarySlicer


def translateMovieByVasculature(mov,parameterPath,movDecimation,mappingDecimation=2):
    '''

    :param mov: movie before translation (could be 2d (just one frame) or 3d)
    :param parameterPath: path to the json file with translation parameters generated by VasculatureMapMatching GUI
    :param movDecimation: decimation factor from movie vasculature image to movie
    :param mappingDecimation: decimation factor from mapping vasculature image to mapped areas, usually 2
    :return: translated movie
    '''

    with open(parameterPath) as f:
        matchingParams = json.load(f)

    movDecimation = float(movDecimation); mappingDecimation=float(mappingDecimation)

    if matchingParams['Xoffset']%movDecimation != 0: print 'Original Xoffset is not divisble by movDecimation. Taking the floor integer.'
    if matchingParams['Yoffset']%movDecimation != 0: print 'Original Yoffset is not divisble by movDecimation. Taking the floor integer.'

    offset =  [int(matchingParams['Xoffset']/movDecimation),
               int(matchingParams['Yoffset']/movDecimation)]

    if matchingParams['ReferenceMapHeight']%movDecimation != 0: print 'Original ReferenceMapHeight is not divisble by movDecimation. Taking the floor integer.'
    if matchingParams['ReferenceMapWidth']%movDecimation != 0: print 'Original ReferenceMapWidth is not divisble by movDecimation. Taking the floor integer.'

    outputShape = [int(matchingParams['ReferenceMapHeight']/movDecimation),
                   int(matchingParams['ReferenceMapHeight']/movDecimation)]

    movT = ia.rigidTransform_cv2(mov,zoom=matchingParams['Zoom'],rotation=matchingParams['Rotation'],offset=offset,outputShape=outputShape)

    if mappingDecimation/movDecimation != 1:
        movT = ia.rigidTransform_cv2(movT, zoom=mappingDecimation/movDecimation)

    print 'shape of output movie:', movT.shape

    return movT

def segmentMappingPhotodiodeSignal(pd,digitizeThr=0.9,filterSize=0.01,segmentThr=0.02,Fs=10000.):
    '''

    :param pd: photodiode from mapping jphys file
    :param digitizeThr: threshold to digitize photodiode readings
    :param filterSize: gaussian filter size to filter photodiode signal, sec
    :param segmentThr: threshold to detect the onset of each stimulus sweep
    :param Fs: sampling rate
    :return:
    '''

    pd[pd<digitizeThr] = 0.; pd[pd>=digitizeThr] = 5.

    filterDataPoint = int(filterSize*Fs)

    pdFiltered = ni.filters.gaussian_filter(pd, filterDataPoint)
    pdFilteredDiff = np.diff(pdFiltered)
    pdFilteredDiff = np.hstack(([0],pdFilteredDiff))
    pdSignal = np.multiply(pd, pdFilteredDiff)

    displayOnsets = ta.getOnsetTimeStamps(pdSignal, Fs, threshold = segmentThr, onsetType='raising')

    print '\nNumber of sweep onsets:', len(displayOnsets)
    print '\nDisplay onsets (sec):',displayOnsets,'\n'

    return displayOnsets

def getlogPathList(date,#string
                   mouseID,#string
                   stimulus='',#string
                   userID='',#string
                   fileNumber='',#string
                   displayFolder=r'\\W7DTMJ007LHW\data\sequence_display_log'):
    logPathList = []
    for f in os.listdir(displayFolder):
        fn, ext = os.path.splitext(f)
        strings = fn.split('-')
        try: dateTime,stim,mouse,user,fileNum=strings[0:5]
        except Exception as e:
            # print 'Can not read path:',f,'\n',e
            continue
        if (dateTime[0:6] == date) and (mouseID in mouse) and (stimulus in stim) and (userID in user) and (fileNumber == fileNum):
            logPathList.append(os.path.join(displayFolder,f))
    print '\n'+'\n'.join(logPathList)+'\n'
    return logPathList

def analysisMappingDisplayLogs(logPathList):
    '''
    :param logFileList: list of paths of all visual display logs of a mapping experiment
    :return:
    B2U: dictionary of all bottom up sweeps
        'ind': indices of these sweeps in the whole experiments
        'startTime': starting time relative to stimulus onset
        'endTime': end time relative to stimulus onset
        'slope': slope of the linear relationship between phase and retinotopic location
        'intercept': intercept of the linear relationship between phase and retinotopic location

    same for U2B, L2R and R2L
    '''

    displayInfo = {
                   'B2U':{'ind':[],'startTime':[],'sweepDur':[],'slope':[],'intercept':[]},
                   'U2B':{'ind':[],'startTime':[],'sweepDur':[],'slope':[],'intercept':[]},
                   'L2R':{'ind':[],'startTime':[],'sweepDur':[],'slope':[],'intercept':[]},
                   'R2L':{'ind':[],'startTime':[],'sweepDur':[],'slope':[],'intercept':[]}
                   }

    ind=0

    for logPath in sorted(logPathList):
        log = ft.loadFile(logPath)

        #get sweep direction
        direction = log['stimulation']['direction']
        if log['presentation']['displayOrder']==-1: direction=direction[::-1]
        currDict = displayInfo[direction]

        #get number of sweeps
        sweepNum = log['stimulation']['iteration'] * log['presentation']['displayIteration']
        currDict['ind'] += range(ind,ind+sweepNum)
        ind += sweepNum

        #get startTime, sweep duration, phase position relationship
        if not currDict['startTime']:
            refreshRate = float(log['monitor']['refreshRate'])
            interFrameInterval = np.mean(np.diff(log['presentation']['timeStamp']))
            if interFrameInterval > (1.01/refreshRate): raise ValueError, 'Mean visual display too long: '+str(interFrameInterval)+'sec' # check display
            if interFrameInterval < (0.99/refreshRate): raise ValueError, 'Mean visual display too short: '+str(interFrameInterval)+'sec' # check display

            if log['presentation']['displayOrder']==1: currDict['startTime'] = -1 * log['stimulation']['preGapFrame'] / refreshRate
            if log['presentation']['displayOrder']==-1: currDict['startTime'] = -1 * log['stimulation']['postGapFrame'] / refreshRate

            currDict['sweepDur'] = len(log['presentation']['displayFrames']) / (log['stimulation']['iteration'] * log['presentation']['displayIteration'] * refreshRate)

            currDict['slope'], currDict['intercept'] = rm.getPhasePositionEquation(log)

    return displayInfo

def getAverageDfMovie(movPath, frameTS, onsetTimes, chunkDur, startTime=0., temporalDownSampleRate=1):
    '''
    :param movPath: path to the image movie
    :param frameTS: the timestamps for each frame of the raw movie
    :param onsetTimes: time stamps of onset of each sweep
    :param startTime: chunck start time relative to the sweep onset time (length of pre gray period)
    :param chunkDur: duration of each chunk
    :param temporalDownSampleRate: decimation factor in time after recording
    :return: averageed movie of all chunks
    '''

    if temporalDownSampleRate == 1:
        frameTS_real = frameTS
    elif temporalDownSampleRate >1:
        frameTS_real = frameTS[::temporalDownSampleRate]
    else:
        raise ValueError, 'temporal downsampling rate can not be less than 1!'

    mov = BinarySlicer(movPath)

    aveMov = ia.getAverageMovie(mov, frameTS_real, onsetTimes+startTime, chunkDur)

    meanFrameDur = np.mean(np.diff(frameTS_real))
    baselineFrameDur = int(abs(startTime) / meanFrameDur)

    baselinePicture = np.mean(aveMov[0:baselineFrameDur,:,:],axis=0)
    _, aveMovNor, _ = ia.normalizeMovie(aveMov,baselinePicture)

    return aveMov, aveMovNor

def saveMappingMovies(movPath,frameTS,displayOnsets,displayInfo,saveFolder,temporalDownSampleRate=1,filePrefix=''):

    for dir in ['B2U','U2B','L2R','R2L']:
        print '\nAnalyzing sweeps with direction:', dir
        aveMov, aveMovNor = getAverageDfMovie(movPath=movPath,
                                              frameTS=frameTS,
                                              onsetTimes=displayOnsets[displayInfo[dir]['ind']],
                                              chunkDur=displayInfo[dir]['sweepDur'],
                                              startTime=displayInfo[dir]['startTime'],
                                              temporalDownSampleRate=temporalDownSampleRate)

        print 'Saving averaged F movie and DF movie for sweep direction:', dir

        if filePrefix: filePrefix += '_'
        tf.imsave(os.path.join(saveFolder,filePrefix+'aveMov_'+dir+'.tif'),aveMov.astype(np.float32))
        tf.imsave(os.path.join(saveFolder,filePrefix+'aveMovNor_'+dir+'.tif'),aveMovNor.astype(np.float32))

        print 'End of movie averaging for sweep direction:', dir, '\n'

def getVasMap(vasMapPaths,
              saveFolder = None,
              dtype = np.dtype('<u2'),
              headerLength = 116,
              tailerLength = 218,
              column = 1024,
              row = 1024,
              frame = 1,
              crop = None,
              mergeMethod = np.mean, # np.median, np.min, np.max
              fileNamePrefix=''
              ):

    vasMaps = []
    for vasMapPath in vasMapPaths:
        currVasMap,_,_= ft.importRawJCamF(vasMapPath,saveFolder=None,dtype=dtype,headerLength=headerLength,tailerLength=tailerLength,
                                          column=column,row=row,frame=frame,crop=crop)
        vasMaps.append(currVasMap[0].astype(np.float32))
    vasMap = vasMapMergeMethod(vasMaps,axis=0)
    if saveFolder is not None:
        if fileNamePrefix: fileNamePrefix += '_'
        tf.imsave(os.path.join(saveFolder,fileNamePrefix+'vasMap.tif'),vasMap)

    return vasMap


def getMappingPKL(movFolder,
                  displayInfo,
                  mouseID,
                  cycles=1,
                  vasculatureMap=None,
                  isReverse=False,
                  trialNum=None,
                  mouseType=None,
                  isAnesthetized=None,
                  dateRecorded=None,
                  visualStimType=None,
                  visualStimBackground=None,
                  imageExposureTime=None,
                  params ={},
                  ):
    #todo: finish this function
    pass



if __name__ == '__main__':

    #===========================================================================
    movPath = r"\\watersraid\data\Jun\150901-M177931\150901JCamF105_1_1_10.npy"
    jphysPath = r"\\watersraid\data\Jun\150901-M177931\150901JPhys105"
    vasMapPaths = [r"\\watersraid\data\Jun\150901-M177931\150901JCamF104"]
    displayFolder = r'\\W7DTMJ007LHW\data\sequence_display_log'
    saveFolder = r'E:\data\2015-09-04-150901-M177931-FlashCameraMapping'

    dateRecorded = '150901'
    mouseID = '177931'
    fileNum = '105'

    vasMapDtype = np.dtype('<u2')
    vasMapHeaderLength = 116
    vasMapTailerLength = 218
    vasMapColumn = 1024
    vasMapRow = 1024
    vasMapFrame = 1
    vasMapCrop = None
    vasMapMergeMethod = np.mean #np.median,np.min,np.max

    temporalDownSampleRate = 10




    vasMap = getVasMap(vasMapPaths,
                       saveFolder = saveFolder,
                       dtype = vasMapDtype,
                       headerLength = vasMapHeaderLength,
                       tailerLength = vasMapTailerLength,
                       column = vasMapColumn,
                       row = vasMapRow,
                       frame = vasMapFrame,
                       crop = vasMapCrop,
                       mergeMethod = vasMapMergeMethod, # np.median, np.min, np.max
                       fileNamePrefix = dateRecorded+'_M'+mouseID)

    _, jphys = ft.importRawNewJPhys(jphysPath)
    pd = jphys['photodiode']
    displayOnsets = segmentMappingPhotodiodeSignal(pd)

    imgFrameTS = ta.getOnsetTimeStamps(jphys['read'])

    logPathList = getlogPathList(dateRecorded,mouseID,fileNumber=fileNum)
    displayInfo = analysisMappingDisplayLogs(logPathList)

    saveMappingMovies(movPath = movPath,
                      frameTS=imgFrameTS,
                      displayOnsets=displayOnsets,
                      displayInfo=displayInfo,
                      saveFolder=saveFolder,
                      temporalDownSampleRate=temporalDownSampleRate,
                      filePrefix=dateRecorded+'_M'+mouseID)

    #===========================================================================

    print 'for debug...'
