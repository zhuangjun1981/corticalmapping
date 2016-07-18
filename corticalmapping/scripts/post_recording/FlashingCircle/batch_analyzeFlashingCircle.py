__author__ = 'junz'


import os
import numpy as np
import matplotlib.pyplot as plt
from toolbox.misc import BinarySlicer
import warnings
import tifffile as tf
import corticalmapping.core.FileTools as ft
import corticalmapping.core.TimingAnalysis as ta
import corticalmapping.HighLevel as hl
import corticalmapping.RetinotopicMapping as rm



dateRecorded = '160219' # str 'yymmdd'
mouseID = 'TEST' # str, without 'M', for example: '214522'
userID = 'Jun' # user name, should be consistent withe the display log user name
vasfileNums = [100] # file numbers of vasculature images, should be a list
fileNumList = [101] # file number of the imaged movie



dataFolder = r"\\aibsdata2\nc-ophys\CorticalMapping\IntrinsicImageData"
dataFolder = os.path.join(dataFolder,dateRecorded+'-M'+mouseID+'-FlashingCircle')
fileList = os.listdir(dataFolder)

# vasculature map parameters
vasMapDtype = np.dtype('<u2')
vasMapHeaderLength = 116
vasMapTailerLength = 218
vasMapColumn = 1024
vasMapRow = 1024
vasMapFrame = 1
vasMapCrop = None
vasMapMergeMethod = np.mean #np.median,np.min,np.max

#jphys parameters
jphysDtype = np.dtype('>f')
jphysHeaderLength = 96 # length of the header for each channel
jphysChannels = ('photodiode','read','trigger','visualFrame','video1','video2','runningRef','runningSig','open1','open2')# name of all channels
jphysFs = 10000.

#photodiode signal parameters
pdDigitizeThr=0.9
pdFilterSize=0.01
pdSegmentThr=0.02
smallestInterval=1.

#image read signal parameters
readThreshold = 3.
readOnsetType='raising'
temporalDownSampleRate = 1



saveFolder = os.path.dirname(os.path.realpath(__file__))
os.chdir(saveFolder)

vasMapPaths = []
if vasfileNums is not None:
    for vasfileNum in vasfileNums:
        fn = [f for f in fileList if 'JCamF'+str(vasfileNum) in f][0]
        vasMapPaths.append(os.path.join(dataFolder,fn))

if vasMapPaths:
    vasMap = hl.getVasMap(vasMapPaths,dtype=vasMapDtype,headerLength=vasMapHeaderLength,tailerLength=vasMapTailerLength,
                          column=vasMapColumn,row=vasMapRow,frame=vasMapFrame,crop=vasMapCrop,mergeMethod=vasMapMergeMethod)
else:
    print 'No vasculature map find. Taking first frame of movie as vasculature map.'
    firstMovPath = os.path.join(dataFolder, [f for f in fileList if (dateRecorded+'JCamF'+str(fileNumList[0]) in f) and ('.npy' in f)][0])
    vasMap = BinarySlicer(firstMovPath)[0,:,:]
tf.imsave(dateRecorded+'_M'+mouseID+'_vasMap.tif',vasMap.astype(np.float32))

for fileNum in fileNumList:
    movPath = os.path.join(dataFolder, [f for f in fileList if (dateRecorded+'JCamF'+str(fileNum) in f) and ('.npy' in f)][0])

    jphysPath = os.path.join(dataFolder, [f for f in fileList if dateRecorded+'JPhys'+str(fileNum) in f][0])

    _, jphys = ft.importRawNewJPhys(jphysPath,dtype=jphysDtype,headerLength=jphysHeaderLength,channels=jphysChannels,sf=jphysFs)

    pd = jphys['photodiode']

    displayOnsets = hl.segmentMappingPhotodiodeSignal(pd,digitizeThr=pdDigitizeThr,filterSize=pdFilterSize,segmentThr=pdSegmentThr,Fs=jphysFs,smallestInterval=smallestInterval)

    imgFrameTS = ta.get_onset_timeStamps(jphys['read'], Fs=jphysFs, threshold=readThreshold, onsetType=readOnsetType)

    logPath = hl.findLogPath(date=dateRecorded,mouseID=mouseID,stimulus='FlashingCircle',userID=userID,fileNumber=str(fileNum),displayFolder=dataFolder)

    log = ft.loadFile(logPath)
    refreshRate = float(log['monitor']['refreshRate'])
    preGapDur = log['stimulation']['preGapFrameNum'] / refreshRate
    postGapDur = log['stimulation']['postGapFrameNum'] / refreshRate
    displayDur = log['stimulation']['flashFrame'] / refreshRate

    # print 'preGapDur:',preGapDur
    # print 'postGapDur:',postGapDur
    # print 'displayDur:',displayDur

    aveMov, aveMovNor = hl.getAverageDfMovie(movPath, imgFrameTS, displayOnsets, preGapDur+postGapDur+displayDur, startTime=-preGapDur, temporalDownSampleRate=temporalDownSampleRate)

    tf.imsave(dateRecorded + '_M' + mouseID + '_aveMov_' + str(fileNum) + '.tif', aveMov)
    tf.imsave(dateRecorded + '_M' + mouseID + '_aveMovNor_' + str(fileNum) + '.tif', aveMovNor)
