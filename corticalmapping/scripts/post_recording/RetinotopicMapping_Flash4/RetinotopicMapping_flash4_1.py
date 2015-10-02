__author__ = 'junz'


import os
import numpy as np
import matplotlib.pyplot as plt
from toolbox.misc import BinarySlicer
import warnings
import corticalmapping.core.tifffile as tf
import corticalmapping.core.FileTools as ft
import corticalmapping.core.TimingAnalysis as ta
import corticalmapping.HighLevel as hl
import corticalmapping.RetinotopicMapping as rm



dateRecorded = '150930'
mouseID = '187474'
fileNum = 101
vasfileNums = None

trialNum='1_2_3_4'
mouseType='Emx1-IRES-Cre;Camk2a-tTA;Ai94(TITL-GCaMP6s)'

displayFolder = r'\\W7DTMJ007LHW\data\sequence_display_log'
dataFolder = r"\\aibsdata2\nc-ophys\CorticalMapping\IntrinsicImageData"
dataFolder = os.path.join(dataFolder,dateRecorded+'-M'+mouseID)
fileList = os.listdir(dataFolder)
movPath = os.path.join(dataFolder, [f for f in fileList if (dateRecorded+'JCamF'+str(fileNum) in f) and ('.npy' in f)][0])
jphysPath = os.path.join(dataFolder, [f for f in fileList if dateRecorded+'JPhys'+str(fileNum) in f][0])
vasMapPaths = []
if vasfileNums is not None:
    for vasfileNum in vasfileNums:
        fn = [f for f in fileList if 'JCamF'+str(vasfileNum) in f][0]
        vasMapPaths.append(os.path.join(dataFolder,fn))

saveFolder = os.path.dirname(os.path.realpath(__file__))
os.chdir(saveFolder)

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
jphysChannels = ('photodiode2','read','trigger','photodiode','sweep','visualFrame','runningRef','runningSig','reward','licking')# name of all channels
jphysFs = 10000.

#photodiode signal parameters
pdDigitizeThr=0.9
pdFilterSize=0.01
pdSegmentThr=0.02

#image read signal parameters
readThreshold = 3.
readOnsetType='raising'

#pos map and power map parameters
FFTmode='peak'
cycles=1
temporalDownSampleRate = 10

#wrap experiment parameters
isAnesthetized=False
visualStimType='KSstim'
visualStimBackground='gray'
analysisParams ={}

if vasMapPaths:
    vasMap = hl.getVasMap(vasMapPaths,dtype=vasMapDtype,headerLength=vasMapHeaderLength,tailerLength=vasMapTailerLength,
                          column=vasMapColumn,row=vasMapRow,frame=vasMapFrame,crop=vasMapCrop,mergeMethod=vasMapMergeMethod)
else:
    print 'No vasculature map find. Taking first frame of movie as vasculature map.'
    vasMap = BinarySlicer(movPath)[0,:,:]

tf.imsave(os.path.join(saveFolder,dateRecorded+'_M'+mouseID+'_vasMap.tif'),vasMap)

_, jphys = ft.importRawNewJPhys(jphysPath,dtype=jphysDtype,headerLength=jphysHeaderLength,channels=jphysChannels,sf=jphysFs)

pd = jphys['photodiode']

displayOnsets = hl.segmentMappingPhotodiodeSignal(pd,digitizeThr=pdDigitizeThr,filterSize=pdFilterSize,segmentThr=pdSegmentThr,Fs=jphysFs)

imgFrameTS = ta.getOnsetTimeStamps(jphys['read'],Fs=jphysFs,threshold=readThreshold,onsetType=readOnsetType)

logPath = hl.findLogPath(date=dateRecorded,mouseID=mouseID,stimulus='KSstimAllDir',userID='',fileNumber=str(fileNum),displayFolder=displayFolder)

displayInfo = hl.analysisMappingDisplayLog(logPath)

sweepNum = len(displayInfo['B2U']['ind']+displayInfo['U2B']['ind']+displayInfo['L2R']['ind']+displayInfo['R2L']['ind'])
if len(displayOnsets) != sweepNum:
    warningMessage = '\nNumber of detected photodiode onsets ('+str(len(displayOnsets))+') is not equal to display sweep number ('+str(sweepNum)+')!\n'
    warnings.warn(warningMessage)

altPosMap,aziPosMap,altPowerMap,aziPowerMap  = hl.getMappingMovies(movPath=movPath,frameTS=imgFrameTS,displayOnsets=displayOnsets,displayInfo=displayInfo,
                                                                   temporalDownSampleRate=temporalDownSampleRate,saveFolder=saveFolder,
                                                                   savePrefix=dateRecorded+'_M'+mouseID,FFTmode='peak',cycles=1)

f = plt.figure(figsize=(12,10))
f.suptitle(dateRecorded+'_M'+mouseID+'_Trial:'+trialNum)
ax1 = f.add_subplot(221); fig1 = ax1.imshow(altPosMap, vmin=-30,vmax=50,cmap='hsv',interpolation='nearest')
f.colorbar(fig1); ax1.set_title('alt position map')
ax2 = f.add_subplot(222); fig2 = ax2.imshow(altPowerMap, vmin=0,vmax=1,cmap='hot',interpolation='nearest')
f.colorbar(fig2); ax2.set_title('alt power map')
ax3 = f.add_subplot(223); fig3 = ax3.imshow(aziPosMap, vmin=0,vmax=120,cmap='hsv',interpolation='nearest')
f.colorbar(fig3); ax3.set_title('azi position map')
ax4 = f.add_subplot(224); fig4 = ax4.imshow(aziPowerMap, vmin=0,vmax=1,cmap='hot',interpolation='nearest')
f.colorbar(fig4); ax4.set_title('alt power map')

plt.show()

f.savefig(os.path.join(saveFolder,dateRecorded+'_M'+mouseID+'_RetinotopicMappingTrial_'+trialNum+'.png'),dpi=300)

trialObj = rm.RetinotopicMappingTrial(mouseID=mouseID,
                                      dateRecorded=int(dateRecorded),
                                      trialNum=trialNum,
                                      mouseType=mouseType,
                                      visualStimType=visualStimType,
                                      visualStimBackground=visualStimBackground,
                                      imageExposureTime=np.mean(np.diff(imgFrameTS)),
                                      altPosMap=altPosMap,
                                      aziPosMap=aziPosMap,
                                      altPowerMap=altPowerMap,
                                      aziPowerMap=altPowerMap,
                                      vasculatureMap=vasMap,
                                      isAnesthetized=isAnesthetized,
                                      params=analysisParams
                                      )

trialDict = trialObj.generateTrialDict()
ft.saveFile(os.path.join(saveFolder,trialObj.getName()+'.pkl'),trialDict)