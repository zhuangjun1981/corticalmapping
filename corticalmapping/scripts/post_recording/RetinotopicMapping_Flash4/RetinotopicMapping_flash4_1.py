__author__ = 'junz'


import os
import numpy as np
import matplotlib.pyplot as plt
import corticalmapping.core.tifffile as tf
import corticalmapping.core.FileTools as ft
import corticalmapping.core.TimingAnalysis as ta
import corticalmapping.HighLevel as hl
import corticalmapping.RetinotopicMapping as rm

movPath = r"\\watersraid\data\Jun\150901-M177931\150901JCamF105_1_1_10.npy"
jphysPath = r"\\watersraid\data\Jun\150901-M177931\150901JPhys105"
vasMapPaths = [r"\\watersraid\data\Jun\150901-M177931\150901JCamF104"]
displayFolder = r'\\W7DTMJ007LHW\data\sequence_display_log'

dateRecorded = '150901'
mouseID = '177931'
fileNum = '105'

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
trialNum='4_5'
mouseType='Emx1-IRES-Cre;Camk2a-tTA;Ai93(TITL-GCaMP6f)'
isAnesthetized=False,
visualStimType='KSstim'
visualStimBackground='gray'
analysisParams ={'phaseMapFilterSigma': 1.,
                 'signMapFilterSigma': 9.,
                 'signMapThr': 0.3,
                 'eccMapFilterSigma': 15.0,
                 'splitLocalMinCutStep': 10.,
                 'closeIter': 3,
                 'openIter': 3,
                 'dilationIter': 15,
                 'borderWidth': 1,
                 'smallPatchThr': 100,
                 'visualSpacePixelSize': 0.5,
                 'visualSpaceCloseIter': 15,
                 'splitOverlapThr': 1.1,
                 'mergeOverlapThr': 0.1}




vasMap = hl.getVasMap(vasMapPaths,dtype=vasMapDtype,headerLength=vasMapHeaderLength,tailerLength=vasMapTailerLength,
                      column=vasMapColumn,row=vasMapRow,frame=vasMapFrame,crop=vasMapCrop,mergeMethod=vasMapMergeMethod)

tf.imsave(os.path.join(saveFolder,dateRecorded+'_M'+mouseID+'_vasMap.tif'),vasMap)

_, jphys = ft.importRawNewJPhys(jphysPath,dtype=jphysDtype,headerLength=jphysHeaderLength,channels=jphysChannels,sf=jphysFs)

pd = jphys['photodiode']

displayOnsets = hl.segmentMappingPhotodiodeSignal(pd,digitizeThr=pdDigitizeThr,filterSize=pdFilterSize,segmentThr=pdSegmentThr,Fs=jphysFs)

imgFrameTS = ta.getOnsetTimeStamps(jphys['read'],Fs=jphysFs,threshold=readThreshold,onsetType=readOnsetType)

logPathList = hl.getlogPathList(date=dateRecorded,mouseID=mouseID,stimulus='',userID='',fileNumber=fileNum,displayFolder=displayFolder)

displayInfo = hl.analysisMappingDisplayLogs(logPathList)

movies, moviesNor = hl.getMappingMovies(movPath=movPath,frameTS=imgFrameTS,displayOnsets=displayOnsets,displayInfo=displayInfo,temporalDownSampleRate=temporalDownSampleRate)

for dir,mov in movies.iteritems():
    tf.imsave(os.path.join(saveFolder,dateRecorded+'_M'+mouseID+'_aveMov_'+dir+'.tif'),mov)
for dir,movNor in moviesNor.iteritems():
    tf.imsave(os.path.join(saveFolder,dateRecorded+'_M'+mouseID+'_aveMovNor_'+dir+'.tif'),movNor)

del moviesNor

altPosMap,aziPosMap,altPowerMap,aziPowerMap = hl.getPositionAndPowerMap(movies=movies,displayInfo=displayInfo,FFTmode=FFTmode,cycles=cycles)

del movies

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