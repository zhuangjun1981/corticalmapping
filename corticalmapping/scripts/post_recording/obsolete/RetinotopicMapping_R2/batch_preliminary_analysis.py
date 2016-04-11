# -*- coding: utf-8 -*-
"""
Created on Tue Apr 08 09:44:02 2014

@author: junz
"""

# version 7


import os
import numpy as np
import scipy.ndimage as ndi
import scipy.stats.stats as stats

import corticalmapping.core.tifffile as tf
from corticalmapping.core.FileTools import loadFile,saveFile,importRawJCam,importRawNewJPhys
from corticalmapping.core.ImageAnalysis import normalize_movie,array_nor
from corticalmapping.core.PlottingTools import mergeNormalizedImages


def preliminary_analysis(imageDate,
                         mouseID,
                         imageFileNum,
                         vasculatureMapList,
                         decimation = None,
                         JPhysChannels=['photodiode2','read','trigger','photodiode','sweep','visualFrame','runningRef','runningSig','reward','licking'],
                         JPhysFs=10000.,
                         photodiodeThr = .003, #threshold of photo diode signal
                         isTruncate=False,
                         truncateArea=[25,475,100,550], #[row start, row end, column start, column end]
                         savingFolder = 'C:\\data',
                         imageFileFolder = r'\\aibsdata2\nc-ophys\CorticalMapping\IntrinsicImageData',
                         displayLogFolder = r'\\aibsdata2\nc-ophys\CorticalMapping\sequence_display_log',
                         dtype = np.dtype('>f'), # data type of JCam and Jphys file
                         headerLength = 96, # length of the header, measured as the data type defined above               
                         columnNumIndex = 14, # index of number of rows in header of JCam file
                         rowNumIndex = 15, # index of number of columns in header of JCam file
                         frameNumIndex = 16, # index of number of frames in header of JCam file
                         exposureTimeIndex = 17, # index of exposure time in header of JCam file (ms)
                         ):    
    print 'Starting analysing file JCam' + str(imageFileNum)
    print 'Mouse:' + mouseID
    print 'Date: ' + imageDate
    
    #get path for all image files
    
    JCamFilePath = os.path.join(imageFileFolder + '\\' + imageDate + '-M' + mouseID, 
                                imageDate + 'JCam' + imageFileNum)
                                
    JPhysFilePath = os.path.join(imageFileFolder + '\\' + imageDate + '-M' + mouseID, 
                                imageDate + 'JPhys' + imageFileNum)
    
    #get path for visual stimulation display log files
    displayLogFileNameList = []
            
    for f in os.listdir(displayLogFolder):
        if os.path.isfile(os.path.join(displayLogFolder,f)) and f[-7:-4] == imageFileNum and f[0:6] == imageDate:
            mouseIndexStart = f.find('mouse') + 5
            mouseIndexEnd = f.find('-', mouseIndexStart)
            if f[mouseIndexStart:mouseIndexEnd] == mouseID:
                displayLogFileNameList.append(f)
    
    if len(displayLogFileNameList) > 1:
        raise LookupError, 'More than one display log files correspont to this image file.'
        
    elif len(displayLogFileNameList) == 0: # no visual stim display
        print "No visual stimulation for this image file. Stop analyzing!!"
    
    else: # there is visual stim display
        print 'There is visual stimulation for this image file. Continue analyzing...\n'
        
        displayLogFilePath = os.path.join(displayLogFolder, displayLogFileNameList[0])
        #read visual stim log file
        displayLog = loadFile(displayLogFilePath)
        
        #load image file
        rawMov, _ = importRawJCam(JCamFilePath, decimation = decimation)
        
        #load JPhys file
        _, JPhysDict = importRawNewJPhys(JPhysFilePath,
                                            channels = JPhysChannels, 
                                            sf=JPhysFs)
                                            
        # generate time stamp for each image frame
        imageFrameNum = rawMov.shape[0]
        imageFrameTS = []
        for i in range(1,len(JPhysDict['read'])):
            if JPhysDict['read'][i-1] < 3.0 and JPhysDict['read'][i] >= 3.0:
                imageFrameTS.append(i*(1./JPhysFs))
        
        if len(imageFrameTS) < imageFrameNum:
            raise LookupError, "Expose period number is smaller than image frame number!"
        imageFrameTS = imageFrameTS[0:imageFrameNum]
        
        #get precise exposure time
        exposure = [imageFrameTS[i+1]-imageFrameTS[i] for i in range(len(imageFrameTS)-1)]
        exposure = np.mean(exposure)
        
        #generate display start for each presentation
        displayOnset = []
        photodiode = JPhysDict['photodiode']
        photodiode[photodiode<1.] = 0.
        photodiode[photodiode>=1.] = 5.
        photodiodeFiltered = ndi.filters.gaussian_filter(photodiode, 50.)
        photodiodeFilteredDiff = np.diff(photodiodeFiltered)
        photodiodeFilteredDiff = np.hstack(([0],photodiodeFilteredDiff))
        photodiodeSignal = np.multiply(photodiode, photodiodeFilteredDiff)
        
        for i in range(1,len(photodiodeSignal)):
            if photodiodeSignal[i] > photodiodeThr and \
               photodiodeSignal[i-1]  < photodiodeThr:
                   
                  displayOnset.append(i * (1. / JPhysFs)) 
            
        displayOnset = np.array(displayOnset)

        print 'Number of presentation:', len(displayOnset)
        
        #get pre-onset image frame number and whole duration image frame number
        refreshRate = float(displayLog['monitor']['refreshRate'])
        preOnsetDur = displayLog['stimulation']['preGapFrame'] / refreshRate
        preOnsetImageFrame = int(np.ceil(preOnsetDur / exposure))
        iterDur = len(displayLog['presentation']['displayFrames']) / \
                  (displayLog['stimulation']['iteration'] * \
                  displayLog['presentation']['displayIteration'] * \
                  refreshRate)
        iterDurImageFrame = int(np.ceil(iterDur / exposure))
        
        
        #find image Frame index of display onset
        imageIterIndex = []
        for i, onset in enumerate(displayOnset):
            Tdiff = np.abs(imageFrameTS - onset)
            imageIterIndex.append(Tdiff.argmin())
        imageIterIndex = np.array(imageIterIndex)
        del Tdiff
        
        #check if there are enough image frames for visual stim, if not adding np.nans
        if imageIterIndex[-1]-preOnsetImageFrame + iterDurImageFrame > rawMov.shape[0]:
            missFrameNum = imageIterIndex[-1]-preOnsetImageFrame + iterDurImageFrame - rawMov.shape[0]
            print 'Missing '+str(missFrameNum)+' image frames.\nAdding np.nans...\n'
        else:
            print 'There are enough image frames.\nKeep analyzing...\n'
        
        #calculate average movie
        aveMov = []
        for i, TS in enumerate(imageIterIndex):
            currStartInd = TS-preOnsetImageFrame
            currEndInd = TS-preOnsetImageFrame + iterDurImageFrame
            preMissingFrameNum = None
            postMissingFrameNum = None
            
            if currStartInd < 0:
                preMissingFrameNum = - (TS-preOnsetImageFrame)
                currStartInd = 0
            elif currEndInd > rawMov.shape[0]:
                postMissingFrameNum = currEndInd - rawMov.shape[0]
                currEndInd = rawMov.shape[0]
                
            currIterMov = rawMov[currStartInd:currEndInd,:,:]
            
            if preMissingFrameNum:
                preMov = np.zeros((preMissingFrameNum,rawMov.shape[1],rawMov.shape[2]))
                preMov[:] = np.nan
                currIterMov = np.concatenate((preMov,currIterMov), axis = 0)
                
            if postMissingFrameNum:
                postMov = np.zeros((postMissingFrameNum,rawMov.shape[1],rawMov.shape[2]))
                postMov[:] = np.nan
                currIterMov = np.concatenate((currIterMov, postMov), axis = 0)
                
            aveMov.append(currIterMov)
            
            del currStartInd, currEndInd, preMissingFrameNum, postMissingFrameNum, currIterMov
            
        aveMov = stats.nanmean(aveMov, axis=0).astype(np.float32)
        
        
        #calculate normalized average movie
        preGapFrame = displayLog['stimulation']['preGapFrame']
        refreshRate = float(displayLog['monitor']['refreshRate'])
        preOnsetImageFrame = int(np.round(preGapFrame / (refreshRate * exposure)))
        
        baseline = np.mean(aveMov[0:preOnsetImageFrame,:,:], axis = 0)
        _, aveMovNor, _ = normalize_movie(aveMov, baselinePic = baseline)
        aveMovNor = aveMovNor.astype(np.float32)
        
        #reading vasculature map
        vasMap = []
        for i, vasculatureMapNum in enumerate(vasculatureMapList):
            vasculatureMapFilepath = os.path.join(imageFileFolder + '\\' + imageDate + '-M' + mouseID,
                                                  imageDate + 'JCam' + str(vasculatureMapNum))
            try:
                currVasMap, _ = importRawJCam(vasculatureMapFilepath)
                if currVasMap.shape[0] > 1:
                    vasMap.append(array_nor(currVasMap[0]))
                    print 'More than one frame in the vasculature map file. Taking the first frame as vasMap ...'
                elif currVasMap.shape[0] == 1:
                    vasMap.append(array_nor(currVasMap[0]))
            except IOError: pass

        if len(vasMap)==0:
            vasMap1 = array_nor(rawMov[0])
            vasMap2 = mergeNormalizedImages([rawMov[0]])
            print 'Did not find vasculature map file. Taking the first frame of the movie as vasMap...'
        else:
            vasMap1 = array_nor(np.mean(vasMap, axis=0))
            vasMap2 = mergeNormalizedImages(vasMap)

        
        fileName = 'File' + imageFileNum
        
        print 'Saving files for file ' + imageFileNum + '...\n'    
        #save file
        if isTruncate:
            tf.imsave(os.path.join(savingFolder, 'aveMovie_' + fileName + '.tif'), aveMov[:,truncateArea[0]:truncateArea[1],truncateArea[2]:truncateArea[3]])
            tf.imsave(os.path.join(savingFolder, 'aveMovieNor_' + fileName + '.tif'), aveMovNor[:,truncateArea[0]:truncateArea[1],truncateArea[2]:truncateArea[3]])
        else:
            tf.imsave(os.path.join(savingFolder, 'aveMovie_' + fileName + '.tif'), aveMov)
            tf.imsave(os.path.join(savingFolder, 'aveMovieNor_' + fileName + '.tif'), aveMovNor)
        data = {
                'imageFrameTS': imageFrameTS, 
                'displayLog': displayLog,
                'displayOnset': displayOnset,
                'imageExposureTime': exposure,
                }
        if isTruncate:
            zoom = vasMap1.shape[-1]/aveMov.shape[-1]
            data.update({'vasculatureMap': vasMap1[truncateArea[0]*zoom:truncateArea[1]*zoom,
                                                   truncateArea[2]*zoom:truncateArea[3]*zoom],
                         'vasculatureMapEnhanced': vasMap2[truncateArea[0]*zoom:truncateArea[1]*zoom,
                                                           truncateArea[2]*zoom:truncateArea[3]*zoom]
                        })
        else:
            data.update({'vasculatureMap': vasMap1,
                         'vasculatureMapEnhanced':vasMap2})
        saveFile(os.path.join(savingFolder,fileName + '.pkl'), data)
        print 'End of saving. \n\n\n'
        
        
            
        
if __name__ == '__main__':
        
    imageDate = '141001' # format: '140508'
    mouseID = '147680' # format: '140174'
    
    imageFileList = [105]#[115,116,117,120,121,122,123,123,124,125,127,129,131,132]
    vasculatureMapList = [100,101,102,103] #None
    
    decimation = None #decimation number
    
    isTruncate = False
    truncateArea=[0,450,130,580]
    
    JPhysChannels=['photodiode2',
                   'read',
                   'trigger',
                   'photodiode',
                   'sweep',
                   'visualFrame',
                   'runningRef',
                   'runningSig',
                   'reward',
                   'licking']
                   
#    JPhysChannels=['photodiode2','read','trigger','photodiode'] # for old files
    
    JPhysFs=10000.
    photodiodeThr = 0.04 #threshold of photo diode signal
    savingFolder = os.path.dirname(os.path.realpath(__file__))
    imageFileFolder = r'\\W7DTMJ19VTX\ImageData2'
    displayLogFolder = r'\\W7DTMJ007LHW\data\sequence_display_log'
   
    
    for i in range(len(imageFileList)):
        imageFileNum = str(imageFileList[i])
        preliminary_analysis(imageDate = imageDate,
                             mouseID = mouseID,
                             imageFileNum = imageFileNum,
                             vasculatureMapList = vasculatureMapList,
                             isTruncate=isTruncate,
                             truncateArea=truncateArea,
                             decimation = decimation,
                             JPhysChannels=JPhysChannels,
                             savingFolder = savingFolder,
                             photodiodeThr =  photodiodeThr,
                             imageFileFolder = imageFileFolder,
                             displayLogFolder = displayLogFolder)
        
        
        
        
