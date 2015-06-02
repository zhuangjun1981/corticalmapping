__author__ = 'junz'

import h5py
import numpy as np
import os
import FileTools as ft
import matplotlib.pyplot as plt
import pandas as pd

try: from aibs.Analysis.InDevelopment.FileTools import pkl2df
except ImportError as e: print e


def getValidTrials(visualFrames,
                   lapFirstFrame,
                   monitorSpan = (-96.,96.),
                   ):
    '''

    get visual frame index, onset and offset timing and posx information for each trial (defined by pkl2df function)

    :param visualFrames: the visualFrames variable from aibs.CorticalMapping.CorticalMappingDev.BehaviorAnalysis.behaviorTimeAlignment function
                         first column: JPhys time of each visual frame onset
                         second column: posx in degrees
    :param lapFirstFrame: the first visual frame of each lap defined by pkl2df function, 1-d array of integer
    :param monitorSpan: # monitor coverage azimuth in degrees
    :return: validTrialInd: visual frame indicies of onset and offset of each trial. (float type, because the first trial is nan, excluded from further analysis
             trialDur: JPhys time of onset and offset of each trial
             trialPosx: first column: relative timing of each visual frame
                        second column: posx in visual degrees
    '''

    visualFrameTS = visualFrames[:,0]
    posx = visualFrames[:,1]

    posx[np.logical_or(posx<=monitorSpan[0],posx>=monitorSpan[1])]=np.nan

    cutPoint = []
    for i in xrange(len(posx)-1):
        currPosx = posx[i]
        nextPosx = posx[i+1]

        if (~np.isnan(currPosx)) and np.isnan(nextPosx):
            cutPoint.append(i)

    cutPoint = np.array(cutPoint)
    trialVisualFrameInd = np.transpose(np.array([cutPoint[0:-1],cutPoint[1:]]))

    validTrials = [np.array([np.nan,np.nan])] * len(lapFirstFrame)

    for i, firstframe in enumerate(lapFirstFrame):

        for j, currTrial in enumerate(list(trialVisualFrameInd)):
            if firstframe >= currTrial[0] and firstframe < currTrial[1]:
                validTrials[i] = currTrial
                break

    validTrialInd = np.array(validTrials)
    validTrialInd[:,0] = validTrialInd[:,0] + 1

    trialDur = [np.array([np.nan,np.nan])] * len(validTrialInd)
    trialPosx = [None] * len(validTrialInd)

    for i, trialInd in enumerate(validTrialInd):

        if ~np.isnan(trialInd[0]) and ~np.isnan(trialInd[0]):
            trialDur[i] = np.array([visualFrameTS[int(trialInd[0])],visualFrameTS[int(trialInd[1])]])
            indexList = range(int(trialInd[0]),int(trialInd[1]))
            currTrialPosx = np.array([visualFrameTS[indexList]-visualFrameTS[int(trialInd[0])],posx[indexList]]).transpose()
            trialPosx[i] = currTrialPosx

    return validTrialInd, np.array(trialDur), trialPosx



def getTrialEvent(event,
                  trialDur):
    '''
    get the event temporal information, given the [timestamp, event] list of the whole file and the duration of a
    given trial

    :param event: an x by 2 array, each row reprents [timestamp, event] in the whole file
                  can be visualFrames, speed from time alignment function
    :param trialDur: duration of the trial in the whole file

    :return: trialEvent, same structure as event, but only contains the events within the trial, and the timestamps
             are relative to the onset of the trail
    '''

    ind = np.logical_and(event[:,0]>=trialDur[0],event[:,0]<=trialDur[1])
    trialEvent = event[ind,:]

    if trialEvent.size == 0:
        trialEvent = None
    else:
        trialEvent[:,0] = trialEvent[:,0] - trialDur[0]

    return trialEvent


def getTrialEvent2(event,
                   trialDur):
    '''
    get the event temporal information, given the timestamp list of the whole file and the duration of a
    given trial

    :param event: 1-d array, each value represents timestamp in the whole file
                  can be rewardOnsetTS, movFrameTS, bodyMovTS, eyeMovTS from time alignment function
    :param trialDur: duration of the trial in the whole file

    :return: trialEvent, [timestamp, index], but only contains the events within the trial, and the timestamps
             are relative to the onset of the trail
    '''

    trialEvent = [np.array([event[i],i]) for i in range(len(event)) if \
                  np.logical_and(event[i]>=trialDur[0],event[i]<=trialDur[1])]

    if len(trialEvent) == 0:
        trialEvent = None
    else:
        trialEvent = np.array(trialEvent)
        trialEvent[:,0] = trialEvent[:,0] - trialDur[0]

    return trialEvent


def getTrialLick(lick,
                 trialDur):

    '''
    get the trialLick information, given the licks of the whole file and the duration of a given trial
    '''

    trialLick = []

    for i, currLick in enumerate(lick):

        if currLick[0] < trialDur[0] and currLick[1] > trialDur[0]:
            trialLick.append(np.array([trialDur[0],currLick[1]]))

        if currLick[0] < trialDur[1] and currLick[1] > trialDur[1]:
            trialLick.append(np.array([currLick[0],trialDur[1]]))

        if currLick[0] >= trialDur[0] and currLick[1] <= trialDur[1]:
            trialLick.append(currLick)

    if len(trialLick) == 0:
        trialLick = None
    else:
        trialLick = np.array(trialLick) - trialDur[0]

    return trialLick

def getTrialSpeed(speed,
                  speedFs,
                  trialDur):

    trialSpeedFs = speedFs
    trialSpeedInterval = 1./speedFs

    startInd = int(np.round(trialDur[0] // trialSpeedInterval + 1))
    endInd = int(np.round(trialDur[1] // trialSpeedInterval + 1))

    trialSpeed = speed[startInd:endInd]

    trialSpeedOnset = startInd * trialSpeedInterval - trialDur[0]

    return trialSpeed, trialSpeedFs, trialSpeedOnset

def downSample(ary,
               downSampleRate):

    if len(ary.shape) == 1:
        length = ary.shape[0]
        reind = range(length//downSampleRate)
        newAry = ary[reind*downSampleRate]
    elif len(ary.shape) == 2:
        length = ary.shape[0]
        reind = range(length//downSampleRate)
        newAry = ary[reind*downSampleRate,:]


    return newAry


def getTrials(dataPath, #standard hdf5 temporal and image data from time alignment
              pklPath,
              saveFolder = None,
              mouseType = None,
              fileLabel = None,
              rewardVolumn = 0.005,
              taskType = 'SimpleDetction',
              visualStimType = 'StationaryGrating',
              speedDownSampleRate = 100,
              speedGain = 0.28,
              monitorPosition = (-96, 96),
              windowPosition = (-25, 25),
              ):


    dataFile = h5py.File(dataPath)
    df = pkl2df(pklPath)

    mouseID = df.loc[0,'mouse'][1:]
    date = df.loc[0,'date']
    dateRecorded = int(date[0:4]+date[5:7]+date[8:10])

    visualFrames = dataFile['visualFrames'].value
    bodyMovTS = dataFile['bodyMovTS'].value
    eyeMovTS = dataFile['eyeMovTS'].value
    lick = dataFile['lickEpoch'].value
    movFrameTS = dataFile['movFrameTS'].value
    reward = dataFile['rewardOnsetTS'].value
    speed = dataFile['speed'].value
    speedFs = dataFile['speedFs'].value

    #down sample speed
    speed = downSample(speed,speedDownSampleRate).astype(np.float16)
    speedFs = speedFs / speedDownSampleRate

    _, trialDur, trialPosx = getValidTrials(visualFrames = visualFrames,
                                            lapFirstFrame = np.array(df.loc[:,'firstframe']),
                                            monitorSpan = monitorPosition, # monitor coverage azimuth in degrees
                                            )


    trials = {}


    for i, currTrialDur in enumerate(trialDur):

        if not np.isnan(currTrialDur[0]) and not np.isnan(currTrialDur[1]):

            print 'Gathering information for trial', str(i), '...'

            trialID = "%05d" % i

            trialName = str(dateRecorded)+'-M'+mouseID+'-'+str(taskType)+'-'+fileLabel+'-Trial' + trialID

            currPosx = trialPosx[i]
            currLick = getTrialLick(lick, currTrialDur)
            currSpeedArray, currSpeedFs, currSpeedOnset = getTrialSpeed(speed,speedFs,currTrialDur)
            currSpeed = {'array':currSpeedArray,
                         'Fs':currSpeedFs,
                         'onset':currSpeedOnset}
            currReward = getTrialEvent2(reward, currTrialDur)
            currImageSequence = getTrialEvent2(movFrameTS, currTrialDur)

            if eyeMovTS != None:
                currEyeMovSequence = getTrialEvent2(eyeMovTS, currTrialDur)
            else:
                currEyeMovSequence = None

            if bodyMovTS != None:
                currBodyMovSequence = getTrialEvent2(bodyMovTS, currTrialDur)
            else:
                currBodyMovSequence = None


            currIsTarget = df.loc[i,'istarget']
            currSelectionTime = df.loc[i,'selectiontime']


            try:
                con = df.loc[i,'Contrast']
            except KeyError:
                con = 0.64

            try:
                ori = df.loc[i,'Ori']
            except KeyError:
                ori = 0.

            try:
                sf = df.loc[i,'SF']
            except KeyError:
                sf = 0.08

            try:
                size = df.loc[i,'Size']
            except KeyError:
                size = 25.


            currTrial = ForagingTrial(trialNum = i,
                                      duration = currTrialDur,
                                      posx = currPosx,
                                      speed = currSpeed,
                                      lick = currLick,
                                      reward = currReward,
                                      isTarget = currIsTarget,
                                      imageSequence = currImageSequence,
                                      eyeMovSequence = currEyeMovSequence,
                                      bodyMovSequence = currBodyMovSequence,
                                      taskType = taskType,
                                      selectionTime = currSelectionTime,
                                      rewardVolumn = rewardVolumn,
                                      visualStim = {
                                                    'type':visualStimType ,
                                                    'con':con,
                                                    'sf':sf,
                                                    'size':size,
                                                    'ori':ori
                                                    },
                                      windowPosition = windowPosition,
                                      monitorPosition = monitorPosition,
                                      fileLabel = fileLabel,
                                      mouseID = mouseID,
                                      mouseType = mouseType,
                                      dateRecorded = dateRecorded,
                                      speedGain = speedGain)


            trials.update({trialName:currTrial.generateTrialDict()})

            print 'trial finished.'


    if saveFolder:
        savePath = os.path.join(saveFolder,str(dateRecorded)+'-M'+mouseID+'-'+str(taskType)+'-'+fileLabel+'.pkl')
        ft.saveFile(savePath,trials)

    return trials


def loadTrial(trialDict):

    trialNum = trialDict['trialNum']
    duration = trialDict['duration']
    posx = trialDict['posx']
    lick = trialDict['lick']
    reward = trialDict['reward']
    isTarget = trialDict['isTarget']
    imageSequence = trialDict['imageSequence']
    taskType = trialDict['taskType']

    try:
        speed = trialDict['speed']
    except KeyError:
        speed = None

    try:
        eyeMovSequence = trialDict['eyeMovSequence']
    except KeyError:
        eyeMovSequence = None

    try:
        bodyMovSequence = trialDict['bodyMovSequence']
    except KeyError:
        bodyMovSequence = None

    try:
        selectionTime = trialDict['selectionTime']
    except KeyError:
        selectionTime = None

    try:
        rewardVolumn = trialDict['rewardVolumn']
    except KeyError:
        rewardVolumn = None

    try:
        visualStim = trialDict['visualStim']
    except KeyError:
        visualStim = {
                      'type':'StationaryGrating',
                      'con':0.64,
                      'sf':0.08,
                      'size':25.,
                      'ori':0.
                      }

    try:
        windowPosition = trialDict['windowPosition']
    except KeyError:
        windowPosition = np.array([-25., 25.])

    try:
        monitorPosition = trialDict['monitorPosition']
    except KeyError:
        monitorPosition = np.array([-96., 96.])

    try:
        fileLabel = trialDict['fileLabel']
    except KeyError:
        fileLabel = '000'

    try:
        mouseID = trialDict['mouseID']
    except KeyError:
        mouseID = 'TEST'

    try:
        mouseType = trialDict['mouseType']
    except KeyError:
        mouseType = None

    try:
        dateRecorded = trialDict['dateRecorded']
    except KeyError:
        dateRecorded = 99999999

    try:
        speedGain = trialDict['speedGain']
    except KeyError:
        speedGain = 0.28

    trial = ForagingTrial(trialNum = trialNum,
                          duration = duration,
                          posx = posx,
                          speed = speed,
                          lick = lick,
                          reward = reward,
                          isTarget = isTarget,
                          imageSequence = imageSequence,
                          taskType = taskType,
                          eyeMovSequence = eyeMovSequence,
                          bodyMovSequence = bodyMovSequence,
                          selectionTime = selectionTime,
                          rewardVolumn = rewardVolumn,
                          visualStim = visualStim,
                          windowPosition = windowPosition ,
                          monitorPosition = monitorPosition,
                          fileLabel = fileLabel,
                          mouseID = mouseID,
                          mouseType = mouseType,
                          dateRecorded = dateRecorded,
                          speedGain = speedGain
                          )

    return trial


class ForagingTrial(object):

    def __init__(self,
                 trialNum,
                 duration,
                 posx,
                 speed,
                 lick,
                 reward,
                 isTarget,
                 imageSequence,
                 taskType,
                 eyeMovSequence = None,
                 bodyMovSequence = None,
                 selectionTime = None,
                 rewardVolumn = None,
                 visualStim = {
                               'type':'StationaryGrating',
                               'con':0.64,
                               'sf':0.08,
                               'size':25.,
                               'ori':0.
                               },
                 windowPosition = np.array([-25, 25]),
                 monitorPosition = np.array([-96.,96.]),
                 fileLabel = '000',
                 mouseID = 'TEST',
                 mouseType = None,
                 dateRecorded = 99999999,
                 speedGain = 0.28
                 ):

        self.trialNum = trialNum
        self.duration = duration
        self.posx = posx
        self.speed = speed
        self.lick = lick
        self.reward = reward
        self.isTarget = isTarget
        self.imageSequence = imageSequence
        self.eyeMovSequence = eyeMovSequence
        self.bodyMovSequence = bodyMovSequence
        self.taskType = taskType
        self.selectionTime = selectionTime
        self.rewardVolumn = rewardVolumn
        self.visualStim = visualStim
        self.windowPosition = windowPosition
        self.monitorPosition = monitorPosition
        self.fileLabel = fileLabel
        self.mouseID = mouseID
        self.mouseType = mouseType
        self.dateRecorded = dateRecorded
        self.speedGain = speedGain


    def generateTrialDict(self,
                          keysToRetain=('trialNum',
                                        'duration',
                                        'posx',
                                        'speed',
                                        'lick',
                                        'reward',
                                        'isTarget',
                                        'imageSequence',
                                        'taskType',
                                        'eyeMovSequence',
                                        'bodyMovSequence',
                                        'selectionTime',
                                        'rewardVolumn',
                                        'visualStim',
                                        'windowPosition',
                                        'monitorPosition',
                                        'fileLabel',
                                        'mouseID',
                                        'mouseType',
                                        'dateRecorded',
                                        'speedGain')
                          ):

        trialDict = {}

        for i, key in enumerate(keysToRetain):

            trialDict.update({key:self.__dict__[key]})

        return trialDict


    def getRunningDistance(self):
        '''
        the real runing distance of the animal during the trial (arc)
        '''

        pass


if __name__ == '__main__':
    dataPath = r"E:\data2\2014-12-03-BehaviorDataAnalysis\testdataset\processed\20140828-M147861-SimpleDetection-107.hdf5"
    pklPath = r"E:\data2\2014-12-03-BehaviorDataAnalysis\testdataset\141203124343-141203_detection-M147861-SimpleDetection-Jun-107.pkl"
    saveFolder = r"E:\data2\2014-12-03-BehaviorDataAnalysis\testdataset\processed"

    dataFile = h5py.File(dataPath)
    df = pkl2df(pklPath)


    mouseType = 'Emx1-IRES-Cre;Ai96(GCaMP6s)'
    filelabel = dataPath[-8:-5]
    rewardVolumn = 0.005
    selectionTime = 1.0


    trials = getTrials(dataPath = dataPath, #standard hdf5 temporal and image data from time alignment
                       pklPath = pklPath,
                       saveFolder = saveFolder,
                       mouseType = mouseType,
                       fileLabel = filelabel,
                       rewardVolumn = rewardVolumn,
                       taskType = 'SimpleDetection',
                       visualStimType = 'StationaryGrating',
                       speedDownSampleRate = 100,
                       speedGain = 0.28,
                       monitorPosition = np.array([-96., 96.]),
                       windowPosition = np.array([-25., 25.]),
                       )


    trials2 = ft.loadFile(r"E:\data2\2014-12-03-BehaviorDataAnalysis\testdataset\processed\20141203-M147861-SimpleDetection-107.pkl")
    trial = loadTrial(trials2[trials2.keys()[0]])

    print trial.__dict__.keys()


    print 'for debug...'



