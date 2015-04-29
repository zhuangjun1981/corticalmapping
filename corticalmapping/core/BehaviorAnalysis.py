__author__ = 'junz'

import FileTools as ft
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ni
import scipy.stats as stats
import h5py
import os

try: import cv2
except ImportError as e: print e

###
### refer to IPython notebook "C:\JunZhuang\labwork\data\ipythonNotebook\WideFieldImaging_behavior_timealignment.ipynb"
###


def getFrameTS(JPhys,
               pklDict,
               digitalVoltage = 5.,
               vSyncThr = 3.5,
               JPhysFs = 10000.,
               pdrawThr = 2.,
               pdThr = 0.03,
               pdSmoothFilterLen = 0.005):

    pd = JPhys['photodiode2'] # photodiode
    vSync = JPhys['visualFrame'] # visualFrame
    # t = np.arange(len(pd))/JPhysFs

    vSync[vSync<vSyncThr] = 0
    vSync[vSync>=vSyncThr] = digitalVoltage

    diffvSync = np.diff(vSync)

    vTS = np.argwhere(diffvSync>vSyncThr)/JPhysFs
    vTS = np.squeeze(vTS)

    # print "Total visual frame number:", len(vTS)

    syncSqrFrameNum = pklDict['syncsqrfreq'] * len(pklDict['syncsqrcolorsequence'])

    # print 'Total visual frame number:', pklDict['vsynccount']
    # print 'Sync square period frame:', syncSqrFrameNum
    # print 'Sync square color seqency', pklDict['syncsqrcolorsequence']

    # first visual frame number of the onset of white sync square
    firstFrameNum = np.squeeze(np.argwhere(np.array(pklDict['syncsqrcolorsequence'])==1)[0]) * pklDict['syncsqrfreq']
    syncSqrOnsetFrame = np.arange(firstFrameNum, pklDict['vsynccount'], syncSqrFrameNum)

    # print '\n\n\nNumber of vSync onsets:', len(syncSqrOnsetJPhysTime),'\n\n'
    # print '\nFirst ten JPhys times of the onset of white sync square:', syncSqrOnsetJPhysTime[0:10],'\n\n'

    # digitize photodiode signal
    pd[pd>=pdrawThr] = digitalVoltage
    pd[pd<pdrawThr] = 0

    # filter photodiode signal
    pdFiltered = ni.filters.gaussian_filter(pd, pdSmoothFilterLen*JPhysFs)

    # derivative of photodiode filted
    pdFilteredDiff = np.diff(pdFiltered)
    pdFilteredDiff = np.hstack(([0],pdFilteredDiff))

    # photodiode signal used to detect onset
    pdSignal = np.multiply(pd, pdFilteredDiff)

    # get photodiode onset JPhys time
    pdOnset = []
    for i in range(1,len(pdSignal)):
        if pdSignal[i] > pdThr and \
           pdSignal[i-1]  < pdThr:

                pdOnset.append(i * (1. / JPhysFs))

    pdOnset = np.array(pdOnset)

    # print '\n\n\nNumber of photodiode onset:', len(pdOnset),'\n\n'
    # print '\n First ten photodiode onset JPhys time:', pdOnset[0:10],'\n\n'

    # check length of syncSqrOnsetJPhysTime, pdOnset
    print '\n\nNumber of sync square onset recorded by behavior pkl file:', len(syncSqrOnsetFrame)
    
    print 'photodiode onset times, Jun extraction', vTS[syncSqrOnsetFrame[:5]]

    print 'Number of photodiode onset recorded by JPhys file:', len(pdOnset)

    #get the frame number of sync square onset which was recorded by JPhys
    syncSqrOnsetFrame = [x for x in syncSqrOnsetFrame if x<len(vTS)]

    # print '\nFirst ten visual frames of the onset of white sync square:',syncSqrOnsetFrame[0:10]
    syncSqrOnsetJPhysTime = vTS[syncSqrOnsetFrame]

    # check the number of sync square onsets getting from behavior pkl and the number of recorded photodiode onsets by
    # JPhys
    if len(syncSqrOnsetJPhysTime)!=len(pdOnset):
        print 'buggy_line', len(syncSqrOnsetJPhysTime), len(pdOnset)
        raise ValueError, ''' The number of sync square onsets detected by the photodiode in JPhys is different from the
        number of sync square onsets recorded in behavior pkl file which are within the duration of JPhys file.'''

    slope, intercept, r_value, p_value, stderr = stats.linregress(syncSqrOnsetJPhysTime,pdOnset)
    
    f = plt.figure(figsize=(18,5))
    plt.plot(pdOnset-syncSqrOnsetJPhysTime)
    plt.xlim([0,20])

    print 'intercept:', intercept, 'sec'
    print 'slope:', slope, 'sec/sec'

    frameTS = vTS * slope + intercept

    # print '\n\nFirst ten visual display time stamps in JPhys time:\n', frameTS[0:10]
    # print '\n\nAverage frame duration:', np.mean(np.diff(frameTS)),'\n\n'

    posx = pklDict['posx']

    # Check posx length
    print "\n\nTotal visual frame number recorded by 'vsync signal' in JPhys file:", len(frameTS)
    print "Total visual frame number recorded by 'posx' in behavior pkl file:", len(posx)

    posxLen = np.min([len(frameTS),len(posx)])
    posx = posx[0:posxLen]
    frameTS = frameTS[0:posxLen]

    # frames = np.array([frameTS[0:-1],posx]).reshape(len(posx),2)
    visualFrames = np.array([frameTS,posx])

    return visualFrames.transpose()


def getReward(JPhys,
              rewardThr = 3.5,
              digitalVoltage = 5.,
              JPhysFs = 10000.):

    reward = JPhys['reward']

    reward[reward>=rewardThr]=digitalVoltage
    reward[reward<rewardThr]=0.
    reward = digitalVoltage-reward

    rewardDiff = np.diff(reward)
    rewardOnsetTS = np.argwhere(rewardDiff>rewardThr)/JPhysFs
    rewardOnsetTS = np.squeeze(rewardOnsetTS)

    # print '\n\nNumber of rewards:', len(rewardOnsetTS)
    # print '\n\nReward onset time stamps in Jphys time:\n',rewardOnsetTS,'\n\n'

    return rewardOnsetTS


def getLicking(JPhys,
               lickThr = 3.5,
               digitalVoltage = 5.,
               JPhysFs = 10000.,
               lickLengthThr = (0.005, 0.2) # threshold of lick duration, for removing artificial lick epochs
               ):


    lick = JPhys['licking']

    #digitize lick signal
    lick[lick>=lickThr]=digitalVoltage
    lick[lick<lickThr]=0.

    lick[0]=0.
    lick[-1]=0.

    lickDiff = np.diff(lick)
    lickOnsetTS = np.argwhere(lickDiff>lickThr)/JPhysFs
    lickOffsetTS = np.argwhere(lickDiff<-lickThr)/JPhysFs

    # print 'number of lickOnsets:', len(lickOnsetTS)
    # print 'number of lickOffset:', len(lickOffsetTS)

    lickEpoch = np.concatenate((lickOnsetTS,lickOffsetTS),axis=1)

    lickEpoch = [x for x in lickEpoch if np.logical_and((x[1]-x[0]>lickLengthThr[0]),(x[1]-x[0]<lickLengthThr[1]))]

    lickEpoch = np.array(lickEpoch)

    # print '\n\nNumber of licks:', lickEpoch.shape[0]
    # print '\n\nFirst ten licking time stamps in Jphys time:\n',lickEpoch[0:10],'\n\n'

    # plt.figure(figsize=(20,5))
    # plt.hist(np.squeeze(lickOffsetTS)-np.squeeze(lickOnsetTS),bins=50,range=lickLengthThr)
    # plt.xlim(lickLengthThr)

    return lickEpoch

def getSpeed(JPhys,
             runningThr = 0.01,
             digitalVoltage = 5.,
             speedSmoothFilterLen = 0.01,
             JPhysFs = 10000.):


    running = JPhys['runningSig'] - JPhys['runningRef'] + digitalVoltage

    #get raw speed
    speed = np.diff(running)
    speed = np.hstack(([0.],speed))

    #interpolate abrupt change in speed signal
    speed = np.array(speed)
    speed[np.logical_or(speed>=runningThr,speed<=-runningThr)]=np.nan
    mask = np.isnan(speed)
    speed[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), speed[~mask])

    #smooth
    speed = ni.filters.gaussian_filter1d(speed,speedSmoothFilterLen*JPhysFs)
    speed = speed * 2. * np.pi * JPhysFs/ digitalVoltage # (arc / second)

    speedFs = JPhysFs

    return speed.astype(np.float16), speedFs


def getMovFrameTS(JPhys,
                  imageNum,
                  readThr = 3.5,
                  digitalVoltage = 5.,
                  JPhysFs = 10000.):

    read = JPhys['read']

    #digitize read signal
    read[read>=readThr]=digitalVoltage
    read[read<readThr]=0.

    readDiff = np.diff(read)

    movFrameTS = np.argwhere(readDiff>readThr)/JPhysFs
    movFrameTS = np.squeeze(movFrameTS)[0:imageNum]

    return movFrameTS


def getAVIFrameTS(txtPath,
                  aviPath,
                  startingTime):
    '''
    temporary function to get time stamps of eye movie and body movie from the .txt and .avi movie recorded by
    Derric's eyetracker software. temporal precision is not very high.
    '''

    with open(txtPath, 'r') as f:
        TS = f.read()

    TS = TS.split('\n')

    TSarray = np.array([float(x) for x in TS if x!=''])

    TSarray = TSarray - TSarray[0] + startingTime

    if len(TSarray.shape) != 1:
        raise ValueError, '.avi movie time stamps are not one dimensional!'

    vc = cv2.VideoCapture(aviPath)
    frameNum = int(vc.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    if int(TSarray.shape[0]) != frameNum:
        raise ValueError, 'Number of time stamps in .txt file and number of frames in .avi file do not match!!'

    with open(aviPath, 'rb') as f2:
        aviMovie = f2.read()

    return TSarray, aviMovie


def behaviorTimeAlignment(JPhysPath,
                          pklPath,
                          JCamPath,
                          eyeTXTPath,
                          eyeAVIPath,
                          bodyTXTPath,
                          bodyAVIPath,
                          savePath = None,
                          dtype = np.dtype('>f'),
                          headerLength = 96,
                          channels = ('photodiode2',
                                      'read',
                                      'trigger',
                                      'photodiode',
                                      'sweep',
                                      'visualFrame',
                                      'runningRef',
                                      'runningSig',
                                      'reward',
                                      'licking'),
                          JPhysFs = 10000.,
                          digitalVoltage = 5.,
                          vSyncThr = 3.5,
                          pdThr = 0.03,
                          pdSmoothFilterLen = 0.005,
                          rewardThr = 3.5,
                          lickThr = 3.5,
                          lickLengthThr = (0.005, 0.2),
                          runningThr = 0.01,
                          speedSmoothFilterLen = 0.05,
                          columnNumIndex = 14, # index of number of rows in header
                          rowNumIndex = 15, # index of number of columns in header
                          frameNumIndex = 16, # index of number of frames in header
                          decimation = None,
                          readThr = 3.5 #threshold for camera read signal
                          ):

    _, JPhys = ft.importRawNewJPhys(path = JPhysPath,
                                    dtype = dtype,
                                    headerLength = headerLength,
                                    channels = channels,
                                    sf = JPhysFs)

    pklDict = ft.loadFile(pklPath)

    visualFrames = getFrameTS(JPhys,
                              pklDict,
                              digitalVoltage = digitalVoltage,
                              vSyncThr = vSyncThr,
                              JPhysFs = JPhysFs,
                              pdThr = pdThr ,
                              pdSmoothFilterLen = pdSmoothFilterLen)

    rewardOnsetTS = getReward(JPhys,
                    rewardThr = rewardThr,
                    digitalVoltage = digitalVoltage,
                    JPhysFs = JPhysFs)

    lickEpoch = getLicking(JPhys,
                             lickThr = lickThr,
                             digitalVoltage = digitalVoltage,
                             lickLengthThr=lickLengthThr,
                             JPhysFs = JPhysFs)

    speed, speedFs = getSpeed(JPhys,
                              runningThr = runningThr,
                              digitalVoltage = digitalVoltage,
                              speedSmoothFilterLen = speedSmoothFilterLen,
                              JPhysFs = JPhysFs)

    mov,_=ft.importRawJCam(JCamPath,
                           dtype = dtype,
                           headerLength = headerLength, # length of the header, measured as the data type defined above
                           columnNumIndex = columnNumIndex, # index of number of rows in header
                           rowNumIndex = rowNumIndex, # index of number of columns in header
                           frameNumIndex = frameNumIndex, # index of number of frames in header
                           decimation = decimation)


    movFrameTS = getMovFrameTS(JPhys,
                               imageNum = mov.shape[0],
                               readThr = readThr,
                               digitalVoltage = digitalVoltage,
                               JPhysFs = JPhysFs)

    if eyeTXTPath and eyeAVIPath:
        eyeMovTS, eyeMovie = getAVIFrameTS(txtPath = eyeTXTPath,
                                           aviPath = eyeAVIPath,
                                           startingTime = visualFrames[0,0])
    else:
        eyeMovTS = None
        eyeMovie = None

    if bodyTXTPath and bodyAVIPath:
        bodyMovTS, bodyMovie = getAVIFrameTS(txtPath = bodyTXTPath,
                                             aviPath = bodyAVIPath,
                                             startingTime = visualFrames[0,0])
    else:
        bodyMovTS = None
        bodyMovie = None




    print '\n\nNumber of visual frames:', len(visualFrames)
    print '\n\nFirst ten visual frames [timse stamp, posx]:\n',visualFrames[0:10,:]

    print '\n\nNumber of rewards:', len(rewardOnsetTS)
    print '\n\nTime stamps for first ten rewards:\n',rewardOnsetTS[0:10]

    print '\n\nNumber of licks:', len(lickEpoch)
    print '\n\nTime epochs for first ten licks [onset, offset]:\n',lickEpoch[0:10]

    print '\n\nAverage running speed:', np.mean(speed),'arc/sec'
    print '\n\nFirst ten time points for running speed:\n', np.array([np.arange(10)/speedFs,speed[0:10]]).transpose()

    print '\n\nNumber of image frames:', len(movFrameTS)
    print '\n\nTime stamps for first ten image frames:\n',movFrameTS[0:10]

    if eyeMovTS != None:
        print '\n\nNumber of eye movie frames:', len(eyeMovTS)
        print '\n\nTime stamps for first ten eye movie frames:\n',eyeMovTS[0:10]

    if bodyMovTS != None:
        print '\n\nNumber of body movie frames:', len(bodyMovTS)
        print '\n\nTime stamps for first ten body movie frames:\n',bodyMovTS[0:10]

    if savePath:
        f = h5py.File(savePath,'w')
        f.create_dataset('mov',data=mov)
        f.create_dataset('movFrameTS',data=movFrameTS)
        f.create_dataset('visualFrames',data=visualFrames)
        f.create_dataset('rewardOnsetTS',data=rewardOnsetTS)
        f.create_dataset('lickEpoch',data=lickEpoch)
        f.create_dataset('speed',data=speed)
        f.create_dataset('speedFs',data=speedFs)
        f.create_dataset('eyeMovTS',data=eyeMovTS)
        f.create_dataset('eyeMovie',data=np.void(eyeMovie))
        f.create_dataset('bodyMovTS',data=bodyMovTS)
        f.create_dataset('bodyMovie',data=np.void(bodyMovie))
        f.close()

    return mov, movFrameTS, visualFrames, rewardOnsetTS, lickEpoch, speed, speedFs, eyeMovTS, eyeMovie, bodyMovTS, bodyMovie




if __name__ == '__main__':

    mouseID = '147861'
    mouseType = 'Emx1-IRES-Cre;Ai96(GCaMP6s)'
    dateRecorded = 20140828
    taskType = 'SimpleDetection'
    fileLabel = '107'



    JPhysPath = r"E:\data2\2014-12-03-BehaviorDataAnalysis\testdataset\141203JPhys107"
    pklPath = r"E:\data2\2014-12-03-BehaviorDataAnalysis\testdataset\141203124343-141203_detection-M147861-SimpleDetection-Jun-107.pkl"
    JCamPath = r"E:\data2\2014-12-03-BehaviorDataAnalysis\testdataset\141203JCam107"

    eyeTXTPath = r"E:\data2\2014-12-03-BehaviorDataAnalysis\testdataset\141203122951-SimpleDetection-mouse147861-Jun-107-avt.txt"
    eyeAVIPath = r"E:\data2\2014-12-03-BehaviorDataAnalysis\testdataset\141203122951-SimpleDetection-mouse147861-Jun-107-avt.avi"

    bodyTXTPath = r"E:\data2\2014-12-03-BehaviorDataAnalysis\testdataset\141203122951-SimpleDetection-mouse147861-Jun-107-0.txt"
    bodyAVIPath = r"E:\data2\2014-12-03-BehaviorDataAnalysis\testdataset\141203122951-SimpleDetection-mouse147861-Jun-107-0.avi"

    saveFolder = r"E:\data2\2014-12-03-BehaviorDataAnalysis\testdataset\processed"
    savePath = None #os.path.join(saveFolder,str(dateRecorded)+'-M'+mouseID+'-'+taskType+'-'+fileLabel+'.hdf5')


    dtype = np.dtype('>f')
    headerLength = 96 # length of the header for each channel
    channels = ['photodiode2',
                'read',
                'trigger',
                'photodiode',
                'sweep',
                'visualFrame',
                'runningRef',
                'runningSig',
                'reward',
                'licking']
    JPhysFs = 10000.

    digitalVoltage = 5.
    digitalThr = 3.0
    runningThr = 0.01
    pdThr = 0.03
    lickLengthThr = (0.005, 0.2)
    pdSmoothFilterLen = 0.005 # second
    speedSmoothFilterLen = 0.01 # second


    columnNumIndex = 14 # index of number of rows in header
    rowNumIndex = 15 # index of number of columns in header
    frameNumIndex = 16 # index of number of frames in header
    decimation = None #decimation number

    everything = behaviorTimeAlignment(JPhysPath = JPhysPath,
                                       pklPath = pklPath,
                                       JCamPath = JCamPath,
                                       eyeTXTPath = eyeTXTPath,
                                       eyeAVIPath = eyeAVIPath,
                                       bodyTXTPath = bodyTXTPath,
                                       bodyAVIPath = bodyAVIPath,
                                       savePath = savePath,
                                       dtype = dtype,
                                       headerLength = headerLength,
                                       channels = channels,
                                       JPhysFs = JPhysFs,
                                       digitalVoltage = digitalVoltage,
                                       vSyncThr = digitalThr,
                                       pdThr = pdThr,
                                       pdSmoothFilterLen = pdSmoothFilterLen,
                                       rewardThr = digitalThr,
                                       lickThr = digitalThr,
                                       lickLengthThr = lickLengthThr,
                                       runningThr = digitalThr,
                                       speedSmoothFilterLen = speedSmoothFilterLen,
                                       columnNumIndex = columnNumIndex,
                                       rowNumIndex = rowNumIndex,
                                       frameNumIndex = frameNumIndex,
                                       decimation = decimation,
                                       readThr = digitalThr)


    mov, movFrameTS, visualFrames, rewardOnsetTS, lickEpoch, speed, speedFs, eyeMovTS, eyeMovie, bodyMovTS, bodyMovie = everything

    plt.show()










