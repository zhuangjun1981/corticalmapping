import matplotlib.pyplot as plt
import corticalmapping.VisualStim2 as vs


mouseID = '228431' #'147861' #'TEST'
userID = 'Naveen'
numOfTrials = 20 # 20
isTriggered = True

psychopyMonitor = 'smartTVgamma' #'smartTVgamma'

logFolder = r'C:\data'
backupFolder = r'\\W7DTMJ38BBB\data'

isRemoteSync = False
remoteSyncIP = 'localhost'
remoteSyncPort = 11001
syncOutputFolder = None

mon=vs.Monitor(resolution=(1080, 1920),
               dis=15.3,
               monWcm=88.8,
               monHcm=50.1,
               C2Tcm=31.1,
               C2Acm=41.91,
               monTilt=26.56,
               downSampleRate=5)
                  
#mon.plot_map()
#plt.show()                  
                  
indicator=vs.Indicator(mon,
                       width_cm=3.,
                       height_cm=3.,
                       position = 'southeast',
                       isSync=True,
                       freq=1.)

stim = vs.KSstimAllDir(mon,
                       indicator,
                       background=0.,
                       iteration=1,
                       squareSize=25.,
                       sweepWidth=20.,
                       stepWidth=0.15,
                       sweepFrame=1,
                       flickerFrame=10,
                       preGapDur=2.,
                       postGapDur=3.)

ds = vs.DisplaySequence(logdir=logFolder,
                        backupdir=backupFolder,
                        displayIteration=numOfTrials,
                        psychopyMonitor=psychopyMonitor,
                        displayOrder=1,
                        mouseid=mouseID,
                        userid=userID,
                        isInterpolate=False,
                        isRemoteSync=isRemoteSync,
                        remoteSyncIP=remoteSyncIP,
                        remoteSyncPort=remoteSyncPort,
                        syncOutputFolder=syncOutputFolder,
                        isTriggered=isTriggered,
                        triggerNIDev='Dev1',
                        triggerNIPort=1,
                        triggerNILine=0,
                        triggerType="NegativeEdge",
                        isSyncPulse=True,
                        syncPulseNIDev='Dev1',
                        syncPulseNIPort=1,
                        syncPulseNILine=1,
                        displayScreen=1,
                        initialBackgroundColor=0.,
                        isVideoRecord=True,
                        videoRecordIP='w7dtmj007lhu',
                        videoRecordPort=10000,
                        displayControlIP = 'localhost',
                        displayControlPort = 10002,
                        fileNumNIDev = 'Dev1',
                        fileNumNIPort = 0,
                        fileNumNILines = '0:7')

ds.set_stim(stim)

ds.trigger_display()

plt.show()
