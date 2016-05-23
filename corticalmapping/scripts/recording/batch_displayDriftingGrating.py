import corticalmapping.VisualStim2 as vs
import matplotlib.pyplot as plt
import numpy as np



mouseID = '225835' #'147861' #'TEST'
userID = 'Natalia'
numOfTrials = 3 # 20

sf_list=(0.04,) # (0.16,0.08,0.04), spatial frequency, cycle/unit
tf_list=(0.5,1.,2.,4.,8)  # (15.,4.,0.5), temporal frequency, Hz
dire_list=np.arange(0,2*np.pi,np.pi/2)  # np.arange(0,2*np.pi,np.pi/2), direction, arc
con_list=(0.7,) # (0.01,0.02,0.05,0.11,0.23,0.43,0.73,0.95), contrast, [0, 1]
size_list=(500.,)  # (1.,2.,5.,10.), radius of the circle, unit defined by self.coordinate
blockDur=4.  # duration of each condition, second
midGapDur=4  # duration of gap between conditions
iteration=3  # iteration of whole sequence

isTriggered = True
isRemoteSync = False

psychopyMonitor = 'smartTVgamma' #'smartTVgamma'

logFolder = r'C:\data'
backupFolder = r'\\W7DTMJ03jgl2\data'
remoteSyncIP = 'w7dtmj19vtx'
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


DriftingGrating=vs.DriftingGratingCircle(mon,
                                         indicator,
                                         background=0., # back ground color [-1,1]
                                         coordinate='degree', # 'degree' or 'linear'
                                         center=(60.,0.), # (azi, alt), unit defined by self.coordinate
                                         sf_list=sf_list, # (0.16,0.08,0.04), spatial frequency, cycle/unit
                                         tf_list=tf_list, # (15.,4.,0.5), temporal frequency, Hz
                                         dire_list=dire_list, # np.arange(0,2*np.pi,np.pi/2), direction, arc
                                         con_list=con_list, # (0.01,0.02,0.05,0.11,0.23,0.43,0.73,0.95), contrast, [0, 1]
                                         size_list=size_list, # (1.,2.,5.,10.), radius of the circle, unit defined by self.coordinate
                                         blockDur=blockDur, # duration of each condition, second
                                         midGapDur=midGapDur, # duration of gap between conditions
                                         iteration=iteration, # iteration of whole sequence
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
                        waitTime=2.,
                        isRemoteSync=isRemoteSync,
                        remoteSyncIP=remoteSyncIP,
                        remoteSyncPort=remoteSyncPort,
                        remoteSyncTriggerEvent="positiveEdge",
                        isTriggered=isTriggered,
                        triggerNIDev='Dev1',
                        triggerNIPort=1,
                        triggerNILine=0,
                        displayTriggerEvent="NegativeEdge",
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

ds.set_stim(DriftingGrating)

ds.trigger_display()

plt.show()
