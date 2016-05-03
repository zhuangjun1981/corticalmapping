import corticalmapping.VisualStim2 as vs
import matplotlib.pyplot as plt


mouseID = '225835' #'147861' #'TEST'
userID = 'Natalia'
numOfTrials = 3 # 20

gridSpace=(10.,10.) #(alt,azi)
probeSize=(10.,10.) #size of flicker probes (width,height)
probeOrientation=0. #orientation of flicker probes
probeFrameNum=6 #how many frames each probe is displayed
subregion=[-15, 50, -20, 70] #mapping region
sign='ON-OFF' # 'ON', 'OFF' or 'ON-OFF'
iteration=1
preGapDur=0.
postGapDur=0.


isTriggered = True



psychopyMonitor = 'smartTVgamma' #'smartTVgamma'

logFolder = r'C:\data'
backupFolder = r'\\W7DTMJ03jgl2\data'

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


SparseNoise=vs.SparseNoise(mon,
                           indicator,
                           coordinate='degree', #'degree' or 'linear'
                           background=0., #back ground color [-1,1]
                           gridSpace=gridSpace, #(alt,azi)
                           probeSize=probeSize, #size of flicker probes (width,height)
                           probeOrientation=probeOrientation, #orientation of flicker probes
                           probeFrameNum=probeFrameNum,
                           subregion=subregion,
                           sign=sign,
                           iteration=iteration,
                           preGapDur=preGapDur,
                           postGapDur=postGapDur)


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

ds.setStim(SparseNoise)

ds.triggerDisplay()

plt.show()
