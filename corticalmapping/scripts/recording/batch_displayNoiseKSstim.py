__author__ = 'junz'

import corticalmapping.core.VisualStim as vs
from corticalmapping.core.FileTools import loadFile
import numpy as np
import matplotlib.pyplot as plt
import os

displayDirection = 'R2L' # 'B2U','U2B','L2R','R2L'
saveFolder = r'C:\data\NoiseKSstim'


mov = np.load(os.path.join(saveFolder,'NoiseKSstim-'+displayDirection+'.npy'))
log = loadFile(os.path.join(saveFolder,'NoiseKSstim-'+displayDirection+'.pkl'))


ds = vs.DisplaySequence(logdir = r'C:\data',
                        backupdir = r'\\aibsdata2\nc-ophys\CorticalMapping',
                        displayIteration = 10,
                        displayOrder = 1, # 1: the right order; -1: the reverse order
                        mouseid = 'Test',
                        userid = 'Jun',
                        isVideoRecord = True,
                        isTriggered = True,
                        triggerNIDev = 'Dev1',
                        triggerNIPort = 0,
                        triggerNILine = 0,
                        isSyncPulse = True,
                        syncPulseNIDev = 'Dev3',
                        syncPulseNIPort = 1,
                        syncPulseNILine = 2,
                        triggerType = "NegativeEdge", # should be one of "NegativeEdge", "PositiveEdge", "HighLevel", or "LowLevel"
                        displayScreen = 1,
                        initialBackgroundColor = 0)

ds.setAnyArray(mov, logDict = log)
ds.triggerDisplay()
plt.show()
