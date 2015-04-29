# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 15:03:04 2014

@author: junz
"""

import corticalmapping.core.VisualStim as vs
import matplotlib.pyplot as plt



mon=vs.MonitorJun(resolution=(1080, 1920),
                  dis=13.5,
                  monWcm=88.8,
                  monHcm=50.1,
                  C2Tcm=33.1,
                  C2Acm=46.4,
                  monTilt=16.22,
                  downSampleRate=5)

indicator=vs.IndicatorJun(mon,
                          width_cm = 3., 
                          height_cm = 3., 
                          position = 'northeast',
                          isSync = True,
                          freq = 1.)
                          
                          
flashCircleStim=vs.FlashCircle(
                               mon,
                               indicator,
                               center = (50., 10.), # center coordinate of the circle (degree)
                               radius = 5, # radius of the circle
                               color = -1, # color of the circle [-1: 1]
                               iteration=1, # total number of flashes
                               flashFrame=3, # frame number for display circle of each flash
                               preGapFrame=120, # gap frame number before flash
                               postGapFrame=180, # gap frame number after flash
                               background = 0.)
                               


ds= vs.DisplaySequence(
                       logdir=r'C:\data',
                       backupdir=r'\\aibsdata2\nc-ophys\CorticalMapping',
                       displayIteration = 20,
                       displayOrder = 1, # 1: the right order; -1: the reverse order
                       mouseid = 'test',
                       userid = 'Jun',
                       isVideoRecord = True,
                       isTriggered = True,
                       triggerNIDev = 'Dev1',
                       triggerNIPort = 1,
                       triggerNILine = 3,
                       triggerType = "NegativeEdge", # should be one of "NegativeEdge", "PositiveEdge", "HighLevel", or "LowLevel"
                       isSyncPulse = True,
                       syncPulseNIDev = 'Dev3',
                       syncPulseNIPort = 1,
                       syncPulseNILine = 2,
                       displayScreen = 0,
                       initialBackgroundColor = 0)

ds.setStim(flashCircleStim)

ds.triggerDisplay()

plt.show()



                      
