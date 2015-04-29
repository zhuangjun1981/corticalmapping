# -*- coding: utf-8 -*-
"""
Created on Wed Apr 08 12:49:08 2015

@author: junz
"""

import corticalmapping.core.VisualStim as vs
import corticalmapping.core.RetinotopicMapping as rm
import matplotlib.pyplot as plt
import numpy as np

downSampleRate=5

mon=vs.MonitorJun(resolution=(1080, 1920),
                  dis=13.5,
                  monWcm=88.8,
                  monHcm=50.1,
                  C2Tcm=33.1,
                  C2Acm=46.4,
                  monTilt=16.22,
                  downSampleRate=downSampleRate)
                  
print min(mon.degCorX.flatten())
print max(mon.degCorX.flatten())

print min(mon.degCorY.flatten())
print max(mon.degCorY.flatten())
                  
screen = rm.Patch(np.ones((1080/downSampleRate,1920/downSampleRate)).astype(np.uint8),sign=1)

coverage, _, _, _=screen.getVisualSpace(mon.degCorY,mon.degCorX,isplot=True)

plt.show()