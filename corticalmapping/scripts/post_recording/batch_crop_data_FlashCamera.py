# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 14:41:44 2015

@author: junz
"""

import os
import numpy as np
import corticalmapping.core.FileTools as ft

mouseID = '146303'
dateRecorded = 150122
vasMapFileNumList = range(116,122)
imageFileNumList = range(122,128) + range(129,134) + [135] + range(138,143)

movieBinning = 1

vMapCrop = np.array([550, 1550, 700, 1700])

decimation = 'deci1011'

column = 2048
row = 2048 
dtype = np.dtype('<u2') 
headerLength = 116
tailerLength = 218

movieCrop = vMapCrop / movieBinning

dataFolder = os.path.join(r'\\WATERSRAID\Data\FlashData',str(dateRecorded)+'-M'+mouseID)
    
vasMapFileList = [str(dateRecorded)+'JCamF'+str(x) for x in vasMapFileNumList]

imageFileList = [str(dateRecorded)+'JCamF'+str(x)+decimation for x in imageFileNumList]  

for i, currVasMap in enumerate(vasMapFileList):
    currVasMapPath = os.path.join(dataFolder,currVasMap)
    mov, header, tailer = ft.importRawJCamF(path=currVasMapPath,
                                            saveFolder = dataFolder,
                                            dtype = dtype,
                                            headerLength = headerLength,
                                            tailerLength = tailerLength,
                                            column = column,
                                            row = row,
                                            crop=vMapCrop)
    
del i, currVasMap, currVasMapPath, mov, header, tailer
    
for i, currImgFile in enumerate(imageFileList):    
    currImgFilePath = os.path.join(dataFolder,currImgFile)
    mov, _, _ = ft.importRawJCamF(path=currImgFilePath,
                                  saveFolder = dataFolder,
                                  dtype = dtype,
                                  headerLength = 0,
                                  tailerLength = 0,
                                  column = column/movieBinning,
                                  row = row/movieBinning,
                                  crop=movieCrop)
                             
del i, currImgFile, currImgFilePath, mov