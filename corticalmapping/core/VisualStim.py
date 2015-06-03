# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 17:12:24 2014

@author: junz
"""


import os
import datetime
from psychopy import visual, event
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
import cPickle as pickle
from random import shuffle

import socket
import tifffile as tf

try: import toolbox.IO.nidaq as iodaq
except ImportError as e:
    print e
    print 'import iodaq from aibs package...'
    try: import aibs.iodaq as iodaq
    except ImportError as er: print er


#def showRGBSequence(path):
#    mov = np.load(path)
#    tf.imshow(mov)
#    plt.show()
    

#def gray2rgb(frame, 
#             backgroundColor = np.array([0,0,0]).astype(np.uint8),
#             foregroundColor = np.array([0,0,255]).astype(np.uint8)):
#    '''
#    convert gray scale frame to an array with RGB format
#    '''
#    
#    flatFrame = frame.flatten()
#    
#    currRGB = np.zeros((len(flatFrame), 3)).astype(np.uint8)
#    
#    for i in xrange(len(flatFrame)):
#        if np.isnan(flatFrame[i]):
#            currRGB[i] = backgroundColor
#        else:
#            currRGB[i] = (flatFrame[i] + 1) / 2 * foregroundColor
#    
#    currRGB = currRGB.reshape([frame.shape[0], frame.shape[1], 3])
#    
#    return currRGB
#    

#def gray2rgb2(frame, 
#              backgroundColor = np.array([0,0,0]).astype(np.uint8),
#              foregroundColor = np.array([0,0,255]).astype(np.uint8)):
#
#    currRGB = np.zeros((np.size(frame, 0), np.size(frame, 1), 3))
#    
#    for i in range(np.size(frame, 0)):
#        for j in range(np.size(frame, 1)):
#            if np.isnan(frame[i,j]):
#                currRGB[i,j,0:3] = backgroundColor
#            else:
#                currRGB[i,j,0:3] = (frame[i,j]+1)/2*foregroundColor
#    
#    return np.uint8(currRGB)



def gaussian(x, mu=0, sig=1):
    
    return np.exp(np.divide(-np.power(x - mu, 2.) , 2 * np.power(sig, 2.)))


def analysisFrames(ts, refreshRate, checkPoint = [0.02, 0.033, 0.05, 0.1]):
    '''
    analyze frame durations. input is the time stamps of each frame and
    the refresh rate of the monitor
    '''
    
    frameDuration = ts[1::] - ts[0:-1]
    plt.figure()
    plt.hist(frameDuration, bins=np.linspace(0.0, 0.05, num=51))
    refreshRate = float(refreshRate)
    
    frameStats = '\n'
    frameStats += 'Total frame number: %d. \n' % (len(ts)-1)
    frameStats += 'Total length of display   : %.5f second. \n' % (ts[-1]-ts[0])
    frameStats += 'Expected length of display: %.5f second. \n' % ((len(ts)-1)/refreshRate)
    frameStats += 'Mean of frame durations: %.2f ms. \n' % (np.mean(frameDuration)*1000)
    frameStats += 'Standard deviation of frame durations: %.2f ms. \n' % (np.std(frameDuration)*1000)
    frameStats += 'Shortest frame: %.2f ms, index: %d. \n' % (min(frameDuration)*1000, np.nonzero(frameDuration==np.min(frameDuration))[0][0])
    frameStats += 'longest frame : %.2f ms, index: %d. \n' % (max(frameDuration)*1000, np.nonzero(frameDuration==np.max(frameDuration))[0][0])
    
    for i in range(len(checkPoint)):
        checkNumber = checkPoint[i]
        frameNumber = len(frameDuration[frameDuration>checkNumber])
        frameStats += 'Number of frames longer than %d ms: %d; %.2f%% \n' % (round(checkNumber*1000), frameNumber, round(frameNumber*10000/(len(ts)-1))/100)
    
    print frameStats
    
    return frameDuration, frameStats
    

def noiseMovie(frameFilter, widthFilter, heightFilter, isplot = False):
    '''
    creating a numpy array with shape [len(frameFilter), len(heightFilter), len(widthFilter)]
    
    this array is random noize filtered by these three filters in Fourier domain
    each pixel of the movie have the value in [-1 1]
    '''
    
    rawMov = np.random.rand(len(frameFilter), len(heightFilter), len(widthFilter))
    
    rawMovFFT = np.fft.fftn(rawMov)
    
    filterX = np.repeat(np.array([widthFilter]), len(heightFilter), axis = 0)
    filterY = np.repeat(np.transpose(np.array([heightFilter])), len(widthFilter), axis = 1)
    
    filterXY = filterX * filterY
    
    for i in xrange(rawMovFFT.shape[0]):
        rawMovFFT[i] = frameFilter[i]* (rawMovFFT[i] * filterXY) 
    
    
#    heightFilter = heightFilter.reshape((len(heightFilter),1))
#    frameFilter = frameFilter.reshape((len(frameFilter),1,1))
#    
#    rawMovFFT = np.multiply(np.multiply(np.multiply(rawMovFFT,widthFilter),heightFilter),frameFilter)
    
    filteredMov = np.real(np.fft.ifftn(rawMovFFT))
    
    rangeFilteredMov = np.amax(filteredMov) - np.amin(filteredMov)    
    noiseMovie = ((filteredMov - np.amin(filteredMov)) / rangeFilteredMov) * 2 - 1
    
    if isplot:
        tf.imshow(noiseMovie, vmin=-1, vmax=1, cmap='gray')
    
    return noiseMovie
        

def generateFilter(length, # length of filter
                   Fs, # sampling frequency
                   Flow, # low cutoff frequency
                   Fhigh, # high cutoff frequency
                   mode = 'box'): # filter mode, '1/f' or 'box'

    '''
    generate one dimensional filter on Fourier domain, with symmetrical structure
    '''
    
    freqs = np.fft.fftfreq(int(length), d = (1./float(Fs)))
    
    filterArray = np.ones(length)
    
    for i in xrange(len(freqs)):
        if ((freqs[i] > 0) and (freqs[i] < Flow) or (freqs[i] > Fhigh)) or \
           ((freqs[i] < 0) and (freqs[i] > -Flow) or (freqs[i] < -Fhigh)):
            filterArray[i] = 0
    
    if mode == '1/f':
        filterArray[1:] = filterArray[1:] / abs(freqs[1:])
        filterArray[0] = 0
        filterArray = (filterArray - np.amin(filterArray)) / (np.amax(filterArray) - np.amin(filterArray))
    elif mode == 'box':
        filterArray[0] = 0
    else: raise NameError, 'Variable "mode" should be either "1/f" or "box"!'
    
    if Flow == 0:
        filterArray[0] = 1
    
    return filterArray


def lookupImage(img, lookupI, lookupJ):
    '''
    generate warpped image from img, using look up talbel: lookupI and lookupJ
    '''
    
    if not img.shape == lookupI.shape:
        raise LookupError, 'The image and lookupI should have same size!!'
        
    if not lookupI.shape == lookupJ.shape:
        raise LookupError, 'The lookupI and lookupJ should have same size!!'

    img2 = np.zeros(img.shape)
    
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            img2[i,j] = img[lookupI[i,j],lookupJ[i,j]]
            
    return img2


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0


def getWarpedFrameWithSquare(degCorX,degCorY,center,width,height,ori,foregroundColor=1,backgroundColor=0):
    '''
    generate a frame (matrix) with single square defined by center, width, height and orientation in degress
    visual degree value of each pixel is defined by degCorX, and degCorY
    dtype = np.float32, color space, -1:black, 1:white

    ori: angle in degree, should be 0~180
    '''

    frame = np.ones(degCorX.shape,dtype=np.float32)*backgroundColor

    if ori < 0. or ori > 180.: raise ValueError, 'ori should be between 0 and 180.'

    k1 = np.tan(ori*np.pi/180.)
    k2 = np.tan((ori+90.)*np.pi/180.)

    disW = np.abs((k1*degCorX - degCorY + center[1] - k1 * center[0]) / np.sqrt(k1**2 +1))
    disH = np.abs((k2*degCorX - degCorY + center[1] - k2 * center[0]) / np.sqrt(k2**2 +1))

    frame[np.logical_and(disW<=width/2.,disH<=height/2.)] = foregroundColor

    return frame


class MonitorJun(object):
    '''
    monitor object created by Jun, has the method "remap" to generate the 
    spherical corrected coordinates in degrees
    '''
    
    def __init__(self, 
                 resolution, 
                 dis, 
                 monWcm, 
                 monHcm, 
                 C2Tcm, 
                 C2Acm, 
                 monTilt, 
                 visualField='right',
                 degCorX=None, 
                 degCorY=None, 
                 name='testMonitor', 
                 gamma=None, 
                 gammaGrid=None, 
                 luminance=None,
                 downSampleRate=10, 
                 refreshRate = 60.):
                     
        if resolution[0] % downSampleRate != 0 or resolution[1] % downSampleRate != 0:           
           raise ArithmeticError, 'Resolution pixel numbers are not divisible by down sampling rate'
        
        self.resolution = resolution
        self.dis = dis
        self.monWcm = monWcm
        self.monHcm = monHcm
        self.C2Tcm = C2Tcm # distance from gaze center to monitor top
        self.C2Acm = C2Acm # distance from gaze center to anterior edge of the monitor
        self.monTilt = monTilt
        self.visualField = visualField
        self.degCorX = degCorX
        self.degCorY = degCorY
        self.name = name
        self.downSampleRate = downSampleRate
        self.gamma = gamma
        self.gammaGrid = gammaGrid
        self.luminance = luminance
        self.refreshRate = 60
        
        #distance form the projection point of the eye to the bottom of the monitor
        self.C2Bcm = self.monHcm - self.C2Tcm
        #distance form the projection point of the eye to the right of the monitor
        self.C2Pcm = self.monWcm - self.C2Acm
        
        resolution=[0,0]        
        resolution[0]=self.resolution[0]/downSampleRate
        resolution[1]=self.resolution[1]/downSampleRate
        
        mapcorX, mapcorY = np.meshgrid(range(resolution[1]), range(resolution[0]))
        
        if self.visualField == "left": 
            mapX = np.linspace(self.C2Acm, -1.0 * self.C2Pcm, resolution[1])
            
        if self.visualField == "right":
            mapX = np.linspace(-1 * self.C2Acm, self.C2Pcm, resolution[1])
            
        mapY = np.linspace(self.C2Tcm, -1.0 * self.C2Bcm, resolution[0])
        oldmapX, oldmapY = np.meshgrid(mapX, mapY, sparse = False)
        
        self.linCorX=oldmapX
        self.linCorY=oldmapY
        
        self.remap()
        
    def setGamma(self, gamma, gammaGrid):
        self.gamma = gamma
        self.gammaGrid = gammaGrid
        
    def setLuminance(self, luminance):
        self.luminance = luminance
        
    def setdownSampleRate(self, downSampleRate):
        
        if self.resolution[0] % downSampleRate != 0 or self.resolution[1] % downSampleRate != 0:
           
           raise ArithmeticError, 'resolutionolution pixel numbers are not divisible by down sampling rate'
        
        self.downSampleRate=downSampleRate
        
        resolution=[0,0]        
        resolution[0]=self.resolution[0]/downSampleRate
        resolution[1]=self.resolution[1]/downSampleRate
        
        mapcorX, mapcorY = np.meshgrid(range(resolution[1]), range(resolution[0]))
        
        if self.visualField == "left": 
            mapX = np.linspace(self.C2Acm, -1.0 * self.C2Pcm, resolution[1])
            
        if self.visualField == "right":
            mapX = np.linspace(-1 * self.C2Pcm, self.C2Pcm, resolution[1])
            
        mapY = np.linspace(self.C2Tcm, -1.0 * self.C2Bcm, resolution[0])
        oldmapX, oldmapY = np.meshgrid(mapX, mapY, sparse = False)
        
        self.linCorX=oldmapX
        self.linCorY=oldmapY
        
        self.remap()
        
        
    def remap(self):
        
        resolution=[0,0]        
        resolution[0]=self.resolution[0]/self.downSampleRate
        resolution[1]=self.resolution[1]/self.downSampleRate
        
        mapcorX, mapcorY = np.meshgrid(range(resolution[1]), range(resolution[0]))
        
        newmapX = np.zeros(resolution,dtype=np.float16)
        newmapY = np.zeros(resolution,dtype=np.float16)
        
        
        for j in range(resolution[1]):
            newmapX[:, j] = (180.0 / np.pi) * np.arctan(self.linCorX[0, j] / self.dis)
            dis2 = np.sqrt(np.square(self.dis) + np.square(self.linCorX[0, j])) #distance from 
            
            for i in range(resolution[0]):
                newmapY[i, j] = (180.0 / np.pi) * np.arctan(self.linCorY[i, 0] / dis2)
                
        self.degCorX = newmapX+90-self.monTilt
        self.degCorY = newmapY
        
    def plotMap(self):
        
        resolution=[0,0]        
        resolution[0]=self.resolution[0]/self.downSampleRate
        resolution[1]=self.resolution[1]/self.downSampleRate
        
        mapcorX, mapcorY = np.meshgrid(range(resolution[1]), range(resolution[0]))
        
        f1 = plt.figure(figsize=(12,5))
        f1.suptitle('Remap monitor', fontsize=14, fontweight='bold')
        
        OMX = plt.subplot(221)
        OMX.set_title('Linear Map X (cm)')
        currfig = plt.imshow(self.linCorX)
        levels1 = range(int(np.floor(self.linCorX.min() / 10) * 10), int((np.ceil(self.linCorX.max() / 10)+1) * 10), 10)
        im1 =plt.contour(mapcorX, mapcorY, self.linCorX, levels1, colors = 'k', linewidth = 2)
#        plt.clabel(im1, levels1, fontsize = 10, inline = 1, fmt='%2.1f')
        f1.colorbar(currfig,ticks=levels1)
        plt.gca().set_axis_off()
        
        OMY = plt.subplot(222)
        OMY.set_title('Linear Map Y (cm)')
        currfig = plt.imshow(self.linCorY)
        levels2 = range(int(np.floor(self.linCorY.min() / 10) * 10), int((np.ceil(self.linCorY.max() / 10)+1) * 10), 10)
        im2 =plt.contour(mapcorX, mapcorY, self.linCorY, levels2, colors = 'k', linewidth = 2)
#        plt.clabel(im2, levels2, fontsize = 10, inline = 1, fmt='%2.2f')
        f1.colorbar(currfig,ticks=levels2)
        plt.gca().set_axis_off()
        
        NMX = plt.subplot(223)
        NMX.set_title('Spherical Map X (deg)')
        currfig = plt.imshow(self.degCorX)
        levels3 = range(int(np.floor(self.degCorX.min() / 10) * 10), int((np.ceil(self.degCorX.max() / 10)+1) * 10), 10)
        im3 =plt.contour(mapcorX, mapcorY, self.degCorX, levels3, colors = 'k', linewidth = 2)
#        plt.clabel(im3, levels3, fontsize = 10, inline = 1, fmt='%2.1f')
        f1.colorbar(currfig,ticks=levels3)
        plt.gca().set_axis_off()
        
        NMY = plt.subplot(224)
        NMY.set_title('Spherical Map Y (deg)')
        currfig = plt.imshow(self.degCorY)
        levels4 = range(int(np.floor(self.degCorY.min() / 10) * 10), int((np.ceil(self.degCorY.max() / 10)+1) * 10), 10)
        im4 =plt.contour(mapcorX, mapcorY, self.degCorY, levels4, colors = 'k', linewidth = 2)
#        plt.clabel(im4, levels4, fontsize = 10, inline = 1, fmt='%2.1f')
        f1.colorbar(currfig,ticks=levels4)
        plt.gca().set_axis_off()
        
    def saveMonitor(self):
        pass
    
    def generate_LookUpTable(self):
        '''
        generate lookup talbe between degree corrdinates and linear corrdinates
        return two matrix: 
        lookupI: i index in linear matrix to this pixel after warping
        lookupJ: j index in linear matrix to this pixel after warping
        '''
        
        #length of one degree on monitor at gaze point
        degDis = np.tan(np.pi / 180) * self.dis
        
        #generate degree coordinate without warpping
        degNoWarpCorX = self.linCorX / degDis
        degNoWarpCorY = self.linCorY / degDis
        
        #deg coordinates
        degCorX = self.degCorX+self.monTilt-90
        degCorY = self.degCorY
        
        lookupI = np.zeros(degCorX.shape).astype(np.int32)
        lookupJ = np.zeros(degCorX.shape).astype(np.int32)
        
        for j in xrange(lookupI.shape[1]):
            currDegX = degCorX[0,j]
            diffDegX = degNoWarpCorX[0,:] - currDegX
            IndJ = np.argmin(np.abs(diffDegX))
            lookupJ[:,j] = IndJ
            
            for i in xrange(lookupI.shape[0]):
                currDegY = degCorY[i,j]
                diffDegY = degNoWarpCorY[:,IndJ] - currDegY
                indI = np.argmin(np.abs(diffDegY))
                lookupI[i,j] = indI
        
        return lookupI, lookupJ
                

class KSstimJun(object):
    '''
    generate Kalatsky & Stryker stimulation integrats flashing indicator for 
    photodiode
    '''
    def __init__(self,
                 monitor,
                 indicator,
                 background = 0, #back ground color [-1,1]
                 squareSize=25, #size of flickering square 
                 squareCenter=(0,0), #coordinate of center point
                 flickerFrame=6,
                 sweepWidth=20., # width of sweeps (unit same as Map, cm or deg)
                 stepWidth=0.15, # width of steps (unit same as Map, cm or deg)
                 direction='B2U', # the direction of sweep movement, should be one of "B2U","U2B","L2R","R2L"
                 sweepFrame=1,
                 coordinate='degree', #'degree' or 'linear'
                 iteration=1, 
                 preGapFrame=120,
                 postGapFrame=180):
                     
        self.stimName = 'KSstim'
        self.monitor = monitor
        self.indicator =  indicator
        self.background = background
        self.squareSize = squareSize
        self.squareCenter = squareCenter
        self.flickerFrame = flickerFrame
        self.flickerFrequency=self.monitor.refreshRate / self.flickerFrame
        self.sweepWidth = sweepWidth
        self.stepWidth = stepWidth
        self.direction = direction
        self.sweepFrame = sweepFrame
        self.coordinate = coordinate
        self.iteration = iteration
        self.preGapFrame = preGapFrame
        self.postGapFrame = postGapFrame
#        self.sequencePath = sequencePath
#        self.color=color
        
        self.sweepSpeed = self.monitor.refreshRate * self.stepWidth / self.sweepFrame #the speed of sweeps deg/sec
        self.flickerHZ = self.monitor.refreshRate / self.flickerFrame
        
        self.squares = self.make_squares()
        #self.plot_squares()
        _, self.sweepTable = self.generate_sweeps()
        self.frames = self.generate_frames()
        

    def make_squares(self):
        '''
        generate checker board squares
        '''
        
        
        if self.coordinate == 'degree':
            mapX = self.monitor.degCorX
            mapY = self.monitor.degCorY
            
        elif self.coordinate == 'linear':
            mapX = self.monitor.linCorX
            mapY = self.monitor.linCorY
            
        else:
            raise LookupError, 'the "coordinate" attributate show be either "degree" or "linear"'
        
        minX = mapX.min()
        maxX = mapX.max()
        
        minY = mapY.min()
        maxY = mapY.max()
        
        negX = np.ceil( abs( ( ( minX - self.squareCenter[0] ) / ( 2 * self.squareSize ) ) ) ) + 1
        posX = np.ceil( abs( ( ( maxX - self.squareCenter[0] ) / ( 2 * self.squareSize ) ) ) ) + 1
        
        negY = np.ceil( abs( ( ( minY - self.squareCenter[0] ) / ( 2 * self.squareSize ) ) ) ) + 1
        posY = np.ceil( abs( ( ( maxY - self.squareCenter[0] ) / ( 2 * self.squareSize ) ) ) ) + 1
        
        squareV = np.ones((np.size(mapX, 0), np.size(mapX, 1)), dtype = np.float16)
        squareV = -1 * squareV
        
        stepV = np.arange(self.squareCenter[0] - ( 2 * negX + 0.5 ) * self.squareSize, 
                          self.squareCenter[0] + ( 2 * posX - 0.5 ) * self.squareSize, self.squareSize*2)
        
        for i in range(len(stepV)):
            squareV[ np.where( np.logical_and( mapX >= stepV[i], mapX < (stepV[i] + self.squareSize)))] = 1.0
        
        squareH = np.ones((np.size(mapY, 0), np.size(mapY, 1)), dtype = np.float16)
        squareH = -1 * squareH
        
        stepH = np.arange(self.squareCenter[1] - ( 2 * negY + 0.5 ) * self.squareSize, 
                          self.squareCenter[1] + ( 2 * posY - 0.5 ) * self.squareSize, self.squareSize*2)
        
        for j in range(len(stepH)):
            squareH[ np.where( np.logical_and( mapY >= stepH[j], mapY < (stepH[j] + self.squareSize)))] = 1
        
        squares = np.multiply(squareV, squareH)
        
        return squares
        
        
        
    def plot_squares(self):
        '''
        plot checkerboare squares
        '''
        plt.figure()
        plt.imshow(self.squares)
        
        
        
    def generate_sweeps(self):
        '''
        generate full screen sweep sequence
        '''
        sweepWidth = self.sweepWidth
        stepWidth =  self.stepWidth
        direction = self.direction
        
        if self.coordinate == 'degree':
            mapX = self.monitor.degCorX
            mapY = self.monitor.degCorY
            
        elif self.coordinate == 'linear':
            mapX = self.monitor.linCorX
            mapY = self.monitor.linCorY
            
        else:
            raise LookupError, 'the "coordinate" attributate show be either "degree" or "linear"'
        
        minX = mapX.min()
        maxX = mapX.max()
        
        minY = mapY.min()
        maxY = mapY.max()
        
        if direction == "B2U":
            stepY = np.arange(minY - sweepWidth, maxY + stepWidth, stepWidth)
        elif direction == "U2B":
            stepY = np.arange(minY - sweepWidth, maxY + stepWidth, stepWidth)[::-1]
            # stepY = np.arange(maxY, minY - sweepWidth - stepWidth, -1 * stepWidth)
        elif direction == "L2R":
            stepX = np.arange(minX - sweepWidth, maxX + stepWidth, stepWidth)
        elif direction == "R2L":
            stepX = np.arange(minX - sweepWidth, maxX + stepWidth, stepWidth)[::-1]
            # stepX = np.arange(maxX, minX - sweepWidth - stepWidth, -1 * stepWidth)
        else:
            raise LookupError, 'attribute "direction" should be "B2U", "U2B", "L2R" or "R2L".'
        
        sweepTable = []
        
        if 'stepX' in locals():
            sweeps = np.zeros((len(stepX), np.size(mapX, 0), np.size(mapX, 1)), dtype = np.float16)
            for i in range(len(stepX)):
                temp=sweeps[i,:,:]
                temp[np.where(np.logical_and(mapX >= stepX[i], mapX < (stepX[i] + sweepWidth)))] = 1.0
                sweepTable.append(('V', stepX[i], stepX[i] + sweepWidth))
                del temp
                
        if 'stepY' in locals():
            sweeps = np.zeros((len(stepY), np.size(mapY, 0), np.size(mapY, 1)), dtype = np.float16)
            for j in range(len(stepY)):
                temp=sweeps[j,:,:]
                temp[np.where(np.logical_and(mapY >= stepY[j], mapY < (stepY[j] + sweepWidth)))] = 1.0
                sweepTable.append(('H', stepY[j], stepY[j] + sweepWidth))
                del temp
                
        return sweeps.astype(np.bool), sweepTable


    def generate_frames(self):
        '''
        function to generate all the frames needed for KS stimulation
        
        returning a list of information of all frames, list of tuples
        
        for each frame:
        
        first element: gap:0 or display:1
        second element: square polarity, 1: not reversed; -1: reversed
        third element: sweeps, index in sweep table
        forth element: color of indicator
                       synchronized: gap:0, then alternating between -1 and 1 for each sweep
                       non-synchronized: alternating between -1 and 1 at defined frequency
        for gap frames the second and third elements should be 'None'
        '''
        
        sweeps, _ = self.generate_sweeps()
        sweepFrame = self.sweepFrame
        flickerFrame = self.flickerFrame
        iteration = self.iteration
        
        sweepNum = np.size(sweeps,0) # Number of sweeps, vertical or horizontal
        displayFrameNum = sweepFrame * sweepNum # total frame number for the visual stimulation of 1 iteration
        
        #frames for one iteration
        iterFrames=[] 
        
        #add frames for gaps
        for i in range(self.preGapFrame):
            iterFrames.append([0,None,None,-1])
        
        
        #add frames for display
        isreverse=[]
        
        for i in range(displayFrameNum):
            
            if (np.floor(i // flickerFrame)) % 2 == 0:
                isreverse = -1
            else:
                isreverse = 1
                
            sweepIndex=int(np.floor(i // sweepFrame))
            
            #add sychornized indicator
            if self.indicator.isSync == True:
                indicatorColor = 1
            else:
                indicatorColor = -1
                
            iterFrames.append([1,isreverse,sweepIndex,indicatorColor])
            
            
        # add gap frames at the end
        for i in range(self.postGapFrame):
            iterFrames.append([0,None,None,-1])
        
        fullFrames = []
        
        #add frames for multiple iteration
        for i in range(int(iteration)):
            fullFrames += iterFrames
        
        #add non-synchronized indicator
        if self.indicator.isSync == False:
            indicatorFrame = self.indicator.frameNum
            
            for j in range(np.size(fullFrames,0)):
                if np.floor(j // indicatorFrame) % 2 == 0:
                    fullFrames[j][3] = 1
                else:
                    fullFrames[j][3] = -1
            
        fullFrames = [tuple(x) for x in fullFrames]
        
        return tuple(fullFrames)
        
    
    def generateMovie(self):
        '''
        Function to Generate Kalatsky & Stryker visual stimulus frame by frame
        '''
        
        sweeps, _ = self.generate_sweeps()
        
        frames = self.frames
        
        fullSequence = np.zeros((len(frames),self.monitor.degCorX.shape[0],self.monitor.degCorX.shape[1]),dtype=np.uint8)
        
        indicatorWmin=self.indicator.centerWpixel - (self.indicator.width_pixel / 2)
        indicatorWmax=self.indicator.centerWpixel + (self.indicator.width_pixel / 2)
        indicatorHmin=self.indicator.centerHpixel - (self.indicator.height_pixel / 2)
        indicatorHmax=self.indicator.centerHpixel + (self.indicator.height_pixel / 2)
        
        background = self.background * np.ones((np.size(self.monitor.degCorX, 0), np.size(self.monitor.degCorX,1)), dtype = np.float32)
        
        for i in range(len(frames)):
            currFrame = frames[i]
            
            if currFrame[0] == 0:
                currNMsequence = background
                
            else:
                currSquare = self.squares * currFrame[1]
                currSweep = sweeps[currFrame[2]]
                currNMsequence = np.array(background)
                
                currNMsequence = (currSweep * currSquare) + ((-1 * (currSweep - 1)) * background)
                
                
            currNMsequence[indicatorHmin:indicatorHmax, indicatorWmin:indicatorWmax] = currFrame[3]
            
            fullSequence[i] = (255*(currNMsequence+1)/2).astype(np.uint8)
            
            print ['Generating numpy sequence: '+str(int(100 * (i+1) / len(frames)))+'%']
        
        
        mondict=dict(self.monitor.__dict__)
        indicatordict=dict(self.indicator.__dict__)
        indicatordict.pop('monitor')        
        KSdict=dict(self.__dict__)
        KSdict.pop('monitor')
        KSdict.pop('indicator')
        fulldictionary={'stimulation':KSdict, 
                        'monitor':mondict,
                        'indicator':indicatordict} 
                        
        return fullSequence, fulldictionary
        
    
    def reset(self):
        self.sweepTable = None
        self.frames = None


    def setDirection(self,direction):
        
        if direction == "B2U" or direction == "U2B" or direction == "L2R" or direction == "R2L":
            self.direction = direction
            self.reset()
        else:
            raise LookupError, 'attribute "direction" should be "B2U", "U2B", "L2R" or "R2L".'


    def setSweepSigma(self,sweepSigma):
        self.sweepSigma = sweepSigma
        self.reset()


    def setSweepWidth(self,sweepWidth):
        self.sweepWidth = sweepWidth
        self.reset()


    def setGapFrame(self,preGapFrame,postGapFrame):
        self.preGapFrame = preGapFrame
        self.postGapFrame = postGapFrame
        self.reset()


class NoiseKSstimJun(object):
    '''
    generate Kalatsky & Stryker stimulation but with noise movie not flashing 
    squares 
    
    it also integrats flashing indicator for photodiode
    '''
    def __init__(self,
                 monitor,
                 indicator,
                 tempFreqCeil = 15, # cutoff temporal frequency (Hz)
                 spatialFreqCeil = 0.05, # cutoff spatial frequency (cycle/degree)
                 filterMode = 'box', # type of filter of movie, '1/f' or 'box'
                 sweepWidth = 10., # width of sweeps (unit same as Map, cm or deg)
                 sweepSigma=5., # sigma of sweep edges (unit same as Map, cm or deg)
                 sweepEdgeWidth=3., # number of sigmas to smooth the edge of sweeps on each side
                 stepWidth=0.12, # width of steps (unit same as Map, cm or deg)
                 isWarp = False, # warp noise or not
                 direction='B2U', # the direction of sweep movement, should be one of "B2U","U2B","L2R","R2L"
                 sweepFrame=1, # display frame numbers for each step
                 coordinate='degree', #'degree' or 'linear'
                 iteration=1, 
                 preGapFrame=60, # gap frame number before flash
                 postGapFrame=300, # gap frame number after flash
                 enhanceExp = None): # exponential factor to enhance stimulus contrast (smaller means more enhancement)
                     
        self.stimName = 'NoiseKSstim'
        self.monitor = monitor
        self.indicator =  indicator
        self.tempFreqCeil = tempFreqCeil
        self.spatialFreqCeil = spatialFreqCeil
        self.filterMode = filterMode
        self.background = 0
        self.sweepSigma = sweepSigma
        self.sweepWidth = sweepWidth
        self.sweepEdgeWidth = sweepEdgeWidth
        self.isWarp = isWarp
        self.stepWidth = stepWidth
        self.direction = direction
        self.sweepFrame = sweepFrame
        self.coordinate = coordinate
        self.iteration = iteration
        self.preGapFrame = preGapFrame
        self.postGapFrame = postGapFrame
        self.enhanceExp = enhanceExp
        
        self.sweepSpeed = self.monitor.refreshRate * self.stepWidth / self.sweepFrame #the speed of sweeps deg/sec
        
        self.sweepTable = None
        self.frames = None
        

    def generate_noiseMovie(self, frameNum):
        '''
        generate filtered noise movie with defined number of frames
        '''
        
        Fs_T = self.monitor.refreshRate
        Flow_T = 0
        Fhigh_T = self.tempFreqCeil
        filter_T = generateFilter(frameNum, Fs_T, Flow_T, Fhigh_T, mode = self.filterMode)
        
        hPixNum = self.monitor.resolution[0]/self.monitor.downSampleRate
        pixHeightCM = self.monitor.monHcm / hPixNum
        Fs_H = 1 / (np.arcsin(pixHeightCM / self.monitor.dis) * 180 /  np.pi)
        #print 'Fs_H:', Fs_H
        Flow_H = 0
        Fhigh_H = self.spatialFreqCeil
        filter_H = generateFilter(hPixNum, Fs_H, Flow_H, Fhigh_H, mode = self.filterMode)
        
        wPixNum = self.monitor.resolution[1]/self.monitor.downSampleRate
        pixWidthCM = self.monitor.monWcm / wPixNum
        Fs_W = 1 / (np.arcsin(pixWidthCM / self.monitor.dis) * 180 / np.pi)
        #print 'Fs_W:', Fs_W
        Flow_W = 0
        Fhigh_W = self.spatialFreqCeil
        filter_W = generateFilter(wPixNum, Fs_W, Flow_W, Fhigh_W, mode = self.filterMode)
        
        movie = noiseMovie(filter_T, filter_W, filter_H, isplot = False)

        if self.enhanceExp:
                movie = (np.abs(movie)**self.enhanceExp)*(np.copysign(1,movie))
        
        return movie
        
        
    def generate_sweeps(self):
        '''
        generate full screen sweep sequence
        '''
        sweepSigma = self.sweepSigma
        stepWidth =  self.stepWidth
        direction = self.direction
        sweepWidth = float(self.sweepWidth)
        edgeWidth = self.sweepEdgeWidth * self.sweepSigma
        
        if self.coordinate == 'degree':
            mapX = self.monitor.degCorX
            mapY = self.monitor.degCorY
            
        elif self.coordinate == 'linear':
            mapX = self.monitor.linCorX
            mapY = self.monitor.linCorY
            
        else:
            raise LookupError, 'the "coordinate" attributate show be either "degree" or "linear"'
        
        minX = mapX.min()
        maxX = mapX.max()
        
        minY = mapY.min()
        maxY = mapY.max()
        
        if direction == "B2U":
            stepY = np.arange(minY - edgeWidth - sweepWidth / 2, maxY + edgeWidth + stepWidth + sweepWidth / 2, stepWidth)
        elif direction == "U2B":
            stepY = np.arange(minY - edgeWidth - sweepWidth / 2, maxY + edgeWidth + stepWidth + sweepWidth / 2, stepWidth)[::-1]
            # stepY = np.arange(maxY + edgeWidth + sweepWidth / 2, minY - edgeWidth - stepWidth - sweepWidth / 2, -1 * stepWidth)
        elif direction == "L2R":
            stepX = np.arange(minX - edgeWidth - sweepWidth / 2, maxX + edgeWidth + stepWidth + sweepWidth / 2, stepWidth)
        elif direction == "R2L":
            stepX = np.arange(minX - edgeWidth - sweepWidth / 2, maxX + edgeWidth + stepWidth + sweepWidth / 2, stepWidth)[::-1]
            # stepX = np.arange(maxX + edgeWidth + sweepWidth / 2, minX - edgeWidth - stepWidth - sweepWidth / 2, -1 * stepWidth)
        else:
            raise LookupError, 'attribute "direction" should be "B2U", "U2B", "L2R" or "R2L".'
        
        sweepTable = []
        
        if 'stepX' in locals():
            sweeps = np.ones((len(stepX), np.size(mapX, 0), np.size(mapX, 1)), dtype = np.float16)
            for i in range(len(stepX)):
                currSweep = sweeps[i,:,:]
                
                sweep1 = gaussian(mapX, mu = stepX[i] - sweepWidth / 2, sig = sweepSigma)
                sweep2 = gaussian(mapX, mu = stepX[i] + sweepWidth / 2, sig = sweepSigma)
                
                currSweep[mapX < (stepX[i] - sweepWidth / 2)] = sweep1[mapX < (stepX[i] - sweepWidth / 2)]
                currSweep[mapX > (stepX[i] + sweepWidth / 2)] = sweep2[mapX > (stepX[i] + sweepWidth / 2)]
                
                sweeps[i,:,:] = currSweep
                
                sweepTable.append(('V', stepX[i] - sweepWidth / 2, stepX[i] + sweepWidth / 2))
                
        if 'stepY' in locals():
            sweeps = np.ones((len(stepY), np.size(mapY, 0), np.size(mapY, 1)), dtype = np.float16)
            for j in range(len(stepY)):
                currSweep = sweeps[j,:,:]
                
                sweep1 = gaussian(mapY, mu = stepY[j] - sweepWidth / 2, sig = sweepSigma)
                sweep2 = gaussian(mapY, mu = stepY[j] + sweepWidth / 2, sig = sweepSigma)
                
                currSweep[mapY < (stepY[j] - sweepWidth / 2)] = sweep1[mapY < (stepY[j] - sweepWidth / 2)]
                currSweep[mapY > (stepY[j] + sweepWidth / 2)] = sweep2[mapY > (stepY[j] + sweepWidth / 2)]
                
                sweeps[j,:,:] = currSweep
                
                sweepTable.append(('H', stepY[j] - sweepWidth / 2, stepY[j] + sweepWidth / 2))
                
        return sweeps, sweepTable
        
        
          
    def generate_frames(self):
        '''
        function to generate all the frames needed for KS stimulation
        
        returning a list of information of all frames, list of tuples
        
        for each frame:
        
        first element: gap:0 or display:1
        second element: square polarity, None for new KSstim
        third element: sweeps, index in sweep table
        forth element: color of indicator, gap:0, then alternating between -1 and 1 for each sweep
        for gap frames the second and third elements should be 'None'
        '''
        
        if not(self.sweepTable):
            _, self.sweepTable = self.generate_sweeps()
        
        sweepTable = self.sweepTable
        sweepFrame = self.sweepFrame
        iteration = self.iteration
        
        sweepNum = len(sweepTable) # Number of sweeps, vertical or horizontal
        displayFrameNum = sweepFrame * sweepNum # total frame number for the visual stimulation of 1 iteration
        
        #frames for one iteration
        iterFrames=[] 
        
        #add frames for gaps
        for i in range(self.preGapFrame):
            iterFrames.append([0,None,None,-1])
        
        
        #add frames for display
        
        for i in range(displayFrameNum):
                
            sweepIndex=int(np.floor(i // sweepFrame))
            
            #add sychornized indicator
            if self.indicator.isSync == True:
                indicatorColor = 1
            else:
                indicatorColor = 0
                
            iterFrames.append([1,None,sweepIndex,indicatorColor])
            
            
        # add gap frames at the end
        for i in range(self.postGapFrame):
            iterFrames.append([0,None,None,-1])
        
        fullFrames = []
        
        #add frames for multiple iteration
        for i in range(int(iteration)):
            fullFrames += iterFrames
            
        
        #add non-synchronized indicator
        if self.indicator.isSync == False:
            indicatorFrame = self.indicator.frameNum
            
            for j in range(np.size(fullFrames,0)):
                if np.floor(j // indicatorFrame) % 2 == 0:
                    fullFrames[j][3] = 1
                else:
                    fullFrames[j][3] = -1
            
        fullFrames = [tuple(x) for x in fullFrames]
        
        
        return tuple(fullFrames)
        
    
    def generateMovie(self):
        '''
        Function to Generate Kalatsky & Stryker visual stimulus frame by frame
        '''
        
        sweeps, self.sweepTable = self.generate_sweeps()
        
        self.frames = self.generate_frames()
        
        noiseMovie = self.generate_noiseMovie(len(self.frames))
        
        if self.isWarp:
            lookupI, lookupJ = self.monitor.generate_LookUpTable()
         
        fullSequence = np.zeros((len(self.frames),self.monitor.degCorX.shape[0],self.monitor.degCorX.shape[1]),dtype=np.uint8)
        
        indicatorWmin=self.indicator.centerWpixel - (self.indicator.width_pixel / 2)
        indicatorWmax=self.indicator.centerWpixel + (self.indicator.width_pixel / 2)
        indicatorHmin=self.indicator.centerHpixel - (self.indicator.height_pixel / 2)
        indicatorHmax=self.indicator.centerHpixel + (self.indicator.height_pixel / 2)
        
        background = np.zeros(self.monitor.degCorX.shape, dtype = np.float32)
        
        for i in range(len(self.frames)):
            currFrame = self.frames[i]
            
            if currFrame[0] == 0:
                currNMsequence = background
            else:
                currImage = noiseMovie[i,:,:]
                if self.isWarp:
                    currImage = lookupImage(currImage, lookupI, lookupJ)
                currNMsequence = currImage * sweeps[currFrame[2]]
                
            currNMsequence[indicatorHmin:indicatorHmax, indicatorWmin:indicatorWmax] = currFrame[3]

            fullSequence[i] = (255*(currNMsequence+1)/2).astype(np.uint8)
            
            print ['Generating numpy sequence: '+str(int(100 * (i+1) / len(self.frames)))+'%']
        
        
        mondict=dict(self.monitor.__dict__)
        indicatordict=dict(self.indicator.__dict__)
        KSdict=dict(self.__dict__)
        KSdict.pop('monitor')
        KSdict.pop('indicator')
        fullDictionary={'stimulation':KSdict, 
                        'monitor':mondict,
                        'indicator':indicatordict}
                        
        return fullSequence, fullDictionary
        
    
    def reset(self):
        self.sweepTable = None
        self.frames = None
    
    def setDirection(self,direction):
        
        if direction == "B2U" or direction == "U2B" or direction == "L2R" or direction == "R2L":
            self.direction = direction
            self.reset()
        else:
            raise LookupError, 'attribute "direction" should be "B2U", "U2B", "L2R" or "R2L".'
            
    def setSweepSigma(self,sweepSigma):
        self.sweepSigma = sweepSigma
        self.reset()
        
    def setSweepWidth(self,sweepWidth):
        self.sweepWidth = sweepWidth
        self.reset()
        
    def setGapFrame(self,preGapFrame,postGapFrame):
        self.preGapFrame = preGapFrame
        self.postGapFrame = postGapFrame
        self.reset()
    
 
class FlashNoise(object):
    
    '''
    generate flashing full field noise with background displayed before and after
    
    it also integrats flashing indicator for photodiode
    '''
    
    def __init__(self,
                 monitor,
                 indicator,
                 spatialFreqCeil = 0.05, # cutoff spatial frequency (cycle/degree)
                 filterMode = 'box', # type of filter of movie, '1/f' or 'box'
                 iteration=50, # time to flash
                 flashFrame=1, # frame number for display noise of each flash
                 preGapFrame=60, # gap frame number before flash
                 postGapFrame=300, # gap frame number after flash
                 isWarp = False, # warp noise or not,
                 background = 0.):
        
        self.stimName = 'FlashNoise'
        self.monitor = monitor
        self.indicator = indicator
        self.spatialFreqCeil = spatialFreqCeil
        self.filterMode = filterMode
        self.iteration = iteration
        self.flashFrame = flashFrame
        self.preGapFrame = preGapFrame
        self.postGapFrame = postGapFrame
        self.isWarp = isWarp
        self.background = background
        
        self.frames = None
        self.iterationFrame = None #frame number for each iteration
    
    def generatNoiseMovie(self):
        '''
        generate filtered noise movie with defined number of frames
        '''
        
        frameNum = self.flashFrame * self.iteration
        filter_T = np.ones((frameNum))
        
        hPixNum = self.monitor.resolution[0]/self.monitor.downSampleRate
        pixHeightCM = self.monitor.monHcm / hPixNum
        Fs_H = 1 / (np.arcsin(pixHeightCM / self.monitor.dis) * 180 /  np.pi)
        Flow_H = 0
        Fhigh_H = self.spatialFreqCeil
        filter_H = generateFilter(hPixNum, Fs_H, Flow_H, Fhigh_H, mode = self.filterMode)
        
        wPixNum = self.monitor.resolution[1]/self.monitor.downSampleRate
        pixWidthCM = self.monitor.monWcm / wPixNum
        Fs_W = 1 / (np.arcsin(pixWidthCM / self.monitor.dis) * 180 / np.pi)
        Flow_W = 0
        Fhigh_W = self.spatialFreqCeil
        filter_W = generateFilter(wPixNum, Fs_W, Flow_W, Fhigh_W, mode = self.filterMode)
        
        movie = noiseMovie(filter_T, filter_W, filter_H, isplot = False)
        
        return movie
        
        
    
    def generatFrames(self):
        '''
        function to generate all the frames needed for KS stimulation
        
        returning a list of information of all frames, list of tuples
        
        for each frame:
        
        first element: gap:0 or display:1
        second element: iteration start, first frame of each iteration: 1; other frames: 0
        third element: current iteration 
        forth element: color of indicator, gap:0, then alternating between -1 and 1 for each sweep
        '''
        
        #frame number for each iteration
        self.iterationFrame = self.preGapFrame+self.flashFrame+self.postGapFrame
        
        frames = np.zeros((self.iteration*(self.iterationFrame),4)).astype(np.int)
        
        #initilize indicator color
        frames[:,3] = -1
        
        for i in xrange(frames.shape[0]):

            # current iteration number
            frames[i,2] = i // self.iterationFrame            
            
            # mark start frame of every iteration
            if i % self.iterationFrame == 0:
                frames[i, 1] = 1
                
            # mark display frame and synchronized indicator 
            if ((i % self.iterationFrame >= self.preGapFrame) and \
               (i % self.iterationFrame < (self.preGapFrame + self.flashFrame))):
                   
                frames[i, 0] = 1
                
                if self.indicator.isSync:
                    frames[i, 3] = 1
            
            # mark unsynchronized indicator
            if not(self.indicator.isSync):
                if np.floor(i // self.indicator.frameNum) % 2 == 0:
                    frames[i,3] = 1
                    
        frames = [tuple(x) for x in frames]
        
        return tuple(frames)


    def set_preGapFrameNum(self, preGapFrameNum):
        self.preGapFrame = preGapFrameNum
        self.frames = None
        self.iterationFrame = None
        
    def set_postGapFrameNum(self, postGapFrameNum):
        self.postGapFrame = postGapFrameNum
        self.frames = None
        self.iterationFrame = None
        
    def set_flashFrameNum(self, flashFrameNum):
        self.flashFrame = flashFrameNum
        self.frames = None
        self.iterationFrame = None
        
    def generateMovie(self):
        '''
        generating movie
        '''            
        
        self.frames = self.generatFrames()
        noiseMovie = self.generatNoiseMovie()
        
        if self.isWarp:
            lookupI, lookupJ = self.monitor.generate_LookUpTable()
        
        fullSequence = np.zeros((len(self.frames),self.monitor.degCorX.shape[0],self.monitor.degCorX.shape[1]),dtype=np.uint8)
        
        indicatorWmin=self.indicator.centerWpixel - (self.indicator.width_pixel / 2)
        indicatorWmax=self.indicator.centerWpixel + (self.indicator.width_pixel / 2)
        indicatorHmin=self.indicator.centerHpixel - (self.indicator.height_pixel / 2)
        indicatorHmax=self.indicator.centerHpixel + (self.indicator.height_pixel / 2)
        
        background = self.background * np.ones((np.size(self.monitor.degCorX, 0), np.size(self.monitor.degCorX,1)), dtype = np.float32)
        
        for i in range(len(self.frames)):
            currFrame = self.frames[i]
            
            if currFrame[0] == 0:
                currFNsequence = background
            else:
                currFNsequence = noiseMovie[currFrame[2],:,:]
                if self.isWarp:
                    currFNsequence = lookupImage(currFNsequence, lookupI, lookupJ)
                
            currFNsequence[indicatorHmin:indicatorHmax, indicatorWmin:indicatorWmax] = currFrame[3]
            
            fullSequence[i] = (255*(currFNsequence+1)/2).astype(np.uint8)
            
            print ['Generating numpy sequence: '+str(int(100 * (i+1) / len(self.frames)))+'%']
        
        mondict=dict(self.monitor.__dict__)
        indicatordict=dict(self.indicator.__dict__)
        indicatordict.pop('monitor')        
        NFdict=dict(self.__dict__)
        NFdict.pop('monitor')
        NFdict.pop('indicator')
        fullDictionary={'stimulation':NFdict, 
                        'monitor':mondict,
                        'indicator':indicatordict}
                        
        return fullSequence, fullDictionary
        
    
class GaussianNoise(object):
    '''
    generate full field noise movie with contrast modulated by gaussian function
    '''
    def __init__(self,
                 monitor,
                 indicator,
                 tempFreqCeil = 15, # cutoff temporal frequency (Hz)
                 spatialFreqCeil = 0.05, # cutoff spatial frequency (cycle/degree)
                 filterMode = 'box', # type of filter of movie, '1/f' or 'box'
                 sweepSigma=10., # sigma of sweep edges (unit same as Map, cm or deg)
                 sweepWidth=10., # width of sweeps (unit same as Map, cm or deg)
                 sweepEdgeWidth=3., # number of sigmas to smooth the edge of sweeps on each side
                 stepWidth=0.12, # width of steps (unit same as Map, cm or deg)
                 sweepFrame=1, # display frame numbers for each step
                 iteration=1, 
                 preGapFrame=60, # gap frame number before flash
                 postGapFrame=300, # gap frame number after flash
                 isWarp = False, # warp noise or not
                 contrast = 0.5, # contrast of the movie from 0 to 1
                 background = 0.,
                 enhanceExp = None):
        
        self.stimName = 'GaussianNoise'     
        self.monitor = monitor
        self.indicator = indicator
        self.tempFreqCeil = tempFreqCeil
        self.spatialFreqCeil = spatialFreqCeil
        self.filterMode = filterMode
        self.sweepSigma = sweepSigma
        self.sweepWidth = sweepWidth
        self.stepWidth = stepWidth
        self.sweepEdgeWidth= sweepEdgeWidth
        self.sweepFrame = sweepFrame
        self.iteration = iteration
        self.preGapFrame = preGapFrame
        self.postGapFrame = postGapFrame
        self.isWarp = isWarp
        self.background = background
        self.contrast = contrast
        self.enhanceExp = enhanceExp
        
        self.frames = None
        self.iterationFrame = None #frame number for each iteration
    
    def set_preGapFrameNum(self, preGapFrameNum):
        self.preGapFrame = preGapFrameNum
        self.frames = None
        self.iterationFrame = None
        
    def set_postGapFrameNum(self, postGapFrameNum):
        self.postGapFrame = postGapFrameNum
        self.frames = None
        self.iterationFrame = None
        
    def set_flashFrameNum(self, flashFrameNum):
        self.flashFrame = flashFrameNum
        self.frames = None
        self.iterationFrame = None 
        
    def set_sweepSigma(self, sweepSigma):
        self.sweepSigma = sweepSigma
        self.frames = None
        self.iterationFrame = None
        
    def set_sweepWidth(self, sweepWidth):
        self.sweepWidth = sweepWidth
        self.frames = None
        self.iterationFrame = None
        
    def set_contrast(self, contrast):
        self.contrast = contrast
        self.frames = None
        self.iterationFrame = None

    def generateNoiseMovie(self, frameNum):
        '''
        generate filtered noise movie with defined number of frames
        '''
        
        Fs_T = self.monitor.refreshRate
        Flow_T = 0
        Fhigh_T = self.tempFreqCeil
        filter_T = generateFilter(frameNum, Fs_T, Flow_T, Fhigh_T, mode = self.filterMode)
        
        hPixNum = self.monitor.resolution[0]/self.monitor.downSampleRate
        pixHeightCM = self.monitor.monHcm / hPixNum
        Fs_H = 1 / (np.arcsin(pixHeightCM / self.monitor.dis) * 180 /  np.pi)
        #print 'Fs_H:', Fs_H
        Flow_H = 0
        Fhigh_H = self.spatialFreqCeil
        filter_H = generateFilter(hPixNum, Fs_H, Flow_H, Fhigh_H, mode = self.filterMode)
        
        wPixNum = self.monitor.resolution[1]/self.monitor.downSampleRate
        pixWidthCM = self.monitor.monWcm / wPixNum
        Fs_W = 1 / (np.arcsin(pixWidthCM / self.monitor.dis) * 180 / np.pi)
        #print 'Fs_W:', Fs_W
        Flow_W = 0
        Fhigh_W = self.spatialFreqCeil
        filter_W = generateFilter(wPixNum, Fs_W, Flow_W, Fhigh_W, mode = self.filterMode)
        
        movie = noiseMovie(filter_T, filter_W, filter_H, isplot = False)

        if self.enhanceExp:
                movie = (np.abs(movie)**self.enhanceExp)*(np.copysign(1,movie))
        
        return movie

        
    def generateFrames(self):
        '''
        function to generate all the frames needed for KS stimulation
        
        returning a list of information of all frames, list of tuples
        
        for each frame:
        
        first element: gap:0 or display:1
        second element: iteration start, first frame of each iteration: 1; other frames: 0
        third element: current iteration 
        forth element: color of indicator, gap:0, then alternating between -1 and 1 for each sweep
        fifth element: if is display, the contrast
        '''
        
        sweepEdge = self.sweepEdgeWidth * self.sweepSigma
        
        steps = np.arange(-sweepEdge - self.sweepWidth / 2, sweepEdge + self.sweepWidth / 2, self.stepWidth)
        
        displayFrameNum = self.sweepFrame * len(steps) # total frame number for the visual stimulation of 1 iteration
        
        stepContrast = np.ones(len(steps))
        stepContrast1 = gaussian(steps, mu=-self.sweepWidth / 2, sig=self.sweepSigma)
        stepContrast2 = gaussian(steps, mu=+self.sweepWidth / 2, sig=self.sweepSigma)
        
        stepContrast[steps < (-self.sweepWidth / 2)] = stepContrast1[steps < (-self.sweepWidth / 2)]
        stepContrast[steps > (self.sweepWidth / 2)] = stepContrast2[steps > (self.sweepWidth / 2)]
        
        
        #frame number for each iteration
        self.iterationFrame = self.preGapFrame + displayFrameNum + self.postGapFrame
        
        frames = np.zeros((self.iteration*(self.iterationFrame),5)).astype(np.float32)
        
        #initilize indicator color
        frames[:,3] = -1
        
        for i in xrange(frames.shape[0]):

            # current iteration number
            frames[i,2] = i // self.iterationFrame            
            
            # mark start frame of every iteration
            if i % self.iterationFrame == 0:
                frames[i, 1] = 1
                
            # mark display frame and synchronized indicator 
            if ((i % self.iterationFrame >= self.preGapFrame) and \
               (i % self.iterationFrame < (self.preGapFrame + displayFrameNum))):
                   
                frames[i, 0] = 1
                
                if self.indicator.isSync:
                    frames[i, 3] = 1
            
            # mark unsynchronized indicator
            if not(self.indicator.isSync):
                if np.floor(i // self.indicator.frameNum) % 2 == 0:
                    frames[i,3] = 1
                    
            # mark display contrast
            currFrameNumInIteration = i % self.iterationFrame
            if (currFrameNumInIteration < self.preGapFrame) or \
               (currFrameNumInIteration >= self.preGapFrame + displayFrameNum):
                frames[i,4] = np.nan
            else:
                displayInd = currFrameNumInIteration - self.preGapFrame
                frames[i,4] = stepContrast[displayInd // self.sweepFrame] * self.contrast
                
        frames = [tuple(x) for x in frames]
        
        return tuple(frames)
        
    def generateMovie(self):
        '''
        generating movie
        '''
        
        if not(self.frames):
            self.frames = self.generateFrames()
        
        if self.isWarp:
            lookupI, lookupJ = self.monitor.generate_LookUpTable()
        
        fullSequence = np.zeros((len(self.frames),self.monitor.degCorX.shape[0],self.monitor.degCorX.shape[1]),dtype=np.uint8)
        
        indicatorWmin=self.indicator.centerWpixel - (self.indicator.width_pixel / 2)
        indicatorWmax=self.indicator.centerWpixel + (self.indicator.width_pixel / 2)
        indicatorHmin=self.indicator.centerHpixel - (self.indicator.height_pixel / 2)
        indicatorHmax=self.indicator.centerHpixel + (self.indicator.height_pixel / 2)
        
        background = self.background * np.ones((self.monitor.degCorX.shape), dtype = np.float32)
        
        for i in range(len(self.frames)):
            
            currFrame = self.frames[i]
            
            if currFrame[1] == 1:
                displayFrameNum = self.iterationFrame - self.preGapFrame - self.postGapFrame
                noiseMovie = self.generateNoiseMovie(displayFrameNum)
            
            if currFrame[0] == 0:
                currGNsequence = background
            else:
                currDisplayInd = (i % self.iterationFrame) - self.preGapFrame
                currGNsequence = noiseMovie[currDisplayInd,:,:] * currFrame[4]
                if self.isWarp:
                    currGNsequence = lookupImage(currGNsequence, lookupI, lookupJ)
            
            currGNsequence[indicatorHmin:indicatorHmax, indicatorWmin:indicatorWmax] = currFrame[3]
            
            fullSequence[i] = (255*(currGNsequence+1)/2).astype(np.uint8)
            
            print ['Generating RGB sequence: '+str(int(100 * (i+1) / len(self.frames)))+'%']
            
        mondict=dict(self.monitor.__dict__)
        indicatordict=dict(self.indicator.__dict__)
        indicatordict.pop('monitor')        
        KSdict=dict(self.__dict__)
        KSdict.pop('monitor')
        KSdict.pop('indicator')
        fullDictionary={'stimulation':KSdict, 
                        'monitor':mondict,
                        'indicator':indicatordict}
                        
        return fullSequence, fullDictionary

        
class FlashCircle(object):
    '''
    flashing circle stimulation.
    '''
    
    def __init__(self,
                 monitor,
                 indicator,
                 center = (90., 10.), # center coordinate of the circle (degree)
                 radius = 10., # radius of the circle
                 color = -1., # color of the circle [-1: 1]
                 iteration= 1, # total number of flashes
                 flashFrame= 3, # frame number for display circle of each flash
                 preGapFrame=120, # gap frame number before flash
                 postGapFrame=180, # gap frame number after flash
                 background = 0.):
        
        self.stimName = 'FlashCircle'
        self.monitor = monitor
        self.indicator = indicator
        self.center = center
        self.radius = radius
        self.color = color
        self.iteration = iteration
        self.flashFrame = flashFrame
        self.preGapFrame = preGapFrame
        self.postGapFrame = postGapFrame
        self.background = background
        
        self.frames = None
        self.iterationFrame = None #frame number for each iteration
        
        
    def set_preGapFrameNum(self, preGapFrameNum):
        self.preGapFrame = preGapFrameNum
        self.frames = None
        self.iterationFrame = None
        
    def set_postGapFrameNum(self, postGapFrameNum):
        self.postGapFrame = postGapFrameNum
        self.frames = None
        self.iterationFrame = None
        
    def set_flashFrameNum(self, flashFrameNum):
        self.flashFrame = flashFrameNum
        self.frames = None
        self.iterationFrame = None
        
    def set_color(self, color):
        self.color = color
        self.frames = None
        self.iterationFrame = None
        
    def set_center(self, center):
        self.center = center
        self.frames = None
        self.iterationFrame = None
    
    def set_radius(self, radius):
        self.radius = radius
        self.frames = None
        self.iterationFrame = None

    def generatFrames(self):
        '''
        function to generate all the frames needed for the stimulation
        
        returning a list of information of all frames, list of tuples
        
        for each frame:
        
        first element: gap:0 or display:1
        second element: iteration start, first frame of each iteration: 1; other frames: 0
        third element: current iteration 
        forth element: color of indicator, gap:0, then alternating between -1 and 1 for each sweep
        '''
        
        #frame number for each iteration
        self.iterationFrame = self.preGapFrame+self.flashFrame+self.postGapFrame
        
        frames = np.zeros((self.iteration*(self.iterationFrame),4)).astype(np.int)
        
        #initilize indicator color
        frames[:,3] = -1
        
        for i in xrange(frames.shape[0]):

            # current iteration number
            frames[i,2] = i // self.iterationFrame            
            
            # mark start frame of every iteration
            if i % self.iterationFrame == 0:
                frames[i, 1] = 1
                
            # mark display frame and synchronized indicator 
            if ((i % self.iterationFrame >= self.preGapFrame) and \
               (i % self.iterationFrame < (self.preGapFrame + self.flashFrame))):
                   
                frames[i, 0] = 1
                
                if self.indicator.isSync:
                    frames[i, 3] = 1
            
            # mark unsynchronized indicator
            if not(self.indicator.isSync):
                if np.floor(i // self.indicator.frameNum) % 2 == 0:
                    frames[i,3] = 1
                    
        frames = [tuple(x) for x in frames]
        
        return tuple(frames)
        
    
    def generateMovie(self):
        '''
        generating movie
        '''            
        
        self.frames = self.generatFrames()
        
        fullSequence = np.zeros((len(self.frames),self.monitor.degCorX.shape[0],self.monitor.degCorX.shape[1]),dtype=np.uint8)
        
        indicatorWmin=self.indicator.centerWpixel - (self.indicator.width_pixel / 2)
        indicatorWmax=self.indicator.centerWpixel + (self.indicator.width_pixel / 2)
        indicatorHmin=self.indicator.centerHpixel - (self.indicator.height_pixel / 2)
        indicatorHmax=self.indicator.centerHpixel + (self.indicator.height_pixel / 2)
        
        background = self.background * np.ones((np.size(self.monitor.degCorX, 0), np.size(self.monitor.degCorX,1)), dtype = np.float32)
        
        
        circleMask = np.zeros((np.size(self.monitor.degCorX, 0), np.size(self.monitor.degCorX,1)), dtype = np.float32)
        for i in xrange(circleMask.shape[0]):
            for j in xrange(circleMask.shape[1]):
                x=self.monitor.degCorX[i,j]
                y=self.monitor.degCorY[i,j]
                if np.sqrt((x-self.center[0])**2 + (y-self.center[1])**2) <= self.radius:
                    circleMask[i,j] = 1        
        
        
        for i in range(len(self.frames)):
            currFrame = self.frames[i]
            
            if currFrame[0] == 0:
                currFCsequence = background
            else:
                currFCsequence = (circleMask * self.color) + ((-1 * (circleMask - 1)) * background)
                
            currFCsequence[indicatorHmin:indicatorHmax, indicatorWmin:indicatorWmax] = currFrame[3]
            
            fullSequence[i] = (255*(currFCsequence+1)/2).astype(np.uint8)
            
            print ['Generating numpy sequence: '+str(int(100 * (i+1) / len(self.frames)))+'%']
        
        mondict=dict(self.monitor.__dict__)
        indicatordict=dict(self.indicator.__dict__)
        indicatordict.pop('monitor')        
        NFdict=dict(self.__dict__)
        NFdict.pop('monitor')
        NFdict.pop('indicator')
        fullDictionary={'stimulation':NFdict, 
                        'monitor':mondict,
                        'indicator':indicatordict}
                        
        return fullSequence, fullDictionary


class SparseNoise(object):
    '''
    generate sparse noise stimulus integrates flashing indicator for photodiode
    '''

    def __init__(self,
                 monitor,
                 indicator,
                 coordinate='degree', #'degree' or 'linear'
                 background = 0., #back ground color [-1,1]
                 gridSpace = (10.,10.), #(alt,azi)
                 probeSize = (10.,10.), #size of flicker probes (width,height)
                 probeOrientation = 0., #orientation of flicker probes
                 probeFrameNum = 3, #number of frames for each square presentation
                 subregion = None, #[minAlt, maxAlt, minAzi, maxAzi]
                 sign = 'ON-OFF', # 'On', 'OFF' or 'ON-OFF'
                 iteration = 1,
                 preGapFrame = 0,
                 postGapFrame = 0):

        self.stimName = 'SparseNoise'
        self.monitor = monitor
        self.indicator = indicator
        self.coordinate = coordinate
        self.background = background
        self.gridSpace = gridSpace
        self.probeSize = probeSize
        self.probeOrientationt = probeOrientation
        self.probeFrameNum = probeFrameNum
        self.subregion = subregion
        self.sign = sign
        self.iteration = iteration
        self.preGapFrame = preGapFrame
        self.postGapFrame = postGapFrame

        if self.subregion is None:
            if coordinate == 'degree':
                self.subregion=[np.amin(self.monitor.degCorY),np.amax(self.monitor.degCorY),
                                np.amin(self.monitor.degCorX),np.amax(self.monitor.degCorX)]
            if coordinate == 'linear':
                self.subregion=[np.amin(self.monitor.linCorY),np.amax(self.monitor.linCorY),
                                np.amin(self.monitor.linCorX),np.amax(self.monitor.linCorX)]


    def _getGridPoints(self):
        '''
        generate all the grid points in display area (subregion and monitor coverage)
        [azi, alt]
        '''

        rows = np.arange(self.subregion[0],self.subregion[1],self.gridSpace[0])
        columns = np.arange(self.subregion[2],self.subregion[3],self.gridSpace[1])

        xx,yy = np.meshgrid(columns,rows)

        gridPoints = np.transpose(np.array([xx.flatten(),yy.flatten()]))

        #get all the visual points for each pixels on monitor
        if self.coordinate == 'degree':monitorPoints = np.transpose(np.array([self.monitor.degCorX.flatten(),self.monitor.degCorY.flatten()]))
        if self.coordinate == 'linear':monitorPoints = np.transpose(np.array([self.monitor.linCorX.flatten(),self.monitor.linCorY.flatten()]))

        #get the grid points within the coverage of monitor
        gridPoints = gridPoints[in_hull(gridPoints,monitorPoints)]

        return gridPoints


    def _getGridPointsSequence(self):
        '''
        generate pseudorandomized grid point sequence. if ON-OFF, continuous frame shold not
        present stimulus at same location
        :return: list of [gridPoint, sign]
        '''

        gridPoints = self._getGridPoints()

        if self.sign == 'ON':
            gridPoints = [[x,1] for x in gridPoints]
            shuffle(gridPoints)
            return gridPoints
        elif self.sign == 'OFF':
            gridPoints = [[x,-1] for x in gridPoints]
            shuffle(gridPoints)
            return gridPoints
        elif self.sign == 'ON-OFF':
            allGridPoints = [[x,1] for x in gridPoints] + [[x,-1] for x in gridPoints]
            shuffle(allGridPoints)
            # remove coincident hit of same location by continuous frames
            print 'removing coincident hit of same location with continuous frames:'
            while True:
                iteration = 0
                coincidentHitNum = 0
                for i, gridPoint in enumerate(allGridPoints[:-3]):
                    if (allGridPoints[i][0] == allGridPoints[i+1][0]).all():
                        allGridPoints[i+1], allGridPoints[i+2] = allGridPoints[i+2], allGridPoints[i+1]
                        coincidentHitNum += 1
                iteration += 1
                print 'iteration:',iteration,'  continous hits number:',coincidentHitNum
                if coincidentHitNum == 0:
                    break

            return allGridPoints


    def generateFramesList(self):
        '''
        function to generate all the frames needed for SparseNoiseStimu

        returning a list of information of all frames, list of tuples

        for each frame:

        first element: gap:0 or display:1
        second element: tuple, retinotopic location of the center of current square,[azi,alt]
        third element: polarity of current square, 1: bright, -1: dark
        forth element: color of indicator
                       synchronized: gap:0, 1 for onset frame for each square, -1 for the rest
                       non-synchronized: alternating between -1 and 1 at defined frequency
        for gap frames the second and third elements should be 'None'
        '''

        self.frames = []

        for i in range(self.iteration):

            if self.preGapFrame>0: self.frames += [[0,None,None,-1]]*self.preGapFrame

            iterGridPoints = self._getGridPointsSequence()

            for gridPoint in iterGridPoints:
                self.frames += [[1,gridPoint[0],gridPoint[1],1]]
                if self.probeFrameNum > 1:
                    self.frames += [[1,gridPoint[0],gridPoint[1],-1]] * (self.probeFrameNum-1)

            if self.postGapFrame>0: self.frames += [[0,None,None,-1]]*self.postGapFrame

        if self.indicator.isSync == False:
            indicatorFrame = self.indicator.frameNum
            for m in range(len(self.frames)):
                if np.floor(m // indicatorFrame) % 2 == 0:self.frames[m][3] = 1
                else:self.frames[m][3] = -1

        self.frames=tuple(self.frames)


    def generateMovie(self):
        '''
        generate movie for display
        '''

        self.generateFramesList()

        if self.coordinate=='degree':corX=self.monitor.degCorX;corY=self.monitor.degCorY
        if self.coordinate=='linear':corX=self.monitor.linCorX;corY=self.monitor.linCorY

        indicatorWmin=self.indicator.centerWpixel - (self.indicator.width_pixel / 2)
        indicatorWmax=self.indicator.centerWpixel + (self.indicator.width_pixel / 2)
        indicatorHmin=self.indicator.centerHpixel - (self.indicator.height_pixel / 2)
        indicatorHmax=self.indicator.centerHpixel + (self.indicator.height_pixel / 2)

        fullSequence = np.zeros((len(self.frames),self.monitor.degCorX.shape[0],self.monitor.degCorX.shape[1]),dtype=np.uint8)

        for i, currFrame in enumerate(self.frames):
            if i == 1 or \
               (currFrame[1]!=self.frames[i-1][1]).any() or \
               currFrame[2]!=self.frames[i-1][2]:
                currDisplayMatrix = getWarpedFrameWithSquare(corX,
                                                             corY,
                                                             center = currFrame[1],
                                                             width=self.probeSize[0],
                                                             height=self.probeSize[1],
                                                             ori=self.probeOrientationt,
                                                             foregroundColor=currFrame[2],
                                                             backgroundColor=self.background)

            #add sync square for photodiode
            currDisplayMatrix[indicatorHmin:indicatorHmax, indicatorWmin:indicatorWmax]=currFrame[3]

            #assign current display matrix to full sequence
            fullSequence[i] = ((((currDisplayMatrix+1)/2))*255).astype(np.uint8)

            print ['Generating numpy sequence: '+str(int(100 * (i+1) / len(self.frames)))+'%']

        #generate log dictionary
        mondict=dict(self.monitor.__dict__)
        indicatordict=dict(self.indicator.__dict__)
        indicatordict.pop('monitor')
        SNdict=dict(self.__dict__)
        SNdict.pop('monitor')
        SNdict.pop('indicator')
        fulldictionary={'stimulation':SNdict,
                        'monitor':mondict,
                        'indicator':indicatordict}

        return fullSequence, fulldictionary

   
class IndicatorJun(object):
    '''
    flashing indicator for photodiode
    '''
    
    def __init__(self,
                 monitor,
                 width_cm = 3., 
                 height_cm = 3., 
                 position = 'northeast',
                 isSync = True,
                 freq = 2.):
        self.monitor=monitor
        self.width_cm = width_cm
        self.height_cm = height_cm
        self.width_pixel, self.height_pixel = self.get_size_pixel()
        self.position = position
        self.centerWpixel, self.centerHpixel = self.get_center()
        self.isSync = isSync
        
        if isSync == False:
            self.freq = freq #if not synchronized with stimulation, self update frquency of the indicator
            self.frameNum = self.get_frames()
        else:
            self.freq = None
            self.frameNum = None
        
    def get_size_pixel(self):
        
        screen_width = self.monitor.resolution[1] / self.monitor.downSampleRate
        screen_height = self.monitor.resolution[0] / self.monitor.downSampleRate
        
        indicator_width = ((self.width_cm / self.monitor.monWcm ) * screen_width) // 1
        indicator_height = ((self.height_cm / self.monitor.monHcm ) * screen_height) // 1
        
        return indicator_width, indicator_height
        
    def get_center(self):
        
        screen_width = self.monitor.resolution[1] / self.monitor.downSampleRate
        screen_height = self.monitor.resolution[0] / self.monitor.downSampleRate
        
        if self.position == 'northeast':
            centerW = screen_width - self.width_pixel / 2
            centerH = self.height_pixel / 2
            
        elif self.position == 'northwest':
            centerW = self.width_pixel / 2
            centerH = self.height_pixel / 2
            
        elif self.position == 'southeast':
            centerW = screen_width - self.width_pixel / 2
            centerH = screen_height - self.height_pixel / 2
            
        elif self.position == 'southwest':
            centerW = self.width_pixel / 2
            centerH = screen_height - self.height_pixel / 2
            
        else:
            raise LookupError, '"position" attributor should be "northeast", "southeast", "northwest" and "southwest"'
            
        return centerW, centerH
        
    def get_frames(self):
        
        '''
        if not synchronized with stimulation, get frame numbers of each update 
        of indicator
        '''
        
        refreshRate = self.monitor.refreshRate        
        
        if refreshRate % self.freq != 0:
            raise ArithmeticError, "self update frequency of should be divisible by monitor's refresh rate."
        
        return refreshRate/self.freq

         
class DisplaySequence(object):
    '''
    Display the numpy sequence from memory
    '''        
    
    def __init__(self,
                 logdir,
                 backupdir,
                 displayIteration = 1,
                 displayOrder = 1, # 1: the right order; -1: the reverse order
                 mouseid = 'Test',
                 userid = 'Jun',
                 isVideoRecord = False,
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
                 initialBackgroundColor = 0,
                 videoRecordIP = 'localhost',
                 videoRecordPort = '10000'):
                     
        self.sequence = None
        self.sequenceLog = {}             
        self.isVideoRecord = isVideoRecord 
        self.isTriggered = isTriggered
        self.triggerNIDev = triggerNIDev
        self.triggerNIPort = triggerNIPort
        self.triggerNILine = triggerNILine
        self.triggerType = triggerType
        self.isSyncPulse = isSyncPulse
        self.syncPulseNIDev = syncPulseNIDev
        self.syncPulseNIPort = syncPulseNIPort
        self.syncPulseNILine = syncPulseNILine
        self.displayScreen = displayScreen
        self.initialBackgroundColor = initialBackgroundColor
        self.videoRecordIP = videoRecordIP
        self.videoRecordPort = videoRecordPort
        
        if displayIteration % 1 == 0:
            self.displayIteration = displayIteration
        else:
            raise ArithmeticError, "displayIteration should be a whole number."
            
        self.displayOrder = displayOrder
        self.logdir = logdir
        self.backupdir = backupdir
        self.mouseid = mouseid
        self.userid = userid
        self.sequenceLog = None
        
        #FROM DW
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        self.clear()

        
    def setAnyArray(self, anyArray, logDict = None):
        '''
        to display any numpy 3-d array.
        '''
        if len(anyArray.shape) != 3:
            raise LookupError, "Input numpy array should have dimension of 3!"
        
        Vmax = np.amax(anyArray).astype(np.float32)
        Vmin = np.amin(anyArray).astype(np.float32)
        Vrange = (Vmax-Vmin).astype(np.float32)
        anyArrayNor = ((anyArray-Vmin)/Vrange).astype(np.float32)
        self.sequence = np.round(255.*anyArrayNor).astype(np.uint8)

        if logDict != None:
            if type(logDict) is dict:
                self.sequenceLog = logDict
            else:
                raise ValueError, '"logDict" should be a dictionary!'
        else:
            self.sequenceLog = {}
        self.clear()
    
    
    def setStim(self, stim):
        '''
        to display defined stim object
        '''
        self.sequence, self.sequenceLog = stim.generateMovie()
        self.clear


    def triggerDisplay(self):
        
        if self.sequence is None: raise LookupError, "Please set the sequence to be displayed!!"
        
        try: resolution = self.sequenceLog['monitor']['resolution'][::-1]
        except KeyError: resolution = (800,600)
           
        window = visual.Window(size=resolution, fullscr=True, screen = self.displayScreen, color = self.initialBackgroundColor)
        stim = visual.ImageStim(window, size=(2,2))
        
        try: refreshRate = self.sequenceLog['monitor']['refreshRate']
        except KeyError:
            print "No monitor refresh rate information, assuming 60Hz."
            refreshRate = 60.

        displayTime = float(self.sequence.shape[0]) * self.displayIteration / refreshRate
        
        print '\n Expected display time: ', displayTime, ' seconds' 
        
        if self.isTriggered:
            self._waitForTrigger()

        self._getFileName()

        if self.isVideoRecord: self.sock.sendto("1"+self.fileName, (self.videoRecordIP, self.videoRecordPort)) #start eyetracker

        self._display(window, stim) #display sequence

        if self.isVideoRecord: self.sock.sendto("0"+self.fileName,(self.videoRecordIP,self.videoRecordPort)) #end eyetracker
        
        #analyze frames
        try: self.frameDuration, self.frameStats = analysisFrames(ts = self.timeStamp, refreshRate = self.sequenceLog['monitor']['refreshRate'])
        except KeyError:
            print "No monitor refresh rate information, assuming 60Hz."
            self.frameDuration, self.frameStats = analysisFrames(ts = self.timeStamp, refreshRate = 60.)
        
        #write log file
        self.saveLog()
        
        #clear display data
        self.clear()


    def _waitForTrigger(self):
        '''
        time place holder for waiting for trigger
        '''

        #check NI signal
        DI = iodaq.DigitalInput(self.triggerNIDev, self.triggerNIPort)
        DI.StartTask()

        if self.triggerType == 'LowLevel':
            lastTTL = DI.Read()[self.triggerNILine]
            while lastTTL != 0:lastTTL = DI.Read()[self.triggerNILine]
        elif self.triggerType == 'HighLevel':
            lastTTL = DI.Read()[self.triggerNILine]
            while lastTTL != 1:lastTTL = DI.Read()[self.triggerNILine]
        elif self.triggerType == 'NegativeEdge':
            lastTTL = DI.Read()[self.triggerNILine]
            while True:
                currentTTL = DI.Read()[self.triggerNILine]
                if (lastTTL == 1) and (currentTTL == 0):break
                else:lastTTL = int(currentTTL)
        elif self.triggerType == 'PositiveEdge':
            lastTTL = DI.Read()[self.triggerNILine]
            while True:
                currentTTL = DI.Read()[self.triggerNILine]
                if (lastTTL == 0) and (currentTTL == 1):break
                else:lastTTL = int(currentTTL)
        else:raise NameError, 'trigger should be one of "NegativeEdge", "PositiveEdge", "HighLevel", or "LowLevel"!'

        DI.StopTask()


    def _getFileName(self):
        '''
        generate the file name of log file
        '''

        try:
            self.fileName = datetime.datetime.now().strftime('%y%m%d%H%M%S') + \
                            '-' + \
                            self.sequenceLog['stimulation']['stimName'] + \
                            '-mouse' + \
                            self.mouseid + \
                            '-' + \
                            self.userid
        except KeyError:
            self.fileName = datetime.datetime.now().strftime('%y%m%d%H%M%S') + \
                            '-' + 'customStim' + '-mouse' + self.mouseid + '-' + \
                            self.userid

        if self.isTriggered: self.fileName += '-' + str(self.fileNumber)
        else: self.fileName += '-notTriggered'


    def _getFileNumber(self):
        '''
        get synced file number for log file name
        '''


        #------------------This piece of code needs to be improved--------------------
        try:
            #generate synchronized file number to write into the file name
            #file nmuber is coded as binary from digital input
            #port0 line0-7 and port1 line 0-2
            DI0 = iodaq.DigitalInput('Dev1', 0)
            DI1 = iodaq.DigitalInput('Dev1', 1)
            DI0.StartTask()
            DI1.StartTask()
            array0 = DI0.Read()
            array1 = DI1.Read()
            #reverse binary string of digital input port0 line0-7
            str0 = ''.join(map(str,array0))[::-1]
            #reverse binary string of digital input port1 line0-2
            str1 = ''.join(map(str,array1))[-2::-1]
            self.fileNumber = int(str1 + str0,2)
            DI0.StopTask()
            DI1.StopTask()
        except:
            self.fileNumber = None
        #------------------This piece of code needs to be improved--------------------


    def _display(self, window, stim):
        
        if self.sequence is None:
            raise LookupError, "Please set the sequence to be displayed!!"
        
        iteration = self.displayIteration
        order = self.displayOrder
        
        try:
            sequenceFrames = self.sequenceLog['stimulation']['frames']
            
            if order == -1: sequenceFrames = sequenceFrames[::-1]
            
            # generate display Frames
            displayFrames=()
            for i in range(iteration):
                displayFrames += sequenceFrames
            self.displayFrames = displayFrames
        except Exception as e:
            print e
            print "No frame information in sequenceLog dictionary. \nSetting displayFrames to 'None'."
            self.displayFrames = None
        
        
        # display frames
        timeStamp=[]
        startTime = time.clock()
        singleRunFrames = self.sequence.shape[0]
        
        if self.isSyncPulse:
            syncPulse = iodaq.DigitalOutput(self.syncPulseNIDev, self.syncPulseNIPort)
            syncPulse.StartTask()
            syncPulse.WriteBit(self.syncPulseNILine,0)
        
        for i in range(singleRunFrames * iteration):
            
            if order == 1:frameNum = i % singleRunFrames
                
            if order == -1:frameNum = singleRunFrames - (i % singleRunFrames) -1
                
            currFrame=Image.fromarray(self.sequence[frameNum])
            stim.setImage(currFrame)
            stim.draw()
            timeStamp.append(time.clock()-startTime)
            
            #check keyboard input 'q' or 'escape'
            keyList = event.getKeys(['q','escape'])
            if len(keyList) > 0:self.fileName = self.fileName + '-incomplete';break
            
            #set syncPuls signal
            if self.isSyncPulse:syncPulse.WriteBit(self.syncPulseNILine,1)
            
            #show visual stim
            window.flip()
            
            #set syncPuls signal
            if self.isSyncPulse:syncPulse.WriteBit(self.syncPulseNILine,0)
            
        timeStamp.append(time.clock()-startTime)
        stopTime = time.clock()
        window.close()
        
        if self.isSyncPulse:syncPulse.StopTask()
        
        self.timeStamp = np.array(timeStamp)
        self.displayLength = stopTime-startTime
    
    
    def setDisplayOrder(self, displayOrder):
        
        self.displayOrder = displayOrder
        self.clear()
    
    
    def setDisplayIteration(self, displayIteration):
        
        if displayIteration % 1 == 0:self.displayIteration = displayIteration
        else:raise ArithmeticError, "displayIteration should be a whole number."
        self.clear()
        
    
    def saveLog(self):
        
        if self.displayLength == None:
            self.clear()
            raise LookupError, "Please display sequence first!"
        
        #set up log object
        directory = self.logdir + '\sequence_display_log'
        if not(os.path.isdir(directory)):os.makedirs(directory)
        
        logFile = dict(self.sequenceLog)
        displayLog = dict(self.__dict__)
        displayLog.pop('sequenceLog')
        displayLog.pop('sock')
        displayLog.pop('sequence')
        logFile.update({'presentation':displayLog})
        
        filename = self.fileName + ".pkl"
        
        #generate full log dictionary
        path = os.path.join(directory, filename)
        output = open(path,'wb')
        pickle.dump(logFile,output)
        output.close()
        print ".pkl file generated successfully."
        
        if self.backupdir:
            backupfolder = self.backupdir + r'\sequence_display_log'
            if not(os.path.isdir(backupfolder)):os.makedirs(backupfolder)
            backuppath = os.path.join(backupfolder,filename)
            backupoutput = open(backuppath,'wb')
            pickle.dump(logFile,backupoutput)
            backupoutput.close()
            print ".pkl backup file generate successfully"
            
    
    def clear(self):
        ''' clear display information. '''
        self.displayLength = None
        self.timeStamp = None
        self.frameDuration = None
        self.displayFrames = None
        self.frameStats = None
        self.fileName = None
        self.fileNum = None


if __name__ == "__main__":

    #==============================================================================================================================
    # mon=MonitorJun(resolution=(1080, 1920),dis=13.5,monWcm=88.8,monHcm=50.1,C2Tcm=33.1,C2Acm=46.4,monTilt=16.22,downSampleRate=20)
    # indicator=IndicatorJun(mon)
    # KSstim=KSstimJun(mon,indicator)
    # ds=DisplaySequence(logdir=r'C:\data',backupdir=None,isTriggered=False,isSyncPulse=False)
    # ds.setStim(KSstim)
    # ds.triggerDisplay()
    # plt.show()
    #==============================================================================================================================

    #==============================================================================================================================
    # mon=MonitorJun(resolution=(1080, 1920),dis=13.5,monWcm=88.8,monHcm=50.1,C2Tcm=33.1,C2Acm=46.4,monTilt=30,downSampleRate=20)
    # monitorPoints = np.transpose(np.array([mon.degCorX.flatten(),mon.degCorY.flatten()]))
    # indicator=IndicatorJun(mon)
    # SparseNoiseStim=SparseNoise(mon,indicator, subregion=(-20.,20.,40.,60.))
    # gridPoints = SparseNoiseStim._generateGridPoints()
    # plt.plot(monitorPoints[:,0],monitorPoints[:,1],'or',mec='#ff0000',mfc='none')
    # plt.plot(gridPoints[:,0],gridPoints[:,1],'.k')
    # plt.show()
    #==============================================================================================================================

    #==============================================================================================================================
    # mon=MonitorJun(resolution=(1080, 1920),dis=13.5,monWcm=88.8,monHcm=50.1,C2Tcm=33.1,C2Acm=46.4,monTilt=30,downSampleRate=20)
    # monitorPoints = np.transpose(np.array([mon.degCorX.flatten(),mon.degCorY.flatten()]))
    # indicator=IndicatorJun(mon)
    # SparseNoiseStim=SparseNoise(mon,indicator,subregion=(-20.,20.,40.,60.))
    # gridPoints = SparseNoiseStim._getGridPointsSequence()
    #==============================================================================================================================

    #==============================================================================================================================
    # mon=MonitorJun(resolution=(1080, 1920),dis=13.5,monWcm=88.8,monHcm=50.1,C2Tcm=33.1,C2Acm=46.4,monTilt=30,downSampleRate=20)
    # monitorPoints = np.transpose(np.array([mon.degCorX.flatten(),mon.degCorY.flatten()]))
    # indicator=IndicatorJun(mon)
    # SparseNoiseStim=SparseNoise(mon,indicator,subregion=(-20.,20.,40.,60.))
    # SparseNoiseStim.generateFramesList()
    #==============================================================================================================================

    #==============================================================================================================================
    # mon = MonitorJun(resolution=(1080, 1920),dis=13.5,monWcm=88.8,monHcm=50.1,C2Tcm=33.1,C2Acm=46.4,monTilt=30,downSampleRate=5)
    # frame = getWarpedFrameWithSquare(mon.degCorX,mon.degCorY,(20.,25.),4.,4.,0.,foregroundColor=1,backgroundColor=0)
    # plt.imshow(frame,cmap='gray',vmin=-1,vmax=1,interpolation='nearest')
    # plt.show()
    #==============================================================================================================================

    #==============================================================================================================================
    mon=MonitorJun(resolution=(1080, 1920),dis=13.5,monWcm=88.8,monHcm=50.1,C2Tcm=33.1,C2Acm=46.4,monTilt=30,downSampleRate=5)
    monitorPoints = np.transpose(np.array([mon.degCorX.flatten(),mon.degCorY.flatten()]))
    indicator=IndicatorJun(mon)
    SparseNoiseStim=SparseNoise(mon,indicator)
    ds=DisplaySequence(logdir=r'C:\data',backupdir=None,isTriggered=False,isSyncPulse=False,isVideoRecord=False)
    ds.setStim(SparseNoiseStim)
    ds.triggerDisplay()
    plt.show()
    #==============================================================================================================================


    print 'for debug...'