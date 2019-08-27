__author__ = 'junz'

import numpy as np
import os
import scipy.ndimage as ni
import scipy.stats as stats
import scipy.sparse as sparse
import math
import matplotlib.pyplot as plt
from itertools import combinations
from operator import itemgetter
import skimage.morphology as sm
import skimage.transform as tsfm
import cv2
import matplotlib.colors as col
from matplotlib import cm

import corticalmapping.core.FileTools as ft
import corticalmapping.core.ImageAnalysis as ia
import corticalmapping.core.PlottingTools as pt
import imaging_behavior.core.tifffile as tf



def loadTrial(trialPath):
    '''
    load single retinotopic mapping trial from database
    '''

    trialDict = ft.loadFile(trialPath)

    try:
        traces = trialDict['traces']
    except KeyError:
        traces = []

    trial = RetinotopicMappingTrial(mouseID = trialDict['mouseID'], # str, mouseID
                                    dateRecorded = trialDict['dateRecorded'], # int, date recorded, yearmonthday
                                    trialNum = trialDict['trialNum'], # str, number of the trail on that day
                                    mouseType = trialDict['mouseType'], # str, mouse Genotype
                                    visualStimType = trialDict['visualStimType'], # str, stimulation type
                                    visualStimBackground = trialDict['visualStimBackground'], # str, background of visual stimulation
                                    imageExposureTime = trialDict['imageExposureTime'], # float, exposure time of image file
                                    altPosMap = trialDict['altPosMap'], # altitude position map
                                    aziPosMap = trialDict['aziPosMap'], # azimuth position map
                                    altPowerMap = trialDict['altPowerMap'], # altitude power map
                                    aziPowerMap = trialDict['aziPowerMap'], # azimuth power map
                                    vasculatureMap = trialDict['vasculatureMap'], # vasculature map
                                    params = trialDict['params'],
                                    isAnesthetized = trialDict['isAnesthetized'])

    try:
        trial.altPosMapf = trialDict['altPosMapf']
    except KeyError:
        pass

    try:
        trial.aziPosMapf = trialDict['aziPosMapf']
    except KeyError:
        pass

    try:
        trial.altPowerMapf = trialDict['altPowerMapf']
    except KeyError:
        pass

    try:
        trial.aziPowerMapf = trialDict['aziPowerMapf']
    except KeyError:
        pass

    try:
        if isinstance(list(trialDict['finalPatches'].values())[0],dict):
            trial.finalPatches = {}
            for area,patchDict in trialDict['finalPatches'].items():
                try:trial.finalPatches.update({area:Patch(patchDict['array'],patchDict['sign'])})
                except KeyError:trial.finalPatches.update({area:Patch(patchDict['sparseArray'],patchDict['sign'])})
        else: pass
    except KeyError:
        pass

    try:
        if isinstance(list(trialDict['finalPatchesMarked'].values())[0],dict):
            trial.finalPatchesMarked = {}
            for area,patchDict in trialDict['finalPatchesMarked'].items():
                try:trial.finalPatchesMarked.update({area:Patch(patchDict['array'],patchDict['sign'])})
                except KeyError:trial.finalPatchesMarked.update({area:Patch(patchDict['sparseArray'],patchDict['sign'])})
        else: pass
    except KeyError:
        pass

    try:
        trial.signMap = trialDict['signMap']
    except KeyError:
        pass

    try:
        trial.signMapf = trialDict['signMapf']
    except KeyError:
        pass

    try:
        trial.rawPatchMap = trialDict['rawPatchMap']
    except KeyError:
        pass

    try:
        trial.rawPatches = trialDict['rawPatches']
    except KeyError:
        pass

    try:
        trial.eccentricityMapf = trialDict['eccentricityMapf']
    except KeyError:
        pass

    return trial, traces


def findPhaseIndex(trace, harmonic=1):
    '''

    return the index of the peak at the given harmonic component of a singal trace

    :param trace: signal
    :param harmonic: the interested harmonic
    :return: index of the peak of the given harmonic
    '''
    traceF = np.fft.fft(trace)
    angle = (-1*np.angle(traceF[harmonic]))%(2*np.pi)
    index = len(trace) * angle / (2*np.pi)

    return index


def generatePhaseMap(movie,cycles = 1,isReverse = False,isFilter = False,sigma = 3.,isplot = False):
    '''
    generating phase map of a 3-d movie, on the frequency defined by cycles.

    the movie should have the same length of 'cycles' number of cycles.

    isFilter: Ture      use sigma filtering
              False     not use sigma filtering

    sigma is the criterion to choose pixels for plotting. Only the pixels
    meet the following criterion will be plotted:

    for this particular pixel: the power at the desired frequency is bigger
    than 'sigma' standard deviation from the mean power of all frequencies.

    '''
    if isReverse:
        movie = np.amax(movie) - movie

    spectrumMovie = np.fft.fft(movie,axis=0)

    #generate power movie
    powerMovie = (np.abs(spectrumMovie) * 2.) / np.size(movie, 0)
    powerMap = np.abs(powerMovie[cycles,:,:])

    #generate phase movie
    phaseMovie = np.angle(spectrumMovie)
    #phaseMap = phaseMovie[cycles,:,:]
    phaseMap = -1 * phaseMovie[cycles,:,:]

    #remove pixels with not enough power in the ideal frequency
    if isFilter == True:
        meanPower = np.mean(powerMovie, axis = 0)
        stdPower = np.std(powerMovie, axis = 0)
        for i in np.arange(powerMap.shape[0]):
            for j in np.arange(powerMap.shape[1]):
                if powerMap[i,j] < meanPower[i,j] + sigma * stdPower[i,j]:
                    phaseMap[i,j] = np.nan

    if isplot == True:
        plt.figure()
        plotMap = 180 * (phaseMap / np.pi)
        plt.imshow(plotMap,aspect='equal')
        plt.colorbar()

    return phaseMap % (2*np.pi), powerMap #value from -pi to pi


def generatePhaseMap2(movie,cycles,isReverse = False,isPlot = False):
    '''
    generating phase map of a 3-d movie, on the frequency defined by cycles.
    the movie should have the same length of 'cycles' number of cycles.
    '''
    if isReverse:
        movie = np.amax(movie) - movie

    spectrumMovie = np.fft.fft(movie, axis=0)

    #generate power movie
    powerMovie = (np.abs(spectrumMovie) * 2.) / np.size(movie, 0)
    powerMap = np.abs(powerMovie[cycles,:,:])

    #generate phase movie
    phaseMovie = np.angle(spectrumMovie)
    #phaseMap = phaseMovie[cycles,:,:]
    phaseMap = -1 * phaseMovie[cycles,:,:]
    phaseMap = phaseMap % (2 * np.pi)

    if isPlot == True:
        plt.figure()
        plotMap = 180 * (phaseMap / np.pi)
        plt.imshow(plotMap,
                   aspect='equal',
                   cmap = 'hsv',
                   vmax = 360,
                   vmin = 0,
                   interpolation = 'nearest')
        plt.colorbar()

    return phaseMap, powerMap
#
# def generatePhaseMap3(movie, filter_size=None, isReverse=False, isPlot=False):
#     '''
#     find the phase by maximum, the movie should be df movie and only contains one cycle
#     '''
#
#     if isReverse:
#         movie = np.amax(movie) - movie
#
#     if filter_size is not None:
#         movie = ni.filters.gaussian_filter(movie, sigma=(filter_size, 0., 0.))
#
#     powerMap = np.sum(movie, axis=0)
#
#     movie = movie.astype(np.float32)
#     movie[movie < 0.] = 0.
#
#     # phaseMap = np.argmax(movie, axis=0) * 2 * np.pi / float(movie.shape[0])
#
#     weighted_sum = np.zeros((movie.shape[1], movie.shape[2]))
#     for i in range(movie.shape[0]):
#         weighted_sum = weighted_sum + movie[i, :, :] * i
#     total_sum = np.sum(movie, axis=0)
#     weighted_average = weighted_sum / total_sum
#     phaseMap = weighted_average * 2 * np.pi / float(movie.shape[0])
#
#     if isPlot == True:
#         plt.figure()
#         plotMap = 180 * (phaseMap / np.pi)
#         plt.imshow(plotMap,
#                    aspect='equal',
#                    cmap = 'hsv',
#                    vmax = 360,
#                    vmin = 0,
#                    interpolation = 'nearest')
#         plt.colorbar()
#
#     return phaseMap, powerMap


def getPhase(trace,cycles,isReverse = False):
    '''
    return phase and power of a certain trace, the trace should have 'cycles' number of cycles
    the returned phase can be plugged direct into equation generated by getPhasePositionEquation() function to calculate
    retinotopic location of a pertical trace
    '''

    if trace.ndim != 1: raise ValueError('input trace should be a 1-d array!')

    if isReverse:
        trace = np.amax(trace) - trace

    spectrum = np.fft.fft(trace)
    power = ((np.abs(spectrum) * 2.) / spectrum.shape[0])[cycles]
    phase = ((-1 * np.angle(spectrum))[cycles]) % (2 * np.pi)

    return phase, power


def phasePosition(phaseMap, displayLog):
    '''
    from phaseMap get the position map, according to the KSstim display
    log
    '''

    if not('KSstim' in displayLog['stimulation']['stimName']):
        raise TypeError('The stimulation is not KSstim!')

    sweepTable = displayLog['stimulation']['sweepTable']
    frames = displayLog['stimulation']['frames']

    phase = np.linspace(0,2*np.pi,len(frames),endpoint=False)

    phaseIndexStart = np.nan
    phaseIndexEnd = np.nan
    phaseStart = np.nan
    phaseEnd = np.nan
    for i in range(0,len(frames)-1):
        if (frames[i][2] == None) & (frames[i+1][2] != None):
            phaseStart = phase[i+1]
            phaseIndexStart = int(i+1)
            #print phaseIndexStart, phaseStart

        if (frames[i][2] != None) & (frames[i+1][2] == None):
            phaseEnd = phase[i]
            phaseIndexEnd = int(i)
            #print phaseIndexEnd, phaseEnd

    if np.isnan(phaseIndexStart):
        print('no gap in the front.')
        phaseIndexStart = 0
        phaseStart = phase[0]

    if np.isnan(phaseIndexEnd):
        print('no gap in the end.')
        phaseIndexEnd = len(phase)
        phaseEnd = phase[-1]

    position = np.zeros(len(frames))
    position[:] = np.nan

    for i, framei in enumerate(frames):
        if framei[2] != None:
            position[i] = (sweepTable[framei[2]][1]+sweepTable[framei[2]][2])/2

    stiDirection = displayLog['stimulation']['direction']

    if displayLog['presentation']['displayOrder'] == -1:
        position = position[::-1]
        stiDirection = stiDirection[::-1]

    print('\nStimulus direction:', stiDirection)

    slope, intercept, r_value, p_value, stderr = stats.linregress(phase[phaseIndexStart:(phaseIndexEnd+1)],
                                                                  position[phaseIndexStart:(phaseIndexEnd+1)])

    positionMap = phaseMap * slope + intercept

    #print slope, intercept, phaseMap[74,66], phaseMap2[74,66], positionMap[74,66]

    #positionMap[phaseMap<phaseStart] = np.nan
    #positionMap[phaseMap>phaseEnd] = np.nan

    return positionMap


def getPhasePositionEquation(displayLog):
    '''
    return the intercept and slope of the linear relationship between phase and retinotopic location for a certain
    display of KSStim
    '''

    if not('KSstim' in displayLog['stimulation']['stimName']):
        raise TypeError('The stimulation is not KSstim!')

    sweepTable = displayLog['stimulation']['sweepTable']
    frames = displayLog['stimulation']['frames']

    phase = np.linspace(0,2*np.pi,len(frames),endpoint=False)

    phaseIndexStart = np.nan
    phaseIndexEnd = np.nan
    for i in range(0,len(frames)-1):
        if (frames[i][2] == None) & (frames[i+1][2] != None):phaseIndexStart = int(i+1)
        if (frames[i][2] != None) & (frames[i+1][2] == None):phaseIndexEnd = int(i)

    if np.isnan(phaseIndexStart):print('no gap in the front.'); phaseIndexStart = 0

    if np.isnan(phaseIndexEnd):print('no gap in the end.'); phaseIndexEnd = len(phase)

    position = np.zeros(len(frames))
    position[:] = np.nan

    for i, framei in enumerate(frames):
        if framei[2] is not None:
            position[i] = (sweepTable[framei[2]][1]+sweepTable[framei[2]][2])/2

    stiDirection = displayLog['stimulation']['direction']

    if displayLog['presentation']['displayOrder'] == -1:
        position = position[::-1]; stiDirection = stiDirection[::-1]

    # print '\nStimulus direction:', stiDirection

    slope, intercept, r_value, p_value, stderr = stats.linregress(phase[phaseIndexStart:(phaseIndexEnd+1)],
                                                                  position[phaseIndexStart:(phaseIndexEnd+1)])

    # print 'slope: \t'+str(slope)
    # print 'intercept: \t'+str(intercept)
    # print 'r_value: \t'+str(r_value)
    # print 'p_value: \t'+str(p_value)
    # print 'stderr: \t'+str(stderr)

    return slope, intercept


def getPhasePositionEquation2(frames, sweepTable):
    '''
    return the intercept and slope of the linear relationship between phase and retinotopic location from a frames and
    sweepTable (in standard corticalmapping.VisualStim.KSstimJun or corticalmapping.VisualStim.KSstimAllDir format).
    Assuming displayOrder is 1 when display this stimulus
    '''

    phase = np.linspace(0,2*np.pi,len(frames),endpoint=False)

    phaseIndexStart = np.nan
    phaseIndexEnd = np.nan
    for i in range(0,len(frames)-1):
        if (frames[i][2] == None) & (frames[i+1][2] != None):phaseIndexStart = int(i+1)
        if (frames[i][2] != None) & (frames[i+1][2] == None):phaseIndexEnd = int(i)

    if np.isnan(phaseIndexStart):print('no gap in the front.'); phaseIndexStart = 0
    if np.isnan(phaseIndexEnd):print('no gap in the end.'); phaseIndexEnd = len(phase)

    position = np.zeros(len(frames))
    position[:] = np.nan

    for i, framei in enumerate(frames):
        if framei[2] is not None:
            position[i] = (sweepTable[framei[2]][1]+sweepTable[framei[2]][2])/2

    # print '\nStimulus direction:', stiDirection

    slope, intercept, r_value, p_value, stderr = stats.linregress(phase[phaseIndexStart:(phaseIndexEnd+1)],
                                                                  position[phaseIndexStart:(phaseIndexEnd+1)])

    # print 'slope: \t'+str(slope)
    # print 'intercept: \t'+str(intercept)
    # print 'r_value: \t'+str(r_value)
    # print 'p_value: \t'+str(p_value)
    # print 'stderr: \t'+str(stderr)

    return slope, intercept


def visualSignMap(phasemap1,phasemap2):
    '''
    calculate visual sign map from two orthogonally oriented phase maps
    '''

    if phasemap1.shape != phasemap2.shape:
        raise LookupError("'phasemap1' and 'phasemap2' should have same size.")

    gradmap1 = np.gradient(phasemap1)
    gradmap2 = np.gradient(phasemap2)

    # gradmap1 = ni.filters.median_filter(gradmap1,100.)
    # gradmap2 = ni.filters.median_filter(gradmap2,100.)

    graddir1 = np.zeros(np.shape(gradmap1[0]))
    # gradmag1 = np.zeros(np.shape(gradmap1[0]))

    graddir2 = np.zeros(np.shape(gradmap2[0]))
    # gradmag2 = np.zeros(np.shape(gradmap2[0]))

    for i in  range(phasemap1.shape[0]):
        for j in range(phasemap2.shape[1]):

            graddir1[i,j] = math.atan2(gradmap1[1][i,j],gradmap1[0][i,j])
            graddir2[i,j] = math.atan2(gradmap2[1][i,j],gradmap2[0][i,j])

            # gradmag1[i,j] = np.sqrt((gradmap1[1][i,j]**2)+(gradmap1[0][i,j]**2))
            # gradmag2[i,j] = np.sqrt((gradmap2[1][i,j]**2)+(gradmap2[0][i,j]**2))

    vdiff = np.multiply(np.exp(1j * graddir1),np.exp(-1j * graddir2))

    areamap = np.sin(np.angle(vdiff))

    return areamap


def dilationPatches(rawPatches,smallPatchThr = 5,borderWidth = 1): #pixel width of the border after dilation

    '''
    dilation patched in a given area untill the border between them are as
    narrow as defined by 'borderWidth'.
    '''

    #get patch borders
    total_area = sm.convex_hull_image(rawPatches)
    patchBorder = np.multiply(-1 * (rawPatches - 1), total_area)

    #thinning patch borders
    patchBorder = sm.skeletonize(patchBorder)

    #thicking patch borders
    if borderWidth > 1:
        patchBorder = ni.binary_dilation(patchBorder, iterations = borderWidth - 1).astype(np.int)

    #genertating new patches
    newPatches = np.multiply(-1 * (patchBorder - 1), total_area)

    #removing small edges
    labeledPatches, patchNum = ni.label(newPatches)

    for i in range(1, patchNum + 1):
        currPatch = np.array(labeledPatches)
        currPatch[currPatch != i] = 0
        currPatch = currPatch / i

        if (np.sum(np.multiply(currPatch, rawPatches)[:]) == 0) or (np.sum(currPatch[:]) < smallPatchThr):
            #revCurrPatch = -1 * (currPatch - 1)
            #newPatches = np.multiply(newPatches, revCurrPatch)
            newPatches[currPatch == 1] = 0

        else:
            currPatch = ni.binary_closing(currPatch,
                                          structure = np.ones((borderWidth+2,borderWidth+2))).astype(np.int)
            newPatches[currPatch == 1] = 1

    return newPatches


def dilationPatches2(rawPatches,dilationIter = 20,borderWidth = 1): #pixel width of the border after dilation

    '''
    dilation patched in a given area untill the border between them are as
    narrow as defined by 'borderWidth'.
    '''


    total_area = ni.binary_dilation(rawPatches, iterations = dilationIter).astype(np.int)
    patchBorder = total_area - rawPatches

    #thinning patch borders
    patchBorder = sm.skeletonize(patchBorder)

    #thickening patch borders
    if borderWidth > 1:
        patchBorder = ni.binary_dilation(patchBorder, iterations = borderWidth - 1).astype(np.int)

    #genertating new patches
    newPatches = np.multiply(-1 * (patchBorder - 1), total_area)

    #removing small edges
    labeledPatches, patchNum = ni.label(newPatches)

    newPatches2 = np.zeros(newPatches.shape, dtype = np.int)

    for i in range(1, patchNum + 1):
        currPatch = np.zeros(labeledPatches.shape, dtype = np.int)
        currPatch[labeledPatches == i] = 1
        currPatch[labeledPatches != i] = 0

        if (np.sum(np.multiply(currPatch, rawPatches)[:]) > 0):
#            currPatch = ni.binary_closing(currPatch,
#                                          structure = np.ones((borderWidth+2,borderWidth+2))).astype(np.int)
            newPatches2[currPatch == 1] = 1

    return newPatches2


def labelPatches(patchmap, signMap, connectivity=4):
    '''
    from a segregated patchmap generate a dictionary with each entry represents
    a single patch, sorted by area
    '''

    labeledPatches, patchNum = ni.label(patchmap)

    #list of area of every patch, first column: patch label, second column: area
    patchArea = np.zeros((patchNum,2),dtype=np.int)

    for i in range(1, patchNum+1):
        currPatch = np.zeros(labeledPatches.shape, dtype = np.int)
        currPatch[labeledPatches == i] = 1
        currPatch[labeledPatches != i] = 0
        patchArea[i-1] = [i, np.sum(currPatch[:])]

    #sort patches by the area, from largest to the smallest
    sortArea=patchArea[patchArea[:,1].argsort(axis=0)][::-1,:]

    patches = {}
    for i, ind in enumerate(sortArea[:,0]):
        currPatch = np.zeros(labeledPatches.shape, dtype = np.int)
        currPatch[labeledPatches == ind] = 1
        currPatch[labeledPatches != ind] = 0
        currSignPatch = np.multiply(currPatch, signMap)

        if np.sum(currSignPatch[:]) > 0:
            currSign = 1
        elif np.sum(currSignPatch[:]) < 0:
            currSign = -1
        else:
            raise LookupError('This patch has no visual Sign!!')

        patchname = 'patch' + ft.int2str(i, 2)

        patches.update({patchname : Patch(currPatch, currSign)})

    return patches


def phaseFilter(phaseMap,
                filterType = 'Gaussian', # 'Gaussian' of 'uniform'
                filterSize = 3,
                isPositive = True): #if Ture return phase [0 2pi], if False return phase [-pi, pi]
    '''
    smooth phaseMap in a circular fashion
    '''

    phaseMapSin = np.sin(phaseMap)
    phaseMapCos = np.cos(phaseMap)

    if filterType == 'Gaussian':
        phaseMapSinf = ni.filters.gaussian_filter(phaseMapSin, filterSize)
        phaseMapCosf = ni.filters.gaussian_filter(phaseMapCos, filterSize)

    if filterType == 'uniform':
        phaseMapSinf = ni.filters.uniform_filter(phaseMapSin, filterSize)
        phaseMapCosf = ni.filters.uniform_filter(phaseMapCos, filterSize)

    phaseMapf = np.arctan2(phaseMapSinf, phaseMapCosf)

    if isPositive:
        phaseMapf = phaseMapf % (2 * np.pi)

    return phaseMapf


def plotVisualSpace(visualSpace, altAxis, aziAxis, tickSpace=10, plotAxis=None, lineColor='#000000', lineWidth=2):
    '''
    plot visual space in given plotAxis

    tidkSpace: int, gap between labelled ticks in pixels
    '''

    if plotAxis is None:
        f = plt.figure()
        ax = f.add_subplot(111)
    else:
        ax = plotAxis

    pt.plot_mask_borders(visualSpace, plotAxis=ax, color=lineColor, borderWidth=lineWidth)
    ax.set_aspect('equal')
    altTickInds = list(range(len(altAxis)))[::tickSpace]
    aziTickInds = list(range(len(aziAxis)))[::tickSpace]

    altTickLabels = [str(int(round(altAxis[altTickInd]))) for altTickInd in altTickInds]
    aziTickLabels = [str(int(round(aziAxis[aziTickInd]))) for aziTickInd in aziTickInds]

    ax.set_aspect(1)
    ax.set_xticks(aziTickInds)
    ax.set_xticklabels(aziTickLabels)
    ax.set_yticks(altTickInds)
    ax.set_yticklabels(altTickLabels)
    ax.invert_yaxis()


def localMin(eccMap, binSize):

    '''
    find local minimum of eccenticity map (in degree), with binning by binSize
    in degree
    '''

    eccMap2 = np.array(eccMap)
    cutStep = np.arange(np.nanmin(eccMap2[:]) - binSize,
                        np.nanmax(eccMap2[:]) + binSize * 2,
                        binSize)

    # cutStep = np.arange(0, np.nanmax(eccMap2.flatten()) + binSize * 2, binSize)

    NumOfMin = 0
    i = 0
    while (NumOfMin <= 1) and (i < len(cutStep)):
        currThr = cutStep[i]
        marker = np.zeros(eccMap.shape, dtype=np.int)
        marker[eccMap2 <= (currThr)] = 1
        marker, NumOfMin = ni.measurements.label(marker)
        i = i + 1

#    if NumOfMin == 1:
#        print 'Only one local minumum was found!!!'
#    elif NumOfMin == 0:
#        print 'No local minumum was found!!!'
#    else:
#        print str(NumOfMin) + ' local minuma were found!!!'
#
#    if NumOfMin > 1:
#        plt.figure()
#        plt.imshow(marker,vmin=np.amin(marker), vmax=np.amax(marker),cmap='jet',interpolation='nearest')
#        plt.colorbar()
#        plt.title('marker from local min')

    return marker


def adjacentPairs(patches,borderWidth = 2):

    '''
    return all the patch pairs with same visual sign and sharing border
    '''

    keyList = list(patches.keys())
    pairKeyList = []

    for pair in combinations(keyList, 2):
        patch1 = patches[pair[0]]
        patch2 = patches[pair[1]]

        if (ia.is_adjacent(patch1.array, patch2.array, borderWidth = borderWidth)) and (patch1.sign == patch2.sign):

            pairKeyList.append(pair)

    return pairKeyList


def mergePatches(array1, array2, borderWidth = 2):
    '''
    merge two binary patches with borderWidth no greater than borderWidth
    '''

    sp = array1 + array2
    spc =  ni.binary_closing(sp, iterations = (borderWidth)).astype(np.int8)

    _, patchNum = ni.measurements.label(spc)
    if patchNum > 1:
        raise LookupError('this two patches are too far apart!!!')
    else:
        return spc


def eccentricityMap(altMap, aziMap, altCenter, aziCenter):
    '''
    calculate eccentricity map of with defined center

    altMap, aziMap, altCenter, aziCenter: in degree

    eccentricity map is returned in degree
    '''

    altMap2 = altMap * np.pi / 180
    aziMap2 = aziMap * np.pi / 180

    altCenter2 = altCenter * np.pi / 180
    aziCenter2 = aziCenter * np.pi / 180

    eccMap = np.zeros(altMap.shape)
    eccMap[:] = np.nan
#    for i in xrange(altMap.shape[0]):
#        for j in xrange(altMap.shape[1]):
#            alt = altMap2[i,j]
#            azi = aziMap2[i,j]
#            eccMap[i,j] = np.arctan(np.sqrt(np.tan(alt-altCenter2)**2 + ((np.tan(azi-aziCenter2)**2)/(np.cos(alt-altCenter2)**2))))
    eccMap = np.arctan(
                       np.sqrt(
                               np.square(np.tan(altMap2-altCenter2))
                               +
                               np.square(np.tan(aziMap2-aziCenter2))/np.square(np.cos(altMap2-altCenter2))
                               )
                       )

    eccMap = eccMap*180 / np.pi
    return eccMap


def sortPatches(patchDict):
    '''
    from a patch dictionary generate an new dictionary with patches sorted by there area
    '''

    patches = []
    newPatchDict = {}

    for key, value in patchDict.items():
        patches.append((value,value.getArea()))

    patches = sorted(patches, key=lambda a:a[1], reverse=True)

    for i, item in enumerate(patches):

        patchName = 'patch' + ft.int2str(i + 1, 2)

        newPatchDict.update({patchName:item[0]})

    return newPatchDict


def plotPatches(patches,plotaxis = None,zoom = 1,alpha = 0.5,markersize = 5):
    '''
    plot a patches in a patch dictionary
    '''

    if plotaxis == None:
        f = plt.figure()
        plotaxis = f.add_axes([1,1,1,1])

    imageHandle = {}
    for key, value in patches.items():

        if zoom > 1:
            currPatch = Patch(ni.zoom(value.array, zoom, order = 0),value.sign)
        else:
            currPatch = value

        h = plotaxis.imshow(currPatch.getSignedMask(),vmax=1,vmin=-1,interpolation='nearest',alpha=alpha)
        plotaxis.plot(currPatch.getCenter()[1], currPatch.getCenter()[0],'.k', markersize = markersize * zoom)
        imageHandle.update({'handle_'+key:h})

    plotaxis.set_xlim([0, currPatch.array.shape[1]-1])
    plotaxis.set_ylim([currPatch.array.shape[0]-1, 0])
    # plotaxis.set_axis_off()
    return imageHandle


def plotPatchBorders(patches,plotaxis = None,borderWidth = 2,color='#ff0000',zoom = 1,isPlotCenter = True,isCenter = True,
                     rotationAngle = 0 ):# rotation of map in degrees, counter-clockwise

    #generating plot axis
    if plotaxis == None:
        f = plt.figure()
        plotaxis = f.add_subplot(111)

    cmap1 = col.ListedColormap(color, 'temp')
    cm.register_cmap(cmap=cmap1)

    borderArray = []

    #initiating center and area
    center = None
    area = 0

    for key, value in patches.items():

        if zoom > 1:
            currPatch = Patch(ni.zoom(value.array, zoom, order = 0),value.sign)
            currBorderWidth = borderWidth * zoom
        else:
            currPatch = value
            currBorderWidth = borderWidth

        #updating center
        currArea = currPatch.getArea()
        currCenter = currPatch.getCenter()
        if currArea > area:
            center = currCenter
            area = np.int(currArea)

        #print 'currArea:', currArea, '   currCenter:', currCenter, '    center:', center

        #generating border array for the current patch
        currBorder = currPatch.getBorder(borderWidth = currBorderWidth)

        #adding center of current patches to the border array
        if isPlotCenter:
            currBorder[currCenter[0]-currBorderWidth-1:currCenter[0]+currBorderWidth+1,
                       currCenter[1]-currBorderWidth-1:currCenter[1]+currBorderWidth+1] = 1
        currBorder[np.isnan(currBorder)] = 0

        borderArray.append(currBorder)

    #binarize border array
    borderArray = np.sum(np.array(borderArray),axis=0)
    borderArray[borderArray >= 1] = 1

    # centering and expanding border array
    if isCenter:
        NW = np.array([0, 0])
        NE = np.array([0, borderArray.shape[1]])
        SW = np.array([borderArray.shape[0], 0])
        SE = np.array([borderArray.shape[0], borderArray.shape[1]])

        # calculate maximum distance to four corners
        maxDis = np.int(np.ceil(np.max([ia.distance(center, NW),
                                        ia.distance(center, NE),
                                        ia.distance(center, SW),
                                        ia.distance(center, SE)
                                        ])))

        # calculate expansion distance to all four directions
        expandN = maxDis - center[0]
        expandS = maxDis - (borderArray.shape[0] - center[0])
        expandE = maxDis - center[1]
        expandW = maxDis - (borderArray.shape[1] - center[1])

        borderArray = np.concatenate((np.zeros((expandN,borderArray.shape[1])),borderArray),axis = 0)
        borderArray = np.concatenate((borderArray,np.zeros((expandS,borderArray.shape[1]))),axis = 0)
        borderArray = np.concatenate((np.zeros((borderArray.shape[0],expandE)),borderArray),axis = 1)
        borderArray = np.concatenate((borderArray,np.zeros((borderArray.shape[0],expandW))),axis = 1)


    # rotating border array
    borderArrayR = tsfm.rotate(borderArray, rotationAngle)
    #binarize rotated border array
    borderArrayR[borderArrayR > 0]=1

    #thinning rotated border array
    #borderArrayR = sm.binary_opening(borderArrayR,np.array([[0,1,0],[1,1,1],[0,1,0]]))
    borderArrayR = sm.skeletonize(borderArrayR)

    #dilating rotated border array
    borderArrayR = sm.binary_dilation(borderArrayR, sm.square(currBorderWidth))

    #clear unwanted pixels
    borderR = np.array(borderArrayR).astype(np.float32)
    borderR[borderArrayR == 0] = np.nan

    #plotting
    imageHandle = plotaxis.imshow(borderR, vmin=0, vmax=1, cmap='temp', interpolation = 'nearest')

    return imageHandle


def plotPatchBorders2(patches,
                      plotAxis = None,
                      plotSize = None, # size of plotting area
                      borderWidth = 2,
                      zoom = 1,
                      centerPatch = 1,
                      rotationAngle = 0, # rotation of map in degrees, counter-clockwise
                      markerSize = 2, # size of center dot
                      closingIteration = None # open iteration for patch borders
                      ):

    '''
    plot rotated and centered patch borders

    centerPatch defines center at which patch
        1: center at the biggest patch
        2: center at the second biggest patch
        ...
    '''

    #generating plot axis
    if plotAxis == None:
        f = plt.figure()
        plotAxis = f.add_subplot(111)

    #generating list for plotting
    #for each patch: first item: center, second item: area, third item: patch array, forth item: sign
    forPlotting=[]

    for key, value in patches.items():


        currPatch = Patch(ni.zoom(value.array, zoom, order = 0),value.sign)

        forPlotting.append([currPatch.getCenter(),
                            currPatch.getArea(),
                            currPatch.getMask(),
                            value.sign])

    # sort borders with area: biggest to smalles
    forPlotting = sorted(forPlotting, key=lambda a: a[1], reverse=True)

    # get the plotting center
    center = forPlotting[centerPatch-1][0]

    # width and height of original plot
    width = forPlotting[0][2].shape[1]
    height = forPlotting[0][2].shape[0]

    #coordinate of four corners
    NW = np.array([0, 0])
    NE = np.array([0, width])
    SW = np.array([height, 0])
    SE = np.array([height, width])

    # calculate maximum distance to four corners
    maxDis = np.int(np.ceil(np.max([ia.distance(center, NW),
                                    ia.distance(center, NE),
                                    ia.distance(center, SW),
                                    ia.distance(center, SE)
                                    ])))

    # calculate expansion distance to all four directions
    expandN = maxDis - center[0]
    expandS = maxDis - (height - center[0])
    expandE = maxDis - center[1]
    expandW = maxDis - (width - center[1])

    for ind, value in enumerate(forPlotting):

        #expanding border map for each patch
        value[2] = np.concatenate((np.zeros((expandN,value[2].shape[1])),value[2]),axis = 0)
        value[2] = np.concatenate((value[2],np.zeros((expandS,value[2].shape[1]))),axis = 0)
        value[2] = np.concatenate((np.zeros((value[2].shape[0],expandE)),value[2]),axis = 1)
        value[2] = np.concatenate((value[2],np.zeros((value[2].shape[0],expandW))),axis = 1)

        value[2][value[2]==0] = np.nan

        #rotate border map for each patch
        value[2] = tsfm.rotate(value[2], rotationAngle)

        # #binarize current border map
        # value[2][value[2]<0.9]=np.nan
        # value[2][value[2]>=0.9]=1

        #ploting current border
        if value[3] == -1:
            pt.plot_mask(value[2], plotAxis=plotAxis, color='#0000ff', borderWidth = borderWidth, closingIteration = closingIteration)
        elif value[3] == 1:
            pt.plot_mask(value[2], plotAxis=plotAxis, color='#ff0000', borderWidth = borderWidth, closingIteration = closingIteration)

        # expanding center coordinate for each patch
        value[0][0] = value[0][0] + expandN
        value[0][1] = value[0][1] + expandE

        #rotate center coordinate for each patch
        x = value[0][1] - maxDis
        y = maxDis - value[0][0]

        xx = x*np.cos(rotationAngle*np.pi/180) - y*np.sin(rotationAngle*np.pi/180)
        yy = y*np.cos(rotationAngle*np.pi/180) + x*np.sin(rotationAngle*np.pi/180)

        value[0][0] = int(np.round(maxDis - yy))
        value[0][1] = int(np.round(maxDis + xx))

        #ploting current center
        if value[3] == -1:
            plotAxis.plot(value[0][1],value[0][0], '.b', markersize = markerSize)
        elif value[3] == 1:
            plotAxis.plot(value[0][1],value[0][0], '.r', markersize = markerSize)


    if plotSize:
        plotAxis.set_xlim([maxDis-plotSize/2, maxDis+plotSize/2])
        plotAxis.set_ylim([maxDis+plotSize/2, maxDis-plotSize/2])
    else:
        plotAxis.set_xlim([0,2*maxDis])
        plotAxis.set_ylim([2*maxDis,0])

    plotAxis.get_xaxis().set_visible(False)
    plotAxis.get_yaxis().set_visible(False)
    return forPlotting


def plotPatchBorders3(patches,
                      altPosMap,
                      aziPosMap,
                      plotAxis = None,
                      plotSize = None, # size of plotting area
                      borderWidth = 2,
                      zoom = 1,
                      centerPatchKey = 'patch01', # center at the largest patch by default
                      markerSize = 2, # size of center dot
                      closingIteration = None, # open iteration for patch borders
                      arrowLength = 10 # length of arrow of gradiant
                      ):
    '''
    plot patch border centered and rotated by a certain patch defined by 'centerPatch'

    also plot vetors of altitude gradiant and azimuth gradiant
    '''


    #generating plot axis
    if plotAxis == None:
        f = plt.figure()
        plotAxis = f.add_subplot(111)

    #calculat rotation angle and center
    try:
        centerPatchObj = patches[centerPatchKey]
    except KeyError:
        area=[]
        for key, value in patches.items():
            area.append([key, value.getArea()])

        area = sorted(area, key=lambda a: a[1], reverse=True)
        centerPatchKey = area[0][0]
        centerPatchObj = patches[centerPatchKey]

    altGradMap = np.gradient(altPosMap)
    aziGradMap = np.gradient(aziPosMap)

    # altGradMapX = np.sum(altGradMap[0] * centerPatchObj.array)
    # altGradMapY = np.sum(altGradMap[1] * centerPatchObj.array)

    aziGradMapX = np.sum(aziGradMap[0] * centerPatchObj.array)
    aziGradMapY = np.sum(aziGradMap[1] * centerPatchObj.array)

    rotationAngle = -(np.arctan2(-aziGradMapX,aziGradMapY)%(2*np.pi))*180/np.pi
    # rotationAngle = 0
    # print (np.arctan2(-altGradMapX,altGradMapY)%(2*np.pi))*180/np.pi

    zoomedCenterPatch = Patch(ni.zoom(centerPatchObj.array, zoom, order = 0),centerPatchObj.sign)
    center = zoomedCenterPatch.getCenter()


    # width and height of original plot
    width = zoomedCenterPatch.array.shape[1]
    height = zoomedCenterPatch.array.shape[0]

    #coordinate of four corners
    NW = np.array([0, 0])
    NE = np.array([0, width])
    SW = np.array([height, 0])
    SE = np.array([height, width])

    # calculate maximum distance to four corners
    maxDis = np.int(np.ceil(np.max([ia.distance(center, NW),
                                    ia.distance(center, NE),
                                    ia.distance(center, SW),
                                    ia.distance(center, SE)
                                    ])))

    # calculate expansion distance to all four directions
    expandN = maxDis - center[0]
    expandS = maxDis - (height - center[0])
    expandE = maxDis - center[1]
    expandW = maxDis - (width - center[1])

    for key, currPatch in patches.items():

        zoomedArray = ni.zoom(currPatch.array, zoom, order = 0)

        #expanding border map for each patch
        zoomedArray = np.concatenate((np.zeros((expandN,zoomedArray.shape[1])),zoomedArray),axis = 0)
        zoomedArray = np.concatenate((zoomedArray,np.zeros((expandS,zoomedArray.shape[1]))),axis = 0)
        zoomedArray = np.concatenate((np.zeros((zoomedArray.shape[0],expandE)),zoomedArray),axis = 1)
        zoomedArray = np.concatenate((zoomedArray,np.zeros((zoomedArray.shape[0],expandW))),axis = 1)

        #rotate border map for each patch
        zoomedArray = tsfm.rotate(zoomedArray, rotationAngle)

        #get center
        zoomedCenter = np.round(np.mean(np.argwhere(zoomedArray).astype(np.float32),axis=0)).astype(np.int)


        #binarize current border map
        zoomedArray[zoomedArray<0.9]=np.nan
        zoomedArray[zoomedArray>=0.9]=1

        #ploting current border
        if currPatch.sign == -1:
            pt.plot_mask(zoomedArray, plotAxis=plotAxis, color='#0000ff', borderWidth = borderWidth, closingIteration = closingIteration)
            plotAxis.plot(zoomedCenter[1],zoomedCenter[0], '.b', markersize = markerSize)
        elif currPatch.sign == 1:
            pt.plot_mask(zoomedArray, plotAxis=plotAxis, color='#ff0000', borderWidth = borderWidth, closingIteration = closingIteration)
            plotAxis.plot(zoomedCenter[1],zoomedCenter[0], '.r', markersize = markerSize)

        #get gradiant vectors for current patch
        currAltGradMapX = np.sum(altGradMap[0] * currPatch.array)
        currAltGradMapY = np.sum(altGradMap[1] * currPatch.array)
        currAltAngle = np.arctan2(-currAltGradMapX,currAltGradMapY)%(2*np.pi)+(rotationAngle*np.pi/180)
        currAziGradMapX = np.sum(aziGradMap[0] * currPatch.array)
        currAziGradMapY = np.sum(aziGradMap[1] * currPatch.array)
        currAziAngle = np.arctan2(-currAziGradMapX,currAziGradMapY)%(2*np.pi)+(rotationAngle*np.pi/180)

        # if key == centerPatchKey:
        #     print currAltAngle*180/np.pi
        #     print np.sin(currAltAngle)
        #     print np.cos(currAltAngle)

        #plotting arrow for the current patch
        plotAxis.arrow(zoomedCenter[1],zoomedCenter[0],arrowLength*zoom*np.cos(currAltAngle),-arrowLength*zoom*np.sin(currAltAngle),color='#ff00ff',linewidth=2,width=0.5)
        plotAxis.arrow(zoomedCenter[1],zoomedCenter[0],arrowLength*zoom*np.cos(currAziAngle),-arrowLength*zoom*np.sin(currAziAngle),color='#00ffff',linewidth=2,width=0.5)


    if plotSize:
        plotAxis.set_xlim([maxDis-plotSize*zoom/2, maxDis+plotSize*zoom/2])
        plotAxis.set_ylim([maxDis+plotSize*zoom/2, maxDis-plotSize*zoom/2])
    else:
        plotAxis.set_xlim([0,2*maxDis])
        plotAxis.set_ylim([2*maxDis,0])

    plotAxis.get_xaxis().set_visible(False)
    plotAxis.get_yaxis().set_visible(False)


def plotPairedPatches(patch1,
                      patch2,
                      altMap,
                      aziMap,
                      title,
                      pixelSize = 1,
                      closeIter = None):

    visualSpace1, _, _ = patch1.getVisualSpace(altMap = altMap,
                                               aziMap = aziMap,
                                               pixelSize = pixelSize,
                                               closeIter = closeIter)
    area1 = np.sum(visualSpace1[:]) * (pixelSize ** 2)

    visualSpace2, _, _ = patch2.getVisualSpace(altMap = altMap,
                                               aziMap = aziMap,
                                               pixelSize = pixelSize,
                                               closeIter = closeIter)
    area2 = np.sum(visualSpace2[:]) * (pixelSize ** 2)

    visualSpace1 = np.array(visualSpace1, dtype = np.float32)
    visualSpace2 = np.array(visualSpace2, dtype = np.float32)

    visualSpace1[visualSpace1 == 0] = np.nan
    visualSpace2[visualSpace2 == 0] = np.nan

    f = plt.figure()
    f.suptitle(title)
    f_121 = f.add_subplot(121)
    patchPlot1 = f_121.imshow(patch1.getMask(), interpolation = 'nearest', alpha = 0.5, vmax=2, vmin=1)
    patchPlot2 = f_121.imshow(patch2.getMask()*2, interpolation = 'nearest', alpha = 0.5, vmax=2, vmin=1)
    f_121.set_title('patch1: blue, patch2: red')

    f_122 = f.add_subplot(122)
    areaPlot1 = f_122.imshow(visualSpace1, interpolation = 'nearest', alpha = 0.5, vmax=2, vmin=1)
    areaPlot2 = f_122.imshow(visualSpace2*2, interpolation = 'nearest', alpha = 0.5, vmax=2, vmin=1)
    f_122.set_title('area1: %.1f, area2: %.1f (deg^2)' % (area1, area2))
    f_122.invert_yaxis()

    # ---------------------------------------------------------------------------------------------
    # reorganize visual space axis label
    altRange = np.array([np.amin(altMap), np.amax(altMap)])
    aziRange = np.array([np.amin(aziMap), np.amax(aziMap)])
    xlist = np.arange(aziRange[0],aziRange[1],pixelSize)
    ylist = np.arange(altRange[0],altRange[1],pixelSize)

    xtick = []
    xticklabel = []
    i = 0
    while i < len(xlist):
        if int(np.floor(xlist[i])) % 10 == 0:
            xtick.append(i)
            xticklabel.append(str(int(np.floor(xlist[i]))))
            i = int(i + 9 / pixelSize)
        else:
            i=i+1

    ytick = []
    yticklabel = []
    i = 0
    while i < len(ylist):
        if int(np.floor(ylist[i])) % 10 == 0:
            ytick.append(i)
            yticklabel.append(str(int(np.floor(ylist[i]))))
            i = int(i + 9 / pixelSize)
        else:
            i=i+1

    f_122.set_xticks(xtick)
    f_122.set_xticklabels(xticklabel)
    f_122.set_yticks(ytick)
    f_122.set_yticklabels(yticklabel)

def getPatchDict(patch):
    return {'sparseArray':patch.sparseArray,'sign':patch.sign}


class RetinotopicMappingTrial(object):


    def __init__(self,
                 mouseID, # str, mouseID
                 dateRecorded, # int, date recorded, yearmonthday
                 trialNum, # str, number of the trail on that day
                 mouseType, # str, mouse Genotype
                 visualStimType, # str, stimulation type
                 visualStimBackground, # str, background of visual stimulation
                 imageExposureTime, # float, exposure time of image file
                 altPosMap, # altitude position map
                 aziPosMap, # azimuth position map
                 altPowerMap, # altitude power map
                 aziPowerMap, # azimuth power map
                 vasculatureMap, # vasculature map
                 params ={
                          'phaseMapFilterSigma':1,
                          'signMapFilterSigma':10,
                          'signMapThr':0.3,
                          'eccMapFilterSigma':5.,
                          'splitLocalMinCutStep':10.,
                          'mergeOverlapThr':0.05,
                          'closeIter':3,
                          'openIter':3,
                          'dilationIter':20,
                          'borderWidth':1,
                          'smallPatchThr':200,
                          'visualSpacePixelSize':0.5,
                          'visualSpaceCloseIter':15,
                          'splitOverlapThr':1.1
                          },
                 isAnesthetized = False,
                 ):

        self.mouseID = mouseID
        self.dateRecorded = dateRecorded
        self.trialNum = trialNum
        self.mouseType = mouseType
        self.visualStimType = visualStimType
        self.visualStimBackground = visualStimBackground
        self.imageExposureTime = imageExposureTime
        self.altPosMap = altPosMap
        self.aziPosMap = aziPosMap
        self.altPowerMap = altPowerMap
        self.aziPowerMap = aziPowerMap
        self.vasculatureMap = vasculatureMap
        self.params = params

        self.isAnesthetized = isAnesthetized


    def getName(self):

        trialName = str(self.dateRecorded)+\
                    '_M'+str(self.mouseID)+\
                    '_Trial' + str(self.trialNum)
                    #'_'+self.mouseType.split('-')[0]+';'+self.mouseType.split(';')[-1][0:4]
                    #'_' + str(self.visualStimType)+\
                    #'_' + str(self.visualStimBackground)

        # if self.isAnesthetized:
        #     trialName += '_Anesth'
        # else:
        #     trialName += '_Awake'

        return trialName


    def _getSignMap(self, isReverse = False, isPlot = False, isFixedRange = True):

        altPosMapf  = ni.filters.gaussian_filter(self.altPosMap,
                                                 self.params['phaseMapFilterSigma'])
        aziPosMapf  = ni.filters.gaussian_filter(self.aziPosMap,
                                                 self.params['phaseMapFilterSigma'])

        if self.altPowerMap is not None:
            altPowerMapf  = ni.filters.gaussian_filter(self.altPowerMap,
                                                       self.params['phaseMapFilterSigma'])
        else: altPowerMapf = None

        if self.aziPowerMap is not None:
            aziPowerMapf  = ni.filters.gaussian_filter(self.aziPowerMap,
                                                       self.params['phaseMapFilterSigma'])
        else: aziPowerMapf = None

        signMap = visualSignMap(altPosMapf, aziPosMapf)

        if isReverse: signMap = signMap * -1

        signMapf = ni.filters.gaussian_filter(signMap,
                                              self.params['signMapFilterSigma'])

        if isPlot:
            f1=plt.figure(figsize=(18,9))
            f1_231 = f1.add_subplot(231)
            if isFixedRange: currfig = f1_231.imshow(self.altPosMap, vmin=-40, vmax=60, cmap='hsv', interpolation='nearest')
            else: currfig = f1_231.imshow(self.altPosMap, cmap='hsv', interpolation='nearest')
            f1.colorbar(currfig)
            f1_231.set_axis_off()
            f1_231.set_title('alt position')
            f1_232 = f1.add_subplot(232)
            if isFixedRange: currfig = f1_232.imshow(self.aziPosMap, vmin=-0, vmax=120, cmap='hsv', interpolation='nearest')
            else: currfig = f1_232.imshow(self.aziPosMap, cmap='hsv', interpolation='nearest')
            f1.colorbar(currfig)
            f1_232.set_axis_off()
            f1_232.set_title('azi position')
            f1_233 = f1.add_subplot(233)
            currfig = f1_233.imshow(signMap, vmin=-1, vmax=1, cmap='jet', interpolation='nearest')
            f1.colorbar(currfig)
            f1_233.set_axis_off()
            f1_233.set_title('sign map')
            f1_234 = f1.add_subplot(234)
            if isFixedRange: currfig = f1_234.imshow(altPosMapf, vmin=-40, vmax=60, cmap='hsv', interpolation='nearest')
            else: currfig = f1_234.imshow(altPosMapf, cmap='hsv', interpolation='nearest')
            f1.colorbar(currfig)
            f1_234.set_axis_off()
            f1_234.set_title('alt position filtered')
            f1_235 = f1.add_subplot(235)
            if isFixedRange: currfig = f1_235.imshow(aziPosMapf, vmin=0, vmax=120, cmap='hsv', interpolation='nearest')
            else: currfig = f1_235.imshow(aziPosMapf, cmap='hsv', interpolation='nearest')
            f1.colorbar(currfig)
            plt.axis('off')
            f1_235.set_title('azi position filtered')
            f1_236 = f1.add_subplot(236)
            currfig = f1_236.imshow(signMapf, vmin=-1, vmax=1, cmap='jet', interpolation='nearest')
            f1.colorbar(currfig)
            plt.axis('off')
            f1_236.set_title('sign map filtered')

            f2=plt.figure(figsize=(12,4))
            f2_121 = f2.add_subplot(121)
            if altPowerMapf is not None:
                currfig = f2_121.imshow(ia.array_nor(self.altPowerMap), cmap ='hot', vmin = 0, vmax=1, interpolation='nearest')
                f2.colorbar(currfig)
                f2_121.set_title('alt power map')
                f2_121.set_axis_off()
            f2_122 = f2.add_subplot(122)
            if aziPowerMapf is not None:
                currfig = f2_122.imshow(ia.array_nor(self.aziPowerMap), cmap ='hot', vmin = 0, vmax=1, interpolation='nearest')
                f2.colorbar(currfig)
                f2_122.set_title('azi power map')
                f2_122.set_axis_off()


        self.altPosMapf = altPosMapf
        self.aziPosMapf = aziPosMapf
        self.altPowerMapf = altPowerMapf
        self.aziPowerMapf = aziPowerMapf
        self.signMap = signMap
        self.signMapf = signMapf

        return altPosMapf, aziPosMapf, altPowerMapf, aziPowerMapf, signMap, signMapf


    def _getRawPatchMap(self, isPlot = False):

        if not hasattr(self, 'signMapf'):
            _ = self._getSignMap()

        signMapf = self.signMapf
        signMapThr = self.params['signMapThr']
        openIter = self.params['openIter']
        closeIter = self.params['closeIter']

        #thresholding filtered signmap
        patchmap = np.zeros(signMapf.shape)
        patchmap[signMapf >= signMapThr] = 1
        patchmap[signMapf <= -1 * signMapThr] = 1
        patchmap[(signMapf < signMapThr) & (signMapf > -1 * signMapThr)] = 0
        patchmap = ni.binary_opening(np.abs(patchmap), iterations = openIter).astype(np.int)
        patches, patchNum = ni.label(patchmap)


        #closing each patch, then put them together
        patchmap2=np.zeros(patchmap.shape).astype(np.int)
        for i in range(patchNum):
            currPatch = np.zeros(patches.shape).astype(np.int)
            currPatch[patches==i+1]=1
            currPatch = ni.binary_closing(currPatch, iterations = closeIter).astype(np.int)
            patchmap2 = patchmap2 + currPatch

        if isPlot:
            plt.figure()
            plt.imshow(patchmap, vmin=0, vmax=1, cmap='jet', interpolation='nearest')
            plt.colorbar()
            plt.title('raw patchmap')
            plt.gca().set_axis_off()

        self.rawPatchMap = patchmap2

        return patchmap2


    def _getRawPatches(self, isPlot = False):

        if not hasattr(self, 'rawPatchMap'):
            _ = self._getRawPatchMap()

        signMapf = self.signMapf
        rawPatchMap = self.rawPatchMap
        dilationIter = self.params['dilationIter']
        borderWidth = self.params['borderWidth']
        smallPatchThr = self.params['smallPatchThr']
        vasculatureMap = self.vasculatureMap

        patchMapDilated = dilationPatches2(rawPatchMap, dilationIter=dilationIter, borderWidth=borderWidth)

        #generate raw patch dictionary
        rawPatches = labelPatches(patchMapDilated, signMapf, connectivity = 4)

        rawPatches2 = dict(rawPatches)
        #remove small patches
        for key, value in rawPatches2.items():
            if (value.getArea() < smallPatchThr):
                rawPatches.pop(key)

        #remove isolated Patches
        rawPatches2 = dict(rawPatches)
        for key in rawPatches2.keys():
            isTouching = 0
            for key2 in rawPatches2.keys():
                if key != key2:
                    if rawPatches2[key].isTouching(rawPatches2[key2], borderWidth*2):
                        isTouching = 1
                        break

            if isTouching == 0:
                rawPatches.pop(key)

        rawPatches = sortPatches(rawPatches)

        if isPlot:
            try:
                zoom = vasculatureMap.shape[0] / rawPatches['patch01'].array.shape[0]
            except:
                zoom = 1
            f = plt.figure()
            f_axis = f.add_subplot(111)
            try:
                f_axis.imshow(vasculatureMap, cmap = 'gray', interpolation = 'nearest')
            except:
                pass
            _ = plotPatches(rawPatches, plotaxis = f_axis, zoom = zoom)
            f_axis.set_title('raw patches')
            plt.gca().set_axis_off()
            del _

        self.rawPatches = rawPatches

        return rawPatches


    def _getDeterminantMap(self, isPlot = False):

        if not hasattr(self, 'altPosMapf') or not hasattr(self, 'aziPosMapf'):
            _ = self._getSignMap()

        altPosMapf = self.altPosMapf
        aziPosMapf = self.aziPosMapf

        gradAltMap = np.gradient(altPosMapf)
        gradAziMap = np.gradient(aziPosMapf)

        detMap = np.array([[gradAltMap[0], gradAltMap[1]],
                           [gradAziMap[0], gradAziMap[1]]])

        detMap = detMap.transpose(2,3,0,1)
        detMap = np.abs(np.linalg.det(detMap))


        if isPlot:
            plt.figure()
            plt.imshow(detMap, vmin = 0, vmax = 1,cmap='hsv', interpolation='nearest')
            plt.colorbar()
            plt.title('determinant map')
            plt.gca().set_axis_off()

        self.determinantMap = detMap

        return detMap


    def _getEccentricityMap(self, isPlot = False):

        if not hasattr(self, 'rawPatches'):
            _ = self._getRawPatches()

        altPosMapf = self.altPosMapf
        aziPosMapf = self.aziPosMapf
        eccMapFilterSigma = self.params['eccMapFilterSigma']
        patches = self.rawPatches

        eccMap = np.zeros(altPosMapf.shape)
        eccMapf = np.zeros(altPosMapf.shape)
        eccMap[:] = np.nan
        eccMapf[:] = np.nan

        for key, value in patches.items():

            patchAltC, patchAziC = value.getPixelVisualCenter(altPosMapf,aziPosMapf)
            patchEccMap = eccentricityMap(altPosMapf, aziPosMapf, patchAltC, patchAziC)
            patchEccMapf  = ni.filters.uniform_filter(patchEccMap, eccMapFilterSigma)

            eccMap[value.array == 1] = patchEccMap[value.array == 1]
            eccMapf[value.array == 1] = patchEccMapf[value.array == 1]

        if isPlot:
            plt.figure()
            plt.imshow(eccMapf, interpolation='nearest')
            plt.colorbar()
            plt.title('filtered eccentricity map')
            plt.gca().set_axis_off()

        self.eccentricityMap = eccMap
        self.eccentricityMapf = eccMapf

        return eccMap, eccMapf


    def _splitPatches(self,isPlot = False):

        if not hasattr(self, 'eccentricityMapf'):
            _ = self._getEccentricityMap()

        if not hasattr(self, 'determinantMap'):
            _ = self._getDeterminantMap()

        altPosMapf = self.altPosMapf
        aziPosMapf = self.aziPosMapf
#        eccMap = self.eccentricityMapf
        eccMapf = self.eccentricityMapf
        patches = dict(self.rawPatches)
        detMap = self.determinantMap

        visualSpacePixelSize = self.params['visualSpacePixelSize']
        visualSpaceCloseIter = self.params['visualSpaceCloseIter']
        splitLocalMinCutStep = self.params['splitLocalMinCutStep']
        splitOverlapThr = self.params['splitOverlapThr']
        borderWidth = self.params['borderWidth']

        overlapPatches = []
        newPatchesDict = {}

        for key, value in patches.items():
            visualSpace, _, _ = value.getVisualSpace(altPosMapf,
                                                     aziPosMapf,
                                                     pixelSize = visualSpacePixelSize,
                                                     closeIter = visualSpaceCloseIter)
            AU = np.sum(visualSpace[:]) * (visualSpacePixelSize ** 2)

            AS = value.getSigmaArea(detMap)
            print(key, 'AU='+str(AU), ' AS='+str(AS), ' ratio='+str(AS/AU))

            if AS/AU >= splitOverlapThr:

                patchEccMapf = eccMapf * value.getMask()
                patchEccMapf[value.array == 0] = np.nan

                minMarker = localMin(patchEccMapf, splitLocalMinCutStep)
                NumOfMin = np.amax(minMarker)

                if NumOfMin == 1:
                    print('Only one local minumum was found!!!')
                elif NumOfMin == 0:
                    print('No local minumum was found!!!')
                else:
                    print(str(NumOfMin) + ' local minuma were found!!!')

                    overlapPatches.append(key)

                    newPatches = value.split2(patchEccMapf,
                                              patchName = key,
                                              cutStep = splitLocalMinCutStep,
                                              borderWidth = borderWidth,
                                              isplot = False)

                    #plotting splitted patches
                    if len(newPatches) > 1:

                        f=plt.figure()
                        f121 = f.add_subplot(121)
                        f121.set_title(key)
                        f122 = f.add_subplot(122)
                        f122.set_title('visual space')
                        currPatchValue = 0
                        for key2, value2 in newPatches.items():
                            currPatchValue += 1
                            currArray = np.array(value2.array, dtype=np.float32)
                            currArray[currArray==0]=np.nan
                            currArray[currArray==1]=currPatchValue
                            f121.imshow(currArray,interpolation='nearest', vmin=0, vmax=len(list(newPatches.keys())))
                            f121.set_axis_off()
                            currVisualSpace, _, _ = value2.getVisualSpace(altPosMapf,
                                                                          aziPosMapf,
                                                                          pixelSize = visualSpacePixelSize,
                                                                          closeIter = visualSpaceCloseIter)
                            currVisualSpace=currVisualSpace.astype(np.float32)
                            currVisualSpace[currVisualSpace==0]=np.nan
                            currVisualSpace[currVisualSpace==1]=currPatchValue
                            f122.imshow(currVisualSpace,interpolation='nearest',alpha=0.5, vmin=0, vmax=len(list(newPatches.keys())))

                        xlabel = np.arange(-20,120,visualSpacePixelSize)
                        ylabel = np.arange(60,-40,-visualSpacePixelSize)

                        indSpace = int(10./visualSpacePixelSize)

                        xtickInd = list(range(0,len(xlabel),indSpace))
                        ytickInd = list(range(0,len(ylabel),indSpace))

                        xtickLabel = [str(int(xlabel[x])) for x in xtickInd]
                        ytickLabel = [str(int(ylabel[y])) for y in ytickInd]

                        f122.xaxis.set_ticks(xtickInd)
                        f122.xaxis.set_ticklabels(xtickLabel)
                        f122.yaxis.set_ticks(ytickInd)
                        f122.yaxis.set_ticklabels(ytickLabel)

                    newPatchesDict.update(newPatches)

        for i in range(len(overlapPatches)):
            patches.pop(overlapPatches[i])

        patches.update(newPatchesDict)

        if isPlot:
            try:
                zoom = self.vasculatureMap.shape[0] / self.altPosMap.shape[0]
            except:
                zoom = 1
            f2=plt.figure()
            f2_111=f2.add_subplot(111)
            try:
                f2_111.imshow(self.vasculatureMap, cmap = 'gray', interpolation = 'nearest')
            except:
                pass
            h = plotPatches(patches, plotaxis = f2_111, zoom = zoom)
            f2_111.set_axis_off()
            f2_111.set_title('patches after split')
            f2_111.set_axis_off()


        self.patchesAfterSplit = patches

        return patches


    def _mergePatches(self, isPlot=False):

        if not hasattr(self, 'patchesAfterSplit'):
            self._splitPatches()

        patches = dict(self.patchesAfterSplit)
        altPosMapf = self.altPosMapf
        aziPosMapf = self.aziPosMapf

        borderWidth = self.params['borderWidth']
        visualSpacePixelSize = self.params['visualSpacePixelSize']
        visualSpaceCloseIter = self.params['visualSpaceCloseIter']
        mergeOverlapThr = self.params['mergeOverlapThr']
        smallPatchThr = self.params['smallPatchThr']

        #merging non-overlaping patches
        mergeIter = 1

        # pairs of patches that meet the criterion of merging
        # have 5 columns:
        # first column: key of first patch of the pair
        # second column: key of second patch of the pair
        # third column: merged patch
        # forth column: sum of overlapping ratio of each patch
        # fifth column: negative of unique visual space area (AU) of the merged patch
        mergePairs = []

        while (mergeIter == 1) or (len(mergePairs) > 0):

            print('merge iteration: ' + str(mergeIter))

            mergePairs = []

            #get adjacent pairs
            adjPairs = adjacentPairs(patches, borderWidth = borderWidth+1)

            for ind, pair in enumerate(adjPairs): #for every adjacent pair
                patch1 = patches[pair[0]]
                patch2 = patches[pair[1]]

                try:
                    #merge these two patches
                    currMergedPatch = Patch(mergePatches(patch1.array,patch2.array,borderWidth=borderWidth),
                                            sign = patch1.sign)

                    #calculate unique area of the merged patch
                    visualSpace, _, _ = currMergedPatch.getVisualSpace(altPosMapf,
                                                                       aziPosMapf,
                                                                       pixelSize = visualSpacePixelSize,
                                                                       closeIter = visualSpaceCloseIter)

                    AU = np.sum(visualSpace[:]) * (visualSpacePixelSize ** 2)

                    #calculate the visual space and unique area of the first patch
                    visualSpace1, _, _ =patch1.getVisualSpace(altPosMapf,
                                                              aziPosMapf,
                                                              pixelSize = visualSpacePixelSize,
                                                              closeIter = visualSpaceCloseIter)
                    AU1 = np.sum(visualSpace1[:]) * (visualSpacePixelSize ** 2)

                    #calculate the visual space and unique area of the second patch
                    visualSpace2, _, _ =patch2.getVisualSpace(altPosMapf,
                                                              aziPosMapf,
                                                              pixelSize = visualSpacePixelSize,
                                                              closeIter = visualSpaceCloseIter)
                    AU2 = np.sum(visualSpace2[:]) * (visualSpacePixelSize ** 2)

                    #calculate the overlapping area of these two patches
                    sumSpace = visualSpace1 + visualSpace2
                    overlapSpace = np.zeros(sumSpace.shape, dtype = np.int)
                    overlapSpace[sumSpace == 2] = 1
                    Aoverlap = np.sum(overlapSpace[:]) * (visualSpacePixelSize ** 2)

                    #calculate the ratio of overlaping area to the unique area of each patch
                    overlapRatio1 = Aoverlap / AU1
                    overlapRatio2 = Aoverlap / AU2

                    #if both ratios are small than merge overlap threshold definded at the beginning of the file
                    if (overlapRatio1 <= mergeOverlapThr) and (overlapRatio2 <= mergeOverlapThr):

                        #put this pair and related information to mergePairs list
                        mergePairs.append([pair[0],
                                           pair[1],
                                           currMergedPatch,
                                           np.max([overlapRatio1,overlapRatio2]),
                                           (-1 * AU)])

                    del visualSpace1, visualSpace2, AU1, AU2, sumSpace, overlapSpace, Aoverlap



                except LookupError:
                    pass

                del patch1, patch2

            if len(mergePairs) > 0:
                #for each identified patch pair to merge sort them with the sum of two
                #overlap ratios, from smallest to biggest and then sort them with the
                #unique area of merged patches from biggest to smallest
                mergePairs.sort(key = itemgetter(3,4))

                for ind, value in enumerate(mergePairs): #for each of these pairs
                    patch1 = value[0]
                    patch2 = value[1]

                    # if both of these two patches are still in the 'patches' dictionary
                    if (patch1 in list(patches.keys())) and (patch2 in list(patches.keys())):

                        #plot these patches and their visual space
                        plotPairedPatches(patches[patch1],
                                          patches[patch2],
                                          altPosMapf,
                                          aziPosMapf,
                                          title = 'merge iteation:'+str(mergeIter)+' patch1:'+patch1+' patch2:'+patch2,
                                          pixelSize = visualSpacePixelSize,
                                          closeIter = visualSpaceCloseIter)

                        #remove these two patches from the 'patches' dictionary
                        patches.pop(patch1)
                        patches.pop(patch2)

                        #add merged patches into the 'patches' dictionare
                        patches.update({patch1+'+'+patch2[5:]:value[2]})

                        print('merging: '+patch1+' & '+patch2 + ', overlap ratio: ' + str(value[3]))

            mergeIter = mergeIter + 1


        #remove small patches
        patches2 = dict(patches)
        for key, value in patches2.items():
            if (value.getArea() < smallPatchThr):
                patches.pop(key)

        del patches2

        self.patchesAfterMerge = patches

        finalPatches = sortPatches(patches)

        if isPlot:
            try:
                zoom = self.vasculatureMap.shape[0] / self.altPosMap.shape[0]
            except:
                zoom = 1
            f=plt.figure()
            f111=f.add_subplot(111)
            try:
                f111.imshow(self.vasculatureMap, cmap = 'gray', interpolation = 'nearest')
            except:
                pass
            h = plotPatches(finalPatches, plotaxis = f111, zoom = zoom)
            f111.set_axis_off()
            f111.set_title('final Patches')

        self.finalPatches = finalPatches

        return patches, finalPatches


    def getTraces(self,
                  moviePath,
                  resampleFrequency = 10, # at which frequency the traces are resampled
                  centerPatch = 'patch01', # the patch to get ROI
                  ROIcenters = ([30.,0.],[60.,0.],[90.,0.]), # visual space centers of ROIs
                  ROIsearchRange = 0.5, #range to search pixels in ROI
                  ROIsize = 10, # ROI size (pixel)
                  ROIcolor = ('#ff0000','#00ff00','#0000ff'),#color for each ROI
                  isPlot = False,
                  ):


        if not hasattr(self, 'finalPatches'):
            self.processTrial()

        if type(ROIsearchRange) == int or type(ROIsearchRange) == float:
            ROIsearchRanges=[]
            for i in range(len(ROIcenters)):
                ROIsearchRanges.append(ROIsearchRange)
        elif type(ROIsearchRange) == list:
             ROIsearchRanges = ROIsearchRange

        if not hasattr(self, 'altPosMapf'):
            self._getSignMap()

        altPosMapf = self.altPosMapf
        aziPosMapf = self.aziPosMapf
        patches = self.finalPatches
        mov = tf.imread(moviePath)

        traces = []

        centerPatchArray = patches[centerPatch].array.astype(np.bool)

        for i, ROIcenter in enumerate(ROIcenters):
            try:
                altIndex = np.logical_and(altPosMapf > ROIcenter[1]-ROIsearchRanges[i], altPosMapf < ROIcenter[1]+ROIsearchRanges[i])
                aziIndex = np.logical_and(aziPosMapf > ROIcenter[0]-ROIsearchRanges[i], aziPosMapf < ROIcenter[0]+ROIsearchRanges[i])
                index = np.logical_and(altIndex, aziIndex)
                index = np.where(np.logical_and(index, centerPatchArray) == True)
                pos = int(len(index[0])/2)
                print(ROIcenter, index)
                print('ROI'+str(i)+' center: ['+str(index[0][pos])+','+str(index[1][pos])+']; visual space: ['+str(altPosMapf[index[0][pos],index[1][pos]])+','+str(aziPosMapf[index[0][pos],index[1][pos]])+']')
                mask = ia.generate_rectangle_mask(mov, (index[0][pos], index[1][pos]), ROIsize, ROIsize)
                trace = ia.get_trace(mov, mask)
                t = np.arange(len(trace)) * self.imageExposureTime
                traceP = findPhaseIndex(trace) * self.imageExposureTime

                t2, y2 = ia.resample(t, trace, 1./resampleFrequency)

                traces.append({'position': ROIcenter,
                               'mask': mask,
                               'trace': [t2,y2],
                               'tracePhaseTime': traceP,
                               'ROIcolor':ROIcolor[i]})
            except:
                pass

        if isPlot:
            vasculatureMap = self.vasculatureMap

            try:
                zoom = vasculatureMap.shape[0] / self.altPosMapf.shape[0]
            except:
                zoom = 1

            f = plt.figure(figsize=(15,5))
            ax1 = f.add_axes([0.1,0.1,0.2,0.8])
            ax2 = f.add_axes([0.4,0.1,0.5,0.8])
            try:
                ax1.imshow(vasculatureMap, cmap='gray', interpolation='nearest',aspect='equal')
            except:
                pass
            ROIlegend = []
            for i in range(len(traces)):
                try:
                    pt.plot_mask(traces[i]['mask'], plotAxis = ax1, borderWidth=5, zoom=zoom, color=traces[i]['ROIcolor'])
                    currT=traces[i]['trace'][0]
                    currTrace=traces[i]['trace'][1]
                    ax2.plot(currT,currTrace-np.mean(currTrace), '-', color=traces[i]['ROIcolor'], lw=2)
                    ROIlegend.append(['center:'+str(traces[i]['position'])+', baseline:' + '{0:.0f}'.format(np.mean(currTrace))])
                except:
                    pass


            ax2.legend(ROIlegend,prop={'size':8})

            ymin = np.floor(ax2.yaxis.get_data_interval()[0]/10)*10
            ymax = np.ceil(ax2.yaxis.get_data_interval()[1]/10)*10

            for i in range(len(traces)):
                try:
                    ax2.plot([traces[i]['tracePhaseTime'],traces[i]['tracePhaseTime']],[ymin,ymax],'--',color=traces[i]['ROIcolor'],lw=2)
                except:
                    pass

            ax1.set_axis_off()
            ax2.set_ylim([ymin, ymax])
            ax2.set_xlabel('time (sec)')
            ax2.set_ylabel('normalized count')

        return traces


    def cleanMaps(self):

        try:del self.altPosMapf
        except AttributeError:pass

        try:del self.aziPosMapf
        except AttributeError:pass

        try:del self.altPowerMapf
        except AttributeError:pass

        try:del self.aziPowerMapf
        except AttributeError:pass

        try:del self.signMap
        except AttributeError:pass

        try:del self.signMapf
        except AttributeError:pass

        try:del self.rawPatchMap
        except AttributeError:pass

        try:del self.rawPatches
        except AttributeError:pass

        try:del self.eccentricityMap
        except AttributeError:pass

        try:del self.eccentricityMapf
        except AttributeError:pass

        try:del self.determinantMap
        except AttributeError:pass

        try:del self.patchesAfterSplit
        except AttributeError:pass

        try:del self.patchesAfterMerge
        except AttributeError:pass

        try:del self.finalPatches
        except AttributeError:pass

        try:del self.finalPatchesMarked
        except AttributeError:pass


    def cleanTraces(self):

        try:del self.traces
        except AttributeError:pass


    def processTrial(self, isPlot = False):
        self.cleanMaps()
        _ = self._getSignMap(isPlot=isPlot)
        if isPlot: plt.show()
        _ = self._getRawPatchMap(isPlot=isPlot)
        if isPlot: plt.show()
        _ = self._getRawPatches(isPlot=isPlot)
        if isPlot: plt.show()
        _ = self._getDeterminantMap(isPlot=isPlot)
        if isPlot: plt.show()
        _ = self._getEccentricityMap(isPlot=isPlot)
        if isPlot: plt.show()
        _ = self._splitPatches(isPlot=isPlot)
        if isPlot: plt.show()
        _ = self._mergePatches(isPlot=isPlot)
        if isPlot: plt.show()


    def refresh(self,
                moviePath = None,
                imageExposureTime = None,
                centerPatch = 'patch01',
                ROIcenters = [[30,0],[60,0],[90,0]], # visual space centers of ROIs
                ROIsearchRange = 0.5,
                ROIsize = 10, # ROI size (pixel)
                ROIcolor = ['#ff0000','#00ff00','#0000ff'], #color for each ROI
                isPlot = False):

        self.cleanSelf()

        self._getSignMap(isPlot=isPlot)
        self._getRawPatchMap(isPlot=isPlot)
        self._getRawPatches(isPlot=isPlot)
        self._getDeterminantMap(isPlot=isPlot)
        self._getEccentricityMap(isPlot=isPlot)
        self._splitPatches(isPlot=isPlot)
        self._mergePatches(isPlot=isPlot)

        try:
            _ = self.getTraces(moviePath = moviePath,
                               imageExposureTime = imageExposureTime,
                               centerPatch = centerPatch,
                               ROIcenters = ROIcenters,
                               ROIsearchRange = ROIsearchRange,
                               ROIsize = ROIsize,
                               ROIcolor = ROIcolor,
                               isPlot = isPlot)

        except:
            pass


    def generateTrialDict(self,
                          keysToRetain = ('altPosMap',
                                          'aziPosMap',
                                          'signMap',
                                          'altPosMapf',
                                          'aziPosMapf',
                                          'signMapf',
                                          'rawPatchMap',
                                          'eccentricityMapf',
                                          'finalPatches',
                                          'finalPatchesMarked',
                                          'mouseID',
                                          'dateRecorded',
                                          'trialNum',
                                          'mouseType',
                                          'visualStimType',
                                          'visualStimBackground',
                                          'imageExposureTime',
                                          'altPowerMap',
                                          'altPowerMapf',
                                          'aziPowerMap',
                                          'aziPowerMapf',
                                          'vasculatureMap',
                                          'params',
                                          'isAnesthetized'
                                          )
                          ):


        trialDict = {}
        keysLeft = list(keysToRetain)

        for key in self.__dict__.keys():

            if key in keysToRetain:
                if key=='finalPatches':
                    finalPatches = {}
                    for area,patch in self.finalPatches.items():
                        finalPatches.update({area:getPatchDict(patch)})
                    trialDict.update({'finalPatches':finalPatches})
                    keysLeft.remove('finalPatches')

                elif key == 'finalPathcesMarked':
                    finalPatchesMarked = {}
                    for area,patch in self.finalPathcesMarked.items():
                        finalPatchesMarked.update({area:getPatchDict(patch)})
                    trialDict.update({'finalPatchesMarked':finalPatchesMarked})
                    keysLeft.remove('finalPatchesMarked')

                else:
                    try:
                        trialDict.update({key:self.__dict__[key]})
                        keysLeft.remove(key)
                    except AttributeError:
                        pass

        if keysLeft:
            print('Can not find wanted key(s): ' + str(keysLeft))

        return trialDict


    def generatePosOverlay(self):

        if (not hasattr(self, 'altPosMapf')) or (not hasattr(self, 'aziPosMapf')):
            self._getSignMap()

        vasMap=self.vasculatureMap
        altPosMap = self.altPosMapf
        aziPosMap = self.aziPosMapf

        zoom = vasMap.shape[0]/altPosMap.shape[0]

        altPosMap = ni.zoom(altPosMap,zoom)
        aziPosMap = ni.zoom(aziPosMap,zoom)

        f=plt.figure(figsize=(20,5))
        ax1=f.add_subplot(121)
        ax1.imshow(vasMap,cmap='gray',interpolation='nearest')
        currfig=ax1.imshow(altPosMap,cmap='jet',interpolation='nearest',vmin=-30,vmax=50,alpha=0.5)
        f.colorbar(currfig)
        ax1.axis('off')
        ax1.set_title('altitude position')

        ax2=f.add_subplot(122)
        ax2.imshow(vasMap,cmap='gray',interpolation='nearest')
        currfig=ax2.imshow(aziPosMap,cmap='jet',interpolation='nearest',vmin=0,vmax=100,alpha=0.5)
        f.colorbar(currfig)
        ax2.axis('off')
        ax2.set_title('azimuth position')


    def generateStandardOutput(self,traces = None,isSave = False,saveFolder = None):

        if not hasattr(self, 'finalPatches'):
            self.processTrial()

        if not hasattr(self, 'signMap'):
            self._getSignMap()

        try:
            zoom = self.vasculatureMap.shape[0]/self.altPosMap.shape[0]
        except:
            zoom = 1

        trialName = self.getName()

        f = plt.figure(figsize=(15,10))
        f.suptitle(trialName)

        f_331 = f.add_subplot(331)
        try:
            f_331.imshow(self.vasculatureMap, cmap='gray', interpolation='nearest')
        except:
            pass
        f_331.set_axis_off()

        if traces:
            f_3389 = f.add_axes([0.4,0.1,0.5,0.23])
            ROIlegend=[]

            for i in range(len(traces)):
                pt.plot_mask(traces[i]['mask'], plotAxis = f_331, borderWidth=5, zoom=zoom, color=traces[i]['ROIcolor'])

                currT=traces[i]['trace'][0]
                currTrace=traces[i]['trace'][1]
                f_3389.plot(currT,currTrace-np.mean(currTrace), '-', color=traces[i]['ROIcolor'], lw=2)

                ROIlegend.append(['center:'+str(traces[i]['position'])+', baseline:' + '{0:.0f}'.format(np.mean(currTrace))])

            f_3389.legend(ROIlegend,prop={'size':8})

            ymin = np.floor(f_3389.yaxis.get_data_interval()[0]/10)*10
            ymax = np.ceil(f_3389.yaxis.get_data_interval()[1]/10)*10

            for i in range(len(traces)):
                f_3389.plot([traces[i]['tracePhaseTime'],traces[i]['tracePhaseTime']],[ymin,ymax],'--',color=traces[i]['ROIcolor'], lw=2)

            f_3389.set_ylim([ymin, ymax])
            f_3389.set_xlabel('time (sec)')
            f_3389.set_ylabel('normalized count')


        f_332 = f.add_subplot(332)
        currfig = f_332.imshow(np.mean([self.altPowerMap,self.aziPowerMap],axis=0), cmap='hot', interpolation='nearest')
        f.colorbar(currfig)
        f_332.set_axis_off()

        f_333 = f.add_subplot(333)
        try:
            vm = self.vasculatureMap.astype(np.float)
            vmin = np.amin(vm)
            vmax = np.amax(vm)
            vmin = vmin - 0.5*(vmax-vmin)
            f_333.imshow(self.vasculatureMap, vmin=vmin, vmax=vmax, cmap='gray', interpolation='nearest')
        except:
            pass
        plotPatches(self.finalPatches,plotaxis=f_333,zoom=zoom, markersize=3)
        f_333.set_axis_off()

        f_334 = f.add_subplot(334)
        currfig = f_334.imshow(self.altPosMapf, vmin=-40, vmax=60, cmap='hsv', interpolation='nearest')
        f.colorbar(currfig)
        f_334.set_title('alt position')
        f_334.set_axis_off()

        f_335 = f.add_subplot(335)
        currfig = f_335.imshow(self.aziPosMapf, vmin=0, vmax=120, cmap='hsv', interpolation='nearest')
        f.colorbar(currfig)
        f_335.set_title('azi position')
        f_335.set_axis_off()

        f_336 = f.add_subplot(336)
        currfig = f_336.imshow(self.signMapf, vmin=-1, vmax=1, cmap='jet', interpolation='nearest')
        f.colorbar(currfig)
        f_336.set_axis_off()

        f_337 = f.add_subplot(337)
        plotPatchBorders3(self.finalPatches,
                          self.altPosMapf,
                          self.aziPosMapf,
                          plotAxis = f_337,
                          plotSize = 500,
                          borderWidth = 2,
                          zoom = 1,
                          centerPatchKey = 'patch01',
                          markerSize = 5,
                          closingIteration = 1,
                          arrowLength = 15)
        f_337.get_xaxis().set_visible(False)
        f_337.get_yaxis().set_visible(False)

        if isSave:
            f.savefig(os.path.join(saveFolder,trialName+'.pdf'), format='pdf', dpi = 300, orientation='landscape', papertype='a4')
            f.savefig(os.path.join(saveFolder,trialName+'.png'), format='png', dpi = 300, orientation='landscape', papertype='a4')
            del f


    def generateNormalizedMaps(self,centerPatchKey='patch01',mapSize = 512,isPlot = False,borderValue=0.):

        if not hasattr(self, 'finalPatches'):
            self.processTrial()

        if not hasattr(self, 'signMap') or not hasattr(self, 'altPosMapf') or not hasattr(self, 'aziPosMapf'):
            self._getSignMap()

        centerPixel, rotationAngle = self.getNormalizeTransform(centerPatchKey=centerPatchKey)

        signMap = self.signMap
        signMapf = self.signMapf

        altPosMapC = ia.center_image(self.altPosMap, centerPixel=centerPixel, newSize=mapSize, borderValue=borderValue)
        altPosMapNor = ia.rotate_image(altPosMapC, rotationAngle, borderValue=borderValue)

        aziPosMapC = ia.center_image(self.aziPosMap, centerPixel=centerPixel, newSize=mapSize, borderValue=borderValue)
        aziPosMapNor = ia.rotate_image(aziPosMapC, rotationAngle, borderValue=borderValue)


        if hasattr(self, 'altPowerMap') and self.altPowerMap is not None:
            altPowerMapC = ia.center_image(self.altPowerMap, centerPixel=centerPixel, newSize=mapSize, borderValue=borderValue)
            altPowerMapNor = ia.rotate_image(altPowerMapC, rotationAngle, borderValue=borderValue)
        else:
            altPowerMapNor = None

        if hasattr(self, 'aziPowerMap') and self.aziPowerMap is not None:
            aziPowerMapC = ia.center_image(self.aziPowerMap, centerPixel=centerPixel, newSize=mapSize, borderValue=borderValue)
            aziPowerMapNor = ia.rotate_image(aziPowerMapC, rotationAngle, borderValue=borderValue)
        else:
            aziPowerMapNor = None

        signMapC = ia.center_image(signMap, centerPixel=centerPixel, newSize=mapSize, borderValue=borderValue)
        signMapNor = ia.rotate_image(signMapC, rotationAngle, borderValue=borderValue)

        signMapfC = ia.center_image(signMapf, centerPixel=centerPixel, newSize=mapSize, borderValue=borderValue)
        signMapfNor = ia.rotate_image(signMapfC, rotationAngle, borderValue=borderValue)

        if isPlot:

            trialName = self.getName()

            f = plt.figure(figsize=(15,8))
            f.suptitle('normalized maps for'+trialName)
            f_231 = f.add_subplot(231)
            currfig = f_231.imshow(altPosMapNor,vmin=-30,vmax=50, cmap = 'hsv', interpolation='nearest')
            f.colorbar(currfig)
            f_231.set_axis_off()
            f_231.set_title('normalized altitude position')

            f_232 = f.add_subplot(232)
            currfig = f_232.imshow(aziPosMapNor,vmin=0,vmax=120, cmap = 'hsv', interpolation='nearest')
            f.colorbar(currfig)
            f_232.set_axis_off()
            f_232.set_title('normalized altitude position')

            f_233 = f.add_subplot(233)
            currfig = f_233.imshow(signMapfNor,vmin=-1,vmax=1,cmap = 'jet', interpolation='nearest')
            f.colorbar(currfig)
            f_233.set_axis_off()
            f_233.set_title('normalized sign map')

            f_234 = f.add_subplot(234)
            currfig = f_234.imshow(altPowerMapNor,cmap = 'hot', interpolation='nearest')
            f.colorbar(currfig)
            f_234.set_axis_off()
            f_234.set_title('normalized altitude power')

            f_235 = f.add_subplot(235)
            currfig = f_235.imshow(aziPowerMapNor,cmap = 'hot', interpolation='nearest')
            f.colorbar(currfig)
            f_235.set_axis_off()
            f_235.set_title('normalized azimuth power')

        return altPosMapNor, aziPosMapNor, altPowerMapNor, aziPowerMapNor, signMapNor, signMapfNor


    def generateNormalizedTrial(self,centerPatchKey='patch01', mapSize=512, isPlot=False, borderValue=0.):

        if not hasattr(self, 'finalPatches'):
            self.processTrial()

        centerPixel, rotationAngle = self.getNormalizeTransform(centerPatchKey=centerPatchKey)

        altPosMapC = ia.center_image(self.altPosMap, centerPixel=centerPixel, newSize=mapSize, borderValue=borderValue)
        altPosMapNor = ia.rotate_image(altPosMapC, rotationAngle, borderValue=borderValue)

        aziPosMapC = ia.center_image(self.aziPosMap, centerPixel=centerPixel, newSize=mapSize, borderValue=borderValue)
        aziPosMapNor = ia.rotate_image(aziPosMapC, rotationAngle, borderValue=borderValue)

        # # normalize visual origin
        # altPosOrigin, aziPosOrigin = self.getVisualFieldOrigin()
        # altPosMapNor = altPosMapNor - altPosOrigin
        # aziPosMapNor = aziPosMapNor - aziPosOrigin

        if hasattr(self, 'altPowerMap') and self.altPowerMap is not None:
            altPowerMapC = ia.center_image(self.altPowerMap, centerPixel=centerPixel, newSize=mapSize, borderValue=borderValue)
            altPowerMapNor = ia.rotate_image(altPowerMapC, rotationAngle, borderValue=borderValue)
        else:
            altPowerMapNor = None

        if hasattr(self, 'aziPowerMap') and self.aziPowerMap is not None:
            aziPowerMapC = ia.center_image(self.aziPowerMap, centerPixel=centerPixel, newSize=mapSize, borderValue=borderValue)
            aziPowerMapNor = ia.rotate_image(aziPowerMapC, rotationAngle, borderValue=borderValue)
        else:
            aziPowerMapNor = None

        if hasattr(self, 'vasculatureMap'):
            vasMap = self.vasculatureMap

            if vasMap.shape[0] % self.altPosMap.shape[0] == 0:
                zoom = vasMap.shape[0] / self.altPosMap.shape[0]
            else:
                raise ValueError('Vasculature map size is not whole number of times larger than position maps!')

            vasMapC = ia.center_image(vasMap, centerPixel=[centerPixel[0] * zoom, centerPixel[1] * zoom],
                                      newSize=mapSize * zoom, borderValue=borderValue)
            vasMapNor = ia.rotate_image(vasMapC, rotationAngle, borderValue=borderValue)
        else:
            vasMap = None
            vasMapNor = None

        if isPlot:

            trialName = self.getName()

            f = plt.figure(figsize=(15,8))
            f.suptitle('normalized maps for'+trialName)

            f_231 = f.add_subplot(231)
            currfig = f_231.imshow(self.altPosMap, vmin=-30, vmax=50, cmap='hsv', interpolation='nearest')
            f.colorbar(currfig)
            f_231.set_axis_off()
            f_231.set_title('original altitude map')

            f_232 = f.add_subplot(232)
            currfig = f_232.imshow(self.aziPosMap, vmin=0, vmax=120, cmap='hsv', interpolation='nearest')
            f.colorbar(currfig)
            f_232.set_axis_off()
            f_232.set_title('original altitude map')

            if vasMap is not None:
                f_233 = f.add_subplot(233)
                currfig = f_233.imshow(vasMap, cmap='gray', interpolation='nearest')
                f.colorbar(currfig)
                f_233.set_axis_off()
                f_233.set_title('original vas map')

            f_234 = f.add_subplot(234)
            currfig = f_234.imshow(altPosMapNor, vmin=-30, vmax=50, cmap='hsv', interpolation='nearest')
            f.colorbar(currfig)
            f_234.set_axis_off()
            f_234.set_title('normalized altitude map')

            f_235 = f.add_subplot(235)
            currfig = f_235.imshow(aziPosMapNor, vmin=0, vmax=120, cmap='hsv', interpolation='nearest')
            f.colorbar(currfig)
            f_235.set_axis_off()
            f_235.set_title('normalized azimuth map')

            if vasMapNor is not None:
                f_235 = f.add_subplot(236)
                currfig = f_235.imshow(vasMapNor, cmap='gray', interpolation='nearest')
                f.colorbar(currfig)
                f_235.set_axis_off()
                f_235.set_title('normalized vas map')

        return RetinotopicMappingTrial(self.mouseID, self.dateRecorded, self.trialNum, self.mouseType,
                                       self.visualStimType, self.visualStimBackground, self.imageExposureTime,
                                       altPosMapNor, aziPosMapNor, altPosMapNor, aziPowerMapNor, vasMapNor, params={},
                                       isAnesthetized=self.isAnesthetized)


    def getNormalizeTransform(self,centerPatchKey = 'patch01'):

        try:
            centerPatchObj = self.finalPatchesMarked[centerPatchKey]
        except (AttributeError, KeyError):
            centerPatchObj = self.finalPatches[centerPatchKey]

        centerPixel = centerPatchObj.getCenter()

        if not hasattr(self, 'aziPosMapf'): self._getSignMap()

        aziGradMap = np.gradient(self.aziPosMapf)
        aziGradMapX = np.sum(aziGradMap[0] * centerPatchObj.array)
        aziGradMapY = np.sum(aziGradMap[1] * centerPatchObj.array)
        rotationAngle = -(np.arctan2(-aziGradMapX,aziGradMapY)%(2*np.pi))*180/np.pi

        return centerPixel, rotationAngle


    def normalize(self,centerPatchKey = 'patch01',mapSize = 800,isPlot = False,borderValue=0.):

        '''
        Generate normalized vasculature map and normalized final patches

        return normalized vascualture and patch dictionary, which will be zoomed to match the vascualture map resolution
        '''

        if hasattr(self, 'finalPatchesMarked'):
            patches = self.finalPatchesMarked
        elif hasattr(self, 'finalPatches'):
            patches = self.finalPatches
        else:
            self.processTrial()
            patches = self.finalPatches

        if not hasattr(self, 'signMap'):
            self._getSignMap()

        centerPixel, rotationAngle = self.getNormalizeTransform(centerPatchKey=centerPatchKey)

        try:
            vasMap = self.vasculatureMap.astype(np.float)
            zoom = int(float(vasMap.shape[0])/float(self.aziPosMapf.shape[0]))
        except AttributeError as e:
            print('Can not find vasculature map!!\n\n')
            print(e)
            zoom = 1

        mapSize = mapSize * zoom
        centerPixel = centerPixel * zoom

        try:
            vasMapNor = ia.center_image(vasMap, centerPixel=centerPixel, newSize=mapSize, borderValue=borderValue)
            vasMapNor = ia.rotate_image(vasMapNor, rotationAngle, borderValue=borderValue)
        except NameError:
            pass

        patchesNor = {}
        for key, patch in patches.items():
            patchArray = patch.array.astype(np.float)
            patchArrayNor = ni.zoom(patchArray,zoom=zoom)
            patchArrayNor = ia.center_image(patchArrayNor, centerPixel=centerPixel, newSize=mapSize, borderValue=borderValue)
            patchArrayNor = ia.rotate_image(patchArrayNor, rotationAngle, borderValue=borderValue)
            patchArrayNor = np.round(patchArrayNor).astype(np.int8)
            newPatch = Patch(patchArrayNor,patch.sign)
            patchesNor.update({key:newPatch})

        if isPlot:
            f = plt.figure(figsize=(12,5))
            ax1 = f.add_subplot(121)
            ax1.set_title('original')
            try:
                ax1.imshow(vasMap, cmap = 'gray', interpolation = 'nearest')
            except:
                pass
            h = plotPatches(patches, plotaxis = ax1, zoom = zoom)
            ax1.set_axis_off()

            ax2 = f.add_subplot(122)
            ax2.set_title('normalized')
            try:
                ax2.imshow(vasMapNor, cmap = 'gray', interpolation = 'nearest')
            except:
                pass
            h = plotPatches(patchesNor, plotaxis = ax2, zoom = 1)
            ax2.set_axis_off()


        return vasMapNor, patchesNor


    def plotNormalizedPatchCenter(self,centerPatchKey = 'patch01',mapSize = 512,plotAxis = None,markerSize = 5.,markerEdgeWidth = 2.):

        if not plotAxis:
            f=plt.figure()
            plotAxis = f.add_subplot(111)

        if not hasattr(self, 'finalPatches'):
            self.processTrial()

        if (not hasattr(self, 'signMap')) or (not hasattr(self, 'aziPosMapf')):
            self._getSignMap()

        centerPatchObj = self.finalPatches[centerPatchKey]

        centerPixel = centerPatchObj.getCenter()

        aziPosMapf = self.aziPosMapf

        aziGradMap = np.gradient(aziPosMapf)
        aziGradMapX = np.sum(aziGradMap[0] * centerPatchObj.array)
        aziGradMapY = np.sum(aziGradMap[1] * centerPatchObj.array)
        rotationAngle = -(np.arctan2(-aziGradMapX,aziGradMapY)%(2*np.pi))*180/np.pi

        for key, patch in self.finalPatches.items():
            patchArray = patch.array.astype(np.float32)
            patchSign = patch.sign

            patchArrayC = ia.center_image(patchArray, centerPixel=centerPixel, newSize=mapSize)
            patchArrayN = ia.rotate_image(patchArrayC, rotationAngle)
            patchArrayN[patchArrayN < 0.5] = 0
            patchArrayN[patchArrayN >= 0.5] = 1
            newPatch = Patch(patchArrayN.astype(np.int), patchSign)
            center = newPatch.getCenter()

            if patchSign == 1:
                plotAxis.plot(center[1],
                              mapSize-center[0],
                              'o',
                              mfc='none',
                              mec = 'r',
                              markersize = markerSize,
                              mew = markerEdgeWidth)

            elif patchSign == -1:
                plotAxis.plot(center[1],
                              mapSize-center[0],
                              'o',
                              mfc='none',
                              mec = 'b',
                              markersize = markerSize,
                              mew = markerEdgeWidth)

        plotAxis.set_xlim([0, mapSize])
        plotAxis.set_ylim([0, mapSize])
        plotAxis.set_axis_off()


    def plotTrial(self,isSave = False,saveFolder = None):

        if not hasattr(self, 'finalPatches'):
            self.processTrial()

        if not hasattr(self, 'signMap'):
            self.processTrial()

        try:
            zoom = self.vasculatureMap.shape[0] / self.altPosMap.shape[0]
        except:
            zoom = 1

        trialName = self.getName()

        #plot figure 1
        f1=plt.figure(figsize=(18,9))
        f1.suptitle(trialName)
        f1_231 = f1.add_subplot(231)
        currfig = f1_231.imshow(self.altPosMapf, vmin=-40, vmax=60, cmap='hsv', interpolation='nearest')
        f1.colorbar(currfig)
        f1_231.set_axis_off()
        f1_231.set_title('alt position')
        f1_232 = f1.add_subplot(232)
        currfig = f1_232.imshow(self.aziPosMapf, vmin=0, vmax=120, cmap='hsv', interpolation='nearest')
        f1.colorbar(currfig)
        f1_232.set_axis_off()
        f1_232.set_title('azi position')
        f1_233 = f1.add_subplot(233)
        currfig = f1_233.imshow(self.signMap, vmin=-1, vmax=1, cmap='jet', interpolation='nearest')
        f1.colorbar(currfig)
        f1_233.set_axis_off()
        f1_233.set_title('visual sign map')
        f1_234 = f1.add_subplot(234)
        currfig = f1_234.imshow(self.signMapf, vmin=-1, vmax=1, cmap='jet', interpolation='nearest')
        f1.colorbar(currfig)
        f1_234.set_axis_off()
        f1_234.set_title('filtered visual sign map')
        f1_235 = f1.add_subplot(235)
        try:
            currfig = f1_235.imshow(np.mean([self.altPowerMap,self.aziPowerMap],axis=0), cmap='hot', interpolation='nearest')
            f1.colorbar(currfig)
            plt.axis('off')
            f1_235.set_title('power map')
        except TypeError: pass
        f1_236 = f1.add_subplot(236)
        currfig = f1_236.imshow(self.rawPatchMap, vmin=0, vmax=1, cmap='jet', interpolation='nearest')
        f1.colorbar(currfig)
        plt.axis('off')
        f1_236.set_title('raw patchmap')


        #plot figure 2
        f2 = plt.figure(figsize=(10,8))
        f2.suptitle(trialName)
        f2_221 = f2.add_subplot(221)
        for key, value in self.rawPatches.items():
            currfig = f2_221.imshow(self.altPosMapf * value.getMask(), vmin=-40, vmax=60, interpolation='nearest')
        f2.colorbar(currfig)
        plt.tick_params(
                        axis='both',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom='off',      # ticks along the bottom edge are off
                        top='off',         # ticks along the top edge are off
                        left='off',
                        right='off',
                        labelbottom='off',
                        labelleft='off')
        f2_221.set_title('patches with altitude postion')

        f2_222 = f2.add_subplot(222)
        for key, value in self.rawPatches.items():
            currfig = f2_222.imshow(self.aziPosMapf * value.getMask(), vmin=-10, vmax=120, interpolation='nearest')
        f2.colorbar(currfig)
        plt.tick_params(
                        axis='both',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom='off',      # ticks along the bottom edge are off
                        top='off',         # ticks along the top edge are off
                        left='off',
                        right='off',
                        labelbottom='off',
                        labelleft='off')
        f2_222.set_title('patches with azimuth postion')

        f2_223 = f2.add_subplot(223)
        try:
            f2_223.imshow(self.vasculatureMap, cmap = 'gray', interpolation = 'nearest')
        except:
            pass
        h = plotPatches(self.rawPatches, plotaxis = f2_223, zoom = zoom)
        f2.colorbar(h[list(h.keys())[0]])
        plt.axis('off')
        plt.title('patches with center and sign')


        f2_224 = f2.add_subplot(224)
        currfig = f2_224.imshow(self.eccentricityMapf, interpolation='nearest')
        plt.tick_params(
                        axis='both',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom='off',      # ticks along the bottom edge are off
                        top='off',         # ticks along the top edge are off
                        left='off',
                        right='off',
                        labelbottom='off')
        f2_224.set_title('patches with patch eccentricity')
        f2.colorbar(currfig)

        #plot figure 3
        f3=plt.figure(figsize=(18,7))
        f3.suptitle(trialName)
        f3_131 = f3.add_subplot(131)
        try:
            f3_131.imshow(self.vasculatureMap, cmap = 'gray', interpolation = 'nearest')
        except:
            pass
        h = plotPatches(self.rawPatches, plotaxis = f3_131, zoom = zoom)
        f3_131.set_axis_off()
        f3_131.set_title('original patches')

        f3_132 = f3.add_subplot(132)
        try:
            f3_132.imshow(self.vasculatureMap, cmap = 'gray', interpolation = 'nearest')
        except:
            pass
        h = plotPatches(self.patchesAfterSplit, plotaxis = f3_132, zoom = zoom)
        f3_132.set_axis_off()
        f3_132.set_title('patches after split')

        f3_133 = f3.add_subplot(133)
        try:
            f3_133.imshow(self.vasculatureMap, cmap = 'gray', interpolation = 'nearest')
        except:
            pass
        h = plotPatches(self.patchesAfterMerge, plotaxis = f3_133, zoom = zoom)
        f3_133.set_axis_off()
        f3_133.set_title('patches after merge')

        if isSave:
            f1.savefig(os.path.join(saveFolder,trialName+'_SignMap.pdf'), format='pdf', dpi = 600, orientation='landscape', papertype='a4')
            f2.savefig(os.path.join(saveFolder,trialName+'_RawPatches.pdf'), format='pdf', dpi = 600, orientation='landscape', papertype='a4')
            f3.savefig(os.path.join(saveFolder,trialName+'_SplitMerge.pdf'), format='pdf', dpi = 600, orientation='landscape', papertype='a4')


    def plotFinalPatches(self,plotAxis = None):


        if not hasattr(self, 'finalPatches'):
            self.processTrial()

        try:
            zoom = self.vasculatureMap.shape[0] / self.altPosMap.shape[0]
        except:
            zoom = 1

        name = self.getName()

        if not plotAxis:
            f=plt.figure(figsize=(10,10))
            plotAxis=f.add_subplot(111)

        try:
            plotAxis.imshow(self.vasculatureMap, cmap = 'gray', interpolation = 'nearest')
        except:
            pass
        h = plotPatches(self.finalPatches, plotaxis = plotAxis, zoom = zoom)
        plotAxis.set_axis_off()
        plotAxis.set_title(name)


    def plotFinalPatchBorders(self,plotAxis = None,plotName=True,plotVasMap=True,isTitle=True,isColor=True,
                              borderWidth=2,fontSize=15,interpolation='bilinear'):

        if hasattr(self,'finalPatchesMarked'):finalPatches=self.finalPatchesMarked
        elif hasattr(self, 'finalPatches'):finalPatches=self.finalPatches
        else:self.processTrial();finalPatches=self.finalPatches

        try:zoom = self.vasculatureMap.shape[0] / self.altPosMap.shape[0]
        except AttributeError:zoom = 1

        name = self.getName()

        if not plotAxis:
            f=plt.figure(figsize=(10,10))
            plotAxis=f.add_subplot(111)

        if plotVasMap:
            try:plotAxis.imshow(self.vasculatureMap, cmap = 'gray', interpolation = 'nearest')
            except AttributeError:pass

        for key, patch in finalPatches.items():
            mask = patch.getMask()
            if isColor:
                if patch.sign == 1:plotColor='#ff0000'
                elif patch.sign == -1:plotColor='#0000ff'
                else:plotColor='#000000'
            else:plotColor='#000000'

            im = pt.plot_mask(mask, plotAxis=plotAxis, color=plotColor, zoom=zoom, borderWidth=borderWidth)
            im.set_interpolation(interpolation)
            if plotName:
                center=patch.getCenter()
                plotAxis.text(center[1]*zoom,center[0]*zoom,key,verticalalignment='center', horizontalalignment='center',color=plotColor,fontsize=fontSize)

        plotAxis.set_axis_off()
        if isTitle:plotAxis.set_title(name)

        return plotAxis.get_figure()


    def plotFinalPatchBorders2(self, plotAxis=None, plotName=True, plotVasMap=True, isTitle=True, isColor=True,
                               positiveColor='#ff0000', negativeColor='#0000ff', borderWidth=2, fontSize=15):

        if hasattr(self,'finalPatchesMarked'):finalPatches=self.finalPatchesMarked
        elif hasattr(self, 'finalPatches'):finalPatches=self.finalPatches
        else:self.processTrial();finalPatches=self.finalPatches

        try:zoom = self.vasculatureMap.shape[0] / self.altPosMap.shape[0]
        except AttributeError:zoom = 1

        name = self.getName()

        if not plotAxis:
            f=plt.figure(figsize=(10,10))
            plotAxis=f.add_subplot(111)

        if (plotVasMap) and (self.vasculatureMap is not None):
            try:plotAxis.imshow(self.vasculatureMap, cmap = 'gray', interpolation = 'nearest')
            except AttributeError: plotAxis.invert_yaxis();pass
        else: plotAxis.invert_yaxis()

        for key, patch in finalPatches.items():
            if isColor:
                if patch.sign == 1:plotColor=positiveColor
                elif patch.sign == -1:plotColor=negativeColor
                else:plotColor='#000000'
            else:plotColor='#000000'

            currArray = ni.binary_erosion(patch.array,iterations=1)
            im = pt.plot_mask_borders(currArray, plotAxis=plotAxis, color=plotColor, zoom=zoom, borderWidth=borderWidth)
            if plotName:
                center=patch.getCenter()
                plotAxis.text(center[1]*zoom,center[0]*zoom,key,verticalalignment='center', horizontalalignment='center',color=plotColor,fontsize=fontSize)

        plotAxis.set_axis_off()

        if isTitle:plotAxis.set_title(name)

        return plotAxis.get_figure()


    def getBaselineFluorscence(self):
        '''
        get mean baseline fluorescence of each visual area
        '''

        try:
            finalPatches = self.finalPatchesMarked
        except AttributeError:
            finalPatches = self.finalPatches

        vasMap = ia.array_nor(self.vasculatureMap)

        #get V1 mean fluorscence
        try:
           V1 = finalPatches['V1']
        except KeyError:
           V1 = finalPatches['patch01']

        V1array = V1.array
        zoom = vasMap.shape[-1]/V1array.shape[-1]

        if zoom!=1:
            V1array = ni.zoom(V1array,zoom)
            V1array = ia.binarize(V1array, 0.5)

        V1area = np.sum(V1array).astype(np.float)
        V1totalF = np.sum(V1array*vasMap).astype(np.float)
        V1meanF = V1totalF/V1area

        #get fluorscence for all visual areas normalized by V1
        baselineDict = {}
        for key, patch in finalPatches.items():
            array = patch.array

            if zoom!=1:
                array = ni.zoom(array,zoom)
                array = ia.binarize(array, 0.5)

            area = np.sum(array).astype(np.float)

            totalF = np.sum(array*vasMap).astype(np.float)

            meanFnor = (totalF/area)/V1meanF

            baselineDict.update({key:meanFnor})


        return baselineDict


    def getMeanPowerAmplitude(self):
        '''
        get mean response power amplitude of each visual area
        '''

        try:
            finalPatches = self.finalPatchesMarked
        except AttributeError:
            finalPatches = self.finalPatches

        try:
            powerMap=ia.array_nor(np.mean([self.altPowerMapf, self.aziPowerMapf], axis=0))
        except AttributeError:
            _=self._getSignMap()
            powerMap=ia.array_nor(np.mean([self.altPowerMapf, self.aziPowerMapf], axis=0))

        #get V1 mean fluorscence
        try:
           V1 = finalPatches['V1']
        except KeyError:
           V1 = finalPatches['patch01']

        V1array = V1.array

        V1area = np.sum(V1array).astype(np.float)
        V1totalPower = np.sum(V1array*powerMap).astype(np.float)
        V1meanPower = V1totalPower/V1area

        #get mean power amplitude for all visual areas normalized by V1
        meanPowerDict = {}
        for key, patch in finalPatches.items():
            array = patch.array

            area = np.sum(array).astype(np.float)

            totalPower = np.sum(array*powerMap).astype(np.float)

            meanPowerNor = (totalPower/area)/V1meanPower

            meanPowerDict.update({key:meanPowerNor})

        return meanPowerDict


    def getCorticalArea(self,pixelSize = 0.0129):
        '''
        get area of each visual area (mm^2)
        unit of pixelSize is mm
        '''

        try:
            finalPatches = self.finalPatchesMarked
        except AttributeError:
            finalPatches = self.finalPatches

        #get mean power amplitude for all visual areas normalized by V1
        areaDict = {}
        for key, patch in finalPatches.items():

            area = patch.getArea().astype(np.float)*(pixelSize**2)

            areaDict.update({key:area})

        return areaDict


    def getMagnification(self,pixelSize = 0.0129, isFilter = False,erodeIter = None,):
        '''
        get magnification of each visual area (mm^2/deg^2)
        unit of pixelSize is mm
        '''

        if not hasattr(self,'determinantMap'):
            _ = self._getDeterminantMap()

        if hasattr(self,'finalPathesMarked'):
            finalPatches = self.finalPatchesMarked
        elif hasattr(self, 'finalPatches'):
            finalPatches = self.finalPatches
        else:
            self.processTrial()
            finalPatches = self.finalPatches

        magMap = 1/self.determinantMap

        if isFilter:
            magMap = ni.filters.gaussian_filter(magMap,self.params['signMapFilterSigma'])

        #get mean power amplitude for all visual areas normalized by V1
        magDict = {}
        for key, patch in finalPatches.items():
            array = patch.array.astype(np.float)

            if erodeIter:
                array = ni.binary_erosion(array,iterations=erodeIter)

            area = np.sum(array)

            totalMag = np.sum(array*magMap)

            magDict.update({key:(pixelSize**2)*totalMag/area})

        return magDict


    def getVisualFieldOrigin(self):
        '''
        get the visual field origin as the retinotopic coordinates at the pixels
        where V1, LM and RL meet.

        algorithm dilate V1, LM and RL until the all meet than calculate the
        mean retinotopic locations of all overlap pixels
        '''

        if not hasattr(self,'finalPatchesMarked'):
            raise LookupError('Please mark the final patches first!!')

        if not hasattr(self,'altPosMapf'):
            _=self._getSignMap()

        try:
            V1 = self.finalPatchesMarked['V1'].array.astype(np.float)
            LM = self.finalPatchesMarked['LM'].array.astype(np.float)
            RL = self.finalPatchesMarked['RL'].array.astype(np.float)

            overlap=0 #number of overlaping pixels
            iterNum = 1 #number of iteration
            while overlap<1:
    #            print 'Iteration number for finding overlapping pixel:', iterNum
                V1=ni.morphology.binary_dilation(V1,iterations=1).astype(np.float)
                LM=ni.morphology.binary_dilation(LM,iterations=1).astype(np.float)
                RL=ni.morphology.binary_dilation(RL,iterations=1).astype(np.float)
                totalField = V1+LM+RL
    #            plt.imshow(totalField)
                overlap = len(np.argwhere(totalField==3))
                iterNum += 1
    #            print 'Number of overlapping pixels:', overlap
    #            plt.show()

            altPosOrigin = np.mean(self.altPosMapf[totalField==3],axis=0)
            aziPosOrigin = np.mean(self.aziPosMapf[totalField==3],axis=0)

        except KeyError:
            print('Can not find necessary visual areas (V1, LM, RL) for normalization. \nSetting origins to 0 ...')
            altPosOrigin = 0.
            aziPosOrigin = 0.

        return altPosOrigin, aziPosOrigin


    def plotMagnificationMap(self,pixelSize = 0.0129,plotAxis = None,isFilter = False):
        '''
        param pixelSize:  mm
        '''
        if not hasattr(self,'determinantMap'):
            _ = self._getDeterminantMap()

        if hasattr(self,'finalPathesMarked'):
            finalPatches = self.finalPatchesMarked
        elif hasattr(self, 'finalPatches'):
            finalPatches = self.finalPatches
        else:
            self.processTrial()
            finalPatches = self.finalPatches

        name = self.getName()
        magMap = (pixelSize**2)/self.determinantMap

        if isFilter:
            magMap = ni.filters.gaussian_filter(magMap,self.params['signMapFilterSigma'])

        if not plotAxis:
            f=plt.figure(figsize=(10,10))
            ax = f.add_subplot(111)
        else:
            ax = plotAxis

        for key, patch in finalPatches.items():
            currMagMap = patch.getMask()*magMap
            ax.imshow(currMagMap,cmap='hot_r',vmin = 0, vmax = 0.015,interpolation='nearest')

        ax.set_aspect(1)
        ax.set_title(name)


    def _generateTotalMask(self):
        '''
        generate a single mask (0s and 1s) of the entire visual areas
        '''

        mask = np.zeros(self.altPosMap.shape)

        for patch in self.finalPatches.values():
            mask = mask + patch.array.astype(np.float)

        mask = ni.binary_closing(mask,
                                 structure = np.array([[1.,1.,1.],[1.,1.,1.],[1.,1.,1.]]),
                                 iterations = self.params['borderWidth'])


        return mask.astype(np.int8)


    def plotRetinotopicLocation(self,plotAxis=None,location=(0.,50.),color='#ff0000',searchRange=3.,borderWidth=1,closeIter=3,openIter=3):

        if not plotAxis:
            f = plt.figure()
            plotAxis = f.add_subplot(111)

        altPosMap = self.altPosMapf
        aziPosMap = self.aziPosMapf

        altMin = location[0]-np.abs(searchRange)
        altMax = location[0]+np.abs(searchRange)

        aziMin = location[1]-np.abs(searchRange)
        aziMax = location[1]+np.abs(searchRange)

        altMask = np.logical_and(altPosMap>=altMin,altPosMap<=altMax)
        aziMask = np.logical_and(aziPosMap>=aziMin,aziPosMap<=aziMax)

        mask = np.logical_and(altMask,aziMask).astype(np.float)
        totalMask = self._generateTotalMask()
        mask = (mask*totalMask).astype(np.int)

        mask = ni.binary_closing(mask,iterations=closeIter)

        mask = ni.binary_opening(mask,iterations=openIter)

        mask=mask.astype(np.float)
        mask[mask==0]=np.nan

        pt.plot_mask(mask, plotAxis=plotAxis, color=color, borderWidth=borderWidth)


    def plotPatchesWithName(self,patchDict,plotAxis=None):

        if not hasattr(self,patchDict): raise LookupError('This RetinotopicMappingTrial object does not have "' + patchDict + '" attribute!')
        patchesForPlotting = self.__dict__[patchDict]

        if plotAxis is None: f = plt.figure(); plotAxis = f.add_subplot(111)

        plotAxis.figure.suptitle(self.getName())
        plotPatches(patchesForPlotting,plotaxis=plotAxis,markersize=0)

        for key,patch in patchesForPlotting.items():

            center = patch.getCenter()
            plotAxis.text(center[1],center[0],key,verticalalignment='center', horizontalalignment='center')

        return plotAxis.figure


    def plotVisualCoverage(self, is_normalize=False, altRange=(-40., 60.), aziRange=(-20., 100.)):
        '''
        plot the visual coverage of each visual area in a compact way

        if is_normalize is True, the retinotopy will correct for visual origin
        '''

        if hasattr(self,'finalPatchesMarked'):
            finalPatches = self.finalPatchesMarked
        elif hasattr(self,'finalPatches'):
            finalPatches = self.finalPatches
        else:
            self.processTrial()
            finalPatches = self.finalPatches

        if is_normalize:
            visualFieldOrigin = self.getVisualFieldOrigin()
        else:
            visualFieldOrigin = None

        figList, axList = pt.grid_axis(3, 4, len(list(finalPatches.keys())), figsize=(12, 10))

        i = 0

        pixelSize = self.params['visualSpacePixelSize']
        closeIter = self.params['visualSpaceCloseIter']

        for key, patch in finalPatches.items():
            currAx = axList[i]
            visualSpace, altAxis, aziAxis=patch.getVisualSpace(self.altPosMapf,
                                                               self.aziPosMapf,
                                                               altRange=altRange,
                                                               aziRange=aziRange,
                                                               visualFieldOrigin=visualFieldOrigin,
                                                               pixelSize = pixelSize,
                                                               closeIter = closeIter,
                                                               isplot = False)

            if patch.sign == 1:
                plotColor = '#ff0000'
            elif patch.sign == -1:
                plotColor = '#0000ff'
            else:
                plotColor = '#000000'

            plotVisualSpace(visualSpace,
                            altAxis=altAxis,
                            aziAxis=aziAxis,
                            plotAxis=currAx,
                            lineColor=plotColor,
                            lineWidth=1,
                            tickSpace=10)

            currAx.set_title(key)

            i=i+1

        return figList, axList


    def plotContours(self,
                     isNormalize = True, #is resetting the origin of visual field
                     altLevels = np.arange(-30.,50.,5.),
                     aziLevels = np.arange(0.,120.,5.),
                     isPlottingBorder=True,
                     inline = False,
                     lineWidth = 3,
                     figSize = (12,12),
                     fontSize = 15,
                     altAxis = None,
                     aziAxis = None):
        '''
        plot contours of altitute posititon and azimuth position
        '''

        if not hasattr(self,'altPosMapf'):
            self._getSignMap()

        altPosMap = self.altPosMapf
        aziPosMap = self.aziPosMapf

        if isNormalize:
            altPosOrigin, aziPosOrigin = self.getVisualFieldOrigin()
            altPosMap = altPosMap - altPosOrigin
            aziPosMap = aziPosMap - aziPosOrigin

        if hasattr(self,'vasculatureMap') and type(self.vasculatureMap)!=type(None) and isPlottingBorder:
            zoom = self.vasculatureMap.shape[0]/altPosMap.shape[0]
            altPosMap = ni.zoom(altPosMap,zoom)
            aziPosMap = ni.zoom(aziPosMap,zoom)
            totalMask = ni.zoom(self._generateTotalMask().astype(np.float32),zoom)
            altPosMap[totalMask<0.5]=np.nan
            aziPosMap[totalMask<0.5]=np.nan
        else:
            totalMask = self._generateTotalMask()
            altPosMap[totalMask==0]=np.nan
            aziPosMap[totalMask==0]=np.nan

        X,Y = np.meshgrid(np.arange(altPosMap.shape[1]),
                          np.arange(altPosMap.shape[0]))


        # plotting altitute contours
        if not altAxis:
            altf=plt.figure(figsize=figSize,facecolor='#ffffff')
            altAxis = altf.add_subplot(111)

        altContour = altAxis.contour(X,
                                     Y,
                                     altPosMap,
                                     inline=inline,
                                     levels=altLevels,
                                     linewidths=lineWidth)

        if inline:
            altContour.clabel(inline=inline, fontsize=fontSize,fmt='%1.1f')
        else:
            altAxis.get_figure().colorbar(altContour)

        if isPlottingBorder:
            self.plotFinalPatchBorders(plotAxis=altAxis,
                                       plotName=False,
                                       isTitle=False,
                                       isColor=False,
                                       borderWidth=lineWidth,
                                       interpolation='bilinear')

        altAxis.set_title('Altitute Positions')


        # plotting azimuth contours
        if not aziAxis:
            azif=plt.figure(figsize=figSize,facecolor='#ffffff')
            aziAxis = azif.add_subplot(111)

        aziContour = aziAxis.contour(X,
                                     Y,
                                     aziPosMap,
                                     inline=inline,
                                     levels=aziLevels,
                                     linewidths=lineWidth)
        if inline:
            aziContour.clabel(inline=1, fontsize=fontSize,fmt='%1.1f')
        else:
            aziAxis.get_figure().colorbar(aziContour)

        if isPlottingBorder:
            self.plotFinalPatchBorders(plotAxis=aziAxis,
                                       plotName=False,
                                       isTitle=False,
                                       isColor=False,
                                       borderWidth=lineWidth,
                                       interpolation='bilinear')


        aziAxis.set_title('Azimuth Positions')

        return altAxis, aziAxis


class Patch(object):

    def __init__(self,patchArray,sign):

        if isinstance(patchArray,sparse.coo_matrix):self.sparseArray=patchArray.astype(np.uint8)
        else:
            arr = patchArray.astype(np.int8)
            arr[arr > 0] = 1
            arr[arr == 0] = 0
            self.sparseArray = sparse.coo_matrix(arr)

        if sign==1 or sign==0 or sign==-1: self.sign = int(sign)
        else: raise ValueError('Sign should be -1, 0 or 1!')

    @property
    def array(self):
        return self.sparseArray.toarray()

    def getCenter(self):
        '''
        return the coordinates of the center of a patch
        [rowIndex, columnIndex]
        '''
        pixels = np.argwhere(self.array)
        center = np.mean(pixels.astype(np.float32), axis = 0)
        return np.round(center).astype(np.int)

    def getArea(self):
        '''
        return pixel number in the patch
        '''
        return np.sum(self.array[:])

    def getMask(self):
        '''
        generating ploting mask for the patch
        '''
        mask = np.array(self.array, dtype = np.float32)
        mask[mask == 0] = np.nan
        return mask

    def getSignedMask(self):
        '''
        generating ploting mask with visual sign for the patch
        '''
        signedMask = np.array(self.array * self.sign, dtype = np.float32)
        signedMask[signedMask == 0] = np.nan
        return signedMask

    def getDict(self):
        return {'sparseArray':self.sparseArray,'sign':self.sign}

    def getTrace(self,mov):
        '''
        return trace of this patch in a certain movie
        '''
        return ia.get_trace(mov, self.array)

    def isTouching(self, patch2, distance = 1):
        '''
        decide if this patch is adjacent to another patch within certain distance
        '''

        if distance < 1:
            raise LookupError('distance should be integer no less than 1.')

        bigPatch = ni.binary_dilation(self.array,
                                      iterations = distance).astype(np.int)

        if np.amax(bigPatch + patch2.array) > 1:
            return True
        else:
            return False

    def getVisualSpace(self, altMap, aziMap, altRange=(-40., 60.), aziRange=(-20., 120.), visualFieldOrigin=None,
                       pixelSize=1., closeIter=None, isplot=False):
        '''
        get the visual response space, visual response space center unique area and
        eccentricity map of a cortical patch
        '''

        #        altRange = np.array([np.amin(altMap)-10., np.amax(altMap)+10.])
        #        aziRange = np.array([np.amin(aziMap)-10., np.amax(aziMap)+10.])

        pixelSize = np.float(pixelSize)

        if visualFieldOrigin:
            altMap = altMap - visualFieldOrigin[0]
            aziMap = aziMap - visualFieldOrigin[1]

        # altAxis = np.arange(altRange[0], altRange[1], pixelSize)[::-1]
        # aziAxis = np.arange(aziRange[0], aziRange[1], pixelSize)

        # visualSpace = np.zeros((np.ceil((altRange[1] - altRange[0]) / pixelSize),
        #                         np.ceil((aziRange[1] - aziRange[0]) / pixelSize)))

        altAxis = np.arange(altRange[0], altRange[1], pixelSize)
        aziAxis = np.arange(aziRange[0], aziRange[1], pixelSize)

        visualSpace = np.zeros((len(altAxis), len(aziAxis)), dtype=np.uint8)

        patchArray = self.array
        for i in range(patchArray.shape[0]):
            for j in range(patchArray.shape[1]):
                if patchArray[i, j]:
                    corAlt = altMap[i, j]
                    corAzi = aziMap[i, j]
                    if (corAlt >= altRange[0]) & (corAlt <= altRange[1]) & (corAzi >= aziRange[0]) & (corAzi <= aziRange[1]):
                        indAlt = (corAlt - altRange[0]) // pixelSize
                        indAzi = (corAzi - aziRange[0]) // pixelSize
                        visualSpace[int(indAlt), int(indAzi)] = 1

        if closeIter >= 1:
            visualSpace = ni.binary_closing(visualSpace, iterations=closeIter)

        if isplot:
            f = plt.figure()
            ax = f.add_subplot(111)
            plotVisualSpace(visualSpace,
                            altAxis,
                            aziAxis,
                            plotAxis=ax)

        return visualSpace, altAxis, aziAxis

    def getSigmaArea(self, detMap):
        '''
        calculate sigma area for the patch given altitude and azimuth maps
        '''
        sigmaArea = np.sum((self.array * detMap)[:])
        return sigmaArea

    def getPixelVisualCenter(self, altMap, aziMap):
        '''
        get the center coordinates in visual response space for all pixels in
        this cortical patch
        '''

        altPatch = self.array * altMap
        meanAlt = np.mean(altPatch[altPatch != 0])
        aziPatch = self.array * aziMap
        meanAzi = np.mean(aziPatch[aziPatch != 0])

        return meanAlt, meanAzi

    def eccentricityMap(self, altMap, aziMap, altCenter, aziCenter):
        '''
        calculate eccentricity map of this patch to a certain center in visual
        space

        altMap, aziMap, altCenter, aziCenter: in degree

        eccentricity map is returned in degree
        '''

        altMap2 = altMap * np.pi / 180
        aziMap2 = aziMap * np.pi / 180

        altCenter2 = altCenter * np.pi / 180
        aziCenter2 = aziCenter * np.pi / 180

        eccMap = np.zeros(self.array.shape)
#        eccMap[:] = np.nan
#        for i in xrange(self.array.shape[0]):
#            for j in xrange(self.array.shape[1]):
#                if self.array[i,j]:
#                    alt = altMap2[i,j]
#                    azi = aziMap2[i,j]
#                    eccMap[i,j] = np.arctan(np.sqrt(np.tan(alt-altCenter2)**2 + ((np.tan(azi-aziCenter2)**2)/(np.cos(alt-altCenter2)**2))))
        eccMap = np.arctan(
                           np.sqrt(
                                   np.square(np.tan(altMap2-altCenter2))
                                   +
                                   np.square(np.tan(aziMap2-aziCenter2))/np.square(np.cos(altMap2-altCenter2))
                                   )
                           )
        eccMap = eccMap * 180 / np.pi
        eccMap[self.array==0]=np.nan
        return eccMap

    def split2(self, eccMap, patchName = 'patch00', cutStep = 1, borderWidth = 2, isplot = False):
        '''
        split this patch into two or more patch, according to the eccentricity
        map (in degree). return a dictionary of patches after split

        patchName: str, original patch name
        '''
        minMarker = localMin(eccMap, cutStep)

        connectivity=np.array([[1,1,1],[1,1,1],[1,1,1]])

        newLabel = sm.watershed(eccMap, minMarker, connectivity=connectivity, mask = self.array)

        border = ni.binary_dilation(self.array).astype(np.int8) - self.array

        for i in range(1,np.amax(newLabel)+1):
            currArray = np.zeros(self.array.shape, dtype = np.int8)
            currArray[newLabel == i] = 1
            currBorder = ni.binary_dilation(currArray).astype(np.int8) - currArray
            border = border+currBorder

        border[border > 1] = 1
        border = sm.skeletonize(border)


        if borderWidth > 1:
            border = ni.binary_dilation(border, iterations = borderWidth - 1).astype(np.int8)

        newPatchMap = ni.binary_dilation(self.array).astype(np.int8) * (-1 * (border - 1))


        labeledNewPatchMap, patchNum = ni.label(newPatchMap)


#        if patchNum != np.amax(newLabel):
#            print 'number of patches: ', patchNum, '; number of local minimum:', np.amax(newLabel)
#            raise ValueError, "Number of patches after splitting does not equal to number of local minimum!"

        newPatchDict = {}

        for j in range(1, patchNum + 1):

            currPatchName = patchName + '.' + str(j)
            currArray = np.zeros(self.array.shape, dtype = np.int8)
            currArray[labeledNewPatchMap == j] = 1
            currArray = currArray * self.array

            if np.sum(currArray[:]) > 0:
                newPatchDict.update({currPatchName : Patch(currArray, self.sign)})

        if isplot:
            plt.figure()
            plt.subplot(121)
            plt.imshow(self.array, interpolation = 'nearest')
            plt.title(patchName + ': before split')
            plt.subplot(122)
            plt.imshow(labeledNewPatchMap, interpolation = 'nearest')
            plt.title(patchName + ': after split')


        return newPatchDict

    def split(self, eccMap, patchName = 'patch00', cutStep = 1, borderWidth = 2, isplot = False):
        '''
        split this patch into two or more patch, according to the eccentricity
        map (in degree). return a dictionary of patches after split

        patchName: str, original patch name
        '''
        minMarker = localMin(eccMap, cutStep)

        plt.figure()
        plt.imshow(minMarker, vmin = 0, interpolation='nearest')
        plt.colorbar()
        plt.title('markers 1')
        plt.show()

        minMarker = minMarker.astype(np.int32)
        selfArray = self.array.astype(np.int32)
        minMarker = minMarker + 1
        minMarker[minMarker==1] = 0
        minMarker = minMarker + (-1 * (selfArray-1))
        #minMarker: marker type for opencv watershed,
        #sure background = 1
        #unknow = 0
        #sure forgrand = 2,3,4... etc

        plt.figure()
        plt.imshow(minMarker, vmin = 0, interpolation='nearest')
        plt.colorbar()
        plt.title('markers 2')
        plt.show()

        eccMapNor = (np.round(ia.array_nor(eccMap) * 255)).astype(np.uint8)
        eccMapRGB = cv2.cvtColor(eccMapNor,cv2.COLOR_GRAY2RGB)
        #eccMapRGB: image type for opencv watershed, RGB, [uint8, uint8, uint8]

        newLabel = cv2.watershed(eccMapRGB, minMarker)

        plt.figure()
        plt.imshow(newLabel, vmin = 0, interpolation='nearest')
        plt.colorbar()
        plt.title('markers 3')
        plt.show()

        newBorder = np.zeros(newLabel.shape).astype(np.int)

        newBorder[newLabel==-1]=1



        border = ni.binary_dilation(self.array).astype(np.int) - self.array

        border = newBorder + border

        border[border > 1] = 1

        border = sm.skeletonize(border)


        if borderWidth > 1:
            border = ni.binary_dilation(border, iterations = borderWidth - 1).astype(np.int8)

        newPatchMap = ni.binary_dilation(self.array).astype(np.int8) * (-1 * (border - 1))


        labeledNewPatchMap, patchNum = ni.label(newPatchMap)


#        if patchNum != np.amax(newLabel):
#            print 'number of patches: ', patchNum, '; number of local minimum:', np.amax(newLabel)
#            raise ValueError, "Number of patches after splitting does not equal to number of local minimum!"

        newPatchDict = {}

        for j in range(1, patchNum + 1):

            currPatchName = patchName + '.' + str(j)
            currArray = np.zeros(self.array.shape, dtype = np.int8)
            currArray[labeledNewPatchMap == j] = 1
            currArray = currArray * self.array

            if np.sum(currArray[:]) > 0:
                newPatchDict.update({currPatchName : Patch(currArray, self.sign)})

        if isplot:
            plt.figure()
            plt.subplot(121)
            plt.imshow(self.array, interpolation = 'nearest')
            plt.title(patchName + ': before split')
            plt.subplot(122)
            plt.imshow(labeledNewPatchMap, interpolation = 'nearest')
            plt.title(patchName + ': after split')


        return newPatchDict

    def getBorder(self, borderWidth = 2):
        '''
        return boder of this patch with boder width defined by "borderWidth"
        '''

        patchMap = np.array(self.array, dtype = np.float32)

        smallPatch = ni.binary_erosion(patchMap, iterations = borderWidth).astype(np.float32)

        border = patchMap - smallPatch

        border[border==0] = np.nan

        return border

    def getCorticalPixelForVisualSpaceCenter(self,eccMap):
        '''
        return the coordinates of the pixel representing the center of the
        visual space of the patch
        '''
        eccMap2=np.array(eccMap).astype(np.float)

        eccMap2[self.array==0]=np.nan

        cor = np.array(np.where(eccMap2 == np.nanmin(eccMap2))).transpose()

        return cor


if __name__ == "__main__":

    plt.ioff()

#    testTrial = ft.loadFile(r'\\aibsdata2\nc-ophys\Jun\exampleData\testTrial.pkl')

#    params = {'borderWidth': 1,
#              'closeIter': 3,
#              'dilationIter': 10,
#              'eccMapFilterSigma': 5.0,
#              'mergeOverlapThr': 0.1,
#              'openIter': 3,
#              'phaseMapFilterSigma': 0.8,
#              'signMapFilterSigma': 9.0,
#              'signMapThr': 0.3,
#              'smallPatchThr': 200,
#              'splitLocalMinCutStep': 10.,
#              'splitOverlapThr': 1.1,
#              'visualSpaceCloseIter': 15,
#              'visualSpacePixelSize': 0.5}
#
#    trial = RetinotopicMappingTrial(mouseID = testTrial['mouseID'], # str, mouseID
#                                    dateRecorded = testTrial['dateRecorded'], # int, date recorded, yearmonthday
#                                    trialNum = testTrial['trialNum'], # str, number of the trail on that day
#                                    mouseType = testTrial['mouseType'], # str, mouse Genotype
#                                    visualStimType = testTrial['visualStimType'], # str, stimulation type
#                                    visualStimBackground = testTrial['visualStimBackground'], # str, background of visual stimulation
#                                    imageExposureTime = testTrial['imageExposureTime'], # exposure time of the recorded image
#                                    altPosMap = testTrial['altPosMap'], # altitute position map
#                                    aziPosMap = testTrial['aziPosMap'], # azimuth position map
#                                    altPowerMap = testTrial['altPowerMap'], # altitude power map
#                                    aziPowerMap = testTrial['aziPowerMap'], # azimuth power map
#                                    vasculatureMap = testTrial['vasculatureMap'], # vasculature map
#                                    params = params, # testTrial['params'], # parameters for imaging analysis
#                                    isAnesthetized = testTrial['isAnesthetized'])
#
#    trial.processTrial(isPlot=True)
#    trial._getSignMap()
#
#    traces = trial.getTraces(moviePath = r'\\aibsdata2\nc-ophys\Jun\exampleData\testMov.tif', isPlot = True)
#    trial.generateStandardOutput(traces=traces)
#
#    trialDict = trial.generateTrialDict()
#    print trialDict.keys()
#
#    trial.plotTrial()
#
#    altPosMapfNor, aziPosMapfNor, altPowerMapfNor, aziPowerMapfNor, signMapfNor = trial.generateNormalizedMaps(isPlot = True)
#
#
#    trial.plotNormalizedPatchCenter()
#
#    trial.plotFinalPatches()
#
#
#    vasMapNor, finalPatchesNor = trial.normalize(isPlot = True)

#    plt.show()


#------------------------------------------------------------------------------------------------
#    testTrial, traces = loadTrial(r"E:\data2\2015-02-03-population-maps\20140623_M140174_Trial3.pkl")
#
#    testTrial.plotFinalPatchBorders()
#    plt.show()
#
#    corArea = testTrial.getCorticalArea()
#    print '\nCortical area for each visual area (mm^2):'
#    for key, item in corArea.iteritems():
#        print key, ':', item
#
#    baselineF = testTrial.getBaselineFluorscence()
#    print '\nNormalized baseline fluroscence for each visual area:'
#    for key, item in baselineF.iteritems():
#        print key, ':', item
#
#    power = testTrial.getMeanPowerAmplitude()
#    print '\nNormalized power amplitude for each visual area:'
#    for key, item in power.iteritems():
#        print key, ':', item
#
#    magnification = testTrial.getMagnification()
#    print '\nMagnification for each visual area (mm^2/deg^2):'
#    for key, item in magnification.iteritems():
#        print key, ':', item

#    altPosOrigin, aziPosOrigin = testTrial.getVisualFieldOrigin()
#    print 'Altitude origin:', altPosOrigin
#    print 'Azimuth origin:', aziPosOrigin


#-------------------------------------------------------------------------------------------------

#    testTrial, traces = loadTrial(r"E:\data2\2015-02-03-population-maps\20150130_M160809_Trial1_2_3_4.pkl")
#    visualArea = 'LM'
#    patch = testTrial.finalPatchesMarked[visualArea]
#
#    testTrial._getSignMap()
#
#    f=plt.figure()
#    ax=f.add_subplot(111)
#    ax.imshow(testTrial.altPosMapf,vmin=-30,vmax=50,cmap='hsv',interpolation='nearest')
#    pt.plot_mask(patch.getMask(),plotAxis=ax)
#
#    VSlist = patch.getVisualSpace(testTrial.altPosMapf,testTrial.aziPosMapf)
#
#    VSSpace, VSCoverage, VSAltCenter, VSAziCenter = VSlist
#    print 'Altitute center for area '+visualArea+': '+str(VSAltCenter)
#    print 'Azimuth center for area '+visualArea+': '+str(VSAziCenter)
#    plt.figure()
#    plt.imshow(VSSpace,interpolation='nearest')
#    plt.title(visualArea)
#    plt.show()
#------------------------------------------------------------------------------------------------

#    testTrial, traces = loadTrial(r"E:\data2\2015-02-03-population-maps\20150116_M156569_Trial1_2_3_4.pkl")
#    visualArea = 'V1'
#    patch = testTrial.finalPatchesMarked[visualArea]
#
#    testTrial._getSignMap()
#
#    f=plt.figure()
#    ax=f.add_subplot(111)
#    ax.imshow(testTrial.altPosMapf,vmin=-30,vmax=50,cmap='hsv',interpolation='nearest')
#    pt.plot_mask(patch.getMask(),plotAxis=ax)
#
#    plt.show()

#----------------------------------------------------------------------------------------------

#    testTrial, traces = loadTrial(r"E:\data2\2015-02-03-population-maps\populationTrial_Ai9330min.pkl")
#    testTrial.plotMagnificationMap(isFilter=False)
#    plt.show()


#----------------------------------------------------------------------------------------------
#    testTrial, traces = loadTrial(r"E:\data2\2015-02-03-population-maps\populationTrial_All.pkl")
#    totalMask = testTrial._generateTotalMask()
#    plt.imshow(totalMask)
#    plt.title('total mask of all visual areas')
#    plt.show()


#----------------------------------------------------------------------------------------------
#    testTrial, traces = loadTrial(r"E:\data2\2015-02-03-population-maps\populationTrial_All.pkl")
#
#    locationList = [[0.,40.],
#                    [0.,50.],
#                    [0.,60.]]
#
#    colorList = pt.random_color(len(locationList))
#
#    f=plt.figure(figsize=(12,12))
#    ax=f.add_subplot(111)
#    testTrial.plotFinalPatchBorders(plotAxis=ax,plotName=False,isTitle=True)
#
#    for i, location in enumerate(locationList):
#        testTrial.plotRetinotopicLocation(plotAxis=ax,location=location,color=colorList[i])
#
#    labels = ['alt:'+str(x[0])+'; azi:'+str(x[1]) for x in locationList]
#    [ax.plot(None,None,ls='',c=c,label=l) for c,l in zip(colorList,labels)]
#    leg = ax.legend(labels,frameon=False)
#    for color,text in zip(colorList,leg.get_texts()):
#        text.set_color(color)
#
#    plt.show()


#----------------------------------------------------------------------------------------------


#    testTrial, traces = loadTrial(r"E:\data2\2015-02-03-population-maps\populationTial_Ai93&Ai9630min.pkl")
#    testTrial.plotVisualCoverage()
#    plt.show()
#
#----------------------------------------------------------------------------------------------

    # testTrial, traces = loadTrial(r"E:\data2\2015-02-03-population-maps\populationTial_Ai93&Ai9630min.pkl")
    testTrial, traces = loadTrial(r"E:\data\2015-11-13-150821-M177931-RetinotopicMapping\20150821_MM177931_Trial1_3_4.pkl")
    # testTrial.processTrial(isPlot=True)
    testTrial.plotFinalPatchBorders2(borderWidth=1)
    plt.show()

#----------------------------------------------------------------------------------------------
#    testTrial, traces = loadTrial(r"E:\data2\2015-02-03-population-maps\populationTial_Ai93&Ai9630min.pkl")
#    eccMap = testTrial.eccentricityMapf
#    V1 = testTrial.finalPatchesMarked['V1']
#    V1center = V1.getCorticalPixelForVisualSpaceCenter(eccMap)
#    print V1center

