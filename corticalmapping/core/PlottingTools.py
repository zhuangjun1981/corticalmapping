# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 11:07:20 2014

@author: junz
"""



import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as col
import scipy.ndimage as ni

import tifffile as tf
import ImageAnalysis as ia



def barGraph(left,
             height,
             error,
             errorDir = 'both', # 'both', 'positive' or 'negative'
             width = 0.1,            
             plotAxis = None,
             lw = 3,
             faceColor = '#000000',
             edgeColor = 'none',
             capSize = 10,
             label = None
             ):
    '''
    plot a single bar with error bar
    '''
    
    if not plotAxis:
        f = plt.figure()
        plotAxis = f.add_subplot(111)
    
    if errorDir == 'both':
        yerr = error
    elif errorDir == 'positive':
        yerr = [[0],[error]]
    elif errorDir == 'negative':
        yerr = [[error],[0]]
    
    plotAxis.errorbar(left+width/2,
                      height,
                      yerr = yerr,
                      lw=lw,
                      capsize = capSize,
                      capthick = lw,
                      color = edgeColor)
    
    plotAxis.bar(left,
                 height,
                 width=width,
                 color = faceColor,
                 edgecolor=edgeColor,
                 lw = lw,
                 label = label)
    
    
                      
    return plotAxis
    

def randomColor(numOfColor=10):
    '''
    generate as list of random colors
    '''
    numOfColor = int(numOfColor)

    colors = []

    Cmatrix = (np.random.rand(numOfColor,3)*255).astype(np.uint8)

    for i in range(numOfColor):

        r = hex(Cmatrix[i][0]).split('x')[1]
        if len(r) == 1:
            r = '0' + r

        g = hex(Cmatrix[i][1]).split('x')[1]
        if len(g) == 1:
            g = '0' + g

        b = hex(Cmatrix[i][2]).split('x')[1]
        if len(b) == 1:
            b = '0' + b

        colors.append('#' + r + g + b)

    return colors


def showMovie(path, #tif file path or numpy arrary of the movie
              mode = 'raw', # 'raw', 'dF' or 'dFoverF'
              baselinePic = None, # picuture of baseline
              baselineType = 'mean', # way to calculate baseline
              cmap = 'gray'):
    
    '''
    plot tf movie in the way defined by mode
    '''
    
    if isinstance(path, str):
        rawMov = tf.imread(path)
    elif isinstance(path, np.ndarray):
        rawMov = path
        
    if mode == 'raw':
        mov = rawMov
    else:
        _, dFMov, dFoverFMov = ia.normalizeMovie(rawMov,
                                                 baselinePic = baselinePic,
                                                 baselineType = baselineType)
        if mode == 'dF':
            mov = dFMov
        elif mode == 'dFoverF':
            mov = dFoverFMov
        else: 
            raise LookupError, 'The "mode" should be "raw", "dF" or "dFoverF"!'
            
    if isinstance(path, str):
        tf.imshow(mov,
                  cmap=cmap,
                  vmax=np.amax(mov),
                  vmin=np.amin(mov),
                  title = mode + ' movie of ' + path)
    elif isinstance(path, np.ndarray):
        tf.imshow(mov,
                  cmap=cmap,
                  vmax=np.amax(mov),
                  vmin=np.amin(mov),
                  title = mode+' Movie')
            
    return mov



def standaloneColorBar(vmin,
                       vmax,
                       cmap,
                       sectionNum=10):
    '''
    plot a stand alone color bar.
    '''
    
    a = np.array([[vmin,vmax]])
    
    plt.figure(figsize=(0.1,9))
    
    img = plt.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.gca().set_visible(False)
    cbar = plt.colorbar()
    cbar.set_ticks(np.linspace(vmin,vmax,num=sectionNum+1))
    
    

def alphaBlending(image,
                  alphaData,
                  vmin,
                  vmax,
                  cmap = 'Paired',
                  sectionNum = 10,
                  background = -1,
                  interpolation = 'nearest',
                  isSave = False,
                  savePath = None):
    '''
    Generate image with transparency weighted by another matrix.
    
    Plot numpy array 'image' with colormap 'cmap'. And define the tranparency 
    of each pixel by the value in another numpy array alphaData.
    
    All the elements in alphaData should be non-negative.
    '''   
    
    
    if image.shape != alphaData.shape:
        raise LookupError, '"image" and "alphaData" should have same shape!!'
    
    if np.amin(alphaData) < 0:
        raise ValueError, 'All the elements in alphaData should be bigger than zero.'
    
    #normalize image
    image[image > vmax] = vmax
    image[image < vmin] = vmin
    
    image = (image - vmin) / (vmax - vmin)
    
    #get colored image of image
    exec('colorImage = cm.' + cmap + '(image)')

    #normalize alphadata
    alphaDataNor = alphaData / np.amax(alphaData)
    alphaDataNor = np.sqrt(alphaDataNor)
    
    colorImage[:,:,3] = alphaDataNor

    #plt.figure()
    #plot dummy figure for colorbar       
    a = np.array([[vmin,vmax]])
    plt.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0)
    #plt.gca().set_visible(False)
    cbar = plt.colorbar()
    cbar.set_ticks(np.linspace(vmin,vmax,num=sectionNum+1))
    cbar.set_alpha(1)
    cbar.draw_all()
    
    #generate black background
    b=np.array(colorImage)
    b[:] = background
    b[:,:,3] = 1
    plt.imshow(b,cmap = 'gray')
    
    #plot map
    plt.imshow(colorImage, interpolation = interpolation)
    
    return colorImage



def plotMask(mask,
             plotAxis=None,
             color='#ff0000',
             zoom=1,
             borderWidth = None,
             closingIteration = None):
    '''
    plot mask borders in a given color
    '''

    if not plotAxis:
        f = plt.figure()
        plotAxis = f.add_subplot(111)

    cmap1 = col.ListedColormap(color, 'temp')
    cm.register_cmap(cmap=cmap1)

    if zoom != 1:
        mask = ni.interpolation.zoom(mask,zoom,order=0)

    mask2 = mask.astype(np.float32)
    mask2[np.invert(np.isnan(mask2))]= 1.
    mask2[np.isnan(mask2)] = 0.

    struc = ni.generate_binary_structure(2, 2)
    if borderWidth:
        border=mask2 - ni.binary_erosion(mask2,struc,iterations=borderWidth).astype(np.float32)
    else:
        border=mask2 - ni.binary_erosion(mask2,struc).astype(np.float32)

    if closingIteration:
        border = ni.binary_closing(border,iterations=closingIteration).astype(np.float32)

    border[border==0] = np.nan

    currfig = plotAxis.imshow(border, cmap = 'temp', interpolation='nearest')

    return currfig


def gridAxis(rowNum,columnNum,totalPlotNum,figSize=(10,10)):
    '''
    return figure handles and axis handels for multiple subplots and figures
    '''
    
    figureNum = totalPlotNum//(rowNum*columnNum)+1
    
    figureHandles = []
    
    for i in range(figureNum):
        f=plt.figure(figsize=figSize)
        figureHandles.append(f)
    
    axisHandles = []    
    for i in range(totalPlotNum):
        currFig = figureHandles[i//(rowNum*columnNum)]
        currIndex = i%(rowNum*columnNum)
        currAxis = currFig.add_subplot(rowNum,columnNum,currIndex+1)
        axisHandles.append(currAxis)
        
    return figureHandles, axisHandles
    
def saveFigureWithoutBorders(f,
                             savePath,
                             removeAxisTitle = True,
                             removeSuperTitle = True,
                             **kwargs):
    '''
    remove borders of a figure
    '''
    
    f.gca().get_xaxis().set_visible(False)
    f.gca().get_yaxis().set_visible(False)
    f.gca().set_title('')
    f.supertitle = None
    f.savefig(savePath,pad_inches = 0,bbox_inches='tight',**kwargs)

def mergeNormalizedImages(imgList,sigma=50,mergeMethod='mean',dtype=np.float32):

    '''
    merge images in a list in to one, for each image, local intensity variability will be removed by subtraction of
    gaussian filtered image. Then all images will be collapsed by the mergeMethod in to single image
    '''

    imgList2 = []

    for currImg in imgList:
        imgList2.append(ia.arrayNor(currImg.astype(dtype)))

    if mergeMethod == 'mean':
        mergedImg = np.mean(np.array(imgList2),axis=0)
    elif mergeMethod == 'min':
        mergedImg = np.min(np.array(imgList2),axis=0)
    elif mergeMethod == 'max':
        mergedImg = np.max(np.array(imgList2),axis=0)
    elif mergeMethod == 'median':
        mergedImg = np.median(np.array(imgList2),axis=0)

    mergedImgf = ni.filters.gaussian_filter(mergedImg.astype(np.float),sigma=sigma)

    return ia.arrayNor(mergedImg - mergedImgf).astype(dtype)




    
if __name__=='__main__':
    
    plt.ioff()
#----------------------------------------------------    
#    ax = barGraph(0.5,1,0.1,label='xx')
#    ax.legend()
#    plt.show()
#----------------------------------------------------
    
#----------------------------------------------------
#    figures, axises = gridAxis(2,3,20)
#    for i, ax in enumerate(axises):
#        ax.imshow(np.random.rand(5,5))        
#    plt.show()
#----------------------------------------------------
    
    
#----------------------------------------------------
    mask = np.zeros((100,100))
    mask[30:50,20:60]=1
    mask[mask==0]=np.nan
    
    plotMask(mask)
    plt.show()
     
#----------------------------------------------------