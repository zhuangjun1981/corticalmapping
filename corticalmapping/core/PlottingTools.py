# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 11:07:20 2014

@author: junz
"""



import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import colorsys
import matplotlib.colors as col
import scipy.ndimage as ni

import tifffile as tf
import ImageAnalysis as ia


def getRGB(colorStr):
    '''
    get R,G,B int value from a hex color string
    '''
    return int(colorStr[1:3],16),int(colorStr[3:5],16),int(colorStr[5:7],16)


def getColorStr(R,G,B):
    '''
    get hex color string from R,G,B value (integer with uint8 format)
    '''
    if not (isinstance(R,(int,long)) and isinstance(G,(int,long)) and isinstance(G,(int,long))):
        raise TypeError, 'Input R, G and B should be integer!'

    if not ((0<=R<=255) and (0<=G<=255) and (0<=B<=255)): raise ValueError, 'Input R, G and B should between 0 and 255!'
    return '#'+''.join(map(chr, (R,G,B))).encode('hex')


def binary2RGBA(img,foregroundColor='#ff0000',backgroundColor='#000000',foregroundAlpha=255,backgroundAlpha=0):
    '''
    generate display image in (RGBA).(np.uint8) format which can be displayed by imshow
    :param img: input image, should be a binary array (np.bool, or np.(u)int
    :param foregroundColor: color for 1 in the array, RGB str, i.e. '#ff0000'
    :param backgroundColor: color for 0 in the array, RGB str, i.e. '#ff00ff'
    :param foregroundAlpha: alpha for 1 in the array, int, 0-255
    :param backgroundAlpha: alpha for 1 in the array, int, 0-255
    :return: displayImg, (RGBA).(np.uint8) format, ready for imshow
    '''

    if img.dtype == np.bool:pass
    elif issubclass(img.dtype.type, np.integer):
        if np.amin(img)<0 or np.amax(img)>1:raise ValueError, 'Values of input image should be either 0 or 1.'
    else: raise TypeError, 'Data type of input image should be either np.bool or integer.'

    if type(foregroundAlpha) is int:
        if foregroundAlpha<0 or foregroundAlpha>255:raise ValueError, 'Value of foreGroundAlpha should be between 0 and 255.'
    else: raise TypeError, 'Data type of foreGroundAlpha should be integer.'

    if type(backgroundAlpha) is int:
        if backgroundAlpha<0 or backgroundAlpha>255:raise ValueError, 'Value of backGroundAlpha should be between 0 and 255.'
    else: raise TypeError, 'Data type of backGroundAlpha should be integer.'

    fR,fG,fB=getRGB(foregroundColor)
    bR,bG,bB=getRGB(backgroundColor)

    displayImg = np.zeros((img.shape[0],img.shape[1],4)).astype(np.uint8)
    displayImg[img==1]=np.array([fR,fG,fB,foregroundAlpha]).astype(np.uint8)
    displayImg[img==0]=np.array([bR,bG,bB,backgroundAlpha]).astype(np.uint8)

    return displayImg


def scalar2RGBA(img,color='#ff0000'):
    '''
    generate display a image in (RGBA).(np.uint8) format which can be displayed by imshow
    alpha is defined by values in the img
    :param img: input image
    :param alphaMatrix: matrix of alpha
    :param foreGroundColor: color for 1 in the array, RGB str, i.e. '#ff0000'
    :return: displayImg, (RGBA).(np.uint8) format, ready for imshow
    '''

    R,G,B=getRGB(color)

    RMatrix = (R * ia.arrayNor(img.astype(np.float32))).astype(np.uint8)
    GMatrix = (G * ia.arrayNor(img.astype(np.float32))).astype(np.uint8)
    BMatrix = (B * ia.arrayNor(img.astype(np.float32))).astype(np.uint8)

    alphaMatrix = (ia.arrayNor(img.astype(np.float32))*255).astype(np.uint8)

    displayImg = np.zeros((img.shape[0],img.shape[1],4)).astype(np.uint8)
    displayImg[:,:,0]=RMatrix; displayImg[:,:,1]=GMatrix; displayImg[:,:,2]=BMatrix; displayImg[:,:,3]=alphaMatrix

    return displayImg


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


def standaloneColorBar(vmin,vmax,cmap,sectionNum=10):
    '''
    plot a stand alone color bar.
    '''
    
    a = np.array([[vmin,vmax]])
    
    plt.figure(figsize=(0.1,9))
    
    img = plt.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.gca().set_visible(False)
    cbar = plt.colorbar()
    cbar.set_ticks(np.linspace(vmin,vmax,num=sectionNum+1))
    

def alphaBlending(image,alphaData,vmin,vmax,cmap = 'Paired',sectionNum = 10,background = -1,interpolation = 'nearest',isSave = False,savePath = None):
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


def plotMask(mask,plotAxis=None,color='#ff0000',zoom=1,borderWidth = None,closingIteration = None):
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


#todo: new plot mask function using contour and return mask border as vector graph


def gridAxis(rowNum,columnNum,totalPlotNum,**kwarg):
    '''
    return figure handles and axis handels for multiple subplots and figures
    '''
    
    figureNum = totalPlotNum//(rowNum*columnNum)+1
    
    figureHandles = []
    
    for i in range(figureNum):
        f=plt.figure(**kwarg)
        figureHandles.append(f)
    
    axisHandles = []    
    for i in range(totalPlotNum):
        currFig = figureHandles[i//(rowNum*columnNum)]
        currIndex = i%(rowNum*columnNum)
        currAxis = currFig.add_subplot(rowNum,columnNum,currIndex+1)
        axisHandles.append(currAxis)
        
    return figureHandles, axisHandles


def tileAxis(f,rowNum,columnNum,topDownMargin=0.05,leftRightMargin=0.05,rowSpacing=0.05,columnSpacing=0.05):

    if 2*topDownMargin+((rowNum-1)*rowSpacing) >= 1: raise ValueError, 'Top down margin or row spacing are too big!'
    if 2*leftRightMargin+((columnNum-1)*columnSpacing) >= 1: raise ValueError, 'Left right margin or column spacing are too big!'

    height = (1-(2*topDownMargin)-(rowNum-1)*rowSpacing)/rowNum
    width = (1-(2*leftRightMargin)-(columnNum-1)*columnSpacing)/columnNum

    xStarts = np.arange(leftRightMargin,1-leftRightMargin,(width+columnSpacing))
    yStarts = np.arange(topDownMargin,1-topDownMargin,(height+rowSpacing))[::-1]

    axisList = [[f.add_axes([xStart,yStart,width,height]) for xStart in xStarts] for yStart in yStarts]

    return axisList


def saveFigureWithoutBorders(f,
                             savePath,
                             removeSuperTitle = True,
                             **kwargs):
    '''
    remove borders of a figure
    '''
    f.gca().get_xaxis().set_visible(False)
    f.gca().get_yaxis().set_visible(False)
    f.gca().set_title('')
    if removeSuperTitle:
        f.suptitle('')
    f.savefig(savePath,pad_inches = 0,bbox_inches='tight',**kwargs)


def mergeNormalizedImages(imgList,isFilter=True,sigma=50,mergeMethod='mean',dtype=np.float32):

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

    if isFilter:
        mergedImgf = ni.filters.gaussian_filter(mergedImg.astype(np.float),sigma=sigma)
        return ia.arrayNor(mergedImg - mergedImgf).astype(dtype)
    else: return ia.arrayNor(mergedImg).astype(dtype)


def hue2RGB(hue):
    '''
    get the RGB value as format as hex string from the decimal ratio of hue (from 0 to 1)
    color model as described in:
    https://en.wikipedia.org/wiki/Hue
    '''
    color = colorsys.hsv_to_rgb(hue,1,1)
    color = [int(x*255) for x in color]
    return getColorStr(*color)







    
if __name__=='__main__':
    
    plt.ioff()
    #----------------------------------------------------
    # ax = barGraph(0.5,1,0.1,label='xx')
    # ax.legend()
    # plt.show()
    #----------------------------------------------------
    
    #----------------------------------------------------
    # figures, axises = gridAxis(2,3,20)
    # for i, ax in enumerate(axises):
    #     ax.imshow(np.random.rand(5,5))
    # plt.show()
    #----------------------------------------------------
    
    
    #----------------------------------------------------
    # mask = np.zeros((100,100))
    # mask[30:50,20:60]=1
    # mask[mask==0]=np.nan
    #
    # plotMask(mask)
    # plt.show()
    #----------------------------------------------------

    #----------------------------------------------------
    # aa=np.random.rand(20,20)
    # mask = np.zeros((20,20),dtype=np.bool)
    # mask[4:7,13:16]=True
    # displayMask = binary2RGBA(mask)
    # plt.figure()
    # plt.imshow(aa)
    # plt.imshow(displayMask,interpolation='nearest')
    # plt.show()
    #----------------------------------------------------

    #----------------------------------------------------
    # b=np.random.rand(5,5)
    # displayImg = scalar2RGBA(b)
    # plt.imshow(displayImg,interpolation='nearest')
    # plt.show()
    #----------------------------------------------------

    #----------------------------------------------------
    # print hue2RGB((2./3.))
    # assert hue2RGB((2./3.)) == '#0000ff'
    #----------------------------------------------------

    #----------------------------------------------------
    # f=plt.figure()
    # f.suptitle('test')
    # ax=f.add_subplot(111)
    # ax.imshow(np.random.rand(20,20))
    # saveFigureWithoutBorders(f,r'C:\JunZhuang\labwork\data\python_temp_folder\test_title.png',removeSuperTitle=False,dpi=300)
    # saveFigureWithoutBorders(f,r'C:\JunZhuang\labwork\data\python_temp_folder\test_notitle.png',removeSuperTitle=True,dpi=300)
    #----------------------------------------------------

    #----------------------------------------------------
    f=plt.figure(figsize=(12,9))
    axisList = tileAxis(f,4,3,0.05,0.05,0.05,0.05)
    print np.array(axisList).shape
    plt.show()
    #----------------------------------------------------

    print 'for debug'