# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 13:13:33 2015

@author: chrism
"""
from corticalmapping import RetinotopicMapping
from corticalmapping.core import FileTools
import corticalmapping.core.PlottingTools as pt

try:
    import ipywidgets as widgets
    from IPython.display import display
    has_ipython = True
except ImportError:
    has_ipython = False
#from utils.progress_bar import ProgressBar

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mpl_color

import scipy.ndimage as ni
import numpy as np
import os

class WrappedRetinotopicMapping(RetinotopicMapping.RetinotopicMappingTrial):
    
    def getSignMap(self,phaseMapFilterSigma,signMapFilterSigma,isReverse=False,
                   isPlot=True,isFixedRange=True):
        """
        _getSignMap will calculate altPosMapf,aziPosMapf,signMap,signMapf
        and set them as attributes of RetinotopicMappingTrial
        """
        phaseMapFilterSigma = float(phaseMapFilterSigma)
        signMapFilterSigma = float(signMapFilterSigma)
        self.params["phaseMapFilterSigma"] = phaseMapFilterSigma
        self.params["signMapFilterSigma"] = signMapFilterSigma
        print "phaseMapFilterSigma: {0}".format(phaseMapFilterSigma)
        print "signMapFilterSigma: {0}".format(signMapFilterSigma)
        print "isReverse: {0}".format(isReverse)
        print "isPlot: {0}".format(isPlot)
        print "isFixedRange: {0}".format(isFixedRange)
        self._getSignMap(isReverse=isReverse,isPlot=isPlot,
                         isFixedRange=isFixedRange)
    
    def getRawPatchMap(self,signMapThr,openIter,closeIter,isPlot=True):
        signMapThr = float(signMapThr)
        openIter = int(openIter)
        closeIter = int(closeIter)
        self.params["signMapThr"] = signMapThr
        self.params["openIter"] = openIter
        self.params["closeIter"] = closeIter
        print "signMapThr: {0}".format(signMapThr)
        print "operIter: {0}".format(openIter)
        print "closeIter: {0}".format(closeIter)
        self._getRawPatchMap(isPlot=isPlot)
    
    def getRawPatches(self,dilationIter,borderWidth,smallPatchThr,isPlot=True):
        dilationIter = int(dilationIter)
        borderWidth = int(borderWidth)
        smallPatchThr = int(smallPatchThr)
        self.params['dilationIter'] = dilationIter
        self.params['borderWidth'] = borderWidth
        self.params['smallPatchThr'] = smallPatchThr
        print "dilationIter: {0}".format(dilationIter)
        print "borderWidth: {0}".format(borderWidth)
        print "smallPatchThr: {0}".format(smallPatchThr)
        self._getRawPatches(isPlot=isPlot)
    
    def getDeterminantMap(self,isPlot=True):
        self._getDeterminantMap(isPlot=isPlot)
    
    # def _getDeterminantMap(self, isPlot = False):
    #     if not hasattr(self, 'altPosMapf') or not hasattr(self, 'aziPosMapf'):
    #         _ = self._getSignMap()
    #
    #     altPosMapf = self.altPosMapf
    #     aziPosMapf = self.aziPosMapf
    #
    #     gradAltMap = np.gradient(altPosMapf)
    #     gradAziMap = np.gradient(aziPosMapf)
    #     stack0=np.dstack((gradAltMap[0],gradAziMap[0]))
    #     stack1=np.dstack((gradAltMap[1],gradAziMap[1]))
    #     stacked_stacks = [np.dstack((stack_0,stack_1)) for stack_0,stack_1 in zip(stack0,stack1)] #PLEASE FIND A NEW WAY, THIS ISN"T OPTIMAL,exponentially slower with higher resoltions
    #     detMap=np.abs(np.linalg.det(stacked_stacks))
    #     if isPlot:
    #         plt.figure()
    #         plt.imshow(detMap, vmin = 0, vmax = 1,cmap='hsv', interpolation='nearest')
    #         plt.colorbar()
    #         plt.title('determinant map')
    #         plt.gca().set_axis_off()
    #
    #     self.determinantMap = detMap
    #     return detMap
    
    def getEccentricityMap(self,eccMapFilterSigma,isPlot=True):
        eccMapFilterSigma = float(eccMapFilterSigma)
        self.params['eccMapFilterSigma'] = eccMapFilterSigma
        print "eccMapFilterSigma: {0}".format(eccMapFilterSigma)
        self._getEccentricityMap(isPlot=isPlot)
    
#     def _getEccentricityMap(self,isPlot=False,showProgressBar=False):
#         if not has_ipython:
#             showProgressBar = False #ONLY WORKS IF YOU HAVE IPYTHON
#
#         if not hasattr(self, 'rawPatches'):
#             _ = self._getRawPatches()
#
#         altPosMapf = self.altPosMapf
#         aziPosMapf = self.aziPosMapf
#         eccMapFilterSigma = self.params['eccMapFilterSigma']
#         patches = self.rawPatches
#
#         eccMap = np.zeros(altPosMapf.shape)
#         eccMapf = np.zeros(altPosMapf.shape)
#         eccMap[:] = np.nan
#         eccMapf[:] = np.nan
#         if showProgressBar:
#             progress_bar_txt = "Calculating eccentricity map...{0}%"
#             progress_bar = widgets.FloatProgress(value=0,min=0,max=len(patches.keys()),step=1,
#                                                  description=progress_bar_txt.format(0))
#             display(progress_bar)
#
#         for patch_idx,(key,value) in enumerate(patches.iteritems()):
#
#             patchAltC, patchAziC = value.getPixelVisualCenter(altPosMapf,aziPosMapf)
#             patchEccMap = RetinotopicMapping.eccentricityMap(altPosMapf, aziPosMapf, patchAltC, patchAziC)
#             patchEccMapf  = ni.filters.uniform_filter(patchEccMap, eccMapFilterSigma)
#
# #            for i in xrange(patchEccMap.shape[0]): #faster method is bellow
# #                for j in xrange(patchEccMap.shape[1]):
# #                    if value.array[i,j] == 1:
# #                        eccMap[i,j] = patchEccMap[i,j]
# #                        eccMapf[i,j] = patchEccMapf[i,j]
#             __idxs = value.array == 1
#             eccMap[__idxs] = patchEccMap[__idxs]
#             eccMapf[__idxs] = patchEccMapf[__idxs]
#
#             if showProgressBar:
#                 progress_bar_percent = int((float(patch_idx+1)/(len(patches.keys())))*100)
#                 progress_bar.description = progress_bar_txt.format(progress_bar_percent)
#                 progress_bar.value = patch_idx+1
#
#         if isPlot:
#             plt.figure()
#             plt.imshow(eccMapf, interpolation='nearest')
#             plt.colorbar()
#             plt.title('filtered eccentricity map')
#             plt.gca().set_axis_off()
#
#         self.eccentricityMap = eccMap
#         self.eccentricityMapf = eccMapf
#         return eccMap, eccMapf
    
    def splitPatches(self,visualSpacePixelSize,visualSpaceCloseIter,
                     splitLocalMinCutStep,splitOverlapThr,borderWidth,
                     isPlot=True):
        visualSpacePixelSize = float(visualSpacePixelSize)
        visualSpaceCloseIter = int(visualSpaceCloseIter)
        splitLocalMinCutStep = float(splitLocalMinCutStep)
        splitOverlapThr = float(splitOverlapThr)
        borderWidth = int(borderWidth)
        self.params['visualSpacePixelSize'] = visualSpacePixelSize
        self.params['visualSpaceCloseIter'] = visualSpaceCloseIter
        self.params['splitLocalMinCutStep'] = splitLocalMinCutStep
        self.params['splitOverlapThr'] = splitOverlapThr
        self.params['borderWidth'] = borderWidth
        print "visualSpacePixelSize: {0}".format(visualSpacePixelSize)
        print "visualSpaceCloseIter: {0}".format(visualSpaceCloseIter)
        print "splitLocalMinCutStep: {0}".format(splitLocalMinCutStep)
        print "splitOverlapThr: {0}".format(splitOverlapThr)
        print "borderWidth: {0}".format(borderWidth)
        self._splitPatches(isPlot=isPlot)
        
    def mergePatches(self,visualSpacePixelSize,visualSpaceCloseIter,
                     mergeOverlapThr,borderWidth,smallPatchThr,isPlot=True):
        borderWidth = int(borderWidth)
        visualSpacePixelSize = float(visualSpacePixelSize)
        visualSpaceCloseIter = int(visualSpaceCloseIter)
        mergeOverlapThr = float(mergeOverlapThr)
        smallPatchThr = int(smallPatchThr)
        self.params['borderWidth'] = borderWidth
        self.params['visualSpacePixelSize'] = visualSpacePixelSize
        self.params['visualSpaceCloseIter'] = visualSpaceCloseIter
        self.params['mergeOverlapThr'] = mergeOverlapThr
        self.params['smallPatchThr'] = smallPatchThr
        print "visualSpacePixelSize: {0}".format(visualSpacePixelSize)
        print "visualSpaceCloseIter: {0}".format(visualSpaceCloseIter)
        print "mergeOverlapThr: {0}".format(mergeOverlapThr)
        print "smallPatchThr: {0}".format(smallPatchThr)
        print "borderWidth: {0}".format(borderWidth)
        self._mergePatches(isPlot=isPlot)
    
    def plotPatchesWithName(self,patches_dict,*args,**kwargs):
        if isinstance(patches_dict,dict):
            setattr(self,"__temp_marked_patches_dict",dict(patches_dict))
            plot_patches_dict = "__temp_marked_patches_dict"
        else:
            plot_patches_dict = patches_dict   
        return super(WrappedRetinotopicMapping,self).plotPatchesWithName(plot_patches_dict,*args,**kwargs)
    
    def plotPatchesWithNameAxes(self,patches_dict,*args,**kwargs):
        ret = self.plotPatchesWithName(patches_dict,*args,**kwargs)
        return ret,ret.axes
    
    def plotFinalPatchBorders(self,final_patch_dict,*args,**kwargs):
        final_patches_marked = getattr(self,"finalPatchesMarker",None)
        setattr(self,"finalPatchesMarked",dict(final_patch_dict))
        ret = super(WrappedRetinotopicMapping,self).plotFinalPatchBorders(*args,**kwargs)
        setattr(self,"finalPatchesMarked",final_patches_marked)
        return ret
    
    def plotColoredPatchOnPatchBorders(self,patch,patches_dict,
                                       desired_patch_names,plotAxis=None,
                                       patch_colors={-1:"#0000ff",1:"#ff0000"},
                                       txt_colors={True:"#00cc00",False:"#F09000"},
                                       default_color="#000000"):
        if not plotAxis:
            f,plotAxis = plt.subplots(1,1)
        
        plotAxis.invert_yaxis()
        plotAxis.set_aspect('equal') 
        
        patch_array = ni.binary_erosion(patch.array,iterations=2)
        patch_hex = patch_colors.get(patch.sign,default_color)
        patch_cmap = get_cmap_from_hex(patch_hex)
        patch = RetinotopicMapping.Patch(ni.zoom(patch_array,1,order=0),patch.sign)
        plotAxis.imshow(patch.getSignedMask(),vmax=1,vmin=-1,
                        interpolation='nearest',cmap=patch_cmap,alpha=0.2)
        
        for key,patch in patches_dict.iteritems():
            patch_color = patch_colors.get(patch.sign,default_color)
            currArray = ni.binary_erosion(patch.array,iterations=2)
            pt.plotMaskBorders(currArray,plotAxis=plotAxis,color=patch_color,
                               alpha=0.6)
            text_color = txt_colors.get((key in desired_patch_names),
                                        default_color)
            plotAxis.text(patch.getCenter()[1],patch.getCenter()[0],key,
                          color=text_color,horizontalalignment='center',
                          verticalalignment='center',fontsize=15,zorder=10,fontweight='bold')
            
        return plotAxis

    
    
    def verification_phase(self,final_patch_dict,patch_name_list,
                           error_color="#FFFF66",success_color="#99FFCC",
                           alpha=1,plotaxis=None,zoom=1,markersize=5):
        patch_dict = dict(final_patch_dict)
        patch_name_list = [p_name.lower() for p_name in patch_name_list]
        error_patches = {}
        success_patches = {}
        p_keys = final_patch_dict.keys()
        l_keys = [key.lower() for key in p_keys]
        lower_dup_idxs = [idx for idx,x in enumerate(l_keys) if l_keys.count(x) > 1]
        for l_idx in lower_dup_idxs:
            p_key = p_keys[l_idx]
            error_patches[p_key] = patch_dict.pop(p_key)
            print "{0} appears to be represented multiple times in patch dictionary.".format(p_key)
        for key in patch_dict.keys():
            if key.lower() in patch_name_list:
                success_patches[key] = patch_dict[key]
            else:
                error_patches[key] = patch_dict[key]
                print "{0} not in patch name list.".format(key)
        if not plotaxis:
            f,plotaxis = plt.subplots(1,1)
        if error_patches:
            self.plotPatchesWithNameAxes(error_patches,plotAxis=plotaxis)
            WrappedRetinotopicMapping.plotPatchesWithColor(error_patches,plotaxis=plotaxis,
                                                           zoom=zoom,alpha=alpha,
                                                           markersize=markersize,
                                                           cmap=get_cmap_from_hex(error_color))
                                                    
        if success_patches:
            self.plotPatchesWithNameAxes(success_patches,plotAxis=plotaxis)
            WrappedRetinotopicMapping.plotPatchesWithColor(success_patches,plotaxis=plotaxis,
                                                           zoom=zoom,alpha=alpha,
                                                           markersize=markersize,
                                                           cmap=get_cmap_from_hex(success_color))
        return plotaxis
        
                
            
        
    def save_TrialDict_pkl(self,f_name=None):
        if not f_name:
            f_name = "{0}.pkl".format(self.getName())
            print "file name: {0}".format(os.path.join(os.getcwd(),f_name))
        else:
            print "file name: {0}".format(f_name)
        FileTools.saveFile(f_name,self.generateTrialDict())
    
    def saveFinalPatchBorders(self,fig,png_path=None,pdf_path=None,png_dpi=300,
                              pdf_dpi=600):
        if not png_path:
            png_path = os.path.join(os.getcwd(),
                                    "{0}_borders.png".format(self.getName()))
        if not pdf_path:
            pdf_path = os.path.join(os.getcwd(),
                                    "{0}_borders.pdf".format(self.getName()))

        print "png path: {0}".format(png_path)
        print "pdf path: {0}".format(pdf_path)
        
        fig.savefig(png_path,dpi=png_dpi)
        fig.savefig(pdf_path,dpi=pdf_dpi)         
            
    @staticmethod
    def plotPatchesWithColor(patches,plotaxis=None,zoom=1,alpha=0.5,
                             markersize=5,cmap=None):
        if plotaxis == None:
            f = plt.figure()
            plotaxis = f.add_axes([1,1,1,1])

        imageHandle = {}
        for key, value in patches.iteritems():
    
            if zoom > 1:
                ni_zoom = ni.zoom(value.array,zoom,order=0)
                currPatch = RetinotopicMapping.Patch(ni_zoom,value.sign)
            else:
                currPatch = value
    
            h = plotaxis.imshow(currPatch.getSignedMask(),vmax=1,vmin=-1,
                                interpolation='nearest',alpha=alpha,
                                cmap=cmap)
            plotaxis.plot(currPatch.getCenter()[1],currPatch.getCenter()[0],
                          '.k', markersize=markersize*zoom)
            imageHandle.update({'handle_'+key:h})
    
        plotaxis.set_xlim([0, currPatch.array.shape[1]-1])
        plotaxis.set_ylim([currPatch.array.shape[0]-1, 0])
        return imageHandle,plotaxis    
    
    
    @staticmethod
    def load_from_pkl(pkl_path):
        trial,traces = RetinotopicMapping.loadTrial(pkl_path)
        _trial_kwargs = {}
        _trial_attrs = {}
        
        _trial_kwargs["mouseID"] = trial.mouseID
        _trial_kwargs["dateRecorded"] = trial.dateRecorded
        _trial_kwargs["trialNum"] = trial.trialNum
        _trial_kwargs["mouseType"] = trial.mouseType
        _trial_kwargs["visualStimType"] = trial.visualStimType
        _trial_kwargs["visualStimBackground"] = trial.visualStimBackground
        _trial_kwargs["imageExposureTime"] = trial.imageExposureTime
        _trial_kwargs["altPosMap"] = trial.altPosMap
        _trial_kwargs["aziPosMap"] = trial.aziPosMap
        _trial_kwargs["altPowerMap"] = trial.altPowerMap
        _trial_kwargs["aziPowerMap"] = trial.aziPowerMap
        _trial_kwargs["vasculatureMap"] = trial.vasculatureMap
        _trial_kwargs["params"] = trial.params
        _trial_kwargs["isAnesthetized"] = trial.isAnesthetized        
        
        _trial_attrs["altPosMapf"] = getattr(trial,"altPosMapf",None)
        _trial_attrs["aziPosMapf"] = getattr(trial,"aziPosMapf",None)
        _trial_attrs["altPowerMapf"] = getattr(trial,"altPowerMapf",None)
        _trial_attrs["aziPowerMapf"] = getattr(trial,"aziPowerMapf",None)
        _trial_attrs["finalPatches"] = getattr(trial,"finalPatches",None)
        _trial_attrs["finalPatchesMarked"] = getattr(trial,
                                                     "finalPatchesMarked",None)
        _trial_attrs["signMap"] = getattr(trial,"signMap",None)
        _trial_attrs["signMapf"] = getattr(trial,"signMapf",None)
        _trial_attrs["rawPatchMap"] = getattr(trial,"rawPatchMap",None)
        _trial_attrs["rawPatches"] = getattr(trial,"rawPatches",None)
        _trial_attrs["eccentricityMapf"] = getattr(trial,"eccentricityMapf",
                                                   None)

        new_trial = WrappedRetinotopicMapping(**_trial_kwargs)
        for attr,val in _trial_attrs.iteritems():
            setattr(new_trial,attr,val)
        
        return new_trial    

patch_colors={-1:"#0000ff",1:"#ff0000"}

def get_cmap_from_hex(hex_color):
    cmap = mpl_color.ListedColormap(hex_color,hex_color)
    cm.register_cmap(cmap=cmap)
    return cmap