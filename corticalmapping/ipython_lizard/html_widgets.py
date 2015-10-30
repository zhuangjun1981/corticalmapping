# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 11:53:37 2015

@author: chrism
"""
#from PyQt4 import QtGui,QtCore
from IPython.display import display,HTML
from ipywidgets import interact
import ipywidgets as widgets
import os
from functools import partial

from export_notebook_functions import export_current_notebook

def getSignMapWidget(RetinotopicTrial,
                     phaseMapFilterSigmaDefault=1.0,
                     signMapFilterSigmaDefault=9.0,
                     phaseMapFilterSigmaRange=(-1,2,0.1),
                     signMapFilterSigmaRange=(0,20,1)):
    phaseSlider = widgets.FloatSlider(value=phaseMapFilterSigmaDefault,
                                      min=phaseMapFilterSigmaRange[0],
                                      max=phaseMapFilterSigmaRange[1],
                                      step=phaseMapFilterSigmaRange[2],
                                      description="phaseMapFilterSigma:")
    signSlider = widgets.FloatSlider(value=signMapFilterSigmaDefault,
                                     min=signMapFilterSigmaRange[0],
                                     max=signMapFilterSigmaRange[1],
                                     step=signMapFilterSigmaRange[2],
                                     description="signMapFilterSigma:")
    
    interact(RetinotopicTrial.getSignMap,phaseMapFilterSigma=phaseSlider,
             signMapFilterSigma=signSlider)
    
def getRawPatchMapWidget(RetinotopicTrial,
                         signMapThrDefault=0.35,
                         openIterDefault=3,
                         closeIterDefault=3,
                         signMapThrRange=(0,1,0.01),
                         openIterRange=(0,5,1),
                         closeIterRange=(0,5,1)):
    signSlider = widgets.FloatSlider(value=signMapThrDefault,
                                     min=signMapThrRange[0],
                                     max=signMapThrRange[1],
                                     step=signMapThrRange[2],
                                     description="signMapThr:")
    openSlider = widgets.FloatSlider(value=openIterDefault,
                                     min=openIterRange[0],
                                     max=openIterRange[1],
                                     step=openIterRange[2],
                                     description="openIter:")
    closeSlider = widgets.FloatSlider(value=closeIterDefault,
                                      min=closeIterRange[0],
                                      max=closeIterRange[1],
                                      step=openIterRange[2],
                                      description="closeIter:")
    
    interact(RetinotopicTrial.getRawPatchMap,signMapThr=signSlider,
             openIter=openSlider,closeIter=closeSlider)

def getRawPatchesWidget(RetinotopicTrial,
                        dilationIterDefault=15,
                        borderWidthDefault=1,
                        smallPatchThrDefault=100,
                        dilationIterRange=(0,30,1),
                        borderWidthRange=(1,10,1),
                        smallPatchThrRange=(0,200,10)):
    dilationSlider = widgets.FloatSlider(value=dilationIterDefault,
                                         min=dilationIterRange[0],
                                         max=dilationIterRange[1],
                                         step=dilationIterRange[2],
                                         description="dilationIter:")
    borderSlider = widgets.FloatSlider(value=borderWidthDefault,
                                       min=borderWidthRange[0],
                                       max=borderWidthRange[1],
                                       step=borderWidthRange[2],
                                       description="borderWidth:")
    smallSlider = widgets.FloatSlider(value=smallPatchThrDefault,
                                      min=smallPatchThrRange[0],
                                      max=smallPatchThrRange[1],
                                      step=smallPatchThrRange[2],
                                      description="smallPatchThr:")
    
    interact(RetinotopicTrial.getRawPatches,dilationIter=dilationSlider,
            borderWidth=borderSlider,smallPatchThr=smallSlider)

def getEccentricityMapWidget(RetinotopicTrial,
                             eccMapFilterSigmaDefault=10.0,
                             eccMapFilterSigmaRange=(10.0,200.0,10.0)):
    eccSlider = widgets.FloatSlider(value=eccMapFilterSigmaDefault,
                                    min=eccMapFilterSigmaRange[0],
                                    max=eccMapFilterSigmaRange[1],
                                    step=eccMapFilterSigmaRange[2],
                                    description="eccMapFilterSigma:")
    interact(RetinotopicTrial.getEccentricityMap,eccMapFilterSigma=eccSlider)

def splitPatchesWidget(RetinotopicTrial,
                       visualSpacePixelSizeDefault=0.5,
                       visualSpaceCloseIterDefault=15,
                       splitLocalMinCutStepDefault=5.0,
                       splitOverlapThrDefault=1.2,
                       borderWidthDefault=1,
                       visualSpacePixelSizeRange=(0,1,0.1),
                       visualSpaceCloseIterRange=(0,30,1),
                       splitLocalMinCutStepRange=(0,10.0,0.2),
                       splitOverlapThrRange=(0,2.4,0.1),
                       borderWidthRange=(1,10,1)):

    visualSpacePSlider = widgets.FloatSlider(value=visualSpacePixelSizeDefault,
                                             min=visualSpacePixelSizeRange[0],
                                             max=visualSpacePixelSizeRange[1],
                                             step=visualSpacePixelSizeRange[2],
                                             description="visualSpacePixelSize:")
    visualSpaceCSlider = widgets.FloatSlider(value=visualSpaceCloseIterDefault,
                                             min=visualSpaceCloseIterRange[0],
                                             max=visualSpaceCloseIterRange[1],
                                             step=visualSpaceCloseIterRange[2],
                                             description="visualSpaceCloseIter:")
    splitLocalSlider = widgets.FloatSlider(value=splitLocalMinCutStepDefault,
                                           min=splitLocalMinCutStepRange[0],
                                           max=splitLocalMinCutStepRange[1],
                                           step=splitLocalMinCutStepRange[2],
                                           description="splitLocalMinCutStep:")
    splitOverlapSlider = widgets.FloatSlider(value=splitOverlapThrDefault,
                                             min=splitOverlapThrRange[0],
                                             max=splitOverlapThrRange[1],
                                             step=splitOverlapThrRange[2],
                                             description="splitOverlapThr:")
    borderSlider = widgets.FloatSlider(value=borderWidthDefault,
                                       min=borderWidthRange[0],
                                       max=borderWidthRange[1],
                                       step=borderWidthRange[2],
                                       description="borderWidth:")
    
    interact(RetinotopicTrial.splitPatches,
             visualSpacePixelSize=visualSpacePSlider,
             visualSpaceCloseIter=visualSpaceCSlider,
             splitLocalMinCutStep=splitLocalSlider,
             splitOverlapThr=splitOverlapSlider,
             borderWidth=borderSlider)

def mergePatchesWidget(RetinotopicTrial,
                 mergeOverlapThrDefault=0.1,
                 visualSpacePixelSizeDefault=0.5,
                 visualSpaceCloseIterDefault=15,
                 borderWidthDefault=1,
                 smallPatchThrDefault=100,
                 mergeOverlapThrRange=(-1,1,0.1),
                 visualSpacePixelSizeRange=(-2,2,0.1),
                 visualSpaceCloseIterRange=(0,30,1),
                 borderWidthRange=(1,10,1),
                 smallPatchThrRange=(0,200,10)):
    mergeSlider = widgets.FloatSlider(value=mergeOverlapThrDefault,
                                      min=mergeOverlapThrRange[0],
                                      max=mergeOverlapThrRange[1],
                                      step=mergeOverlapThrRange[2],
                                      description="mergeOverlapThr:")
    visualSpacePSlider = widgets.FloatSlider(value=visualSpacePixelSizeDefault,
                                             min=visualSpacePixelSizeRange[0],
                                             max=visualSpacePixelSizeRange[1],
                                             step=visualSpacePixelSizeRange[2],
                                             description="visualSpacePixelSize:")
    visualSpaceCSlider = widgets.FloatSlider(value=visualSpaceCloseIterDefault,
                                             min=visualSpaceCloseIterRange[0],
                                             max=visualSpaceCloseIterRange[1],
                                             step=visualSpaceCloseIterRange[2],
                                             description="visualSpaceCloseIter:")
    borderSlider = widgets.FloatSlider(value=borderWidthDefault,
                                       min=borderWidthRange[0],
                                       max=borderWidthRange[1],
                                       step=borderWidthRange[2],
                                       description="borderWidth:")
    smallSlider = widgets.FloatSlider(value=smallPatchThrDefault,
                                      min=smallPatchThrRange[0],
                                      max=smallPatchThrRange[1],
                                      step=smallPatchThrRange[2],
                                      description="smallPatchThr:")

    interact(RetinotopicTrial.mergePatches,
             mergeOverlapThr=mergeSlider,
             visualSpacePixelSize=visualSpacePSlider,
             visualSpaceCloseIter=visualSpaceCSlider,
             borderWidth=borderSlider,
             smallPatchThr=smallSlider)

def saveFinalResultWidget(trial,fig,pkl_path,pkl_save_path=None,png_dpi=300,pdf_dpi=600):
    allSaveButton = widgets.Button(description="save all")
    allSaveButton.width = "400px"
    allSaveButton.font_size = "20px"
    allSaveButton.margin = "10px 0px 10px 72px"
    png_path = os.path.join(os.path.dirname(pkl_path),
                            "{0}_borders.png".format(trial.getName()))
    pdf_path = os.path.join(os.path.dirname(pkl_path),
                            "{0}_borders.pdf".format(trial.getName()))
    if not pkl_save_path:
        pkl_save_path = os.path.join(os.path.dirname(pkl_path),
                                     "{0}.pkl".format(trial.getName()))
    save_all_func = partial(save_all_callback,fig,trial,png_path,pdf_path,png_dpi,pdf_dpi,pkl_save_path)
    allSaveButton.on_click(save_all_func)
    display(allSaveButton)

def save_all_callback(fig,trial,png_path,pdf_path,png_dpi,pdf_dpi,pkl_save_path,button):
    save_all(fig,trial,png_path,pdf_path,png_dpi,pdf_dpi,pkl_save_path)

def save_all(fig,trial,png_path,pdf_path,png_dpi,pdf_dpi,pkl_save_path):
    saveFigure(fig,png_path,png_dpi)
    saveFigure(fig,pdf_path,pdf_dpi)
    if pkl_save_path and os.path.exists(pkl_save_path):
        pkl_save_path = avoidPathOverwrite(pkl_save_path)
    trial.save_TrialDict_pkl(pkl_save_path)
    
    export_current_notebook(os.path.dirname(png_path))
    

#def saveTrialDictPkl_button_callback(trial,fpath,button):
#    if fpath and os.path.exists(fpath):
#        fpath = avoidPathOverwrite(fpath)
#    trial.save_TrialDict_pkl(fpath)
#
#def saveFigure_button_callback(fig,fpath,dpi,button):
#    saveFigure(fig,fpath,dpi)
    
def saveFigure(fig,fpath,dpi):
    if os.path.exists(fpath):
        fpath = avoidPathOverwrite(fpath)
    fig.savefig(fpath,dpi=dpi)
    print "Saved figure to: {0}".format(fpath)

def avoidPathOverwrite(fpath,ntries=10):
    dirname = os.path.dirname(fpath)
    partial_fname,ftype = os.path.basename(fpath).split(".")
    for ntry in range(1,ntries+1):
        fname = "{0}_{1}.{2}".format(partial_fname,ntry,ftype)
        path = os.path.join(dirname,fname)
        if not os.path.exists(path):
            break
    else:
        raise Exception("Can't save without overwritting existant filenames.")
    return path

def raw_code_toggle():
    #stolen from: http://stackoverflow.com/questions/27934885/how-to-hide-code-from-cells-in-ipython-notebook-visualized-with-nbviewer
    display(HTML('''<script>
                    code_show=true; 
                    function code_toggle() {
                        if (code_show){
                        $('div.input').hide();
                        } else {
                        $('div.input').show();
                        }
                        code_show = !code_show
                    } 
                    $( document ).ready(code_toggle);
                    </script>
                    <form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>'''))

#class SaveFinalPatchBorderFigureWidget(object):
#    
#    PNG_DPI = 300
#    PDF_DPI = 600
#    DPI_HARD_LIMIT = 10000
#    
#    def __init__(self,trial,fig,png_dpi_default=None,pdf_dpi_default=None):
#        self.default_png_name = os.path.join(os.getcwd(),
#                                             "{0}_borders.png".format(trial.getName()))
#        self.default_pdf_name = os.path.join(os.getcwd(),
#                                             "{0}_borders.pdf".format(trial.getName()))
#        
#        self.fig = fig
#        
#        self._build_widgets()
#        self._style_widgets()
#        self._setup_callbacks()
#        
#    def _build_widgets(self):
#        self.pngDpiFloatBox = widgets.BoundedIntText(max=self.DPI_HARD_LIMIT,
#                                                     description="png dpi:")
#        self.pdfDpiFloatBox = widgets.BoundedIntText(max=self.DPI_HARD_LIMIT,
#                                                     description="pdf dpi:")
#        self.pngSaveButton = widgets.Button(description="save png")
#        self.pdfSaveButton = widgets.Button(description="save pdf")
#    
#    def _style_widgets(self):
#        self.pngSaveButton.width = "280px"
#        self.pngSaveButton.margin = "10px 0px 10px 72px"
#        self.pdfSaveButton.width = "280px"
#        self.pdfSaveButton.margin = "10px 0px 0px 72px"           
#        
#        self.pngDpiFloatBox.value = self.PNG_DPI
#        self.pdfDpiFloatBox.value = self.PDF_DPI
#    
#    def _setup_callbacks(self):
#        self.pngSaveButton.on_click(self.save_png)
#        self.pdfSaveButton.on_click(self.save_pdf)
#
#    def show(self):
#        display(self.pngDpiFloatBox)
#        display(self.pngSaveButton)
#        
#        display(self.pdfDpiFloatBox)
#        display(self.pdfSaveButton)
#
#    def save_png(self,button=None):
#        path = QtGui.QFileDialog.getSaveFileName(None,
#                                                 'Save file',
#                                                 "*.png")
#        dpi = self.pngDpiFloatBox.value
#        
#        if not dpi:
#            dpi = self.PNG_DPI
#            
#        if path:
#            path = str(path)
#        elif not path:
#            path = self.default_png_name
#            
#        self.fig.savefig(path,dpi=dpi)
#    
#    def save_pdf(self,button=None):
#        path = QtGui.QFileDialog.getSaveFileName(None,
#                                                 'Save file',
#                                                 "*.pdf") 
#        dpi = self.pdfDpiFloatBox.value
#        
#        if not dpi:
#            dpi = self.PNG_DPI
#            
#        if path:
#            path = str(path)
#        elif not path:
#            path = self.default_png_name
#
#        self.fig.savefig(path,dpi=dpi)


