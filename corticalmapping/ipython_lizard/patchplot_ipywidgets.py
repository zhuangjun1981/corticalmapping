# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:55:00 2015

@author: chrism
"""
from IPython.display import display
from ipywidgets import interact
import ipywidgets as widgets
import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mpl_color

class PatchPlotWidgets(object):
    
    #HIGHLIGHT_COLOR = "#0CF4C7"
    HIGHLIGHT_COLOR = "#ffffff"
    GUIDE_IMG = os.path.join(os.path.dirname(__file__),r"res\visual_area_summary.png") #need an absolute path for plt.imread? so damn annoying...

    @property
    def highlight_color(self):
        return self._highlight_color
    @highlight_color.setter
    def highlight_color(self,value):                     
        temp_cmap = mpl_color.ListedColormap(value,'temp')
        cm.register_cmap(cmap=temp_cmap)
        self._highlight_color = temp_cmap

    @property
    def guide_img(self):
        return self._guide_img
    @guide_img.setter
    def guide_img(self,path):
        self._guide_img = plt.imread(path)


    def __init__(self,retinotopic_mapping_trial,patch_dict,desired_patch_names,
                 highlight_color=HIGHLIGHT_COLOR,guide_img=GUIDE_IMG,
                 *ax_args,**ax_kwargs):
        """
        Requires a WrappedRetinotopicMapping obj NOT a RetinotopicMappingTrial
        """
        self.patches_dict = dict(patch_dict)
        self.trial = retinotopic_mapping_trial
        self.guide_img = guide_img
        self.highlight_color = highlight_color
        self.desired_patch_names = desired_patch_names
        
        self.ax_args = ax_args
        self.ax_kwargs = ax_kwargs        
        
        self._build_widgets()
        self._style_widgets()
        self._setup_callbacks()

    def _build_widgets(self):
        t_options = sorted(self.patches_dict.keys())
        self.patch_toggle_button_widget = widgets.ToggleButtons(description='Patches:',
                                                                options=t_options)
        self.patch_rename_text_widget = widgets.Text(description="Rename:")
        self.patch_rename_button_widget = widgets.Button(description="Submit")
    
    def _style_widgets(self):
        self.patch_rename_text_widget.margin = "10px 0px 0px 0px"
        self.patch_rename_button_widget.width = "280px"
        self.patch_rename_button_widget.margin = "10px 0px 0px 72px"
    
    def _setup_callbacks(self):
        self.patch_rename_button_widget.on_click(self._rename_patch)
        self.patch_rename_text_widget.on_submit(self._rename_patch)
    
    def show(self):
        interact(self._plot_patch,patch_name=self.patch_toggle_button_widget)
        display(self.patch_rename_text_widget)
        display(self.patch_rename_button_widget)

    def _plot_patch(self,patch_name):
#        if isinstance(self.guide_img,np.ndarray):
#            _,(p_ax,p_ax1) = plt.subplots(1,2,*self.ax_args,**self.ax_kwargs)
#            p_ax.imshow(self.guide_img,interpolation='nearest')
#            p_ax.set_title("Reference Map")
#            #p_ax.set_title("Sample Annotated Visual Sign Map from Garrett et. al. 2014")
#        else:
#            p_ax = None
        _,p_ax1 = plt.subplots(1,1,*self.ax_args,**self.ax_kwargs)
        self.trial.plotColoredPatchOnPatchBorders(self.patches_dict[patch_name],
                                                  self.patches_dict,self.desired_patch_names,
                                                  plotAxis=p_ax1)
                                    
#        fig,ax = self.trial.plotPatchesWithNameAxes(self.patches_dict,
#                                                    plotAxis=p_ax1)
#        temp_dict = {patch_name:self.patches_dict[patch_name]}
#        self.trial.plotPatchesWithColor(temp_dict,cmap=self.highlight_color,
#                                        plotaxis=p_ax1,alpha=0.5)
        p_ax1.set_title("Current Sign Map")

    #@staticmethod
    def plot_reference_img(self,ax=None,img=None,aspect="equal"):
        if not ax:
            fig,ax = plt.subplots(1,1,*self.ax_args,**self.ax_kwargs)
        
        if not img:
            img = self.guide_img
        
        ax.imshow(img,interpolation='nearest',aspect=aspect)
        ax.set_title("Reference Map")
            
    def _rename_patch(self,button):
        new_name = self.patch_rename_text_widget.value
        patch_list = list(self.patch_toggle_button_widget.options)
        
        if new_name in patch_list: 
            raise ValueError("Cannot have more than one patch with name: {0}".format(new_name))
        
        old_name = self.patch_toggle_button_widget.value
        old_idx = patch_list.index(old_name)
        patch_list.pop(old_idx)
        patch_list.append(new_name)
        self.patches_dict[new_name] = self.patches_dict.pop(old_name)
        toggle_button_options = self.patch_toggle_button_widget.options[old_idx:] + \
                                self.patch_toggle_button_widget.options[:old_idx]
                                
        for option in toggle_button_options:
            if option in patch_list:
                self.patch_toggle_button_widget.value = option
                break
        
        self.patch_toggle_button_widget.options = patch_list
        self.patch_rename_text_widget.value = ""
    


