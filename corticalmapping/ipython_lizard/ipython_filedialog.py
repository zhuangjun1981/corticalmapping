# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 17:32:35 2015

@author: chrism
"""
import Tkinter
import tkFileDialog

import ipywidgets as widgets
from IPython.display import display,Javascript

def _raise_above_all(window):
    window.attributes('-topmost', 1)
    window.attributes('-topmost', 0)

def get_file_path(initial_dir):
    root = Tkinter.Tk()
    root.withdraw()
    
    root.overrideredirect(True)
    root.geometry('0x0+0+0')
    
    root.deiconify()
    root.lift()
    _raise_above_all(root)
    root.focus_force()
    file_types = [('pickles', '.pkl'),('all files', '.*')]
    file_path = tkFileDialog.askopenfilename(parent=root,initialdir=initial_dir,
                                             filetypes=file_types)
        
    #root.destroy()
        
    return file_path

class IPythonTkinterFileDialog(object):
    DEFAULT_DIR = r"C:"  
    
    def __init__(self,initial_dir=DEFAULT_DIR):
        self.initial_dir = initial_dir
        self.file_path = None
        
        self._build_ui()
        self._setup_callbacks()
        self._style_widgets()
        self.execute_below = True

    def _style_widgets(self):
        self.dialog_trigger_button.width = "400px"
        self.dialog_trigger_button.font_size = "20px" 
     
    def _build_ui(self):
        self.dialog_trigger_button = widgets.Button(description="Get File Path")
        
    def _setup_callbacks(self):
        self.dialog_trigger_button.on_click(self.set_file_path)
    
    def set_file_path(self,button):
        self.file_path = get_file_path(self.initial_dir)
        if self.execute_below:
            print "ha"
            display(Javascript('IPython.notebook.execute_cells_below()'))
    
    def show(self):
        display(self.dialog_trigger_button)