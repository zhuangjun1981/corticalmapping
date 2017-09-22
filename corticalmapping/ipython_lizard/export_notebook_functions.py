# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 17:09:48 2015

@author: chrism
"""

import json
import os
import urllib2
from IPython.lib import kernel
import shutil
#conversion-related imports
from nbconvert import export_by_name
from nbconvert.writers import FilesWriter
from nbformat import read,NO_CONVERT
from nbconvert.utils.exceptions import ConversionException

from warnings import warn

file_writer = FilesWriter() #writer for incase the default implementation fails, i think creating it like this is fine for now...

def get_current_notebook_path(notebook_url='http://127.0.0.1:8888/api/sessions'):
    connection_file_path = kernel.get_connection_file()
    connection_file = os.path.basename(connection_file_path)
    kernel_id = connection_file.split('-', 1)[1].split('.')[0]
    
    sessions = json.load(urllib2.urlopen(notebook_url))
    for sess in sessions:
        if sess['kernel']['id'] == kernel_id:
            notebook_path = str(sess["notebook"]["path"])
            break
    else:
        notebook_path = None
    return notebook_path

def export(name,nb,export_ftype="html"):
    notebook_name = name[:name.rfind('.')]
    resources = {}
    resources['unique_key'] = notebook_name
    resources['output_files_dir'] = '%s_files' % notebook_name
    try:
        output, resources = export_by_name(export_ftype,nb)
    except ConversionException as e:
        warn(str(e))
    else:
        write_results = file_writer.write(output,resources,
                                          notebook_name=notebook_name)
    
    print "Successfully exported: {0} as {1}".format(notebook_name,
                                                      export_ftype)

def export_current_notebook(backup_dir=None,export_ftype="html"):
    notebook_path = get_current_notebook_path()
    notebook_name = os.path.basename(notebook_path)
    
    with open(notebook_name,"r") as f:
        export(notebook_name,read(f,NO_CONVERT),export_ftype)
    
    if backup_dir:
        full_notebook_path = os.path.abspath(notebook_path)
        if not os.path.exists(full_notebook_path): #if default method fails to create a real path, use the more hacky method which makes alot of assumptions about the cwd
            full_notebook_path = os.path.join(os.getcwd(),notebook_name)
        exported_notebook_path = full_notebook_path.replace("ipynb","html")
        shutil.copy(full_notebook_path,backup_dir)
        shutil.copy(exported_notebook_path,backup_dir)
        
        print "Successfully backed_up to: {0}".format(backup_dir)