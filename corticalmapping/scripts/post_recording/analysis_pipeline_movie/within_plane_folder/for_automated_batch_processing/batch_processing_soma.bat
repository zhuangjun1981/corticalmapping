call activate bigmess

set PYTHONPATH=%PYTHONPATH%;E:\data\python_packages\corticalmapping;E:\data\python_packages\allensdk_internal;E:\data\python_packages\ainwb\ainwb

python 120_get_cells_file_soma.py
python 130_refine_cells_soma.py
python 140_get_weighted_rois_and_surrounds.py
python 150_get_raw_center_and_surround_traces.py
python 160_get_neuropil_subtracted_traces.py
python 135_generate_marked_avi.py