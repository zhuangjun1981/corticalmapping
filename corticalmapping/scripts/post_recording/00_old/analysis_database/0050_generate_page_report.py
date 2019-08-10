import os
import corticalmapping.DatabaseTools as dt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import h5py
import datetime

area_lim = 100.
nwb_folder = 'nwbs'

summary_fn = 'big_roi_table_190404145421.xlsx'
sheet_n = 'f_center_raw'

params = dt.ANALYSIS_PARAMS
params['trace_type'] = sheet_n
params['response_window_dgc'] = [0., 1.5]

plot_params = dt.PLOTTING_PARAMS
plot_params['response_type_for_plot'] = 'zscore'

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

db_df = pd.read_excel(summary_fn, sheet_name=sheet_n)
print(db_df.head())

pdf_fn = 'page_report_{}.pdf'.format(datetime.datetime.now().strftime('%y%m%d%H%M%S'))
pdff = PdfPages(pdf_fn)

for row_i, row in db_df.iterrows():
    # if row['roi_area'] >= area_lim and row['skew_fil'] >= 0.5:
    if row['roi_area'] >= area_lim:

        print('{}_{}; {}; {}'.format(row['date'], row['mouse_id'], row['plane_n'], row['roi_n']))

        nwb_fn = '{}_{}_110.nwb'.format(row['date'], row['mouse_id'])
        nwb_f = h5py.File(os.path.join(nwb_folder, nwb_fn), 'r')

        f = dt.roi_page_report(nwb_f=nwb_f,
                               plane_n=row['plane_n'],
                               roi_n=row['roi_n'],
                               params=params,
                               plot_params=plot_params)
        nwb_f.close()
        pdff.savefig(f)
        f.clear()
        plt.close()

pdff.close()