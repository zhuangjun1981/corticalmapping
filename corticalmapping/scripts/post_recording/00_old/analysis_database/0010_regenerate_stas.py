import os
import time
from multiprocessing import Pool
import corticalmapping.NwbTools as nt
import corticalmapping.DatabaseTools as dt

process_num = 5
nwb_folder = 'repacked'
strf_t_win = [-0.5, 2.]
dgc_t_win = [-1., 2.5]

curr_folder = os.path.dirname(os.path.abspath(__file__))
os.chdir(curr_folder)

def run_single_file_for_multi_process(params):

    file_path, file_ind, total_file_num, t0 = params

    print('{:6.1f} min; {} / {}; {}: regenerating strf ...'.format((time.time() - t0) / 60.,
                                                                   file_ind + 1,
                                                                   total_file_num,
                                                                   file_path))

    nwb_f = nt.RecordedFile(file_path)
    lsn_name = '001_LocallySparseNoiseRetinotopicMapping'
    if dt.get_strf_grp_key(nwb_f.file_pointer) is None:
        if lsn_name in nwb_f.file_pointer['analysis/photodiode_onsets']:
            nwb_f.get_spatial_temporal_receptive_field_retinotopic_mapping(stim_name=lsn_name,
                                                                           time_window=strf_t_win,
                                                                           verbose=False)

    dgc_name = '003_DriftingGratingCircleRetinotopicMapping'
    if dt.get_dgcrm_grp_key(nwb_f.file_pointer) is None:
        if dgc_name in nwb_f.file_pointer['analysis/photodiode_onsets']:
            nwb_f.get_drifting_grating_response_table_retinotopic_mapping(stim_name=dgc_name,
                                                                          time_window=dgc_t_win)

    dgc_name = '001_DriftingGratingCircleRetinotopicMapping'
    if dt.get_dgcrm_grp_key(nwb_f.file_pointer) is None:
        if dgc_name in nwb_f.file_pointer['analysis/photodiode_onsets']:
            nwb_f.get_drifting_grating_response_table_retinotopic_mapping(stim_name=dgc_name,
                                                                          time_window=dgc_t_win)

    nwb_f.close()


def run():

    t0 = time.time()

    nwb_fns = [f for f in os.listdir(nwb_folder) if f[-4:] == '.nwb']
    nwb_fns.sort()
    print('nwb files:')
    print('\n'.join(nwb_fns))

    param_lst = [(os.path.join(curr_folder, nwb_folder, nwb_fn),
    	          file_ind,
    	          len(nwb_fns),
    	          t0) for file_ind, nwb_fn in enumerate(nwb_fns)]

    p = Pool(process_num)
    p.map(run_single_file_for_multi_process, param_lst)

if __name__ == "__main__":
    run()