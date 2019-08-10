import os
import sys
import h5py
import time
from multiprocessing import Pool
import corticalmapping.NwbTools as nt
import corticalmapping.DatabaseTools as dt

process_num = 5
save_folder = 'repacked'

curr_folder = os.path.dirname(os.path.abspath(__file__))
os.chdir(curr_folder)

def run_single_file_for_multi_process(params):

    file_path, save_dir, file_ind, total_file_num, t0 = params

    nwb_f = h5py.File(file_path)
    if 'STRFs' in nwb_f['analysis']:
        del nwb_f['analysis/STRFs']

    if 'strf_001_LocallySparseNoiseRetinotopicMapping' in nwb_f['analysis']:
        del nwb_f['analysis/strf_001_LocallySparseNoiseRetinotopicMapping']

    if 'response_table_003_DriftingGratingCircleRetinotopicMapping' in nwb_f['analysis']:
        del nwb_f['analysis/response_table_003_DriftingGratingCircleRetinotopicMapping']

    if 'response_table_001_DriftingGratingCircleRetinotopicMapping' in nwb_f['analysis']:
        del nwb_f['analysis/response_table_001_DriftingGratingCircleRetinotopicMapping']

    nwb_f.close()

    print('{:6.1f} min; {} / {}; {}: repacking ...'.format((time.time() - t0) / 60.,
                                                           file_ind+1,
                                                           total_file_num,
                                                           file_path))
    # save_path = os.path.join(save_dir, os.path.splitext(os.path.split(file_path)[1])[0] + '_repacked.nwb')
    save_path = os.path.join(save_dir, os.path.splitext(os.path.split(file_path)[1])[0] + '_repacked.nwb')
    sys_str = "h5repack {} {}".format(file_path, save_path)

    # print(sys_str)

    os.system(sys_str)

    print('{:6.1f} min; {} / {}; {}: Done.'.format((time.time() - t0) / 60.,
                                                   file_ind + 1,
                                                   total_file_num,
                                                   file_path))


def run():

    t0 = time.time()

    nwb_fns = [f for f in os.listdir(curr_folder) if f[-4:] == '.nwb']
    nwb_fns.sort()
    print('nwb files:')
    print('\n'.join(nwb_fns))

    param_lst = [(os.path.join(curr_folder, nwb_fn),
    	          os.path.join(curr_folder, save_folder),
    	          file_ind,
    	          len(nwb_fns),
    	          t0) for file_ind, nwb_fn in enumerate(nwb_fns)]

    p = Pool(process_num)
    p.map(run_single_file_for_multi_process, param_lst)

if __name__ == "__main__":
    run()
