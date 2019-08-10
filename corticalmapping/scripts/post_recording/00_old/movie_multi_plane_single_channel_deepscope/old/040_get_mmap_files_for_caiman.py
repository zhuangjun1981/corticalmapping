import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend([r"E:\data\github_packages\CaImAn"])

import os
import numpy as np
import tifffile as tf
import caiman as cm
import h5py

def run():

    data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data_rabies_project" \
                  r"\180323-M360495-deepscope\02\02_"
    base_name = '180323_M360495_02'
    t_downsample_rate = 10.

    plane_ns = [p for p in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, p))]
    plane_ns.sort()
    print('planes:')
    print('\n'.join(plane_ns))

    ## start cluster
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=3, single_thread=False)

    for plane_n in plane_ns:
        print('\nprocessing {} ...'.format(plane_n))

        plane_folder = os.path.join(data_folder, plane_n, 'corrected')
        os.chdir(plane_folder)

        f_ns = [f for f in os.listdir(plane_folder) if f[-14:] == '_corrected.tif']
        f_ns.sort()
        print('\n'.join(f_ns))

        min_tot = 0
        for fn in f_ns:
            min_tot = min([min_tot, np.min(tf.imread(os.path.join(plane_folder, fn)))])
        print('minimum pixel value of entire movie: ' + str(min_tot))

        add_to_movie = 10. - min_tot  # the movie must be positive!!!
        t_ds_factor = 1. / t_downsample_rate  # use .2 or .1 if file is large and you want a quick answer
        f_paths = [os.path.join(plane_folder, f) for f in f_ns]

        name_new = cm.save_memmap_each(f_paths,
                                       dview=dview,
                                       base_name=base_name + '_' + plane_n + '_each',
                                       resize_fact=(1., 1., t_ds_factor),
                                       add_to_movie=add_to_movie)
        name_new.sort()

        fname_new = cm.save_memmap_join(name_new, base_name=base_name + '_' + plane_n, dview=dview,
                                        n_chunks=100)
        print('\n{}'.format(fname_new))

        save_file = h5py.File(os.path.join(plane_folder, 'caiman_segmentation_results.hdf5'))
        save_file['bias_added_to_movie'] = add_to_movie
        save_file.close()

        single_fns = [f for f in os.listdir(plane_folder) if '_each' in f]
        for single_fn in single_fns:
            os.remove(os.path.join(plane_folder, single_fn))

if __name__ == '__main__':
    run()