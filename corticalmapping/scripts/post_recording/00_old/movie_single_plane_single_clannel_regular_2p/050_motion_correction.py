import os
import stia.motion_correction as mc

def run():
    data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data_rabies_project" \
                  r"\180404-M360495-2p\2p_movie\reorged"

    curr_folder = os.path.dirname(os.path.realpath(__file__))
    os.chdir(curr_folder)

    mc.motion_correction(input_folder=data_folder,
                         input_path_identifier='.tif',
                         process_num=6,
                         output_folder=os.path.join(data_folder, 'corrected'),
                         anchor_frame_ind_chunk=10,
                         anchor_frame_ind_projection=0,
                         iteration_chunk=10,
                         iteration_projection=10,
                         max_offset_chunk=(40., 40.),
                         max_offset_projection=(40., 40.),
                         align_func=mc.phase_correlation,
                         preprocessing_type=0,
                         fill_value=0.)

    offsets_path = os.path.join(data_folder, 'corrected', 'correction_offsets.hdf5')
    fns = [f for f in os.listdir(data_folder) if f[-4:] == '.tif']
    fns.sort()
    f_paths = [os.path.join(data_folder, f) for f in fns]
    print('\nfile paths:')
    print('\n'.join(f_paths))

    mc.apply_correction_offsets(offsets_path=offsets_path,
                                path_pairs=zip(f_paths, f_paths),
                                output_folder=os.path.join(data_folder, 'corrected'),
                                process_num=6,
                                fill_value=0.,
                                avi_downsample_rate=10,
                                is_equalizing_histogram=True)

if __name__ == "__main__":
    run()