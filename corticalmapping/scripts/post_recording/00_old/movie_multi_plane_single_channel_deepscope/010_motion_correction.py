import os
import stia.motion_correction as mc

def run():

    data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data_rabies_project" \
                  r"\180328-M360495-deepscope\04\04_"

    curr_folder = os.path.dirname(os.path.realpath(__file__))
    os.chdir(curr_folder)

    plane_ns = [p for p in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, p))]
    plane_ns.sort()
    print('planes:')
    print('\n'.join(plane_ns))

    for plane_n in plane_ns:
        print('\nprocessing plane: {}'.format(plane_n))
        plane_folder = os.path.join(data_folder, plane_n)
        f_paths, _ = mc.motion_correction(input_folder=plane_folder,
                                          input_path_identifier='.tif',
                                          process_num=3,
                                          output_folder=os.path.join(plane_folder, 'corrected'),
                                          anchor_frame_ind_chunk=10,
                                          anchor_frame_ind_projection=0,
                                          iteration_chunk=10,
                                          iteration_projection=10,
                                          max_offset_chunk=(50., 50.),
                                          max_offset_projection=(50., 50.),
                                          align_func=mc.phase_correlation,
                                          preprocessing_type=0,
                                          fill_value=0.)

        offsets_path = os.path.join(plane_folder, 'corrected', 'correction_offsets.hdf5')

        mc.apply_correction_offsets(offsets_path=offsets_path,
                                    path_pairs=zip(f_paths, f_paths),
                                    output_folder=os.path.join(plane_folder, 'corrected'),
                                    process_num=3,
                                    fill_value=0.,
                                    avi_downsample_rate=20,
                                    is_equalizing_histogram=True)

if __name__ == "__main__":
    run()

