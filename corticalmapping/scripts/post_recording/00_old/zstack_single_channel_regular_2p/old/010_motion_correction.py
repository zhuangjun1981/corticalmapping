import os
import stia.motion_correction as mc

data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data_rabies_project" \
              r"\180322-M360495-2p\zstack_zoom4"

input_path_identifier = '.tif'
process_num = 1
anchor_frame_ind_chunk = 100
anchor_frame_ind_projection = 0
iteration_chunk = 10
iteration_projection = 5
max_offset_chunk = (30., 30.)
max_offset_projection = (100., 100.)
align_func = mc.phase_correlation
fill_value = 0.
avi_downsample_rate = 20

sub_folder_ns = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]
sub_folder_ns.sort()
print('\n'.join(sub_folder_ns))

for sub_folder_n in sub_folder_ns:

    sub_folder = os.path.join(data_folder, sub_folder_n)

    f_paths, _ = mc.motion_correction(input_folder=sub_folder,
                                      input_path_identifier=input_path_identifier,
                                      process_num=process_num,
                                      output_folder=sub_folder,
                                      anchor_frame_ind_chunk=anchor_frame_ind_chunk,
                                      anchor_frame_ind_projection=anchor_frame_ind_projection,
                                      iteration_chunk=iteration_chunk,
                                      iteration_projection=iteration_projection,
                                      max_offset_chunk=max_offset_chunk,
                                      max_offset_projection=max_offset_projection,
                                      align_func=align_func,
                                      fill_value=fill_value)

    print('\n'.join(f_paths))

    offsets_path = os.path.join(sub_folder, 'correction_offsets.hdf5')
    path_pairs = zip(f_paths, f_paths)
    mc.apply_correction_offsets(offsets_path=offsets_path,
                                path_pairs=path_pairs,
                                process_num=process_num,
                                fill_value=fill_value,
                                output_folder=sub_folder,
                                avi_downsample_rate=avi_downsample_rate)
