import os
import stia.motion_correction as mc

input_folder = 'cell1'

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

input_path_identifier = 'zstack'

output_folder = os.path.join(input_folder, 'corrected')
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)

process_num = 1
anchor_frame_ind_chunk = 10
anchor_frame_ind_projection = 0
iteration_chunk = 10
iteration_projection = 5
max_offset_chunk = (30., 30.)
max_offset_projection = (30., 30.)
align_func = mc.phase_correlation
fill_value = 0.
avi_downsample_rate = 20

f_paths, _ = mc.motion_correction(input_folder=input_folder,
                                  input_path_identifier=input_path_identifier,
                                  process_num=process_num,
                                  output_folder=output_folder,
                                  anchor_frame_ind_chunk=anchor_frame_ind_chunk,
                                  anchor_frame_ind_projection=anchor_frame_ind_projection,
                                  iteration_chunk=iteration_chunk,
                                  iteration_projection=iteration_projection,
                                  max_offset_chunk=max_offset_chunk,
                                  max_offset_projection=max_offset_projection,
                                  align_func=align_func,
                                  fill_value=fill_value)

print('\n'.join(f_paths))

offsets_path = os.path.join(output_folder, 'correction_offsets.hdf5')
path_pairs = zip(f_paths, f_paths)
mc.apply_correction_offsets(offsets_path=offsets_path,
                            path_pairs=path_pairs,
                            process_num=process_num,
                            fill_value=fill_value,
                            output_folder=output_folder,
                            avi_downsample_rate=avi_downsample_rate)
