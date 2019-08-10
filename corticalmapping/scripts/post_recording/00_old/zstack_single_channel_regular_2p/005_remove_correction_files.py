import os

base_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data_rabies_project" \
              r"\180502-M386507-2p\FOV4_zoom10"

step_fns = [f for f in os.listdir(base_folder) if f.split('_')[-2] == 'step']
step_fns.sort()
print('\n'.join(step_fns))

for step_fn in step_fns:

    print('\n' + step_fn)
    step_folder = os.path.join(base_folder, step_fn)
    fns = os.listdir(step_folder)

    if 'corrected_max_projection.tif' in fns:
        print('removing corrected_max_projection.tif')
        os.remove(os.path.join(step_folder, 'corrected_max_projection.tif'))

    if 'corrected_mean_projection.tif' in fns:
        print('removing corrected_mean_projection.tif')
        os.remove(os.path.join(step_folder, 'corrected_mean_projection.tif'))

    if 'correction_offsets.hdf5' in fns:
        print('removing correction_offsets.hdf5')
        os.remove(os.path.join(step_folder, 'correction_offsets.hdf5'))

    fn_cor = [f for f in fns if f[-14:] == '_corrected.tif']
    if len(fn_cor) == 1:
        print('removing ' + fn_cor[0])
        os.remove(os.path.join(step_folder, fn_cor[0]))