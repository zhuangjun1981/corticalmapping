import numpy as np
import scipy.ndimage as ni


def interpolate_nans(arr):
    """
    fill the nans in a 1d array by interpolating value on both sides
    """

    if len(arr.shape) != 1:
        raise ValueError('input arr should be 1d array.')

    nan_ind = np.isnan(arr)

    nan_pos = nan_ind.nonzero()[0]
    # print(nan_pos)
    data_pos = (~nan_ind).nonzero()[0]
    # print(data_pos)
    data = arr[~nan_ind]
    # print(data)

    arr1 = np.array(arr)
    arr1[nan_ind] = np.interp(nan_pos, data_pos, data)

    return arr1


def get_pupil_area(pupil_shapes, fs, ell_thr=0.5, median_win=3.):
    """
    from elliptic pupil shapes, calculate pupil areas and filter out outliers.

    step 1: calculate area
    step 2: nan the shape with ellipticity larger than ell_thr
            ellipticity = (a - b) / b
    step 3: interpolate the nans
    step 4: median filter with length of median_win

    :param pupil_shapes: 2d array, each row: each sampling point; column0: axis0; column1: axis1; column2: angle
    :param fs: float, Hz, sampling rate
    :param ell_thr: float, (0. 1.], threshold for ellipticity
    :param median_win: float, sec, window length of median filter

    :return: 1d array of pupil area.
    """

    if len(pupil_shapes.shape) != 2:
        raise ValueError('input pupil_shapes should be 2d array.')

    if pupil_shapes.shape[1] < 2:
        raise ValueError('input pupil_shapes should have at least 2 columns.')

    area = np.pi * pupil_shapes[:, 0] * pupil_shapes[:, 1]
    ax1 = np.max(pupil_shapes[:, 0:2], axis=1)
    ax2 = np.min(pupil_shapes[:, 0:2], axis=1)
    ell = (ax1 - ax2) / ax1
    area[ell > ell_thr] = np.nan
    area = interpolate_nans(area)
    area = ni.median_filter(area, int(fs * median_win))

    return area

if __name__ == '__main__':

    # ============================================================================================================
    y = np.array([1, 1, 1, np.nan, np.nan, 2, 2, np.nan, 0])
    y1 = interpolate_nans(y)
    print(y1)
    # ============================================================================================================