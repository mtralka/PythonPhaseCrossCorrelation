"""

 @title: Optimized Sentinel-2 Coregistration using Phase Cross Correlation
 @author: Matthew Tralka
 @date: May 2021
 @version: 0.1

    Algorithm References
    .. [0] Harris CR, Millman KJ, van der Walt SJ, Gommers R, Virtanen P, Cournapeau D, et al.
            Array programming with NumPy. Nature. 2020; 585:357–62.
    .. [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
           "Efficient subpixel image registration algorithms,"
           Optics Letters 33, 156-158 (2008). :DOI:`10.1364/OL.33.000156`
    .. [2] James R. Fienup, "Invariant error metrics for image reconstruction"
           Optics Letters 36, 8352-8357 (1997). :DOI:`10.1364/AO.36.008352`
    .. [3] Dirk Padfield. Masked Object Registration in the Fourier Domain.
           IEEE Transactions on Image Processing, vol. 21(5),
           pp. 2706-2718 (2012). :DOI:`10.1109/TIP.2011.2181402`
    .. [4] D. Padfield. "Masked FFT registration". In Proc. Computer Vision and
           Pattern Recognition, pp. 2918-2925 (2010).
           :DOI:`10.1109/CVPR.2010.5540032`

"""

from cython.parallel import prange
import numpy as np
import scipy.fft as fft
cimport cython
cimport numpy as np

DTYPE = np.int16

cdef find_shift(int[:, :] reference_image, int[:, :] moving_image,
                            int upsample_factor=1,
                            double overlap_ratio=0.3):

    src_freq = fft.fftn(reference_image)
    target_freq = fft.fftn(moving_image)
    image_product = src_freq * target_freq.conj()
    cross_correlation = fft.ifftn(image_product)

    cdef Py_ssize_t cc_x, cc_y

    cc_x = cross_correlation.shape[0]
    cc_y = cross_correlation.shape[1]

    cdef Py_ssize_t src_freq_x = reference_image.shape[0]
    cdef Py_ssize_t  src_freq_y = reference_image.shape[1]

    maxima = np.unravel_index(np.argmax(np.abs(cross_correlation)),(cc_x, cc_y))      
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in (src_freq_x, src_freq_y)])
    
    shifts = np.stack(maxima).astype(np.float64)
    shifts[shifts > midpoints] -= np.array((src_freq_x, src_freq_y))[shifts > midpoints]

    for dim in range(src_freq.ndim):
        if (src_freq_x, src_freq_y)[dim] == 1:
            shifts[dim] = 0

    return shifts

@cython.boundscheck(False)
@cython.wraparound(False)
def phase_cross_correlation(int[:, :] reference_arr, int[:, :] moving_arr, int window_size = 64, 
    int window_step = 64, double no_data = -9999., int upsample):
    
    cdef Py_ssize_t x_max = reference_arr.shape[0]
    cdef Py_ssize_t y_max = reference_arr.shape[1]

    assert tuple(reference_arr.shape) == tuple(moving_arr.shape)

    cdef np.ndarray total_shift = no_data * np.ones((x_max, y_max), dtype=DTYPE)
    cdef double[:,:] total_shift_view = total_shift

    cdef int window_start = window_size / 2

    cdef int[:] x_arrange = np.arange(window_start,x_max, window_step, dtype=DTYPE).astype('int')
    cdef int[:] y_arrange = np.arange(window_start, y_max, window_step, dtype=DTYPE).astype('int')

    cdef Py_ssize_t x, y
    cdef int row_start, row_end
    cdef int col_start, col_end
    cdef int offset_x, offset_y
    cdef int[:,:] reference_window, moving_window

    for x in x_arrange:
        for y in y_arrange:

            row_start = x - window_start
            row_end = x + window_start
            col_start = y - window_start
            col_end = y + window_start
            
            reference_window = reference_arr[row_start:row_end, col_start:col_end]
            moving_window = moving_arr[row_start:row_end, col_start:col_end]
            offset_pixels = find_shift(reference_window, moving_window)

            offset_x = int(offset_pixels[0])
            offset_y = int(offset_pixels[1])
            total_shift_view[row_start:row_end,col_start:col_end] = \
                np.sqrt(1.* offset_y * offset_y + offset_x * offset_x, dtype=np.double)

    return total_shift