@cython.boundscheck(False)
@cython.wraparound(False)
def find_dlims(
        np.ndarray[np.float64_t, ndim=1] mag_arr,
        double ab_mag_limit):
    """
    Find the indices of the stars in an array that are visible given a limit.
    Extended description of function.
    Parameters
    ----------
    mag_arr : np arr
        Numpy array of absolute magnitude.
    ab_mag_limit : float
        The minimum intrinsic brightness needed to be seen.
    Returns
    -------
    np arr
        Returns a Numpy array containing the indices of visible stars
    """
    line = '-' * 85
    print(line)
    print('\n[ find_dlims ]\nfinding indices of visible stars : ',
          ab_mag_limit, ' abs mag lim')
    print('mean of mag_arr:', mag_arr.mean())
    return np.nonzero(mag_arr < ab_mag_limit)[0]


@cython.boundscheck(False)
@cython.wraparound(False)
def box_lims(
        np.ndarray[np.float64_t, ndim=1] px,
        np.ndarray[np.float64_t, ndim=1] py,
        np.float64_t box_size,  np.float64_t box_step):
    cdef:
        size_t i
        np.int32_t n_stars = px.shape[0]
        np.float64_t outter_box = (box_size + box_step)
        np.float64_t neg_outter_box = -(outter_box)
        np.float64_t neg_box_size = -(box_size)
        np.ndarray[np.int64_t, ndim = 1, mode = 'c'] idx_arr = np.zeros(
            (n_stars), dtype=np.int64)
        np.int_t n_threads = np.int(px.shape[0]/1e6)
    # Set (threads)
    if n_threads >= NUM_PROCESSORS:
        n_threads = NUM_PROCESSORS - 2
        if n_threads <= 0:
            n_threads = 1
        print('box_lims(): number of processors=' +
              str(NUM_PROCESSORS) + ' - number of threads=' + str(n_threads))
        with nogil, parallel(num_threads=n_threads):
            for i in prange(n_stars, schedule='dynamic'):
                if (px[i] > neg_outter_box and px[i] < outter_box):
                    if (py[i] >= box_size and py[i] < outter_box):
                        idx_arr[i] = 1
                    elif (py[i] <= neg_box_size and py[i] > neg_outter_box):
                        idx_arr[i] = 1
                if (py[i] > neg_outter_box and py[i] < outter_box):
                    if (px[i] >= box_size and px[i] < outter_box):
                        idx_arr[i] = 1
                    elif (px[i] <= neg_box_size and px[i] > neg_outter_box):
                        idx_arr[i] = 1
        return idx_arr


cdef extern from "math.h":
    double M_PI