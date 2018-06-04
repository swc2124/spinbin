@cython.boundscheck(False)
@cython.wraparound(False)
def deposit_data(np.ndarray[np.float64_t, ndim=2] in_arr,
                 np.ndarray[np.float64_t, ndim=1] out_arr,
                 np.ndarray[np.int64_t, ndim=1] bidx):
    cdef:
        size_t j, idx
        size_t n_data = 3
        np.float64_t n_stars = len(bidx)
    if n_stars <= 0:
        return 0
    out_arr[26] += n_stars
    for j in range(in_arr.shape[0]):
        idx = j * n_data
        out_arr[idx] += in_arr[j][bidx].min()
        out_arr[idx+1] += in_arr[j][bidx].mean()
        out_arr[idx+2] += in_arr[j][bidx].max()
    return n_stars
