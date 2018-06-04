
@cython.boundscheck(False)
@cython.wraparound(False)
def integerize(
        np.ndarray[np.float64_t, ndim=1] x,
        np.ndarray[np.float64_t, ndim=1] y):
    '''
    Convert np.array of float64 to int32 for binning into a grid.
    TODO Extended description of function.
    Parameters
    ----------
    x : np.array
            np.array of x coordinates.
    y : np.array
            np.array of x coordinates.
    center : float
            float representing the midpoint of the grid.
    scale : float
            float representing the scale factor for matching Kpc to the grid.
    Returns
    -------
    np.array,np.array
            Returns two equivalent length arrays in int32 form.
    '''
    center = Config.getint('grid_options', 'size') / 2
    scale = Config.getint('grid_options', 'size') / (x.max() + np.abs(x.min()))
    line = '-' * 85
    # print('converting px and py arrays to integers')
    print(line)
    print('[ integerize ]', '\n')
    print('[before ] px min, mean, max : ',
          x.min(), ',', x.mean(), ',', x.max())
    cdef np.ndarray[np.float64_t, ndim = 1, mode = 'c'] x1 = x.round(3)
    x1 += center
    x1 *= scale
    x2 = x1.astype(np.int32)
    print('[ after ] px min, mean, max : ',
          x2.min(), ',', x2.mean(), ',', x2.max())
    print('[before ] py min, mean, max : ',
          y.min(), ',', y.mean(), ',', y.max())
    cdef np.ndarray[np.float64_t, ndim = 1, mode = 'c'] y1 = y.round(3)
    y1 += center
    y1 *= scale
    y2 = y1.astype(np.int32)
    print('[ after ] py min, mean, max : ',
          y2.min(), ',', y2.mean(), ',', y2.max())
    print(line)
    return x2, y2

