@cython.boundscheck(False)
@cython.wraparound(False)
def bin(
        np.ndarray[np.int32_t, ndim=1] px,
        np.ndarray[np.int32_t, ndim=1] py,
        np.ndarray[np.float64_t, ndim=1] ab_mags,
        np.ndarray[np.float64_t, ndim=1] ap_mags,
        np.ndarray[np.float64_t, ndim=1] r_proj,
        np.ndarray[np.float64_t, ndim=1] lims,
        np.ndarray[np.int32_t, ndim=1] satid):
    """[summary]

    [description]

    Parameters
    ----------
    np.ndarray[np.int32_t : {[type]}
        [description]
    np.ndarray[np.int32_t : {[type]}
        [description]
    np.ndarray[np.float64_t : {[type]}
        [description]
    np.ndarray[np.float64_t : {[type]}
        [description]
    np.ndarray[np.float64_t : {[type]}
        [description]
    np.ndarray[np.float64_t : {[type]}
        [description]
    np.ndarray[np.int32_t : {[type]}
        [description]
    ndim : {[type]}, optional
        [description] (the default is 1] px, which [default_description])
    ndim : {[type]}, optional
        [description] (the default is 1] py, which [default_description])
    ndim : {[type]}, optional
        [description] (the default is 1] ab_mags, which [default_description])
    ndim : {[type]}, optional
        [description] (the default is 1] ap_mags, which [default_description])
    ndim : {[type]}, optional
        [description] (the default is 1] r_proj, which [default_description])
    ndim : {[type]}, optional
        [description] (the default is 1] lims, which [default_description])
    ndim : {[type]}, optional
        [description] (the default is 1] satid, which [default_description])

    Returns
    -------
    [type]
        [description]
    """
    line = '-' * 85
    print(line)
    print('\n', '[ bin ]', '\n')
    cdef:
        # Itierable
        size_t i, ii
        # np.float64_t _one = 1.0
        # Magnitude limits (mlim_min, mlim_med, mlim_max).
        np.float64_t mlim_min = lims[0]
        np.float64_t mlim_med = lims[1]
        np.float64_t mlim_max = lims[2]
        # Apparent magnitude (apparent_mag).
        np.float64_t apparent_mag
        # The grid array (grid)
        np.ndarray[np.float64_t, ndim = 3, mode = 'c'] grid = np.zeros(
            (
                Config.getint('grid_options', 'size'),
                Config.getint('grid_options', 'size'),
                Config.getint('grid_options', 'n_slices')),
            dtype=np.float64)
        # Satprop arrays. TODO package data file
        np.ndarray[np.float32_t, ndim = 1] tsat = SATPROP['tsat']
        np.ndarray[np.int32_t, ndim = 1] bsat = SATPROP['bsat']
        np.ndarray[np.int32_t, ndim = 1] nsat = SATPROP['nsat']
        np.ndarray[np.int32_t, ndim = 1] nsatc = SATPROP['nsatc']
        # Index arrays
        np.ndarray[np.int64_t, ndim = 1] idx_1, idx_2
        # np.ndarray[np.int64_t, ndim=1] idx_1 = np.nonzero(
        #     bsat[satid + nsatc[2]] == 0)[0]
        # np.ndarray[np.int64_t, ndim=1] idx_2 = np.nonzero(
        #     bsat[satid + nsatc[2]] == 0)[1]
        # Number of stars (n_stars) to bin.
        np.int_t n_stars = len(px)
        # Number of threads
        np.int_t n_threads = int(px.shape[0]/(5 * 1e5))
        # X and Y grid boundaries (boundary_x, boundary_y)
        np.int_t boundary_y = grid.shape[0]
        np.int_t boundary_x = grid.shape[1]
        # Satellite number of star.
        np.int32_t sat_number
        # Age of sat (Gyr).
        np.float32_t sat_age
        # Bound or unbound status.
        np.int32_t sat_bound
        # Number of (Bound) or (unbound) stars.
        np.int32_t bound = 0
        np.int32_t unbound = 0
        # Number of stars outside (missed) of the superimposed grid.
        np.int32_t missed = 0
        np.int32_t bad = 0

    # Set (threads)
    if n_threads >= NUM_PROCESSORS:
        n_threads = NUM_PROCESSORS - 2
        if n_threads <= 0:
            n_threads = 1

    # Initial print statement.
    print('sample px and py before starting bin loop:')
    print('  -> ', np.random.choice(px, 10))
    print('  -> ', np.random.choice(py, 10))
    print('\n')
    print('threads : ', n_threads)
    print('mlim_min: ', mlim_min)
    print('mlim_med: ', mlim_med)
    print('mlim_max: ', mlim_max)

    with nogil, parallel(num_threads=n_threads):
        for i in prange(n_stars, schedule='dynamic'):

            # Check to see if star is in FOV of grid.
            if (
                    px[i] >= boundary_x or
                    px[i] <= 0 or
                    py[i] >= boundary_y or
                    py[i] <= 0):
                missed += 1

            # If star is in the grid.
            else:
                sat_number = satid[i]
                sat_age = tsat[sat_number]
                sat_bound = bsat[sat_number]
                apparent_mag = ap_mags[i]

                # If the star is unbound [0].
                if not sat_bound:
                    unbound += 1
                    # Bin stars.
                    grid[px[i], py[i], 0] += 1.0
                    # Magnitudes.
                    grid[px[i], py[i], 1] += ab_mags[i]
                    # grid[px[i], py[i], 2] += apparent_mag
                    # if apparent_mag < mlim_min:
                    #    grid[px[i], py[i], 3] += 1.0
                    # if apparent_mag < mlim_med:
                    #    grid[px[i], py[i], 4] += 1.0
                    # if apparent_mag < mlim_max:
                    #    grid[px[i], py[i], 5] += 1.0
                    # Accretion time (Gyr) of satellite.
                    grid[px[i], py[i], 2] += sat_age
                    # Satid.
                    grid[px[i], py[i], 3] += sat_number

                # If the star is bound [1].
                else:
                    bound += 1
                    # Bin stars.
                    # grid[px[i], py[i], 8] += 1.0
                    # Magnitudes.
                    # grid[px[i], py[i], 9] += ab_mags[i]
                    # grid[px[i], py[i], 10] += apparent_mag
                    # Accretion time (Gyr).
                    # grid[px[i], py[i], 11] += sat_age
                    # Satid.
                    # grid[px[i], py[i], 12] += sat_number
                    if not sat_bound:
                        bad += 1

                # grid[px[i], py[i], 13] += 1
                grid[px[i], py[i], 4] += r_proj[i]

    # Slices that need to be divided by the number of stars in each bin.
    idx_1, idx_2 = np.nonzero(grid[:, :, 0] > 0.)
    grid[idx_1, idx_2, 1] /= grid[idx_1, idx_2, 0]
    grid[idx_1, idx_2, 2] /= grid[idx_1, idx_2, 0]
    grid[idx_1, idx_2, 3] /= grid[idx_1, idx_2, 0]
    grid[idx_1, idx_2, 4] /= grid[idx_1, idx_2, 0]
    # grid[idx_1, idx_2, 14] /= grid[idx_1, idx_2, 0]
    # idx_1, idx_2 = np.nonzero(grid[:, :, 8] > 0.)
    # grid[idx_1, idx_2, 9] /= grid[idx_1, idx_2, 8]
    # grid[idx_1, idx_2, 10] /= grid[idx_1, idx_2, 8]
    # grid[idx_1, idx_2, 11] /= grid[idx_1, idx_2, 8]
    # grid[idx_1, idx_2, 12] /= grid[idx_1, idx_2, 8]
    # idx_1, idx_2 = np.nonzero(grid[:, :, 13] > 0.)
    # grid[idx_1, idx_2, 14] /= grid[idx_1, idx_2, 13]

    print('\n-------------------------------')
    print('    bound stars: ', bound)
    print('  unbound stars: ', unbound)
    print('percent unbound: ', round(
        1e2 * (float(unbound)/float(len(px))), 2), '%')
    print('   binned stars: ', (n_stars - missed))
    print('    total stars: ', n_stars)
    print('   missed stars: ', missed)
    print('-------------------------------\n')
    return grid