@cython.wraparound(False)
def rotate_positions(
        np.ndarray[np.float64_t, ndim=2] xyz_arr,
        np.ndarray[np.float64_t, ndim=2] out_arr):
    cdef:
        int _theta1 = rand() % 4000
        int _theta2 = rand() % 2000
        int _theta3 = rand() % 6000
        double theta1 = (_theta1 / 1e3) * M_PI  # 0.01745
        double theta2 = (_theta2 / 1e3) * M_PI  # 0.01745
        double theta3 = (_theta3 / 1e3) * M_PI  # 0.01745
        size_t i, j, k
        double rot_matrix[3][3]
        int I = len(rot_matrix)
        int J = xyz_arr.shape[1]
        int K = xyz_arr.shape[0]
        np.float64_t s, a, aa, bb, cc, dd, bc, ad, ac, ab, bd, cd, b, c, d
    for a, b, c, d in [
            (cos(theta1), 0, 0, -sin(theta1)),
            (cos(theta2), 0, -sin(theta2), 0),
            (cos(theta3), -sin(theta3), 0, 0)]:
        aa, bb, cc, dd, bc = a * a, b * b, c * c, d * d, b * c
        ad, ac, ab, bd, cd = a * d, a * c, a * b, b * d, c * d
        rot_matrix[0][:] = [aa + bb - cc - dd,
                            2.0 * (bc + ad), 2.0 * (bd - ac)]
        rot_matrix[1][:] = [2.0 * (bc - ad), aa +
                            cc - bb - dd, 2.0 * (cd + ab)]
        rot_matrix[2][:] = [2.0 * (bd + ac), 2.0 *
                            (cd - ab), aa + dd - bb - cc]
        for i in range(I):
            for j in xrange(J):
                s = 0
                for k in range(K):
                    s += (xyz_arr[k, j] * rot_matrix[i][k])
                out_arr[i, j] = s
                
@cython.boundscheck(False)
@cython.wraparound(False)
def rotation_matrix(
        np.ndarray[np.float64_t, ndim=1, mode='c'] ax,
        np.float64_t th):
    '''
    Description : used to rotate x,y,z arrays. usually called by <rotate>.
    Parameters
    ----------
    ax : list
            an axis to rotate about.
    th : float
            the amount to rotate by in radians
    Returns
    -------
    list,list,list
            returns a rotation matrix
    '''
    print('calculating rotation matrix')
    cdef np.float64_t a, b, c, d, aa, bb, cc, dd
    a = np.cos(th / 2.0)
    b, c, d = -(ax / (np.dot(ax, ax))**2) * np.sin(th / 2.0)
    aa, bb, cc, dd, bc = a * a, b * b, c * c, d * d, b * c
    ad, ac, ab, bd, cd = a * d, a * c, a * b, b * d, c * d
    return (
        [aa + bb - cc - dd, 2.0 * (bc + ad), 2.0 * (bd - ac)],
        [2.0 * (bc - ad), aa + cc - bb - dd, 2.0 * (cd + ab)],
        [2.0 * (bd + ac), 2.0 * (cd - ab), aa + dd - bb - cc])


@cython.boundscheck(False)
@cython.wraparound(False)
def rotate(
        np.ndarray[np.float64_t, ndim=2, mode='c'] xyz,
        np.ndarray[np.float64_t, ndim=1, mode='c'] axis,
        np.float64_t theta):
    '''
    Consolidates data and rotation options into a single function.
    Parameters
    ----------
    xyz : list
            a list of numpy arrays [px,py,pz]
    axis : list
            an axis to rotate about [0,1,0]
    theta : float
            amount to rotate by in radians
    Returns
    -------
    array
            returns the rotated positions in the same form as they were input.
    '''
    print('rotating x, y and z position arrays')
    return np.asarray(
        np.dot(
            rotation_matrix(axis, theta),
            [xyz[0], xyz[1], xyz[2]]),
        dtype=np.float64)


@cython.boundscheck(False)
@cython.wraparound(False)
def trippel_rotate(
        np.ndarray[np.float64_t, ndim=2, mode='c'] xyz):
    '''
    '''
    print('starting trippel rotatation')
    cdef np.float64_t theta_1 = np.float64(
        np.divide(
            (np.random.randint(360) * np.pi),
            180.0))
    cdef np.float64_t theta_2 = np.float64(
        np.divide(
            (np.random.randint(360) * np.pi),
            180.0))
    cdef np.float64_t theta_3 = np.float64(
        np.divide(
            (np.random.randint(360) * np.pi),
            180.0))
    return rotate(
        rotate(
            rotate(xyz,
                   np.asarray([0, 0, 1], dtype=np.float64), theta_1),
            np.asarray([0, 1, 0], dtype=np.float64), theta_2),
        np.asarray([1, 0, 0], dtype=np.float64), theta_3)
