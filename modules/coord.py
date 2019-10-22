import numpy as np

# the following functions has been copied from an astrophysics python library:
# https://git.rwth-aachen.de/astro/astrotools


def ang2vec(phi, theta):
    """
    Get vector from spherical angles (phi, theta)

    :param phi: range (pi, -pi), 0 points in x-direction, pi/2 in y-direction
    :param theta: range (pi/2, -pi/2), pi/2 points in z-direction
    :return: vector of shape (3, n)
    """
    assert np.ndim(phi) == np.ndim(theta), "Inputs phi and theta in 'coord.ang2vec()' must have same shape!"
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)
    return np.array([x, y, z])


def rand_phi(n=1):
    """
    Random uniform phi (-pi, pi).

    :param n: number of samples that are drawn
    :return: random numbers in range (-pi, pi)
    """
    return (np.random.random(n) * 2 - 1) * np.pi


def rand_theta(n=1, theta_min=-np.pi/2, theta_max=np.pi/2):
    """
    Random theta (pi/2, -pi/2) from uniform cos(theta) distribution.

    :param n: number of samples that are drawn
    :return: random theta in range (-pi/2, pi/2) from cos(theta) distribution
    """
    assert theta_max > theta_min
    u = np.sin(theta_min) + (np.sin(theta_max) - np.sin(theta_min)) * np.random.random(n)
    return np.pi / 2 - np.arccos(u)


def rand_theta_plane(n=1):
    """
    Random theta (pi/2, 0) on a planar surface from uniform cos(theta)^2 distribution.

    :param n: number of samples that are drawn
    :return: random theta on plane in range (pi/2, 0)
    """
    return np.pi / 2 - np.arccos(np.sqrt(np.random.random(n)))


def rand_vec(n=1):
    """
    Random spherical unit vectors.

    :param n: number of vectors that are drawn
    :return: random unit vectors of shape (3, n)
    """
    return ang2vec(rand_phi(n), rand_theta(n))


def normed(v):
    """
    Return the normalized (lists of) vectors.

    :param v: vector(s) of shape (3, n)
    :return: corresponding normalized vectors of shape (3, n)
    """
    return np.asarray(v) / np.linalg.norm(v, axis=0)


def angle(v1, v2, each2each=False):
    """
    Angular distance in radians for each pair from two (lists of) vectors.
    Use each2each=True to calculate every combination.

    :param v1: vector(s) of shape (3, n)
    :param v2: vector(s) of shape (3, n)
    :param each2each: if true calculates every combination of the two lists v1, v2
    :return: angular distance in radians
    """
    a = normed(v1)
    b = normed(v2)
    if each2each:
        d = np.outer(a[0], b[0]) + np.outer(a[1], b[1]) + np.outer(a[2], b[2])
    else:
        if len(a.shape) == 1:
            a = a.reshape(3, 1)
        if len(b.shape) == 1:
            b = b.reshape(3, 1)
        d = np.sum(a * b, axis=0)
    return np.arccos(np.clip(d, -1., 1.))


def rotate_zaxis_to_x(v, x0):
    """
    Transfers the relative orientation between vectors v and the z-axis towards
    v and the reference vectors x0. Mathematically, the scalar products z_axis*v
    before the rotation and x0*v after rotation are the same (see e.g. unit test).

    :param v: vectors that should be rotated, shape: (3) or (3, n)
    :param x0: reference vectors of same shape like v
    """
    # defines rotation axis by the cross-product with z-axis
    u = np.array([x0[1], -x0[0], np.zeros_like(x0[0])])
    u[2, np.sum(u**2, axis=0) < 1e-10] = 1      # fix z-axis itself
    angles = angle(x0, (0, 0, 1))
    return rotate(v, normed(u), angles)


def rotation_matrix(rotation_axis, rotation_angle):
    """
    Rotation matrix for given rotation axis and angle.
    See http://en.wikipedia.org/wiki/Euler-Rodrigues_parameters

    :param rotation_axis: rotation axis, either np.array([x, y, z]) or ndarray with shape (3, n)
    :param rotation_angle: rotation angle in radians, either float or array size n
    :return: rotation matrix R, either shape (3, 3) or (3, 3, n)

    Example:
    R = rotation_matrix( np.array([4,4,1]), 1.2 )
    v1 = np.array([3,5,0])
    v2 = np.dot(R, v1)
    """
    assert np.ndim(rotation_axis) == np.ndim(rotation_angle) + 1, "Shape of rotation axis and angle do not not match"
    rotation_axis = normed(rotation_axis)
    a = np.cos(rotation_angle / 2.)
    b, c, d = - rotation_axis * np.sin(rotation_angle / 2.)
    r = np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                  [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                  [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])

    return np.squeeze(r)


def rotate(v, rotation_axis, rotation_angle):
    """
    Perform rotation for given rotation axis and angle(s).

    :param v: vector that is supposed to be rotated, either np.array([x, y, z]) or ndarray with shape (3, n)
    :param rotation_axis: rotation axis, either np.array([x, y, z]) or ndarray with shape (3, n)
    :param rotation_angle: rotation angle in radians, either float or array size n
    :return: rotated vector, same shape as input v
    """
    shape = v.shape
    v, rotation_axis, rotation_angle = np.squeeze(v), np.squeeze(rotation_axis), np.squeeze(rotation_angle)
    if np.ndim(rotation_angle) == np.ndim(rotation_axis):
        rotation_axis = rotation_axis[:, np.newaxis]
    r = rotation_matrix(rotation_axis, rotation_angle)
    if np.ndim(rotation_axis) == 1:
        return np.dot(r, v).reshape(shape)

    if np.ndim(v) == np.ndim(rotation_axis) - 1:
        v = v[:, np.newaxis]
    rotated_vector = np.sum(r * v[np.newaxis], axis=1)
    if rotated_vector.size == v.size:
        rotated_vector = rotated_vector.reshape(shape)
    return rotated_vector


def rand_vec_on_surface(x0, n=1):
    """
    Given unit normal vectors x0 orthogonal on a surface, samples one isotropic
    direction for each given vector x0 from a cos(theta)*sin(theta) distribution

    :param x0: ortogonal unit vector on the surface, shape: (3, N)
    :return: isotropic directions for the respective normal vectors x0
    """
    if np.ndim(np.squeeze(x0)) > 1:
        n = x0.shape[1]
    theta = rand_theta_plane(n)
    v = ang2vec(rand_phi(n), theta)   # produce random vecs on plane through z-axis
    return rotate_zaxis_to_x(v, x0), theta      # rotation to respective surface vector x0
