# -*- coding: utf-8 -*-

import itertools
import multiprocessing

from math import cos, radians
from dipy.data import get_sphere
import numpy as np
from scipy.integrate import nquad

from dipy.direction import peak_directions
from dipy.reconst.shm import sh_to_sf_matrix
from scilpy.reconst.utils import get_sh_order_and_fullness


# Constants
NB_PARAMS = 9


class BinghamDistribution(object):
    """
    Scaled bingham distribution.
        B(u) = f0 * exp(-k1 * (mu1 * u)**2 - k2 * (mu2 * u)**2)

    Params
    ------
    f0: float
        Scaling parameter.
    mu1, mu2: ndarray (3,)
        Axes.
    k1, k2: float
        Concentration parameters.
    """
    def __init__(self, f0, mu1, mu2, k1, k2):
        self.f0 = f0  # scaling factor
        self.mu1 = mu1.reshape((1, 3))  # vec3
        self.mu2 = mu2.reshape((1, 3))  # vec3
        self.k1 = k1  # scalar
        self.k2 = k2  # scalar

    def evaluate(self, vertices):
        bu = np.exp(- self.k1 * self.mu1.dot(vertices.T)**2.
                    - self.k2 * self.mu2.dot(vertices.T)**2.)
        bu *= self.f0

        return bu.reshape((-1))  # (1, N)

    def peak_direction(self):
        return np.cross(self.mu1, self.mu2)

    def get_flatten(self):
        ret = np.zeros((9))
        ret[0] = self.f0
        ret[1:4] = self.mu1.reshape((-1))
        ret[4:7] = self.mu2.reshape((-1))
        ret[7] = self.k1
        ret[8] = self.k2
        return ret


def bingham_from_array(arr):
    """
    Instantiate and return a bingham distribution
    with parameters contained in `arr`.

    Params
    ======
    arr: ndarray (9,)
        Parameters for the bingham distribution, with:
        arr[0]   => f0
        arr[1:4] => mu1
        arr[4:7] => mu2
        arr[7]   => k1
        arr[8]   => k2

    Returns
    =======
    out: BinghamDistribution
        Bingham distribution initialized with the parameters
        from `arr`.
    """
    return BinghamDistribution(arr[0], arr[1:4], arr[4:7], arr[7], arr[8])


def bingham_fit_sh_parallel(data, max_lobes, abs_th=0.,
                            rel_th=0., min_sep_angle=25.):
    order, full_basis = get_sh_order_and_fullness(data.shape[-1])
    shape = data.shape

    sphere = get_sphere('symmetric724').subdivide(2)
    B_mat = sh_to_sf_matrix(sphere, order,
                            full_basis=full_basis,
                            return_inv=False)

    nbr_processes = multiprocessing.cpu_count()
    data = data.reshape((-1, data.shape[-1]))
    data = np.array_split(data, nbr_processes)
    pool = multiprocessing.Pool(nbr_processes)
    out = pool.map(_bingham_fit_sh, zip(data, itertools.repeat(B_mat),
                                        itertools.repeat(sphere),
                                        itertools.repeat(abs_th),
                                        itertools.repeat(min_sep_angle),
                                        itertools.repeat(rel_th),
                                        itertools.repeat(max_lobes)))
    pool.close()
    pool.join()

    out = np.concatenate(out, axis=0)
    out = out.reshape(np.append(shape[:3], max_lobes*NB_PARAMS))
    return out


def _bingham_fit_sh(args):
    sh_chunk = args[0]
    B_mat = args[1]
    sphere = args[2]
    abs_th = args[3]
    min_sep_angle = args[4]
    rel_th = args[5]
    max_lobes = args[6]

    out = np.zeros((len(sh_chunk), max_lobes*NB_PARAMS))
    for i, sh in enumerate(sh_chunk):
        odf = sh.dot(B_mat)
        odf[odf < abs_th] = 0.
        if (odf > 0.).any():
            lobes = \
                _bingham_fit_multi_peaks(odf, sphere,
                                         min_sep_angle=min_sep_angle,
                                         rel_th=rel_th)
            for ll in range(min(len(lobes), max_lobes)):
                lobe = lobes[ll]
                out[i, ll*NB_PARAMS:(ll+1)*NB_PARAMS] = lobe.get_flatten()
    return out


def _bingham_fit_peak(sf, peak, sphere, max_angle):
    """
    Fit bingham function on the lobe aligned with peak.
    """
    # abs for twice the number of pts to fit
    dot_prod = np.abs(sphere.vertices.dot(peak))
    min_dot = cos(radians(max_angle))

    p = sphere.vertices[dot_prod > min_dot]
    v = sf[dot_prod > min_dot].reshape((-1, 1))  # (N, 1)

    # test that the peak contains at least 3 non-zero directions
    if np.count_nonzero(v) < 3:
        return BinghamDistribution(0, np.zeros(3), np.zeros(3), 0, 0)

    x, y, z = (p[:, 0:1], p[:, 1:2], p[:, 2:])

    # create an orientation matrix to approximate mu0, mu1 and mu2
    T = np.zeros((3, 3))
    T[0, 0] = np.sum(x**2 * v)
    T[1, 1] = np.sum(y**2 * v)
    T[2, 2] = np.sum(z**2 * v)
    T[1, 0] = np.sum(x * y * v)
    T[2, 0] = np.sum(x * z * v)
    T[2, 1] = np.sum(y * z * v)
    T[0, 1] = T[1, 0]
    T[0, 2] = T[2, 0]
    T[1, 2] = T[2, 1]
    T = T / np.sum(v)

    eval, evec = np.linalg.eig(T)

    ordered = np.argsort(eval)
    mu1 = evec[:, ordered[1]].reshape((3, 1))
    mu2 = evec[:, ordered[0]].reshape((3, 1))
    f0 = v.max()

    if np.iscomplex(mu1).any() or np.iscomplex(mu2).any():
        print('uh oh... \n', eval)
        1/0

    A = np.zeros((len(v), 2), dtype=float)  # (N, 2)
    A[:, 0:1] = p.dot(mu1)**2
    A[:, 1:] = p.dot(mu2)**2

    # Test that AT.A is invertible for pseudo-inverse
    ATA = A.T.dot(A)
    if np.linalg.matrix_rank(ATA) != ATA.shape[0]:
        return BinghamDistribution(0, np.zeros(3), np.zeros(3), 0, 0)

    B = np.zeros_like(v)
    B[v > 0] = np.log(v[v > 0] / f0)  # (N, 1)
    k = np.abs(np.linalg.inv(ATA).dot(A.T).dot(B))
    k1 = k[0]
    k2 = k[1]
    if k[0] > k[1]:
        k1 = k[1]
        k2 = k[0]
        mu2 = evec[:, ordered[1]].reshape((3, 1))
        mu1 = evec[:, ordered[0]].reshape((3, 1))

    return BinghamDistribution(f0, mu1, mu2, k1, k2)


def _bingham_fit_multi_peaks(odf, sphere, max_angle=15.,
                             min_sep_angle=25., rel_th=0.1):
    """
    Peak extraction followed by Bingham fit for each peak.
    """
    peaks, _, _ = peak_directions(odf, sphere,
                                  relative_peak_threshold=rel_th,
                                  min_separation_angle=min_sep_angle)

    lobes = []
    for peak in peaks:
        peak_fit = _bingham_fit_peak(odf, peak, sphere, max_angle)
        lobes.append(peak_fit)

    return lobes


def compute_fiber_density_parallel(data, m=50):
    """
    Fiber density (FD) is given by integrating
    the bingham function over the sphere. Its unit is
    in 1/mm**3.
    """
    shape = data.shape

    phi = np.linspace(0, 2 * np.pi, 2 * m, endpoint=False)  # [0, 2pi[
    theta = np.linspace(0, np.pi, m)  # [0, pi]
    coords = np.array([[p, t] for p in phi for t in theta]).T
    dphi = phi[1] - phi[0]
    dtheta = theta[1] - theta[0]

    nbr_processes = multiprocessing.cpu_count()
    data = data.reshape((-1, shape[-1]))
    data = np.array_split(data, nbr_processes)
    pool = multiprocessing.Pool(nbr_processes)
    res = pool.map(_compute_fiber_density,
                   zip(data,
                       itertools.repeat(coords),
                       itertools.repeat(dphi),
                       itertools.repeat(dtheta)))
    pool.close()
    pool.join()

    nbr_lobes = shape[-1] // NB_PARAMS
    res = np.concatenate(res, axis=0)
    res = np.reshape(np.array(res), np.append(shape[:3], nbr_lobes))
    return res


def _compute_fiber_density(args):
    binghams_chunk = args[0]
    coords = args[1]
    dphi = args[2]
    dtheta = args[3]
    theta = coords[1]
    u = np.array([np.cos(coords[0]) * np.sin(coords[1]),
                  np.sin(coords[0]) * np.sin(coords[1]),
                  np.cos(coords[1])]).T

    nbr_lobes = binghams_chunk.shape[1] // NB_PARAMS
    out = np.zeros((len(binghams_chunk), nbr_lobes))
    for i, binghams in enumerate(binghams_chunk):
        for lobe_i in range(nbr_lobes):
            lobe = bingham_from_array(binghams[lobe_i*NB_PARAMS:
                                               (lobe_i+1)*NB_PARAMS])
            if lobe.f0 > 0:
                fd = np.sum(lobe.evaluate(u) * np.sin(theta) * dtheta * dphi)
                out[i, lobe_i] = fd
    return out


def compute_fiber_spread(binghams, fd):
    """
    Fiber spread (FS) characterizes the spread of the lobe.
    The higher FS is, the wider the lobe. The unit of the
    FS is radians.
    """
    f0 = binghams[..., ::NB_PARAMS]
    fs = np.zeros_like(fd)
    fs[f0 > 0] = fd[f0 > 0] / f0[f0 > 0]

    return fs


def compute_structural_complexity(fd):
    """
    Structural complexity (CX) increases when the fiber structure
    becomes more complex inside a voxel and fewer of the fibers in
    the voxel are contained in the largest bundle alone. This value
    is normalized between 0 and 1.
    """
    cx = np.zeros(fd.shape[:3])
    n = fd.shape[-1]
    mask = np.max(fd, axis=-1) > 0
    cx[mask] = n / (n - 1) * \
        (1 - np.max(fd, axis=-1)[mask]/np.sum(fd, axis=-1)[mask])
    return cx
