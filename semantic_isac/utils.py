# filename "utils.py"
# Global methods: contains general methods used everywhere
import os
import multiprocessing
import GPUtil
from datetime import date
import colorsys
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.spatial import ConvexHull, Voronoi
from scipy.spatial.transform import Rotation
from scipy.constants import k, speed_of_light
from scipy.stats import rice, norm
from scipy.special import erf
from tqdm import tqdm
import sys
import numpy as np
from sionna import rt
import mitsuba as mi
import pandas as pd
import trimesh
import shapely as sha
from shapely.geometry import Polygon, Point, MultiPolygon
import shapely.ops as shops
import dataclasses
from rtree import index
import inspect

# TODO: Maybe split this in multiple files?

def boundary_normal(points: np.ndarray, polygon: Polygon, eps: float = 1e-6) -> np.ndarray:
    """Outward unit normal of the polygon boundary at the point nearest each query point --
    fully vectorized (all points against all boundary segments at once).

    :param points: np.ndarray, shape (N, 3), (x, y, z) query points -- z is ignored.
    :param polygon: shapely.geometry.Polygon.
    :param eps: small offset used to disambiguate the outward side.

    :returns: np.ndarray, shape (N, 3), unit (x, y, 0.) outward normal vectors.
    """
    rings = [np.array(polygon.exterior.coords)] + [np.array(r.coords) for r in polygon.interiors]
    # Collect all segments while removing the third dimension
    seg_start = np.vstack([r[:-1, :2] for r in rings])          # (M, 3)
    seg_end = np.vstack([r[1:, :2] for r in rings])             # (M, 3)
    seg_vec = seg_end - seg_start                            # (M, 3)
    seg_len2 = np.sum(seg_vec ** 2, axis=1)
    seg_len2_safe = np.where(seg_len2 == 0, 1.0, seg_len2)


    xy = points[:, :2]                                     # (N, 2)

    # Project every point onto every segment at once: (N, R, 2)
    w = xy[:, None, :] - seg_start[None, :, :]
    t = np.clip(np.sum(w * seg_vec[None, :, :], axis=2) / seg_len2_safe[None, :], 0.0, 1.0)  # (U, N)
    proj = seg_start[None, :, :] + t[:, :, None] * seg_vec[None, :, :]                         # (U, N, 2)
    dist2 = np.sum((proj - xy[:, None, :]) ** 2, axis=2)                                       # (U, N)

    seg_idx = np.argmin(dist2, axis=1)                       # (N,)
    nxy = proj[np.arange(len(xy)), seg_idx]                  # (N, 2) -- nearest boundary point

    tangent = seg_vec[seg_idx]                                # (N, 2)
    t_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    t_norm[t_norm == 0] = 1.0
    tangent = tangent / t_norm
    cand = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)  # (N, 2), one of the two perpendiculars

    # Vectorized outward test: nudge along `cand`, check which side is outside the polygon
    test_xy = nxy + eps * cand
    test_points = sha.points(test_xy[:, 0], test_xy[:, 1])
    inside = sha.contains(polygon, test_points)            # (N,) bool, broadcasts one polygon vs many points

    cand = np.hstack([cand, np.zeros((cand.shape[0], 1))])      # Coming back to 3 dimensions (0 for z)
    return np.where(inside[:, None], -cand, cand)

def ue_ap_distances(ue_position: np.ndarray, ap_position: np.ndarray) -> np.ndarray:
    """Compute the distances between UEs and APs. It assumes UEs position is an array of (..., U, d) while AP position
    is an array of (L, d).

    :param ue_position: np.ndarray, (..., U, d) position of the UE.
    :param ap_position: np.ndarray, (L, d) position of the AP.
    :returns: distance between UEs and APs, shape (..., L, U)
    """
    return np.linalg.norm(ap_position[np.newaxis, np.newaxis] - ue_position[..., np.newaxis, :], axis=-1)


def distance_to_boundary(points: np.ndarray, polygon: Polygon) -> np.ndarray:
    """Compute the distance between a series of points and a polygon.
    :param points: np.ndarray, (N, 3) coordinates of points. (z-dimension is neglected)
    :param polygon: Polygon, a shapely polygon.

    :returns: np.ndarray, (N,) distance between each point in the polygon.
    """
    boundary = polygon.boundary
    pts = sha.points(points[:, 0], points[:, 1])   # shape (N,) array of Point geometries
    return sha.distance(pts, boundary)              # vectorized, shape (N,)

def qfun(input):
    """Compute the Gaussian Q-function"""
    return 0.5 - 0.5 * erf(input / np.sqrt(2))

def filter_kwargs(method, **kwargs):
    """Filter kwargs passed to method based on the supported one.
     when collecting all parameters into a dictionary.

     :param method: Method to call.
     :returns: filtered dictionary.
     """
    # Get the signature of the method
    sig = inspect.signature(method)

    # Get the parameter names that the method accepts
    supported_params = sig.parameters.keys()

    # Filter parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_params}
    return filtered_kwargs


# Random variables
def sample_multivariate_gaussian(cov: np.ndarray, mean: np.ndarray | None = None,
                                 rng: np.random.Generator = np.random.default_rng(42)) -> np.ndarray:
    r"""Sample collection of multivariate Gaussian distribution from the covariance matrices in a vectorized version
    It uses Cholesky and explicit multiplication to avoid loops, but it needs a positive definitive covariances.

    :param cov: np.ndarray, (A, d, d) covariance matrix where A is the number of covariances matrix and d is the
        dimensionality of the random variable.
    :param mean: np.ndarray, (A, d) mean of the random variable, with similar dimensions.
    :param rng: np.random.Generator, random number generator.
    :returns: np.ndarray, (A, d) collection of multivariate Gaussian realizations.
    """
    if mean is None:
        mean = np.zeros((cov.shape[0], cov.shape[1]))
    # Precompute Cholesky decompositions: shape (A, d, d)
    trian = np.linalg.cholesky(cov)  # Upper triangular Cholesky factors
    # Generate standard normal samples: shape (U, 1, 3)
    standard_samples = rng.standard_normal((cov.shape[0], 1, cov.shape[-1]))
    # Apply Cholesky to obtain X = μ + L @ Z, where Z ~ N(0, I)
    return mean + np.squeeze(np.einsum('nij,nmj->nmi', trian, standard_samples), axis=1)


def square_positioning(side: float, num_ant: int, antenna_spacing: float) -> tuple:
    """Returning the position of the APs and their antennas for a symmetric room with APs on each side (centered)"""
    # Getting antenna position relative to the center
    ant_pos_line = antenna_spacing * (np.arange(num_ant) - (num_ant - 1) / 2)
    # Set orientation (clockwise)
    ap_orientation = np.array([[0., 1.], [1., 0.], [0., -1.], [-1., 0.]])
    # Set center position
    ap_cen_pos = side * np.array([[-1., 0.], [0., 1.], [1., 0.], [0., -1]])
    # Compute relative position given the main direction of the array.
    ap_ant_pos = ap_orientation[:, :, np.newaxis] @ ant_pos_line[np.newaxis]
    # Return tuple
    return ap_cen_pos, ap_ant_pos


def eval_fspl(ue_pos: np.ndarray, ap_pos: np.ndarray, wavelength: float, gain: int = 2.15) -> np.ndarray:
    """Evaluate the FSPL (linear scale)"""
    dist_ue_ap = eval_ue_ap_dist(ue_pos, ap_pos)
    return 10 ** (gain / 10) * (wavelength / (4 * np.pi * dist_ue_ap)) ** 2

def eval_ue_ap_dist(ue_pos: np.ndarray, ap_pos: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.tile(ue_pos[:, np.newaxis], (1, ap_pos.shape[0], 1)) - ap_pos, axis=2)
# Standard array generation

def effective_se(beta, snr: float, num_ant: int, num_ap: int):
    effective_snr = snr * num_ant * np.sum(beta, axis = 1) ** 2 / (2 * np.sum(beta, axis = 1) + num_ap / snr)
    return np.log2(1 + effective_snr)

def move_ues(ue_pos,
             v_max: float,
             time_frame: float,
             side: float,
             rng: np.random.RandomState = np.random.default_rng(42)) -> np.ndarray:
    num_ue = ue_pos.shape[0]
    # Generate user angle of trajectory
    angle = rng.uniform(0, 2 * np.pi, (num_ue, 1))
    # Compute the x and y components of the unit vector
    direction = np.hstack([np.cos(angle), np.sin(angle)])
    # Move the UET_sym = 1 / SCS
    new_pos = ue_pos + v_max * direction * time_frame
    # If it goes out of the room move it back
    new_pos[np.any(new_pos > 0.99 * side, axis=1)] = ue_pos[np.any(new_pos > 0.99 * side, axis=1)]
    # Return new the position
    return new_pos

def std_bar(total_iteration):
    return tqdm(total_iteration, file=sys.stdout, leave=False, ncols=60, ascii=True)

def latex_look(plt):
    # Set LaTeX as the default text interpreter
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']  # LaTeX default font


def check_data(data_shape: tuple, prefix: str, folder: str, overwrite_flag: bool = False):
    filename = os.path.join(folder, prefix + '.csv')

    if os.path.exists(filename) and not overwrite_flag:
        results = pd.read_csv(filename).iloc[:, 1:].to_numpy()
    else:
        results = np.full(data_shape, np.nan)
    return results, filename

# Global dictionaries
cluster_shapes = {'box', 'circle', 'semicircle', 'road'}
channel_stats = {'LoS', 'fading'}
node_labels = {'BS': 0, 'UE': 1, 'RIS': 2}

# The following are defined for graphic purpose only
node_color = {'BS': '#DC2516',  'UE': '#36F507', 'RIS': '#0F4EEA'}
node_mark = {'BS': 'o', 'UE': 'x', 'RIS': '^'}

# The supported channel types are the following.
channel_types = {'LoS', 'No', 'AWGN', 'Rayleigh', 'Rice', 'Shadowing'}

# Path loss computation
def eval_path_loss(carrier_freq: float, pl_exponent: float, gain: float | np.ndarray, distance: float | np.ndarray, ref_distance: float) -> float or np.ndarray:
    """Evaluate path loss based on distance and standard parameters.
    (REMARK: path loss is positive when it is actually a loss. Only exception is when distance is 0 for simplicity)"""
    try:
        output = np.zeros(distance.shape)
        output[distance > 0] = (20 * np.log10(4 * np.pi * carrier_freq / speed_of_light) - gain[distance > 0] + 10 * pl_exponent * np.log10(distance[distance > 0]) + 10 * (2 - pl_exponent) * np.log10(ref_distance)).flatten()
        output[distance == 0] = np.infty
        return output
    except AttributeError:
        if distance == 0:
            return np.infty
        else:
            return  20 * np.log10(4 * np.pi * carrier_freq / speed_of_light) - gain + 10 * pl_exponent * np.log10(distance) + 10 * (2 - pl_exponent) * np.log10(ref_distance)

# Custom distributions
def circular_uniform(n: int, r_outer: float, r_inner: float = 0, rng: np.random.RandomState = None):
    """Generate n points uniform distributed on an annular region. The output is in polar coordinates.

    Parameters
    ----------
    :param n: int, number of points.
    :param r_outer: float, outer radius of the annular region.
    :param r_inner: float, inner radius of the annular region.
    :param rng: np.random.RandomState, random generator needed for reproducibility

    Returns
    -------
    rho: np.ndarray, distance of each point from center of the annular region.
    phi: np.ndarray, azimuth angle of each point.
    """
    if rng is None:
        rho = np.sqrt((r_outer ** 2 - r_inner ** 2) * np.random.rand(n, 1) + r_inner ** 2)
        phi = 2 * np.pi * np.random.rand(n, 1)
    else:
        rho = np.sqrt((r_outer ** 2 - r_inner ** 2) * rng.rand(n, 1) + r_inner ** 2)
        phi = 2 * np.pi * rng.rand(n, 1)
    return np.hstack((rho, phi))


def semicircular_uniform(n: int, r_outer: float, r_inner: float = 0, rng: np.random.RandomState = None):
    """Generate n points uniform distributed on an semi-annular region. The outputs is in polar coordinates.

    Parameters
    ----------
    :param n: int, number of points.
    :param r_outer: float, outer radius of the annular region.
    :param r_inner: float, inner radius of the annular region.
    :param rng: np.random.RandomState, random generator needed for reproducibility

    Returns
    -------
    rho: np.ndarray, distance of each point from center of the annular region.
    phi: np.ndarray, azimuth angle of each point.
    """
    if rng is None:
        rho = np.sqrt((r_outer ** 2 - r_inner ** 2) * np.random.rand(n, 1) + r_inner ** 2)
        phi = np.pi * np.random.rand(n, 1)
    else:
        rho = np.sqrt((r_outer ** 2 - r_inner ** 2) * rng.rand(n, 1) + r_inner ** 2)
        phi = np.pi * rng.rand(n, 1)
    return np.hstack((rho, phi))

def fading(typ: str, dim: tuple = (1,),
           shape: float = 6, seed: int = None) -> np.ndarray:
    """Create a sampled fading channel from a given distribution and given
    dimension.

    Parameters
    __________
    typ : str in dic.channel_types,
        type of the fading channel to be used.
    dim : tuple,
        dimension of the resulting array.
    shape : float [dB],
        shape parameters used in the rice distribution modeling the power of
        the LOS respect to the NLOS rays.
    seed : int,
        seed used in the random number generator to provide the same arrays if
        the same is used.
    """

    if typ not in channel_types:
        raise ValueError(f'Type can only be in {channel_types}')
    elif typ == 'AWGN' or typ == 'LoS':
        return np.ones(dim)
    elif typ == 'No':
        return np.zeros(dim)
    elif typ == "Rayleigh":
        vec = norm.rvs(size=2 * np.prod(dim), random_state=seed)
        return (vec[0:1] + 1j * vec[1:2]).reshape(dim) / np.sqrt(2)  # TODO FIXING THIS BUG
    elif typ == "Rice":
        return rice.rvs(10 ** (shape / 10), size=np.prod(dim), random_state=seed).reshape(dim)  # TODO: FIX ALSO THIS
    elif typ == "Shadowing":
        return norm.rvs(scale=10 ** (shape / 10), random_state=seed)


# Physical noise
def thermal_noise(bandwidth: float, noise_figure: float = 3., t0: float = 293) -> float:
    """Compute the noise power [W] according to bandwidth and ambient temperature.

    :param bandwidth : float, receiver total bandwidth [Hz]
    :param noise_figure: float, noise figure of the receiver [dB]
    :param t0: float, ambient temperature [K]

    :return: power of the noise [W]
    """
    return k * bandwidth * t0 * 10 ** (noise_figure / 10)  # [W]

def thermal_noise_dbm(bandwidth: float, noise_figure: float = 3., t0: float = 293.) -> float:
    """Compute the noise power [dBm] according to bandwidth and ambient temperature.

    :param bandwidth : float, receiver total bandwidth [Hz]
    :param noise_figure: float, noise figure of the receiver [dB]
    :param t0: float, ambient temperature [K]

    :returns: power of the noise [dBm]
    """
    return watt2dbm(thermal_noise(bandwidth, noise_figure, t0))  # [dBm]


# Numeric transformations
def lin2db(lin):
    return 10 * np.log10(lin)

def db2lin(db):
    return 10 ** (db / 10)

def dbm2watt(dbm):
    """Simply converts dBm to Watt"""
    return 10 ** (dbm / 10 - 3)


def watt2dbm(watt):
    """Simply converts Watt to dBm"""
    # with np.errstate(divide='ignore'):
    return 10 * np.log10(watt * 1e3)

def np_ceil(a, precision=0):
    return np.round(a + 0.5 * 10 ** (-precision), precision)


# Coordinate system
def cart2cyl(pos: np.ndarray):
    """ Transformation from cartesian to cylindrical coordinates.

    :param pos: np.array (n,3), position to be transformed
    :return: np.array (n,3), cartesian coordinate
    """
    rho = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    phi = np.arctan2(pos[:, 1], pos[:, 0])
    z = pos[:, 2]
    return np.vstack((rho, phi, z)).T


def cyl2cart(pos: np.ndarray):
    """ Transformation from cylinder to cartesian coordinates.

    :param pos: np.array (n,3), position to be transformed
    :return: np.array (n,3), polar coordinate
    """
    x = pos[:, 0] * np.cos(pos[:, 1])
    y = pos[:, 0] * np.sin(pos[:, 1])
    z = pos[: , 2]
    return np.vstack((x, y, z)).T


def cart2spher(pos: np.ndarray):
    """ Transformation from cartesian to cylindrical coordinates.

    :param pos: np.array (n,3), position to be transformed
    :return: np.array (n,3), cartesian coordinate
    """
    rho = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2 + pos[:, 2] ** 2)
    phi = np.arctan2(pos[:, 1], pos[:, 0])
    theta = np.arccos(pos[:, 2] / rho)
    return np.vstack((rho, theta, phi)).T


def spher2cart(pos: np.ndarray):
    """ Transformation from cylinder to cartesian coordinates.

    :param pos: np.array (n,3), position to be transformed
    :return: np.array (n,3), polar coordinate
    """
    x = pos[:, 0] * np.sin(pos[:, 1]) * np.cos(pos[:, 2])
    y = pos[:, 0] * np.sin(pos[:, 1]) * np.sin(pos[:, 2])
    z = pos[:, 0] * np.cos(pos[:, 1])
    return np.vstack((x, y, z)).T


## Polygon helping functions
def uniform_2d_grid(shape: Polygon, n: int, margin: float = 1.) -> tuple:
    """Generate a 2D grid of uniformly spaced points in a rectangular shape.

    :param shape: Polygon, giving the boundaries where the grid must be done
    :param n: int, number of points in the grid
    :param margin: float, safety margin to reduce the shape. Default is 1. (no margin)
    """

    if n == 1:
       return np.array(shape.centroid.coords), np.zeros(2)

    min_x, min_y, max_x, max_y = shape.bounds
    min_x *= margin
    min_y *= margin
    max_x *= margin
    max_y *= margin
    width_ratio = (max_x - min_x) / (max_y - min_y)

    num_y = np.int32(np.sqrt(n // width_ratio))
    num_x = np.int32(n // num_y)

    x = np.linspace(min_x, max_x, num_x)

    if n == 2:
        y = np.zeros(num_y)
    else:
        # create regularly spaced neurons
        y = np.linspace(min_y, max_y, num_y)

    coords = np.stack(np.meshgrid(x, y), -1).reshape(-1, 2)

    # Compute spacing among elements
    try:
        spacing = np.array([x[1] - x[0], y[1] - y[0]])
    except IndexError:
        spacing = np.array([x[0], y[0]])

    return coords, spacing


def _bounded_voronoi_cells(centers: np.ndarray, polygon: Polygon) -> list:
    """Voronoi cells for `centers`, each clipped to `polygon`. Mirroring
    across the bounding box guarantees every real cell is finite.

    :param centers: np.ndarray, (n,2) array of centers.
    :param polygon: Polygon, the polygon to create the cells from.
    """
    minx, miny, maxx, maxy = polygon.bounds
    x, y = centers[:, 0], centers[:, 1]

    mirrored_left = np.column_stack([2 * minx - x, y])
    mirrored_right = np.column_stack([2 * maxx - x, y])
    mirrored_down = np.column_stack([x, 2 * miny - y])
    mirrored_up = np.column_stack([x, 2 * maxy - y])

    mirrored = np.vstack([mirrored_left, mirrored_right, mirrored_down, mirrored_up])
    vor = Voronoi(np.vstack([centers, mirrored]))

    cells = []
    for i in range(len(centers)):
        region = vor.regions[vor.point_region[i]]
        if not region or -1 in region:
            cells.append(polygon)  # degenerate fallback, shouldn't trigger post-mirroring
            continue
        cell = Polygon(vor.vertices[region])
        if not cell.is_valid:
            cell = cell.buffer(0)
        cells.append(cell.intersection(polygon))
    return cells


def lloyd_relaxation(polygon: Polygon, n: int, n_iter: int = 30, tol: float = 1e-3,
                      rng: np.random.Generator | None = None) -> tuple:
    """Generate n quasi-uniform points inside an arbitrary (convex or not)
    polygon via Lloyd's algorithm (Centroidal Voronoi Tessellation).

    :returns: (centers (n,2), cells (list of n Shapely polygons, one per center))
    """
    rng = rng or np.random.default_rng()

    if n == 1:
        return np.array(polygon.centroid.coords), [polygon]

    # Seed with rejection sampling (works for any polygon shape, convex or not)
    minx, miny, maxx, maxy = polygon.bounds
    centers = np.empty((n, 2))
    filled = 0
    while filled < n:
        cand = rng.uniform([minx, miny], [maxx, maxy], size=(2 * n, 2))
        inside = cand[[polygon.contains(Point(p)) for p in cand]]
        k = min(len(inside), n - filled)
        centers[filled:filled + k] = inside[:k]
        filled += k

    cells = _bounded_voronoi_cells(centers, polygon)
    for _ in range(n_iter):
        new_centers = np.array([
            c.centroid.coords[0] if not c.is_empty else centers[i]
            for i, c in enumerate(cells)
        ])
        shift = np.linalg.norm(new_centers - centers, axis=1).max()
        centers = new_centers
        if shift < tol:
            break
        cells = _bounded_voronoi_cells(centers, polygon)

    return centers, cells

def split_polygon_voronoi(polygon: Polygon, n: int, centers: np.ndarray | None = None,
                          rng: np.random.Generator = np.random.default_rng()) -> MultiPolygon:
    """
    Split a polygon into n sub-polygons using a Voronoi diagram.

    :param polygon: Polygon, the polygon to be split
    :param n: int, number of sub-polygons to generate
    :param centers: np.ndarray, optional, the coordinates of the seed points.
        If not provided, a grid of n uniform points is used.
    :param rng: np.random.Generator, optional, random number generator.

    :returns: MultiPolygon containing each Polygon as a region.
    """
    if n == 1 and centers is None:
        return MultiPolygon([polygon])

    if centers is None:
        centers, cells = lloyd_relaxation(polygon, n, rng=rng)
    else:
        centers = np.asarray(centers, dtype=float)
        cells = _bounded_voronoi_cells(centers, polygon)

    sub_polygons = []
    for c in cells:
        if c.is_empty:
            continue
        if c.geom_type == "Polygon":
            sub_polygons.append(c)
        elif c.geom_type == "MultiPolygon":
            sub_polygons.extend(c.geoms)
    return MultiPolygon(sub_polygons)


def which_region(points: np.ndarray, multi_polygon: Polygon | MultiPolygon,
                 rtree_idx: index.Index | None = None) -> np.ndarray:
    """For each point in `points`, find the indices of polygons in `multi_polygon` that contain it.

    :param points: np.ndarray of points with shape (n, 3). The z-coordinate is ignored.
    :param multi_polygon: sha.MultiPolygon or shapely.Polygon.
    :param rtree_idx: rtree spatial index of multi_polygon, optional. If not provided it is computed
        for the method to succeed.

    :returns np.ndarray: For each point, give the index of the polygon that contains it. -1 if the point is outside
        all polygons.
    """
    # Transform to Multipolygon if needed
    if isinstance(multi_polygon, sha.Polygon):
        multi_polygon = MultiPolygon([multi_polygon])

    # Create spatial index if None
    if rtree_idx is None:
        rtree_idx = build_spatial_index(multi_polygon)

    # Convert points to Shapely Point objects (ignore z-coordinate)
    shapely_points = [sha.Point(p) for p in points]

    # Initialize result: for each point, store a list of polygon indices
    result = -1 * np.zeros(points.shape[0], dtype=np.int32)

    for i, point in enumerate(shapely_points):
        # Find candidate polygons using the spatial index
        for j in rtree_idx.intersection((point.x, point.y, point.x, point.y)):
            if multi_polygon.geoms[j].contains(point):
                result[i] = np.int32(j)
    return result


def get_gaussian_from_polygon(polygon: Polygon) -> tuple:
    """
    Compute a Gaussian covariance matrix whose 3σ ellipse (99.7% confidence region)
    matches the minimum-area bounding ellipse of the input polygon.

    :param polygon: sha.Polygon, Input Shapely Polygon.

    :returns tuple: (mean, covariance_matrix)
            - mean: (x, y, z) center of the Gaussian.
            - covariance_matrix: 3 x 3 covariance matrix of the Gaussian (deficient on z).
    """
    polygon = polygon.convex_hull  # Handle non-convex polygons

    vertices = np.array(polygon.exterior.coords)[:-1]  # Exclude repeated last point
    mean = np.mean(vertices, axis=0)  # Center of the Gaussian

    # Center the vertices
    centered = vertices - mean

    # Compute the covariance matrix of the centered vertices
    cov = np.cov(centered.T)

    # Scale the covariance matrix so that its 3σ ellipse matches the bounding ellipse
    covariance_matrix = cov / 9.0

    return mean, covariance_matrix


def build_spatial_index(multi_polygon: MultiPolygon) -> index.Index:
    """
    Build a spatial index (rtree) for the polygons in a MultiPolygon.

    :param multi_polygon: MultiPolygon: Input MultiPolygon.
    :returns rtree.index.Index: Spatial index for the polygons.
    """
    idx = index.Index()
    for i, polygon in enumerate(multi_polygon.geoms):
        idx.insert(i, polygon.bounds)  # Insert polygon bounds with its index
    return idx


def build_floor_mesh(polygon: Polygon | list | np.ndarray) -> trimesh.Trimesh:
    """Triangulate a polygon and export it as a flat mesh (.ply) at
    z=0, so it can be loaded as a Sionna RT floor SceneObject. Requires `trimesh`.

    :returns: str, path to the exported .ply mesh.
    """
    if isinstance(polygon, list):
        polygon = Polygon(*polygon)
    elif isinstance(polygon, np.ndarray):
        polygon = Polygon(polygon)
    # Generate the mesh
    triangles = shops.triangulate(polygon)
    # Delaunay triangulation works on the vertex set alone, so for a non-convex
    # boundary it can produce triangles that fall outside the polygon. Keep
    # only the ones whose representative point is actually inside it.
    triangles = [t for t in triangles if polygon.contains(t.representative_point())]

    vertex_index, vertices, faces = {}, [], []

    def _idx(pt):
        key = (round(pt[0], 6), round(pt[1], 6))
        if key not in vertex_index:
            vertex_index[key] = len(vertices)
            vertices.append([pt[0], pt[1], 0.0])
        return vertex_index[key]

    for tri in triangles:
        coords = list(tri.exterior.coords)[:-1]  # drop the closing repeated point
        faces.append([_idx(c) for c in coords])
    # Create the mesh
    mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))
    return mesh


def scene_area_coordinates(scene: rt.Scene, z_tol: float = 1e-3) -> np.ndarray:
    """Extract the ground-level boundary vertices of a Sionna scene, in the
    scene's own (world) coordinate system.

    Gathers vertices from every mesh in the scene, keeps only those lying at
    the scene's minimum z (i.e. sitting on the ground/floor), and returns the
    convex hull of their (x, y) projection -- ordered counter-clockwise, at
    the floor's z height. This is the inverse of building a floor mesh from
    a Shapely polygon: it recovers `area_coordinates` from an existing scene.

    :param scene: sionna.rt.Scene, the scene to inspect (e.g. loaded via load_scene()).
    :param z_tol: float, tolerance [m] for considering a vertex to be "on the floor."
    :returns: np.ndarray, shape (V, 3), (x, y, z) of the boundary vertices.
    """
    mi_scene = scene.mi_scene
    all_xyz = []

    for shape in mi_scene.shapes():
        params = mi.traverse(shape)
        if "vertex_positions" not in params:
            continue  # skip non-mesh shapes (e.g. analytic primitives without exposed vertices)
        pos = params["vertex_positions"].numpy().reshape(-1, 3)
        all_xyz.append(pos)

    if not all_xyz:
        raise ValueError("Scene contains no mesh geometry to compute boundaries from.")

    all_xyz = np.vstack(all_xyz)
    floor_z = all_xyz[:, 2].min()

    # Keep only vertices actually sitting on the ground plane
    on_floor = all_xyz[np.abs(all_xyz[:, 2] - floor_z) <= z_tol]
    xy = on_floor[:, :2]

    if len(xy) < 3:
        raise ValueError("Fewer than 3 ground-level vertices found -- cannot form a boundary.")

    hull = ConvexHull(xy)
    hull_xy = xy[hull.vertices]  # scipy orders these counter-clockwise

    area_coordinates = np.hstack([hull_xy, np.full((hull_xy.shape[0], 1), floor_z)])
    return area_coordinates

# Matplotlib helps
def shade_of(base_color: str, rng: np.random.Generator = np.random.default_rng(),
               hue_jitter: float = 0.08, sat_jitter: float = 0.15, val_jitter: float = 0.15):
    """Return an RGB tuple that is a random variation around `base_color`,
    by perturbing its hue/saturation/value slightly and clamping to [0, 1]."""
    r, g, b = mcolors.to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    h = (h + rng.uniform(-hue_jitter, hue_jitter)) % 1.0
    s = np.clip(s + rng.uniform(-sat_jitter, sat_jitter), 0.0, 1.0)
    l = np.clip(l + rng.uniform(-val_jitter, val_jitter), 0.1, 0.9)  # avoid near-black/near-white

    return colorsys.hls_to_rgb(h, l, s)


def printplot(fig: plt.Figure = None,
              ax: plt.Axes or np.array = None,
              render: bool = False,
              filename: str = '',
              dirname: str = '',
              title: str or list = None,
              labels: list = None,
              grid: bool = True,
              orientation: str = 'vertical',
              tight_layout: bool = True):
    # Common print options with LaTeX type definitions
    rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)

    if fig is None:
        fig = plt.gcf()
        ax = plt.gca()
    if isinstance(ax, plt.Axes):    # Single axis plot
        if grid:
            ax.grid(axis='both', color='#E9E9E9', linestyle='--', linewidth = 0.8)
        try:
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.set_ylabel(labels[2])
        except (TypeError, IndexError):
            pass
        if ax.get_legend_handles_labels()[1]:   # Is there is at least a label, print the legend
            ax.legend()
        if not render:
            ax.set_title(title)
            fig.show()
            plt.close(fig=fig)
        else:
            filename = os.path.join(dirname, filename)
            ax.set_title(title)
            fig.savefig(filename + '.jpg', dpi=300)
            ax.set_title('')
            # tikzplotlib.save(filename + '.tex', figure=fig)
            plt.close(fig=fig)
            return
    else:   # Multi-axes plot
        if ax[0].get_legend_handles_labels()[1]:  # Is there is at least a label, print the legend
            ax[0].legend()
        if grid:
            for a in ax:
                a.grid(axis='both', color='#E9E9E9', linestyle='--', linewidth = 0.8)
        try:
            if orientation == 'vertical':
                ax[-1].set_xlabel(labels[0])
                for i, a in enumerate(ax):
                    a.set_ylabel(labels[i+1])
            elif orientation == 'horizontal':
                ax[0].set_ylabel(labels[-1])
                for i, a in enumerate(ax):
                    a.set_xlabel(labels[i])
            else:   # No particular orientation
                i = 0
                for a in ax:
                    a.set_xlabel(labels[i])
                    a.set_ylabel(labels[i + 1])
                    i += 2
        except (TypeError, IndexError):
            pass
        # Keep the title for more than one figure
        try:
            for i, a in enumerate(ax):
                a.set_title(title[i])
        except IndexError:
            ax[0].set_title(title)
        except TypeError:
            pass
        if not render:
            if tight_layout:
                fig.tight_layout()
            fig.show()
            plt.close(fig=fig)
        else:
            filename = os.path.join(dirname, filename)
            if tight_layout:
                fig.tight_layout()
            fig.savefig(filename + '.jpg', dpi=300, bbox_inches='tight')


# Save data
def save_params_npz(filepath: str, **dataclass_instances) -> str:
    """Flatten one or more dataclass instances (SharedParams, GeometryParams, EstimationParams,
    Metrics, ...) into a single compressed .npz archive, one array per field, keyed as
    "<name>_<field>" (name = the kwarg you passed it under). np.load() on a .npz is lazy
    per-key, so a later CSV-of-a-subset only needs to touch the keys it actually wants,
    not the whole archive.

    Field-conversion rules:
      - scipy Rotation -> quaternion array (`.as_quat()`)
      - shapely Polygon/MultiPolygon -> WKT string
      - dict -> stored as a 0-d object array (allow_pickle)
      - a field that's a *reference to another one of the passed dataclass instances*
        (e.g. GeometryParams.c, EstimationParams.geo) is skipped -- avoids saving the
        same data twice; pass every dataclass you want saved explicitly.
      - genuinely unserializable live objects (rt.PlanarArray, rtree.index.Index, a live
        np.random.Generator) are skipped.

    :param filepath: str, output path (".npz" appended automatically if missing).
    :param dataclass_instances: dataclasses passed as name=instance, e.g.
        save_params_npz(path, shared=env.c, geo=env.geo, estimate=env.estimate, metrics=env.metrics)

    :returns: str, the actual path written (with ".npz" ensured).
    """
    instance_ids = {id(v) for v in dataclass_instances.values()}
    flat = {}

    for group_name, obj in dataclass_instances.items():
        for f in dataclasses.fields(obj):
            value = getattr(obj, f.name)
            key = f"{f.name}"

            if id(value) in instance_ids:
                continue  # back-reference to another saved dataclass -- skip, avoid duplication
            if value is None:
                continue
            if isinstance(value, Rotation):
                flat[key] = value.as_quat()
            elif isinstance(value, (sha.geometry.Polygon, sha.geometry.MultiPolygon)):
                flat[key] = np.array(value.wkt)
            elif isinstance(value, (rt.PlanarArray, index.Index, np.random.Generator)):
                # print(f"save_params_npz: skipping non-serializable field '{key}' ({type(value).__name__}).")
                continue
            elif isinstance(value, dict):
                flat[key] = np.array(value, dtype=object)
            else:
                flat[key] = np.asarray(value)

    if not filepath.endswith(".npz"):
        filepath += ".npz"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez_compressed(filepath, **flat)
    return filepath


# Progress bars
def std_progressbar(total_iteration, **kwargs):
    return tqdm(total_iteration, file=sys.stdout, leave=False, ncols=60, ascii=True, **kwargs)

def fixed_progressbar(total_iteration, **kwargs):
    return tqdm(total_iteration, file=sys.stdout, leave=True, ncols=60, ascii=True, **kwargs)

## Folder management
def standard_output_dir(subdirname: str) -> str:
    basedir = os.path.join(os.path.expanduser('~'), 'OneDrive/plots')
    if not os.path.exists(basedir):
        basedir = os.path.join(uppath(__file__, 2), 'plots')
        if not os.path.exists(basedir):
            os.mkdir(basedir)
    subdir = os.path.join(basedir, subdirname)
    if not os.path.exists(subdir):
        os.mkdir(subdir)
    output_dir = os.path.join(subdir, str(date.today()))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    return output_dir

uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])


def get_processing_resources(num_scripts: int,
                             on_gpu: bool, num_ue: int = 1, mib_vram_per_ue: int | float = 200,
                             mib_vram_baseline: int = 210):
    """Estimate the number of CPU workers, threads per worker, and GPU semaphore value based on the number of scripts,
    episodes, and GPU availability.
    It considers a mib_vram_per_ue = 200 if not specified otherwise.

    :param num_scripts: Number of scripts to run simultaneously.
    :param on_gpu: Whether to use GPU or not.
    :param num_ue: Number of UE instances to simulate.
    :param mib_vram_per_ue: Amount of VRAM (in MiB) required per episode to run on GPU.
    :param mib_vram_baseline: Amount of VRAM (in MiB) required for initialization processes.
    """

    cpu_slots = int(max(1, (os.cpu_count() // 2)  // num_scripts))
    if on_gpu:
        gpus = GPUtil.getGPUs()
        free_vram_mib = int(np.sum([gpu.memoryFree for gpu in gpus]))
        vram_per_episode = mib_vram_baseline + mib_vram_per_ue * num_ue
        gpu_slots = int(max(1, free_vram_mib / vram_per_episode / num_scripts))
    else:
        gpu_slots = 0
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    return cpu_slots, gpu_slots
