# Geometry: polygons, Voronoi tessellation, coordinate transforms, meshes, scenes
import numpy as np
from scipy.spatial import ConvexHull, Voronoi
import shapely as sha
from shapely.geometry import Polygon, MultiPolygon
import shapely.ops as shops
from rtree import index
import trimesh
from sionna import rt
import mitsuba as mi
import hashlib
import json


def boundary_dist_normal(points: np.ndarray, polygon: Polygon, eps: float = 1e-6,
                         exclude_polygon: Polygon | None = None,
                         exclude_tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Outward unit normal of the polygon boundary at the point nearest each query point --
    fully vectorized (all points against all boundary segments at once).

    :param points: np.ndarray, shape (N, 3), (x, y, z) query points -- z is ignored.
    :param polygon: shapely.geometry.Polygon.
    :param eps: small offset used to disambiguate the outward side.
    :param exclude_polygon: shapely.geometry.Polygon, optional. If provided, segments lying entirely on its boundary
        will be excluded from the computation.
    :param exclude_tol: float, optional. Tolerance for considering a point to lie on the boundary of exclude_polygon.

    :returns: tuple[np.ndarray (N,), np.ndarray (N, 3)], bein the latter the minimum distance between points and the
        boundary and the former the outward normal vector with unit (x, y, 0.)
    """
    rings = [np.array(polygon.exterior.coords)] + [np.array(r.coords) for r in polygon.interiors]
    # Collect all segments while removing the third dimension
    seg_start = np.vstack([r[:-1, :2] for r in rings])          # (M, 3)
    seg_end = np.vstack([r[1:, :2] for r in rings])             # (M, 3)
    seg_vec = seg_end - seg_start                            # (M, 3)
    seg_len2 = np.sum(seg_vec ** 2, axis=1)
    seg_len2_safe = np.where(seg_len2 == 0, 1.0, seg_len2)

    # Mask out segments that lie entirely on exclude_polygon's boundary
    excluded_mask = np.zeros(seg_start.shape[0], dtype=bool)
    if exclude_polygon is not None:
        ex_boundary = exclude_polygon.boundary
        start_pts = sha.points(seg_start[:, 0], seg_start[:, 1])
        end_pts = sha.points(seg_end[:, 0], seg_end[:, 1])
        start_on = sha.distance(start_pts, ex_boundary) <= exclude_tol
        end_on = sha.distance(end_pts, ex_boundary) <= exclude_tol
        excluded_mask = start_on & end_on  # both endpoints on it -> whole segment lies on it

    # Project points on the plane
    xy = points[:, :2]                                     # (N, 2)

    # Project every point onto every segment at once: (N, R, 2)
    w = xy[:, None, :] - seg_start[None, :, :]
    t = np.clip(np.sum(w * seg_vec[None, :, :], axis=2) / seg_len2_safe[None, :], 0.0, 1.0)  # (U, N)
    proj = seg_start[None, :, :] + t[:, :, None] * seg_vec[None, :, :]                         # (U, N, 2)
    dist2 = np.sum((proj - xy[:, None, :]) ** 2, axis=2)                                       # (U, N)
    dist2[:, excluded_mask] = np.inf    # Excluding segments from the excluded polygon

    # Take index of nearest segment and nearest boundary point
    seg_idx = np.argmin(dist2, axis=1)                       # (N,)
    nxy = proj[np.arange(len(xy)), seg_idx]                  # (N, 2) -- nearest boundary point
    # Compute min distance
    min_distance = np.linalg.norm(nxy - xy, axis=-1)

    # Take the tangent vector and compute the possible candidate for the normal vector
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
    return min_distance, np.where(inside[:, None], -cand, cand)


def distance_to_boundary(points: np.ndarray, polygon: Polygon) -> np.ndarray:
    """Compute the distance between a series of points and a polygon.
    :param points: np.ndarray, (N, 3) coordinates of points. (z-dimension is neglected)
    :param polygon: Polygon, a shapely polygon.

    :returns: np.ndarray, (N,) distance between each point in the polygon.
    """
    boundary = polygon.boundary
    pts = sha.points(points[:, 0], points[:, 1])   # shape (N,) array of Point geometries
    return sha.distance(pts, boundary)              # vectorized, shape (N,)


## Coordinate system
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
                      rng: np.random.Generator = np.random.default_rng()) -> tuple:
    """Generate n quasi-uniform points inside an arbitrary (convex or not)
    polygon via Lloyd's algorithm (Centroidal Voronoi Tessellation).

    :returns: (centers (n,2), cells (list of n Shapely polygons, one per center))
    """
    if n == 1:
        return np.array(polygon.centroid.coords), [polygon]

    # Seed with rejection sampling (works for any polygon shape, convex or not)
    minx, miny, maxx, maxy = polygon.bounds
    centers = np.empty((n, 2))
    filled = 0
    while filled < n:
        cand = rng.uniform([minx, miny], [maxx, maxy], size=(2 * n, 2))
        inside = cand[[polygon.contains(sha.Point(p)) for p in cand]]
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
    # If a single region, return region 0 for each point
    if isinstance(multi_polygon, sha.Polygon):
        return np.zeros(points.shape[0], dtype=np.int32)
    elif len(multi_polygon.geoms) == 1:
        return np.zeros(points.shape[0], dtype=np.int32)

    # Create spatial index if None
    if rtree_idx is None:
        rtree_idx = build_spatial_index(multi_polygon)

    # Flatten point if it is a tensor
    shape = points.shape
    if len(shape) > 2:
        points = points.reshape(-1, shape[-1])

    # Convert points to Shapely Point objects (ignore z-coordinate)
    shapely_points = [sha.Point(p) for p in points]

    # Initialize result: for each point, store a list of polygon indices
    result = -1 * np.ones(points.shape[0], dtype=np.int32)

    for i, point in enumerate(shapely_points):
        # Find candidate polygons using the spatial index
        for j in rtree_idx.intersection((point.x, point.y, point.x, point.y)):
            if multi_polygon.geoms[j].contains(point):
                result[i] = np.int32(j)
    if len(shape) > 2:
        result = result.reshape(shape[:-1])
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
        polygon = sha.polygons(*polygon)
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


def floor_mesh_cache_key(area_coordinates: list | np.ndarray) -> str:
    """Return a cache key incorporating all coordinates and polygon holes."""

    def normalize(value):
        if isinstance(value, np.ndarray):
            return normalize(value.tolist())
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return float(value)

    payload = json.dumps(
        {
            "format": "floor-v1",
            "coordinates": normalize(area_coordinates),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()