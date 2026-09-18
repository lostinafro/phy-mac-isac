import numpy as np
from dataclasses import dataclass, field, fields, asdict, make_dataclass
import rtree
from scipy.spatial.transform import Rotation
from scipy.constants import speed_of_light
import scipy.stats as stats

import utils as ut
import shapely as sha
import matplotlib as mpl
import matplotlib.pyplot as plt
import sionna.rt as rt
import os
from argparse import ArgumentParser

## DEFAULT PARAMETERS
# Scenario
NUM_AP = 3
NUM_ANT = 2
NUM_UE = 18
NUM_EP = int(1e2)
ROOM_SIDE = 200             # [m]
VEL_STD = 10                 # [m/s]
REGIONS = 16
UE_HEIGHT = 1.5             # [m]

# Wireless
CARRIER = 3e9       # [Hz]
NUMEROLOGY = 0
NUM_SCS = 108
NUM_SYM = 14
NUM_SLOTS = 5 * 2 ** NUMEROLOGY
NUM_FRAMES = 100
PPM_XO = 5 / 50                # Residual PPM after initial synchronization from PRACH
PPM_OCXO = 0.5 / 50            # Residual PPM after initial synchronization
DATASIZE_AVG = 2048
QUEUE_MAX = int(10 * DATASIZE_AVG)
QUEUE_RELIABILITY = 0.95         # 1 - epsilon in the paper
UE_POW = ut.dbm2watt(-10)

# DEFAULT FOLDERS
FIGURE_FOLDER = os.path.join(os.getcwd(), "figs")
DATA_FOLDER = os.path.join(os.getcwd(), "data")
SCENE_FOLDER = os.path.join(os.getcwd(), "scenes")
MESH_FOLDER = os.path.join(SCENE_FOLDER, "meshes")

# RayTracer and RadioMap parameters
RT_PARAM = {
    "max_depth": 3,
    "los": True,
    "specular_reflection": False,
    "diffuse_reflection": False,
    "refraction": False,
    "diffraction": False,
    "edge_diffraction": False,
    "synthetic_array": False,
    "samples_per_tx": int(5*1e8),
    "cell_size": (.5, .5),
}

scattering_coefficient_dict = {
    "vacuum": 0.00,
    "glass": 0.00,
    "concrete": 0.15,
    "brick": 0.1,
    "marble": 0.05,
    "metal": 0.30,
    "wood": 0.05
}

scattering_pattern_dict = {
    "vacuum": rt.LambertianPattern(),
    "glass": rt.DirectivePattern(1),
    "concrete": rt.LambertianPattern(),
    "brick": rt.LambertianPattern(),
    "marble": rt.DirectivePattern(1),
    "metal": rt.DirectivePattern(1),
    "wood": rt.LambertianPattern()
}

def common_parser():
    # TODO: to be moved on the run_episode script. Also, give for granted gpu, and ask for cpu instead (but doing it properly would be nice)
    parser = ArgumentParser(description="Common parser to run simulations.")

    parser.add_argument("--ra_method", type=str, default="", help="Method to use for RA simulation")
    parser.add_argument('--parallel', action='store_true', help="Enable parallel processing")
    parser.add_argument('--savesubdir', type=str, default="", help="Optional name of the subdirectory to save results in")
    parser.add_argument('--quiet', action='store_true', help="Quiet outputs")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite results  if already present in the savedir")
    parser.add_argument("--concurrent", type=int, default=1, help="Set the number of concurrent scripts run")
    parser.add_argument("--gpu", action='store_true', help="Set the GPU usage")

    args = parser.parse_args()

    return args.ra_method, args.parallel, args.savesubdir, args.quiet, args.overwrite, args.concurrent, args.gpu

## CLASSES
@dataclass
class SharedParams:
    """Shared parameters across different components of the simulation.

    :ivar num_ap: int, number of Access Points (APs).
    :ivar num_ant_h: int, number of horizontal antennas per AP.
    :ivar num_ant_v: int, number of vertical antennas per AP.
    :ivar num_ant: int, total number of antennas per AP (M_h * M_v).
    :ivar num_ue: int, number of User Equipment (UE) devices.
    :ivar num_scs: int, number of subcarriers.
    :ivar num_sym: int, number of OFDM symbols per slot.
    :ivar num_slots: int, number of slots per frame.
    :ivar num_frames: int, number of frames. Frame 0 is the initial state of the environment.
    :ivar num_regions: int, number of semantic regions within the environment.
    :ivar rays: dict, ray tracing key-value parameters.
    :ivar num_target: int, number of passive targets in the environment.
    :ivar num_car: int, number of cars in the environment.
    :ivar rng: np.random.Generator, random number generator used for simulation.
    """

    num_ap: int = NUM_AP
    num_ant_h: int = NUM_ANT
    num_ant_v: int = 1
    num_ant: int = field(init=False)
    num_ue: int = NUM_UE
    num_scs: int = NUM_SCS
    num_sym: int = NUM_SYM
    num_slots: int = NUM_SLOTS
    num_frames: int = NUM_FRAMES
    num_regions: int = REGIONS
    # RayTracerSimulator
    rays: dict | None = None
    # Not used yet
    num_target: int = 0
    num_car: int = 0

    # Generator
    rng: np.random.Generator = np.random.default_rng()

    def __post_init__(self):
        self.num_ant = self.num_ant_h * self.num_ant_v
        
        if self.rays is None:
            self.rays = RT_PARAM

@dataclass
class WirelessParams:
    """Parameters related to wireless communication.

    :ivar numerology: 5G NR numerology index (μ). Determines subcarrier spacing and slot duration.
        μ=0 → 15 kHz, μ=1 → 30 kHz, etc.
    :ivar carrier: Carrier frequency in Hz (e.g., 3.5e9 for 3.5 GHz).
    :ivar c: SharedParams object containing common simulation parameters.
    :ivar slot_time: Duration of a slot in seconds, computed as 1e-3 / 2^μ.
    :ivar sym_time: Duration of an OFDM symbol in seconds (useful symbol time 1/sc_spacing plus cp_time).
    :ivar frame_time: Duration of a frame in seconds (includes DL/UL signaling overhead).
    :ivar sc_spacing: Subcarrier spacing in Hz (15e3 * 2^μ).
    :ivar rb_spacing: Resource block (RB) spacing in Hz (12 * sc_spacing).
    :ivar sc_index: Subcarrier indices centered on the carrier (shape: (F,)).
    :ivar sym_index: OFDM symbol indices within a slot, i.e. [0, ..., T - 1] (shape: (T,)).
    :ivar slot_index: Indices of the first symbol of each slot within a frame (shape: (S,)).
    :ivar freqs: Baseband frequencies of the subcarriers (shape: (F,)).
    :ivar bw: Total bandwidth in Hz (sc_spacing * num_scs).
    :ivar nob_ap: Thermal noise power at APs (W).
    :ivar nob_ue: Thermal noise power at UEs (W).
    :ivar lo_ppm: Nominal part-per-million error of the transmitter's Local oscillators (LO).
    """
    c: SharedParams
    numerology: int
    carrier: float
    lo_ppm: float | np.ndarray = 0.
    ue_power: float | np.ndarray = None
    ap_power: float | np.ndarray = None
    avg_datasize : np.ndarray = None
    traffic_pdf: stats.rv_continuous = stats.expon
    # Performance guarantees
    queue_reliability: float = None
    queue_max: np.ndarray = None

    # Computed attributes (excluded from __init__)
    cp_time: float = field(init=False)
    sym_time: float = field(init=False)
    slot_time: float = field(init=False)
    frame_time: float = field(init=False)
    sc_spacing: float = field(init=False)
    rb_spacing: float = field(init=False)
    sc_index: np.ndarray = field(init=False)
    sym_index: np.ndarray = field(init=False)
    slot_index: np.ndarray = field(init=False)
    freqs: np.ndarray = field(init=False)
    bw: float = field(init=False)
    nob_ap: float = field(init=False)
    nob_ue: float = field(init=False)
    cfo_base: np.ndarray = field(init=False)
    cto_base: np.ndarray = field(init=False)
    channel: np.ndarray = field(init=False)

    def __post_init__(self):
        """Compute derived attributes after initialization."""
        # Compute slot_time from numerology
        self.slot_time = 1e-3 / 2 ** self.numerology

        # Compute frame_time from slot_time and S (assuming the same number of DL slots)
        guard = 0.09
        self.frame_time = self.slot_time * (2 * self.c.num_slots) + guard

        # Compute subcarrier spacing
        self.sc_spacing = 15e3 * 2 ** self.numerology

        # Compute RB spacing
        self.rb_spacing = 12 * self.sc_spacing

        # Compute cp_time and sym time from numerology
        self.cp_time = 4.69e-6 / 2 ** self.numerology
        self.sym_time = self.cp_time + 1 / self.sc_spacing

        # Compute indexes
        self.freqs = np.array(rt.utils.subcarrier_frequencies(self.c.num_scs, self.sc_spacing))
        self.sc_index = np.array(self.freqs / self.sc_spacing, dtype=np.int32)
        self.sym_index = np.arange(self.c.num_slots * self.c.num_sym)
        self.times = self.sym_index * self.sym_time
        self.slot_index = np.arange(self.c.num_slots) * self.c.num_sym

        # Compute bandwidth
        self.bw = self.sc_spacing * self.c.num_scs

        # Check LO_PPM
        if isinstance(self.lo_ppm, float):
            self.lo_ppm = np.ones(self.c.num_ue) * self.lo_ppm
        # Set CTO and CFO at zero at first
        self.cfo_base = np.zeros(self.c.num_ue)
        self.cto_base = np.zeros(self.c.num_ue)

        # Compute power
        if self.ue_power is None:
            self.ue_power = np.ones(self.c.num_ue) * ut.dbm2watt(10)
        elif isinstance(self.ue_power, float):
            self.ue_power = np.ones(self.c.num_ue) * self.ue_power
        if self.ap_power is None:
            self.ap_power = np.ones(self.c.num_ap) * ut.dbm2watt(30)
        elif isinstance(self.ap_power, float):
            self.ap_power = np.ones(self.c.num_ap) * self.ap_power

        # Compute thermal noise
        self.nob_ap = ut.thermal_noise(self.bw, noise_figure=9.0)
        self.nob_ue = ut.thermal_noise(self.bw, noise_figure=9.0)

        # Compute datasize and queue
        if self.avg_datasize is None:
            self.avg_datasize = DATASIZE_AVG * np.ones(self.c.num_ue, dtype=int)
        elif isinstance(self.avg_datasize, float):
            self.avg_datasize = np.ones(self.c.num_ue) * self.avg_datasize
        if self.queue_max is None:
            self.queue_max = QUEUE_MAX * np.ones(self.c.num_ue, dtype=np.int32)
        if self.queue_reliability is None:
            self.queue_reliability = QUEUE_RELIABILITY * np.ones(self.c.num_ue)

        # Initialize channel
        self.channel = np.zeros((self.c.num_frames, self.c.num_ap, self.c.num_ant, self.c.num_ue,
                                 self.c.num_slots, self.c.num_sym, self.c.num_scs), dtype=np.complex64)

    def update(self, frame: int | np.ndarray = 0, **kwargs):
        """Update wireless parameters with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                if key == "traffic_pdf":
                    if isinstance(value, np.ndarray):
                        value = value.item()
                    # Take the right generator
                    value = getattr(stats, value)
                    setattr(self, key, value)
                elif key == "channel":
                    attr = getattr(self, key)
                    attr[frame] = value
                else:
                    setattr(self, key, value)
            else:
                raise AttributeError(f"WirelessParams has no attribute '{key}'")

    def sample_lo_offset(self,
                         model: str,  #["deterministic", "gaussian", "uniform"],
                         ppm: float | np.ndarray | None = None,
                         t: float | None = None,
                         store: bool = True,
                         rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Carrier Frequency Offset (CFO) in Hz and Clock Timing Offset (CTO) in seconds.

        :param model: str, model for ppm error ("deterministic", "gaussian", "uniform").
        :param ppm: float or np.ndarray, nominal ppm error of the LO (positive). Its shape determines the output shape.
            If `model` is "deterministic", this is the ppm error of the LO.
            If `model` is "gaussian", this is 3 times the standard deviation of the Gaussian distribution (99.7% rule).
            If `model` is "uniform", this is the range of the uniform distribution.
        :param t: Observation time in seconds (default: self.sym_time).
        :param store: If True, store the computed CFO and CTO in self.cfo_base and self.cto_base.
        :param rng: Random number generator (default: self.c.rng).

        :return: Tuple of (CFO [Hz], CTO [s]).
        """
        # Checking inputs
        if t is None:
            t = self.sym_time
        if rng is None:
            rng = self.c.rng
        if ppm is None:
            ppm = self.lo_ppm
        # Take a positive value for ppm
        ppm = np.abs(ppm)

        # Chack different models
        if model == "deterministic":
            cfo_hz = self.carrier * ppm * 1e-6
        elif model == "gaussian":
            ppm_random = rng.normal(0, ppm / 3)
            cfo_hz = self.carrier * ppm_random * 1e-6
        elif model == "uniform":
            if isinstance(ppm, (float, int)):
                shape = (1,)
                ppm = np.ndarray([-np.abs(ppm), np.abs(ppm)])
            elif isinstance(ppm, np.ndarray):
                shape = ppm.shape
                ppm = np.vstack((-np.abs(ppm.flatten()), np.abs(ppm.flatten())))
            else:
                raise ValueError(f"ppm must be a float or np.ndarray, not {type(ppm)}")
            ppm_random = rng.uniform(ppm[0], ppm[1])
            ppm_random = np.reshape(ppm_random, shape)
            cfo_hz = self.carrier * ppm_random * 1e-6
        else:
            raise ValueError(f"Unknown model: {model}. Supported are 'deterministic', 'gaussian', and 'uniform'.")

        cto_s = (cfo_hz / self.carrier) * t
        # If store is True keep the values in the class
        if store:
            self.cfo_base = np.array(cfo_hz)
            self.cto_base = np.array(cto_s)
        return np.array(cfo_hz), np.array(cto_s)

    def sample_new_data(self):
        """Generate new data [bits] sampling from the defined distribution."""
        return self.traffic_pdf.rvs(scale=self.avg_datasize, random_state=self.c.rng).astype(int)

    def datasize_from_load(self, load_factor: float, avg_se: float) -> tuple[float, float]:
        """Initialize the arrival rate for a given number of UEs, slots, symbols, and average data size."""
        return arrival_rate_init(self.c.num_ue, self.c.num_slots, avg_se, self.bw, self.slot_time, load_factor)

    @property
    def ap_power_dbm(self):
        return ut.watt2dbm(self.ap_power)

    @property
    def ue_power_dbm(self):
        return ut.watt2dbm(self.ue_power)

    @property
    def nob_ap_dbm(self):
        return ut.watt2dbm(self.nob_ap)

    @property
    def nob_ue_dbm(self):
        return ut.watt2dbm(self.nob_ue)

    @property
    def wavelength(self) -> float:
        return speed_of_light / self.carrier

    @property
    def wavenum(self) -> float:
        return 2 * np.pi / self.wavelength

@dataclass
class GeometryParams:
    """Parameters related to the geometric configuration of the simulation environment.

    :ivar area_coordinates: Coordinates defining the boundaries of the simulation area.
        If a np.array, it is a convex polygon with shape (A, 3) where A is the number of vertices, and each row
        represents (x, y, z) coordinates of a boundary vertex.
        If a list, it is [shell, holes] with both shell and holes being np.ndarray of vertices of size (A, 3).
    :ivar environment_shape: Shapely polygon representing the outer boundary of the environment
        using the x,y coordinates (first two columns of area_coordinates).
    :ivar semantic_regions: List of Shapely polygons representing semantic regions within the environment.
        Each polygon defines a distinct region with specific properties.
    :ivar ue_location: Positions of User Equipment devices in the absolute coordinate system.
        Shape: (U, 3) where U is the number of UEs. Each row is (x, y, z) in meters.
    :ivar ap_location: Positions of Access Points in the absolute coordinate system.
        Shape: (L, 3) where L is the number of APs. Each row is (x, y, z) in meters.
    :ivar ap_orientation: Rotation matrices for AP antenna array orientation.
        List of scipy Rotation instances (length L) defining rotation from z-boresight
        to actual pointing direction for each AP.
    :ivar antenna_location: Positions of individual AP antennas in the absolute coordinate system.
        Shape: (L, M, 3) where M is the number of antennas per AP.
        Each element is (x, y, z) coordinates in meters.
    :ivar antenna_spacing: Uniform inter-element spacing between antennas in wavelengths.
        Applied to both horizontal and vertical dimensions. Default is λ/2.
    :ivar ue_velocity: Current velocities of User Equipment devices.
        Shape: (K, U, 3) where each last dimension is (vx, vy, vz) in meters/second.
    :ivar ue_vel_cov: Velocity covariance matrices for UEs on the ground plane.
        Shape: (U, 2, 2) where each (2, 2) matrix is the covariance for (vx, vy)
        components of velocity for each UE.
    """
    c: SharedParams
    area_coordinates: np.ndarray | list
    semantic_regions: sha.geometry.MultiPolygon | None = None
    ap_location: np.ndarray | None = None
    ap_orientation: Rotation | None = None
    antenna_location: np.ndarray | None = None
    antenna_spacing: float | None = None
    ue_vel_cov: np.ndarray | None = None

    # Computed attributes (excluded from __init__)
    environment_shape: sha.geometry.Polygon = field(init=False)
    regions_idx: rtree.index.Index = field(init=False)
    ue_location: np.ndarray = field(init=False)
    ue_velocity: np.ndarray = field(init=False)
    ue_region: np.ndarray = field(init=False)

    def __post_init__(self):
        # Build environment geometry using Shapely, checking if there are holes defined in the area coordinates
        if isinstance(self.area_coordinates, list):
            self.environment_shape = sha.geometry.Polygon(*self.area_coordinates)
            self.area_coordinates = np.array(self.area_coordinates[0])
        else:
            self.environment_shape = sha.geometry.Polygon(self.area_coordinates)
        # For UE's location and velocity, account for different position each frame
        self.ue_location = np.zeros((self.c.num_frames + 1, self.c.num_ue, 3))
        self.ue_velocity = np.zeros((self.c.num_frames + 1, self.c.num_ue, 3))
        self.ue_region = -1 * np.ones((self.c.num_frames + 1, self.c.num_ue), dtype=np.int32)

        if self.semantic_regions is None:
            self.semantic_regions = ut.split_polygon_voronoi(polygon=self.environment_shape,
                                                             n=self.c.num_regions,
                                                             rng=self.c.rng)
            self.regions_idx = ut.build_spatial_index(self.semantic_regions)

    def update(self, frame: int | np.ndarray = 0, **kwargs):
        """Update geometry parameters with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                if key in ['ue_location', 'ue_velocity']:
                    # If these keys are used, the value must be inserted into the current frame
                    # If value has not the correct shape, numpy will raise IndexError
                    attr = getattr(self, key)
                    attr[frame] = value
                elif key in ["semantic_regions", "environment_shape"]:
                    if isinstance(value, (sha.Polygon, sha.MultiPolygon)):
                        setattr(self, key, value)
                    else:   # If the value is a wkt, rebuild build from that
                        setattr(self, key, sha.from_wkt(value))
                    if key == "semantic_regions":
                        self.regions_idx = ut.build_spatial_index(self.semantic_regions)
                elif key == "ap_orientation":
                    if isinstance(value, np.ndarray):
                        value = Rotation.from_quat(value)
                    setattr(self, key, value)
                else:
                    setattr(self, key, value)
            else:
                raise AttributeError(f"GeometryParams has no attribute '{key}'")
        # At the end, because it requires everything else in place
        if "ue_location" in kwargs.keys():
            self.ue_region[frame] = self.in_which_region(kwargs["ue_location"])

    def in_which_region(self, points: np.ndarray) -> np.ndarray:
        """
        For each point in `points`, find the indices of polygons in `multi_polygon` that contain it.

        :param points: np.ndarray of points with shape (n, 3). The z-coordinate is ignored.

        :returns np.ndarray: For each point, give the index of the polygon that contains it. -1 if the point is outside
            all polygons.
        """
        return ut.which_region(points, self.semantic_regions, self.regions_idx)

    def show(self, frame: int = -1,
             ax: plt.Axes | None = None, figsize: tuple=(6, 6), cmap_name: str = "tab20",
             ue_radius: float = 1., arrow_scale: float = 1,
             ap_width: float = 1., ap_length: float = 5.0) -> plt.Axes:
        """Plot the environment boundary and semantic regions.

        :param frame: the index of the frame to plot.
        :param ax: existing matplotlib Axes to draw on. If None, a new figure is created.
        :param figsize: size of the created figure (ignored if `ax` is given).
        :param cmap_name: matplotlib colormap used to color the semantic regions.
        :param ue_radius: radius of the UEs' circles in the plot.
        :param arrow_scale: scale factor for the UEs' velocity arrows.
        :param ap_width: width of the APs' rectangles in the plot.
        :param ap_length: length of the APs' rectangles in the plot.

        :returns: the matplotlib Axes used for the plot.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        # --- Environment boundary: outline only, no fill ---
        ex, ey = self.environment_shape.exterior.xy
        ax.plot(ex, ey, color="black", linewidth=1.5, zorder=1)

        # Place the $\mathcal{E}$ label just outside one corner.
        # Use the polygon's centroid as a reference to push the label radially outward from the chosen corner,
        # so it lands outside the boundary regardless of the polygon's orientation/shape.
        cx, cy = self.environment_shape.centroid.x, self.environment_shape.centroid.y
        corner_x, corner_y = self.area_coordinates[0, 0], self.area_coordinates[0, 1]
        dx, dy = corner_x - cx, corner_y - cy
        norm = np.hypot(dx, dy) or 1.0
        offset = 0.06 * np.hypot(*(np.array(self.environment_shape.bounds[2:]) -
                                   np.array(self.environment_shape.bounds[:2])))
        label_x = corner_x + offset * dx / norm
        label_y = corner_y + offset * dy / norm
        ax.text(label_x, label_y, r"$\mathcal{E}$", fontsize=14, ha="center", va="center", zorder=3)

        # --- Semantic regions: filled + colored + labeled ---
        if self.semantic_regions is not None:
            regions = list(self.semantic_regions.geoms)
            cmap = mpl.colormaps[cmap_name].resampled(max(len(regions), 1))

            for l, region in enumerate(regions):
                if region.is_empty:
                    continue
                color = cmap(l % cmap.N)

                # A region could itself be a MultiPolygon after clipping;
                # handle both cases when drawing the fill/outline.
                polys = list(region.geoms) if region.geom_type == "MultiPolygon" else [region]
                for poly in polys:
                    rx, ry = poly.exterior.xy
                    ax.fill(rx, ry, color=color, alpha=0.4, zorder=1)
                    ax.plot(rx, ry, color=color, linewidth=1.0, zorder=2)

                # Label at the region's center.
                # Use representative_point() instead of centroid because the former is guaranteed to lie inside.
                label_pt = region.representative_point()
                ax.text(label_pt.x, label_pt.y, rf"$\mathcal{{E}}_{{{l}}}$",
                        fontsize=12, ha="center", va="center", zorder=3)

        # --- UEs: circles + velocity arrows ---
        # Skip entirely if every UE sits at the origin (i.e. positions haven't
        # been generated yet), but still allow individual UEs to legitimately be at [0,0].
        if self.ue_location is not None and not np.allclose(self.ue_location, 0.0):
            ue_xy = self.ue_location[frame][:, :2]
            ue_vel_xy = self.ue_velocity[frame][:, :2]
            for u, (pos, vel) in enumerate(zip(ue_xy, ue_vel_xy)):
                color = ut.shade_of("blue", rng=self.c.rng)
                circle = plt.Circle(pos, ue_radius, color=color, fill=True, zorder=4)
                ax.add_patch(circle)
                ax.text(pos[0], pos[1], f"{u}", fontsize="small", ha="center", va="bottom", zorder=6)
                speed = np.hypot(*vel)
                if speed > 1e-9:
                    ax.arrow(pos[0], pos[1], vel[0] * arrow_scale, vel[1] * arrow_scale,
                             head_width=ue_radius * 0.6, head_length=ue_radius * 0.8,
                             fc=color, ec=color, zorder=5, length_includes_head=True)

        # --- APs: oriented rectangles ---
        if self.ap_location is not None:
            # Evaluate length of the sides
            half_w, half_l = ap_width / 2, ap_length / 2
            # ap_orientation rotates the local frame; applying it to the local x-axis gives the vector from the AP's
            # center to the midpoint of its longer side. That vector is, by construction, perpendicular to the long
            # axis of the rectangle (it points along the *width*).
            ap_xy = self.ap_location[:, :2]
            # Take width direction of each AP's orientation matrix and normalize it.
            width_dir = self.ap_orientation.apply([0.0, 0.0, 1.0])[:, :2]
            w_norm = np.hypot(width_dir[:, 0], width_dir[:, 1])
            width_dir[w_norm >= 1e-9] = width_dir[w_norm >= 1e-9] / w_norm[w_norm >= 1e-9][:, np.newaxis]
            width_dir[w_norm < 1e-9] = np.array([1.0, 0.0])
            # Rotate 90° on the ground to obtain length_dir
            length_dir = np.vstack([-width_dir[:, 1], width_dir[:, 0]]).T
            # Iterate to construct the rectangles
            for l, pos in enumerate(ap_xy):
                corners = np.array([
                    pos + half_w * width_dir[l] + half_l * length_dir[l],
                    pos + half_w * width_dir[l] - half_l * length_dir[l],
                    pos - half_w * width_dir[l] - half_l * length_dir[l],
                    pos - half_w * width_dir[l] + half_l * length_dir[l],
                ])
                rect = plt.Polygon(corners, closed=True, color="red", alpha=0.8,
                                   zorder=4, linewidth=0.5)
                ax.add_patch(rect)
                # Insert orientation
                ax.arrow(pos[0], pos[1], width_dir[l, 0] * arrow_scale, width_dir[l, 1] * arrow_scale,
                         head_width=ap_width * 0.6, head_length=ap_width * 0.8,
                         fc="gray", ec="gray", zorder=5, length_includes_head=True)
                # Insert AP number
                ax.text(pos[0], pos[1], rf"{l}", fontsize=10, ha="center", va="bottom", zorder=6)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        return ax

    @property
    def center(self) -> np.ndarray:
        """Compute the center of the environment."""
        return np.hstack([np.array(self.environment_shape.centroid.coords[0]).round(3), 0])


@dataclass
class EstimationParams:
    """Collect estimation and tracking parameters.
     All estimation variables are saved, from frame = 0 to frame = self.c.num_frames - 1.
     If frame = self.c.num_frames (indexed also with [-1]), the element is the initialization variables.

     :ivar c:
     :ivar geo: GeometryParams, needed to the
    """
    c: SharedParams
    geo: GeometryParams

    # Computed attributes (excluded from __init__)
    ue_pos_hat_prio: np.ndarray | None = field(default=None, init=False)
    ue_pos_hat_post: np.ndarray | None = field(default=None, init=False)
    ue_pos_cov_prio: np.ndarray | None = field(default=None, init=False)
    ue_pos_hat_meas: np.ndarray | None = field(default=None, init=False)
    ue_pos_cov_meas: np.ndarray | None = field(default=None, init=False)
    ue_pos_cov_post: np.ndarray | None = field(default=None, init=False)
    ue_region_hat: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self):
        self.ue_pos_hat_prio = np.zeros((self.c.num_frames + 1, self.c.num_ue, 3))
        self.ue_pos_hat_post = np.zeros((self.c.num_frames + 1, self.c.num_ue, 3))
        self.ue_pos_cov_prio = np.zeros((self.c.num_frames + 1, self.c.num_ue, 3, 3))
        self.ue_pos_hat_meas = np.full((self.c.num_frames + 1, self.c.num_ue, 3), np.nan)
        self.ue_pos_cov_meas = np.full((self.c.num_frames + 1, self.c.num_ue, 3, 3), np.nan)
        self.ue_pos_cov_post = np.zeros((self.c.num_frames + 1, self.c.num_ue, 3, 3))
        self.ue_region_hat = -1 * np.ones((self.c.num_frames + 1, self.c.num_ue), dtype=np.int32)

    def update(self, frame: int | np.ndarray = 0, idx: np.ndarray | None = None, **kwargs):
        """Update estimate parameters for the selected frame with new values."""
        if idx is None:
            idx = np.arange(self.c.num_ue)
        for key, value in kwargs.items():
            if hasattr(self, key):
                # Get the attribute
                attr = getattr(self, key)
                # If value has not the correct shape, numpy will raise IndexError
                if isinstance(attr, np.ndarray):
                    if "meas" in key:
                        attr[frame, idx] = value
                    else:
                        attr[frame] = value
                    if key in ("ue_pos_hat_prio", "ue_pos_hat_post"):
                        self.ue_region_hat[frame] = self.geo.in_which_region(value)
                        out_of_regions =  self.ue_region_hat[frame] == -1
                        self.ue_region_hat[frame, out_of_regions] = self.ue_region_hat[frame - 1, out_of_regions]
                else:
                    # If the attribute is not an array, update it directly
                    setattr(self, key, value)
            else:
                raise AttributeError(f"EstimateParams has no attribute '{key}'")

    def show(self, ax: plt.Axes, frame: int = -1, kind: str = "post", confidence: float = 0.99,
             ue_idx: np.ndarray | None = None) -> plt.Axes:
        """Plot each UE's estimated (x, y) position as an 'x' marker, with a Gaussian
        confidence-ellipse outline derived from the corresponding position covariance.

        :param ax: matplotlib Axes to draw on.
        :param frame: int, which stored frame to plot (default: -1, the initialization frame).
        :param kind: str, one of {"prio", "meas", "post"} -- which estimate/covariance pair to use.
        :param confidence: float, confidence level in (0, 1) for the covariance ellipse. Default 0.99.
        :param ue_idx: np.ndarray | None, which UEs to plot. Default: all.

        :returns: the Axes used for the plot.
        """
        if kind not in ("prio", "meas", "post"):
            raise ValueError(f"kind must be one of 'prio', 'meas', 'post', got '{kind}'")
        if not 0. < confidence < 1.:
            raise ValueError(f"p must be in (0, 1), got {confidence}")

        pos = getattr(self, f"ue_pos_hat_{kind}")[frame]  # (U, 3)
        cov = getattr(self, f"ue_pos_cov_{kind}")[frame]  # (U, 3, 3)
        ue_idx = np.arange(self.c.num_ue) if ue_idx is None else ue_idx

        chi2_val = stats.chi2.ppf(confidence, df=2)  # Mahalanobis-distance^2 threshold enclosing probability p

        for u in ue_idx:
            mu = pos[u, :2]
            sigma = cov[u, :2, :2]  # exact marginal (x,y) covariance -- drop the z row/column

            if np.any(np.isnan(mu)):
                continue  # e.g. "meas" wasn't updated for this UE at this frame

            # Take color
            color = ut.shade_of("grey", rng=self.c.rng)
            # Add UE estimation and number
            ax.plot(mu[0], mu[1], marker="x", color=color, markersize=4, zorder=5)
            ax.text(mu[0], mu[1], f"{u}", color="gray", fontsize="x-small", ha="center", va="bottom", zorder=6)

            # Eigendecomposition: eigenvalues = variances along the ellipse's principal axes,
            # eigenvectors = their directions.
            eigvals, eigvecs = np.linalg.eigh(sigma)
            eigvals = np.clip(eigvals, a_min=0., a_max=None)  # guard tiny negative numerical noise

            # Eigenvalues are sorted in ascending ordered
            # Semi-axis lengths sqrt(eigenvalue * chi2 quantile); Ellipse wants full width/height.
            width, height = 2 * np.sqrt(eigvals[::-1] * chi2_val)
            # width is along the `angle` direction

            # Orientation: angle of the eigenvector for the LARGER eigenvalue (major axis).
            angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))

            if kind == "meas":
                # The measured covariance are centered at the UE true location
                mu = self.geo.ue_location[frame, u, :2]

            ellipse = mpl.patches.Ellipse(xy=mu, width=width, height=height, angle=angle,
                              edgecolor=color, facecolor="none", linewidth=1.2, zorder=4)
            ax.add_patch(ellipse)
        return ax


@dataclass
class Metrics:
    """Collect relevant metrics.
     All metric variables are saved, from frame = 0 to frame = self.c.num_frames - 1.
     If frame = self.c.num_frames (indexed also with [-1]), the element is the initialization variable.

     :ivar c: SharedParams, shared parameters of the simulation.
     :ivar
     """
    c: SharedParams

    # Computed attributes (excluded from __init__)
    # Allocation
    ue_slot_map: np.ndarray = field(default=None, init=False)
    # Estimation
    large_scale: np.ndarray = field(default=None, init=False)
    large_scale_hat: np.ndarray = field(default=None, init=False)
    # Communication parameters
    gen_bit: np.ndarray = field(default=None, init=False)
    queue: np.ndarray = field(default=None, init=False)
    delta_queue: np.ndarray = field(default=None, init=False)
    sinr: np.ndarray = field(default=None, init=False)
    se_set: np.ndarray = field(default=None, init=False)
    se_achievable: np.ndarray = field(default=None, init=False)
    se_oracle: np.ndarray = field(default=None, init=False)
    # ber: np.ndarray = field(default=None, init=False)
    # latency: np.ndarray = field(default=None, init=False)
    # packet_loss_rate: np.ndarray  = field(default=None, init=False)
    # Localization parameters
    # mse_position: np.ndarray = field(default=None, init=False)
    # mse_velocity: np.ndarray = field(default=None, init=False)
    # detection_probability: np.ndarray = field(default=None, init=False)
    # false_alarm_rate: np.ndarray = field(default=None, init=False)
    # range_resolution: np.ndarray = field(default=None, init=False)
    # doppler_resolution: np.ndarray = field(default=None, init=False)
    efim_oracle: np.ndarray = field(default=None, init=False)
    efim_predicted: np.ndarray = field(default=None, init=False)
    # Semantic parameters
    misclassification: np.ndarray = field(default=None, init=False)
    risk_predicted: np.ndarray = field(default=None, init=False)
    risk_predicted_full: np.ndarray = field(default=None, init=False)
    risk_hat: np.ndarray = field(default=None, init=False)

    def __post_init__(self):
        self.ue_slot_map = np.zeros((self.c.num_frames + 1, self.c.num_slots, self.c.num_ue), dtype=bool)
        self.large_scale = np.zeros((self.c.num_frames + 1, self.c.num_ap, self.c.num_ue, ))
        self.large_scale_hat = np.zeros((self.c.num_frames + 1, self.c.num_ap, self.c.num_ue))
        self.gen_bit = np.zeros((self.c.num_frames + 1, self.c.num_ue))
        self.queue = np.zeros((self.c.num_frames + 1, self.c.num_ue), dtype=np.int32)
        self.delta_queue = np.zeros((self.c.num_frames + 1, self.c.num_ue), dtype=np.int32)
        self.se_achievable = np.full((self.c.num_frames + 1, self.c.num_ue), np.nan)
        self.se_set = np.full((self.c.num_frames + 1, self.c.num_ue), np.nan)
        self.se_oracle = np.full((self.c.num_frames + 1, self.c.num_ue), np.nan)
        self.sinr = np.full((self.c.num_frames + 1, self.c.num_ue), np.nan)
        self.efim_oracle = np.zeros((self.c.num_frames + 1, self.c.num_slots, self.c.num_ue, 3, 3))
        self.efim_predicted = np.zeros((self.c.num_frames + 1, self.c.num_slots, self.c.num_ue, 3, 3))
        self.misclassification = np.zeros((self.c.num_frames + 1, self.c.num_ue), dtype=bool)
        self.risk_predicted_full = np.zeros((self.c.num_frames + 1, self.c.num_ue, 2))
        self.risk_predicted = np.zeros((self.c.num_frames + 1, self.c.num_ue))
        self.risk_hat = np.zeros((self.c.num_frames + 1, self.c.num_ue))

    def update(self, frame: int | np.ndarray = 0, idx: np.ndarray | None = None, **kwargs):
        """Update metrics for the selected frames with new values."""
        if idx is None:
            idx = np.arange(self.c.num_ue)
        for key, value in kwargs.items():
            if hasattr(self, key):
                # Get the attribute
                attr = getattr(self, key)
                # If value has not the correct shape, numpy will raise IndexError
                if key in ["se_achievable", "se_set", "se_oracle", "sinr"]:
                    attr[frame, idx] = value
                elif isinstance(attr, np.ndarray):
                    attr[frame] = value
                else:
                    # If the attribute is not an array, update it directly
                    setattr(self, key, value)
            else:
                raise AttributeError(f"EstimateParams has no attribute '{key}'")


##### Useful methods for saving/loading #####
# Check parameter file
def check_dataclass(filepath: str, parameters: type[dataclass], keys: list[str] | None = None) -> bool:
    # Adding extension
    if not filepath.endswith(".npz"):
        filepath += ".npz"

    # Check file existence
    if not os.path.exists(filepath):
        return False

    # Filter data
    param_dict = asdict(parameters)
    if keys is not None:
        filtered_dict = [(k, v) for (k, v) in zip(param_dict.keys(), param_dict.values()) if k in keys]
    else:
        filtered_dict = param_dict.items()
    #   Check data
    with np.load(filepath, allow_pickle=True) as data:
        try:
            are_equal = np.all([np.all(data[key] == value) for key, value in filtered_dict])
        except KeyError as e:
            print(f"Missing key in parameter file: {e}")
            return False
    if are_equal:
        return True
    else:
        return False


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
      - fields referencing another one of the passed dataclass instances, e.g. GeometryParams.c, are skipped.
      - genuinely unserializable live objects (rtree.index.Index, np.random.Generator) are skipped.

    :param filepath: str, output path (".npz" appended automatically if missing).
    :param dataclass_instances: dataclasses passed as name=instance, e.g.
        save_params_npz(path, shared=env.c, geo=env.geo, estimate=env.estimate, metrics=env.metrics)

    :returns: str, the actual path written (with ".npz" ensured).
    """
    instance_ids = {id(v) for v in dataclass_instances.values()}
    flat = {}

    for group_name, obj in dataclass_instances.items():
        for f in fields(obj):
            value = getattr(obj, f.name)
            key = f"{f.name}"

            if id(value) in instance_ids:
                continue  # back-reference to another saved dataclass -- skip, avoid duplication
            if value is None:
                continue
            if isinstance(value, (SharedParams, GeometryParams)):
                continue
            if isinstance(value, Rotation):
                flat[key] = value.as_quat()
            elif isinstance(value, stats.rv_continuous):
                flat[key] = value.name
            elif isinstance(value, (sha.geometry.Polygon, sha.geometry.MultiPolygon)):
                flat[key] = np.array(value.wkt)
            elif isinstance(value, (rtree.index.Index, np.random.Generator)):
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

def save_radiomap(filepath: str, rm: rt.RadioMap, ap_location: np.ndarray) -> str:
    """Extract a RadioMap's numeric data to plain NumPy arrays and save as a single .npz.
    Geometry metadata (cell_centers, cell_size) is included so the grid can be reconstructed
    for plotting/lookup later without needing the original scene or solver.

    :param filepath: str, output path (".npz" appended automatically if missing).
    :param rm: sionna.rt.RadioMap, the computed radio map to save.
    :param ap_location: np.ndarray, the location of the access points (TX) in the radio map.

    :returns: str, the actual path written.
    """
    data = {
        "ap_location": ap_location,
        "cell_centers": np.array(rm.cell_centers),
        "cell_size": np.array(rm.cell_size),
        "cells_count": np.array(rm.cells_count),
        "cells_per_dim": np.array(rm.cells_per_dim),
        "center": np.array(rm.center),
        "num_rx": rm.num_rx,
        "num_tx": rm.num_tx,
        "orientation": np.array(rm.orientation),
        "path_gain": np.array(rm.path_gain),
        "rx_cell_indices": np.array(rm.rx_cell_indices),
        "size": np.array(rm.size),
        "tx_cell_indices": np.array(rm.tx_cell_indices)
    }
    # RSS/SINR require tx power / noise assumptions baked in at query time in some
    # Sionna versions -- include them only if directly available as attributes, so a
    # version mismatch here doesn't take down the whole save.
    for attr in ("rss", "sinr"):
        if hasattr(rm, attr):
            try:
                data[attr] = np.array(getattr(rm, attr))
            except Exception:
                pass

    if not filepath.endswith(".npz"):
        filepath += ".npz"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez_compressed(filepath, **data)
    return filepath


def load_radiomap(filepath: str):
    if not filepath.endswith(".npz"):
        filepath += ".npz"
    with np.load(filepath) as data:
        FakeRadioMap = make_dataclass("FakeRadioMap", data.keys())
        return FakeRadioMap(**data)


def arrival_rate_init(avg_se: float, num_ue: int = NUM_UE, num_slots: int = NUM_SLOTS,
                      bw: float = NUM_SCS * 15e3 * 2 ** NUMEROLOGY, slot_time: float = 1e-3, rho: float= 0.75):
    lambda_max = num_slots / num_ue * avg_se * bw * slot_time
    return rho * lambda_max, 10 * lambda_max
