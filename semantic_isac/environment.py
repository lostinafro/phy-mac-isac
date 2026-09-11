import threading
import os
import numpy as np
import torch
import mitsuba as mi
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
import utils as ut
import shapely as sha
import sionna.rt as rt
import sionna.phy as phy
import contextlib
import parameters as p
import resource_allocation as ra
from transmission import TransmissionBase, TransmissionAbstraction
from trackers import TrackerBase, LinearKalmanTracker
from theory import PerformanceBase, DMimoFarFieldOMA


class SimulationEnvironment:
    """
    Configures and manages the setup for a dynamic scenario in a wireless communication simulation.

    Function:
    This class provides methods to generate UE positions and velocities, check for overlapping positions, create vehicle objects,
    save vehicle data3 to a CSV file, set up a simulation scene, and retrieve scene information and velocities.

    """
    def __init__(self, episode : int,
                 num_scs: int = p.NUM_SCS,
                 num_sym: int = p.NUM_SYM,
                 num_slot: int = p.NUM_SLOTS,
                 num_frames: int = p.NUM_FRAMES,
                 numerology: int = p.NUMEROLOGY,
                 carrier_freq: float = p.CARRIER,
                 scene_name: str | None = None,
                 area_coordinates: list | np.ndarray | None = None,
                 rt_parameters: dict | None = None,
                 num_regions : int = 2,
                 boundaries: np.ndarray | None = None,
                 num_ap : int = p.NUM_AP,
                 num_ant_h : int = p.NUM_ANT, num_ant_v : int = 1,
                 num_ue : int = p.NUM_UE,
                 ue_power : float = p.UE_POW,
                 lo_ppm: float | np.ndarray = p.PPM_XO,
                 debugging: bool = False,
                 env_folder: str = p.ENV_FOLDER,
                 scene_folder: str = p.SCENE_FOLDER,
                 generate_channel: bool = False,
                 ):
        r"""Initializes the simulation environment for a wireless communication scenario.

        This method sets up the core parameters for a wireless network simulation, including:
        
        - The physical environment (area coordinates).
        - Access Points (APs) and their antenna arrays.
        - User Equipment (UEs) and their mobility (positions and velocities).
        - OFDM (Orthogonal Frequency-Division Multiplexing) parameters for the physical layer.

        :param episode: int, unique identifier for the simulation episode. Used for logging, seeding randomness,
            or distinguishing between multiple runs.

        :param num_sym: int, number of OFDM symbols per slot. Defaults to 14.
        :param num_slot: int, number of slots per frame. Defaults to 10.
        :param num_frames: int, total number of frames in the simulation. Defaults to 100.
            This is `math:K` in the paper; the internal variable is set to `math:K + 1` to account for the initial state.

        :param numerology: int, 5G NR numerology index (μ), where subcarrier spacing is 2 ** μ * 15 kHz. Defaults to 0.
        :param carrier_freq: float, carrier frequency in Hz. Defaults to ``central_freq`` (a predefined constant).

        :param area_coordinates: Coordinates defining the boundaries of the simulation area as vertices of a polygon.
            Shape: (A, 3) for a Polygon area of A vertices (e.g., [[x_min, y_min, 0], [x_max, y_max, 0]] for a rectangle).
            If None, a default area of a 200-meter side square centered on the origin is created.

        :param num_ap: int, number of Access Points (APs) in the simulation. Defaults to ``L``.

        :param num_ant_h: int, number of horizontal antennas per AP array. Defaults to 4.
        :param num_ant_v: int, number of vertical antennas per AP array. Defaults to 1.

        :param num_ue: Number of User Equipment (UE) devices. Defaults to ``U``.
        :type num_ue: int, optional
        
        :param num_scs: Number of subcarriers in the OFDM system. Defaults to 144.
        :type num_scs: int, optional

        .. note::
           - If ``pos_ap``, ``pos_ue``, or ``pos_ant`` are not provided, the method will generate default positions
             (e.g., random or grid-based) within ``area_coordinates``.
           - The ``numerology`` parameter affects the subcarrier spacing and symbol duration in 5G NR systems.
           - The ``episode`` parameter can be used to seed random number generators for reproducibility.
           - All positions (``pos_ap``, ``pos_ue``, ``pos_ant``) are assumed to be in meters.
           - Velocities (``vel_ue``) are assumed to be in meters/second.
           - ``area_coordinate`` does not really use the last dimension, but it is required to have the same shape
             as other geometric properties.
        """
        # Randomization
        self.episode = episode                          # Set episode number
        self.rng = np.random.default_rng(self.episode)  # Random number generator used everywhere
        phy.config.seed = self.episode                  # Configuring sionna.phy to use seed equal to episode no.
        self.debug = debugging                          # Boolean declaring debug session
        if self.debug:
            self.channel_fig = None
            self.env_fig = None

        # Useful folders
        self.env_folder = env_folder
        self.scene_folder = scene_folder

        os.makedirs(self.env_folder, exist_ok=True)
        os.makedirs(self.scene_folder, exist_ok=True)

        # SHARED PARAMETERS
        # General constants (+1 in num_frames accounts for the initial state being frame 0)
        sha_kwargs = {
            "num_ap": num_ap,
            "num_ue": num_ue,
            "num_regions": num_regions,
            "num_ant_h": num_ant_h,
            "num_ant_v": num_ant_v,
            "num_scs": num_scs,
            "num_sym": num_sym,
            "num_slots": num_slot,
            "num_frames": num_frames,
            "rays": rt_parameters
        }
        self.c = p.SharedParams(**sha_kwargs, rng=self.rng)       # Parameters collecting shared variables

        # WIRELESS attributes
        # Compute wireless parameters from numerology and carrier (check dataclass for help)
        wir_kwargs = {
            "numerology": numerology,
            "carrier": carrier_freq,
            "lo_ppm": lo_ppm,
            "ue_power": ue_power,
        }
        self.wireless = p.WirelessParams(**wir_kwargs, c=self.c)  # Parameters collecting wireles variables

        # GEOMETRY and SCENE
        self._scene_name = scene_name
        self.scene, area_coordinates = self._build_base_scene(scene_name, area_coordinates)
        # Compute boundaries of semantic regions
        if boundaries is None:
            # Default is splitting the area in self.c.num_regions equal regions (see __post_init__ in GeometryParams)
           semantic_regions = None
        else:
            assert boundaries.shape[0] == self.c.num_regions, "boundaries must comprises as many regions as num_regions"
            semantic_regions = sha.MultiPolygon([sha.Polygon(boundaries[i]) for i in range(self.c.num_regions)])

        # Initialize geometrical shape
        self.geo = p.GeometryParams(c=self.c,
                                    area_coordinates=area_coordinates,
                                    semantic_regions=semantic_regions)      # Parameters collecting geometric variables

        # Initialize default camera for the scene
        bbox = self.geo.environment_shape.bounds  # (min_x, min_y, max_x, max_y)
        extent = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * np.array([0, -1, 1.3])
        self.default_camera = rt.Camera(position=mi.Point3f(self.geo.center + extent),
                                        look_at=self.geo.center)

        # Initialize estimation parameters
        self.estimate = p.EstimationParams(c=self.c, geo=self.geo)   # Parameters collecting estimation variables

        # Initialize metrics parameters
        self.metrics = p.Metrics(c=self.c)      # Metrics to be saved for performance evaluation

        ## Before loading the module, check if the env for the episode has been already generated
        self.env_filename = os.path.join(self.env_folder, f"ep{self.episode:04d}.npz")
        data_check = (p.check_dataclass(self.env_filename, self.c, keys=list(sha_kwargs.keys())) and
                      p.check_dataclass(self.env_filename, self.wireless, keys=list(wir_kwargs.keys())) and
                      p.check_dataclass(self.env_filename, self.geo, keys=["area_coordinates"]))
        if data_check and not generate_channel:
            self._load_env = True
        else:
            self._load_env = False

        # RadioMap and RayTracer
        self.rm: rt.RadioMap = None             # Instance of a RadioMap. Set in _setup_scene()
        self.rm_solver = rt.RadioMapSolver()    # Solver module for RadioMap computation
        self.path_solver = rt.PathSolver()      # Solver module for path evaluation in the ray tracer

        # Resource allocation, transmission and tracker modules
        self.ra_method: ra.ResourceAllocationBase = None    # Resource allocation module, giving the allocation on a frame
        self.transmission_module: TransmissionBase = None   # Transmission module, simulating wireless signal processing
        self.tracker_module: TrackerBase = None             # Tracker module, needed for location tracking

        # Theoretical evaluation modules
        self.performance: PerformanceBase = None   # Module for theoretic performance evaluation

        # Set up the environment at frame -1 (last element for the dataclasses attributes)
        self.frame = -1             # Denote the current number of the frame
        self.frame_sync = 0         # Denote the last frame in which PRACH synchronization was performed

    def setup_modules(self,
                      ra_algorithm: ra.ResourceAllocationBase,
                      transmission_module: TransmissionBase,
                      tracker_module: TrackerBase,
                      performance: PerformanceBase):
        """Set scheduling, transmission, tracking and evaluation modules to be used."""
        self.ra_method = ra_algorithm
        self.transmission_module = transmission_module
        self.tracker_module = tracker_module
        self.performance = performance

    def setup_scenario(self,
                       ap_pos: float | np.ndarray | None = None,
                       antenna_spacing: float | None = None,
                       init_ue_pos: np.ndarray | None = None,
                       ue_vel_cov: np.ndarray | float = p.VEL_STD):
        """Set up the actual environment as it was at frame -1 (initial condition of the system).

        This method initializes the physical environment by placing Access Points (APs) and User Equipment (UEs),
        configuring antenna arrays, computing distances, setting velocity covariances, and initializing 
        estimation variables. It represents the state of the system at the beginning (frame -1).

        :param ap_pos: position(s) of the Access Points. Can be:
            - ``None``: APs are placed at default symmetric positions with height = 5 [m].
            - ``float``: z-axis height for symmetric AP placement.
            - ``np.ndarray``: Explicit (L, 3) array of AP positions.

        :param antenna_spacing: float > 0, antenna spacing in wavelengths for horizontal and vertical dimensions.
            If ``None``, defaults to half-wavelength spacing (λ/2).

        :param init_ue_pos: initial positions of the User Equipment devices. Can be:
            - ``None``: UEs are placed randomly within the area boundaries.
            - ``np.ndarray``: Explicit (U, 3) array of UE positions.

        :param ue_vel_cov: velocity covariance for the UEs. Can be:
            - ``float``: Standard deviation for a diagonal covariance matrix (σ²I) applied to all UEs.
            - ``np.ndarray``: (U, 2, 2) array of covariance matrices, one for each UE.

        :raises AssertionError: If ``ue_vel_cov`` is an ndarray but does not have shape (U, 3, 3).
        :raises AttributeError: If ``ue_vel_cov`` is not an ndarray (handled internally to create default covariance).

        :returns: None. This method updates instance attributes in place.

        .. note::
           - All positions are in meters in the absolute coordinate system.
           - Velocities and covariances are on the ground plane (2D: x, y).
           - The method initializes distances, path loss, and sensing coverage maps.
           - Estimated variables (e.g., ``hat_pos_ue``, ``cov_pos_prior``) are initialized to default values.

        .. seealso::
           :meth:`place_aps`, :meth:`ue_random_positioning`, :meth:`compute_ue_ap_distance`
        """
        ### Check input and populate geometry variables
        if self._load_env:
            with np.load(self.env_filename, allow_pickle=True) as data:
                self.geo.update(np.arange(self.K + 1), **ut.filter_kwargs(self.geo, **data))
                self.wireless.update(**ut.filter_kwargs(self.wireless, **data))
                ap_array = None
        else:
            # [λ] default half-wavelength spacing else whatever was input
            antenna_spacing = antenna_spacing or 1 / 2

            # Place APs, UEs and compute their distances
            ap_pos, ap_rot, ant_pos = self.place_aps(antenna_spacing, ap_pos)
            init_ue_pos = self.ue_random_positioning() if init_ue_pos is None else init_ue_pos

            # Set velocity covariance
            try:
                assert ue_vel_cov.shape == (self.U, 3, 3), "ue_vel_cov must have shape (U, 3, 3)"
            except AttributeError:
                # Set a standard diagonal matrix for all UE with sigma_x, sigma_y = V_max / 3 and sigma_z = 1e-9
                ue_vel_cov = ue_vel_cov ** 2 * np.repeat(np.array(np.eye(3))[np.newaxis], self.U, axis=0)
                ue_vel_cov[:, 2, 2] = 1e-9

            # Update geometry variables
            self.geo.update(self.frame,
                            ap_location=ap_pos,
                            ap_orientation=ap_rot,
                            antenna_location=ant_pos,
                            antenna_spacing=antenna_spacing,
                            ue_location=init_ue_pos,
                            ue_vel_cov=ue_vel_cov)

            # Sample LO offset at the initial frame for CTO and CFO residual baselines
            self.wireless.sample_lo_offset("gaussian", store=True)

        ### Initialize estimated variable
        # Assumption: the UE estimated position is the center of the overall area they are in. Then, the prior
        # estimate at initialization is done by generating an ellipse including each semantic region, and using it to
        # instantiate a Gaussian covariance with 99.7% confidence interval equal to the ellipse.
        mean, covariance = ut.get_gaussian_from_polygon(self.geo.environment_shape)
        init_pos_hat = np.repeat(mean[np.newaxis], self.U, axis=0)
        init_cov_hat = np.repeat(covariance[np.newaxis], self.U, axis=0)

        self.estimate.update(self.frame,
                             ue_pos_hat_prio=init_pos_hat,
                             ue_pos_cov_prio=init_cov_hat,
                             ue_pos_hat_post=init_pos_hat,
                             ue_pos_cov_post=init_cov_hat)

        # Initialize the scene
        self._setup_scene()  # Instance of sionna.rt Scene class

        if self.debug:
            self.env_fig = self.show_environment()
            self.env_fig.show()

    def step(self, cto: bool = True, cfo: bool = True, sync: bool = False):
        """Evaluates operations in a frame"""
        ### Update frame and geometry
        # Start by updating the frame number
        self.frame += 1

        # Move the UEs to a new position under Brownian motion
        new_pos, new_vel = self.move_ue()

        ## Take channel instance through Sionna.rt
        # Check if synchronization has been performed during the previous frame (all UEs)
        if sync:
            # Resample LO offset in the current frame for CTO and CFO baselines
            cto_base, cfo_base = self.wireless.sample_lo_offset("gaussian",
                                                                ppm=self.wireless.lo_ppm, store=True)
            self.frame_sync = self.frame
        else:
            cto_base, cfo_base = self.wireless.cto_base, self.wireless.cfo_base
        # Check if CTO or CFO are neglected
        if not cto:
            cto_base = np.zeros_like(cto_base)
        if not cfo:
            cfo_base = np.zeros_like(cfo_base)

        # Generate channel realization for the whole frame
        if not self._load_env:
            paths = self.path_solver(self.scene, seed=self.episode, **ut.filter_kwargs(self.path_solver, **self.c.rays))
            channel = self.compute_cfr(paths, cfo_base, cto_base)   # [L, M, U, S, T, F]
        else: # The environment is loaded from a file
            paths = None
            channel = self.wireless.channel[self.frame]

        ### Prior
        # Update through Kalman equations
        ue_pos_hat_pri, ue_pos_cov_pri = self.tracker_module.prior(self.estimate.ue_pos_hat_post[self.frame - 1],
                                                                   self.estimate.ue_pos_cov_post[self.frame - 1])

        self.estimate.update(self.frame,
                             ue_pos_hat_prio=ue_pos_hat_pri,
                             ue_pos_cov_prio=ue_pos_cov_pri)

        # Generate data at the UE and check the queue
        delta_queue = self.wireless.queue_max - self.metrics.queue[self.frame - 1]
        new_data = self.wireless.sample_new_data()
        queue = self.metrics.queue[self.frame - 1] + new_data

        ### Perform allocation based on prior
        # Evaluate common SNR related to eFIM
        snr = self.wireless.ue_power / self.wireless.nob_ap * len(self.ra_method.pilot_ofdm_symbol_indices())
        # Average energy over antennas (far-field approximation), symbols and subcarriers in a slot (quasi-static fading)
        beta = np.mean(np.abs(channel) ** 2, axis=(1, 4, 5))     # (L, U, S)

        # Compute eFIM and rate assuming perfect knowledge (oracle) (shape (S, U, d, d))
        oracle_efim = self.performance.eval_efim(snr, self.geo.ue_location[self.frame], beta.transpose())
        oracle_se = self.performance.eval_achievable_se(beta.transpose(), len(self.ra_method.pilot_ofdm_symbol_indices()))

        # Compute predicted eFIM and rate with estimated values (shape (U, d, d))
        beta_hat = self.estimate_gain_dl(self.estimate.ue_pos_hat_prio[self.frame])
        predicted_efim = self.performance.eval_efim(snr, self.estimate.ue_pos_hat_prio[self.frame], beta_hat[..., np.newaxis].transpose())
        predicted_se = self.performance.eval_achievable_se(beta_hat[..., np.newaxis].transpose(), len(self.ra_method.pilot_ofdm_symbol_indices()))

        # Compute allocation matrix based on the predictions
        kwargs = {"efim": predicted_efim, "covariance_pri": ue_pos_cov_pri, "delta_queue": delta_queue}
        ue_slot_map = self.ra_method(self.frame, **kwargs)
        # Take indexing variables
        ue_active_per_slot, slot_idx, ue_idx = self.ra_method.scheduled_ues(ue_slot_map)
        not_selected = ~np.isin(np.arange(self.U), ue_idx)
        # Compute approximate semantic risk through oracle eFIM
        risk_hat = np.zeros(self.U)
        risk0, risk1, _ = self.ra_method.eval_semantic_risk(self.frame, oracle_efim, ue_pos_cov_pri)
        risk_hat[ue_idx] = risk1[slot_idx, ue_idx]
        risk_hat[not_selected] = risk0[not_selected]
        # Compute predicted risk
        risk_predicted_full = self.ra_method.output_risks()
        risk_predicted = np.zeros(self.U)
        risk_predicted[ue_idx] = risk_predicted_full[ue_idx, 1]
        risk_predicted[not_selected] = risk_predicted_full[not_selected, 0]

        ### Simulating abstract transmission for CF-mMIMO operations
        # Transpose to put the slots as batches for the transmission module (needed for GPU enhancement)
        # Also, merge the dimensions on the antennas to consider a single receiver with distributed placement and
        # add a single UE stream dimension
        channel_cf = np.moveaxis(channel.copy(), 3, 0)  # [S, L, M, U, T, F]
        channel_cf = channel_cf.reshape(self.S, 1, self.L * self.M, self.U, 1, self.T, self.F)    # [S, 1, L*M, U, 1, T, F]

        # Apply the allocation mask
        channel_cf *= self.ra_method.broadcast_slot_ue_mask(ue_slot_map) * self.ra_method.broadcast_ap_ue_mask()

        # TODO: future should implement per-UE modulation for PHY accurate tests.
        # Perform (abstracted) usual centralized CF-mMIMO transmission
        sinr_eff, se_eff = self.transmission_module(channel_cf)

        ### Update queues
        # Checking SE achievability
        se_achievable = se_eff[slot_idx, ue_idx]
        se_per_scheduled_ue = predicted_se[ue_idx]
        se_per_scheduled_ue[se_per_scheduled_ue > se_achievable] = 0.

        # Taking the amount of bits transmitted and update the queue
        bits_tx = np.asarray(np.floor(se_per_scheduled_ue * self.wireless.bw * self.wireless.slot_time), dtype=np.int32)
        queue[ue_idx] = np.maximum(queue[ue_idx] - bits_tx, np.zeros_like(queue[ue_idx]))

        # Create the observation for the scheduled UEs (assuming asymptotic performance are achievable)
        asymptotic_cov = np.linalg.inv(oracle_efim[ue_slot_map])
        obs_vec = ut.sample_multivariate_gaussian(cov=asymptotic_cov, mean=self.geo.ue_location[self.frame, ue_idx],
                                                  rng=self.rng)

        ### Posterior
        # Compute posterior for UE observed
        ue_pos_hat_post, ue_pos_cov_post = self.tracker_module.posterior(obs_vec, ue_pos_hat_pri, ue_pos_cov_pri, ue_idx)
        self.estimate.update(self.frame,
                             ue_pos_hat_post=ue_pos_hat_post,
                             ue_pos_cov_post=ue_pos_cov_post)

        ### Evaluate Metrics
        self.metrics.update(self.frame, ue_idx,
                            queue=queue,
                            delta_queue=delta_queue,
                            large_scale=beta.mean(axis=-1),
                            large_scale_hat=beta_hat,
                            sinr=sinr_eff.mean(axis=-1)[slot_idx, ue_idx],
                            se_set=se_per_scheduled_ue,
                            se_achievable=se_achievable,
                            efim_oracle=oracle_efim,
                            efim_predicted=predicted_efim,
                            misclassification=self.geo.ue_region[self.frame] != self.estimate.ue_region_hat[self.frame],
                            risk_predicted_full=risk_predicted_full,
                            risk_predicted=risk_predicted,
                            risk_hat=risk_hat,
                            ue_slot_map=ue_slot_map,
                            )
        # Print errors
        if self.debug and not (self.frame % 1):
            # Visualize
            self.env_fig = self.show_environment(paths=paths)
            self.channel_fig = self.plot_channel(channel)
            self.env_fig.show()
            self.channel_fig.show()

            # UE movement
            for u in range(min(10, self.U)):
                print(f"UE {u}: old_pos={self.geo.ue_location[self.frame - 1, u]}, new_pos={new_pos[u]}, new_vel={new_vel[u]}")

            # Test quasi-constant beta
            beta_full = np.abs(channel) ** 2
            mask = np.nonzero(np.any(np.abs(beta_full - beta[:, np.newaxis, :, :, np.newaxis, np.newaxis]) > 1e-6, axis=(1, 4, 5)))
            if len(mask[0]) > 0:
                for idx in range(len(mask[0])):
                    print(f"Not quadi-constant beta for AP {mask[0][idx]}, UE {mask[1][idx]} and slot {mask[2][idx]}")

            # Compute absolute and relative error between RadioMap and RayTracer as metrics
            beta_gt = beta.mean(axis=-1)
            beta_rm = self.estimate_gain_dl(self.geo.ue_location[self.frame])
            print(f"Absolute RM error is: {beta_rm - beta_gt} [dB]")
            print(f"Relative EM error is: {(beta_rm - beta_gt) / beta_gt * 100} %")

            # Check issues in the oracle if any
            unachievable_err = np.nonzero(se_achievable < oracle_se[slot_idx, ue_idx])[0]
            if len(unachievable_err) > 0:
                print(f"Unachievable oracle rate for UE {unachievable_err}")
            else:
                print(f"Achievable oracle rate for all UEs")

            # Queue statuts
            queue_exceeded = np.nonzero(queue > self.wireless.queue_max)
            if len(queue_exceeded) > 0:
                print(f"Queue exceeded for UE {queue_exceeded}")
            else:
                print(f"Queue within limits for all UEs")

            # Semantic errors
            print(f"The current region error is: {self.geo.ue_region[self.frame] == self.estimate.ue_region_hat[self.frame]}")

        if (not self._load_env) and (self.frame == self.K - 1):    # Last frame
            p.save_params_npz(self.env_filename, sha=self.c, geo=self.geo, wir=self.wireless)
            print(f"Environment saved into file: {self.env_filename}")

        # Check feasibility of the current allocation
        if np.any(np.sum(ue_slot_map, axis=-1) > 1):
            # print("Allocation is not OMA")
            return False
        if np.any(np.nonzero(queue > self.wireless.queue_max)):
            # print("Queue is exceeding its maximum for at least one UE")
            return False
        return True

    def show_environment(self, camera: rt.Camera | None = None, show_devices: bool = True,
                         paths: rt.Paths | None = None, figsize: tuple = (14, 7)) -> plt.Figure:
        """Combine the top-down environment/region view and the Mitsuba ray-traced
        rendering into a single Figure with two side-by-side panels.

        :param camera: sionna.rt.Camera, If None, uses `self.default_camera`.
        :param show_devices: bool, whether to show UEs/APs in the render. Defaults to True.
        :param paths: rt.Paths, propagation paths to overlay on the render.
        :param figsize: size of the combined figure.

        :returns: the combined matplotlib Figure (not shown automatically -- call
            `fig.show()`, `plt.show()`, or `fig.savefig(...)` on the result).
        """
        fig, (ax_geo, ax_render) = plt.subplots(1, 2, figsize=figsize)

        # Left panel: top-down environment/region view, drawn directly into ax_geo
        self.geo.show(frame=self.frame, ax=ax_geo)
        ax_geo.set_title("Environment & regions (From above)")

        # Right panel: Mitsuba ray-traced rendering
        if hasattr(self, 'scene'):
            camera = camera or self.default_camera
            render_kwargs = dict(camera=camera, show_devices=show_devices,
                                 resolution=(1024, 1024), return_bitmap=True)
            if paths is not None:
                render_kwargs["paths"] = paths

            # Render as bitmap
            image = self.scene.render(**render_kwargs)
            to_show = image.convert(component_format=mi.Struct.Type.UInt8, srgb_gamma=True)
            ax_render.imshow(to_show)
            ax_render.set_title("Ray-traced scene")
            ax_render.axis("off")
        else:
            ax_render.text(0.5, 0.5, "No scene available for rendering.\nCall _setup_scene() first.",
                           ha="center", va="center", transform=ax_render.transAxes)
            ax_render.axis("off")
        return fig

    def plot_channel(self, channel: np.ndarray,  ap_idx: int = 0, ant_idx: int = 0, ue_idx: int = 0,
                     tol: float = 1e-2) -> plt.Figure:
        """Plot the channel of a single ap-ant-ue for the current frame as a sanity check."""

        # Print four equispaced subcarriers
        subcarriers = np.linspace(0, self.F, 4, dtype=np.int32, endpoint=False)
        freqs_ghz = (self.wireless.freqs + self.wireless.carrier) * 1e-9      # [GHz]
        # Print the channel for first, and last symbol of the first and last slot
        timesteps = [0, self.T - 1, (self.S - 1) * self.T - 1, self.S * self.T - 1]
        np.hstack([self.wireless.slot_index[s] + np.array([0, self.T - 1]) for s in [0, self.S - 1]])
        times_ms = self.wireless.times * 1e3    # [ms]

        # Channel dimensions are: [L, M, U, S, T, F]
        h = channel[ap_idx, ant_idx, ue_idx].reshape(self.S * self.T, self.F)
        power = 10 * np.log10(np.abs(h) ** 2)
        phase = np.angle(h)

        # Beautify the slots
        slot_color = [ut.shade_of("blue", rng=self.rng) for _ in range(self.S)]
        slot_boundaries = (self.wireless.slot_index + np.array([[0, self.T - 1]]).T).T * self.wireless.sym_time * 1e3  # [ms]

        # Beautify the RBs
        num_rbs = np.ceil(self.F / 12).astype(int)
        rb_color = [ut.shade_of("red", rng=self.rng) for _ in range(num_rbs)]
        rb_inits = np.arange(0, self.F, 12)
        rb_ends = np.arange(11, self.F, 12)
        if rb_ends[-1] < self.F - 1:
            rb_ends = np.append(rb_ends, self.F - 1)
        rb_boundaries = np.vstack((freqs_ghz[rb_inits], freqs_ghz[rb_ends])).T

        # Plot
        fig, axs = plt.subplots(4, 4, figsize=(17, 13))
        fig.suptitle(f"Channel between AP {ap_idx} (antenna {ant_idx}) and UE {ue_idx} in frame {self.frame}")

        # Plots vs time
        for i, j in enumerate(subcarriers):
            # Power
            axs[0, i].plot(times_ms, power[:, j])
            axs[0, i].set_xlabel("Time [ms]")
            axs[0, i].set_ylabel(r"Power, $|h(f,t)|^2$ [dB]")
            axs[0, i].set_title(f"Subcarrier: {j}")
            axs[0, i].grid(True)
            if np.abs(power[:, j].max() - power[:, j].min()) <= tol:    # Power is constant
                axs[0, i].set_ylim(power[:, j].mean() + np.array([-6, 6]))

            # Phase
            axs[1, i].plot(times_ms, phase[:, j])
            axs[1, i].set_xlabel("Time [ms]")
            axs[1, i].set_ylabel(r"Phase, $\angle h(f,t)$")
            axs[1, i].set_ylim(-np.pi, np.pi)
            axs[1, i].set_title(f"Subcarrier {j}")
            axs[1, i].grid(True)

            # Final annotation
            for s in np.arange(self.S):
                axs[0, i].fill_betweenx(axs[0, i].get_ybound(), slot_boundaries[s, 0], slot_boundaries[s, 1],
                                        facecolor=slot_color[s], alpha=0.2)
                axs[1, i].fill_betweenx(axs[1, i].get_ybound(), slot_boundaries[s, 0], slot_boundaries[s, 1],
                                        facecolor=slot_color[s], alpha=0.2)

        # Plot vs frequencies
        for i, j in enumerate(timesteps):
            # Power
            axs[2, i].plot(freqs_ghz, power[j, :])
            axs[2, i].set_xlabel("Subcarrier [GHz]")
            axs[2, i].set_ylabel(r"Power, $|h(f,t)|^2$ [dB]")
            axs[2, i].set_title(f"Symbol {j}")
            axs[2, i].grid(True)
            if np.abs(power[j, :].max() - power[j, :].min()) <= tol:    # Power is constant
                axs[2, i].set_ylim(power[j, :].mean() + np.array([-6, 6]))

            # Phase
            axs[3, i].plot(freqs_ghz, phase[j, :])
            axs[3, i].set_xlabel("Subcarrier [GHz]")
            axs[3, i].set_ylabel(r"Phase, $\angle h(f,t)$")
            axs[3, i].set_ylim(-np.pi, np.pi)
            axs[3, i].set_title(f"Symbol {j}")
            axs[3, i].grid(True)

            # Last annotations
            for b in range(num_rbs):
                axs[2, i].fill_betweenx(axs[2, i].get_ybound(), rb_boundaries[b, 0], rb_boundaries[b, 1],
                                        facecolor=rb_color[b], alpha=0.2)
                axs[3, i].fill_betweenx(axs[3, i].get_ybound(), rb_boundaries[b, 0], rb_boundaries[b, 1],
                                        facecolor=rb_color[b], alpha=0.2)

        fig.tight_layout()
        return fig

    def place_aps(self,
                  antenna_spacing : float,
                  ap_location : float | np.ndarray | None = None) -> tuple:
        """Generate APs positions.
        Returning the position of the APs and their antennas assuming symmetric antenna positioning

        :param antenna_spacing: float, it represents the inter-element distance in a symmetric placement
        :param ap_location: float: z-axis position of the APs; or ndarray: absolute position of the center of the array (AP)

        :returns: tuple, containing the AP positions (np.ndarray), AP rotation class, antenna positions (np.ndarray), and sionna.rt PlanarArray
        """
        try:
            assert ap_location.shape == (self.L, 3), "AP position must be of shape (L, 3)"
            ap_cen_pos = ap_location
        except AttributeError:
            # If location is not given, default is a symmetric manner with height = 5 m.
            ap_height = (ap_location or 5.) * np.ones((self.L,))

            # Place L APs regularly spaced along the environment's perimeter
            boundary = self.geo.environment_shape.exterior
            perimeter = boundary.length
            # Half-step offset so APs are centered along each "side" segment
            offset = perimeter / (2 * self.L)
            distances = np.linspace(0, perimeter, self.L, endpoint=False) + offset

            # Set the position (already 3D) and add the height
            ap_cen_pos = np.array([boundary.interpolate(d).coords[0] for d in distances])
            ap_cen_pos[:, 2] = ap_height

        # Create the base array through Sionna built-in method
        ap_array = self._default_mimo_array(antenna_spacing)

        ## Point the array to the center of the environment
        # Direction vector from the array to the center
        direction = self.geo.center - ap_cen_pos
        direction = direction / np.linalg.norm(direction, axis=1)[:, np.newaxis]  # Normalize

        # Default boresight in Sionna is +Z axis: [0, 0, 1]
        boresight = np.array([0, 0, 1])

        # Compute the rotation matrix to align boresight with the direction using the Rodrigues' rotation formula
        # Axis of rotation: cross product of boresight and direction
        axis = np.cross(boresight, direction)
        axis = axis / np.linalg.norm(axis)  # Normalize
        # Angle of rotation: angle between boresight and direction
        angle = np.arccos(np.clip(direction @ boresight[:, np.newaxis], -1.0, 1.0))
        # Create rotation matrix
        ap_rotation = Rotation.from_rotvec(axis * angle)

        ant_relative_positions = np.ones((self.L, self.M, 3))
        # For each array, rotate it to face the center using sionna built-in methods with euler angles from rotation class
        for l in range(self.L):
            ant_relative_positions[l, :, :] = np.array(ap_array.rotate(self.wireless.wavelength, mi.Point3f(ap_rotation.as_euler('zyx')[l]))).T

        # Write antenna position coordinates in absolute coordinate system
        ap_ant_pos = ant_relative_positions + ap_cen_pos[:, np.newaxis, :]

        # Return tuple
        return ap_cen_pos, ap_rotation, ap_ant_pos

    def ue_random_positioning(self, height: float | np.ndarray = p.UE_HEIGHT,
                              min_dist: np.ndarray = np.ones(2), margin: float = 0.95) -> np.ndarray:
        """Generate UE random positions with minimum distance >= min_dist.
        Modified version of https://stackoverflow.com/questions/27499139/how-can-i-set-a-minimum-distance-constraint-for-generating-points-with-numpy-ran

        :param height: float or np.ndarray, height of the UEs
        :param min_dist: np.ndarray, minimum distance between UEs in x and y direction
        :param margin: float, margin from the environment to avoid UE on the boundaries of the environment.

        :returns np.array: positions of UEs
        """
        if np.isscalar(height):
            height = height * np.ones((self.U, 1))

        shrunk = self.geo.environment_shape.buffer(-(1 - margin) * np.sqrt(self.geo.environment_shape.area))
        centers, cells = ut.lloyd_relaxation(shrunk, self.U, rng=self.rng)

        coords = np.empty_like(centers)
        for i, (pt, cell) in enumerate(zip(centers, cells)):
            minx, miny, maxx, maxy = cell.bounds
            max_shift = np.array([max((maxx - minx) / 2 - min_dist[0], 0),
                                  max((maxy - miny) / 2 - min_dist[1], 0),
                        ])
            # Reject-resample until the perturbed point stays inside its own cell
            for _ in range(10):
                noise = self.rng.uniform(-max_shift, max_shift)
                cand = pt + noise
                if cell.contains(sha.geometry.Point(cand)):
                    coords[i] = cand
                    break
            else:
                coords[i] = pt  # fall back to unperturbed center

        # Permute users for index randomization
        return self.rng.permutation(np.hstack((coords, height)))

    def move_ue(self) -> tuple:
        """Advance UEs by one time step with a linear mobility model, updating both the
        stored geometry and the Sionna scene (position + velocity of each Tx, for Doppler).

        :returns: tuple (new_position, new_velocity), each of shape (U, 3).
        """
        # Take previous position
        old_position = self.geo.ue_location[self.frame - 1]

        if self._load_env:
            new_position = self.geo.ue_location[self.frame]
            new_velocity = self.geo.ue_velocity[self.frame]
        else:
            # Non-zero only on x, y (see sample_velocities)
            new_velocity = ut.sample_multivariate_gaussian(self.geo.ue_vel_cov, rng=self.rng)    # self.sample_velocities()
            new_position = old_position + new_velocity * self.wireless.frame_time

            # Clip new position within the environment boundaries
            idx = np.array([self.geo.environment_shape.contains(sha.geometry.Point(p)) for p in new_position])
            new_position[~idx] = old_position[~idx]

            # Advance the frame counter, then store the new state there
            self.geo.update(self.frame, ue_location=new_position, ue_velocity=new_velocity)

        # Reflect the update in the scene
        for u in range(self.U):
            tx = self.scene.get(f"ue_{u}")
            tx.position = new_position[u]
            tx.velocity = new_velocity[u]

        return new_position, new_velocity

    def compute_cfr(self, paths: rt.Paths,
                    fo: np.ndarray | None = None, to: np.ndarray | None = None) -> np.ndarray:
        """Compute the channel frequency response from `paths`, accounting for UE mobility (Doppler-driven), then apply
        timing offset (TO) and frequency offset (FO), assuming ISI and ICI are negligible.

        :param paths: sionna.rt.Paths, output of raytrace_channel().
        :param fo: np.ndarray, shape (U,), frequency offset [Hz]. Default: 0.
        :param to: np.ndarray, shape (U,), timing offset [s] per OFDM symbol. Default: 0.

        :returns: np.ndarray, complex CFR of shape (L, M, U, S, T, F).
        """
        fo = fo if fo is not None else np.zeros(self.U)
        to = to if to is not None else np.zeros(self.U)
        # Check that to and fo are small enough to have ICI negligible
        if np.any((fo / self.wireless.sc_spacing) > 0.1):
            idx = ((fo / self.wireless.sc_spacing) > 0.1).nonzero()
            ue_list = np.arange(self.U)[idx].tolist()
            print(f"Warning: the ICI is not negligible for UEs in {ue_list}.")

        num_time_steps = self.S * self.T

        # [L, M, U, 1, S*T, F] -- singleton axis is num_tx_ant (UEs are single-antenna)
        h = paths.cfr(frequencies=self.wireless.freqs,
                      sampling_frequency=1 / self.wireless.sym_time,
                      num_time_steps=num_time_steps,
                      normalize_delays=False,
                      out_type="numpy")
        # Squeeze num_tx_ant axis
        h = np.squeeze(h, axis=3) # [L, M, U, S*T, F]

        # Apply time and frequency offsets assuming ISI and ICI are negligible
        # Start by checking the drift of the CTO through time, considering synchronization was made at frame 0.
        to_in_time = to[:, np.newaxis] *  (self.wireless.sym_index + (self.S * self.T * (self.frame - 1)))  # (U, S*T)
        # Check if the TO is larger than the CP generating non-negligible ISI in theory
        if np.any(to_in_time > self.wireless.cp_time):
            idx = (to_in_time > self.wireless.cp_time).nonzero()
            ue_list = np.arange(self.U)[idx].tolist()
            print(f"\tWarning: the ISI is not negligible for UEs in {ue_list}.")

        to_over_freqs = to_in_time[:, :, np.newaxis] * self.wireless.freqs[np.newaxis, np.newaxis, :]   # (U, S*T, F)
        # Compute effect of CFO
        fo_in_time = fo[:, np.newaxis] * (self.wireless.sym_index + (self.S * self.T * (self.frame - 1))) * self.wireless.sym_time # (U, S*T)
        cross_fo_to = fo[:, np.newaxis] * to_in_time    # (U, S*T)

        # Put into the exponential and reshape to [L, M, U, S * T, F]
        to_applied = np.exp(-2j * np.pi * to_over_freqs[np.newaxis, np.newaxis, :, :, :])
        fo_applied = np.exp(2j * np.pi * (fo_in_time + cross_fo_to)[np.newaxis, np.newaxis, :, :, np.newaxis])

        h *= to_applied * fo_applied

        return h.reshape(self.L, self.M, self.U, self.S, self.T, self.F)  # [L, M, U, S, T, F]

    def estimate_gain_dl(self, positions: np.ndarray) -> np.ndarray:
        """Estimate the large-scale path gain (linear scale) between each given point and
        every AP, using the AP-side radio map computed in `_setup_scene` (`self.rm`).

        :param positions: np.ndarray, shape (U, 3), query (x, y, z) positions. Only (x, y)
            is used -- the radio map is evaluated on a fixed horizontal measurement plane
            (see the `center`/`orientation` passed to `rm_solver`), so z is ignored.

        :returns: np.ndarray, shape (L, U), path gain (linear scale) between UE u and AP l.
        """
        positions = np.asarray(positions)

        # Flatten the radio map's spatial grid; cell_centers and path_gain share the same
        # spatial ordering (they describe the same grid), so a shared flat index into both
        # is consistent regardless of the underlying grid layout convention.
        centers = np.array(self.rm.cell_centers).reshape(-1, 3)  # (num_cells, 3)
        gain = np.array(self.rm.path_gain).reshape(self.L, -1)  # (L, num_cells)

        diff = centers[np.newaxis, :, :2] - positions[:, np.newaxis, :2]  # (U, num_cells, 2)
        nearest_cell = np.argmin(np.sum(diff ** 2, axis=-1), axis=1)  # (U,)

        return gain[:, nearest_cell]  # (L, U)

    def estimate_gain_ul(self, rm: rt.RadioMap, positions: np.ndarray) -> np.ndarray:
        """Estimate the large-scale path gain (linear scale) between each given point and
        every AP, using the AP-side radio map computed in `_setup_scene` (`self.rm`).

        :param positions: np.ndarray, shape (U, 3), query (x, y, z) positions. Only (x, y)
            is used -- the radio map is evaluated on a fixed horizontal measurement plane
            (see the `center`/`orientation` passed to `rm_solver`), so z is ignored.

        :returns: np.ndarray, shape (L, U), path gain (linear scale) between UE u and AP l.
        """
        positions = np.asarray(positions)

        # Flatten the radio map's spatial grid; cell_centers and path_gain share the same
        # spatial ordering (they describe the same grid), so a shared flat index into both
        # is consistent regardless of the underlying grid layout convention.
        centers = np.array(rm.cell_centers).reshape(-1, 3)  # (num_cells, 3)
        gain = np.array(rm.path_gain).reshape(self.U, -1)  # (L, num_cells)

        diff = centers[np.newaxis, :, :2] - positions[:, np.newaxis, :2]  # (U, num_cells, 2)
        nearest_cell = np.argmin(np.sum(diff ** 2, axis=-1), axis=1)  # (U,)

        return gain[:, nearest_cell].transpose()  # (L, U)

    def _setup_scene(self):
        """Populate the Sionna RT scene with UEs (single-antenna transmitters) and APs (multi-antenna receivers)
        at their current (``self.frame``) location, and compute an AP-side radio map (path gain) using an independently
        rebuilt scene with the APs as transmitters.
        """
        # Single-antenna array shared by all UEs (transmitters)
        self.scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                             pattern="iso", polarization="V")

        # Multi-antenna array shared by all APs (receivers)
        self.scene.rx_array = self._default_mimo_array(self.geo.antenna_spacing)

        # Add UEs as transmitters at their current location
        for u in range(self.U):
            self.scene.add(rt.Transmitter(name=f"ue_{u}", position=self.geo.ue_location[self.frame, u],
                                          power_dbm=self.wireless.ue_power_dbm[u]))

        # Add APs as receivers, oriented per ap_orientation (z-boresight convention, see place_aps)
        for l in range(self.L):
            orientation = self.geo.ap_orientation[l].as_euler('zyx')
            self.scene.add(rt.Receiver(name=f"ap_{l}",
                                       position=self.geo.ap_location[l],
                                       orientation=orientation))

        # Check if a radio map with the same AP position is available
        scene_name = self._scene_name or "default_room"
        rm_filename = os.path.join(self.scene_folder, "rm", f"{scene_name}_AP{self.c.num_ap}_M{self.c.num_ant}")
        if p.check_dataclass(rm_filename, self.geo, keys=["ap_location"]):
            self.rm = p.load_radiomap(rm_filename)
        else:
            # Copy scene to compute RadioMap considering AP as transmitters (working due to symmetric UL/DL channel)
            scene_for_rm = self._build_base_scene(self._scene_name)[0]
            scene_for_rm.tx_array = self.scene.tx_array
            for l in range(self.L):
                scene_for_rm.add(rt.Transmitter(name=f"ap_{l}",
                                                position=self.geo.ap_location[l],
                                                orientation=self.geo.ap_orientation[l].as_euler('zyx')))

            # Set targets/cars
            if self.c.num_car:
                # Start car setup
                car_material = rt.ITURadioMaterial("car-material",
                                                "metal",
                                                thickness=0.01,
                                                color=(0.8, 0.1, 0.1))
                # TODO: randomize positions
                cars = [rt.SceneObject(fname=rt.scene.low_poly_car, name=f"car_{i}", radio_material=car_material) for i in range(self.c.num_car)]
                self.scene.edit(add=cars)

            # Setting some default scattering coefficients and patterns for the scene
            if self.c.rays["diffuse_reflection"]:
                for material in self.scene.radio_materials.keys():
                    try:
                        self.scene.radio_materials[material].scattering_coefficient = p.scattering_coefficient_dict[material]
                        self.scene.radio_materials[material].scattering_pattern = p.scattering_pattern_dict[material]
                    except (AttributeError, KeyError):
                        print(f"Scattering coefficient and pattern not supported for {material}. "
                              f"Scattering is disabled for this material.")

            # Compute RadioMap
            self.rm = self.rm_solver(scene_for_rm,
                                     seed=self.episode,
                                     **ut.filter_kwargs(self.rm_solver, **self.c.rays))
            p.save_radiomap(rm_filename, self.rm, self.geo.ap_location)

    def _build_base_scene(self, scene_name: str | None,
                          area_coordinates: list | np.ndarray | None = None) -> tuple[rt.Scene, list | np.ndarray]:
        """Build the base propagation scene (loaded from file, or a bare floor), with
        frequency/bandwidth set -- independent of any Tx/Rx additions. Called twice: once
        for `self.scene`, once (fresh) for the AP-side radio map scene in `_setup_scene`.
        """
        # Check for scene_name first, then area_coordinates.
        if scene_name:
            try:
                scene = rt.load_scene(os.path.join(self.scene_folder, scene_name))
            except FileNotFoundError:
                print(f"Scene {scene_name} not found. Checking if a default scene has the same name.")
                scene = rt.load_scene(getattr(rt.scene, scene_name))
                print(f"Default scene {scene_name} loaded.")
            except Exception as e:
                raise e
                # TODO: the following works on the ConvexHull, while it should account for holes in concave polygon.
            area_coordinates = ut.scene_area_coordinates(scene)  # Generate base polygon coordinates from scene
        # Generate default area coordinate if reached here
        else:
            if area_coordinates is None:
                # Default is a square room
                first_quadrant = p.ROOM_SIDE / 2 * np.array([1, 1, 0])
                # Order is left-top first, then clockwise.
                area_coordinates = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]]) * first_quadrant
            scene = rt.load_scene()  # Empty sionna.rt scene
            # Generate just the floor by meshing the Polygon of the Area
            floor_material = rt.ITURadioMaterial("floor-material", "concrete",
                                                 thickness=0.2, color=(0.1, 0.1, 0.1))
            floor_path = self._build_default_floor_mesh(area_coordinates)  # already cached to disk after first call
            floor = rt.SceneObject(fname=floor_path, name="floor", radio_material=floor_material)
            scene.edit(add=[floor])

            scene.frequency = self.wireless.carrier
            scene.bandwidth = self.wireless.freqs[-1] - self.wireless.freqs[0]
        return scene, area_coordinates

    def _build_default_floor_mesh(self, area_coordinates: list | np.ndarray) -> str:
        """Triangulate a polygon generated from `area_coordinates` and export it as a flat mesh (.ply) at
        z=0, so it can be loaded as a Sionna RT floor SceneObject. Requires `trimesh`.

        :param area_coordinates: list or np.ndarray, vertex coordinates of the area to triangulate.
        :returns: str, path to the exported .ply mesh.
        """
        # TODO: move this function into utils and specify square size in the name
        # Check if meshes already exist first
        mesh_path = os.path.join(self.scene_folder, "meshes", "_default_room_floor.ply")
        if os.path.exists(mesh_path):
            return mesh_path
        else:
            mesh = ut.build_floor_mesh(area_coordinates)

            # Generate the folder if not there
            os.makedirs(os.path.join(p.MESH_FOLDER), exist_ok=True)
            # Export and return mesh
            mesh.export(mesh_path)
            return mesh_path

    def _default_mimo_array(self, antenna_spacing: float):
        return rt.PlanarArray(num_rows=self.M_h,
                              num_cols=self.M_v,
                              vertical_spacing=antenna_spacing,
                              horizontal_spacing=antenna_spacing,
                              pattern="hw_dipole",
                              polarization="V")

    @property
    def L(self) -> int:
        return self.c.num_ap

    @property
    def M_h(self) -> int:
        return self.c.num_ant_h

    @property
    def M_v(self) -> int:
        return self.c.num_ant_v

    @property
    def M(self) -> int:
        return self.c.num_ant

    @property
    def U(self) -> int:
        return self.c.num_ue

    @property
    def F(self) -> int:
        return self.c.num_scs

    @property
    def T(self) -> int:
        return self.c.num_sym

    @property
    def S(self) -> int:
        return self.c.num_slots

    @property
    def K(self) -> int:
        return self.c.num_frames

    @property
    def mib_tensor(self):
        return self.U * self.S * self.L * self.M * self.T * self.F / 1024 ** 2





def run_episode(episode: int,
                ra_method: type[ra.ResourceAllocationBase],
                ra_kwargs: dict | None = None,
                env_kwargs: dict | None = None,
                step_kwargs: dict | None = None,
                gpu_guard: threading.Semaphore | None = None,
                save_results: bool = True,
                results_folder: str = p.DATA_FOLDER,
                verbose: bool = False) -> dict:
    """Run one full simulation episode: build the environment, wire up the modules, step
    through every frame, then save the whole run's Shared/Geometric/Estimation/Metrics
    parameters to a single .npz.

    :param episode: int, episode index -- seeds the environment's RNG, and disambiguates
        this run's output filename (must be the first positional argument).
    :param ra_method: type[ResourceAllocationBase], a class from resource_allocation.py
        (not an instance -- it's constructed here once env.c/wireless/geo/estimate exist),
        e.g. `ra.RoundRobinAllocation` (must be the second positional argument).
    :param ra_kwargs: dict | None, extra kwargs forwarded to `ra_method`'s constructor.
    :param env_kwargs: dict | None, kwargs forwarded to `SimulationEnvironment.__init__`
        (num_ue, num_ap, area_coordinates, lo_ppm, ue_power, ...). `episode` is filled in
        automatically -- don't include it here.
    :param step_kwargs: dict | None, kwargs forwarded to every `env.step(...)` call
        (cto, cfo, sync).

    :param gpu_guard: threading.Semaphore, set to the GPU access to grant.
    :param save_results: bool, if False, skip saving (useful for quick interactive runs).
    :param results_folder: str, directory the output .npz is written to.
    :param verbose: bool, print a per-frame progress bar.

    :returns: dict, small scalar summary (mean over frames) of a few key metrics, plus the
        saved file path under "save_path" (None if save=False). Useful for a parameter-sweep
        script to aggregate across episodes without re-reading every .npz.
    """
    ### Set up machine variables
    on_gpu = (gpu_guard is not None) and torch.cuda.is_available()
    guard = gpu_guard if on_gpu else contextlib.nullcontext()
    device = torch.device("cuda") if on_gpu else torch.device("cpu")

    # Take the kwargs
    ra_kwargs = ra_kwargs or {}
    env_kwargs = env_kwargs or {}
    step_kwargs = step_kwargs or {}

    with guard:
        # --- Build environment ---
        env = SimulationEnvironment(episode, **env_kwargs)
        env.setup_scenario()

        # --- Build and wire up modules ---
        ra_module = ra_method(env.c, env.wireless, env.geo, env.estimate, **ra_kwargs)
        tx_module = TransmissionAbstraction(env.wireless, ra_module, device)

        trans_mat = np.eye(3)[np.newaxis, :, :].repeat(env.U, axis=0)
        obs_mat = np.eye(3)[np.newaxis, :, :].repeat(env.U, axis=0)
        proc_noise_cov = env.geo.ue_vel_cov * env.wireless.frame_time
        obs_noise_cov = np.eye(3)[np.newaxis, :, :].repeat(env.U, axis=0)
        tracker = LinearKalmanTracker(trans_mat, obs_mat, proc_noise_cov, obs_noise_cov)

        performance = DMimoFarFieldOMA(env.c, env.wireless, env.geo, env.estimate)

        env.setup_modules(ra_module, tx_module, tracker, performance)

        # --- Run the simulation ---
        frame_iter = ut.std_bar(range(env.c.num_frames)) if verbose else range(env.c.num_frames)
        for _ in frame_iter:
            env.step(**step_kwargs)

        # --- Save the whole run: Estimation, and Metrics parameters ---
        if save_results:
            ra_method_label = next(key for key, val in ra.available_methods.items() if val == ra_method)
            filename = os.path.join(results_folder, f"{ra_method_label}_ep{episode:04d}.npz")
            save_path = p.save_params_npz(filename, estimate=env.estimate, metrics=env.metrics)

        # --- Small scalar summary for quick cross-episode aggregation ---
        summary = {
            "save_path": save_path,
            "avg_se_achievable": np.nanmean(env.metrics.se_achievable[:-1], axis=0),
            "avg_queue": np.nanmean(env.metrics.queue[:-1], axis=0),
            "avg_misclassification": np.nanmean(env.metrics.misclassification[:-1], axis=0),
            "avg_risk_predicted": np.nanmean(env.metrics.risk_predicted[:-1], axis=0),
            "avg_risk_hat": np.nanmean(env.metrics.risk_hat[:-1], axis=0),
        }
        return summary

if __name__ == "__main__":
    num_ue = 10
    env_folder = os.path.join(p.ENV_FOLDER, f"UE{num_ue:03d}")
    env_args = {"debugging": False, "num_frames": 5, "num_ue": 10, "env_folder": env_folder}
    step_args = {"cfo": False}
    summary = run_episode(1, ra.SemanticRiskAllocation, gpu_guard=threading.Semaphore(1), env_kwargs=env_args, step_kwargs=step_args, verbose=True,
                          results_folder=os.path.join(os.getcwd(), "data_test"))
    print(summary)