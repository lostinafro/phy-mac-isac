from copy import deepcopy
import numpy as np
from dataclasses import dataclass, field
import math

from .. import utils as ut
from ..modules.params import SCENE_FOLDER, DATA_FOLDER, CACHE_FOLDER
from ..modules.params import SharedParams as BaseShared
from ..modules.params import WirelessParams as BaseWireless
from ..modules.params import Metrics as BaseMetrics


## DEFAULT PARAMETERS
NUM_EP = int(1e2)
# Scenario
NUM_AP = 1
NUM_ANT = 1
NUM_UE = 100
NUM_CLUSTERS = 20
CLUSTER_SIZE = 4
NUM_REGIONS = 1
UE_HEIGHT = 1             # [m]
ROOM_SIDE = 150

# Wireless
CARRIER = 3e9       # [Hz]
NUMEROLOGY = 0
NUM_SCS = 72
NUM_SYM = 14
NUM_SLOTS = 20
NUM_FRAMES = 100
PPM_XO = 0.     # Perfect synchronization is assumed among nodes and AP
UE_POW = ut.dbm2watt(0)
TARGET_BLER = 0.05


# RayTracer and RadioMap parameters
RT_PARAM = {
    "max_depth": 3,
    "los": True,
    "specular_reflection": True,
    "diffuse_reflection": False,
    "refraction": False,
    "diffraction": False,
    "edge_diffraction": False,
    "synthetic_array": False,
    "samples_per_tx": int(5*1e8),
    "cell_size": (.5, .5),
}


## CLASSES
@dataclass
class SharedParams(BaseShared):
    """Shared parameters across different components of the simulation.

    :ivar num_ap: int, number of Access Points (APs).
    :ivar num_ant_h: int, number of horizontal antennas per AP.
    :ivar num_ant_v: int, number of vertical antennas per AP.
    :ivar num_ue: int, number of User Equipment (UE) devices.
    :ivar num_scs: int, number of subcarriers.
    :ivar num_sym: int, number of OFDM symbols per slot.
    :ivar num_slots: int, number of slots per frame.
    :ivar num_frames: int, number of frames. Frame 0 is the initial state of the environment.
    :ivar num_regions: int, number of semantic regions within the environment.
    :ivar rays: dict, ray tracing key-value parameters.
    """
    num_ap: int = NUM_AP
    num_ant_h: int = NUM_ANT
    num_ant_v: int = 1
    num_ue: int = NUM_UE
    num_scs: int = NUM_SCS
    num_sym: int = NUM_SYM
    num_slots: int = NUM_SLOTS
    num_frames: int = NUM_FRAMES
    num_regions: int = NUM_REGIONS
    rays: dict = field(default_factory=lambda: deepcopy(RT_PARAM))

    # Anomaly tracking parameters
    num_clusters: int = NUM_CLUSTERS
    cluster_size: int = CLUSTER_SIZE

    def __post_init__(self):
        super().__post_init__()

        # Force SISO communication
        if self.num_ap != 1 or self.num_ant != 1:
            raise ValueError("Anomaly tracking requires a single SISO AP")

        # Force the total number of UEs
        if self.num_ue < self.num_clusters * self.cluster_size:
            raise ValueError("The total number of UEs must be greater than or equal to the number of clusters times the cluster size")


@dataclass
class WirelessParams(BaseWireless):
    """Parameters related to wireless communication.

    :ivar numerology: 5G NR numerology index (μ).
    :ivar carrier: Carrier frequency in Hz.
    :ivar lo_ppm: Nominal part-per-million error of the transmitter's Local oscillators (LO).
    :ivar ue_power: UE available transmit power in Watts.
    :ivar queue_reliability: Reliability of queue not exceeding queue_max (e.g., 0.99 for 99% reliability).
    """
    numerology: int = NUMEROLOGY
    carrier: float = CARRIER
    lo_ppm: float | np.ndarray = PPM_XO
    ue_power: float | np.ndarray = UE_POW
    guard_time: float = 0.      # Guard time between frames [s]

    # Geometry-based TR 38.901 UMi channel and SYS abstraction settings.
    direction: str = "uplink"
    mcs_table_index: int = 1
    bler_target: float = TARGET_BLER
    force_los: bool | None = None
    enable_shadow_fading: bool = False
    # Absolute complex-CFR coefficient MSE: E[abs(h_hat - h)**2].
    ce_error_variance: float = 0.0

    # Computed attribute (excluded from __init__)
    mcs_category: int = field(init=False)

    def __post_init__(self):
        super().__post_init__()

        # Check input
        if self.direction not in ("uplink", "downlink"):
            raise ValueError("direction must be uplink or downlink")
        tables = (1, 2) if self.direction == "uplink" else (1, 2, 3, 4)
        if self.mcs_table_index not in tables or not 0 < self.bler_target < 1:
            raise ValueError("Invalid MCS table or BLER target")
        self.mcs_category = int(self.direction == "downlink")
        if not 0 < self.bler_target < 1:
            raise ValueError("bler_target must lie strictly between 0 and 1")
        if self.force_los is not None and not isinstance(self.force_los, bool):
            raise ValueError("force_los must be None or bool")
        variance = self.ce_error_variance
        if not math.isfinite(variance) or variance < 0:
            raise ValueError("channel_estimation_error_variance must be finite and nonnegative")


class Metrics(BaseMetrics):
    """Compact occupied-slot channels and PHY results for a single frame.

    Channel arrays have shape [S,1,1,1,1,T,F], where S is the number of
    occupied slots. ``slot_indices`` maps these rows to the original allocation.
    ``channel`` is physical propagation; ``channel_estimate`` includes error.
    """
    # Allocation performed
    allocation: np.ndarray = field(init=False)
    # Communication parameters
    mcs_index: np.ndarray = field(init=False)
    decoded_bits:  np.ndarray = field(init=False)
    harq_feedback:  np.ndarray = field(init=False)
    sinr_eff: np.ndarray = field(init=False)
    tbler: np.ndarray = field(init=False)
    bler: np.ndarray = field(init=False)

    def __post_init__(self):
        self.allocation = -1 * np.ones((self.c.num_frames + 1, self.c.num_slots), dtype=int)
        self.mcs_index = np.full((self.c.num_frames + 1, self.c.num_ue), np.nan)
        self.decoded_bits = np.full((self.c.num_frames + 1, self.c.num_ue), np.nan)
        self.harq_feedback = np.full((self.c.num_frames + 1, self.c.num_ue), np.nan)
        self.sinr_eff = np.full((self.c.num_frames + 1, self.c.num_ue), np.nan)
        self.tbler = np.full((self.c.num_frames + 1, self.c.num_ue), np.nan)
        self.bler = np.full((self.c.num_frames + 1, self.c.num_ue), np.nan)

    def decoded(self, frame: int) -> np.ndarray:
        """Success flags in original slot order; idle slots are False."""
        allocation_frame = self.allocation[frame]
        return self.harq_feedback[frame, allocation_frame] == 1

    def successful_ues(self, frame: int) -> np.ndarray:
        return self.allocation[frame, self.decoded(frame)]