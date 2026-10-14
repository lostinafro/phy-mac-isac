from copy import deepcopy
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field

from .. import utils as ut
from ..modules.params import SCENE_FOLDER, DATA_FOLDER
from ..modules.params import SharedParams as BaseShared
from ..modules.params import WirelessParams as BaseWireless
from ..modules.params import Metrics as BaseMetrics

## DEFAULT PARAMETERS
NUM_EP = int(1e3)
# Scenario
NUM_AP = 3
NUM_ANT = 2
NUM_UE = 12
VEL_MAG = 5.                 # [m/s]
VEL_STD = 10.                 # [m/s]
NUM_REGIONS = 16
UE_HEIGHT = 1.5             # [m]

# Wireless
CARRIER = 3e9       # [Hz]
NUMEROLOGY = 0
NUM_SCS = 108
NUM_SYM = 14
NUM_SLOTS = 5
NUM_FRAMES = 100
PPM_XO = 5 / 50                # Residual PPM after initial synchronization from PRACH
QUEUE_RELIABILITY = 0.95         # 1 - epsilon in the paper
UE_POW = ut.dbm2watt(-10)

# DEFAULT FOLDERS
RM_FOLDER =  str(Path(__file__).resolve().parent / "rm")

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
    # Performance guarantees
    queue_reliability: float = QUEUE_RELIABILITY

    # Guard time between frames [s]
    guard_time: float = 0.09


@dataclass
class Metrics(BaseMetrics):
    """Collect relevant metrics for semantic isac simulations.
     All metric variables are saved, from frame = 0 to frame = self.c.num_frames - 1.
     If frame = self.c.num_frames (indexed also with [-1]), the element is the initialization variable.
    """
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


def arrival_rate_init(avg_se: float, num_ue: int = NUM_UE, num_slots: int = NUM_SLOTS,
                      bw: float = NUM_SCS * 15e3 * 2 ** NUMEROLOGY, slot_time: float = 1e-3, rho: float= 0.75):
    lambda_max = num_slots / num_ue * avg_se * bw * slot_time
    return rho * lambda_max, 10 * lambda_max
