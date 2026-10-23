from copy import deepcopy
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field

from .. import utils as ut
from ..modules.params import SCENE_FOLDER, DATA_FOLDER, CACHE_FOLDER, _directory_from_env
from ..modules.params import SharedParams as BaseShared
from ..modules.params import WirelessParams as BaseWireless


## DEFAULT PARAMETERS
NUM_EP = int(1e2)
# Scenario
NUM_AP = 1
NUM_ANT = 1
NUM_UE = 100
VEL_MAG = 0.                 # [m/s]
VEL_STD = 1e-9                 # [m/s]
NUM_REGIONS = 1
UE_HEIGHT = 1             # [m]

# Wireless
CARRIER = 3e9       # [Hz]
NUMEROLOGY = 0
NUM_SCS = 72
NUM_SYM = 14
NUM_SLOTS = 20
NUM_FRAMES = 100
PPM_XO = 0.     # Perfect synchronization is assumed among nodes and AP
UE_POW = ut.dbm2watt(0)


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

    # Guard time between frames [s]
    guard_time: float = 0.
