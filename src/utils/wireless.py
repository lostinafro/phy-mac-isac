# Wireless: path loss, fading, noise, and network-node placement/mobility
import numpy as np
from scipy.constants import speed_of_light
from scipy.stats import rice, norm

# The supported channel types are the following.
channel_types = {'LoS', 'No', 'AWGN', 'Rayleigh', 'Rice', 'Shadowing'}


def ue_ap_distances(ue_position: np.ndarray, ap_position: np.ndarray) -> np.ndarray:
    """Compute the distances between UEs and APs. It assumes UEs position is an array of (..., U, d) while AP position
    is an array of (L, d).

    :param ue_position: np.ndarray, (..., U, d) position of the UE.
    :param ap_position: np.ndarray, (L, d) position of the AP.
    :returns: distance between UEs and APs, shape (..., L, U)
    """
    return np.linalg.norm(ap_position[np.newaxis, np.newaxis] - ue_position[..., np.newaxis, :], axis=-1)


def eval_ue_ap_dist(ue_pos: np.ndarray, ap_pos: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.tile(ue_pos[:, np.newaxis], (1, ap_pos.shape[0], 1)) - ap_pos, axis=2)


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


def fading(typ: str, dim: tuple = (1,),
           shape: float = 6, seed: int = None) -> np.ndarray:
    """Create a sampled fading channel from a given distribution and given
    dimension.

    Parameters
    __________
    typ : str in channel_types,
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
    from scipy.constants import k
    return k * bandwidth * t0 * 10 ** (noise_figure / 10)  # [W]


def thermal_noise_dbm(bandwidth: float, noise_figure: float = 3., t0: float = 293.) -> float:
    """Compute the noise power [dBm] according to bandwidth and ambient temperature.

    :param bandwidth : float, receiver total bandwidth [Hz]
    :param noise_figure: float, noise figure of the receiver [dB]
    :param t0: float, ambient temperature [K]

    :returns: power of the noise [dBm]
    """
    from .units import watt2dbm
    return watt2dbm(thermal_noise(bandwidth, noise_figure, t0))  # [dBm]


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
    # Move the UE
    new_pos = ue_pos + v_max * direction * time_frame
    # If it goes out of the room move it back
    new_pos[np.any(new_pos > 0.99 * side, axis=1)] = ue_pos[np.any(new_pos > 0.99 * side, axis=1)]
    # Return new the position
    return new_pos