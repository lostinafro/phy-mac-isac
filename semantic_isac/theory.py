import numpy as np
from scipy.constants import speed_of_light
import sionna.rt as rt
from parameters import SharedParams, WirelessParams, GeometryParams, EstimationParams
from abc import ABC, abstractmethod

class PerformanceBase(ABC):
    """Generic class for theoretical performance calculations"""

    def __init__(self, c: SharedParams, wireless: WirelessParams, geo: GeometryParams, estimate: EstimationParams):
        """Initialize the performance module.

        :param c: SharedParams object containing constant parameters
        :param wireless: WirelessParams object containing wireless parameters
        :param geo: GeometryParams object containing geometric parameters
        """
        self.c = c
        self.wireless = wireless
        self.geo = geo
        self.estimate = estimate

    @abstractmethod
    def eval_efim(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def eval_achievable_se(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError


class DMimoFarFieldOMA(PerformanceBase):
    """Compute theoretical performance for a distributed MIMO scenario where far-field propagation, orthogonal UE
    allocation, and negligible CFO are assumed."""

    def __init__(self, c: SharedParams, wireless: WirelessParams, geo: GeometryParams, estimate: EstimationParams):
        """Initialize the performance module for a distributed MIMO scenario where far-field propagation, orthogonal
        UE allocation, and negligible CFO are assumed.

        :param c: SharedParams object containing constant parameters
        :param wireless: WirelessParams object containing wireless parameters
        :param geo: GeometryParams object containing geometric parameters
        """
        super().__init__(c, wireless, geo, estimate)

    def eval_efim(self, snr: np.ndarray, x: np.ndarray, beta: np.ndarray) -> np.ndarray:
        r"""Evaluate effective FIM through algebraic manipulation.

        :param snr: np.ndarray (U,), power to noise ratio accounting for the number of pilots used, e.g., :math:`\frac{p_u P}{\sigma^2}`
        :param x: np.ndarray (U, d) collecting the UE position in the absolute coordinate system, e.g., :math:`\mathbf{x}_{u}`
        :param beta: np.ndarray (S, U, L) collecting the large scale fading for each AP, e.g., :math:`[\beta_{u,\ell}] \forall \ell \in\mathcal{L}`
        """
        d = x.shape[1]
        F = self.c.num_scs
        M = self.c.num_ant

        # Constants
        Sigma2 = sum_set(2, self.wireless.sc_index) / F

        # Take relative positions
        x_rel = x[:, np.newaxis] - self.geo.ap_location[np.newaxis, :, :d]     # (U, L, d)

        # Nabla xx
        x_norm_sq = np.linalg.norm(x_rel, axis=-1) ** 2  # (U, L)
        x_outer = np.einsum('uli,ulj->ulij', x_rel, x_rel)  # (U, L, d, d)

        A = np.eye(d)[np.newaxis, np.newaxis] - x_outer / x_norm_sq[..., np.newaxis, np.newaxis]  # (U, L, d, d)
        B = (np.matrix_transpose(self.geo.antenna_location) - self.geo.ap_location[:, :, np.newaxis])[:, :d, :d] # (L, d, d)
        a_dotdot = (self.wireless.sc_spacing ** 2 * Sigma2 / (speed_of_light ** 2) * x_outer
                    + self.wireless.wavelength ** (-2) / M * A @ (B @ np.matrix_transpose(B))[np.newaxis] @ A)  # (U, L, d, d)
        nabla_xx = (2 * np.pi) ** 2 * np.sum((beta / x_norm_sq[np.newaxis])[..., np.newaxis, np.newaxis] * a_dotdot[np.newaxis], axis=2) # (S, U, d, d)

        # Nabla xxi
        nabla_xxi = np.concat((
            np.zeros((beta.shape[0], self.c.num_ue, d, 2 * self.c.num_ap)),
            ((2 * np.pi * self.wireless.sc_spacing) ** 2 * Sigma2 / speed_of_light *
            np.sum((beta[..., np.newaxis] * (x_rel / np.sqrt(x_norm_sq)[:,:,np.newaxis])[np.newaxis]), axis=2))[..., np.newaxis]), # (S, U, d, 1)
            axis=-1)

        # Nabla \csi
        nabla_xi = np.concat((beta ** (-1) / 4,
                              beta,
                              (2 * np.pi) ** 2 * self.wireless.sc_spacing ** 2 * Sigma2 * np.sum(beta, axis=2)[..., np.newaxis]),
                             axis=-1) # (S, U, 2L + 1)
        # nabla_xi = nabla_xi[..., np.newaxis] * np.eye(nabla_xi.shape[-1])[np.newaxis, np.newaxis]
        nabla_xi_inv = nabla_xi[..., np.newaxis] ** (-1) * np.eye(nabla_xi.shape[-1])[np.newaxis, np.newaxis]

        # Shur complement
        efim = 2 * snr[np.newaxis, :, np.newaxis, np.newaxis] * M * F * (nabla_xx - nabla_xxi @ nabla_xi_inv @ np.matrix_transpose(nabla_xxi))

        # Test the purely theoretical version
        # excess = (2 * np.pi) ** 2 * self.wireless.sc_spacing ** 2 * Sigma2 / (speed_of_light ** 2 * np.sum(beta, axis=2))
        # somma = np.zeros((self.c.num_slots, self.c.num_ue, d, d))
        # for l in range(self.c.num_ap):
        #     for k in range(self.c.num_ap):
        #         outer_lk = np.einsum('ui,uj->uij',x_rel[:, l] / np.sqrt(x_norm_sq)[:, l, np.newaxis], x_rel[:, k] / np.sqrt(x_norm_sq)[:, k, np.newaxis])
        #         somma += beta[..., l, np.newaxis, np.newaxis] * beta[..., k, np.newaxis, np.newaxis] * outer_lk[np.newaxis]
        # efim2 = 2 * snr[np.newaxis, :, np.newaxis, np.newaxis] * M * F * (nabla_xx - excess[..., np.newaxis, np.newaxis] * somma)
        # YES; unfortunately the formula is right.
        return np.squeeze(efim)

    def eval_achievable_se(self, beta: np.ndarray, pilot_length: int):
        r"""Evaluate an achievable data rate obtained through use-and-then-forget-bound.

          :param beta: np.ndarray (S, U, L) collecting the large scale fading for each AP, e.g., :math:`[\beta_{u,\ell}] \forall \ell \in\mathcal{L}`
          :param pilot_length: int, the length of the pilot sequence used for channel estimation.
          """
        snr = self.wireless.ue_power[np.newaxis] / self.wireless.nob_ap

        # Compute with the specific environment formula
        sum_beta = np.sum(beta, axis=-1)
        se = np.log2(1 + snr * self.c.num_ant * sum_beta ** 2 / (2 * sum_beta + self.c.num_ap / snr))
        return np.squeeze((1 - pilot_length / self.c.num_sym) * se)


def eval_speb(efim: np.ndarray, covariance_pri: np.ndarray) -> tuple:
    """Compute (squared) Position Error Bound (PEB) based on UE eFIM and covariance prior.

    :param efim: np.ndarray, shape (..., U, d, d) representing the effective FIM of the position observations.
    :param covariance_pri: np.ndarray, shape (..., U, d, d) representing the covariance matrix of the prior model.
    :returns: tuple of (risk0, risk1, delta) with risk0 being a np.ndarray (U,) representing the sPEB
        with no allocation (only prior is used); risk1 is a np.ndarray (U) representing the sPEB when
        allocating users; delta is a np.ndarray (U) representing their difference.
    """
    covariance_post = np.linalg.inv(efim + np.linalg.inv(covariance_pri))
    speb_prior = np.linalg.trace(covariance_pri)
    speb_post = np.linalg.trace(covariance_post)
    delta_speb = speb_prior - speb_post
    return speb_prior, speb_post, delta_speb



def sum_set(order: int, sets: np.ndarray) -> float:
    return np.sum(sets ** order)


def shur_complement(constant: float, a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray):
    """Wrapper returning the Shur complement of 2 by 2 block matrix"""
    cond_exp = np.log10(np.linalg.cond(d))
    shur_com = constant * (a - b @ np.linalg.inv(d) @ c)
    return shur_com, cond_exp