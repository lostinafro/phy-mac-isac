import numpy as np
from abc import ABC, abstractmethod

class TrackerBase(ABC):
    """Abstract base class for a recursive Bayesian tracker (Kalman-family filters and similar)
    operating on batched state vectors of shape (..., Z).

    Subclasses implement the model-specific `prior` (predict) and `posterior` (update) steps.
    `step` is provided here as a concrete method: it's pure composition of the two abstract
    steps and is identical across every subclass, so it doesn't belong in each of them.
    """

    @abstractmethod
    def prior(self, x: np.ndarray, P: np.ndarray, u: np.ndarray | None = None) -> tuple:
        """Predict step.

        :returns: tuple (x_pri, P_pri).
        """
        raise NotImplementedError

    @abstractmethod
    def posterior(self, obs: np.ndarray, x_pri: np.ndarray, P_pri: np.ndarray, obs_idx: np.ndarray | None = None,
                  obs_cov: np.ndarray | None = None) -> tuple:
        """Update step.

        :returns: tuple (x_post, P_post).
        """
        raise NotImplementedError

    def step(self, obs: np.ndarray, x: np.ndarray, P: np.ndarray, u: np.ndarray | None = None) -> tuple:
        """Convenience method chaining `prior` then `posterior` in a single call."""
        x_pri, P_pri = self.prior(x, P, u)
        return self.posterior(obs, x_pri, P_pri)


class LinearKalmanTracker(TrackerBase):
    """Generic linear Kalman filter operating on arbitrary leading (batch) dimensions.

    Given a state vector of shape (..., Z), every operation is applied along the last axis
    (or last two axes for covariances) via vectorized matrix multiplication -- no explicit
    loop over the leading (batch) dimensions, e.g. (num_frames, num_ue, ...), is used.

    :param F: np.ndarray, state-transition matrix, shape (d, d).
    :param H: np.ndarray, observation matrix, shape (d_obs, d).
    :param Q: np.ndarray, process noise covariance, shape (d, d).
    :param R: np.ndarray, observation noise covariance, shape (d_obs, d_obs).
    :param B: np.ndarray | None, control-input matrix, shape (d, d_u). If None, `prior` raises
        if a control vector `u` is supplied.
    """

    def __init__(self, F: np.ndarray, H: np.ndarray, Q: np.ndarray, R: np.ndarray,
                 B: np.ndarray | None = None):
        self.F = np.asarray(F)
        self.H = np.asarray(H)
        self.Q = np.asarray(Q)
        self.R = np.asarray(R)
        self.B = np.asarray(B) if B is not None else None
        self.dim = self.F.shape[-1]

    def update_model(self, F: np.ndarray | None = None, H: np.ndarray | None  = None, Q: np.ndarray | None = None,
                     R: np.ndarray | None = None, B: np.ndarray | None = None):
        """Update the model parameters if needed."""
        if F is not None:
            self.F = np.asarray(F)
        if H is not None:
            self.H = np.asarray(H)
        if Q is not None:
            self.Q = np.asarray(Q)
        if R is not None:
            self.R = np.asarray(R)
        if B is not None:
            self.B = np.asarray(B)

    def prior(self, x: np.ndarray, P: np.ndarray, u: np.ndarray | None = None) -> tuple:
        """Predict step: x_pri = F x (+ B u), P_pri = F P F^T + Q.

        :param x: np.ndarray, shape (..., Z), previous posterior state estimate.
        :param P: np.ndarray, shape (..., Z, Z), previous posterior covariance.
        :param u: np.ndarray | None, shape (..., Z_u), optional control input (requires `B`).

        :returns: tuple (x_pri (..., Z), P_pri (..., Z, Z)).
        """
        x, P = np.asarray(x), np.asarray(P)
        x_col = x[..., :, np.newaxis]                     # (..., Z, 1)
        x_pri_col = self.F @ x_col

        if u is not None:
            if self.B is None:
                raise ValueError("u was given but no control-input model B was provided.")
            u_col = np.asarray(u)[..., :, np.newaxis]
            x_pri_col = x_pri_col + self.B @ u_col

        x_pri = np.squeeze(x_pri_col, axis=-1)
        P_pri = self.F @ P @ np.matrix_transpose(self.F) + self.Q             # broadcasts over leading dims
        return x_pri, P_pri

    def posterior(self, obs: np.ndarray, x_pri: np.ndarray, P_pri: np.ndarray, obs_idx: np.ndarray | None = None,
                  obs_cov: np.ndarray | None = None) -> tuple:
        """Update step: standard Kalman correction given a new observation.

        :param obs: np.ndarray, shape (..., Z_obs), new observation.
        :param x_pri: np.ndarray, shape (..., Z), prior state estimate (from `prior`).
        :param P_pri: np.ndarray, shape (..., Z, Z), prior covariance (from `prior`).
        :param obs_idx: np.ndarray, shape (..., Z_obs), index vector of observation available. Needed to compute the
            posteriors for elements where an observation is obtained and keep the other posterior equal to the prior
        :param obs_cov: np.ndarray, shape (..., Z_obs, Z_obs), observation covariance. Needed to compute the
            posteriors for elements where an observation is obtained and keep the other posterior equal to the prior

        :returns: tuple (x_post (..., Z), P_post (..., Z, Z)).
        """
        if obs_idx is None:
            obs_idx = np.ones(self.H.shape[:-2], dtype=bool)
        if obs_cov is None:
            obs_cov = self.R[obs_idx]

        # processing for posteriors
        x_post_out = np.asarray(x_pri).copy()
        P_post_out = np.asarray(P_pri).copy()

        obs, x_pri, P_pri = np.asarray(obs), np.asarray(x_pri)[obs_idx], np.asarray(P_pri)[obs_idx]
        x_pri_col = x_pri[..., :, np.newaxis]

        innovation = obs[..., :, np.newaxis] - self.H[obs_idx] @ x_pri_col      # (..., Z_obs, 1)
        S = self.H[obs_idx] @ P_pri @ np.matrix_transpose(self.H[obs_idx]) + obs_cov               # (..., Z_obs, Z_obs)
        K = P_pri @ np.matrix_transpose(self.H[obs_idx]) @ np.linalg.inv(S)                        # (..., Z, Z_obs)

        x_post = np.squeeze(x_pri_col + K @ innovation, axis=-1)       # (..., Z)
        I = np.eye(self.dim)
        P_post = (I - K @ self.H[obs_idx]) @ P_pri                              # (..., Z, Z)

        x_post_out[obs_idx] = x_post
        P_post_out[obs_idx] = P_post
        return x_post_out, P_post_out