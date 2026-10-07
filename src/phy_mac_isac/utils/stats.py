# Statistics: random sampling, custom distributions, Q-function
import numpy as np
from scipy.special import erf


def qfun(input):
    """Compute the Gaussian Q-function"""
    return 0.5 - 0.5 * erf(input / np.sqrt(2))


def sample_multivariate_gaussian(cov: np.ndarray, mean: np.ndarray | None = None,
                                 rng: np.random.Generator = np.random.default_rng()) -> np.ndarray:
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
    # Generate standard normal samples: shape (A, 1, d)
    standard_samples = rng.standard_normal((cov.shape[0], 1, cov.shape[-1]))
    # Apply Cholesky to obtain X = μ + L @ Z, where Z ~ N(0, I)
    return mean + np.squeeze(np.einsum('nij,nmj->nmi', trian, standard_samples), axis=1)


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