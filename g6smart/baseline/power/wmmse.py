from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt


def _subband_iterative_wmmse(
    G: Annotated[npt.NDArray[np.complex128], Literal["N", "N"]],
    max_power: float,
    noise_power: float,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.float64]]:
    """
    Iterative WMMSE algorithm for subband processing.

    Args:
        G: (N, N) complex channel matrix (users → APs)
        max_power: maximum transmit power per user
        noise_power: noise variance
        max_iter: maximum number of iterations
        tol: convergence tolerance

    Returns:
        W: (N, N) precoding matrix (columns are precoding vectors)

    References:
        [1] Z. Wang, J. Zhang, H. Q. Ngo, B. Ai and M. Debbah, "Iteratively Weighted MMSE Uplink Precoding for Cell-Free Massive MIMO," ICC 2022 - IEEE International Conference on Communications, Seoul, Korea, Republic of, 2022, pp. 231-236, doi: 10.1109/ICC45855.2022.9838843. keywords: {Spectral efficiency;Precoding;Massive MIMO;Rayleigh channels;Mean square error methods;Uplink;Iterative decoding},


    """
    N = G.shape[0]
    p = np.full(N, max_power / 2)  # Initial power allocation

    for _ in range(max_iter):
        p_prev = p.copy()

        # Compute SINRs
        SINR = np.zeros(N)
        for i in range(N):
            interference = np.sum([p[j] * G[i, j] for j in range(N) if j != i])
            SINR[i] = (p[i] * G[i, i]) / (interference + noise_power)

        # MMSE receiver weight
        w = 1 / (1 + SINR)

        # Utility weight (derivative of log(1 + SINR))
        u = 1 / (np.log(2) * (1 + SINR))

        # Update power
        for i in range(N):
            interference = sum([
                u[j] * G[j, i] * w[j] for j in range(N) if j != i
            ])
            numerator = u[i] * G[i, i]
            denominator = w[i] * (interference + 1e-9)
            p[i] = min(max_power, numerator / denominator)

        # Check convergence
        if np.linalg.norm(p - p_prev) < tol:
            break

    return p


def iterative_wmmse(
    channel_gain: Annotated[npt.NDArray[np.float64], Literal["K", "N", "N"]],
    allocation: Annotated[npt.NDArray[np.integer], Literal["N"]],
    max_power: float,
    noise_power: float,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> npt.NDArray[np.float64]:
    """
    Computes power allocation using WMMSE for all subnetworks.

    Args:
        channel_gain: (K, N, N) channel gain matrices for K subbands
        allocation: (N,) subband allocation for each user
        config: Simulation configuration with max_power and noise_power
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        power_allocation: (N,) array of total power allocated to each user
    """
    # initialization
    channel_gain = channel_gain.copy()
    K, N, _ = channel_gain.shape
    power = np.zeros(N, dtype = np.float64)

    # each subband is treated separately
    for k in range(K):
        active_users = np.flatnonzero(allocation == k)
        if not active_users.size:
            continue

        # compute the most efficient allocation
        band_gains = channel_gain[k][active_users][:, active_users]
        w = _subband_iterative_wmmse(
            band_gains, max_power, noise_power,
            max_iter, tol
        )[1]

        # w = np.abs(w) ** 2
        power[active_users] = w
        # power[active_users] = np.sum(w, axis = 0)

    return power
