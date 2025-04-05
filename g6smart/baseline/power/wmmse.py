from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt

from g6smart.sim_config import SimConfig


def _subband_iterative_wmmse(
    C: Annotated[npt.NDArray[np.complex128], Literal["N", "N"]],
    max_power: float,
    noise_power: float,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.float64]]:
    """
    Iterative WMMSE algorithm for subband processing.

    Args:
        C: (N, N) complex channel matrix (users → APs)
        max_power: maximum transmit power per user
        noise_power: noise variance
        max_iter: maximum number of iterations
        tol: convergence tolerance

    Returns:
        W: (N, N) precoding matrix (columns are precoding vectors)

    References:
        [1] Z. Wang, J. Zhang, H. Q. Ngo, B. Ai and M. Debbah, "Iteratively Weighted MMSE Uplink Precoding for Cell-Free Massive MIMO," ICC 2022 - IEEE International Conference on Communications, Seoul, Korea, Republic of, 2022, pp. 231-236, doi: 10.1109/ICC45855.2022.9838843. keywords: {Spectral efficiency;Precoding;Massive MIMO;Rayleigh channels;Mean square error methods;Uplink;Iterative decoding},


    """
    N = C.shape[0]
    dtype = np.complex128

    # Initialize variables
    W = np.sqrt(max_power / N) * np.eye(N, dtype=dtype)
    C_H = C.conj().T  # Precompute Hermitian

    # Numerical stability scaling
    scale = max(np.max(np.abs(C)), 1e-8)
    C_scaled = C / scale
    noise_scaled = noise_power / (scale ** 2)

    # Precompute outer products
    C_outer = np.einsum('ij,ik->ijk', C_scaled, C_scaled.conj())

    prev_sinr = np.zeros(N)
    for _ in range(max_iter):
        # MMSE receiver update (vectorized)
        interference = noise_scaled * np.eye(N)
        interference += np.einsum('ij,ik,jkl->il', W.conj(), W, C_outer).real
        u = np.diag(W.conj().T @ C_H) / np.diag(C_H @ interference @ C_scaled)

        # MSE weights with numerical safety
        CW = C_H @ W
        signal = np.abs(u * np.diag(CW)) ** 2
        total = noise_scaled * np.abs(u) ** 2 + np.sum(np.abs(u[:, None] * CW) ** 2, axis=1)
        w = 1 / (1 - 2 * np.real(signal) + total + 1e-12)

        # Update precoders
        W = (C_H * u * w).conj().T

        # Normalize to power constraint
        norms = np.linalg.norm(W, axis=0) + 1e-10
        W *= np.sqrt(max_power) / norms

        # Compute SINRs
        CW = C_H @ W
        signal = np.abs(np.diag(CW)) ** 2
        interference = noise_scaled + np.sum(np.abs(CW) ** 2, axis=1) - signal
        sinrs = signal / interference

        # Check convergence
        if np.max(np.abs(sinrs - prev_sinr)) < tol:
            break
        prev_sinr = sinrs.copy()

    return W


def iterative_wmmse(
    channel_gain: Annotated[npt.NDArray[np.float64], Literal["K", "N", "N"]],
    allocation: Annotated[npt.NDArray[np.int64], Literal["N"]],
    config: SimConfig,
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
    K, N, _ = channel_gain.shape
    power = np.zeros(N, dtype = np.float64)

    # each subband is treated separately
    for k in range(K):
        active_users = np.flatnonzero(allocation == k)
        if not active_users.size:
            continue

        # compute the most efficient allocation
        band_gains = channel_gain[k][active_users][:, active_users]
        w, _ = _subband_iterative_wmmse(
            band_gains, config.max_power, config.noise_power,
            max_iter, tol
        )

        power[active_users] += np.sum(np.abs(w) ** 2, axis = 0)

    return power
