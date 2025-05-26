import torch

from g6smart.sim_config import SimConfig


def onehot_allocation(A: torch.Tensor, K: int, N: int) -> torch.Tensor:
    B = A.size(0)
    if A.dim() == 2 and A.shape[1] == N:  # Hard allocation case (B, N)
        # Convert hard allocation indices to one-hot (B, K, N)
        A_one_hot = torch.zeros((B, K, N), device=A.device).float()
        A_one_hot.scatter_(1, A.unsqueeze(1), 1)
    elif A.dim() == 3 and A.shape[1] == K and A.shape[2] == N:  # One-hot allocation case (B, K, N)
        A_one_hot = A  # Already in the correct format
    else:
        raise ValueError("Invalid allocation shape. Expected (B, N) for hard allocation or (B, K, N) for one-hot.")
    return A_one_hot

def signal_interference_ratio(
        config: SimConfig,
        C: torch.Tensor,
        A: torch.Tensor,
        P: torch.Tensor | float | None = None,
        return_dbm: bool = False
    ) -> torch.Tensor:
    """Computes the signal-to-interference ratio (SINR) using PyTorch, supporting batch processing on GPU.

    Args:
        config: Configuration object containing system parameters (must have `noise_power` as an attribute).
        C (torch.Tensor): (B, K, N, N) tensor with the channel gain values, where B is batch size.
        A (torch.Tensor): (B, N) or (B, K, N) tensor with subband allocations of each node.
        P (torch.Tensor | float): (B, N) tensor or a scalar representing transmission power assigned to each subnetwork. If it is None, we take the power from `config.transmit_power`.
        return_dbm (bool): Whether to return the output in decibels (dBm). Defaults to False.

    Returns:
        torch.Tensor: (B, K, N) tensor with the SINR of each subnetwork per subband
    """

    B, K, N, _ = C.shape
    NE         = torch.tensor(config.noise_power, device = C.device).float()
    # define power settings
    if P is None or isinstance(P, (int, float)):
        P = torch.full((B, K, N), P or config.transmit_power, device=C.device).float()
    elif isinstance(P, torch.Tensor) and len(P.shape) == 2: # power not expanded
        P = P.unsqueeze(1).expand(-1, K, -1)

    # Standarize the allocation format to one-hot allocation
    A = onehot_allocation(A, K, N)

    # signal calculation
    signal = A * P * torch.diagonal(C, dim1=-2, dim2=-1)

    # interference calculation
    interference = torch.sum((A * P).unsqueeze(-1) * C, dim=-2)
    interference = interference - signal # remove self-interference
    interference = interference + NE + 1e-9

    # calculate signal-interference ratio
    sinr = signal / interference # added epsilon for stability

    # convert to dbm if required
    if return_dbm:
        return 10 * torch.log10(sinr)
    else:
        return sinr

def bit_rate(config: SimConfig, sinr: torch.Tensor, alloc: torch.Tensor | None = None) -> torch.Tensor:
    """Computes the bit rate for each subnetwork based on the allocated subbands,
    channel gain, and transmission power.

    The bit rate is calculated using Shannon's capacity formula:

    $Rate = B * log2(1 + SINR)$

    where:
    - B is the total bandwidth (Hz) available in the system.
    - SINR (Signal-to-Interference-and-Noise Ratio) is computed for each subnetwork.

    Args:
        config (SimConfig): Configuration object with simulation details
        sinr (torch.Tensor): (B, N) tensor with the SINR of each subnetwork. It can be calculated using `signal_interference_ratio` function.
        alloc (torch.Tensor): (B, K, N) or (B, N) tensor with the allocation of each subband. It is only required when the `sinr` has dimension (B, K, N)
    Returns:
        torch.Tensor: (B, N) tensor with the bit rate of each subnetwork in bps
    """
    bandwidth = torch.tensor(config.ch_bandwidth, device = sinr.device).float()
    if len(sinr.shape) == 3:

      _, K, N = sinr.size(0), sinr.size(1), sinr.size(2)
      assert alloc is not None and isinstance(alloc, torch.Tensor), "We required to give the allocation to weight the channels"
      A = onehot_allocation(alloc, K, N)
      return bandwidth * torch.sum(A * torch.log2(1 + sinr), dim = 1)

    else:
      # common scenario
      return bandwidth * torch.log2(1 + sinr)

def proportional_loss_factor(
        config,
        C: torch.Tensor,
        A: torch.Tensor,
        P: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
    """PLF measures how much a subnetwork's rate deviates from the ideal rate due to interference, bandwidth allocation,
    and other system constraints.

    Args:
    * config: Configuration object containing system parameters (must have `noise_power` as an attribute).
    * channel_gain (torch.Tensor): (B, K, N, N) tensor with the channel gain values, where B is batch size.
    * allocation (torch.Tensor): (B, N) or (B, K, N) tensor with subband allocations of each node.
    * power (torch.Tensor | float): (B, N) tensor or a scalar representing transmission power assigned to each subnetwork. If it is None, we take the power from `config.transmit_power`.

    Returns:
    * torch.Tensor: (B, ) tensor with the PLF of each simulation.
    """
    B, K, N, _ = C.shape
    NE         = torch.tensor(config.noise_power, device = C.device).float()

    # define power settings
    if P is None or isinstance(P, (int, float)):
        P = torch.full((B, K, N), P or config.transmit_power, device=C.device).float()
    elif isinstance(P, torch.Tensor) and len(P.shape) == 2: # power not expanded
        P = P.unsqueeze(1).expand(-1, K, -1)

    A = onehot_allocation(A, K, N)

    # obtain real conditions
    sinr = signal_interference_ratio(config, C, A, P, False)
    rate = bit_rate(config, sinr, A)

    # obtain ideal conditions
    ideal = A * P * torch.diagonal(C, dim1=-2, dim2=-1) / (NE + 1e-9)
    ideal = bit_rate(config, ideal, A)

    # return plf
    return torch.sum(rate, dim=1) / (torch.sum(ideal, dim=1) + 1e-9)

def jain_fairness(rate: torch.Tensor) -> float:
    """
    Jain's Fairness Index is used to evaluate fairness in resource allocation. It ranges
    from 0 (extremely unfair) to 1 (perfectly fair). It helps in evaluating how evenly
    the resources are distributed among users in a network.

    Args:
        rate (torch.Tensor): (B, N) tensor with the bit rate of each subnetwork in bps
    Returns:
        float: Jain's Fairness Index.
    """
    _, N  = rate.shape
    upper = torch.sum(rate, dim = 1) ** 2
    lower = N * torch.sum(rate ** 2, dim = 1)
    return upper / lower

def spectral_efficency(
        config: SimConfig,
        rate: torch.Tensor
    ) -> torch.Tensor:
    """
    Computes the spectral efficency for each subnetwork based on the allocated subbands,
    channel gain, and transmission power.

    The bit rate is calculated using Shannon’s capacity formula:

    $Rate = Bit Rate / B$

    where:
    - Bit Rate is obtained using the function `bit_rate`
    - B is the total bandwidth (Hz) available in the system.

    Args:
    * config (SimConfig): Configuration object containing system parameters
    * rate (torch.Tensor): (B, N) tensor with the bit rate of each subnetwork in bps

    Returns:
    * (torch.Tensor): (B, N) tensor with the spectral efficency of each subnetwork in bps
    """
    B  = config.ch_bandwidth
    return rate / B
