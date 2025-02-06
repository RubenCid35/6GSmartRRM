import torch
import torch.nn.functional as F

from g6smart.sim_config import SimConfig

def __onehot_allocation(A: torch.Tensor, K: int, N: int) -> torch.Tensor:
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
        config, 
        C: torch.Tensor,
        A: torch.Tensor,
        P: torch.Tensor | float | None,
        return_dbm: bool = False
    ) -> torch.Tensor:
    """Computes the signal-to-interference ratio (SINR) using PyTorch, supporting batch processing on GPU.

    Args:
    * config: Configuration object containing system parameters (must have `noise_power` as an attribute).
    * channel_gain (torch.Tensor): (B, K, N, N) tensor with the channel gain values, where B is batch size.
    * allocation (torch.Tensor): (B, N) or (B, K, N) tensor with subband allocations of each node.
    * power (torch.Tensor | float): (B, N) tensor or a scalar representing transmission power assigned to each subnetwork. If it is None, we take the power from `config.transmit_power`.
    * return_dbm (bool): Whether to return the output in decibels (dBm). Defaults to False.

    Returns:
    * torch.Tensor: (B, N) tensor with the SINR of each subnetwork.
    """

    B, K, N, _ = C.shape
    NE         = torch.tensor(config.noise_power, device = C.device).float()
    # define power settings
    if P is None or isinstance(P, (int, float)):
        P = torch.full((B, N), P or config.transmit_power, device=C.device).float()
   
    # Standarize the allocation format to one-hot allocation
    A = __onehot_allocation(A, K, N)

    # Ensure power is a tensor of shape (B, N)
    signal = torch.einsum('bknn,bkn,bn->bn', C, A, P)
    interference = torch.einsum('bknm,bkm,bm->bn', C, A, P) - signal
    interference = interference + NE + 1e-9

    # calculate signal-interference ratio
    sinr = signal / interference # added epsilon for stability
    
    # convert to dbm if required
    if return_dbm: return 10 * torch.log10(sinr)
    else: return sinr

def bit_rate(config: SimConfig, sinr: torch.Tensor) -> torch.Tensor:
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

    Returns:
        torch.Tensor: (B, N) tensor with the bit rate of each subnetwork in bps
    """
    B, N = sinr.size(0), sinr.size(1)
    bandwidth = torch.tensor(config.ch_bandwidth, device = sinr.device).float()
    return bandwidth * torch.log2(1 + sinr)

def proportional_loss_factor(
        config, 
        C: torch.Tensor,
        A: torch.Tensor,
        P: torch.Tensor | float | None,
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
    bandwidth  = torch.tensor(config.ch_bandwidth, device = C.device).float()
    NE         = torch.tensor(config.noise_power, device = C.device).float()

    # define power settings
    if P is None or isinstance(P, (int, float)):
        P = torch.full((B, N), P or config.transmit_power, device=C.device).float()
   
    A = __onehot_allocation(A, K, N)

    # obtain real conditions
    sinr = signal_interference_ratio(config, C, A, P, False)
    rate = bit_rate(config, sinr)

    # obtain ideal conditions
    ids   = torch.arange(C.size(2))
    ideal = torch.einsum('bknn,bkn,bn->bn', C, A, P) / (NE + 1e-9)
    ideal = bit_rate(config, ideal)
   
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
    B, N  = rate.shape
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
