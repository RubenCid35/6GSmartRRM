import numpy as np
from g6smart.sim_config import SimConfig

def signal_interference_ratio(
        config: SimConfig, 
        channel_gain: np.ndarray,
        allocation: np.ndarray,
        power: np.ndarray | float,
        return_dbm: bool = False
    ) -> np.ndarray:
    """Computes the signal-to-interference ratio for each subnetwork in their allocated channel. For each subnetwork,
    the ratio of signal is computed gains the interference of the activity of the rest of subnetworks in the same subband. 

    Args:
    * config (SimConfig): Configuration object containing system parameters
    * channel_gain (np.ndarray): a (K, N, N) matrix with the channel gain values
    * allocation (np.ndarray): (N, ) array with the subband allocations of each node
    * power (np.ndarray | float): transmission power that is assigned to each subnetwork. If the values is a float, then all subnetworks
    have the power.  
    * return_dbm (bool). Wheter the final output is measured in decibels (dbm) for comparison purposes. Defauls to False

    Returns:
    *  np.ndarray: (N, ) array with the SINR of each subnetwork.  

    """
    K, N, _ = channel_gain.shape
    noise = config.noise_power

    # Ensure allocations are within valid subband indices [0, K-1]
    assert np.max(allocation) < K and allocation.shape[0] == N, \
           "Each subnetwork must have an allocated subband from 0 to K - 1"

    # Ensure power is an array of shape (N,)
    power = np.full((N,), power) if isinstance(power, (float, int)) else power
    assert power.shape[0] == N, "Each subnetwork must have a power assignment at least"

    # calculate the SINR
    sinr = np.zeros((N, ))
    for n, k in enumerate(allocation):
        signal = channel_gain[k, n, n] * power[n]

        # estimate the interference
        masked_power = power * (allocation == k)
        interference = np.dot(channel_gain[allocation, np.arange(N), n], masked_power)
        interference = interference - signal # remove self-interference

        sinr[n] = signal / (interference + noise + 1e-9)
    
    # change units
    if return_dbm: return 10 * np.log10(sinr)
    else: return sinr

def bit_rate(
        config: SimConfig, 
        channel_gain: np.ndarray,
        allocation: np.ndarray,
        power: np.ndarray | float
    ) -> np.ndarray:
    """
    Computes the bit rate for each subnetwork based on the allocated subbands, 
    channel gain, and transmission power.

    The bit rate is calculated using Shannon's capacity formula:
    
    $Rate = B * log2(1 + SINR)$
    
    where:
    - B is the total bandwidth (Hz) available in the system.
    - SINR (Signal-to-Interference-and-Noise Ratio) is computed for each subnetwork.
    
    Args:
    * config (SimConfig): Configuration object containing system parameters
    * channel_gain (np.ndarray): a (K, N, N) matrix with the channel gain values
    * allocation (np.ndarray): (N, ) array with the subband allocations of each node
    * power (np.ndarray | float): transmission power that is assigned to each subnetwork. If the values is a float, then all subnetworks
    have the power.  

    Returns:
        np.ndarray: (N, ) array with the bit rate of each subnetwork. The bit rate is measured in Mbps (10^6 bps). 

    Note:

    If the values may appear as invalid, it has been study that 6G networks could perform with great SINR ratios
    in comparison with other networks. The current setup is short-range. For example, this [publication](https://arxiv.org/abs/2305.18309)
    display results with great SINR (>20dbm) in short range scenario.
    
    """
    B  = config.ch_bandwidth
    SINR = signal_interference_ratio(config, channel_gain, allocation, power, False)
    return B * np.log2(SINR + 1) / 1e6

def proportional_loss_factor(
        config: SimConfig, 
        channel_gain: np.ndarray,
        allocation: np.ndarray,
        power: np.ndarray | float
    ) -> float:
    """
    PLF measures how much a subnetwork's rate deviates from the ideal rate due to interference, bandwidth allocation, 
    and other system constraints. 
    
    Args:
    * config (SimConfig): Configuration object containing system parameters
    * channel_gain (np.ndarray): a (K, N, N) matrix with the channel gain values
    * allocation (np.ndarray): (N, ) array with the subband allocations of each node
    * power (np.ndarray | float): transmission power that is assigned to each subnetwork. If the values is a float, then all subnetworks
    have the power.  

    Returns:
        float: proportional loss factor of the whole network 
    """
    K, N, _ = channel_gain.shape
    # all subnetworks have the same bandwidth
    B  = config.ch_bandwidth
    N0 = config.noise_power

    # estimate SINR 
    rate = bit_rate(config, channel_gain, allocation, power)

    # calculate ideal conditions
    ideal_snir = channel_gain[allocation, np.arange(N), np.arange(N)] * power / (N0 + 1e-9)
    ideal = B * np.log2(ideal_snir + 1) / 1e6

    return np.sum(rate) / np.sum(ideal)

def spectral_efficency(
        config: SimConfig, 
        rate: np.ndarray
    ) -> np.ndarray:
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
    * rate (np.ndarray): a (N, ) array with the bit rate of each subnetwork

    Returns:
        np.ndarray: (N, ) array with the spectral efficency of each subnetwork. The bit rate is measured in Mbps (10^6 bps). 
    """
    B  = config.ch_bandwidth
    return rate / B

def jain_fairness(rate: np.ndarray) -> float:
    """
    Jain's Fairness Index is used to evaluate fairness in resource allocation. It ranges
    from 0 (extremely unfair) to 1 (perfectly fair). It helps in evaluating how evenly 
    the resources are distributed among users in a network.
    
    Args:
        rate (np.ndarray): bit rate of each subnetwork. These values can be estimated using `bit_rate`

    Returns:
        float: Jain's Fairness Index.
    """
    upper = np.sum(rate) ** 2
    lower = rate.shape[0] * np.sum(np.power(rate, 2))
    return upper / lower
