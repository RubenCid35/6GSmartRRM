import numpy as np

def weighted_interference_matrix(channel_gain: np.ndarray) -> np.ndarray:
    """
    Computes the weighted interference matrix W based on the channel gain matrix.

    Parameters:
    - channel_gain (np.ndarray): A (K, N, N) array representing the channel gains 
      between different nodes and subnetworks.

    Returns:
    - W (np.ndarray): A (K, N, N) matrix where W[k, i, j] represents the 
      interference weight from node i to node j in subnetwork k.
    """
    # Compute direct channel gains and normalize interference
    Hd = np.expand_dims(np.diagonal(channel_gain, 0, 1, 2), 2)
    W  = np.where(Hd > 0, channel_gain / Hd , 0)
    
    #  remove self-interference
    for k in range(W.shape[0]): np.fill_diagonal(W[k], 0)

    return W

def sisa_algoritm(channel_gain: np.ndarray, max_iter: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """
    Implements the Sequential Iterative Subband Allocation (SISA) Algorithm

    The algorithm iteratively assigns each subnetwork to a subband that minimizes the sum weighted
    interference of the whole network while ensuring a fair allocation.

    Args:
        - channel_gain (np.ndarray): A (K, N, N) array representing the channel gains.
        - max_iter (int): The maximum number of iterations for optimization. Defaults to 20 iterations.

    Returns:
        - A (np.ndarray): An (N, ) array where A[n] represents the assignment of the subnetwork n to the subband k.
        - F (np.ndarray): An (N x max_iter) array with a list of values of the sum interference of each step. 
    
    Reference:
        - [Advanced Frequency Resource Allocation for Industrial Wireless Control in 6G subnetworks](https://ieeexplore.ieee.org/document/10118695) 
    """
    K, N, _ = channel_gain.shape
    
    # inititalize inputs
    A = np.zeros((N)   , dtype=int) # A : N -> K
    B = np.zeros((K, N), dtype=int) # B_k : {n \in N : A(n) = k}
    
    # first all networks are assigned to the same subband
    B[0, :] = 1

    W = weighted_interference_matrix(channel_gain)
    total_interference = []

    # procedure
    for _ in range(1, max_iter + 1):
        for n in range(N):
            # 1. compute iteration number : not required
            # 2. compute w_k (d) for all k
            w_k  = np.sum(B * (W[:, n, :] + W[:, :, n]), axis = 1)

            # 3. determine interim allocation A
            A[n] = np.argmin(w_k)

            # 4. determine interim allocation B based on A
            B[:, :] = 0 # reset allocations
            B[A, np.arange(N)] = 1
            
            # 5. compute sum weighted interference
            mask = B[A, :]
            F = np.sum(W[A[:, None], np.arange(N)[:, None], np.where(mask == 1)[1]])            
            total_interference.append(F)
    
    return A, np.array(total_interference)