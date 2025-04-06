import networkx as nx
import numpy as np


def cgc_algoritm(channel_gain: np.ndarray, n_channel: int = 4) -> np.ndarray:
    """
    Implements the Centralized Graph Coloring (CGC) Algorithm

    This method corresponds to a improper graph coloring algorithm that assings subbands to
    minimizes the interference.

    Args:
        - channel_gain (np.ndarray): a (K, N, N) matrix with the channel gain values
        - n_channel (int, optional): max number of subbands to allocate. Defaults to 4.

    Returns:
        - np.ndarray: (N, ) array with the allocations of each node

    References:
        - [Learning to Dynamically Allocate Radio Resources in Mobile 6G in-X Subnetworks](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9569345)
    """

    # estimate the pair-wise interference matrix
    X = np.sum(channel_gain, axis = 0)
    np.fill_diagonal(X, np.inf) # disable self-interference

    GM = np.ones(X.shape, dtype = int)
    np.fill_diagonal(GM, 0) # disable self-interference

    # create conflict graph
    G = nx.from_numpy_array(X)

    # apply gready coloring
    C = nx.coloring.greedy_color(G, strategy='largest_first')

    # remove edges while
    n_iteration = 0
    n_colors    = 100000000000000000000000000000
    while n_colors >= n_channel:
        # remove edge with least interference
        edge = np.unravel_index(np.argmin(X), X.shape)
        X[edge] = np.inf
        GM[edge] = 0

        # recompute coloring
        G = nx.from_numpy_array(GM)
        C = nx.coloring.greedy_color(G, strategy='largest_first')
        n_colors = max(C.values())
        n_iteration += 1

    # set allocations
    allocation = np.zeros((X.shape[0], ), dtype = int)
    for n in range(X.shape[0]):
        allocation[n] = C[n]
    return allocation
