import os
from typing import Tuple

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


def load_data(simulations_path: str, n_samples: int | None = 110_000) -> npt.NDArray | None:
    """Load simulation data from the simulation data folder. This function can take a subsample of the data or return the whole dataset

    This function will look for the following file: `Channel_matrix_gain.npy`

    Args:
        simulations_path (str): simulations data folder that contains the simulations data. It requires to contained the file `Channel_matrix_gain.npy`.
        n_samples (int | None, optional): number of samples to take. If it is None, then it takes all simulations. Defaults to 110_000.

    Raises:
        FileNotFoundError: if the simulations data folder does not exists

    Returns:
        npt.NDArray: simulations data as a numpy array. It has dimensions B x K x N x N
    """
    if not os.path.exists(simulations_path):
        print(f"The provided data folder does not exist. Data Folder: {simulations_path}")
        raise FileNotFoundError

    assert n_samples is None or (isinstance(n_samples, int) and n_samples > 1), (
        "The number of samples must be None or number greater than 0."
    )

    cmg = np.load(simulations_path + "/Channel_matrix_gain.npy")
    cmg = cmg[:n_samples] if n_samples is not None else cmg
    assert len(cmg.shape) == 4 and cmg.shape[2] == cmg.shape[3], (
        "The file does not correspond with simulations CSI data."
    )
    return cmg

def create_datasets(
        csi_data: npt.NDArray, split_sizes: Tuple[int, int, int],
        batch_size: int = 512, seed: int = 101
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Splits the data and creates the data loaders for each part.

    Args:
        csi_data (npt.NDArray): original simulations data
        split_sizes (Tuple[int, int, int]): split sizes. It is a tuple with the following mapping: [train_size, valid_size, test_size]
        batch_size (int, optional): batch size for each data loader. Defaults to 512.
        seed (int, optional): random state. It is used for reproducibility. Defaults to 101.

    Returns:
        Tuple[DataLoader,DataLoader,DataLoader]: newly generated data loaders. The data loaders are in order: training, validation, testing.
    """

    assert len(csi_data.shape) == 4 and csi_data.shape[2] == csi_data.shape[3], (
        "The file does not correspond with simulations CSI data."
    )

    assert sum(split_sizes) == csi_data.shape[0], (
        "The data split must cover the whole dataset."
        f"Currently the splits cover {sum(split_sizes)}, while data has {csi_data.shape[0]} records."
    )

    # split the data
    whole_data       = torch.tensor(csi_data).float()
    train_idx, valid_idx, tests_idx = random_split(
        range(len(whole_data)), split_sizes,
        generator=torch.Generator().manual_seed(seed)
    )

    # create dataset and loaders
    train_dataset = TensorDataset(whole_data[train_idx])
    valid_dataset = TensorDataset(whole_data[valid_idx])
    tests_dataset = TensorDataset(whole_data[tests_idx])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    tests_loader = DataLoader(tests_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, tests_loader
