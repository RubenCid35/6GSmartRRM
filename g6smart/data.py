import os
import zipfile
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
        *csi_datasets: npt.NDArray,
        split_sizes: Tuple[int, int, int] | None= None,
        batch_size: int = 512, seed: int = 101
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates DataLoaders for multi-target data, using the first dataset as input.

    Args:
        *csi_datasets (npt.NDArray): The first dataset is input, others are targets. All must have the same number of samples.
        split_sizes (Tuple[int, int, int] or None, optional): split sizes. It is a tuple with the following mapping: [train_size, valid_size, test_size]. If it set to None, then the following split will be done 70% / 15% / 15%.
        batch_size (int, optional): batch size for each data loader. Defaults to 512.
        seed (int, optional): random state. It is used for reproducibility. Defaults to 101.

    Returns:
        Tuple[DataLoader,DataLoader,DataLoader]: newly generated data loaders. The data loaders are in order: training, validation, testing.
    """
    num_samples = csi_datasets[0].shape[0]
    for i, data in enumerate(csi_datasets):
        assert data.shape[0] == num_samples, (
            f"Dataset at index {i} has {data.shape[0]} samples, expected {num_samples}."
        )

    assert len(csi_datasets[0].shape) == 4 and csi_datasets[0].shape[2] == csi_datasets[0].shape[3], (
        "The file does not correspond with simulations CSI data."
    )
    if split_sizes is None:
        split_sizes = [
            int(num_samples * 0.7), int(num_samples * 0.15), int(num_samples * 0.15)
        ]

    assert sum(split_sizes) == num_samples, (
        "The data split must cover the whole dataset."
        f"Currently the splits cover {sum(split_sizes)}, while data has {num_samples} records."
    )

    # split the data
    train_idx, valid_idx, tests_idx = random_split(
        list(range(num_samples)), split_sizes,
        generator=torch.Generator().manual_seed(seed)
    )
    torch_datasets = [torch.tensor(data).float() for data in csi_datasets]
    def create_split_dataset(indices):
        split_tensors = [tensor[indices] for tensor in torch_datasets]
        return TensorDataset(*split_tensors)

    train_dataset = create_split_dataset(train_idx)
    valid_dataset = create_split_dataset(valid_idx)
    test_dataset  = create_split_dataset(tests_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader

# default simulations locations
SIMULATION_DEFAULT_LOCAL: str = './data/'
SIMULATION_DEFAULT_DRIVE: str = '/content/drive/MyDrive/TFM/simulations'

def download_simulations_data(
        is_colab: bool = False,
        simulations_local_path: str = SIMULATION_DEFAULT_LOCAL,
        simulations_drive_path: str = SIMULATION_DEFAULT_DRIVE
) -> Tuple[str, str]:
    """Special function that serves downloads and unzips the simulations data from google drive or a local zip.
    The google drive usage only works inside the google colab instance and it expects a given file location.

    Args:
        is_colab (bool, optional): google drive flag. Defaults to False.

    Returns:
        Tuple[str, str]: tuple with the simulations data path and the pre-trained models location.
    """
    simulation_path = "./data/simulations" if not is_colab  else "/content/simulations"
    models_path = "./models" if not is_colab  else "/content/drive/MyDrive/TFM/models/"
    os.makedirs(simulation_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)

    # Moung Google Drive Code
    if is_colab:
        try:
            from distutils.dir_util import copy_tree

            # Move Simulations to avoid cluttering the drive folder
            files = os.listdir(simulation_path)
            if '.ipynb_checkpoints' in files:
              files.remove(".ipynb_checkpoints")

            if len(files) == 0:
                copy_tree(simulations_drive_path, simulation_path)

        except Exception as err:
            print(f"failed to mount drive, cause: {err}")
            pass

    data_location = simulation_path if is_colab else simulations_local_path
    for zip_file in os.listdir(data_location):
        if zip_file.endswith('.zip'):
            print(" ----> " + zip_file)
            file_path = os.path.join(data_location, zip_file)
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(simulation_path)

    return simulation_path, models_path
