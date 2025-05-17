import os
import zipfile
from typing import Tuple

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
        from distutils.dir_util import copy_tree

        from google.colab import drive

        drive.mount('/content/drive')
        # Move Simulations to avoid cluttering the drive folder
        if len(os.listdir(simulation_path)) == 0:
            copy_tree(simulations_drive_path, simulation_path)

    data_location = simulation_path if is_colab else simulations_local_path
    for zip_file in os.listdir(data_location):
        if zip_file.endswith('.zip'):
            print(" ----> " + zip_file)
            file_path = os.path.join(data_location, zip_file)
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(simulation_path)

    return simulation_path, models_path
