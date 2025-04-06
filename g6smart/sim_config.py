import numpy as np


class SimConfig:
    def __init__(self, rng, original: bool = False):
        self.num_of_subnetworks = 20                              # Number of subnetworks
        self.n_subchannel = 4                                     # Number of sub-bands
        self.deploy_length = 20                                   # Length and breadth of the factory area (m)
        self.subnet_radius = 1                                    # Radius of the subnetwork cell (m)
        self.minD = 0.8                                           # Minimum distance from device to controller (access point) (m)
        self.minDistance = 2 * self.subnet_radius                 # Minimum controller to controller distance (m)
        self.rng_value = np.random.RandomState(rng)
        self.bandwidth = 40e6                                     # Bandwidth (Hz) - updated to 40 MHz from 100 MHz
        self.ch_bandwidth = self.bandwidth / self.n_subchannel    # Channel bandwidth per sub-band
        self.fc = 6e9                                             # Carrier frequency (Hz)
        self.lambdA = 3e8 / self.fc                               # Wavelength
        self.clutType = 'dense'                                   # Type of clutter (dense)
        self.clutSize = 2.0                                       # Clutter element size (m)
        self.clutDens = 0.6                                       # Clutter density [%]
        self.shadStd = 7.2                                        # Shadowing standard deviation (dB)
        self.max_power = 1 if not original else 0                 # Transmit power (dBm) - updated to 0 dBm from 1 dBm
        self.no_dbm = -174                                        # Noise power in dBm
        self.noise_figure_db = 5                                  # Noise figure (dB)
        self.noise_power = 10 ** ((self.no_dbm + self.noise_figure_db + 10 * np.log10(self.ch_bandwidth)) / 10)
        self.mapXPoints = np.linspace(0, self.deploy_length, num=401, endpoint=True)
        self.mapYPoints = np.linspace(0, self.deploy_length, num=401, endpoint=True)
        self.correlationDistance = 5
        self.transmit_power = 10 ** (self.max_power / 10 - 3)  # max tranmit power in Wats

    def __repr__(self) -> str:
        def format_value(val) -> str:
            if isinstance(val, str):
                return f"{val:>25s}"
            else:
                return f"{val:>25.4f}"

        header = f"| {'name':>25s} | {'value':>25s} |\n{'-'*(25+25+7)}\n"
        table = "\n".join([f"| {name:>25s} | {format_value(value)} |" for name, value in self.__dict__.items() if isinstance(value, (float, int, str))])
        return "Simulation Parameters: \n\n" + header + table + f"\n{'-'*(25+25+7)}\n"
