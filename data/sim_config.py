import numpy as np
import configparser

class SimConfig:
    def __init__(self, rng, config_file="config.ini"):
        config = configparser.ConfigParser()
        config.read(config_file)
        
        self.num_of_subnetworks = int(config.get("General", "num_of_subnetworks", fallback=20))
        self.n_subchannel = int(config.get("General", "n_subchannel", fallback=4))
        self.deploy_length = float(config.get("General", "deploy_length", fallback=20))
        self.subnet_radius = float(config.get("General", "subnet_radius", fallback=1))
        self.minD = float(config.get("General", "minD", fallback=0.8))
        self.minDistance = 2 * self.subnet_radius
        
        self.bandwidth = float(config.get("Network", "bandwidth", fallback=100e6))
        self.ch_bandwidth = self.bandwidth / self.n_subchannel
        self.fc = float(config.get("Network", "fc", fallback=6e9))
        self.lambdA = 3e8 / self.fc
        
        self.clutType = config.get("Environment", "clutType", fallback="dense")
        self.clutSize = float(config.get("Environment", "clutSize", fallback=2.0))
        self.clutDens = float(config.get("Environment", "clutDens", fallback=0.6))
        self.shadStd = float(config.get("Environment", "shadStd", fallback=7.2))
        
        self.max_power = float(config.get("Power", "max_power", fallback=1))
        self.no_dbm = float(config.get("Power", "no_dbm", fallback=-174))
        self.noise_figure_db = float(config.get("Power", "noise_figure_db", fallback=5))
        
        self.noise_power = 10 ** ((self.no_dbm + self.noise_figure_db + 10 * np.log10(self.ch_bandwidth)) / 10)
        self.mapXPoints = np.linspace(0, self.deploy_length, num=401, endpoint=True)
        self.mapYPoints = np.linspace(0, self.deploy_length, num=401, endpoint=True)
        self.correlationDistance = float(config.get("General", "correlationDistance", fallback=5))
        
        self.rng_value = np.random.RandomState(rng)

    def __repr__(self) -> str:
        def format_value(val) -> str:
            if isinstance(val, str): return f"{val:>25s}"
            else: return f"{val:>25.4f}"
        
        header = f"| {'name':>25s} | {'value':>25s} |\n{'-'*(25+25+7)}\n"
        table = "\n".join([f"| {name:>25s} | {format_value(value)} |" for name, value in self.__dict__.items() if isinstance(value, (float, int, str))])
        return "Simulation Parameters: \n\n" + header + table + f"\n{'-'*(25+25+7)}\n"
