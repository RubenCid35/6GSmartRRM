
# input data format
import numpy as np

# deep learning framework
import torch
import torch.nn as nn
import torch.nn.functional as F

# Simulation Settings
from g6smart.sim_config import SimConfig
from g6smart.evaluation.rate_torch import signal_interference_ratio

# define device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# First Step: Subband allocation problem
class RateConfirmAllocModel(nn.Module):
    def __init__(self, n_subnetworks: int, n_bands: int,
                 hidden_dim: int = 1000, hidden_layers: int = 4,
                 dropout: float | None = 0.1,
                 use_weighted: bool = False,
                 keep_band_wise: bool = False) -> None:
        """Rate Confirming Model with modifications 

        Args:
            n_subnetworks (int): Nº of subnetworks in the whole network
            n_bands (int): Nº of subbands to allocate
            hidden_dim (int, optional): Nº of neurons in each hidden layer. Defaults to 1000.
            hidden_layers (int, optional): Nº of hidden layers. Defaults to 4.
            dropout (float | None, optional): Dropout probability. If it is None, the model won't use dropout. Defaults to 0.1.
            use_weighted (bool, optional): Wheter to use weighted representation of channel inputs. This idea was copied from the SISA algorithm. Defaults to False.
            keep_band_wise (bool, optional): wheter the model uses all the subbands information or it uses the mean value. Defaults to False.
        
        References:
        * Rate-conforming Sub-band Allocation for In-factory Subnetworks: A Deep Neural Network Approach
            
        """

        super().__init__()

        # initialize state
        self.n = n_subnetworks
        self.k = n_bands

        # preprocessing options
        self.use_weighted   = use_weighted
        self.keep_band_wise = keep_band_wise

        # DNN architecture
        self.input_size = self.n * self.n if not self.keep_band_wise else self.n * self.n * self.k
        self.output_size = self.n * self.k

        last_layer = -2 if dropout is None else -3

        # learn the normalization preprocessing while training, the subnetworks are disordered
        # cann't know the arrangement or make assumptions about mean and std
        
        layers = [nn.BatchNorm1d(self.input_size)] # with batch norm at start
        dims = [self.input_size] + [hidden_dim] * (hidden_layers + 1) + [self.output_size]
        for i in range(1, len(dims)):
        
            # linear layers with HE initialization
            layers.append(nn.Linear(dims[i - 1], dims[i]))
            torch.nn.init.kaiming_normal_(layers[-1].weight, nonlinearity='relu')
            layers.append(nn.ReLU())

            # apply dropout. We have a lot of parameters, it is required
            layers.append(nn.BatchNorm1d(dims[i]))
            if isinstance(dropout, float):
              layers.append(nn.Dropout(dropout))

        layers = layers[:last_layer]
        self.model = nn.Sequential(*layers)

    def preprocess(self, channel_gain: np.ndarray | torch.Tensor ) -> torch.Tensor:
        channel_gain = torch.tensor(channel_gain, requires_grad=False).float()

        if self.keep_band_wise and len(channel_gain.shape[1:]) != 3:
            raise ValueError("The model expects a channel gain matrix (BxKxNxN)")

        if self.keep_band_wise and self.use_weighted:
            # normalize the interference matrix
            Hd = torch.diagonal(channel_gain, dim1 = -2, dim2 = -1).unsqueeze(-1)

            zero = torch.zeros_like(channel_gain).to(channel_gain.device)
            channel_gain = torch.where(Hd > 0, channel_gain / Hd, zero)

            # remove self-interference
            B, K, N, _ = channel_gain.shape
            self_signal = torch.eye(N, device = DEVICE).expand(B, K, -1, -1)
            channel_gain = channel_gain * (1 - self_signal)

        elif not self.keep_band_wise and len(channel_gain.shape[1:]) == 3:
            channel_gain = torch.mean(channel_gain, dim = 1)

        # flatten the channel information
        channel_gain = channel_gain.flatten(start_dim=1)

        # transform to dbm scale to restrict value range
        channel_gain = 10 * torch.log10(channel_gain + 1e-9) # transform to Dbm

        return channel_gain

    def forward(self, channel_gain: torch.Tensor, t: float = 1.0 ) -> torch.Tensor:
        # preprocess to obtain a NxN channel gain
        channel_gain = self.preprocess(channel_gain)
        # apply model
        channel_network = self.model(channel_gain)
        # determine best allocation
        channel_network = channel_network.reshape(-1, self.k, self.n)
        # derive probabilities
        return F.softmax(channel_network / t, dim = 1)
    

# ----------------------------------------------------------------------------------------------
# LOSS Functions
# ----------------------------------------------------------------------------------------------

def loss_fullfield_req(config: SimConfig, C: torch.Tensor, A: torch.Tensor, req: float, mode: str = 'mean') -> torch.Tensor:
    # calculate shannon rate
    sinr = signal_interference_ratio(config, C, A, None)
    rate = torch.sum(A * torch.log2(1 + sinr), dim = 1)

    rate = F.sigmoid(req - rate) / req
    rate = torch.sum(rate, dim=1)
    return rate

def min_approx(x: torch.Tensor, p: float = 1e2):
    """
    Differentiable Approximation of Minimum Function. This function approximates
    the value of min(x)

      # based on fC https://mathoverflow.net/questions/35191/a-differentiable-approximation-to-the-minimum-function
    """
    mu = 0
    inner = torch.mean(torch.exp(- p * (x - mu)), dim = 1)
    return mu - (1 / p) * torch.log(inner)

def loss_pure_rate(config: SimConfig, C: torch.Tensor, A: torch.Tensor, mode: str = 'sum', p: int = 10) -> torch.Tensor:
    sinr = signal_interference_ratio(config, C, A, None)
    rate = torch.sum(A * torch.log2(1 + sinr), dim = 1)

    if mode == 'sum':
      loss_rate = torch.sum(rate, dim = 1)
    elif mode == 'min':
      loss_rate = min_approx(rate, p)
    elif mode == 'max':
      loss_rate = torch.sum(rate, dim = 1)
    return - loss_rate