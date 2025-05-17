import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F


# First Step: Subband allocation problem
class RateConfirmAllocModel(nn.Module):
    def __init__(self, n_subnetworks: int, n_bands: int,
                 hidden_dim: int = 1000, hidden_layers: int = 4,
                 dropout: float | None = 0.01,
                 use_weighted: bool = False,
                 keep_band_wise: bool = False) -> None:
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

    def preprocess(self, channel_gain: npt.NDArray[np.float32] | torch.Tensor ) -> torch.Tensor:
        channel_gain = torch.tensor(channel_gain, requires_grad=False).float()
        device       = next(self.parameters()).device

        if self.keep_band_wise and len(channel_gain.shape[1:]) != 3:
            raise ValueError("The model expects a channel gain matrix (BxKxNxN)")

        if self.keep_band_wise and self.use_weighted:
            # normalize the interference matrix
            Hd = torch.diagonal(channel_gain, dim1 = -2, dim2 = -1).unsqueeze(-1)

            zero = torch.zeros_like(channel_gain).to(channel_gain.device)
            channel_gain = torch.where(Hd > 0, channel_gain / Hd, zero)

            # remove self-interference
            B, K, N, _ = channel_gain.shape
            self_signal = torch.eye(N, device = device).expand(B, K, -1, -1)
            channel_gain = channel_gain * (1 - self_signal)

        elif not self.keep_band_wise and len(channel_gain.shape[1:]) == 3:
            channel_gain = torch.mean(channel_gain, dim = 1)

        # flatten the channel information
        channel_gain = channel_gain.flatten(start_dim=1)

        # transform to dbm scale to restrict value range
        channel_gain = 10 * torch.log10(channel_gain + 1e-9) # transform to Dbm

        # normalize to values.
        # cavg = channel_gain.mean(dim = 1, keepdim = True)
        # cstd = channel_gain.std( dim = 1, keepdim = True)
        # channel_gain = (channel_gain - cavg) / cstd
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
