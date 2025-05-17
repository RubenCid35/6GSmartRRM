from typing import Tuple

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_cnn_block(
    in_channels: int, out_channels: int,
    kernel_size: int, pad_size: int = 0, stride_size: int = 1,
    dropout: float | None = None, norm: bool = True
) -> nn.Sequential:
    """This function creates a simple and repeatable CNN Block function. This block
    contains the following layers:

    Conv -> BatchNorm2d (optional) -> ReLU -> Dropout (optional).

    Args:
        in_channels (int): nº of input channels
        out_channels (int): nº of output channels
        kernel_size (int): kernel size of the block
        pad_size (int, optional): padding size. Defaults to 0.
        stride_size (int, optional): stride size. Defaults to 1.
        dropout (float | None, optional): dropout probability. If None, then it is not added. Defaults to None.
        norm (bool, optional): batch normalization flag. Defaults to True.

    Returns:
        nn.Sequential: fully created CNN Block as a sequential model.
    """

    # base cnn layer
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride_size, pad_size)]

    if norm:
        # normalization layer
        layers.append(nn.BatchNorm2d(out_channels))
    # activation function layer
    layers.append(nn.ReLU())

    # dropout layer
    if dropout is not None:
        assert 0 < dropout < 1, "The dropout probability needs to be between 0 and 1 if it is set."
        layers.append(nn.Dropout(dropout))
    # build model
    return nn.Sequential(*layers)


def calculate_flatten_size(
    input_size: Tuple[int, int, int, int],
    cnn_layers: list[dict[str, int]]
) -> Tuple[Tuple[int, int, int, int], int]:
    """Simple helper function that allows to calculate the output size of different CNN Blocks

    Args:
        input_size (Tuple[int, int, int, int]): original input size
        cnn_layers (list[dict[str, int]]): list of different input CNN blocks. Each cnn block is represented as a dictionary

    Returns:
        Tuple[Tuple[int, int, int, int], int]: output dimensions of the last CNN block and the number of items per batch.
    """

    B, C, H, W = input_size  # Unpack initial dimensions

    for layer in cnn_layers:
        if layer["type"] == "conv":
            # Apply convolution formula
            H = (H + 2 * layer["padding"] - layer["kernel_size"]) // layer["stride"] + 1
            W = (W + 2 * layer["padding"] - layer["kernel_size"]) // layer["stride"] + 1
            C = layer["out_channels"]  # Update channel count

        elif layer["type"] == "pool":
            # Apply pooling formula (assuming stride = pool_size for simplicity)
            H //= layer["pool_size"]
            W //= layer["pool_size"]

        elif layer["type"] == "flatten":
            # Stop tracking spatial dimensions, return final shape
            return (B, C, H, W), C * H * W

    return (B, C, H, W), C * H * W

# First Step: Subband allocation problem
class RateConfirmAllocCNNModel(nn.Module):
    def __init__(self,
        n_subnetworks: int, n_bands: int,
        dropout: float | None = 0.01, use_weighted: bool = False
    ):
        super().__init__()

        # initialize state
        self.n = n_subnetworks
        self.k = n_bands

        # preprocessing options
        self.use_weighted   = use_weighted

        # DNN architecture
        self.output_size = self.n * self.k

        # built-in input normalization
        layers = [nn.BatchNorm2d(n_bands)]
        # add CNN blocks
        layers.append(create_cnn_block(n_bands, 32,  3, 1, 1, dropout, True))
        layers.append(create_cnn_block(32, 64,  3, 0, 1, dropout, True))
        layers.append(create_cnn_block(64, 256, 3, 0, 1, dropout, True))
        layers.append(nn.AvgPool2d(2))
        layers.append(nn.Flatten())

        self.cnn_layer_info = [
            {"type": "conv", "out_channels": 32, "kernel_size": 3, "padding": 1, "stride": 1},
            {"type": "conv", "out_channels": 64, "kernel_size": 3, "padding": 0, "stride": 1},
            # {"type": "pool", "pool_size": 2},  # AvgPool2d(2)
            {"type": "conv", "out_channels": 256, "kernel_size": 3, "padding": 0, "stride": 1},
            # {"type": "pool", "pool_size": 2},  # AvgPool2d(2)
            {"type": "flatten"},
        ]

        flatten_size = calculate_flatten_size((1, n_bands, n_subnetworks, n_subnetworks), self.cnn_layer_info)[1]
        print(f"flattened sized: {flatten_size:6d}")

        # add FNN blocks
        dims = [flatten_size] + [1024, 512] + [self.output_size]
        for i in range(1, len(dims)):
            # linear layers with HE initialization
            layers.append(nn.Linear(dims[i - 1], dims[i]))
            torch.nn.init.kaiming_normal_(layers[-1].weight, nonlinearity='relu')

            layers.append(nn.ReLU())

            # apply dropout. We have a lot of parameters, it is required
            layers.append(nn.BatchNorm1d(dims[i]))
            layers.append(nn.Dropout(dropout))

        layers = layers[:-3]
        self.model = nn.Sequential(*layers)

    def preprocess(self, channel_gain: npt.NDArray[np.floating] | torch.Tensor ) -> torch.Tensor:
        channel_gain = torch.tensor(channel_gain, requires_grad=False).float()
        device       = next(self.parameters()).device
        if self.use_weighted:
            # normalize the interference matrix
            Hd = torch.diagonal(channel_gain, dim1 = -2, dim2 = -1).unsqueeze(-1)

            zero = torch.zeros_like(channel_gain).to(channel_gain.device)
            channel_gain = torch.where(Hd > 0, channel_gain / Hd, zero)

            # remove self-interference
            B, K, N, _ = channel_gain.shape
            self_signal = torch.eye(N, device = device).expand(B, K, -1, -1)
            channel_gain = channel_gain * (1 - self_signal)

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
