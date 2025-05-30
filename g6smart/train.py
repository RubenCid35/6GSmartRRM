from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from g6smart.evaluation.rate_torch import signal_interference_ratio
from g6smart.proposals.loss import binarization_error, loss_pure_rate
from g6smart.sim_config import SimConfig


def _bit_rate(config: SimConfig, C: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    A    = torch.argmax(A, dim = 1)
    sinr = signal_interference_ratio(config, C, A, None)
    rate = torch.sum( 10 * torch.log2(1 + sinr), dim = 1)
    return torch.mean(rate, dim = 1).mean()


def train_model(
    model: nn.Module,
    config: SimConfig,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.DeviceObjType,
    loss_mode: str = "min",
    epochs: int   = 50,
    lr: float     = 1e-3,
    lr_min: float = 1e-4,
    lr_dt : int   = 50,
    a     : float = 0.5
) -> nn.Module:
    """Train model using a pure bit rate unsupervised learning method.

    Args:
        model (nn.Module): base model
        config (SimConfig): simulation configuration.
        train_loader (DataLoader): training data loader.
        valid_loader (DataLoader): validation data loader
        device (torch.DeviceObjType): data device
        loss_mode (str, optional): loss function mode (See `loss_pure_rate` documentation). Defaults to "min".
        epochs (int, optional): max number of training epochs. Defaults to 50
        lr (float, optional): initial learning rate. Defaults to 1e-3.
        lr_min (float, optional): min learning rate. Defaults to 1e-4.
        lr_dt (int, optional): number of epochs to reach min learning rate (cyclic). Defaults to 50.
        a (float, optional): if hybrid mode, this parameter indicate the influence of the min over mean . Defaults to 0.5

    Returns:
        torch.nn.Module: trained model reference
    """

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lrs.CosineAnnealingLR(optimizer, T_max=lr_dt, eta_min=lr_min)

    loss_func = partial(loss_pure_rate, mode = loss_mode, p = 1e6, a = a)
    for step in range(epochs):
        model.train()
        total_loss = 0.
        total_bin_error = 0.
        total_bit_rate  = 0.
        for sample in tqdm(train_loader, desc = "training: ", unit=" batch", total = len(train_loader), leave = False):
            optimizer.zero_grad()

            sample = sample[0].to(device)
            alloc_prob = model(sample)        # soft output
            loss = loss_func(config, sample, alloc_prob).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            # training metrics
            total_loss += loss.item()
            total_bin_error += binarization_error(alloc_prob)
            total_bit_rate  += _bit_rate(config, sample, alloc_prob)

            del sample, alloc_prob, loss
            torch.cuda.empty_cache()

        ttotal_loss = total_loss / len(train_loader)
        ttotal_bin_error = total_bin_error / len(train_loader)
        ttotal_bit_rate  = total_bit_rate / len(train_loader)

        model.eval()
        total_loss = 0.
        total_bin_error = 0.
        total_bit_rate  = 0.

        with torch.no_grad():
            for sample in tqdm(valid_loader, desc = "validation: ", unit=" batch", total = len(valid_loader), leave = False):
                sample = sample[0].to(device)
                alloc_prob = model(sample)        # soft output
                loss = loss_func(config, sample, alloc_prob).mean()

                # loss = loss_interference(sample, alloc_prob).mean()
                total_loss += loss.item()
                total_bin_error += binarization_error(alloc_prob)
                total_bit_rate  += _bit_rate(config, sample, alloc_prob)

                del sample, alloc_prob, loss
                torch.cuda.empty_cache()

        total_loss = total_loss / len(valid_loader)
        total_bin_error = total_bin_error / len(valid_loader)
        total_bit_rate  = total_bit_rate / len(valid_loader)

        lr = scheduler.get_last_lr()[-1]
        print(
            f"[{step:>3d}] (lr: {lr:1.2e})",
            f"train loss: {ttotal_loss:7.4f}",
            f"(bin error: {ttotal_bin_error:5.3e}, bit rate: {ttotal_bit_rate:4.2f})",
            f"valid loss: { total_loss:7.4f}",
            f"(bin error: { total_bin_error:5.3e}, bit rate: { total_bit_rate:4.2f})",
            sep = " "
        )

        scheduler.step()
        return model
