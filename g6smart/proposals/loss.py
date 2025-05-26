import torch
import torch.nn.functional as F

from g6smart.evaluation import rate_torch as metrics
from g6smart.sim_config import SimConfig


def loss_fullfield_req(config: SimConfig, C: torch.Tensor, A: torch.Tensor, req: float) -> torch.Tensor:
    """Basic Required Fullment Loss Function

    Args:
        config (SimConfig): simulation configuration
        C (torch.Tensor): channel state matrix (B x K x N x N)
        A (torch.Tensor): soft probabilistic subband allocation (BxKxN)
        req (float): minimum required spectral efficency

    Returns:
        torch.Tensor: vector with the loss function value per batch sample (B, )
    """
    # calculate shannon rate
    sinr = metrics.signal_interference_ratio(config, C, A, None)
    rate = torch.sum(A * torch.log2(1 + sinr), dim = 1)

    rate = F.sigmoid(req - rate) / req
    rate = torch.sum(rate, dim=1)
    return rate

def min_approx(x: torch.Tensor, p: float = 1e5, mu: float = 0.) -> torch.Tensor:
    """

    Differentiable Approximation of Minimum Function. This function approximates
    the value of min(x).

    **Note**:
    if the parameter `p` is set to negative number, this function will compute the approximate
    maximum of the batch.

    This function was copied and adapted from:
    * https://mathoverflow.net/questions/35191/a-differentiable-approximation-to-the-minimum-function

    Args:
        x (torch.Tensor): input tensor with dimension B x N
        p (float, optional): Approximation parameter. Large values of this parameter corresponds with a more precise approximation of the minimum. Defaults to 1e5.
        mu (float, optional): shift for calculations. Defaults to 0.

    Returns:
        torch.Tensor: approximate minimum per batch sample (B, )
    """
    return mu - (1 / p) * torch.logsumexp(-p * (x - mu), dim = 1)

def loss_pure_rate(
    config: SimConfig, C: torch.Tensor, A: torch.Tensor, P: torch.Tensor | None = None,
    mode: str = 'sum', p: int = 1e5
) -> torch.Tensor:
    """Raw loss function that returns an aggregation of the raw spectral efficency.

    Args:
        config (SimConfig): simulation configuration
        C (torch.Tensor): channel state matrix (B x K x N x N)
        A (torch.Tensor): soft probabilistic subband allocation (BxKxN)
        P (torch.Tensor, optional): power allocation (BxN). If None, then the function uses the maximum transmit power. Defaults to None
        mode (str, optional): rate loss aggregation mode (mean, sum, min, max). The max and min are differentiable approximations that are affected by the parameter `p`. Defaults to sum.
        p (float, optional): aggregation approximation parameter. Defaults  to 1e5
    Returns:
        torch.Tensor: vector with the loss function value per batch sample (B, )
    """
    sinr = metrics.signal_interference_ratio(config, C, A, None)
    mask = torch.sigmoid(10 * (sinr - 0.01))
    rate = torch.sum(torch.log2(1 + sinr) * A * mask, dim = 1)
    rate = rate / torch.sum(A, dim = 1)

    if mode == 'mean':
      loss_rate = torch.mean(rate, dim = 1)
    if mode == 'sum':
      loss_rate = torch.sum(rate, dim = 1)
    elif mode == 'min':
      loss_rate = min_approx(rate, + p)
    elif mode == 'max':
      loss_rate = min_approx(rate, - p)
    return - loss_rate

def binarization_error(alloc: torch.Tensor) -> torch.Tensor:
    """Binarization Loss Function. This loss is tracked in most models but it is not
    used for training. It tracks the ability of the model to focus on one allocation only.

    Args:
        alloc (torch.Tensor): base allocation probabilities BxKxN

    Returns:
        torch.Tensor: single scalar value with the binarization error. (1, )
    """
    rounded = torch.round(alloc)
    return torch.mean(torch.abs(alloc - rounded))

def update_metrics(
    prev_metrics: dict[str, float], A: torch.Tensor, C: torch.Tensor, P: torch.Tensor | None,
    config : SimConfig, req: float
    ) -> dict[str, float]:
    """Update a metric dictionary with the sum of the previous value and the new metrics.
    This function is intended for the model training loop and after each epoch, the mean must
    be computed.

    This function calculates the following metrics:
    * Mean Bit Rate (MBps)
    * Jain Fairness Metric
    * Mean Spectral Efficency
    * Proportional Path Loss. This metrics is the ratio of the spectral efficency against ideal conditions (no interference).
    * Over Requirements. Proportion of subnetworks that reach a given spectral efficencty requirements thresshold.

    Args:
        prev_metrics (dict[str, float]): previous metrics dictionary.
        C (torch.Tensor): channel state matrix (B x K x N x N)
        A (torch.Tensor): soft probabilistic subband allocation (BxKxN)
        P (torch.Tensor, optional): power allocation (BxN). If None, then the function uses the maximum transmit power. Defaults to None
        config (SimConfig): simulation configuration
        req (float): thresshold for spectral efficency. It allows the calculation of the percentage of networks that reach a certain level of spectral efficency.

    Returns:
        dict[str, float]: updated metrics dictionary
    """
    A        = metrics.onehot_allocation(A, 4, 20)
    sinr     = metrics.signal_interference_ratio(config, C, A, P, False)
    rate     = metrics.bit_rate(config, sinr, A)
    fairness = metrics.jain_fairness(rate)
    spectral = metrics.spectral_efficency(config, rate)
    plf      = metrics.proportional_loss_factor(config, C, A, P)

    shannon  = torch.sum(A * torch.log2(1 + sinr), dim = 1)
    ecf_req  = torch.mean((shannon >= req).float(), dim = 1)

    prev_metrics['bit-rate'] += rate.mean().item() / 1e6
    prev_metrics['jain-fairness'] += fairness.mean().item()
    prev_metrics['spectral-efficency'] += spectral.mean().item()
    prev_metrics['proportional-loss' ] += plf.mean().item()
    prev_metrics['over-requirement' ] += ecf_req.mean().item()
    return prev_metrics
