from typing import Tuple

import numpy as np
import numpy.typing as npt


def get_cdf(
        sample: npt.NDArray[np.floating]
    ) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Calculate the CDF position and density values for a sample

    Args:
        sample (npt.NDArray[np.floating]): data sample with dimension (N, )

    Returns:
        Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]: tuple with the value positions and cdf values (in this order).
    """
    assert len(sample.shape) == 1 and sample.shape[0] > 2, (
        "The sample needs to flatten and it needs at least 2 data points."
    )
    pdf, bins = np.histogram(sample, bins = 500, density=True)
    cdf = np.cumsum(pdf * np.diff(bins))
    return bins[1:], cdf
