import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon


def map_to_output(x: np.ndarray, y, fh: ForecastingHorizon = None, full_posterior: bool = False):
    """
    Maps `x` to the same type as `y`.

    Args:
        x: Model output, always a numpy array.
        y: Model input.
        fh: Forecasting horizon.
        full_posterior: Whether data is of the full posterior or not.

    Returns:
        An object that depends on stuffy.
    """

    if y is None:
        return x

    index = fh.to_pandas()

    # TODO: need to verify the reshape s.t. it makes sense
    if full_posterior:
        index = pd.MultiIndex.from_product([np.arange(x.shape[0]), index])

    if isinstance(y, pd.Series):
        return pd.Series(x.reshape(-1), index=index, name=y.name)

    if isinstance(y, pd.DataFrame):
        return pd.DataFrame(x.reshape(-1, y.shape[-1]), columns=y.columns, index=y.index)

    return x
