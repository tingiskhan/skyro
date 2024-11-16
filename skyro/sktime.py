import sys
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict

import numpy as np
import pandas as pd
from numpyro.infer import Predictive
from skpro.distributions.empirical import Empirical
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from xarray import DataArray

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from ._mixin import BaseNumpyroMixin


# TODO: do NOT support vectorization
class BaseNumpyroForecaster(BaseNumpyroMixin, BaseForecaster):
    """
    Implements a base class for forecasters utilizing :class:`numpyro`.
    """

    dynamic_args: Dict[str, Any] = {}

    _tags = {
        "capability:insample": True,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "requires-fh-in-fit": False,
    }

    def __init__(
        self,
        *,
        num_samples: int,
        num_warmup: int,
        num_chains: int = 1,
        chain_method: str = "parallel",
        seed: int = None,
        progress_bar: bool = False,
        kernel_kwargs: Dict[str, Any] = None,
        model_kwargs: Dict[str, Any] = None,
    ):
        super().__init__(
            num_samples=num_samples,
            num_warmup=num_warmup,
            num_chains=num_chains,
            chain_method=chain_method,
            seed=seed,
            progress_bar=progress_bar,
            kernel_kwargs=kernel_kwargs,
            model_kwargs=model_kwargs,
        )
        BaseForecaster.__init__(self)

    def build_model(self, y, length: int, X=None, future: int = 0, **kwargs):
        raise NotImplementedError("abstract method")

    def _fit(self, y, X, fh):
        key = self._get_key()

        length = y.shape[0]

        self.mcmc.run(key, X=X, y=y, length=length, future=0, **(self.model_kwargs or {}), **self.dynamic_args)
        self.result_set_ = self._process_results(self.mcmc)

        return

    def _do_sample(self, length: int, horizon: int, X=None, prior_predictive: bool = False) -> Dict[str, np.ndarray]:
        """
        Helper function for performing sampling via :class:`Predictive`.

        Args:
            length: Length to forecast.
            horizon: Forecasting horizon.
            X: Exogenous variables.
            prior_predictive: Whether to use prior samples.

        Returns:
            Returns the full trace returned by :class:`Predictive`. Samples are __not__ grouped by chain.
        """

        samples = None if prior_predictive else self.result_set_.get_samples(group_by_chain=False)
        predictive = Predictive(
            self.build_model, posterior_samples=samples, num_samples=self.num_samples if samples is None else None
        )

        output = predictive(
            self._get_key(),
            y=self._y,
            X=X,
            length=length,
            future=horizon,
            **(self.model_kwargs or {}),
            **self.dynamic_args,
        )

        return {k: np.array(v) for k, v in output.items()}

    def format_output(self, x: Dict[str, np.ndarray], horizon: ForecastingHorizon) -> DataArray:
        """
        Formats output.

        Args:
            x: Sample trace.
            horizon: Forecasting horizon.

        Returns:
            Returns a :class:`DataArray`.
        """

        raise NotImplementedError("abstract method")

    def _do_predict(self, fh: ForecastingHorizon, X=None, full_posterior: bool = False) -> DataArray:
        if self._X is not None and not self.get_tag("ignores-exogeneous-X"):
            # TODO: need to use numpy depending on mtype
            X = pd.concat([self._X, X], axis=0, verify_integrity=True)

        length = self._y.shape[0]
        future_index = fh.to_relative(self.cutoff).to_numpy()
        future = future_index.max()

        predictions = self._do_sample(length, horizon=future, X=X)
        actual_index = fh.to_absolute(self.cutoff)

        # TODO: need to figure out how to do this one...
        slice_index = future_index + length - 1

        sliced_predictions = {k: v[:, slice_index] for k, v in predictions.items()}
        output = self.format_output(sliced_predictions, actual_index)

        if not full_posterior:
            output = self.reduce(output)

        return output

    def _predict(self, fh, X):
        output = self._do_predict(fh, X)
        return output

    def _predict_proba(self, fh, X, marginal=True):
        predictions = self._do_predict(fh, X, full_posterior=True)

        as_frame = predictions.to_dataframe(name=predictions.name)

        return Empirical(as_frame, time_indep=False)

    def sample_prior_predictive(self, length: int, X=None, **kwargs) -> Dict[str, np.ndarray]:
        """
        Does posterior/prior predictive checking.

        Returns:
            Returns samples.
        """

        return self._do_sample(length, horizon=0, X=X, prior_predictive=True)

    @contextmanager
    def set_dynamic_args(self, **kwargs) -> Self:
        """
        When modelling in numpyro you sometimes require setting parameters that don't fit into sktime's input API (but
        scikit-learns' in terms of allowing passing kwargs). This is a workaround for that by utilizing a context
        manager.

        Args:
            **kwargs: Any variables.

        Returns:
            Returns :class:`Self`.
        """

        old = deepcopy(self.dynamic_args)

        try:
            self.dynamic_args.update(kwargs)
            yield self
        finally:
            self.dynamic_args = old

        return
