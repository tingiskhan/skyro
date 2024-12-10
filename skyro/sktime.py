import sys
import warnings
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, List

import numpy as np
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
    names_in_trace: List[str]

    def __init__(
        self,
        *,
        num_samples: int,
        num_warmup: int,
        num_chains: int = 1,
        chain_method: str = "parallel",
        seed: int = None,
        progress_bar: bool = True,
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
        self._do_ppc = False

    def build_model(self, y, length: int, X=None, future: int = 0, **kwargs):
        raise NotImplementedError("abstract method")

    def _fit(self, y, X, fh):
        key = self._get_key()

        length = y.shape[0]

        self.mcmc.run(key, X=X, y=y, length=length, future=0, **(self.model_kwargs or {}), **self.dynamic_args)
        self.result_set_ = self._process_results(self.mcmc)

        return

    def _do_sample(self, length: int, horizon: int, y, X=None, prior_predictive: bool = False) -> Dict[str, np.ndarray]:
        """
        Helper function for performing sampling via :class:`Predictive`.

        Args:
            length: Length to forecast.
            horizon: Forecasting horizon.
            y: Observed series values.
            X: Exogenous variables.
            prior_predictive: Whether to use prior samples.

        Returns:
            Returns the full trace returned by :class:`Predictive`. Samples are __not__ grouped by chain.
        """

        samples = None if prior_predictive else self.result_set_.get_samples(group_by_chain=False)
        predictive = Predictive(
            self.build_model,
            posterior_samples=samples,
            num_samples=self.num_samples if samples is None else None,
            return_sites=self.names_in_trace,
        )

        output = predictive(
            self._get_key(),
            y=y,
            X=X,
            length=length,
            future=horizon,
            **(self.model_kwargs or {}),
            **self.dynamic_args,
        )

        return {k: np.array(v) for k, v in output.items()}

    def format_output(self, posterior: Dict[str, np.ndarray], fh: ForecastingHorizon) -> DataArray:
        """
        Formats output.

        Args:
            posterior: Sample trace. Keys are guaranteed to be :attr:`names_in_trace`.
            fh: An absolute forecasting horizon.

        Returns:
            Returns a :class:`DataArray`.
        """

        raise NotImplementedError("abstract method")

    def select_and_slice(
        self, posterior: Dict[str, np.ndarray], fh: ForecastingHorizon, length: int
    ) -> Dict[str, np.ndarray]:
        """
        Method for slicing and returning the trace that forms the output. Default behaviour assumes that the traces
        contain the full history, s.t. we select using absolute indexes rather than relative. Override this method in
        that case.

        Args:
            posterior: Posterior.
            fh: Relative forecasting horizon.
            length: Length of original series.

        Returns:
            Returns a dictionary of samples.
        """

        slice_index = fh.to_numpy() + length - 1
        return {k: v[:, slice_index] for k, v in posterior.items()}

    def _do_predict(self, fh: ForecastingHorizon, X=None, full_posterior: bool = False) -> DataArray:
        # TODO: gah, this needs to be handled a lot better
        length = self._y.shape[0]
        relative_fh = fh.to_relative(self.cutoff)

        fh_as_numpy = relative_fh.to_numpy()
        future = fh_as_numpy.max()

        y = self._y if not self._do_ppc else None
        predictions = self._do_sample(length, horizon=future, y=y, X=X)

        # TODO: I think this needs to be handled a lot better
        sliced_predictions = self.select_and_slice(predictions, relative_fh, length)

        actual_fh = fh.to_absolute(self.cutoff)
        output = self.format_output(sliced_predictions, actual_fh)

        if not full_posterior:
            output = self.reduce(output)

        return output

    def _predict(self, fh, X):
        output = self._do_predict(fh, X)
        return output

    def _predict_proba(self, fh, X, marginal=True):
        predictions = self._do_predict(fh, X, full_posterior=True)

        columns = self._y_metadata["feature_names"]
        if predictions.name is not None:
            warnings.warn(f"Name of the frame will be overwritten with '{columns}'!")

        as_frame = predictions.to_dataframe(columns)

        if predictions.ndim > 2:
            as_frame = as_frame.squeeze(1).unstack(level=-1)

        return Empirical(as_frame, time_indep=False)

    def sample_prior_predictive(self, length: int, X=None, **kwargs) -> Dict[str, np.ndarray]:
        """
        Samples from the prior predictive distribution.

        Returns:
            Returns samples.
        """

        return self._do_sample(length, horizon=0, y=None, X=X, prior_predictive=True)

    @contextmanager
    def set_dynamic_args(self, **kwargs) -> Self:
        """
        When modelling in numpyro you sometimes require setting parameters that don't fit into sktime's input API (but
        scikit-learns' in terms of allowing passing kwargs) as the parameters aren't known until runtime. This is a
        workaround for that by utilizing a context manager.

        Args:
            **kwargs: Any variables.

        Returns:
            Returns :class:`Self`.
        """

        old = deepcopy(self.dynamic_args)

        try:
            self.dynamic_args.update(kwargs)

            dynamic_parameters = set(self.dynamic_args.keys())
            model_parameters = set((self.model_kwargs or {}).keys())

            if intersection := dynamic_parameters.intersection(model_parameters):
                raise ValueError(f"You're overriding a model parameter with a runtime parameter: '{intersection}'")

            yield self
        finally:
            self.dynamic_args = old

        return

    @contextmanager
    def ppc(self):
        """
        Sets observed value to `None`.

        Returns:
            Nothing.
        """

        try:
            self._do_ppc = True
            yield
        finally:
            self._do_ppc = False

        return
