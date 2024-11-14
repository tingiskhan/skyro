import sys
from contextlib import contextmanager
from copy import deepcopy
from operator import attrgetter
from typing import Any, Dict

import numpy as np
import pandas as pd
from numpyro.infer import Predictive
from skpro.distributions.empirical import Empirical
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from ._mixin import BaseNumpyroMixin
from ._typing import OutputType
from ._utils import map_to_output


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

    def _do_sample(self, fh: ForecastingHorizon, X=None) -> Dict[str, np.ndarray]:
        """
        Helper function for performing sampling via :class:`Predictive`.

        Args:
            fh: Forecasting horizon.
            X: Exogenous variables.

        Returns:
            Returns the full trace returned by :class:`Predictive`. Samples are __not__ grouped by chain.
        """

        samples = None if self._prior_predictive else self.result_set_.get_samples(group_by_chain=False)
        predictive = Predictive(
            self.build_model, posterior_samples=samples, num_samples=self.num_samples if samples is None else None
        )

        length = self._y.shape[0] if self._y is not None else 0

        future = 0
        horizon = fh.to_relative(cutoff=self.cutoff).max() + 1

        if self._prior_predictive:
            length = horizon
            y = None
        else:
            future = horizon
            y = self._y

        output = predictive(
            self._get_key(), y=y, X=X, length=length, future=future, **(self.model_kwargs or {}), **self.dynamic_args
        )

        return {k: np.array(v) for k, v in output.items()}

    def _do_predict(self, fh: ForecastingHorizon, X=None, full_posterior: bool = False) -> np.ndarray:
        if self._X is not None and not self.get_tag("ignores-exogeneous-X"):
            # TODO: need to use numpy depending on mtype
            X = pd.concat([self._X, X], axis=0, verify_integrity=True)

        predictions = self._do_sample(fh, X)

        actual_index = fh.to_absolute(self.cutoff)

        # TODO: need to figure out how to do this one...
        slice_index = fh.to_absolute_int(self._y.index.min(), self.cutoff) if self._y is not None else actual_index

        output = self.select_output(predictions)[:, slice_index]

        if not full_posterior:
            output = self.reduce(output)

        output = map_to_output(output, self._y, fh=actual_index, full_posterior=full_posterior)

        return output

    def _predict(self, fh, X):
        output = self._do_predict(fh, X)
        return output

    def _predict_proba(self, fh, X, marginal=True):
        predictions = self._do_predict(fh, X, full_posterior=True)

        if isinstance(predictions, pd.Series):
            predictions = predictions.to_frame(self._y.name)

        return Empirical(predictions, time_indep=False)

    @contextmanager
    def prior_predictive(self, *, output: OutputType = "np.ndarray") -> Self:
        """
        Does posterior/prior predictive checking.

        Args:
            output:

        Returns:
            Returns samples.
        """

        base_mtype = {"mtype": output}
        attrs_to_override = [
            ("_cutoff", 0),
            ("_is_fitted", True),
            ("_prior_predictive", True),
            ("_y_metadata", base_mtype),
        ]

        current_attrs = [(a, attrgetter(a)(self)) for a, _ in attrs_to_override if hasattr(self, a)]

        try:
            for a, v in attrs_to_override:
                setattr(self, a, v)

            yield self
        except Exception:
            raise
        finally:
            for a, v in current_attrs:
                setattr(self, a, v)

            delta = set([a for a, _ in attrs_to_override]) - set([a for a, _ in current_attrs])
            for d in delta:
                delattr(self, d)

        return

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
