import sys
from contextlib import contextmanager
from typing import Any, Dict

import numpy as np
from numpyro.infer import Predictive
from skbase.base import BaseEstimator

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from skyro._mixin import BaseNumpyroMixin


class BaseNumpyroEstimator(BaseNumpyroMixin, BaseEstimator):
    """
    Implements a base class for estimators utilizing :class:`numpyro`.
    """

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
    ):
        super().__init__(
            num_samples=num_samples,
            num_warmup=num_warmup,
            num_chains=num_chains,
            chain_method=chain_method,
            seed=seed,
            progress_bar=progress_bar,
            kernel_kwargs=kernel_kwargs,
        )
        BaseEstimator.__init__(self)

    def build_model(self, X, y=None, **kwargs):
        raise NotImplementedError("abstract method")

    def fit(self, X, y=None):
        key = self._get_key()

        self.mcmc.run(key, X=X, y=y)
        self.result_set_ = self._process_results(self.mcmc)

        self._is_fitted = True

        return

    def _do_sample(self, X, **kwargs) -> Dict[str, np.ndarray]:
        samples = None if self._prior_predictive else self.result_set_.get_samples(group_by_chain=False)
        predictive = Predictive(
            self.build_model, posterior_samples=samples, num_samples=self.num_samples if samples is None else None
        )

        output = predictive(self._get_key(), X=X, **kwargs)

        return {k: np.array(v) for k, v in output.items()}

    def predict(self, X, full_posterior: bool = False, **kwargs):
        ppc = self._do_sample(X, **kwargs)

        output = self.select_output(ppc)

        if full_posterior:
            return output

        return self.reduce(output)

    @contextmanager
    def prior_predictive(self, **kwargs) -> Self:
        try:
            self._prior_predictive = True
            yield self
        except Exception:
            raise
        finally:
            self._prior_predictive = False

        return
