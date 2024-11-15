import sys
from functools import cached_property
from operator import attrgetter
from random import randint
from typing import Any, Dict

import numpy as np
from jax.random import PRNGKey
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS
from numpyro.infer.mcmc import MCMCKernel

if sys.version_info >= (3, 11):
    pass
else:
    pass

from ._result import NumpyroResultSet
from .exc import ConvergenceError

# TODO: use this https://forum.pyro.ai/t/parallelising-numpyro/2442/10


class BaseNumpyroMixin:
    """
    Implements a base mixin for :class:`numpyro` models.

    Args:
        See :class:`MCMC`.
    """

    group_by_chain: bool = True
    max_rhat: float = 1.1

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
        self.num_samples = num_samples
        self.num_warmup = num_warmup
        self.num_chains = num_chains
        self.chain_method = chain_method
        self.kernel_kwargs = kernel_kwargs
        self.seed = seed
        self.progress_bar = progress_bar

        self.model_kwargs = model_kwargs

        self.result_set_: NumpyroResultSet = None

        self._is_vectorized = False

    def reduce(self, posterior: np.ndarray) -> np.ndarray:
        """
        Reduces the prediction posterior to a single moment.

        Args:
            posterior: Posterior of the variable.

        Returns:
            Returns a :class:`numpy.ndarray`.
        """

        return np.mean(posterior, axis=0)

    def build_model(self, *args, **kwargs):
        """
        Builds numpyro model

        Returns:
            Returns nothing.
        """

        raise NotImplementedError("abstract method")

    def _get_key(self) -> PRNGKey:
        return PRNGKey(self.seed or randint(0, 1_000))

    def build_kernel(self, **kwargs) -> MCMCKernel:
        """
        Utility for building model specific kernel. Otherwise defaults to NUTS.

        Args:
            **kwargs: Kwargs passed in class' __init__.

        Returns:
            Returns a :class:`MCMCKernel`.
        """

        return NUTS(self.build_model, **kwargs)

    @cached_property
    def mcmc(self) -> MCMC:
        kernel = self.build_kernel(**(self.kernel_kwargs or {}))

        mcmc = MCMC(
            kernel,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
            chain_method=self.chain_method,
            progress_bar=self.progress_bar,
        )

        return mcmc

    # TODO: consider building from samples instead of MCMC
    def to_idata(self):
        """
        Returns the MCMC object as an :class:`InferenceData` object.

        Returns:
            Returns :class:`arviz.InferenceData` object.
        """

        import arviz

        self.check_is_fitted()
        return arviz.from_numpyro(self.mcmc)

    def __getstate__(self):
        if hasattr(super(), "__getstate__"):
            state = super().__getstate__()
        else:
            state = self.__dict__.copy()

        result_set = state.pop("result_set_", None)
        state.pop("mcmc", None)

        if result_set:
            state["result_set_"] = result_set.__getstate__()

        return state

    def __setstate__(self, state):
        # Check if the superclass has __setstate__
        result_set_state = state.pop("result_set_", None)

        if hasattr(super(), "__setstate__"):
            super().__setstate__(state)
        else:
            self.__dict__.update(state)

        if result_set_state:
            result_set = NumpyroResultSet.__new__(NumpyroResultSet)
            result_set.__setstate__(result_set_state)
            self.result_set_ = result_set

        return

    def sample_prior_predictive(self, **kwargs) -> Dict[str, np.ndarray]:
        """
        Samples from the prior predictive density.

        Returns:
            Returns samples.
        """

        raise NotImplementedError("abstract method")

    def _process_results(self, mcmc: MCMC) -> NumpyroResultSet:
        # fetch sample sites
        sites = attrgetter(mcmc._sample_field)(mcmc._last_state)

        samples = mcmc.get_samples(group_by_chain=self.group_by_chain)

        sub_samples = {k: v for k, v in samples.items() if k in sites}
        s = summary(sub_samples, group_by_chain=self.group_by_chain)

        # TODO: need to handle case when some of the dimensions of the variables are nan
        for name, summary_ in s.items():
            # NB: some variables have deterministic elements (s.a. samples from LKJCov).
            mask = np.isnan(summary_["n_eff"])
            r_hat = summary_["r_hat"][~mask]

            if (r_hat <= self.max_rhat).all():
                continue

            raise ConvergenceError(f"Parameter '{name}' did not converge!")

        return NumpyroResultSet(samples, self.group_by_chain, self.num_chains, self.num_samples)
