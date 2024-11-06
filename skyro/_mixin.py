import sys
from contextlib import contextmanager
from functools import cached_property
from operator import attrgetter
from random import randint
from typing import Any, Dict

import jax.numpy as jnp
from jax.random import PRNGKey
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

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
        progress_bar: bool = False,
        kernel_kwargs: Dict[str, Any] = None,
    ):
        self.num_samples = num_samples
        self.num_warmup = num_warmup
        self.num_chains = num_chains
        self.chain_method = chain_method
        self.kernel_kwargs = kernel_kwargs
        self.seed = seed
        self.progress_bar = progress_bar

        self.result_set_: NumpyroResultSet = None

        self._is_vectorized = False
        self._prior_predictive = False

    def build_model(self, *args, **kwargs):
        """
        Builds numpyro model

        Returns:
            Returns nothing.
        """

        raise NotImplementedError("abstract method")

    def _get_key(self) -> PRNGKey:
        return PRNGKey(self.seed or randint(0, 1_000))

    @cached_property
    def mcmc(self) -> MCMC:
        kernel = NUTS(self.build_model, **(self.kernel_kwargs or {}))

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

    @contextmanager
    def prior_predictive(self, **kwargs) -> Self:
        """
        Does posterior/prior predictive checking.

        Returns:
            Returns samples.
        """

        raise NotImplementedError("abstract method")

    def make_output(self, x: Dict[str, jnp.ndarray], context: str = None) -> jnp.ndarray:
        """
        Abstract method overridden by derived classes to format output given returned predictions.

        Args:
            x: Samples.
            context: Returns the context in which the function is called. Can be safely ignored for :class:`sklearn`
                like estimators.

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

        for name, summary_ in s.items():
            if (summary_["r_hat"] <= self.max_rhat).all():
                continue

            raise ConvergenceError(f"Parameter '{name}' did not converge!")

        return NumpyroResultSet(samples, self.group_by_chain, self.num_chains, self.num_samples)
