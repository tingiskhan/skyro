from functools import partial
from io import BytesIO

import jax.numpy as jnp
import joblib
import numpy as np
import numpyro
import pandas as pd
import pytest
from numpyro.contrib.control_flow import scan
from numpyro.distributions import Normal, TransformedDistribution, HalfNormal, TruncatedNormal
from numpyro.distributions.transforms import SigmoidTransform
from xarray import DataArray

from skyro import BaseNumpyroForecaster

from numpyro_sts import AutoRegressive as AR


def _body_fn(state_tm1, _, alpha, beta, sigma):
    loc = alpha + beta * state_tm1
    x_t = numpyro.sample("y", Normal(loc, sigma))

    return x_t, x_t


class AutoRegressive(BaseNumpyroForecaster):
    """
    Implements a basic Auto-Regressive model of order 1.
    """

    moment_selector = np.mean
    names_in_trace = ["y"]

    def set_default_tags(self):
        super().set_default_tags()
        tags = {
            "capability:insample": False,
            "capability:pred_int:insample": False,
        }

        self.set_tags(**tags)
        return

    def build_model(self, y, length: int, X=None, future=0, use_mean: bool = True, **kwargs):
        # parameters
        if use_mean:
            mu = numpyro.sample("mu", Normal())
        else:
            mu = 0.0

        phi = numpyro.sample("phi", TransformedDistribution(Normal(), SigmoidTransform()))
        sigma = numpyro.sample("sigma", HalfNormal())

        # init
        y_0 = mu + sigma / (1.0 - phi**2.0) ** 0.5 * numpyro.sample("y_0", Normal())

        # observed
        with numpyro.handlers.condition(data={"y": jnp.array(y.values) if y is not None else y}):
            fun = partial(_body_fn, alpha=mu * (1.0 - phi), beta=phi, sigma=sigma)
            _, result = scan(fun, y_0, jnp.arange(0, length + future))

        return

    def format_output(self, posterior, index):
        return DataArray(posterior["y"], dims=["draw", "time"], coords={"time": index.to_numpy()}, name="y")


@pytest.mark.parametrize("use_mean", [True, False])
def test_autoregressive(use_mean: bool):
    model = AutoRegressive(num_warmup=1_000, num_samples=500, seed=123)

    samples = model.sample_prior_predictive(100)
    train = samples["y"][0]

    assert train.shape == (100,)

    train = pd.Series(train, index=pd.date_range("2024-01-01", periods=train.shape[0], freq="W"))
    fh = np.arange(1, 12)

    with model.set_dynamic_args(use_mean=use_mean):
        model.fit(train)
        predictions = model.predict(fh)

    posterior = model.result_set_.get_samples(group_by_chain=False)

    if use_mean:
        assert "mu" in posterior
    else:
        assert "mu" not in posterior

    assert predictions.shape[0] == fh.shape[0]

    with BytesIO() as f:
        joblib.dump(model, f)

        f.seek(0)
        new_model = joblib.load(f)

    with new_model.set_dynamic_args(use_mean=use_mean):
        new_predictions = new_model.predict(fh)
        proba = new_model.predict_proba(fh)

    assert new_predictions.index.equals(predictions.index)
    assert proba.shape == (fh.shape[0], 1)


def body_fn(state_t, x_tp1, phi, mu):
    vol_tp1 = x_tp1
    y_t = state_t

    y_tp1 = numpyro.sample("y", Normal(mu + phi * (y_t - mu), vol_tp1))

    return y_tp1, y_tp1


class StochasticVolatilityModel(BaseNumpyroForecaster):
    """
    Implements a stochastic volatility model with auto-correlation and a business cycle.
    """

    names_in_trace = ["y"]

    def build_model(self, y, length: int, X=None, future: int = 0, **kwargs):
        with numpyro.handlers.scope(prefix="volatility"):
            mu = numpyro.sample("mu", Normal())
            phi = numpyro.sample("phi", TransformedDistribution(Normal(), SigmoidTransform()))
            sigma = numpyro.sample("sigma", HalfNormal())

            x_0 = mu + sigma / jnp.sqrt(1.0 - phi**2.0) * numpyro.sample("eps_0", Normal())
            log_volatility_model = AR(length, phi, sigma, 1, mu=mu, initial_value=x_0)

            log_volatility = numpyro.sample("x", log_volatility_model)

            if future > 0:
                future_log_volatility = numpyro.sample(
                    "x_future", log_volatility_model.predict(future, log_volatility[-1])
                )
                log_volatility = jnp.concat([log_volatility, future_log_volatility], axis=-2)

            volatility = jnp.exp(log_volatility).squeeze(-1)

        # observed
        mu = numpyro.sample("mu", Normal())
        phi = numpyro.sample("phi", TruncatedNormal(low=-1.0, high=1.0))

        y_0 = mu + jnp.exp(x_0) / jnp.sqrt(1.0 - phi**2.0) * numpyro.sample("eps_0", Normal())
        fun = partial(body_fn, phi=phi, mu=mu)

        with numpyro.handlers.condition(data={"y": jnp.array(y) if y is not None else None}):
            _, y = scan(fun, y_0, volatility)

        return

    def format_output(self, x, horizon):
        return DataArray(
            x["y"],
            dims=["draw", "time"],
            coords={"time": horizon.to_numpy()},
            name="y",
        )


def test_in_sample():
    model = StochasticVolatilityModel(
        num_warmup=50, num_samples=10, num_chains=1, chain_method="parallel", progress_bar=True
    )

    model.max_rhat = float("inf")
    log_returns = model.sample_prior_predictive(length=100)["y"][0]

    index = pd.date_range("2024-01-01", periods=log_returns.shape[0], freq="W")
    log_returns = pd.Series(log_returns, index=index)

    model.fit(log_returns)

    with model.ppc():
        samples = model.predict_proba(log_returns.index)

    return
