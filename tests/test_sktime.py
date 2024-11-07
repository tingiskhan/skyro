from functools import partial
from io import BytesIO

import jax.numpy as jnp
import joblib
import numpy as np
import numpyro
import pandas as pd
from numpyro.contrib.control_flow import scan
from numpyro.distributions import Normal, TransformedDistribution, HalfNormal
from numpyro.distributions.transforms import SigmoidTransform

from skyro import BaseNumpyroForecaster


def _body_fn(state_tm1, _, alpha, beta, sigma):
    loc = alpha + beta * state_tm1
    x_t = numpyro.sample("y", Normal(loc, sigma))

    return x_t, x_t


class AutoRegressive(BaseNumpyroForecaster):
    """
    Implements a basic Auto-Regressive model of order 1.
    """

    moment_selector = np.mean

    def build_model(self, y, length: int, X=None, future=0, **kwargs):
        # parameters
        mu = numpyro.sample("mu", Normal())
        phi = numpyro.sample("phi", TransformedDistribution(Normal(), SigmoidTransform()))
        sigma = numpyro.sample("sigma", HalfNormal())

        # init
        y_0 = mu + sigma / (1.0 - phi**2.0) ** 0.5 * numpyro.sample("y_0", Normal())

        # observed
        with numpyro.handlers.condition(data={"y": jnp.array(y.values) if y is not None else y}):
            fun = partial(_body_fn, alpha=mu * (1.0 - phi), beta=phi, sigma=sigma)
            _, result = scan(fun, y_0, jnp.arange(0, length + future))

        return

    def select_output(self, x):
        return x["y"]


def test_autoregressive():
    model = AutoRegressive(num_warmup=1_000, num_samples=500, seed=123)

    with model.prior_predictive(output="np.ndarray"):
        to_predict = np.arange(0, 100)
        train = model.predict(to_predict)

        assert train.shape == to_predict.shape

    train = pd.Series(train, index=pd.date_range("2024-01-01", periods=train.shape[0], freq="W"))

    model.fit(train)

    fh = np.arange(1, 12)
    predictions = model.predict(fh)
    assert predictions.shape[0] == fh.shape[0]

    with BytesIO() as f:
        joblib.dump(model, f)

        f.seek(0)
        new_model = joblib.load(f)

    new_predictions = new_model.predict(fh)
    assert new_predictions.index.equals(predictions.index)

    fh = np.arange(1, 12)
    proba = new_model.predict_proba(fh)

    assert proba.shape == (fh.shape[0], 1)
