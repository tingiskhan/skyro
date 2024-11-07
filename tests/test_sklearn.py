import numpy as np
import numpyro
from numpyro.distributions import Normal, LogNormal

from skyro import BaseNumpyroEstimator


class LinearRegression(BaseNumpyroEstimator):

    def build_model(self, X, y=None, **kwargs):
        with numpyro.plate("temp", X.shape[-1]):
            beta = numpyro.sample("beta", Normal())

        alpha = numpyro.sample("alpha", Normal())
        sigma = numpyro.sample("sigma", LogNormal())

        numpyro.sample("y", Normal(alpha + X @ beta, sigma), obs=y)

        return

    def select_output(self, x, context: str = None):
        return x["y"]


def test_sklearn():
    np.random.seed(123)

    x = np.random.normal(size=(1_000, 3))

    weights = np.random.normal(size=3)
    alpha = np.random.normal()
    sigma = np.random.normal() ** 2.0

    y = alpha + x @ weights + sigma * np.random.normal(size=x.shape[0])

    model = LinearRegression(num_samples=1_000, num_warmup=1_000, num_chains=4, seed=123)
    model.fit(x, y)

    # do some predictions
    y_hat = model.predict(x)
    delta = y_hat - y

    y_hat_posterior = model.predict(x, full_posterior=True)

    assert y_hat_posterior.shape == (model.num_samples * model.num_chains,) + y.shape
