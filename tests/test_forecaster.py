import joblib
import io

import pytest

from skyro import BaseNumpyroForecaster, BaseNumpyroMixin


class DummyModel(BaseNumpyroMixin):
    expected_parameters = ["parameter_1"]


def test_forecaster_base_functionality():
    forecaster = BaseNumpyroForecaster(num_warmup=1, num_samples=1)
    params = forecaster.get_params()

    expected = {
        "chain_method": "parallel",
        "kernel_kwargs": None,
        "num_chains": 1,
        "num_samples": 1,
        "progress_bar": True,
        "seed": None,
    }

    for k, v in expected.items():
        assert params[k] == v

    file = io.BytesIO()
    joblib.dump(forecaster, file)

    file.seek(0)
    other = joblib.load(file)

    assert other == forecaster


def test_assert_raises():
    with pytest.raises(ValueError):
        model = DummyModel(num_samples=1, num_warmup=1)

    model = DummyModel(num_samples=1, num_warmup=1, model_kwargs={"parameter_1": 2})
