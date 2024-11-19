import joblib
import io

from skyro import BaseNumpyroForecaster


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
