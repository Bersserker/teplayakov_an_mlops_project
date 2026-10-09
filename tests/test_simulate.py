"""Check reference replay against the API contract and HTTP failures."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import requests

from src.api import app as api
from src.api.models import CreditRequest
from src.model_training.training_data import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.monitoring import simulate


@pytest.fixture
def reference_path(tmp_path, monkeypatch):
    sample = {
        "limit_bal": 50000,
        "sex": 1,
        "education": 2,
        "marriage": 2,
        "age": 46,
        "pay_0": -1,
        **{f"pay_{i}": 0 for i in range(2, 7)},
        **{f"bill_amt{i}": 1000 for i in range(1, 7)},
        **{f"pay_amt{i}": 100 for i in range(1, 7)},
    }
    data = pd.DataFrame(
        [
            {**sample, "id": i, "default": i, "agebin": 3, "avg_exp_1": 0.1}
            for i in range(2)
        ]
    )
    path = tmp_path / "test_reference.csv"
    data.to_csv(path, index=False)
    monkeypatch.setattr(simulate, "TEST_REF_DATA", path)
    return path


def test_replay_sends_valid_requests_and_returns_predictions(
    reference_path, monkeypatch
):
    model = SimpleNamespace(
        feature_names_in_=NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        classes_=[0, 1],
        predict=lambda features: np.array([1]),
        predict_proba=lambda features: np.array([[0.2, 0.8]]),
    )
    monkeypatch.setattr(api, "load_model", lambda: model)

    def post_to_api(url, **kwargs):
        request = CreditRequest.model_validate(kwargs["json"])
        result = api.predict(request)
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(result.model_dump()).encode()
        return response

    post = Mock(side_effect=post_to_api)
    monkeypatch.setattr(simulate.requests, "post", post)
    results = simulate.test_my_api(pd.read_csv(reference_path), port=8123)

    assert post.call_count == 2
    assert post.call_args.args == ("http://127.0.0.1:8123/predict",)
    assert post.call_args.kwargs["timeout"] == 30
    assert results.to_dict(orient="list") == {
        "default": [0, 1],
        "prediction": [1, 1],
        "default_probability": [0.8, 0.8],
    }


def test_replay_stops_on_http_error(reference_path, monkeypatch):
    response = requests.Response()
    response.status_code = 503
    post = Mock(return_value=response)
    monkeypatch.setattr(simulate.requests, "post", post)

    with pytest.raises(requests.HTTPError):
        simulate.test_my_api(pd.read_csv(reference_path))

    assert post.call_count == 1


def test_replay_rejects_empty_reference(reference_path, monkeypatch):
    data = pd.read_csv(reference_path)
    post = Mock()
    monkeypatch.setattr(simulate.requests, "post", post)

    with pytest.raises(ValueError, match="empty"):
        simulate.test_my_api(data.head(0))

    post.assert_not_called()


def test_main_reports_feature_and_probability_psi(reference_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["simulate", "--port", "8123"])
    normal = pd.DataFrame(
        {"default": [0, 1], "prediction": [0, 1], "default_probability": [0.2, 0.8]}
    )
    drifted = normal.copy()
    drifted["default_probability"] = [0.8, 0.8]
    replay = Mock(side_effect=[normal, drifted])
    monkeypatch.setattr(simulate, "test_my_api", replay)

    simulate.main()

    original = replay.call_args_list[0].args[0]
    changed = replay.call_args_list[1].args[0]
    pd.testing.assert_frame_equal(original, pd.read_csv(reference_path))
    pd.testing.assert_series_equal(changed["limit_bal"], original["limit_bal"] * 0.7)
    pd.testing.assert_frame_equal(
        changed.drop(columns="limit_bal"), original.drop(columns="limit_bal")
    )
    assert all(call.kwargs == {"port": 8123} for call in replay.call_args_list)
    output = capsys.readouterr().out
    expected = simulate.psi_calculation([0.2, 0.8], [0.8, 0.8])
    assert "PSI limit_bal:" in output
    assert f"PSI default_probability: {expected:.6f}" in output
