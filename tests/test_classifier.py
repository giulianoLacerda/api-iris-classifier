import pytest
import numpy as np
from unittest import mock

from settings import Settings
from v1.modules.classifier import Classifier


def get_pred():
    return np.asarray([0.8, 0.1, 0.1])


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        (
            {
                "sepal_length": 7.7,
                "sepal_width": 2.6,
                "petal_length": 6.9,
                "petal_width": 2.3,
            },
            (0.8, "Iris-setosa"),
        ),
        (None, None),
    ],
)
@mock.patch.object(Classifier, "_predict_proba", return_value=get_pred())
def test_predict(mock_predict_proba, input, expected):
    sett = Settings()
    sett.set_env("dev")
    sett.load_model()
    classifier = Classifier(sett.model)

    out = classifier.predict(input)

    if input:
        assert isinstance(out, tuple)
        assert isinstance(out[0], float)
        assert isinstance(out[1], str)
    assert out == expected
