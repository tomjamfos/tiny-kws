from typing import get_args

from tiny_kws import models


def test_models_match_model_names() -> None:

    assert set(models.MODELS) == set(get_args(models.ModelName))
