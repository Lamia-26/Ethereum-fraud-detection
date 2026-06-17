"""Tests du preprocesseur."""

from sklearn.compose import ColumnTransformer

from ethereum_fraud.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from ethereum_fraud.features import build_preprocessor


def test_build_preprocessor_returns_column_transformer():
    preprocessor = build_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer)


def test_preprocessor_has_numeric_and_categorical_transformers():
    preprocessor = build_preprocessor()
    names = [name for name, _, _ in preprocessor.transformers]
    assert "num" in names
    assert "cat" in names


def test_numeric_features_not_empty():
    assert len(NUMERIC_FEATURES) > 0


def test_categorical_features_not_empty():
    assert len(CATEGORICAL_FEATURES) > 0
