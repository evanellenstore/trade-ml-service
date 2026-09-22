import pandas as pd

from app.dataset.dataset_generator import DatasetGenerator


def test_target_excluded_from_feature_list():
    dataset = DatasetGenerator()
    assert "future_close" not in dataset.FEATURE_COLUMNS
    assert "future_return_pct" not in dataset.FEATURE_COLUMNS
    assert "label" not in dataset.FEATURE_COLUMNS


def test_dataset_generator_feature_count_matches_expected_columns():
    dataset = DatasetGenerator()
    feature_columns = dataset.FEATURE_COLUMNS
    assert isinstance(feature_columns, list)
    assert len(feature_columns) > 0
    assert "return_1" in feature_columns
    assert "is_doji" in feature_columns
