from __future__ import annotations


class FeatureSchema:
    """Canonical model feature contract used for generation and prediction."""

    version = "v1"
    columns = [
        "return_1",
        "return_5",
        "return_15",
        "price_range_pct",
        "body_size_pct",
        "upper_wick_pct",
        "lower_wick_pct",
        "volume_change_pct",
        "rolling_volume_mean",
        "relative_volume",
        "is_doji",
        "is_hammer",
        "is_shooting_star",
        "is_bullish_engulfing",
        "is_bearish_engulfing",
    ]

    @classmethod
    def list(cls) -> list[str]:
        return list(cls.columns)

    @classmethod
    def as_dict(cls) -> dict[str, object]:
        return {"version": cls.version, "count": len(cls.columns), "features": list(cls.columns)}


FEATURE_SCHEMA = FeatureSchema()
