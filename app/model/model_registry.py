from __future__ import annotations

from pathlib import Path

from app.domain.trading_style import TradingStyle


def model_path_for(
    trading_style: TradingStyle | str,
    model_directory: str | Path,
) -> Path:
    """Return the style-specific model path."""
    normalized_style = TradingStyle.normalize(trading_style)
    filename = f"trade_ml_model_{normalized_style.value}.joblib"
    return Path(model_directory) / filename
