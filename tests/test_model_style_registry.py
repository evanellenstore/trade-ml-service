from pathlib import Path

import pytest

from app.domain.trading_style import TradingStyle
from app.model.model_registry import model_path_for


@pytest.mark.parametrize(
    ("style", "filename"),
    [
        (TradingStyle.INTRADAY, "trade_ml_model_INTRADAY.joblib"),
        (TradingStyle.SWING, "trade_ml_model_SWING.joblib"),
        (TradingStyle.LONG_TERM, "trade_ml_model_LONG_TERM.joblib"),
    ],
)
def test_model_path_for_style(style: TradingStyle, filename: str) -> None:
    path = model_path_for(style, Path("/tmp/models"))

    assert path == Path(f"/tmp/models/{filename}")


def test_model_path_for_rejects_invalid_style() -> None:
    with pytest.raises(ValueError, match="Unsupported tradingStyle"):
        model_path_for("INVALID", Path("/tmp/models"))
