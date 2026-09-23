from __future__ import annotations

import pandas as pd

from app.dataset.swing_target_policy import SwingTargetPolicy
from app.domain.trading_style import TradingStyle


class LongTermTargetPolicy(SwingTargetPolicy):
    trading_style = TradingStyle.LONG_TERM

    def group_columns(self) -> list[str]:
        return ["symbol_token", "timeframe"]
