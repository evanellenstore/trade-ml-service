from __future__ import annotations

from dataclasses import dataclass

from app.domain.trading_style import TradingStyle


@dataclass(frozen=True)
class TargetPolicyMetadata:
    same_session_only: bool
    cross_session_allowed: bool
    horizon_unit: str = "MARKET_BAR"


class TargetPolicyResolver:
    """Central source of truth for target metadata and policy semantics."""

    @staticmethod
    def metadata_for(trading_style: TradingStyle | str) -> TargetPolicyMetadata:
        normalized = TradingStyle.normalize(trading_style)
        if normalized == TradingStyle.INTRADAY:
            return TargetPolicyMetadata(True, False)
        return TargetPolicyMetadata(False, True)

    @staticmethod
    def description_for(trading_style: TradingStyle | str) -> str:
        return TargetPolicyResolver.metadata_for(trading_style).horizon_unit
