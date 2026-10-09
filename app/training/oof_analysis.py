from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support
from pathlib import Path

from app.domain.trading_style import TradingStyle


OOF_CLASS_ORDER = ["SELL", "HOLD", "BUY"]
DEFAULT_CONFIDENCE_THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
DEFAULT_MARGIN_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25]
DEFAULT_COMBINED_THRESHOLDS = [
    {"confidenceThreshold": 0.60, "marginThreshold": 0.10},
    {"confidenceThreshold": 0.65, "marginThreshold": 0.15},
    {"confidenceThreshold": 0.70, "marginThreshold": 0.20},
]


class OOFAnalyzer:
    """Analyze validation-only predictions and simulate their next-candle signals."""

    @staticmethod
    def create_records(
        *,
        validation_rows: pd.DataFrame,
        actual_labels: pd.Series,
        predicted_labels: np.ndarray,
        probabilities: np.ndarray,
        fold_number: int,
    ) -> list[dict[str, Any]]:
        probabilities = np.asarray(probabilities, dtype=float)
        predicted_labels = np.asarray(predicted_labels, dtype=object)
        if probabilities.shape != (len(validation_rows), 3):
            raise ValueError("OOF probability matrix must contain three columns per validation row")
        if len(actual_labels) != len(validation_rows) or len(predicted_labels) != len(validation_rows):
            raise ValueError("OOF labels and probabilities must align with validation rows")
        if not np.isfinite(probabilities).all() or (probabilities < 0.0).any():
            raise ValueError("OOF probabilities contain invalid values")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("OOF probability rows do not sum to approximately 1")

        sorted_probabilities = np.sort(probabilities, axis=1)
        records: list[dict[str, Any]] = []
        for row_index, (_, row) in enumerate(validation_rows.iterrows()):
            sell_probability, hold_probability, buy_probability = probabilities[row_index]
            records.append(
                {
                    "candleId": str(row["candle_id"]),
                    "timestamp": pd.Timestamp(row["featureTimestamp"]).isoformat(),
                    "targetEndTimestamp": pd.Timestamp(row["targetEndTimestamp"]).isoformat(),
                    "tradingDate": str(row["trading_date"]),
                    "foldNumber": int(fold_number),
                    "actualLabel": str(actual_labels.iloc[row_index]),
                    "predictedLabel": str(predicted_labels[row_index]),
                    "sellProbability": float(sell_probability),
                    "holdProbability": float(hold_probability),
                    "buyProbability": float(buy_probability),
                    "confidence": float(sorted_probabilities[row_index, 2]),
                    "secondHighestProbability": float(sorted_probabilities[row_index, 1]),
                    "margin": float(sorted_probabilities[row_index, 2] - sorted_probabilities[row_index, 1]),
                }
            )
        return records

    @staticmethod
    def persist_records(
        records: list[dict[str, Any]],
        *,
        output_directory: str | Path,
        dataset_fingerprint: str,
        walk_forward_fingerprint: str,
    ) -> Path:
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        artifact_path = output_directory / (
            f"{dataset_fingerprint[:12]}_{walk_forward_fingerprint[:16]}_oof.csv.gz"
        )
        pd.DataFrame.from_records(records).to_csv(
            artifact_path,
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )
        return artifact_path

    @staticmethod
    def _classification_metrics(selected: pd.DataFrame, total_count: int) -> dict[str, Any]:
        if selected.empty:
            empty_classes = {
                label: {"precision": None, "recall": None, "f1": None, "predictedCount": 0}
                for label in OOF_CLASS_ORDER
            }
            return {
                "coveragePct": 0.0,
                "predictionCount": 0,
                "metrics": {"accuracy": None, "balancedAccuracy": None, "macroF1": None},
                **empty_classes,
            }

        truth = selected["actualLabel"].astype(str)
        predicted = selected["predictedLabel"].astype(str)
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth,
            predicted,
            labels=OOF_CLASS_ORDER,
            average=None,
            zero_division=0,
        )
        classes = {
            label: {
                "predictedCount": int((predicted == label).sum()),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index, label in enumerate(OOF_CLASS_ORDER)
        }
        return {
            "coveragePct": float(len(selected) / total_count * 100.0) if total_count else 0.0,
            "predictionCount": int(len(selected)),
            "metrics": {
                "accuracy": float(accuracy_score(truth, predicted)),
                "balancedAccuracy": float(balanced_accuracy_score(truth, predicted)),
                "macroF1": float(f1_score(truth, predicted, labels=OOF_CLASS_ORDER, average="macro", zero_division=0)),
            },
            **classes,
        }

    @staticmethod
    def _fold_stability(selected: pd.DataFrame) -> dict[str, Any]:
        fold_macro_f1: list[float] = []
        fold_balanced_accuracy: list[float] = []
        fold_directional_precision: list[float] = []
        fold_class_precision: dict[str, list[float]] = {label: [] for label in OOF_CLASS_ORDER}
        for _, fold_rows in selected.groupby("foldNumber", sort=True):
            truth = fold_rows["actualLabel"].astype(str)
            predicted = fold_rows["predictedLabel"].astype(str)
            fold_macro_f1.append(
                float(f1_score(truth, predicted, labels=OOF_CLASS_ORDER, average="macro", zero_division=0))
            )
            fold_balanced_accuracy.append(float(balanced_accuracy_score(truth, predicted)))
            for label in OOF_CLASS_ORDER:
                predicted_count = int(predicted.eq(label).sum())
                fold_class_precision[label].append(
                    float(((truth == label) & (predicted == label)).sum() / predicted_count)
                    if predicted_count
                    else 0.0
                )
            directional = predicted.isin(["BUY", "SELL"])
            fold_directional_precision.append(
                float((truth[directional] == predicted[directional]).mean()) if directional.any() else 0.0
            )
        if not fold_macro_f1:
            return {
                "foldCount": 0,
                "macroF1StdDev": None,
                "balancedAccuracyStdDev": None,
                "directionalPrecisionStdDev": None,
                "classPrecisionStdDev": {label: None for label in OOF_CLASS_ORDER},
            }
        return {
            "foldCount": len(fold_macro_f1),
            "macroF1StdDev": float(np.std(fold_macro_f1, ddof=1)) if len(fold_macro_f1) > 1 else 0.0,
            "macroF1Min": float(np.min(fold_macro_f1)),
            "macroF1Max": float(np.max(fold_macro_f1)),
            "balancedAccuracyStdDev": float(np.std(fold_balanced_accuracy, ddof=1)) if len(fold_balanced_accuracy) > 1 else 0.0,
            "directionalPrecisionStdDev": float(np.std(fold_directional_precision, ddof=1)) if len(fold_directional_precision) > 1 else 0.0,
            "classPrecisionStdDev": {
                label: float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                for label, values in fold_class_precision.items()
            },
        }

    @staticmethod
    def _attach_execution_prices(
        oof_rows: pd.DataFrame,
        development_rows: pd.DataFrame,
        trading_style: TradingStyle | str,
    ) -> pd.DataFrame:
        required = {"candle_id", "open", "future_close", "targetEndTimestamp", "featureTimestamp"}
        missing = sorted(required - set(development_rows.columns))
        if missing:
            raise ValueError(f"OOF backtest requires development price columns: {missing}")

        ordered = development_rows.copy()
        ordered["candle_id"] = ordered["candle_id"].astype(str)
        ordered["featureTimestamp"] = pd.to_datetime(ordered["featureTimestamp"], errors="raise")
        group_columns = [column for column in ("symbol_token", "timeframe") if column in ordered.columns]
        if TradingStyle.normalize(trading_style) == TradingStyle.INTRADAY:
            if "trading_date" not in ordered.columns:
                raise ValueError("INTRADAY OOF execution requires trading_date")
            group_columns.append("trading_date")
        if not group_columns:
            group_columns = ["candle_id"]

        ordered = ordered.sort_values([*group_columns, "featureTimestamp"], kind="mergesort")
        grouped = ordered.groupby(group_columns, dropna=False, sort=False)
        ordered["nextEligibleOpen"] = grouped["open"].shift(-1)
        ordered["nextEligibleTimestamp"] = grouped["featureTimestamp"].shift(-1)
        open_lookup = ordered.set_index("candle_id")[[
            "nextEligibleOpen",
            "nextEligibleTimestamp",
            "future_close",
        ]]

        result = oof_rows.copy()
        result["candleId"] = result["candleId"].astype(str)
        result = result.join(open_lookup, on="candleId", how="left")
        return result

    @staticmethod
    def _backtest(
        selected: pd.DataFrame,
        *,
        policy: dict[str, Any],
        costs: dict[str, float],
        capital_per_trade: float,
        locked_test_start: pd.Timestamp,
    ) -> dict[str, Any]:
        per_side_rate = sum(costs[key] for key in ("transactionCostPct", "slippagePct", "exchangeFeePct"))
        fixed_brokerage_pct = costs["brokerage"] / capital_per_trade * 100.0
        round_trip_cost_pct = 2.0 * (per_side_rate + fixed_brokerage_pct) + costs["sttPct"]
        directional = selected.loc[selected["predictedLabel"].isin(["BUY", "SELL"])].copy()
        directional["signal"] = np.where(directional["predictedLabel"].eq("BUY"), "BUY_SIGNAL", "SELL_SIGNAL")
        executable = directional.loc[
            directional["nextEligibleOpen"].notna()
            & directional["future_close"].notna()
            & directional["nextEligibleTimestamp"].notna()
        ].copy()
        if not executable.empty:
            executable["targetEndTimestamp"] = pd.to_datetime(executable["targetEndTimestamp"], errors="raise")
            executable["nextEligibleTimestamp"] = pd.to_datetime(executable["nextEligibleTimestamp"], errors="raise")
            executable = executable.loc[
                executable["targetEndTimestamp"].lt(locked_test_start)
                & executable["targetEndTimestamp"].gt(executable["nextEligibleTimestamp"])
            ].sort_values(["nextEligibleTimestamp", "timestamp", "candleId"], kind="mergesort")

        trades: list[dict[str, Any]] = []
        active_until: pd.Timestamp | None = None
        overlapping_skips = 0
        for _, row in executable.iterrows():
            entry_time = pd.Timestamp(row["nextEligibleTimestamp"])
            exit_time = pd.Timestamp(row["targetEndTimestamp"])
            if active_until is not None and entry_time < active_until:
                overlapping_skips += 1
                continue
            entry_price = float(row["nextEligibleOpen"])
            exit_price = float(row["future_close"])
            if not np.isfinite(entry_price) or not np.isfinite(exit_price) or entry_price <= 0.0:
                continue
            direction = 1.0 if row["predictedLabel"] == "BUY" else -1.0
            gross_return_pct = direction * (exit_price - entry_price) / entry_price * 100.0
            net_return_pct = gross_return_pct - round_trip_cost_pct
            trades.append(
                {
                    "signal": row["signal"],
                    "grossReturnPct": gross_return_pct,
                    "netReturnPct": net_return_pct,
                    "costPct": round_trip_cost_pct,
                    "exitTime": exit_time,
                }
            )
            active_until = exit_time

        if not trades:
            return {
                **policy,
                "signalCount": int(len(directional)),
                "buySignalCount": int(directional["predictedLabel"].eq("BUY").sum()),
                "sellSignalCount": int(directional["predictedLabel"].eq("SELL").sum()),
                "tradeCount": 0,
                "unexecutableSignalCount": int(len(directional) - len(executable)),
                "overlappingSignalCount": overlapping_skips,
                "buyTradeCount": 0,
                "sellTradeCount": 0,
                "winRate": None,
                "averageReturnPct": None,
                "averageGrossReturnPct": None,
                "averageWinPct": None,
                "averageLossPct": None,
                "grossReturnPct": 0.0,
                "netReturnPct": 0.0,
                "roundTripCostPct": round_trip_cost_pct,
                "profitFactor": None,
                "profitFactorStatus": "NO_TRADES",
                "maxDrawdownPct": 0.0,
                "expectancyPct": None,
            }

        gross_returns = np.asarray([trade["grossReturnPct"] for trade in trades], dtype=float)
        net_returns = np.asarray([trade["netReturnPct"] for trade in trades], dtype=float)
        wins = net_returns[net_returns > 0.0]
        losses = net_returns[net_returns < 0.0]
        equity = np.cumprod(1.0 + net_returns / 100.0)
        peaks = np.maximum.accumulate(np.concatenate(([1.0], equity)))
        drawdowns = (peaks[1:] - equity) / peaks[1:]
        gross_equity = float(np.prod(1.0 + gross_returns / 100.0))
        net_equity = float(equity[-1])
        profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) else None
        return {
            **policy,
            "signalCount": int(len(directional)),
            "buySignalCount": int(directional["predictedLabel"].eq("BUY").sum()),
            "sellSignalCount": int(directional["predictedLabel"].eq("SELL").sum()),
            "tradeCount": int(len(trades)),
            "unexecutableSignalCount": int(len(directional) - len(executable)),
            "overlappingSignalCount": int(overlapping_skips),
            "buyTradeCount": int(sum(trade["signal"] == "BUY_SIGNAL" for trade in trades)),
            "sellTradeCount": int(sum(trade["signal"] == "SELL_SIGNAL" for trade in trades)),
            "winRate": float((net_returns > 0.0).mean() * 100.0),
            "averageReturnPct": float(net_returns.mean()),
            "averageGrossReturnPct": float(gross_returns.mean()),
            "averageWinPct": float(wins.mean()) if len(wins) else 0.0,
            "averageLossPct": float(losses.mean()) if len(losses) else 0.0,
            "grossReturnPct": float((gross_equity - 1.0) * 100.0),
            "netReturnPct": float((net_equity - 1.0) * 100.0),
            "roundTripCostPct": round_trip_cost_pct,
            "profitFactor": profit_factor,
            "profitFactorStatus": "NO_LOSSES" if not len(losses) else "CALCULATED",
            "maxDrawdownPct": float(drawdowns.max() * 100.0) if len(drawdowns) else 0.0,
            "expectancyPct": float(net_returns.mean()),
        }

    @staticmethod
    def analyze(
        *,
        records: list[dict[str, Any]],
        development_rows: pd.DataFrame,
        trading_style: TradingStyle | str,
        locked_test_start: str | pd.Timestamp,
        confidence_thresholds: list[float] | None = None,
        margin_thresholds: list[float] | None = None,
        combined_thresholds: list[dict[str, float]] | None = None,
        costs: dict[str, float] | None = None,
        capital_per_trade: float = 100000.0,
        minimum_trade_count: int = 30,
        maximum_drawdown_pct: float = 25.0,
        prediction_artifact_path: str | None = None,
    ) -> dict[str, Any]:
        confidence_thresholds = confidence_thresholds or DEFAULT_CONFIDENCE_THRESHOLDS
        margin_thresholds = margin_thresholds or DEFAULT_MARGIN_THRESHOLDS
        combined_thresholds = combined_thresholds or DEFAULT_COMBINED_THRESHOLDS
        costs = costs or {
            "brokerage": 0.0,
            "transactionCostPct": 0.0,
            "slippagePct": 0.0,
            "sttPct": 0.0,
            "exchangeFeePct": 0.0,
        }
        oof_rows = pd.DataFrame.from_records(records)
        if oof_rows.empty:
            raise ValueError("WALK_FORWARD_OOF_EMPTY: no validation predictions were collected")
        oof_rows["timestamp"] = pd.to_datetime(oof_rows["timestamp"], errors="raise")
        if oof_rows["candleId"].duplicated().any():
            raise ValueError("WALK_FORWARD_OOF_DUPLICATE: a candle has multiple OOF predictions")
        oof_rows = oof_rows.sort_values(["timestamp", "candleId"], kind="mergesort").reset_index(drop=True)
        total_count = len(oof_rows)

        confidence_analysis = []
        confidence_policies: list[tuple[pd.DataFrame, dict[str, Any]]] = []
        for threshold in confidence_thresholds:
            selected = oof_rows.loc[oof_rows["confidence"] >= threshold].copy()
            analysis = {
                "threshold": float(threshold),
                **OOFAnalyzer._classification_metrics(selected, total_count),
                "foldStability": OOFAnalyzer._fold_stability(selected),
            }
            confidence_analysis.append(analysis)
            confidence_policies.append((selected, {"policyType": "CONFIDENCE", "confidenceThreshold": float(threshold)}))

        margin_analysis = []
        for threshold in margin_thresholds:
            selected = oof_rows.loc[oof_rows["margin"] >= threshold]
            margin_analysis.append(
                {
                    "threshold": float(threshold),
                    **OOFAnalyzer._classification_metrics(selected, total_count),
                    "foldStability": OOFAnalyzer._fold_stability(selected),
                }
            )

        combined_analysis = []
        combined_policies: list[tuple[pd.DataFrame, dict[str, Any]]] = []
        for thresholds in combined_thresholds:
            confidence = float(thresholds["confidenceThreshold"])
            margin = float(thresholds["marginThreshold"])
            selected = oof_rows.loc[
                (oof_rows["confidence"] >= confidence) & (oof_rows["margin"] >= margin)
            ].copy()
            policy = {
                "policyType": "CONFIDENCE_AND_MARGIN",
                "confidenceThreshold": confidence,
                "marginThreshold": margin,
            }
            analysis = {
                **policy,
                **OOFAnalyzer._classification_metrics(selected, total_count),
                "foldStability": OOFAnalyzer._fold_stability(selected),
            }
            baseline = next(
                (item for item in confidence_analysis if item["threshold"] == confidence),
                None,
            )
            for label in ("BUY", "SELL"):
                baseline_precision = baseline[label]["precision"] if baseline else None
                selected_precision = analysis[label]["precision"]
                analysis[f"{label.lower()}PrecisionDelta"] = (
                    float(selected_precision - baseline_precision)
                    if selected_precision is not None and baseline_precision is not None
                    else None
                )
            baseline_directional_count = (
                baseline["BUY"]["predictedCount"] + baseline["SELL"]["predictedCount"]
                if baseline
                else 0
            )
            selected_directional_count = analysis["BUY"]["predictedCount"] + analysis["SELL"]["predictedCount"]
            baseline_directional_precision = (
                (
                    baseline["BUY"]["precision"] * baseline["BUY"]["predictedCount"]
                    + baseline["SELL"]["precision"] * baseline["SELL"]["predictedCount"]
                )
                / baseline_directional_count
                if baseline and baseline_directional_count
                else None
            )
            selected_directional_precision = (
                (
                    analysis["BUY"]["precision"] * analysis["BUY"]["predictedCount"]
                    + analysis["SELL"]["precision"] * analysis["SELL"]["predictedCount"]
                )
                / selected_directional_count
                if selected_directional_count
                else None
            )
            analysis["directionalPrecisionDelta"] = (
                float(selected_directional_precision - baseline_directional_precision)
                if selected_directional_precision is not None and baseline_directional_precision is not None
                else None
            )
            combined_analysis.append(analysis)
            combined_policies.append((selected, policy))

        price_rows = OOFAnalyzer._attach_execution_prices(oof_rows, development_rows, trading_style)
        test_start = pd.Timestamp(locked_test_start)
        analysis_by_key: dict[tuple[Any, ...], dict[str, Any]] = {
            ("CONFIDENCE", item["threshold"], None): item for item in confidence_analysis
        }
        analysis_by_key.update(
            (
                ("CONFIDENCE_AND_MARGIN", item["confidenceThreshold"], item["marginThreshold"]),
                item,
            )
            for item in combined_analysis
        )
        backtests: list[dict[str, Any]] = []
        for selected, policy in [*confidence_policies, *combined_policies]:
            key = (
                policy["policyType"],
                policy["confidenceThreshold"],
                policy.get("marginThreshold"),
            )
            selected_ids = set(selected["candleId"].astype(str))
            policy_rows = price_rows.loc[price_rows["candleId"].isin(selected_ids)]
            backtest = OOFAnalyzer._backtest(
                policy_rows,
                policy=policy,
                costs=costs,
                capital_per_trade=capital_per_trade,
                locked_test_start=test_start,
            )
            backtests.append(backtest)

        signal_policies = []
        for backtest in backtests:
            label = (
                f"confidence>={backtest['confidenceThreshold']:.2f}"
                if backtest["policyType"] == "CONFIDENCE"
                else f"confidence>={backtest['confidenceThreshold']:.2f} AND margin>={backtest['marginThreshold']:.2f}"
            )
            signal_policies.append(
                {
                    **{key: value for key, value in backtest.items() if key in {"policyType", "confidenceThreshold", "marginThreshold"}},
                    "name": label,
                    "buyRule": "predictedLabel == BUY and all policy thresholds pass",
                    "sellRule": "predictedLabel == SELL and all policy thresholds pass",
                    "holdRule": "NO_TRADE",
                    "buySignalCount": backtest["buySignalCount"],
                    "sellSignalCount": backtest["sellSignalCount"],
                    "tradeCount": backtest["tradeCount"],
                }
            )

        ranked = []
        for backtest in backtests:
            analysis = analysis_by_key[
                (
                    backtest["policyType"],
                    backtest["confidenceThreshold"],
                    backtest.get("marginThreshold"),
                )
            ]
            directional_count = analysis["BUY"]["predictedCount"] + analysis["SELL"]["predictedCount"]
            directional_precision = (
                (
                    analysis["BUY"]["precision"] * analysis["BUY"]["predictedCount"]
                    + analysis["SELL"]["precision"] * analysis["SELL"]["predictedCount"]
                )
                / directional_count
                if directional_count
                else 0.0
            )
            eligible = (
                backtest["tradeCount"] >= minimum_trade_count
                and backtest["netReturnPct"] is not None
                and backtest["netReturnPct"] > 0.0
                and backtest["maxDrawdownPct"] <= maximum_drawdown_pct
            )
            ranked.append(
                {
                    "policy": backtest,
                    "analysis": analysis,
                    "eligible": eligible,
                    "directionalPrecision": float(directional_precision),
                    "foldStability": analysis["foldStability"],
                }
            )
        ranked.sort(
            key=lambda item: (
                item["eligible"],
                item["policy"]["netReturnPct"] or 0.0,
                item["directionalPrecision"],
                item["policy"]["winRate"] or 0.0,
                -(item["policy"]["maxDrawdownPct"] or 0.0),
                -(item["foldStability"]["macroF1StdDev"] or 0.0),
                -(item["foldStability"]["directionalPrecisionStdDev"] or 0.0),
            ),
            reverse=True,
        )

        def policy_recommendation(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
            observed = [item for item in candidates if item["policy"]["tradeCount"] > 0]
            if not observed:
                return None
            winner = observed[0]
            return {
                "recommended": bool(winner["eligible"]),
                "policyType": winner["policy"]["policyType"],
                "confidenceThreshold": winner["policy"]["confidenceThreshold"],
                "marginThreshold": winner["policy"].get("marginThreshold"),
                "tradeCount": winner["policy"]["tradeCount"],
                "netReturnPct": winner["policy"]["netReturnPct"],
                "winRate": winner["policy"]["winRate"],
                "profitFactor": winner["policy"]["profitFactor"],
                "maxDrawdownPct": winner["policy"]["maxDrawdownPct"],
                "directionalPrecision": winner["directionalPrecision"],
                "reason": (
                    "Passed trade-count, positive-net-return, and drawdown guards; ranked by net return, directional precision, win rate, and drawdown."
                    if winner["eligible"]
                    else "No policy passed all profitability, trade-count, and drawdown guards."
                ),
            }

        recommendation = policy_recommendation(ranked)
        best_confidence_policy = policy_recommendation(
            [item for item in ranked if item["policy"]["policyType"] == "CONFIDENCE"]
        )
        best_combined_policy = policy_recommendation(
            [item for item in ranked if item["policy"]["policyType"] == "CONFIDENCE_AND_MARGIN"]
        )

        return {
            "predictionCount": int(total_count),
            "oofPredictionCount": int(total_count),
            "predictionArtifact": prediction_artifact_path,
            "confidenceAnalysis": confidence_analysis,
            "marginAnalysis": margin_analysis,
            "combinedAnalysis": combined_analysis,
            "signalPolicies": signal_policies,
            "backtests": backtests,
            "recommendation": recommendation,
            "bestConfidencePolicy": best_confidence_policy,
            "bestCombinedPolicy": best_combined_policy,
            "backtestAssumptions": {
                "entry": "next eligible candle open",
                "exit": "target horizon future_close at targetEndTimestamp",
                "positionPolicy": "one open position at a time; overlapping signals are skipped",
                "costModel": "transactionCostPct, slippagePct, and exchangeFeePct apply on each entry/exit side; sttPct applies once on exit; fixed brokerage currency is charged per side and converted using capitalPerTrade",
                "percentageUnit": "percentage points, so 0.01 means 0.01%",
                "winRateUnit": "percent",
                "brokerageUnit": "currency units per order side",
                "costs": costs,
                "capitalPerTrade": float(capital_per_trade),
                "minimumTradeCount": int(minimum_trade_count),
                "maximumDrawdownPct": float(maximum_drawdown_pct),
            },
        }