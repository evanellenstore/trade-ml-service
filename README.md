# Trade ML Service

A Python microservice for feature engineering, dataset generation, and future label creation from the historical market data already produced by the Java history pipeline.

## Purpose

This service is intentionally separate from the Java stack. The Java services continue to own candle backfill, indicator calculation, and historical pipeline orchestration. This Python service is responsible for ML-oriented preparation tasks:

- loading candle + indicator data
- feature engineering
- candlestick pattern features
- target generation
- dataset validation
- leakage prevention

## Architecture

```text
trade-history-service
        │
        ▼
market_candles_backtest
        +
market_indicators_backtest
        │
        │ JOIN ON
        │ candle.id = indicator.candle_id
        ▼
trade-ml-service
        │
        ▼
Feature Engineering
        │
        ├── Price Features
        ├── Volume Features
        ├── Technical Features
        └── Candle Patterns
        │
        ▼
Label Generation
        │
        ▼
ML-ready Dataset
        │
        ▼
Later:
XGBoost / LightGBM
        │
        ▼
Model Evaluation
        │
        ▼
Walk-forward Backtesting
        │
        ▼
Live Prediction
```

## Responsibility split

### Java / trade-history-service

- historical candle fetching
- candle backfill
- market_candles_backtest
- indicator calculation
- indicator backfill
- market_indicators_backtest
- historical orchestration

### Python / trade-ml-service

- loading candle + indicator data
- ML feature engineering
- technical transforms
- candlestick pattern features
- target generation
- BUY/HOLD/SELL label generation
- chronological dataset preparation

## Folder structure

```text
trade-ml-service/
├── app/
│   ├── api/
│   ├── dataset/
│   ├── features/
│   ├── repository/
│   ├── config/
│   ├── schemas/
│   ├── training/
│   ├── prediction/
│   ├── backtesting/
│   └── main.py
├── models/
├── tests/
├── .env.example
├── .gitignore
├── requirements.txt
├── Dockerfile
├── README.md
└── ...
```

## Prerequisites

- Python 3.11+
- MySQL instance with the historical backtest tables populated
- access to the database used by the Java services

## Installation

```bash
cd trade-ml-service
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Set the following variables:

```bash
DB_HOST=localhost
DB_PORT=3306
DB_NAME=tradding_db
DB_USERNAME=root
DB_PASSWORD=ellenverma
LOG_LEVEL=INFO
DEFAULT_PREDICTION_HORIZON_BARS=15
DEFAULT_BUY_THRESHOLD_PCT=0.5
DEFAULT_SELL_THRESHOLD_PCT=-0.5
MODEL_DIRECTORY=./models
```

## Database configuration

This service connects to the existing MySQL tables:

- market_candles_backtest
- market_indicators_backtest

The repository query follows the required relationship:

```sql
SELECT ...
FROM market_candles_backtest m
INNER JOIN market_indicators_backtest i
  ON m.id = i.candle_id
ORDER BY m.symbol_token, m.timeframe, m.candle_time ASC
```

## Starting the API

```bash
uvicorn app.main:app --reload
```

Open Swagger / OpenAPI at:

- http://localhost:8000/docs
- http://localhost:8000/redoc

## Tests

```bash
pytest -q
```

## Timeframe data

`ONE_MINUTE` remains the canonical database timeframe and continues to use the
existing candle-to-indicator join. `FIVE_MINUTE`, `FIFTEEN_MINUTE`,
`THIRTY_MINUTE`, `ONE_HOUR`, `ONE_DAY`, and `ONE_WEEK` are generated in memory
from raw one-minute OHLCV candles, with session-aware aggregation and strict
partial-candle removal by default.

Example five-minute dataset request:

```json
{
        "symbolToken": "14154",
        "tradingStyle": "INTRADAY",
        "timeframe": "FIVE_MINUTE",
        "predictionHorizonBars": 6,
        "buyThresholdPct": 0.15,
        "sellThresholdPct": -0.15
}
```

## Docker

```bash
docker build -t trade-ml-service .
docker run --rm -p 8000:8000 --env-file .env trade-ml-service
```

## Dataset generation flow

1. Retrieve candle + indicator rows for a symbol/timeframe.
2. Sort chronologically.
3. Engineer features.
4. Generate future-return targets and labels.
5. Remove warm-up and invalid rows.
6. Return a dataset summary.

## Feature engineering

This service currently provides:

- price features
- volume features
- technical transforms from existing Java indicator values
- candle pattern flags

## Label generation

Labels are derived from future market movement using the configured prediction horizon.

- BUY when future return >= buy threshold
- SELL when future return <= sell threshold
- HOLD otherwise

## Data leakage rules

The implementation explicitly prevents leakage by:

- only using past/historical data when engineering features
- never allowing future candles into the feature matrix
- excluding target columns from model features
- requiring explicit feature selection rather than using all numeric columns

## Assumptions

This service assumes the Java pipeline already fills the tables with the expected columns and values. The actual indicator columns are not hard-coded beyond the names already present in the Java model. The implementation gracefully handles absent indicator fields by only generating features when the required DB columns exist.
