# Trade ML Service Flow Diagram

```mermaid
flowchart TD
    A[Client\nPOST /api/v1/model/train] --> B[app/api/model_api.py\ntrain_model]
    B --> C[ModelTrainingService.train]
    C --> D[DatasetGenerator.generate_dataset]

    D --> D1[MarketDataProvider.get_market_data]
    D1 --> D2[Raw OHLC + indicator rows]
    D2 --> D3[DatasetGenerator._validate_and_order]
    D3 --> E[FeatureEngineering.transform]

    E --> E1[add_price_features]
    E --> E2[add_volume_features]
    E --> E3[add_technical_features]
    E --> E4[add_candle_pattern_features]

    E3 --> E5[Map persisted Java indicator columns\n momentum_rsi14 -> rsi\n momentum_macd -> macd\n volatility_atr -> atr_pct\n trend_adx -> adx\n trend_ema20/50 -> ema distance\n volatility_bb -> bollinger features]

    E --> F[FeatureEngineering.feature_matrix\nselect canonical schema columns]
    F --> G[LabelGenerator.generate_labels]
    G --> H[TargetPolicyFactory\nfuture_close / future_return_pct\nBUY / HOLD / SELL]
    H --> I[DatasetGenerator._finalize_dataset]
    I --> I1[Drop invalid features + invalid target rows]

    I1 --> J[PurgedChronologicalDatasetSplitter]
    J --> K[train_df / validation_df / test_df]
    K --> L[XGBoostTrainer.train]
    L --> M[Persist model artifact]
    M --> N[Return dataset summary + split summary + training result]

    subgraph PredictionFlow
        P[Client\nPOST /api/v1/prediction] --> Q[PredictionService.predict_market_data]
        Q --> R[FeatureEngineering.feature_matrix]
        R --> S[Load saved model]
        S --> T[Prediction output]
    end
```

## Runtime flow summary
- Request comes in through [app/api/model_api.py](app/api/model_api.py).
- Training orchestration lives in [app/training/model_training_service.py](app/training/model_training_service.py).
- Dataset assembly happens in [app/dataset/dataset_generator.py](app/dataset/dataset_generator.py).
- Feature engineering happens in [app/features/feature_engineering.py](app/features/feature_engineering.py).
- Indicator conversion happens in [app/features/technical_features.py](app/features/technical_features.py).
- The canonical training schema is defined in [app/features/feature_schema.py](app/features/feature_schema.py).
