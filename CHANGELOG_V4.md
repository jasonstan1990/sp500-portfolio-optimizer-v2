# V4 implementation checklist

- [x] Walk-forward backtest
- [x] Transaction costs + turnover + weight drift
- [x] Current S&P 500 universe updater with fallback
- [x] Point-in-time constituent reconstruction
- [x] Point-in-time fundamental snapshot accumulation; price-only safe fallback for unavailable history
- [x] Alpha, Beta, Tracking Error, Information Ratio, Calmar, concentration metrics
- [x] Modular refactor
- [x] Scorecard + workflow UI + sector/drawdown/rolling/turnover charts
- [x] Expanded historical and synthetic stress tests
- [x] Expanded automated tests, including corrupted DB and no-future-fundamentals cases
- [x] Optional GitHub Actions monthly update/deployment data workflow
