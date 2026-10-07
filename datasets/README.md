# Datasets

Names are date-independent. Files are unchanged except for their names.

| Dataset | CSV | Rows | Start | End | Frequency |
|---|---|---:|---|---|---|
| AAPL | `datasets/AAPL/AAPL.csv` | 1,255 | 2020-06-05 | 2025-06-03 | daily trading observations |
| Amazon | `datasets/Amazon/Amazon.csv` | 1,510 | 2017-07-31 | 2023-07-31 | daily trading observations |
| Google | `datasets/Google/Google.csv` | 1,510 | 2017-07-31 | 2023-07-31 | daily trading observations |
| Nasdaq | `datasets/Nasdaq/Nasdaq.csv` | 1,510 | 2017-07-31 | 2023-07-31 | daily trading observations |
| NYSE | `datasets/NYSE/NYSE.csv` | 1,510 | 2017-07-31 | 2023-07-31 | daily trading observations |
| CSI300 | `datasets/CSI300/CSI300.csv` | 1,459 | 2017-07-31 | 2023-07-31 | daily trading observations |
| SP500 | `datasets/SP500/SP500.csv` | 1,510 | 2017-07-31 | 2023-07-31 | daily trading observations |
| BTC | `datasets/BTC/BTC.csv` | 600,526 | 2025-01-07 | 2026-02-28 | 1 minute |

Dates for BTC are UTC. Daily dates are trading-date labels.
The Nasdaq filename corrects the legacy spelling `Nasdaque`; CSI300 and SP500 omit spaces and punctuation.
Each daily CSV needs Date and Close. BTC needs timestamp (Unix seconds) and close. Extra OHLCV columns are preserved.
No missing entries or duplicate timestamps were found. BTC contains consecutive one-minute observations.
