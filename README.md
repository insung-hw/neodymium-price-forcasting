# Neodymium Price Forecasting

**Forecasting neodymium (Nd) prices with external factors (policy and trade events) and assessing supply chain risk**

- Research report for the Sejong University × Gachon University Joint Academic Festival (Aug 2025)
- Team "AI 길들이기" (2 members) · Team lead Insung Hwang: topic selection, data collection, modeling, report writing
- Report: [thesis.pdf](thesis.pdf)
  *Assessing Supply Chain Vulnerability and Price Volatility in the Neodymium Market: Evidence from Machine Learning Models*

---

## 1. Background and Goals

Neodymium is the key raw material for NdFeB permanent magnets used in EV motors and wind turbines. Because refining and processing are concentrated in China, its price is highly sensitive to events such as export controls.

1. Build a neodymium price forecasting model that reflects production, demand, and policy/trade events
2. Forecast 12-month price paths under three scenarios in which China's export controls are **intensified, maintained, or relaxed**, and propose procurement responses

## 2. Data

Monthly, Aug 2015 – Jul 2025 (120 observations)

| Category | Variable | Source |
|---|---|---|
| Price | Neodymium metal price (China FOB, 99%) | Trading Economics (cross-checked with SMM, Argus) |
| Supply | China Nd metal exports, estimated global exports | China Customs, IEA |
| Demand proxy | Permanent magnet exports (HS 850511, 850519) | UN Comtrade |
| Demand proxy | Wind energy index | Trading Economics |
| Demand proxy | New energy vehicle (EV, PHV, FCV) sales | MarkLines |
| Events | Rare earth news event score | BIGKINDS |

> The raw data file (`data.csv`) is not included in this repository.

### News event score
News articles were collected from BIGKINDS with the keywords "rare earths" and "neodymium", and duplicate reports of the same event were reduced to one article. Each article was scored with **Gemini-2.5-pro** on price-impact **direction** (up / down / neutral), **magnitude** (1–3), **channel** (direct / indirect), **duration** (half-life of 7–60 days), and **confidence** (0–1). Scores were aggregated at each month-end with exponential decay so that an event's effect carries over into following months; contributions below 1% of their initial value were dropped, and the series was standardized over the full sample. (Report, Section 3.2 and Appendix A)

## 3. Method

**Validation design: training and testing across COVID-19 regimes**

| Design | Train | Test |
|---|---|---|
| R1 | Pre-COVID (Aug 2015 – Sep 2020) | COVID impact (Oct 2020 – Apr 2023) |
| R2 | COVID impact | Recovery (May 2023 –) |
| R3 | Pre-COVID + COVID impact | Recovery |

- **7 variable sets**: price only (C0), all covariates (Call), and leave-one-out sets removing one group at a time (production, magnet exports, wind, NEV, events)
- **4 models**: Random Forest, XGBoost, LightGBM, CatBoost (tree ensembles chosen over deep learning because the series has only 120 monthly points)
- 3 × 7 × 4 = **84 models** compared
- **Metrics**: MAE, RMSE, MAPE, MASE

## 4. Key Results (XGBoost)

| Design | Result |
|---|---|
| R1 | Structural break: MAPE about 51% or higher in every setting |
| R2 | All covariates MAPE 64.8% → **15.1% without the event variable** |
| R3 | All covariates **MAPE 9.2%**; 53.5% without the wind energy index |

- Training on both pre-COVID and COVID data forecast the recovery period best, and the **wind energy index** emerged as the key medium-term demand driver
- Event patterns learned during the shock period degraded forecasts after normalization → event information is better used for **scenario analysis** than for routine forecasting

### 12-month export control scenarios (forecast origin Jul 2025, spot 650,000, median)

| Scenario | 12-month mean | Change | Month 12 |
|---|---|---|---|
| Intensification | 590,316 | −9.2% | 669,693 |
| Maintenance | 612,839 | −5.7% | 720,397 |
| Relaxation | 602,732 | −7.3% | 705,737 |

- The three paths move together for about 3–6 months and diverge from months 6–9 → prices depend on how long controls **persist** rather than on announcements alone
- Proposed responses: short term, hold 3–4 months of safety stock and lock in part of the volume at current prices; medium term, adjust the share of forward contracts by whether controls persist; long term, diversify suppliers and invest in magnet recycling

## 5. Repository Contents

| File | Description |
|---|---|
| [`thesis.pdf`](thesis.pdf) | Research report |
| [`tree-forcasting-with-scenario.ipynb`](tree-forcasting-with-scenario.ipynb) | Main model in the report: R1–R3 × variable sets × 4 tree models, covariate forecasting, scenario forecasting |
| [`forcasting_naive_ets_no-event.ipynb`](forcasting_naive_ets_no-event.ipynb) | Exploratory: Naive, exponential smoothing, seasonal decomposition, NeuralProphet |
| [`forcasting_transformer.ipynb`](forcasting_transformer.ipynb), [`forcasting_transformer_no-event.ipynb`](forcasting_transformer_no-event.ipynb) | Exploratory: Transformer (with / without the event variable) |
| [`linear-regression.ipynb`](linear-regression.ipynb) | Exploratory: EDA, linear regression and regression diagnostics |

> Results in the report come from `tree-forcasting-with-scenario.ipynb`; the other notebooks are exploratory experiments run before model selection.

**Main libraries**: Python, pandas, numpy, darts (RandomForest, XGBoost, LightGBM, CatBoost models), scikit-learn, statsmodels, matplotlib

## 6. Limitations and Next Steps

- Small sample (120 monthly points); some variables rely on proxies and interpolated estimates
- No hyperparameter tuning or time-series cross-validation
- News events limited to Korean-language articles
- Next steps: time-series cross-validation with systematic tuning, multilingual news and policy calendars, regime-switching models