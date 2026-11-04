# Case study: PowerCo — SME customer churn & price sensitivity

Domain: Energy utility (gas & electricity). Engagement framing: BCG-style diagnostic.
Source project: Customer-Churn-Analysis (EDA + Random Forest modelling).

## Business situation
PowerCo supplies corporate, SME, and residential customers. After power-market
liberalization in Europe, churn rose sharply in the SME segment. The SME division head
hypothesised that **price changes drive churn** and proposed a **blanket 20% discount**
to customers likely to churn.

## Data
- 14,606 SME clients, 26 raw features (consumption, forecast prices, margins, contract
  dates, sales channel, power subscribed, `has_gas`).
- **Observed churn rate: 9.7% (1,419 / 14,606).** The dataset is imbalanced — a model
  that predicts "nobody churns" is already ~90% accurate.

## Feature engineering
- Price-sensitivity feature: **difference between off-peak prices in Dec and the
  preceding Jan**, plus finer mean/max price changes across off-peak / peak / mid-peak
  periods and months (micro vs macro trends).
- Tenure features from dates: months active, months to contract end, months since last
  product modification, months since renewal (reference: Jan 2016).
- `has_gas` turned into a binary flag; skewed numeric features log/transformed;
  low-frequency `channel_sales` / `origin_up` categories dropped after dummy encoding.

## Model
- Random Forest classifier, `n_estimators=1000`, 75/25 train/test split.
- **Evaluation: Accuracy 0.9033, Precision 0.9166, Recall 0.0589, F1 0.1108, AUC 0.5291,
  K-fold 0.9083.**
- Honest read: the high accuracy is an artifact of the 9.7% base rate. **Recall 0.06 and
  AUC ~0.53 mean the model barely beats random at actually catching churners** — it finds
  almost none of them. This is the single most important caveat in the whole case.

## Key findings
- **Price sensitivity alone does NOT strongly explain churn.** The Dec–Jan price feature
  is weak; the proposed "price is the cause" hypothesis is only partially supported.
- **Top churn drivers (feature importance):** net margin on power subscription and
  12-month consumption; forecasted meter-rental bill (next ~2 months); and **tenure** —
  months active, time since last product update/renewal.
- Clients with **≤4 months tenure churn markedly more**; the jump between 4 and 5 months
  is ~4 percentage points.
- Customers who **also buy gas churn ~2% less** (brand-loyal, multi-product).

## Strategy implications
- A blanket 20% discount is expensive and poorly targeted given weak price signal and
  near-random churn detection. Model it against forecast revenue before rolling out, and
  restrict to the few genuinely high-propensity, high-margin accounts.
- Prioritise **early-life onboarding (first 4 months)**, margin-aware retention, and
  cross-sell to gas as cheaper, better-evidenced levers than an across-the-board discount.
- Fix the model before trusting it: rebalance (class weights / resampling), tune the
  decision threshold, and optimise for recall/PR-AUC, not accuracy.
