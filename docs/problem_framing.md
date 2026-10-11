# Problem framing

Written **before** any modelling, so metrics and success criteria can't be chosen after seeing results.
Numbers marked *(data)* come from the dataset; numbers marked *(assumption)* are business assumptions,
kept in one place (`src/retention/config.py`, from module 3) so they can be changed and re-evaluated.

## 1. Decision supported

Each week the customer-success team can contact **about 100 customers**. The model decides
**which 100 to contact** to save the most revenue.

- Output: a **risk score per customer, used to rank them**, not a yes/no label.
- What matters most is the quality of the **top of the list** (capacity-limited).
- Mode: a **weekly batch** scoring job; an online `/predict` endpoint for single customers (e.g. a CRM view).

## 2. Prediction target

`P(Churn = Yes)`: the probability that a customer has cancelled, from the dataset's `Churn` label.
Base rate: **26.5%** of 7,043 customers *(data)*.

**Limitation (leakage risk):** the dataset is a single snapshot without dates. In production the target would
be *"given what we know on date X, will the customer cancel within the next 30 days?"*, with features built
only from data available before X (point-in-time correctness). This can't be reproduced here, so offline
results are likely optimistic compared with a real time-based split.

## 3. Cost of mistakes

| Assumption | Value | Source |
|---|---|---|
| Monthly revenue of a churning customer | $74.44 | *(data: mean `MonthlyCharges` of churners)* |
| Months a saved customer keeps paying | 12 | *(assumption)* |
| Value of keeping a customer | $74.44 × 12 ≈ **$893** | derived |
| Cost of one retention contact (staff time) | $10 | *(assumption)* |
| Retention discount offered on contact | $50 | *(assumption)* |
| Save rate: share of would-be churners who stay after contact | 30% | *(assumption)* |

Value of each outcome for one contacted customer:

| Outcome | Meaning | Value |
|---|---|---|
| True positive | Contacted, would have left | 0.30 × $893 − $10 − $50 ≈ **+$208** |
| False positive | Contacted, would have stayed | −$10 − $50 = **−$60** |
| False negative | Not contacted, leaves | $0 cost, but a missed **+$208** opportunity |
| True negative | Not contacted, stays | $0 |

**Consequence:** contacting a customer pays off when `p × 0.30 × $893 > $60`, i.e. when the predicted churn
probability **p > ~22%**, not the default 0.5. With a 100-contact capacity, we contact the top 100 by
expected value. This is only valid if probabilities are **calibrated** (see metrics).

## 4. Metrics (fixed now, before training)

- **Primary model metric: PR-AUC** (average precision), which ranks churners above stayers and is robust to
  the 26.5% imbalance.
- **Capacity metrics:** **precision@100** and **recall@100** (the weekly call list).
- **Business metric:** **expected profit of contacting the top 100**, using the table above.
- **Also reported:** ROC-AUC (for comparability) and **Brier score / calibration curve** (probabilities feed
  the profit calculation).
- **Not used: accuracy.** Predicting "nobody leaves" is already **73.5% accurate** *(data)* and useless.

## 5. Baselines and success criteria

| Baseline | Rule | Notes |
|---|---|---|
| B0. Nobody leaves | score = 0 for everyone | Shows why accuracy misleads |
| B1. Contract rule | Flag month-to-month contracts | Flags **3,875 customers** (55%) with precision **0.43**, recall **0.89** *(data)*: far more than the team can contact |
| B2. Contract + tenure rule | Month-to-month first, newest customers (lowest `tenure`) first | **The real competitor for the top-100 list**: what a manager could do in a spreadsheet |
| B3. Logistic regression | Simple, explainable model | If gradient boosting can't clearly beat it, ship the simpler model |

**Success criteria:** a model is shipped only if, on the held-out test set, it beats **B2** on **PR-AUC** and
on **expected profit@100**, and beats **B3** by a clear margin (or B3 is shipped). If no model beats B2, the
rule is shipped.

## 6. Other requirements

| Requirement | Decision |
|---|---|
| Explainability | Each flagged customer comes with its top reasons (SHAP), for the success team |
| Fairness | `gender` and `SeniorCitizen` are excluded as features at first, and model quality is reported for each group so differences are visible |
| Privacy | `customerID` is an identifier, never a feature; personal data stays out of logs |
| Freshness | Weekly batch scoring; retrain monthly, or when drift is detected |
| Latency | Batch: minutes are acceptable. API: p95 < 200 ms per customer |
| Reproducibility | Fixed data version (checksum / DVC), pinned dependencies, logged random seeds |

## 7. Known limitations of this dataset

- No timestamps: no time-based split, no point-in-time features (see section 2).
- Small (7,043 rows): metrics have noticeable variance, so compare models with cross-validation.
- Fictional company (IBM sample): costs and save rate are assumptions, not measured values.
- Data quality: `TotalCharges` is stored as text with **11 blank values** (customers with `tenure = 0`), to be
  handled in module 2.
- No information on past retention offers, so the effect of contacting customers (save rate) can't be learned.
