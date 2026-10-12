# Model selection

**Decision: logistic regression with default settings** (L2 regularisation, `C = 1`, scaled numeric
features, one-hot categories). It clearly beats the best rule, ties the gradient-boosted models, and is
better calibrated, faster and directly explainable. **Runner-up: tuned XGBoost.**

All numbers below are from **5-fold stratified cross-validation on the training set** (5,634 customers).
The test set (1,409 customers) has **not** been used: it is evaluated once, in module 5.
Reproduce with `uv run python -m retention.models.train` and `uv run python -m retention.models.tune`.

## 1. Comparison (same 5 folds for every model, seed 42)

"Top 10%" is the call list (see `docs/problem_framing.md`): about 113 customers per fold.

| Model | PR-AUC | ROC-AUC | Brier ↓ | Precision@10% | Recall@10% | Profit@10% per fold | Fit time |
|---|---|---|---|---|---|---|---|
| **B3 logistic regression** | **0.662** ± 0.022 | **0.846** | **0.135** | 0.754 | 0.285 | $16,052 ± 1,127 | **0.02 s** |
| XGBoost | 0.661 ± 0.018 | 0.843 | 0.136 | 0.758 | 0.286 | $16,159 ± 1,207 | 0.29 s |
| LightGBM | 0.653 ± 0.021 | 0.839 | 0.139 | 0.752 | 0.284 | $15,999 ± 1,020 | 0.66 s |
| B2 contract + tenure | 0.513 ± 0.021 | 0.777 | 0.281 | 0.589 | 0.223 | $11,068 ± 1,207 | — |
| B1 contract | 0.409 ± 0.002 | 0.729 | 0.345 | 0.434 | 0.164 | $6,351 ± 628 | — |
| B0 nobody leaves | 0.265 | 0.500 | 0.265 | 0.262 | 0.099 | $1,152 ± 817 | — |

## 2. Paired comparisons (same folds, so each fold is a head-to-head)

| Comparison | PR-AUC difference | Folds won | Verdict |
|---|---|---|---|
| Logistic regression vs **B2 rule** | **+0.148** ± 0.025 | **5/5** | Clear win: ~6× the noise |
| Logistic regression vs B2 rule, profit@10% | **+$4,985** ± 940 per fold (≈ +45%) | **5/5** | Clear win |
| XGBoost vs logistic regression | −0.0008 ± 0.0065 | 3/5 | **Tie** (8× smaller than the noise) |
| LightGBM vs logistic regression | −0.0089 ± 0.0061 | 0/5 | Slightly worse |

**Success criteria from the framing document:**
- ✅ Beats B2 on PR-AUC and on profit: yes, on every fold.
- ❌ Gradient boosting beats B3 "by a clear margin": no, so **B3 is shipped**, as committed in advance.

## 3. Tuning (Optuna, TPE sampler + median pruning)

| Model | Trials (pruned) | Best CV PR-AUC on its tuning folds | PR-AUC on fresh folds (seed 43) | vs default logistic regression, fresh folds |
|---|---|---|---|---|
| XGBoost | 50 (26) | 0.6719 | 0.6698 ± 0.012 | +0.0064 ± 0.0187, 3/5 folds |
| Logistic regression | 30 (18) | 0.6627 | 0.6634 ± 0.028 | +0.0000 ± 0.0044, 3/5 folds |

- **Tuning optimism:** XGBoost's tuned score fell from 0.6719 (the folds it was tuned on) to 0.6698
  (fresh folds). Part of the "best trial" was luck, which is why the re-check uses different folds.
- **The remaining gain (+0.006) is well inside the fold noise (± 0.019).** Not a reliable difference.
- **What the tuner chose is informative:** XGBoost's best settings were `max_depth = 2`,
  `min_child_weight ≈ 11` and a slow learning rate (0.028): the shallowest, most conservative trees.
  The signal is mostly additive, which a linear model already captures.
- Logistic regression gained nothing from tuning (`C = 0.18`, elastic net), so the default is kept.

## 4. Why logistic regression

| | Logistic regression | Tuned XGBoost |
|---|---|---|
| Ranking quality | Tied | Tied |
| Calibration (Brier) | **0.135**, best | 0.136 |
| Training time | **~14× faster** | — |
| Explainability | **One weight per feature**, readable by the success team | Needs SHAP |
| Size, serving, maintenance | Tiny, few dependencies | Larger, extra native library |

Calibration matters because the profit rule uses the probabilities directly (contact above ~22% risk).

## 5. Observations

- **Capacity, not the model, is the main limit:** the top 10% contains only ~28.5% of all churners
  (recall@10%), even with 75% precision. Calling more customers would catch more churners; that is a
  business decision, explored with the cost curve in module 5.
- **Excluded features:** `gender` and `senior_citizen` are not used. Results per group come in module 5.
- **Rules are a strong floor:** B2 alone finds 59% churners in the top 10%, from two columns.

## 6. What would change this decision

- More data (tens of thousands of customers), or richer features such as usage history, support
  tickets or billing events, where interactions matter: re-run the comparison; boosting may then win.
- A requirement for non-linear patterns that the per-group results (module 5) show logistic regression
  misses.
- A clear-margin win for any model on a future re-run (paired, on fresh folds).
