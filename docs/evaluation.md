# Evaluation (module 5)

`customer-retention-model` **v1** (default logistic regression), evaluated **once** on the held-out test set
(1,409 customers, churn rate 26.5%) that no earlier step had seen. MLflow run: `test evaluation v1`.
Reproduce with `uv run python -m retention.evaluation.evaluate` (it refuses a second look without `--force`).

## How it was done

1. **Decisions first, without the test set.** Out-of-fold predictions on the 5,634 training customers (each
   scored by a model that never saw them) were used to check calibration and to choose the contact threshold.
2. **Then the test set, once:** metrics against the B2 rule with **paired bootstrap 95% intervals** (2,000
   resamples of customers), the success criteria from [problem_framing.md](problem_framing.md), the
   policy's profit, results per segment, and SHAP explanations.
3. The verdict and the threshold are written onto the model version, so serving uses what was evaluated.

## Verdict

| | Champion | B2 rule | Difference [95% interval] | P(not better) |
|---|---:|---:|---:|---:|
| **PR-AUC** (primary) | **0.632** | 0.544 | **+0.088** [+0.034, +0.140] | 0.001 |
| **Profit, top 10%** | **$19,678** | $15,927 | **+$3,752** [−$268, +$7,236] | 0.035 |
| ROC-AUC | 0.841 | 0.792 | | |
| Brier (lower is better) | 0.139 | 0.279 | | |
| Precision @ 10% | 0.745 | 0.645 | | |
| Recall @ 10% | 0.281 | 0.243 | | |

Success criteria, read strictly ("beat" = the whole 95% interval above zero):

- Beats B2 on PR-AUC: **yes**.
- Beats B2 on profit for the top 10%: **not conclusively**. +$3,752 (+24%), but the interval just
  crosses zero: in 3.5% of the bootstrap resamples, the champion did not beat B2.

**Result: `passed_success_criteria = false`** (tag on model version 1). The decision whether to ship is open; see
[What this means](#what-this-means).

Test PR-AUC (0.632) is below cross-validation (0.662 ± 0.022). That's within normal sampling variation for 1,409
customers, and the ranking against B2 is the same as in cross-validation, where the model won 5 out of 5 folds
on both PR-AUC (+0.148) and profit (+$4,985 per fold).

## Calibration: can the probabilities be trusted?

| | Out-of-fold (train) | Test |
|---|---:|---:|
| Expected calibration error | 0.013 | 0.023 |
| Mean predicted − observed rate | | +0.002 |

Below the 0.03 tolerance, so **no recalibration step is needed**: "30% risk" means about 30% leave. This matters
because the contact rule multiplies probabilities by money. The B2 rule can't do this at all; its Brier score is
twice as bad because its scores are rankings, not probabilities.

## Decision policy

**Contact a customer if P(leaves) ≥ 0.28, highest risk first, at most 10% of customers per cycle.**

- **Threshold 0.28**, chosen on out-of-fold data as the most profitable cut-off *without* the capacity limit,
  close to the theoretical break-even of 0.224 (as expected for a calibrated model).
- **Capacity 10%** is applied separately. *A note on a bug found and fixed:* the first run chose the threshold
  *with* the limit. Every threshold up to the capacity line then gives the same list, and the tie-break
  returned 0.66 (the 10%-th customer's score). That isn't a decision rule: on a quiet week it would skip
  profitable calls. The fix only changed the policy result; no test metric depends on the threshold.
- **On the test set:** 141 calls, 74% reach a real churner, $19,678 expected profit ($140 per call).

**Capacity is the bottleneck.** Without a limit, profit is maximised by contacting about **41%** of customers
(out-of-fold), and expected profit on the training population rises from **$80k to $184k**, under the assumptions
in `config.py`. That's an argument for more retention staff, or cheaper contact channels for the next tier of risk.

## Segments and fairness

Being contacted brings a discount, so it is a benefit. The fairness question is therefore whether churners in
every group have the same chance of being reached (recall), not whether everyone is flagged at the same rate.

| Group | Customers | Churn rate | Mean predicted | PR-AUC | Recall @ policy |
|---|---:|---:|---:|---:|---:|
| Female | 687 | 0.281 | 0.268 | 0.631 | 0.244 |
| Male | 722 | 0.251 | 0.267 | 0.638 | 0.320 |
| Not senior | 1,187 | 0.233 | 0.243 | 0.612 | 0.268 |
| Senior | 222 | 0.441 | 0.402 | 0.693 | 0.316 |

- **Ranking quality is the same for women and men** (PR-AUC 0.63 vs 0.64), but **female churners are reached less
  often** (recall 0.24 vs 0.32). That gap of 0.08 is under the 0.10 investigation threshold, but it points the
  wrong way, since women churn slightly more, and the model slightly under-predicts their risk (0.268 vs 0.281
  observed). With about 190 female churners in the test set, the gap is noisy. It is tracked as a metric
  (`recall_gap_gender`) and will be monitored in module 10.
- **Seniors**: well ranked (PR-AUC 0.69). Risk is slightly under-predicted (0.40 vs 0.44 observed). Recall is
  similar to non-seniors.
- **Where the model is (and isn't) used:** no one-year, two-year, no-internet or 49+-month customers make the
  list. Their churn rates (3–12%) are far below the threshold, which is the right call. Customers with 13–24
  months of tenure are **over-predicted** (0.35 vs 0.29 observed, 206 customers): the largest calibration gap
  of any segment.

## What drives the score (SHAP)

The champion is linear, so SHAP values are exact: each feature's push on the log-odds of leaving, relative to an
average customer. One-hot columns are summed back to the original feature.

| Feature | Mean \|SHAP\| | | Feature | Mean \|SHAP\| |
|---|---:|---|---|---:|
| tenure | 1.10 | | total_charges | 0.42 |
| internet_service | 0.58 | | streaming_movies | 0.22 |
| contract | 0.52 | | streaming_tv | 0.20 |
| monthly_charges | 0.50 | | online_security | 0.19 |

Typical reasons for the highest-risk customers: *tenure = 1 month; internet = fiber optic; contract =
month-to-month*. New fiber customers on monthly contracts are the core risk group. This matches the EDA, and it
gives the success team something to act on (onboarding, fiber service quality, contract offers).

## What this means

The champion is **clearly better at ranking** (the primary metric), is **calibrated** (so the profit rule
works), and **earns more** on the test set (+24%). It is also the only option that produces probabilities.
The profit gain is **not statistically conclusive on a single test set of 1,409 customers**, and the
pre-registered strict criterion is therefore not met.

Options (a product decision, recorded here once made):

1. **Ship v1** with this result documented: superiority on the primary metric, a consistent but
   inconclusive profit gain, and strong cross-validation evidence (5 out of 5 folds).
2. **Ship the B2 rule**, as the framing doc says to do if no model beats it.
3. **For future versions,** pre-register a two-part criterion before the next evaluation: *superior* on the
   primary metric and *not worse* on profit (for example, the lower bound above −$2,000). This is the usual
   design for a secondary business metric. It must be decided before the next test-set look, not now, to fit this result.

## Limitations

- Test set of 1,409 customers: about 140 contacts, so profit figures are noisy (hence the intervals).
- Profit uses assumed costs and save rate (`config.py`), not measured ones.
- No time dimension (see the framing doc): real performance on future months is likely lower.
