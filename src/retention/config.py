"""Every business assumption and modelling setting, in one place (see docs/problem_framing.md).

Change a number here and re-run: metrics, thresholds and profit figures all follow.
"""

from dataclasses import dataclass

from retention.data.schema import CATEGORY_COLUMNS

SEED = 42  # one seed for the split, cross-validation folds and every model
TEST_SIZE = 0.2
CV_FOLDS = 5


@dataclass(frozen=True)
class BusinessAssumptions:
    """The value of a retention contact. Marked (data) or (assumption) as in the framing doc."""

    monthly_revenue: float = 74.44  # (data) mean MonthlyCharges of customers who churned
    months_retained: int = 12  # (assumption) how long a saved customer keeps paying
    contact_cost: float = 10.0  # (assumption) staff time per contact
    discount: float = 50.0  # (assumption) retention offer made on contact
    save_rate: float = 0.30  # (assumption) share of would-be churners who stay after a contact

    @property
    def value_of_saving(self) -> float:
        return self.monthly_revenue * self.months_retained

    @property
    def cost_per_contact(self) -> float:
        return self.contact_cost + self.discount

    @property
    def true_positive_value(self) -> float:
        """Expected value of contacting someone who would have left."""
        return self.save_rate * self.value_of_saving - self.cost_per_contact

    @property
    def false_positive_value(self) -> float:
        """Value of contacting someone who would have stayed (negative: a wasted contact)."""
        return -self.cost_per_contact

    @property
    def break_even_probability(self) -> float:
        """Contacting pays off when the predicted churn probability is above this."""
        return self.cost_per_contact / (self.save_rate * self.value_of_saving)


BUSINESS = BusinessAssumptions()

# Capacity: ~100 contacts a week. Evaluated as the top 10% of customers (~7 weeks of contacts),
# because 100/7,043 customers is only ~16 per cross-validation fold: too few for a stable metric.
CONTACT_FRACTION = 0.10
CALLS_PER_WEEK = 100  # the weekly call list's length (the batch job, module 7)

# ---------- Shipping and promotion criteria (problem_framing.md section 5, amended 2026-10-12) ----------
# Pre-registered before any version after v1 is evaluated. Primary metric: must be clearly better
# (95% interval above zero). Profit: must not be clearly worse (interval's lower bound above -10% of the
# reference's profit). A challenger must also not be worse than the current champion on PR-AUC.
CRITERIA_VERSION = "2"
PROFIT_NON_INFERIORITY_FRACTION = 0.10
PR_AUC_NON_INFERIORITY_MARGIN = 0.01

# ---------- Features ----------
TARGET = "churn"
ID_COLUMN = "customer_id"
NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges"]
EXCLUDED_FOR_FAIRNESS = ["gender", "senior_citizen"]  # excluded at first; results are reported per group
CATEGORICAL_FEATURES = [c for c in CATEGORY_COLUMNS if c not in EXCLUDED_FOR_FAIRNESS]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
