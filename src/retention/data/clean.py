"""Turn validated raw data into model-ready data.

Pure: a DataFrame in, a new DataFrame out; the input is never modified. Nothing here learns from the
data (no averages, scaling or encoding): that happens after the train/test split, inside the model
pipeline, so no information leaks from the test set. The API reuses this function (module 7).
"""

import pandas as pd

from retention.config import ID_COLUMN, TARGET
from retention.data.schema import CATEGORY_COLUMNS, CLEAN_SCHEMA, RAW_SCHEMA

__all__ = ["ID_COLUMN", "RENAMES", "TARGET", "clean"]

RENAMES = {
    "customerID": "customer_id",
    "gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churn",
}


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate raw data, clean it, and validate the result."""
    df = RAW_SCHEMA.validate(raw.copy(), lazy=True).rename(columns=RENAMES)

    # Blank total charges belong to customers with tenure 0 (never billed yet): they are 0, not missing.
    # astype(float): to_numeric returns integers when every value is whole (e.g. "100"), which the
    # clean schema rejects. Explicit, so a single API request like TotalCharges="100" works too.
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce").fillna(0.0).astype(float)

    df[TARGET] = (df[TARGET] == "Yes").astype(int)

    # A yes/no flag stored as 0/1 in the raw data: store it like every other yes/no column.
    df["senior_citizen"] = df["senior_citizen"].map({1: "Yes", 0: "No"})

    df[CATEGORY_COLUMNS] = df[CATEGORY_COLUMNS].astype("category")

    return CLEAN_SCHEMA.validate(df, lazy=True)
