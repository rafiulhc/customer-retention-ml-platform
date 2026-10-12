"""Small, valid test data shared by the test modules."""

import pandas as pd

from retention.data.clean import clean


def raw_row(**overrides: object) -> dict[str, object]:
    """One valid raw customer. Override any field to build a test case."""
    row: dict[str, object] = {
        "customerID": "0001-ABCDE",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 50.0,
        "TotalCharges": "600.0",
        "Churn": "No",
    }
    row.update(overrides)
    return row


def frame(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def clean_customers(*specs: dict[str, object]) -> pd.DataFrame:
    """Cleaned customers built from raw overrides; each gets a unique ID automatically."""
    rows = [raw_row(**{"customerID": f"{i:04d}-ABCDE", **spec}) for i, spec in enumerate(specs, start=1)]
    return clean(frame(*rows))
