"""Data contracts: what valid data looks like at each stage of the pipeline.

The same schemas are reused by the API (module 7), so training and serving accept exactly the same data.
"""

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

YES_NO = ["Yes", "No"]
INTERNET_ADDON = ["Yes", "No", "No internet service"]


def _is_blank(s: pd.Series) -> pd.Series:
    return s.str.strip() == ""


def _is_number(s: pd.Series) -> pd.Series:
    """True where the text can be parsed as a number."""
    return pd.Series(pd.to_numeric(s, errors="coerce"), index=s.index).notna()


RAW_SCHEMA = DataFrameSchema(
    columns={
        "customerID": Column(str, Check.str_matches(r"^\d{4}-[A-Z]{5}$"), unique=True),
        "gender": Column(str, Check.isin(["Female", "Male"])),
        "SeniorCitizen": Column(int, Check.isin([0, 1])),
        "Partner": Column(str, Check.isin(YES_NO)),
        "tenure": Column(int, Check.in_range(0, 100)),
        "OnlineSecurity": Column(str, Check.isin(INTERNET_ADDON)),
        "TotalCharges": Column(
            str,
            Check(
                lambda s: _is_blank(s) | _is_number(s),
                error="TotalCharges must be a number or blank",
            ),
        ),
        "Dependents": Column(str, Check.isin(YES_NO)),
        "PhoneService": Column(str, Check.isin(YES_NO)),
        "MultipleLines": Column(str, Check.isin(["Yes", "No", "No phone service"])),
        "InternetService": Column(str, Check.isin(["DSL", "Fiber optic", "No"])),
        "OnlineBackup": Column(str, Check.isin(INTERNET_ADDON)),
        "DeviceProtection": Column(str, Check.isin(INTERNET_ADDON)),
        "TechSupport": Column(str, Check.isin(INTERNET_ADDON)),
        "StreamingTV": Column(str, Check.isin(INTERNET_ADDON)),
        "StreamingMovies": Column(str, Check.isin(INTERNET_ADDON)),
        "Contract": Column(str, Check.isin(["Month-to-month", "One year", "Two year"])),
        "PaperlessBilling": Column(str, Check.isin(YES_NO)),
        "PaymentMethod": Column(
            str,
            Check.isin(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            ),
        ),
        "MonthlyCharges": Column(float, Check.in_range(0, 500)),
        "Churn": Column(str, Check.isin(YES_NO)),
    },
    checks=[
        Check(
            lambda df: ~_is_blank(df["TotalCharges"]) | (df["tenure"] == 0),
            error="TotalCharges may only be blank for brand-new customers (tenure == 0)",
        ),
    ],
    strict=True,  # no unexpected columns
    name="raw_telco",
)
