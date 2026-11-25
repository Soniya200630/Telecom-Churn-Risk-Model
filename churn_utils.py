"""Reusable pieces for the Telecom Churn Risk Model.
Kept in a module so the saved joblib pipeline can be un-pickled anywhere."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SERVICE_COLS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
PROTECT_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
STREAM_COLS = ["StreamingTV", "StreamingMovies"]

# feature groups used in the ablation study
FEATURE_GROUPS = {
    "tenure":   ["tenure_band", "is_new_customer"],
    "charges":  ["charge_per_tenure_month", "charge_gap", "high_monthly_charge"],
    "bundles":  ["n_services", "n_protect_services", "n_streaming_services", "has_internet"],
    "contract": ["is_month_to_month", "autopay"],
    "risk":     ["fiber_no_support", "household_size"],
}
ALL_ENGINEERED = [f for g in FEATURE_GROUPS.values() for f in g]


def clean_telco(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise cleaning only (no statistics learned from data -> no leakage)."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    obj = df.select_dtypes("object").columns
    for c in obj:
        df[c] = df[c].astype(str).str.strip()
    # blank TotalCharges are brand-new customers (tenure == 0) -> they have been billed nothing yet
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace("", np.nan), errors="coerce")
    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0
    # inconsistent labels: 'No internet service' / 'No phone service' are really 'No'
    df = df.replace({"No internet service": "No", "No phone service": "No"})
    for c in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        df[c] = df[c].str.title()
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless feature engineering. `groups` lets us switch feature families off for ablation."""
    def __init__(self, groups=tuple(FEATURE_GROUPS.keys())):
        self.groups = groups

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        yes = lambda c: (X[c] == "Yes").astype(int)
        f = {}
        if "tenure" in self.groups:
            f["tenure_band"] = pd.cut(X["tenure"], [-1, 6, 12, 24, 48, 100],
                                      labels=["0-6m", "7-12m", "13-24m", "25-48m", "49m+"]).astype(str)
            f["is_new_customer"] = (X["tenure"] <= 6).astype(int)
        if "charges" in self.groups:
            f["charge_per_tenure_month"] = X["TotalCharges"] / np.maximum(X["tenure"], 1)
            f["charge_gap"] = X["MonthlyCharges"] - f["charge_per_tenure_month"]  # >0 => bill has risen
            f["high_monthly_charge"] = (X["MonthlyCharges"] > 70).astype(int)
        if "bundles" in self.groups:
            f["n_services"] = sum(yes(c) for c in SERVICE_COLS) + (X["InternetService"] != "No").astype(int)
            f["n_protect_services"] = sum(yes(c) for c in PROTECT_COLS)
            f["n_streaming_services"] = sum(yes(c) for c in STREAM_COLS)
            f["has_internet"] = (X["InternetService"] != "No").astype(int)
        if "contract" in self.groups:
            f["is_month_to_month"] = (X["Contract"] == "Month-to-month").astype(int)
            f["autopay"] = X["PaymentMethod"].str.contains("automatic", case=False).astype(int)
        if "risk" in self.groups:
            f["fiber_no_support"] = ((X["InternetService"] == "Fiber optic") &
                                     (X["TechSupport"] == "No") & (X["OnlineSecurity"] == "No")).astype(int)
            f["household_size"] = 1 + yes("Partner") + yes("Dependents")
        return pd.concat([X, pd.DataFrame(f, index=X.index)], axis=1)

    def get_feature_names_out(self, input_features=None):
        return None


def make_preprocessor(groups=tuple(FEATURE_GROUPS.keys()), scale=True):
    """FeatureEngineer -> ColumnTransformer (impute + scale numerics, impute + one-hot categoricals)."""
    from sklearn.compose import ColumnTransformer, make_column_selector as sel
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    num_steps = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else [])
    ct = ColumnTransformer([
        ("num", Pipeline(num_steps), sel(dtype_include=np.number)),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
         sel(dtype_include=object)),
    ], verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    return Pipeline([("fe", FeatureEngineer(groups)), ("ct", ct)])
