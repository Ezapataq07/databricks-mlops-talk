# src/features/compute_features.py

from __future__ import annotations

from typing import Iterable, List

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


# --- Contract: we keep the label OUT of the feature table to avoid leakage.
REQUIRED_COLUMNS: List[str] = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


def _require_columns(df: DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _lower_trim(col_name: str) -> F.Column:
    return F.lower(F.trim(F.col(col_name).cast("string")))


def _yn_to_int(col_name: str) -> F.Column:
    """
    Maps "Yes" -> 1, everything else -> 0
    (Works well for: No, No internet service, No phone service, nulls)
    """
    return F.when(_lower_trim(col_name) == F.lit("yes"), F.lit(1)).otherwise(F.lit(0)).cast("int")


def _to_double_robust(col_name: str) -> F.Column:
    """
    Robust conversion to double:
    - Accepts numeric columns or strings
    - Handles blanks / whitespace -> null
    - Handles NaN -> null
    """
    raw = F.trim(F.col(col_name).cast("string"))
    as_double = F.when(raw == "", F.lit(None)).otherwise(raw).cast("double")
    return F.when(F.isnan(as_double), F.lit(None)).otherwise(as_double)


def compute_features_fn(df_raw: DataFrame) -> DataFrame:
    """
    Feature engineering for Telco Customer Churn dataset.

    Input:
      - df_raw: Spark DataFrame with original Kaggle schema (including customerID).
    Output:
      - Spark DataFrame suitable for Unity Catalog Feature Store table:
        * primary key: customer_id
        * no label column
        * scalar columns only (int/double/string)
    """
    _require_columns(df_raw, REQUIRED_COLUMNS)

    # --------------------------
    # 0) Base normalization
    # --------------------------
    df = (
        df_raw
        .select(
            F.trim(F.col("customerID").cast("string")).alias("customer_id"),
            F.trim(F.col("gender").cast("string")).alias("gender"),
            F.col("SeniorCitizen").cast("int").alias("senior_citizen"),
            F.col("tenure").cast("int").alias("tenure_months"),
            F.col("MonthlyCharges").cast("double").alias("monthly_charges"),
            _to_double_robust("TotalCharges").alias("total_charges"),
            F.trim(F.col("Partner").cast("string")).alias("partner"),
            F.trim(F.col("Dependents").cast("string")).alias("dependents"),
            F.trim(F.col("PhoneService").cast("string")).alias("phone_service"),
            F.trim(F.col("MultipleLines").cast("string")).alias("multiple_lines"),
            F.trim(F.col("InternetService").cast("string")).alias("internet_service"),
            F.trim(F.col("OnlineSecurity").cast("string")).alias("online_security"),
            F.trim(F.col("OnlineBackup").cast("string")).alias("online_backup"),
            F.trim(F.col("DeviceProtection").cast("string")).alias("device_protection"),
            F.trim(F.col("TechSupport").cast("string")).alias("tech_support"),
            F.trim(F.col("StreamingTV").cast("string")).alias("streaming_tv"),
            F.trim(F.col("StreamingMovies").cast("string")).alias("streaming_movies"),
            F.trim(F.col("Contract").cast("string")).alias("contract_type"),
            F.trim(F.col("PaperlessBilling").cast("string")).alias("paperless_billing"),
            F.trim(F.col("PaymentMethod").cast("string")).alias("payment_method"),
        )
        .filter(F.col("customer_id").isNotNull() & (F.col("customer_id") != ""))
        .dropDuplicates(["customer_id"])
        .withColumn("senior_citizen", F.coalesce(F.col("senior_citizen"), F.lit(0)))
        .withColumn("tenure_months", F.coalesce(F.col("tenure_months"), F.lit(0)))
        .withColumn("monthly_charges", F.coalesce(F.col("monthly_charges"), F.lit(0.0)))
    )

    # Helpful missingness signal (in the full dataset TotalCharges can be blank)
    df = df.withColumn("is_total_charges_missing", F.when(F.col("total_charges").isNull(), 1).otherwise(0).cast("int"))

    # --------------------------
    # 1) Binary / household features
    # --------------------------
    df = (
        df
        .withColumn("has_partner", _yn_to_int("partner"))
        .withColumn("has_dependents", _yn_to_int("dependents"))
        .withColumn("paperless_billing_flag", _yn_to_int("paperless_billing"))
    )

    # --------------------------
    # 2) Service features (phone / internet)
    # --------------------------
    df = df.withColumn("phone_service_flag", _yn_to_int("phone_service"))

    # MultipleLines can be "No phone service". Force it to 0 when no phone service.
    df = df.withColumn(
        "multiple_lines_flag",
        F.when(F.col("phone_service_flag") == 0, F.lit(0)).otherwise(_yn_to_int("multiple_lines")).cast("int"),
    )

    # Internet provider flags
    internet_l = _lower_trim("internet_service")
    df = (
        df
        .withColumn("has_internet", F.when(internet_l == "no", 0).otherwise(1).cast("int"))
        .withColumn("internet_is_dsl", F.when(internet_l == "dsl", 1).otherwise(0).cast("int"))
        .withColumn("internet_is_fiber", F.when(internet_l == "fiber optic", 1).otherwise(0).cast("int"))
    )

    # Add-on services (common churn drivers)
    df = (
        df
        .withColumn("online_security_flag", _yn_to_int("online_security"))
        .withColumn("online_backup_flag", _yn_to_int("online_backup"))
        .withColumn("device_protection_flag", _yn_to_int("device_protection"))
        .withColumn("tech_support_flag", _yn_to_int("tech_support"))
        .withColumn("streaming_tv_flag", _yn_to_int("streaming_tv"))
        .withColumn("streaming_movies_flag", _yn_to_int("streaming_movies"))
    )

    df = (
        df
        .withColumn(
            "addon_services_cnt",
            (
                F.col("online_security_flag")
                + F.col("online_backup_flag")
                + F.col("device_protection_flag")
                + F.col("tech_support_flag")
                + F.col("streaming_tv_flag")
                + F.col("streaming_movies_flag")
            ).cast("int"),
        )
        .withColumn("security_support_cnt", (F.col("online_security_flag") + F.col("tech_support_flag")).cast("int"))
        .withColumn("streaming_cnt", (F.col("streaming_tv_flag") + F.col("streaming_movies_flag")).cast("int"))
    )

    # --------------------------
    # 3) Contract / payments
    # --------------------------
    df = (
        df
        .withColumn(
            "contract_months",
            F.when(F.col("contract_type") == "Month-to-month", 1)
             .when(F.col("contract_type") == "One year", 12)
             .when(F.col("contract_type") == "Two year", 24)
             .otherwise(None)
             .cast("int"),
        )
        .withColumn("is_month_to_month", F.when(F.col("contract_type") == "Month-to-month", 1).otherwise(0).cast("int"))
        .withColumn("is_long_contract", F.when(F.col("contract_type").isin("One year", "Two year"), 1).otherwise(0).cast("int"))
        .withColumn("is_auto_payment", F.when(F.col("payment_method").contains("(automatic)"), 1).otherwise(0).cast("int"))
        .withColumn("is_electronic_check", F.when(F.col("payment_method") == "Electronic check", 1).otherwise(0).cast("int"))
    )

    # --------------------------
    # 4) Tenure / billing behavior
    # --------------------------
    df = (
        df
        .withColumn("tenure_years", (F.col("tenure_months") / F.lit(12.0)).cast("double"))
        .withColumn(
            "tenure_bucket",
            F.when(F.col("tenure_months") < 6, "0_5m")
             .when(F.col("tenure_months") < 12, "6_11m")
             .when(F.col("tenure_months") < 24, "12_23m")
             .when(F.col("tenure_months") < 48, "24_47m")
             .otherwise("48m_plus"),
        )
        .withColumn("is_new_customer", F.when(F.col("tenure_months") < 6, 1).otherwise(0).cast("int"))
    )

    # Total charges: if missing, impute with monthly_charges * tenure
    df = df.withColumn(
        "total_charges_filled",
        F.coalesce(
            F.col("total_charges"),
            (F.col("monthly_charges") * F.col("tenure_months")).cast("double"),
        ),
    )

    # Averages and “consistency” signals
    df = (
        df
        .withColumn(
            "avg_monthly_charge_lifetime",
            (
                F.col("total_charges_filled")
                / F.when(F.col("tenure_months") > 0, F.col("tenure_months")).otherwise(F.lit(1))
            ).cast("double"),
        )
        .withColumn(
            "charges_gap",
            (F.col("total_charges_filled") - (F.col("monthly_charges") * F.col("tenure_months"))).cast("double"),
        )
        .withColumn("abs_charges_gap", F.abs(F.col("charges_gap")).cast("double"))
        .withColumn(
            "monthly_charge_bucket",
            F.when(F.col("monthly_charges") < 35, "low")
             .when(F.col("monthly_charges") < 70, "mid")
             .otherwise("high"),
        )
    )

    # --------------------------
    # 5) Final selection (Feature Store friendly)
    # --------------------------
    feature_cols = [
        "customer_id",

        # low-card categoricals (use OHE later)
        "gender",
        "internet_service",
        "contract_type",
        "payment_method",
        "tenure_bucket",
        "monthly_charge_bucket",

        # demographics / household
        "senior_citizen",
        "has_partner",
        "has_dependents",

        # tenure
        "tenure_months",
        "tenure_years",
        "is_new_customer",

        # services
        "phone_service_flag",
        "multiple_lines_flag",
        "has_internet",
        "internet_is_dsl",
        "internet_is_fiber",

        # add-ons
        "online_security_flag",
        "online_backup_flag",
        "device_protection_flag",
        "tech_support_flag",
        "streaming_tv_flag",
        "streaming_movies_flag",
        "addon_services_cnt",
        "security_support_cnt",
        "streaming_cnt",

        # contract/payment signals
        "contract_months",
        "is_month_to_month",
        "is_long_contract",
        "paperless_billing_flag",
        "is_auto_payment",
        "is_electronic_check",

        # charges
        "monthly_charges",
        "total_charges_filled",
        "avg_monthly_charge_lifetime",
        "is_total_charges_missing",
        "abs_charges_gap",
    ]

    return df.select(*feature_cols)
