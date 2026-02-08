import pyspark.sql
import pytest
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Adjust this import to match your project structure.
# If compute_features.py is under mlops_dbx/feature_engineering/features/, use:
from mlops_dbx.feature_engineering.features.compute_features import compute_features_fn
# from compute_features import compute_features_fn


@pytest.fixture(scope="session")
def spark(request):
    """fixture for creating a spark session"""
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("pytest-pyspark-local-testing")
        .getOrCreate()
    )
    request.addfinalizer(lambda: spark.stop())
    return spark


@pytest.mark.usefixtures("spark")
def test_compute_features_fn_basic(spark):
    # Minimal-but-realistic sample covering:
    # - blank TotalCharges -> is_total_charges_missing=1 and fill logic
    # - "No phone service" -> multiple_lines_flag must be 0
    # - contract mapping month-to-month vs two year
    input_df = pd.DataFrame(
        {
            "customerID": ["0001-A", "0002-B"],
            "gender": ["Male", "Female"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "Yes"],
            "tenure": [1, 24],
            "PhoneService": ["No", "Yes"],
            "MultipleLines": ["No phone service", "Yes"],
            "InternetService": ["No", "Fiber optic"],
            "OnlineSecurity": ["No internet service", "Yes"],
            "OnlineBackup": ["No internet service", "No"],
            "DeviceProtection": ["No internet service", "Yes"],
            "TechSupport": ["No internet service", "No"],
            "StreamingTV": ["No internet service", "Yes"],
            "StreamingMovies": ["No internet service", "Yes"],
            "Contract": ["Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Credit card (automatic)"],
            "MonthlyCharges": [70.35, 99.90],
            # TotalCharges in Telco is often a string + can be blank
            "TotalCharges": ["", "2397.60"],
        }
    )

    spark_df = spark.createDataFrame(input_df)

    output_df = compute_features_fn(spark_df)

    # basic assertions
    assert isinstance(output_df, pyspark.sql.DataFrame)
    assert output_df.count() == 2

    # expected output schema basics
    assert "customer_id" in output_df.columns
    assert "Churn" not in output_df.columns  # label should not be present

    # compute_features_fn returns a fixed feature set (38 columns in current implementation)
    assert len(output_df.columns) == 38

    # Row-level checks
    row_1 = (
        output_df.filter(F.col("customer_id") == "0001-A")
        .select(
            "phone_service_flag",
            "multiple_lines_flag",
            "is_total_charges_missing",
            "monthly_charges",
            "tenure_months",
            "total_charges_filled",
            "contract_months",
            "is_month_to_month",
        )
        .collect()[0]
    )

    # PhoneService == "No" => phone_service_flag=0 AND multiple_lines_flag must be 0
    assert row_1["phone_service_flag"] == 0
    assert row_1["multiple_lines_flag"] == 0

    # TotalCharges blank => missing flag 1 and filled with monthly_charges * tenure_months
    assert row_1["is_total_charges_missing"] == 1
    expected_filled = float(row_1["monthly_charges"]) * int(row_1["tenure_months"])
    assert abs(float(row_1["total_charges_filled"]) - expected_filled) < 1e-6

    # Contract Month-to-month => 1 month, month-to-month flag 1
    assert row_1["contract_months"] == 1
    assert row_1["is_month_to_month"] == 1

    row_2 = (
        output_df.filter(F.col("customer_id") == "0002-B")
        .select("contract_months", "is_long_contract", "is_auto_payment")
        .collect()[0]
    )

    # Two year => 24 months and long contract flag 1
    assert row_2["contract_months"] == 24
    assert row_2["is_long_contract"] == 1

    # "Credit card (automatic)" contains "(automatic)" => auto payment flag 1
    assert row_2["is_auto_payment"] == 1
