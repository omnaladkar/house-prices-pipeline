import os
import json
import pandas as pd
import numpy as np
from datetime import datetime


def generate_dataset(n_samples=200):
    np.random.seed(42)

    square_footage = np.random.uniform(800, 5000, n_samples)
    bedrooms = np.random.randint(1, 7, n_samples)
    bathrooms = np.random.uniform(1, 5, n_samples)
    age = np.random.uniform(0, 100, n_samples)
    garage_spaces = np.random.randint(0, 4, n_samples)
    location_score = np.random.uniform(1, 10, n_samples)
    has_pool = np.random.choice([0, 1], n_samples)
    stories = np.random.choice([1, 2, 3], n_samples)

    price = (
        150 * square_footage
        + 20000 * bedrooms
        + 15000 * bathrooms
        - 500 * age
        + 10000 * garage_spaces
        + 30000 * location_score
        + 50000 * has_pool
        + 25000 * stories
        + np.random.normal(0, 50000, n_samples)
    )

    price = np.maximum(price, 50000)

    return pd.DataFrame({
        "Square_Footage": square_footage,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Age_Years": age,
        "Garage_Spaces": garage_spaces,
        "Location_Score": location_score,
        "Has_Pool": has_pool,
        "Stories": stories,
        "Price": price,
    })


def validate(df):
    checks = []

    assert len(df) > 0, "Empty dataset"
    checks.append("Row count: PASS")

    assert df["Price"].min() > 0, "Negative prices found"
    checks.append("Positive prices: PASS")

    assert df["Square_Footage"].min() > 0, "Invalid square footage"
    checks.append("Positive square footage: PASS")

    assert df["Age_Years"].min() >= 0, "Negative age"
    checks.append("Non-negative age: PASS")

    assert not df.duplicated().any(), "Duplicate rows found"
    checks.append("No duplicates: PASS")

    return checks


def transform(df):
    df["Price_Per_SqFt"] = (df["Price"] / df["Square_Footage"]).round(2)
    df["Price"] = df["Price"].round(2)

    df["Age_Category"] = pd.cut(
        df["Age_Years"],
        bins=[0, 10, 30, 60, 100],
        labels=["New", "Moderate", "Old", "Very Old"],
    )

    df["Size_Category"] = pd.cut(
        df["Square_Footage"],
        bins=[0, 1500, 2500, 3500, float("inf")],
        labels=["Small", "Medium", "Large", "Extra Large"],
    )

    return df


def aggregate(df):
    summary = {
        "avg_price_by_size": df.groupby("Size_Category")["Price"]
        .mean()
        .round(2)
        .to_dict(),
        "avg_price_by_age": df.groupby("Age_Category")["Price"]
        .mean()
        .round(2)
        .to_dict(),
        "avg_price_by_bedrooms": df.groupby("Bedrooms")["Price"]
        .mean()
        .round(2)
        .to_dict(),
        "total_rows": len(df),
        "price_mean": round(df["Price"].mean(), 2),
        "price_median": round(df["Price"].median(), 2),
        "price_min": round(df["Price"].min(), 2),
        "price_max": round(df["Price"].max(), 2),
    }
    return summary


def main():
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 50)
    print("PIPELINE START")
    print("=" * 50)

    print("\n[1/5] GENERATING DATA")
    df = generate_dataset()
    print(f"  Generated {len(df)} rows")

    print("\n[2/5] VALIDATING")
    for check in validate(df):
        print(f"  {check}")

    print("\n[3/5] TRANSFORMING")
    df = transform(df)
    print(f"  Columns: {len(df.columns)}")

    print("\n[4/5] AGGREGATING")
    summary = aggregate(df)
    print(f"  Avg price: ${summary['price_mean']:,.2f}")

    print("\n[5/5] SAVING")
    csv_path = os.path.join(output_dir, f"house_prices_{run_id}.csv")
    parquet_path = os.path.join(output_dir, f"house_prices_{run_id}.parquet")
    summary_path = os.path.join(output_dir, f"summary_{run_id}.json")

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "schema": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }
    with open(summary_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  CSV:    {csv_path}")
    print(f"  Parquet: {parquet_path}")
    print(f"  Summary: {summary_path}")

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
