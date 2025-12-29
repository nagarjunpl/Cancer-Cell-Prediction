import pandas as pd
import numpy as np
from faker import Faker
from ctgan import CTGAN


# 1. Generate Synthetic Data

def generate_synthetic_data():
    print("Generating synthetic patient data...\n")

    fake = Faker()
    rows = 1300

    data = {
        "age": np.random.randint(18, 90, rows),
        "gender": np.random.choice(["Male", "Female"], rows),
        "blood_group": np.random.choice(
            ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"], rows
        ),
        "ethnicity": np.random.choice(
            ["Asian", "African", "Caucasian", "Hispanic"], rows
        ),
        "height_cm": np.random.randint(120, 200, rows),
        "weight_kg": np.random.randint(30, 150, rows),
        "smoking_status": np.random.choice(["Never", "Former", "Current"], rows),
        "alcohol_consumption": np.random.choice(["Never", "Moderate", "High"], rows),
        "family_history_cancer": np.random.choice([0, 1], rows),
        "physical_activity_level": np.random.choice(["Low", "Medium", "High"], rows),
        "diet_quality": np.random.choice(["Poor", "Average", "Good"], rows),
        "diabetes": np.random.choice([0, 1], rows),
        "hypertension": np.random.choice([0, 1], rows),
        "asthma": np.random.choice([0, 1], rows),
        "cardiac_disease": np.random.choice([0, 1], rows),
        "prior_radiation_exposure": np.random.choice([0, 1], rows),
        "unexplained_weight_loss": np.random.choice([0, 1], rows),
        "persistent_fatigue": np.random.choice([0, 1], rows),
        "chronic_pain": np.random.choice([0, 1], rows),
        "abnormal_bleeding": np.random.choice([0, 1], rows),
        "persistent_cough": np.random.choice([0, 1], rows),
        "lump_presence": np.random.choice([0, 1], rows),
        "hemoglobin_level": np.round(np.random.uniform(8, 17, rows), 1),
        "wbc_count": np.random.randint(3000, 12000, rows),
        "platelet_count": np.random.randint(150000, 450000, rows),
        "tumor_marker_level": np.round(np.random.uniform(0.5, 100, rows), 2),
        "imaging_abnormality": np.random.choice([0, 1], rows),
        "tumor_size_cm": np.round(np.random.uniform(0.5, 10, rows), 2),
    }

    df = pd.DataFrame(data)

    df.insert(0, "patient_id", [fake.uuid4() for _ in range(rows)])

    df["bmi"] = df["weight_kg"] / ((df["height_cm"] / 100) ** 2)

    discrete_cols = [
        col for col in df.columns
        if df[col].dtype == "object" or df[col].nunique() <= 5
    ]

    # Train the CTGAN
    ctgan = CTGAN(epochs=200, batch_size=500, verbose=True)
    ctgan.fit(df, discrete_columns=discrete_cols)

    synthetic_data = ctgan.sample(1200)

    synthetic_data.to_csv("AI_Thunderball_Dataset.csv", index=False)

    print(f"\nGenerated {synthetic_data.shape[0]} synthetic records")
    return synthetic_data


# 2. Introduce Anomalies

def introduce_anomalies(
    input_file="AI_Thunderball_Dataset.csv",
    output_file="AI_Thunderball_RawDataset.csv"
):
    print("\nIntroducing anomalies...")

    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: '{input_file}' not found")
        return None

    # NULL values
    null_idx = df.sample(frac=0.03, random_state=42).index
    df.loc[null_idx, ["wbc_count", "hemoglobin_level", "platelet_count"]] = np.nan

    dup_rows = df.sample(frac=0.01, random_state=1)
    df = pd.concat([df, dup_rows], ignore_index=True)

    df.loc[df.sample(frac=0.005, random_state=3).index, "age"] = 150
    df.loc[df.sample(frac=0.005, random_state=4).index, "tumor_size_cm"] = 40

    # Random deletions
    del_idx = df.sample(frac=0.005, random_state=10).index
    df.drop(del_idx, inplace=True)

    df.to_csv(output_file, index=False)
    print(f"Saved anomalous dataset: {output_file}")
    return df


# 3. Clean the Dataset

def clean_data(
    input_file="AI_Thunderball_RawDataset.csv",
    output_file="AI_Thunderball_CleanedDataset.csv"
):
    print("\nCleaning dataset...")

    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: '{input_file}' not found")
        return None

    original_rows = len(df)

    df = df.fillna(df.median(numeric_only=True))
    df = df.fillna(df.mode().iloc[0])

    df.drop_duplicates(inplace=True)

    df = df[(df["age"] >= 18) & (df["age"] <= 90)]
    df = df[(df["tumor_size_cm"] >= 0.5) & (df["tumor_size_cm"] <= 10)]

    # Ensure 1200 rows
    if len(df) < 1200:
        df = pd.concat(
            [df, df.sample(1200 - len(df), replace=True, random_state=42)],
            ignore_index=True
        )

    if len(df) > 1200:
        df = df.sample(1200, random_state=42)

    # Data types
    int_cols = [
        "age", "family_history_cancer", "diabetes", "hypertension",
        "asthma", "cardiac_disease", "prior_radiation_exposure",
        "unexplained_weight_loss", "persistent_fatigue", "chronic_pain",
        "abnormal_bleeding", "persistent_cough", "lump_presence",
        "imaging_abnormality", "wbc_count", "platelet_count"
    ]

    float_cols = [
        "bmi", "hemoglobin_level", "tumor_marker_level", "tumor_size_cm"
    ]

    for col in int_cols:
        df[col] = df[col].astype(int)

    for col in float_cols:
        df[col] = df[col].astype(float)

    df.reset_index(drop=True, inplace=True)
    df.to_csv(output_file, index=False)

    print(f"Final cleaned dataset saved: {output_file}")
    print(f"Rows: {len(df)} (from {original_rows})")
    return df


generate_synthetic_data()
introduce_anomalies()
clean_data()
