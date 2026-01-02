import pandas as pd
import numpy as np
from ctgan import CTGAN


# 1. GENERATE SYNTHETIC DATA

def generate_synthetic_data():
    print("Generating synthetic patient data...\n")

    rows = 1300

    data = {
        "patient_id": np.arange(1000, 1000 + rows),
        "age": np.random.randint(18, 90, rows),
        "gender": np.random.choice(["Male", "Female"], rows),
        "blood_group": np.random.choice(["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"], rows),
        "ethnicity": np.random.choice(["Asian", "African", "Caucasian", "Hispanic"], rows),
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
        "cancer_presence": np.random.choice([0, 1], rows, p=[0.7, 0.3])
    }

    df = pd.DataFrame(data)

    discrete_cols = [
        col for col in df.columns
        if df[col].dtype == "object" or df[col].nunique() <= 5
    ]

    ctgan = CTGAN(epochs=200, batch_size=500, verbose=True)
    ctgan.fit(df, discrete_columns=discrete_cols)

    synthetic_data = ctgan.sample(1200)
    synthetic_data.to_csv("AI_Thunderball_Dataset.csv", index=False)

    print(f"\nGenerated {synthetic_data.shape[0]} records")
    return synthetic_data


generate_synthetic_data()


# 2. INTRODUCE ANOMALIES

def introduce_anomalies(
    input_file="AI_Thunderball_Dataset.csv",
    output_file="AI_Thunderball_RawDataset.csv"
):
    print("\nIntroducing anomalies...")

    df = pd.read_csv(input_file)
    print(f"Input shape: {df.shape}")

    # NULL values
    null_idx = df.sample(frac=0.03, random_state=42).index
    df.loc[null_idx, ["wbc_count", "hemoglobin_level", "platelet_count"]] = np.nan

    # Duplicate rows
    dup_rows = df.sample(frac=0.01, random_state=1)
    df = pd.concat([df, dup_rows], ignore_index=True)

    # Outliers
    df.loc[df.sample(frac=0.005, random_state=3).index, "age"] = 150
    df.loc[df.sample(frac=0.005, random_state=4).index, "tumor_size_cm"] = 40

    # Random deletions
    del_idx = df.sample(frac=0.005, random_state=10).index
    df.drop(del_idx, inplace=True)

    df.to_csv(output_file, index=False)
    print(f"Final shape after anomalies: {df.shape}")
    print(f"Saved raw dataset: {output_file}")

    return df


introduce_anomalies()


# 3. CLEAN DATA

def clean_data(
    input_file="AI_Thunderball_RawDataset.csv",
    output_file="AI_Thunderball_CleanedDataset.csv"
):
    print("\nCleaning data...")

    df = pd.read_csv(input_file)
    original_rows = len(df)
    print(f"Input shape: {df.shape}")

    # Handle NULLs
    df = df.fillna(df.median(numeric_only=True))
    df = df.fillna(df.mode().iloc[0])

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Remove outliers
    df = df[(df["age"].between(18, 90))]
    df = df[(df["tumor_size_cm"].between(0.5, 10))]

    # Enforce exactly 1200 rows
    if len(df) < 1200:
        extra = df.sample(1200 - len(df), replace=True, random_state=42)
        df = pd.concat([df, extra], ignore_index=True)

    if len(df) > 1200:
        df = df.sample(1200, random_state=42)

    # Safe type casting
    int_cols = [
        "age", "family_history_cancer", "wbc_count", "platelet_count",
        "unexplained_weight_loss", "persistent_fatigue", "chronic_pain",
        "abnormal_bleeding", "persistent_cough", "lump_presence",
        "imaging_abnormality", "cancer_presence"
    ]

    float_cols = ["hemoglobin_level", "tumor_marker_level", "tumor_size_cm"]

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    df.reset_index(drop=True, inplace=True)
    df.to_csv(output_file, index=False)

    print("\nCleaning completed")
    print(f"Final dataset shape: {df.shape}")
    print(f"Rows preserved: {len(df)} / {original_rows}")
    print(f"Age range: {df['age'].min()} - {df['age'].max()}")
    print(f"Saved cleaned dataset: {output_file}")

    return df


clean_data()
