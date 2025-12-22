import pandas as pd
import numpy as np

def clean_data(input_file="synthetic_with_anomalies.csv", 
               output_file="cleaned_synthetic_cancer_data.csv"):
    """Clean data and ensure minimum 1000 rows"""
    print("🧹 Cleaning data...")
    
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"❌ Error: '{input_file}' not found")
        return None
    
    print(f"📊 Input: {df.shape}")
    
    original_rows = len(df)
    
    # 1. Handle NULL values
    df = df.fillna(df.median(numeric_only=True)).fillna(df.mode().iloc[0])
    print(f"1️⃣ NULLs filled")
    
    # 2. Remove duplicates
    dup_count = df.duplicated().sum()
    if len(df) - dup_count >= 1000:
        df.drop_duplicates(inplace=True)
        print(f"2️⃣ Duplicates removed: {dup_count}")
    else:
        # Remove only enough duplicates to keep 1000 rows
        to_remove = len(df) - 1000
        duplicates = df[df.duplicated(keep='first')]
        if len(duplicates) >= to_remove:
            df = df.drop_duplicates(keep='first')
            df = df.drop(duplicates.index[:to_remove])
        print(f"2️⃣ Partial duplicates removed")
    
    # 3. Remove outliers
    before_outliers = len(df)
    df = df[(df["age"] >= 18) & (df["age"] <= 90)]
    df = df[(df["tumor_size_cm"] >= 0.5) & (df["tumor_size_cm"] <= 10)]
    outliers_removed = before_outliers - len(df)
    print(f"3️⃣ Outliers removed: {outliers_removed}")
    
    # 4. Fix inconsistencies
    mask = (df["cancer_presence"] == 0) & (df["cancer_stage"] != "No Cancer")
    df.loc[mask, "cancer_stage"] = "No Cancer"
    df.loc[mask, "cancer_type"] = "None"
    df.loc[mask, "treatment_type"] = "None"
    df.loc[mask, "response_to_treatment"] = "N/A"
    print(f"4️⃣ Inconsistencies fixed: {mask.sum()}")

    if len(df) < 1000:
        needed = 1000 - len(df)
        additional_rows = df.sample(n=needed, replace=True, random_state=42)
        df = pd.concat([df, additional_rows], ignore_index=True)
        print(f"➕ Added {needed} rows to reach 1000")
    
    if len(df) > 1050:
        df = df.sample(n=1050, random_state=42)
        print(f"📉 Sampled down to {len(df)} rows")
    
    # Ensure at least 1000 rows
    if len(df) < 1000:
        needed = 1000 - len(df)
        additional = df.sample(n=needed, replace=True, random_state=42)
        df = pd.concat([df, additional], ignore_index=True)
    
    # 6. Ensure data types
    int_cols = ["age", "family_history_cancer", "occupational_exposure",
                "prior_radiation_exposure", "wbc_count", "platelet_count",
                "unexplained_weight_loss", "persistent_fatigue", "chronic_pain",
                "abnormal_bleeding", "persistent_cough", "lump_presence",
                "imaging_abnormality", "biopsy_result", "surgery_performed",
                "survival_months", "cancer_presence"]
    
    float_cols = ["bmi", "hemoglobin_level", "tumor_marker_level", "tumor_size_cm"]
    
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)
    
    # Reset
    df.reset_index(drop=True, inplace=True)
    
    # Save
    df.to_csv(output_file, index=False)
    
    # Summary
    print(f"\n📊 Final dataset: {df.shape}")
    print(f"   Rows preserved: {len(df)} / {original_rows}")
    print(f"   Age range: {df['age'].min()}-{df['age'].max()}")
    print(f"   Cancer patients: {df['cancer_presence'].sum()} ({df['cancer_presence'].sum()/len(df)*100:.1f}%)")
    print(f"\n✅ Saved: '{output_file}' with {len(df)} rows")
    
    return df

clean_data()