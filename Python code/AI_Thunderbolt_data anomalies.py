import pandas as pd
import numpy as np

def introduce_anomalies(input_file="synthetic_cancer_patient_data.csv", 
                        output_file="synthetic_with_anomalies.csv"):
    """Introduce anomalies while maintaining ~1100+ rows"""
    print("⚠️ Introducing anomalies...")
    
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"❌ Error: '{input_file}' not found")
        return None
    
    print(f" Input: {df.shape}")
    
    # 1. NULL values
    null_idx = df.sample(frac=0.03, random_state=42).index
    df.loc[null_idx, ["bmi", "hemoglobin_level"]] = np.nan
    print(f"1️. NULLs: {df[['bmi', 'hemoglobin_level']].isna().sum().sum()}")
    
    # 2. DUPLICATE rows
    dup_rows = df.sample(frac=0.01, random_state=1)
    df = pd.concat([df, dup_rows], ignore_index=True)
    print(f"2️. Duplicates: +{len(dup_rows)}")
    
    # 3. OUTLIERS 
    df.loc[df.sample(frac=0.005, random_state=3).index, "age"] = 150
    df.loc[df.sample(frac=0.005, random_state=4).index, "tumor_size_cm"] = 40
    print(f"3️. Outliers: Age(150), Tumor(40cm)")
    
    # 4. RANDOM DELETIONS 
    del_idx = df.sample(frac=0.005, random_state=10).index
    df.drop(del_idx, inplace=True)
    print(f"4️. Deletions: -{len(del_idx)}")
    
    # 5. INCONSISTENCIES
    non_cancer = df[df["cancer_presence"] == 0]
    if not non_cancer.empty:
        inc_idx = non_cancer.sample(frac=0.02, random_state=5).index
        df.loc[inc_idx, "cancer_stage"] = "Stage III"
        print(f"5️. Inconsistencies: {len(inc_idx)}")
    
    # Save
    df.to_csv(output_file, index=False)
    
    print(f"\n📊 Final: {df.shape}")
    print(f"✅ Saved: '{output_file}'")
    
    return df

introduce_anomalies()