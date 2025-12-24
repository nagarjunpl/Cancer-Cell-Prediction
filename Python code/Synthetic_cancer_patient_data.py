import pandas as pd
import numpy as np
from faker import Faker
from ctgan import CTGAN

def generate_synthetic_data():
    print(" Generating synthetic cancer patient data... \n")
    
    fake = Faker()
    
    rows = 1300
    
    # Generate names
    first_names = [fake.first_name() for _ in range(rows)]
    last_names = [fake.last_name() for _ in range(rows)]
    
    # Data structure
    data = {
        "first_name": first_names,
        "last_name": last_names,
        "age": np.random.randint(18, 90, rows),
        "gender": np.random.choice(["Male", "Female"], rows),
        "blood_group": np.random.choice(["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"], rows),
        "ethnicity": np.random.choice(["Asian", "African", "Caucasian", "Hispanic"], rows),
        "bmi": np.round(np.random.uniform(16, 40, rows), 1),
        "height_cm": np.random.randint(120, 200, rows),
        "weight_kg": np.random.randint(30, 150, rows),
        "smoking_status": np.random.choice(["Never", "Former", "Current"], rows),
        "alcohol_consumption": np.random.choice(["None", "Moderate", "High"], rows),
        "family_history_cancer": np.random.choice([0, 1], rows),
        "physical_activity_level": np.random.choice(["Low", "Medium", "High"], rows),
        "diet_quality": np.random.choice(["Poor", "Average", "Good"], rows),
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
        "biopsy_result": np.random.choice([0, 1], rows),
        "tumor_size_cm": np.round(np.random.uniform(0.5, 10, rows), 2),
        "treatment_type": np.random.choice(["Chemotherapy", "Radiation", "Surgery", "Combined", "None"], rows),
        "surgery_performed": np.random.choice([0, 1], rows),
        "response_to_treatment": np.random.choice(["Poor", "Stable", "Improved", "N/A"], rows),
        "cancer_presence": np.random.choice([0, 1], rows, p=[0.3, 0.7])
    }
    
    df = pd.DataFrame(data)
    
    # Add patient_id and full_name
    df.insert(0, "patient_id", [fake.uuid4() for _ in range(rows)])
    df["full_name"] = df["first_name"] + " "+ df["last_name"]
    
    non_cancer_mask = df["cancer_presence"] == 0
    df.loc[non_cancer_mask, "treatment_type"] = "None"
    df.loc[non_cancer_mask, "response_to_treatment"] = "N/A"
    df.loc[non_cancer_mask, "surgery_performed"] = 0
    
    # Generate synthetic data 
    discrete_cols = [col for col in df.columns if df[col].dtype == 'object' or df[col].nunique() <= 5]
    
    ctgan = CTGAN(epochs=200, batch_size=500, verbose=True)
    ctgan.fit(df, discrete_columns=discrete_cols)
    
    synthetic_data = ctgan.sample(1200)
    
    # Save only synthetic data
    synthetic_data.to_csv("synthetic_cancer_patient_data.csv", index=False)
    
    print(f"\n Generated: {synthetic_data.shape[0]} records")
    print(f"Cancer presence: {synthetic_data['cancer_presence'].value_counts().to_dict()}")
    
    return synthetic_data

generate_synthetic_data()
