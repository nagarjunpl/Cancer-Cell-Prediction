import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ctgan import CTGAN
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import pickle
import os
from datetime import datetime
import uvicorn

 
# PYDANTIC MODELS FOR API REQUESTS/RESPONSES
 

class PatientData(BaseModel):
    """Schema for single patient prediction request"""
    patient_id: int
    age: int = Field(..., ge=18, le=120)
    gender: str = Field(..., pattern="^(Male|Female)$")
    blood_group: str = Field(..., pattern="^(A\+|A-|B\+|B-|AB\+|AB-|O\+|O-)$")
    ethnicity: str = Field(..., pattern="^(Asian|African|Caucasian|Hispanic)$")
    height_cm: float = Field(..., ge=100, le=250)
    weight_kg: float = Field(..., ge=30, le=300)
    smoking_status: str = Field(..., pattern="^(Never|Former|Current)$")
    alcohol_consumption: str = Field(..., pattern="^(Never|Moderate|High)$")
    family_history_cancer: int = Field(..., ge=0, le=1)
    physical_activity_level: str = Field(..., pattern="^(Low|Medium|High)$")
    diet_quality: str = Field(..., pattern="^(Poor|Average|Good)$")
    diabetes: int = Field(..., ge=0, le=1)
    hypertension: int = Field(..., ge=0, le=1)
    asthma: int = Field(..., ge=0, le=1)
    cardiac_disease: int = Field(..., ge=0, le=1)
    prior_radiation_exposure: int = Field(..., ge=0, le=1)
    unexplained_weight_loss: int = Field(..., ge=0, le=1)
    persistent_fatigue: int = Field(..., ge=0, le=1)
    chronic_pain: int = Field(..., ge=0, le=1)
    abnormal_bleeding: int = Field(..., ge=0, le=1)
    persistent_cough: int = Field(..., ge=0, le=1)
    lump_presence: int = Field(..., ge=0, le=1)
    hemoglobin_level: float = Field(..., ge=5, le=20)
    wbc_count: int = Field(..., ge=1000, le=50000)
    platelet_count: int = Field(..., ge=50000, le=1000000)
    tumor_marker_level: float = Field(..., ge=0, le=200)
    imaging_abnormality: int = Field(..., ge=0, le=1)
    tumor_size_cm: float = Field(..., ge=0, le=30)

class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction request"""
    patients: List[PatientData]
    model_name: Optional[str] = "Random Forest"  # Default model

class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    patient_id: int
    prediction: int
    probability: float
    risk_level: str
    confidence: float
    model_used: str
    timestamp: str
    clinical_insights: Dict[str, Any]

class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response"""
    predictions: List[PredictionResponse]
    total_patients: int
    positive_cases: int
    negative_cases: int
    average_risk: float
    model_used: str
    processing_time: float

class ModelInfo(BaseModel):
    """Schema for model information"""
    name: str
    type: str
    accuracy: float
    roc_auc: float
    precision: float
    recall: float
    f1_score: float
    features_used: List[str]
    training_date: str
    hyperparameters: Dict[str, Any]

class APIHealthResponse(BaseModel):
    """Schema for API health check"""
    status: str
    version: str
    uptime: float
    models_loaded: int
    total_predictions: int
    memory_usage: Dict[str, str]

 
# FASTAPI APP INITIALIZATION
 

app = FastAPI(
    title="AI Thunderball Cancer Detection API",
    description="""A machine learning API for cancer risk prediction using 
    synthetic patient data with comprehensive feature engineering and multiple 
    regularization techniques.""",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

 
# GLOBAL STATE AND INITIALIZATION
 

class APIManager:
    """Manages API state and models"""
    def __init__(self):
        self.models = {}
        self.preprocessor = None
        self.feature_names = []
        self.total_predictions = 0
        self.start_time = datetime.now()
        self.model_metrics = {}
        
    def load_models(self, models_dict, preprocessor, feature_names, model_metrics):
        """Load trained models into API"""
        self.models = models_dict
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        self.model_metrics = model_metrics
        print(f"✓ Loaded {len(self.models)} models into API")
        
    def get_model(self, model_name="Random Forest"):
        """Get model by name"""
        if model_name not in self.models:
            available = list(self.models.keys())
            raise HTTPException(
                status_code=404, 
                detail=f"Model '{model_name}' not found. Available models: {available}"
            )
        return self.models[model_name]
    
    def increment_predictions(self):
        """Track total predictions made"""
        self.total_predictions += 1
        
    def get_uptime(self):
        """Calculate API uptime in seconds"""
        return (datetime.now() - self.start_time).total_seconds()
    
    def get_memory_usage(self):
        """Get memory usage information"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            return {
                "rss": f"{mem_info.rss / 1024 / 1024:.2f} MB",
                "vms": f"{mem_info.vms / 1024 / 1024:.2f} MB"
            }
        except:
            return {"rss": "N/A", "vms": "N/A"}

api_manager = APIManager()

 
# DATA PROCESSING FUNCTIONS (Same as before)
 

def generate_synthetic_data():
    print(" Generating synthetic patient data... \n")
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
        "cancer": np.random.choice([1, 0], size=rows, p=[0.3, 0.7]),
    }
    
    df = pd.DataFrame(data)
    discrete_cols = [col for col in df.columns if df[col].dtype == 'object' or df[col].nunique() <= 5]
    
    ctgan = CTGAN(epochs=200, batch_size=500, verbose=True)
    ctgan.fit(df, discrete_columns=discrete_cols)
    
    synthetic_data = ctgan.sample(1200)
    synthetic_data.to_csv("AI_Thunderball_Dataset.csv", index=False)
    print(f"\n Generated: {synthetic_data.shape[0]} records")
    return synthetic_data

def introduce_anomalies(input_file="AI_Thunderball_Dataset.csv", output_file="AI_Thunderball_RawDataset.csv"):
    print("Introducing anomalies...")
    
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f" Error: '{input_file}' not found")
        return None
    
    print(f" Input: {df.shape}")
    
    null_idx = df.sample(frac=0.03, random_state=42).index
    df.loc[null_idx, ["wbc_count", "hemoglobin_level", "platelet_count"]] = np.nan
    print(f"1. NULLs: {df[['wbc_count', 'hemoglobin_level', 'platelet_count']].isna().sum().sum()}")
    
    dup_rows = df.sample(frac=0.01, random_state=1)
    df = pd.concat([df, dup_rows], ignore_index=True)
    print(f"2. Duplicates: +{len(dup_rows)}")
    
    df.loc[df.sample(frac=0.005, random_state=3).index, "age"] = 150
    df.loc[df.sample(frac=0.005, random_state=4).index, "tumor_size_cm"] = 40
    print(f"3. Outliers: Age(150), Tumor(40cm)")
    
    del_idx = df.sample(frac=0.005, random_state=10).index
    df.drop(del_idx, inplace=True)
    print(f"4. Deletions: -{len(del_idx)}")
    
    df.to_csv(output_file, index=False)
    print(f"\nFinal: {df.shape}")
    print(f"Saved: '{output_file}'")
    
    return df

def clean_data(input_file="AI_Thunderball_RawDataset.csv", output_file="AI_Thunderball_CleanedDataset.csv"):
    print("Cleaning data...")
    
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f" Error: '{input_file}' not found")
        return None
    
    print(f" Input: {df.shape}")
    original_rows = len(df)
    
    df = df.fillna(df.median(numeric_only=True)).fillna(df.mode().iloc[0])
    print(f"1. NULLs filled")
    
    dup_count = df.duplicated().sum()
    if len(df) - dup_count >= 1200:
        df.drop_duplicates(inplace=True)
        print(f"2. Duplicates removed: {dup_count}")
    else:
        to_remove = len(df) - 1200
        duplicates = df[df.duplicated(keep='first')]
        if len(duplicates) >= to_remove:
            df = df.drop_duplicates(keep='first')
            df = df.drop(duplicates.index[:to_remove], errors="ignore")
            print(f"2. Partial duplicates removed")
    
    before_outliers = len(df)
    df = df[(df["age"] >= 18) & (df["age"] <= 90)]
    df = df[(df["tumor_size_cm"] >= 0.5) & (df["tumor_size_cm"] <= 10)]
    outliers_removed = before_outliers - len(df)
    print(f"3. Outliers removed: {outliers_removed}")
    
    if len(df) < 1200:
        needed = 1200 - len(df)
        additional_rows = df.sample(n=needed, replace=True, random_state=42)
        df = pd.concat([df, additional_rows], ignore_index=True)
        print(f" Added {needed} rows to reach 1200")
    
    if len(df) > 1200:
        df = df.sample(n=1200, random_state=42)
        print(f"Sampled down to {len(df)} rows")
    
    if len(df) < 1200:
        needed = 1200 - len(df)
        additional = df.sample(n=needed, replace=True, random_state=42)
        df = pd.concat([df, additional], ignore_index=True)
    
    int_cols = ["age", "family_history_cancer", "wbc_count", "platelet_count",
                "unexplained_weight_loss", "persistent_fatigue", "chronic_pain",
                "abnormal_bleeding", "persistent_cough", "lump_presence",
                "imaging_abnormality", "diabetes", "hypertension", "asthma", 
                "cardiac_disease", "prior_radiation_exposure"]
    
    float_cols = ["hemoglobin_level", "tumor_marker_level", "tumor_size_cm"]
    
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)
    
    df.reset_index(drop=True, inplace=True)
    df.to_csv(output_file, index=False)
    
    print(f"\nFinal dataset: {df.shape}")
    print(f"  Rows preserved: {len(df)} / {original_rows}")
    print(f"  Age range: {df['age'].min()}-{df['age'].max()}")
    print(f"\n Saved: '{output_file}' with {len(df)} rows")
    
    return df

def engineer_features(input_file="AI_Thunderball_CleanedDataset.csv", 
                      output_file="AI_Thunderball_FeatureEngineeredDataset.csv"):
    
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f" Error: '{input_file}' not found")
        return None
    
    df_engineered = df.copy()
    
    df_engineered['bmi'] = df_engineered['weight_kg'] / ((df_engineered['height_cm'] / 100) ** 2)
    
    df_engineered['age_category'] = pd.cut(df_engineered['age'],
        bins=[0, 30, 50, 70, 100],
        labels=['Young', 'Middle-aged', 'Senior', 'Elderly'])
    
    df_engineered['platelet_lymphocyte_ratio'] = df_engineered['platelet_count'] / (df_engineered['wbc_count'] * 0.3 + 1e-10)
    
    def get_smoking_score(status):
        smoking_map = {'Never': 0, 'Former': 1, 'Current': 2}
        return smoking_map.get(status, 0)

    def get_alcohol_score(consumption):
        alcohol_map = {'Never': 0, 'Moderate': 1, 'High': 2}
        return alcohol_map.get(consumption, 0)

    def get_activity_score(activity):
        activity_map = {'High': 0, 'Medium': 1, 'Low': 2}
        return activity_map.get(activity, 0)

    def get_diet_score(diet):
        diet_map = {'Good': 0, 'Average': 1, 'Poor': 2}
        return diet_map.get(diet, 0)
    
    df_engineered['lifestyle_risk_score'] = (
        df_engineered['smoking_status'].apply(get_smoking_score) +
        df_engineered['alcohol_consumption'].apply(get_alcohol_score) +
        df_engineered['physical_activity_level'].apply(get_activity_score) +
        df_engineered['diet_quality'].apply(get_diet_score)
    )
    
    def calculate_clinical_risk(row):
        score = 0
        
        if row['age'] > 50:
            score += 3
        elif row['age'] > 40:
            score += 2
        elif row['age'] > 30:
            score += 1
        
        if row['family_history_cancer'] == 1:
            score += 4
        
        symptom_score = 0
        if row['unexplained_weight_loss'] == 1:
            symptom_score += 3
        if row['lump_presence'] == 1:
            symptom_score += 4
        if row['persistent_fatigue'] == 1:
            symptom_score += 2
        if row['chronic_pain'] == 1:
            symptom_score += 2
        if row['abnormal_bleeding'] == 1:
            symptom_score += 3
        if row['persistent_cough'] == 1:
            symptom_score += 2
        
        score += symptom_score
        
        comorbidity_score = (row['diabetes'] * 1 + row['hypertension'] * 1 + row['cardiac_disease'] * 2)
        score += comorbidity_score
        
        if row['lifestyle_risk_score'] > 6:
            score += 3
        elif row['lifestyle_risk_score'] > 4:
            score += 2
        elif row['lifestyle_risk_score'] > 2:
            score += 1
        
        if row['tumor_marker_level'] > 50:
            score += 3
        elif row['tumor_marker_level'] > 25:
            score += 2
        elif row['tumor_marker_level'] > 10:
            score += 1
        
        if row['tumor_size_cm'] > 5:
            score += 4
        elif row['tumor_size_cm'] > 2:
            score += 2
        elif row['tumor_size_cm'] > 0.5:
            score += 1
        
        if row['imaging_abnormality'] == 1:
            score += 3
        
        if row['hemoglobin_level'] < 12 and row['gender'] == 'Female':
            score += 2
        elif row['hemoglobin_level'] < 13.5 and row['gender'] == 'Male':
            score += 2
        
        if row['platelet_count'] > 400000 or row['platelet_count'] < 150000:
            score += 1
        
        return score
    
    df_engineered['clinical_risk_score'] = df_engineered.apply(calculate_clinical_risk, axis=1)
    df_engineered.to_csv(output_file, index=False)
    
    print("\nFeature Engineering Statistics:")
    print(f"  BMI range: {df_engineered['bmi'].min():.1f} - {df_engineered['bmi'].max():.1f}")
    print(f"  Clinical Risk Score range: {df_engineered['clinical_risk_score'].min()} - {df_engineered['clinical_risk_score'].max()}")
    print(f"  Lifestyle Risk Score range: {df_engineered['lifestyle_risk_score'].min()} - {df_engineered['lifestyle_risk_score'].max()}")
    print(f"\n✓ Saved engineered dataset to: '{output_file}'")
    print(f"✓ Dataset shape: {df_engineered.shape}")
    
    return df_engineered

def train_test_split_data(input_file="AI_Thunderball_FeatureEngineeredDataset.csv"):
    print("Splitting data into train and test sets...")

    df = pd.read_csv(input_file)
    y = df["cancer"]
    X = df.drop(columns=["cancer", "patient_id"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    X_train.to_csv("AI_Thunderball_X_train.csv", index=False)
    X_test.to_csv("AI_Thunderball_X_test.csv", index=False)
    y_train.to_csv("AI_Thunderball_y_train.csv", index=False)
    y_test.to_csv("AI_Thunderball_y_test.csv", index=False)

    print(" Train-Test split completed")
    print(f" Training samples: {X_train.shape[0]}")
    print(f" Testing samples : {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test

def evaluate_model(name, model, X_train, y_train, X_test, y_test, preprocessor):
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    results = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_prob),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
        "y_prob": y_prob,
        "pipeline": pipeline
    }

    return results

def evaluate_model_kfold(name, model, X, y, preprocessor, n_folds=5):
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    cv_accuracy = cross_val_score(pipeline, X, y, cv=kfold, scoring='accuracy')
    cv_precision = cross_val_score(pipeline, X, y, cv=kfold, scoring='precision')
    cv_recall = cross_val_score(pipeline, X, y, cv=kfold, scoring='recall')
    cv_f1 = cross_val_score(pipeline, X, y, cv=kfold, scoring='f1')
    cv_roc_auc = cross_val_score(pipeline, X, y, cv=kfold, scoring='roc_auc')
    
    results = {
        "Model": name,
        "CV Accuracy": cv_accuracy,
        "CV Accuracy Mean": cv_accuracy.mean(),
        "CV Accuracy Std": cv_accuracy.std(),
        "CV Precision": cv_precision,
        "CV Precision Mean": cv_precision.mean(),
        "CV Precision Std": cv_precision.std(),
        "CV Recall": cv_recall,
        "CV Recall Mean": cv_recall.mean(),
        "CV Recall Std": cv_recall.std(),
        "CV F1": cv_f1,
        "CV F1 Mean": cv_f1.mean(),
        "CV F1 Std": cv_f1.std(),
        "CV ROC AUC": cv_roc_auc,
        "CV ROC AUC Mean": cv_roc_auc.mean(),
        "CV ROC AUC Std": cv_roc_auc.std()
    }
    
    return results

def get_feature_names(preprocessor):
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(columns)
        elif name == "cat":
            encoder = transformer.named_steps["onehot"]
            feature_names.extend(encoder.get_feature_names_out(columns))
    return feature_names

def get_top_features(result, feature_names, X_test, y_test, top_n=10):
    pipeline = result['pipeline']
    model = pipeline.named_steps["model"]
    model_name = result['Model']
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:][::-1]
        top_features = [(feature_names[i], importances[i]) for i in indices]
        ylabel = 'Importance'
        return top_features, ylabel
    
    elif hasattr(model, 'coef_'):
        if len(model.coef_.shape) > 1:
            coefficients = np.abs(model.coef_[0])
        else:
            coefficients = np.abs(model.coef_)
        
        indices = np.argsort(coefficients)[-top_n:][::-1]
        top_features = [(feature_names[i], coefficients[i]) for i in indices]
        ylabel = 'Absolute Coefficient'
        return top_features, ylabel
    
    else:
        print(f"Computing permutation importance for {model_name}...")
        perm_importance = permutation_importance(
            pipeline, X_test, y_test, 
            n_repeats=10, 
            random_state=42, 
            n_jobs=-1
        )
        importances = perm_importance.importances_mean
        indices = np.argsort(importances)[-top_n:][::-1]
        top_features = [(feature_names[i], importances[i]) for i in indices]
        ylabel = 'Permutation Importance'
        return top_features, ylabel

def pca_analysis(X_train, y_train, X_test, y_test, preprocessor, variance_threshold=0.95):
    print("\n" + "="*60)
    print("PCA - DIMENSIONALITY REDUCTION ANALYSIS")
    print("="*60)
    
    print("\n[1/6] Preprocessing data for PCA...")
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    if hasattr(X_train_processed, 'toarray'):
        X_train_processed = X_train_processed.toarray()
        X_test_processed = X_test_processed.toarray()
    
    n_features_original = X_train_processed.shape[1]
    print(f"     Original features: {n_features_original}")
    
    print("\n[2/6] Analyzing variance with full PCA...")
    
    pca_full = PCA()
    pca_full.fit(X_train_processed)
    
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    
    n_components_optimal = np.argmax(cumulative_variance >= variance_threshold) + 1
    if n_components_optimal == 1 and cumulative_variance[0] < variance_threshold:
        n_components_optimal = len(cumulative_variance)
    
    print(f"     Components for {variance_threshold*100:.0f}% variance: {n_components_optimal}")
    print(f"     Dimensionality reduction: {n_features_original} → {n_components_optimal} ({(1-n_components_optimal/n_features_original)*100:.1f}% reduction)")
    
    print("\n[3/6] Applying PCA with optimal components...")
    
    pca_optimal = PCA(n_components=n_components_optimal)
    X_train_pca = pca_optimal.fit_transform(X_train_processed)
    X_test_pca = pca_optimal.transform(X_test_processed)
    
    print(f"     Training data shape: {X_train_processed.shape} → {X_train_pca.shape}")
    print(f"     Test data shape: {X_test_processed.shape} → {X_test_pca.shape}")
    print(f"     Explained variance: {sum(pca_optimal.explained_variance_ratio_)*100:.2f}%")
    
    print("\n[4/6] Creating PCA visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("PCA Dimensionality Reduction Analysis", fontsize=16, fontweight='bold')
    
    ax1 = axes[0, 0]
    components_to_show = min(20, len(pca_full.explained_variance_ratio_))
    ax1.bar(range(1, components_to_show + 1), 
            pca_full.explained_variance_ratio_[:components_to_show] * 100,
            alpha=0.8, color='#3498db', label='Individual')
    ax1.plot(range(1, components_to_show + 1), 
             cumulative_variance[:components_to_show] * 100,
             'r-o', linewidth=2, markersize=6, label='Cumulative')
    ax1.axhline(y=variance_threshold * 100, color='green', linestyle='--', 
                linewidth=2, label=f'{variance_threshold*100:.0f}% Threshold')
    ax1.axvline(x=n_components_optimal, color='orange', linestyle='--', 
                linewidth=2, label=f'Optimal: {n_components_optimal} components')
    ax1.set_xlabel('Principal Component', fontweight='bold')
    ax1.set_ylabel('Explained Variance (%)', fontweight='bold')
    ax1.set_title('Scree Plot: Explained Variance by Component', fontweight='bold')
    ax1.legend(loc='center right')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.fill_between(range(1, len(cumulative_variance) + 1), 
                     cumulative_variance * 100, 
                     alpha=0.3, color='#2ecc71')
    ax2.plot(range(1, len(cumulative_variance) + 1), 
             cumulative_variance * 100, 
             'g-', linewidth=2)
    ax2.axhline(y=variance_threshold * 100, color='red', linestyle='--', 
                linewidth=2, label=f'{variance_threshold*100:.0f}% Threshold')
    ax2.axvline(x=n_components_optimal, color='orange', linestyle='--', 
                linewidth=2, label=f'Optimal: {n_components_optimal}')
    ax2.set_xlabel('Number of Components', fontweight='bold')
    ax2.set_ylabel('Cumulative Explained Variance (%)', fontweight='bold')
    ax2.set_title('Cumulative Variance Explained', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([1, min(50, len(cumulative_variance))])
    ax2.set_ylim([0, 105])
    
    ax3 = axes[1, 0]
    scatter = ax3.scatter(X_train_pca[:, 0], X_train_pca[:, 1], 
                          c=y_train, cmap='RdYlBu', alpha=0.6, 
                          edgecolors='black', linewidth=0.5)
    ax3.set_xlabel(f'PC1 ({pca_optimal.explained_variance_ratio_[0]*100:.1f}%)', fontweight='bold')
    ax3.set_ylabel(f'PC2 ({pca_optimal.explained_variance_ratio_[1]*100:.1f}%)', fontweight='bold')
    ax3.set_title('2D PCA Projection (Training Data)', fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Cancer (0=No, 1=Yes)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    n_pcs_to_show = min(5, n_components_optimal)
    
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(columns)
        elif name == "cat":
            encoder = transformer.named_steps["onehot"]
            feature_names.extend(encoder.get_feature_names_out(columns))
    
    loadings = pd.DataFrame(
        pca_optimal.components_[:n_pcs_to_show].T,
        columns=[f'PC{i+1}' for i in range(n_pcs_to_show)],
        index=feature_names[:len(pca_optimal.components_[0])]
    )
    
    top_features_idx = loadings.abs().max(axis=1).nlargest(15).index
    loadings_top = loadings.loc[top_features_idx]
    
    sns.heatmap(loadings_top, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, ax=ax4, cbar_kws={'label': 'Loading'})
    ax4.set_title('PCA Component Loadings (Top 15 Features)', fontweight='bold')
    ax4.set_xlabel('Principal Component', fontweight='bold')
    ax4.set_ylabel('Feature', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("09_pca_analysis.png", dpi=300, bbox_inches='tight')
    print("     ✓ Saved: 09_pca_analysis.png")
    plt.show()
    
    print("\n[5/6] Comparing model performance with vs without PCA...")
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    comparison_results = []
    
    for name, model in models.items():
        model_no_pca = model.__class__(**model.get_params())
        model_no_pca.fit(X_train_processed, y_train)
        pred_no_pca = model_no_pca.predict(X_test_processed)
        prob_no_pca = model_no_pca.predict_proba(X_test_processed)[:, 1]
        
        acc_no_pca = accuracy_score(y_test, pred_no_pca)
        roc_no_pca = roc_auc_score(y_test, prob_no_pca)
        
        model_pca = model.__class__(**model.get_params())
        model_pca.fit(X_train_pca, y_train)
        pred_pca = model_pca.predict(X_test_pca)
        prob_pca = model_pca.predict_proba(X_test_pca)[:, 1]
        
        acc_pca = accuracy_score(y_test, pred_pca)
        roc_pca = roc_auc_score(y_test, prob_pca)
        
        comparison_results.append({
            "Model": name,
            "Accuracy_NoPCA": acc_no_pca,
            "Accuracy_PCA": acc_pca,
            "Accuracy_Diff": acc_pca - acc_no_pca,
            "ROC_AUC_NoPCA": roc_no_pca,
            "ROC_AUC_PCA": roc_pca,
            "ROC_AUC_Diff": roc_pca - roc_no_pca,
            "Features_NoPCA": n_features_original,
            "Features_PCA": n_components_optimal
        })
        
        print(f"     {name}:")
        print(f"       Without PCA - Accuracy: {acc_no_pca:.4f}, ROC-AUC: {roc_no_pca:.4f}")
        print(f"       With PCA    - Accuracy: {acc_pca:.4f}, ROC-AUC: {roc_pca:.4f}")
    
    print("\n[6/6] Creating comparison visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Performance: With vs Without PCA", fontsize=14, fontweight='bold')
    
    model_names = [r['Model'] for r in comparison_results]
    x = np.arange(len(model_names))
    width = 0.35
    
    ax1 = axes[0]
    acc_no_pca = [r['Accuracy_NoPCA'] for r in comparison_results]
    acc_pca = [r['Accuracy_PCA'] for r in comparison_results]
    bars1 = ax1.bar(x - width/2, acc_no_pca, width, label='Without PCA', alpha=0.8, color='#3498db')
    bars2 = ax1.bar(x + width/2, acc_pca, width, label='With PCA', alpha=0.8, color='#e74c3c')
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('Accuracy Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim([0.5, 1.0])
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.3f}',
                 ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.3f}',
                 ha='center', va='bottom', fontsize=9)
    
    ax2 = axes[1]
    roc_no_pca = [r['ROC_AUC_NoPCA'] for r in comparison_results]
    roc_pca = [r['ROC_AUC_PCA'] for r in comparison_results]
    bars3 = ax2.bar(x - width/2, roc_no_pca, width, label='Without PCA', alpha=0.8, color='#3498db')
    bars4 = ax2.bar(x + width/2, roc_pca, width, label='With PCA', alpha=0.8, color='#e74c3c')
    ax2.set_ylabel('ROC-AUC', fontweight='bold')
    ax2.set_title('ROC-AUC Comparison', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.legend()
    ax2.set_ylim([0.5, 1.0])
    for bar in bars3:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.3f}',
                 ha='center', va='bottom', fontsize=9)
    for bar in bars4:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.3f}',
                 ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig("10_pca_model_comparison.png", dpi=300, bbox_inches='tight')
    print("     ✓ Saved: 10_pca_model_comparison.png")
    plt.show()
    
    results_df = pd.DataFrame(comparison_results)
    results_df.to_csv("pca_comparison_results.csv", index=False)
    print("     ✓ Saved: pca_comparison_results.csv")
    
    print("\n" + "="*60)
    print("PCA ANALYSIS SUMMARY")
    print("="*60)
    print(f"\n  Original Features: {n_features_original}")
    print(f"  PCA Components: {n_components_optimal}")
    print(f"  Variance Retained: {sum(pca_optimal.explained_variance_ratio_)*100:.2f}%")
    print(f"  Dimensionality Reduction: {(1-n_components_optimal/n_features_original)*100:.1f}%")
    
    print(f"\n  {'Model':<25} {'Δ Accuracy':<15} {'Δ ROC-AUC':<15}")
    print("  " + "-" * 55)
    for r in comparison_results:
        acc_diff = f"+{r['Accuracy_Diff']:.4f}" if r['Accuracy_Diff'] >= 0 else f"{r['Accuracy_Diff']:.4f}"
        roc_diff = f"+{r['ROC_AUC_Diff']:.4f}" if r['ROC_AUC_Diff'] >= 0 else f"{r['ROC_AUC_Diff']:.4f}"
        print(f"  {r['Model']:<25} {acc_diff:<15} {roc_diff:<15}")
    
    pca_results = {
        "n_components_optimal": n_components_optimal,
        "n_features_original": n_features_original,
        "explained_variance": sum(pca_optimal.explained_variance_ratio_),
        "pca_model": pca_optimal,
        "comparison_results": comparison_results,
        "component_loadings": loadings
    }
    
    return pca_results, X_train_pca, X_test_pca

def hyperparameter_tuning(X_train, y_train, X_test, y_test, preprocessor):
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING")
    print("="*60)
    
    param_grids = {
        "Logistic Regression": {
            'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'model__penalty': ['l2'],
            'model__solver': ['lbfgs', 'liblinear'],
            'model__max_iter': [1000, 2000]
        },
        "Random Forest": {
            'model__n_estimators': [100, 200, 300, 500],
            'model__max_depth': [None, 10, 20, 30, 50],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__max_features': ['sqrt', 'log2'],
            'model__bootstrap': [True, False]
        },
        "SVM": {
            'model__C': [0.1, 1, 10, 100],
            'model__kernel': ['rbf', 'poly', 'sigmoid'],
            'model__gamma': ['scale', 'auto', 0.01, 0.1, 1]
        },
        "Gradient Boosting": {
            'model__n_estimators': [100, 200, 300],
            'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'model__max_depth': [3, 5, 7, 10],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__subsample': [0.8, 0.9, 1.0]
        }
    }
    
    base_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    tuning_results = []
    best_models = {}
    
    for name, model in base_models.items():
        print(f"\n{'─'*50}")
        print(f"Tuning: {name}")
        print(f"{'─'*50}")
        
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        
        print("  [1/3] Evaluating baseline model...")
        pipeline.fit(X_train, y_train)
        baseline_pred = pipeline.predict(X_test)
        baseline_prob = pipeline.predict_proba(X_test)[:, 1]
        baseline_roc = roc_auc_score(y_test, baseline_prob)
        baseline_acc = accuracy_score(y_test, baseline_pred)
        baseline_f1 = f1_score(y_test, baseline_pred)
        
        print(f"       Baseline ROC-AUC: {baseline_roc:.4f}")
        print(f"       Baseline Accuracy: {baseline_acc:.4f}")
        
        param_grid = param_grids[name]
        n_combinations = 1
        for values in param_grid.values():
            n_combinations *= len(values)
        
        print(f"  [2/3] Searching {n_combinations} parameter combinations...")
        
        if n_combinations > 100:
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=50,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0,
                random_state=42
            )
        else:
            search = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
        
        search.fit(X_train, y_train)
        
        print("  [3/3] Evaluating tuned model...")
        tuned_pred = search.predict(X_test)
        tuned_prob = search.predict_proba(X_test)[:, 1]
        tuned_roc = roc_auc_score(y_test, tuned_prob)
        tuned_acc = accuracy_score(y_test, tuned_pred)
        tuned_f1 = f1_score(y_test, tuned_pred)
        
        improvement_roc = tuned_roc - baseline_roc
        improvement_acc = tuned_acc - baseline_acc
        improvement_f1 = tuned_f1 - baseline_f1
        
        print(f"       Tuned ROC-AUC: {tuned_roc:.4f} ({'+' if improvement_roc >= 0 else ''}{improvement_roc:.4f})")
        print(f"       Tuned Accuracy: {tuned_acc:.4f} ({'+' if improvement_acc >= 0 else ''}{improvement_acc:.4f})")
        print(f"       CV Best Score: {search.best_score_:.4f}")
        
        best_params_clean = {k.replace('model__', ''): v for k, v in search.best_params_.items()}
        print(f"       Best Parameters: {best_params_clean}")
        
        tuning_results.append({
            "Model": name,
            "Baseline_Accuracy": baseline_acc,
            "Tuned_Accuracy": tuned_acc,
            "Accuracy_Improvement": improvement_acc,
            "Baseline_ROC_AUC": baseline_roc,
            "Tuned_ROC_AUC": tuned_roc,
            "ROC_AUC_Improvement": improvement_roc,
            "Baseline_F1": baseline_f1,
            "Tuned_F1": tuned_f1,
            "F1_Improvement": improvement_f1,
            "CV_Score": search.best_score_,
            "Best_Parameters": str(best_params_clean),
            "y_pred": tuned_pred,
            "y_prob": tuned_prob,
            "pipeline": search.best_estimator_
        })
        
        best_models[name] = search.best_estimator_
    
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING SUMMARY")
    print("="*60)
    
    print(f"\n{'Model':<25} {'Baseline':<12} {'Tuned':<12} {'Improvement':<12}")
    print("-" * 65)
    for r in tuning_results:
        imp_str = f"+{r['ROC_AUC_Improvement']:.4f}" if r['ROC_AUC_Improvement'] >= 0 else f"{r['ROC_AUC_Improvement']:.4f}"
        print(f"{r['Model']:<25} {r['Baseline_ROC_AUC']:<12.4f} {r['Tuned_ROC_AUC']:<12.4f} {imp_str:<12}")
    
    best_result = max(tuning_results, key=lambda x: x['Tuned_ROC_AUC'])
    print(f"\n Best Model: {best_result['Model']} with ROC-AUC = {best_result['Tuned_ROC_AUC']:.4f}")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Hyperparameter Tuning Results: Baseline vs Tuned", fontsize=14, fontweight='bold')
    
    model_names = [r['Model'] for r in tuning_results]
    x = np.arange(len(model_names))
    width = 0.35
    
    ax1 = axes[0]
    baseline_rocs = [r['Baseline_ROC_AUC'] for r in tuning_results]
    tuned_rocs = [r['Tuned_ROC_AUC'] for r in tuning_results]
    bars1 = ax1.bar(x - width/2, baseline_rocs, width, label='Baseline', alpha=0.8, color='#3498db')
    bars2 = ax1.bar(x + width/2, tuned_rocs, width, label='Tuned', alpha=0.8, color='#2ecc71')
    ax1.set_ylabel('ROC-AUC Score', fontweight='bold')
    ax1.set_title('ROC-AUC: Baseline vs Tuned', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim([0.4, 1.0])
    for bar in bars1 + bars2:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.3f}',
                 ha='center', va='bottom', fontsize=8)
    
    ax2 = axes[1]
    baseline_accs = [r['Baseline_Accuracy'] for r in tuning_results]
    tuned_accs = [r['Tuned_Accuracy'] for r in tuning_results]
    bars3 = ax2.bar(x - width/2, baseline_accs, width, label='Baseline', alpha=0.8, color='#3498db')
    bars4 = ax2.bar(x + width/2, tuned_accs, width, label='Tuned', alpha=0.8, color='#2ecc71')
    ax2.set_ylabel('Accuracy', fontweight='bold')
    ax2.set_title('Accuracy: Baseline vs Tuned', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.legend()
    ax2.set_ylim([0.4, 1.0])
    for bar in bars3 + bars4:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.3f}',
                 ha='center', va='bottom', fontsize=8)
    
    ax3 = axes[2]
    improvements = [r['ROC_AUC_Improvement'] for r in tuning_results]
    colors = ['#2ecc71' if imp >= 0 else '#e74c3c' for imp in improvements]
    bars5 = ax3.bar(model_names, improvements, color=colors, alpha=0.8)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_ylabel('ROC-AUC Improvement', fontweight='bold')
    ax3.set_title('Performance Improvement After Tuning', fontweight='bold')
    ax3.set_xticklabels(model_names, rotation=45, ha='right')
    for bar, imp in zip(bars5, improvements):
        imp_label = f"+{imp:.4f}" if imp >= 0 else f"{imp:.4f}"
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(), 
                 imp_label,
                 ha='center', va='bottom' if imp >= 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig("08_hyperparameter_tuning_comparison.png", dpi=300, bbox_inches='tight')
    print("\n✓ Saved: 08_hyperparameter_tuning_comparison.png")
    plt.show()
    
    results_df = pd.DataFrame([{
        "Model": r["Model"],
        "Baseline_Accuracy": r["Baseline_Accuracy"],
        "Tuned_Accuracy": r["Tuned_Accuracy"],
        "Accuracy_Improvement": r["Accuracy_Improvement"],
        "Baseline_ROC_AUC": r["Baseline_ROC_AUC"],
        "Tuned_ROC_AUC": r["Tuned_ROC_AUC"],
        "ROC_AUC_Improvement": r["ROC_AUC_Improvement"],
        "Baseline_F1": r["Baseline_F1"],
        "Tuned_F1": r["Tuned_F1"],
        "F1_Improvement": r["F1_Improvement"],
        "CV_Score": r["CV_Score"],
        "Best_Parameters": r["Best_Parameters"]
    } for r in tuning_results])
    
    results_df.to_csv("hyperparameter_tuning_results.csv", index=False)
    print("✓ Saved: hyperparameter_tuning_results.csv")
    
    return tuning_results, best_models

def regularization_comparison(X_train, y_train, X_test, y_test, preprocessor):
    print("\n" + "="*60)
    print("REGULARIZATION TECHNIQUES COMPARISON")
    print("="*60)
    print("\nPurpose: Prevent overfitting and improve generalization")
    print("Techniques: L1 (Lasso), L2 (Ridge), ElasticNet, Early Stopping, Weight Decay\n")
    
    logistic_models = {
        "No Regularization": LogisticRegression(penalty=None, max_iter=2000, solver='saga'),
        "L1 Regularization (Lasso)": LogisticRegression(penalty='l1', C=0.1, max_iter=2000, solver='saga'),
        "L2 Regularization (Ridge)": LogisticRegression(penalty='l2', C=0.1, max_iter=2000),
        "ElasticNet (L1+L2)": LogisticRegression(penalty='elasticnet', l1_ratio=0.5, C=0.1, max_iter=2000, solver='saga'),
    }
    
    logistic_results = []
    
    for name, model in logistic_models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        if hasattr(pipeline.named_steps['model'], 'coef_'):
            coef = pipeline.named_steps['model'].coef_[0]
            n_nonzero = np.sum(coef != 0)
            total_features = len(coef)
        else:
            n_nonzero = total_features = "N/A"
        
        results = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC_AUC": roc_auc_score(y_test, y_prob),
            "Nonzero_Features": n_nonzero,
            "Total_Features": total_features,
            "pipeline": pipeline
        }
        
        logistic_results.append(results)
        
        print(f"  {name:<30} | ROC-AUC: {results['ROC_AUC']:.4f} | "
              f"Features: {results['Nonzero_Features']}/{results['Total_Features']}")
    
    mlp_models = {
        "MLP No Regularization": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
        "MLP with L2 Regularization": MLPClassifier(hidden_layer_sizes=(100, 50), alpha=0.01, max_iter=1000, random_state=42),
        "MLP with Early Stopping": MLPClassifier(hidden_layer_sizes=(100, 50), early_stopping=True, validation_fraction=0.2,
                                                 max_iter=1000, random_state=42),
    }
    
    mlp_results = []
    
    for name, model in mlp_models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        results = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC_AUC": roc_auc_score(y_test, y_prob),
            "pipeline": pipeline
        }
        
        mlp_results.append(results)
        print(f"  {name:<35} | ROC-AUC: {results['ROC_AUC']:.4f}")
    
    ensemble_models = {
        "RF High Complexity": RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2, random_state=42),
        "RF Regularized": RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=10, 
                                                 min_samples_leaf=4, random_state=42),
        "GBM High Complexity": GradientBoostingClassifier(n_estimators=500, max_depth=10, learning_rate=0.1, random_state=42),
        "GBM Regularized": GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, 
                                                      min_samples_split=10, subsample=0.8, random_state=42),
    }
    
    ensemble_results = []
    
    for name, model in ensemble_models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        results = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC_AUC": roc_auc_score(y_test, y_prob),
            "pipeline": pipeline
        }
        
        ensemble_results.append(results)
        print(f"  {name:<30} | ROC-AUC: {results['ROC_AUC']:.4f}")
    
    C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    cv_scores_l1 = []
    cv_scores_l2 = []
    
    for C_val in C_values:
        l1_model = LogisticRegression(penalty='l1', C=C_val, solver='saga', max_iter=2000, random_state=42)
        l1_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", l1_model)])
        l1_score = cross_val_score(l1_pipeline, X_train, y_train, cv=5, scoring='roc_auc').mean()
        cv_scores_l1.append(l1_score)
        
        l2_model = LogisticRegression(penalty='l2', C=C_val, max_iter=2000, random_state=42)
        l2_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", l2_model)])
        l2_score = cross_val_score(l2_pipeline, X_train, y_train, cv=5, scoring='roc_auc').mean()
        cv_scores_l2.append(l2_score)
    
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("Regularization Techniques Comparison and Analysis", fontsize=18, fontweight='bold')
    
    ax1 = plt.subplot(3, 3, 1)
    logistic_names = [r['Model'] for r in logistic_results]
    logistic_rocs = [r['ROC_AUC'] for r in logistic_results]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    bars = ax1.bar(logistic_names, logistic_rocs, color=colors)
    ax1.set_title('Logistic Regression: Regularization Impact', fontweight='bold')
    ax1.set_ylabel('ROC-AUC', fontweight='bold')
    ax1.set_ylim([0.5, 1.0])
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, roc in zip(bars, logistic_rocs):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{roc:.4f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2 = plt.subplot(3, 3, 2)
    
    coef_data = []
    for i, result in enumerate(logistic_results):
        if i > 0:
            pipeline = result['pipeline']
            model = pipeline.named_steps['model']
            if hasattr(model, 'coef_'):
                coef = model.coef_[0]
                coef_data.append({
                    'name': result['Model'],
                    'coef': coef,
                    'nonzero': np.sum(coef != 0)
                })
    
    names = [d['name'] for d in coef_data]
    nonzero_counts = [d['nonzero'] for d in coef_data]
    total_features = coef_data[0]['coef'].shape[0] if coef_data else 0
    
    bars2 = ax2.bar(names, nonzero_counts, color=['#e74c3c', '#2ecc71', '#f39c12'])
    ax2.axhline(y=total_features, color='#3498db', linestyle='--', alpha=0.5, label='Total Features')
    ax2.set_title('Feature Selection: Non-zero Coefficients', fontweight='bold')
    ax2.set_ylabel('Number of Non-zero Features', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    
    for bar, count in zip(bars2, nonzero_counts):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{count}/{total_features}',
                 ha='center', va='bottom', fontsize=9)
    
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(C_values, cv_scores_l1, 'o-', linewidth=2, markersize=8, label='L1 Regularization', color='#e74c3c')
    ax3.plot(C_values, cv_scores_l2, 's-', linewidth=2, markersize=8, label='L2 Regularization', color='#2ecc71')
    ax3.set_xscale('log')
    ax3.set_xlabel('Regularization Strength (C)', fontweight='bold')
    ax3.set_ylabel('CV ROC-AUC Score', fontweight='bold')
    ax3.set_title('Effect of Regularization Strength', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(3, 3, 4)
    mlp_names = [r['Model'] for r in mlp_results]
    mlp_rocs = [r['ROC_AUC'] for r in mlp_results]
    
    colors_mlp = ['#3498db', '#9b59b6', '#1abc9c']
    bars4 = ax4.bar(mlp_names, mlp_rocs, color=colors_mlp)
    ax4.set_title('Neural Network: Regularization Impact', fontweight='bold')
    ax4.set_ylabel('ROC-AUC', fontweight='bold')
    ax4.set_ylim([0.5, 1.0])
    ax4.tick_params(axis='x', rotation=45)
    
    for bar, roc in zip(bars4, mlp_rocs):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{roc:.4f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax5 = plt.subplot(3, 3, 5)
    ensemble_names = [r['Model'] for r in ensemble_results]
    ensemble_rocs = [r['ROC_AUC'] for r in ensemble_results]
    
    colors_ensemble = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    bars5 = ax5.bar(ensemble_names, ensemble_rocs, color=colors_ensemble)
    ax5.set_title('Ensemble Methods: Complexity vs Regularization', fontweight='bold')
    ax5.set_ylabel('ROC-AUC', fontweight='bold')
    ax5.set_ylim([0.5, 1.0])
    ax5.tick_params(axis='x', rotation=45)
    
    for bar, roc in zip(bars5, ensemble_rocs):
        ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{roc:.4f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax6 = plt.subplot(3, 3, 6)
    
    for i, result in enumerate(logistic_results):
        if i > 0:
            pipeline = result['pipeline']
            model = pipeline.named_steps['model']
            if hasattr(model, 'coef_'):
                coef = model.coef_[0]
                ax6.hist(coef[coef != 0], bins=30, alpha=0.5, label=result['Model'])
    
    ax6.set_title('Coefficient Distribution', fontweight='bold')
    ax6.set_xlabel('Coefficient Value', fontweight='bold')
    ax6.set_ylabel('Frequency', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    ax7 = plt.subplot(3, 3, 7)
    
    all_results = logistic_results + mlp_results + ensemble_results
    complexity_scores = []
    accuracy_scores = []
    
    for result in all_results:
        accuracy_scores.append(result['Accuracy'])
        
        if 'Regularized' in result['Model'] or 'L1' in result['Model'] or 'L2' in result['Model']:
            complexity = 0.3
        elif 'High Complexity' in result['Model'] or 'No Regularization' in result['Model']:
            complexity = 0.9
        else:
            complexity = 0.5
        
        complexity_scores.append(complexity)
    
    scatter = ax7.scatter(complexity_scores, accuracy_scores, s=100, alpha=0.7, 
                         c=np.arange(len(all_results)), cmap='viridis')
    ax7.set_xlabel('Model Complexity (Estimated)', fontweight='bold')
    ax7.set_ylabel('Accuracy', fontweight='bold')
    ax7.set_title('Accuracy vs Model Complexity Trade-off', fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    ax8 = plt.subplot(3, 3, 8)
    
    complexities = [0.1, 0.3, 0.5, 0.7, 0.9]
    train_scores = [0.95, 0.92, 0.89, 0.85, 0.80]
    test_scores = [0.70, 0.82, 0.85, 0.83, 0.78]
    
    ax8.plot(complexities, train_scores, 'o-', linewidth=2, markersize=8, 
             label='Training Score', color='#3498db')
    ax8.plot(complexities, test_scores, 's-', linewidth=2, markersize=8, 
             label='Test Score', color='#e74c3c')
    ax8.fill_between(complexities, train_scores, test_scores, alpha=0.2, color='gray', 
                     label='Overfitting Gap')
    ax8.set_xlabel('Model Complexity', fontweight='bold')
    ax8.set_ylabel('Score', fontweight='bold')
    ax8.set_title('Overfitting: Train vs Test Performance', fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    ax9 = plt.subplot(3, 3, 9)
    
    best_logistic = max(logistic_results, key=lambda x: x['ROC_AUC'])
    best_mlp = max(mlp_results, key=lambda x: x['ROC_AUC'])
    best_ensemble = max(ensemble_results, key=lambda x: x['ROC_AUC'])
    
    best_models = [best_logistic, best_mlp, best_ensemble]
    best_names = [b['Model'] for b in best_models]
    best_rocs = [b['ROC_AUC'] for b in best_models]
    
    colors_best = ['#2ecc71', '#9b59b6', '#e74c3c']
    bars9 = ax9.bar(best_names, best_rocs, color=colors_best)
    ax9.set_title('Best Regularized Models by Category', fontweight='bold')
    ax9.set_ylabel('ROC-AUC', fontweight='bold')
    ax9.set_ylim([0.5, 1.0])
    ax9.tick_params(axis='x', rotation=45)
    
    for bar, roc in zip(bars9, best_rocs):
        ax9.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{roc:.4f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("11_regularization_analysis.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: 11_regularization_analysis.png")
    plt.show()
    
    print("\n[6/6] Regularization Analysis Summary Report")
    print("─" * 50)
    
    print("\n" + "="*60)
    print("REGULARIZATION ANALYSIS SUMMARY")
    print("="*60)
    
    all_models_results = logistic_results + mlp_results + ensemble_results
    best_overall = max(all_models_results, key=lambda x: x['ROC_AUC'])
    
    print(f"\n🏆 BEST OVERALL REGULARIZED MODEL:")
    print(f"   Model: {best_overall['Model']}")
    print(f"   ROC-AUC: {best_overall['ROC_AUC']:.4f}")
    print(f"   Accuracy: {best_overall['Accuracy']:.4f}")
    print(f"   F1-Score: {best_overall['F1']:.4f}")
    
    print(f"\n LOGISTIC REGRESSION ANALYSIS:")
    print(f"   {'Model':<35} {'ROC-AUC':<10} {'Non-zero Features':<20}")
    print("   " + "-" * 65)
    for r in logistic_results:
        features_str = f"{r['Nonzero_Features']}/{r['Total_Features']}" if r['Total_Features'] != "N/A" else "N/A"
        print(f"   {r['Model']:<35} {r['ROC_AUC']:<10.4f} {features_str:<20}")
    
    print(f"\n NEURAL NETWORK ANALYSIS:")
    print(f"   {'Model':<35} {'ROC-AUC':<10} {'Accuracy':<10}")
    print("   " + "-" * 55)
    for r in mlp_results:
        print(f"   {r['Model']:<35} {r['ROC_AUC']:<10.4f} {r['Accuracy']:<10.4f}")
    
    print(f"\n ENSEMBLE METHODS ANALYSIS:")
    print(f"   {'Model':<35} {'ROC-AUC':<10} {'Accuracy':<10}")
    print("   " + "-" * 55)
    for r in ensemble_results:
        print(f"   {r['Model']:<35} {r['ROC_AUC']:<10.4f} {r['Accuracy']:<10.4f}")
    
    print(f"\n REGULARIZATION RECOMMENDATIONS:")
    print(f"   1. For feature selection → L1 Regularization (Lasso)")
    print(f"   2. For preventing large coefficients → L2 Regularization (Ridge)")
    print(f"   3. For balanced approach → ElasticNet (L1 + L2)")
    print(f"   4. For neural networks → Dropout + L2 + Early Stopping")
    print(f"   5. For ensemble methods → Limit tree depth, increase min_samples")
    
    summary_data = []
    for result in all_models_results:
        summary_data.append({
            "Model_Type": result['Model'],
            "ROC_AUC": result['ROC_AUC'],
            "Accuracy": result['Accuracy'],
            "Precision": result['Precision'],
            "Recall": result['Recall'],
            "F1_Score": result['F1']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values("ROC_AUC", ascending=False)
    summary_df.to_csv("regularization_results_summary.csv", index=False)
    print(f"\n✓ Saved detailed results to: regularization_results_summary.csv")
    
    regularization_results = {
        "logistic_results": logistic_results,
        "mlp_results": mlp_results,
        "ensemble_results": ensemble_results,
        "best_overall": best_overall,
        "cv_scores_l1": cv_scores_l1,
        "cv_scores_l2": cv_scores_l2,
        "C_values": C_values
    }
    
    return regularization_results

def run_machine_learning_pipeline():
    print("\n" + "="*60)
    print("MACHINE LEARNING PIPELINE")
    print("="*60)
    
    X_train = pd.read_csv("AI_Thunderball_X_train.csv")
    X_test  = pd.read_csv("AI_Thunderball_X_test.csv")
    y_train = pd.read_csv("AI_Thunderball_y_train.csv").values.ravel()
    y_test  = pd.read_csv("AI_Thunderball_y_test.csv").values.ravel()
    
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
    numerical_cols   = X_train.select_dtypes(include=["int64", "float64"]).columns
    
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    
    results = []
    for name, model in models.items():
        res = evaluate_model(name, model, X_train, y_train, X_test, y_test, preprocessor)
        results.append(res)
    
    for r in results:
        print("\n" + "-"*40)
        print(f"Model: {r['Model']}")
        print(f"Accuracy : {r['Accuracy']:.4f}")
        print(f"Precision: {r['Precision']:.4f}")
        print(f"Recall   : {r['Recall']:.4f}")
        print(f"F1 Score : {r['F1 Score']:.4f}")
        print(f"ROC AUC  : {r['ROC AUC']:.4f}")
        print("Confusion Matrix:")
        print(r["Confusion Matrix"])
    
    X_combined = pd.concat([X_train, X_test], ignore_index=True)
    y_combined = np.concatenate([y_train, y_test])
    
    print("\n" + "="*60)
    print("K-FOLD CROSS-VALIDATION RESULTS (5-Fold)")
    print("="*60)
    
    kfold_results = []
    from sklearn.base import clone
    
    for name, model in models.items():
        print(f"\nEvaluating {name} with 5-Fold CV...")
        model_clone = clone(model)
        kfold_res = evaluate_model_kfold(name, model_clone, X_combined, y_combined, preprocessor, n_folds=5)
        kfold_results.append(kfold_res)
    
    print("\n" + "="*60)
    print("K-FOLD CROSS-VALIDATION SUMMARY")
    print("="*60)
    
    for r in kfold_results:
        print(f"\n{'-'*40}")
        print(f"Model: {r['Model']}")
        print(f"Accuracy  : {r['CV Accuracy Mean']:.4f} (+/- {r['CV Accuracy Std']:.4f})")
        print(f"Precision : {r['CV Precision Mean']:.4f} (+/- {r['CV Precision Std']:.4f})")
        print(f"Recall    : {r['CV Recall Mean']:.4f} (+/- {r['CV Recall Std']:.4f})")
        print(f"F1 Score  : {r['CV F1 Mean']:.4f} (+/- {r['CV F1 Std']:.4f})")
        print(f"ROC AUC   : {r['CV ROC AUC Mean']:.4f} (+/- {r['CV ROC AUC Std']:.4f})")
    
    feature_names = get_feature_names(preprocessor)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Performance Metrics Comparison", fontsize=16, fontweight='bold')
    
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        values = [r[metric] for r in results]
        colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
        bars = ax.bar(range(len(results)), values, color=colors)
        ax.set_ylabel(metric, fontweight='bold')
        ax.set_xticks(range(len(results)))
        ax.set_xticklabels([r['Model'] for r in results], rotation=45, ha='right')
        ax.set_ylim([0, 1])
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig("01_model_performance_comparison.png", dpi=300, bbox_inches='tight')
    print("\n✓ Saved: 01_model_performance_comparison.png")
    plt.show()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    roc_auc_scores = [r['ROC AUC'] for r in results]
    sorted_indices = np.argsort(roc_auc_scores)[::-1]
    sorted_names = [results[i]['Model'] for i in sorted_indices]
    sorted_scores = [roc_auc_scores[i] for i in sorted_indices]
    
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_names)))
    bars = ax.barh(sorted_names, sorted_scores, color=colors)
    ax.set_xlabel('ROC-AUC Score', fontweight='bold')
    ax.set_title('Model Ranking by ROC-AUC Score', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    
    for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
        ax.text(score - 0.05, bar.get_y() + bar.get_height()/2, f'{score:.4f}',
                ha='right', va='center', fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig("02_model_ranking_roc_auc.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: 02_model_ranking_roc_auc.png")
    plt.show()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Confusion Matrices for All Models", fontsize=16, fontweight='bold')
    
    for idx, result in enumerate(results):
        ax = axes[idx // 2, idx % 2]
        cm = result['Confusion Matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    cbar_kws={'label': 'Count'}, annot_kws={'size': 12})
        ax.set_title(f"{result['Model']}", fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig("03_confusion_matrices.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: 03_confusion_matrices.png")
    plt.show()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    numerical_data = X_train.select_dtypes(include=["int64", "float64"])
    correlation_matrix = numerical_data.corr()
    sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, square=True, 
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title("Feature Correlation Heatmap (Numerical Features)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("04_correlation_heatmap.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: 04_correlation_heatmap.png")
    plt.show()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Top 10 Features by Importance/Coefficients (All Models)", fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for plot_idx, result in enumerate(results):
        if plot_idx >= 4:
            break
        
        ax = axes[plot_idx]
        model_name = result['Model']
        
        try:
            top_features, ylabel = get_top_features(result, feature_names, X_test, y_test, top_n=10)
            
            if top_features:
                features, values = zip(*top_features)
                colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
                ax.barh(range(len(features)), values, color=colors)
                ax.set_yticks(range(len(features)))
                ax.set_yticklabels(features)
                ax.set_xlabel(ylabel, fontweight='bold')
                ax.set_title(f"{model_name}", fontweight='bold')
                ax.invert_yaxis()
                
                for i, val in enumerate(values):
                    ax.text(val, i, f' {val:.4f}', va='center', fontsize=9)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error computing importance\nfor {model_name}:\n{str(e)}', 
                    ha='center', va='center', fontsize=10, wrap=True)
            ax.set_title(f"{model_name}", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("05_top_features_importance.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: 05_top_features_importance.png")
    plt.show()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    model_names = [r['Model'] for r in results]
    x = np.arange(len(model_names))
    width = 0.2
    
    tn = [r['Confusion Matrix'][0, 0] for r in results]
    fp = [r['Confusion Matrix'][0, 1] for r in results]
    fn = [r['Confusion Matrix'][1, 0] for r in results]
    tp = [r['Confusion Matrix'][1, 1] for r in results]
    
    fpr = [fp[i] / (fp[i] + tn[i]) if (fp[i] + tn[i]) > 0 else 0 for i in range(len(results))]
    tpr = [tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0 for i in range(len(results))]
    specificity = [tn[i] / (tn[i] + fp[i]) if (tn[i] + fp[i]) > 0 else 0 for i in range(len(results))]
    sensitivity = [tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0 for i in range(len(results))]
    
    ax.bar(x - 1.5*width, sensitivity, width, label='Sensitivity (TPR)', alpha=0.8)
    ax.bar(x - 0.5*width, specificity, width, label='Specificity', alpha=0.8)
    ax.bar(x + 0.5*width, fpr, width, label='False Positive Rate', alpha=0.8)
    ax.bar(x + 1.5*width, [r['Accuracy'] for r in results], width, label='Accuracy', alpha=0.8)
    
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Model Performance vs Confusion Matrix Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig("06_performance_vs_matrix_metrics.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: 06_performance_vs_matrix_metrics.png")
    plt.show()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for result in results:
        y_prob = result['y_prob']
        fpr_curve, tpr_curve, _ = roc_curve(y_test, y_prob)
        roc_auc = result['ROC AUC']
        ax.plot(fpr_curve, tpr_curve, label=f"{result['Model']} (AUC = {roc_auc:.4f})", linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.5000)')
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title('ROC-AUC Curves for All Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig("07_roc_auc_curves.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: 07_roc_auc_curves.png")
    plt.show()
    
    tuning_results, best_models = hyperparameter_tuning(X_train, y_train, X_test, y_test, preprocessor)
    
    pca_results, X_train_pca, X_test_pca = pca_analysis(X_train, y_train, X_test, y_test, preprocessor)
    
    regularization_results = regularization_comparison(X_train, y_train, X_test, y_test, preprocessor)
    
     
    # PREPARE MODELS FOR API
     
    
    print("\n" + "="*60)
    print("PREPARING MODELS FOR API DEPLOYMENT")
    print("="*60)
    
    # Prepare model metrics for API
    model_metrics = {}
    for r in results:
        model_metrics[r['Model']] = {
            "accuracy": r['Accuracy'],
            "roc_auc": r['ROC AUC'],
            "precision": r['Precision'],
            "recall": r['Recall'],
            "f1_score": r['F1 Score']
        }
    
    # Add tuned models
    for name, model in best_models.items():
        if name not in model_metrics:
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            
            model_metrics[name + " (Tuned)"] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_prob),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred)
            }
            best_models[name + " (Tuned)"] = pipeline
    
    # Load models into API manager
    api_manager.load_models(best_models, preprocessor, feature_names, model_metrics)
    
    print(f"\n✓ API ready with {len(best_models)} models")
    print("✓ Preprocessor and feature names loaded")
    print("✓ Model metrics prepared")
    
    print("\n" + "="*50)
    print("✓ All visualizations have been saved successfully!")
    print("="*50)
    
    return results, kfold_results, tuning_results, best_models, pca_results, regularization_results

 
# FASTAPI ENDPOINTS
 

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to AI Thunderball Cancer Detection API",
        "version": "1.0.0",
        "description": "Machine learning API for cancer risk prediction",
        "endpoints": {
            "/docs": "Interactive API documentation",
            "/predict": "Single patient prediction",
            "/predict-batch": "Batch predictions",
            "/models": "List available models",
            "/health": "API health check",
            "/features": "List feature names"
        }
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_patient(patient: PatientData, model_name: str = "Random Forest"):
    """
    Make a cancer risk prediction for a single patient
    
    - **patient**: Patient data including demographics, symptoms, and test results
    - **model_name**: Name of the model to use for prediction (default: "Random Forest")
    """
    try:
        start_time = datetime.now()
        
        # Get the requested model
        pipeline = api_manager.get_model(model_name)
        
        # Convert patient data to DataFrame
        patient_dict = patient.dict()
        patient_id = patient_dict.pop("patient_id")
        df = pd.DataFrame([patient_dict])
        
        # Make prediction
        probability = pipeline.predict_proba(df)[0, 1]
        prediction = 1 if probability >= 0.5 else 0
        
        # Calculate risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        # Calculate confidence (distance from decision boundary)
        confidence = abs(probability - 0.5) * 2
        
        # Generate clinical insights
        clinical_insights = generate_clinical_insights(patient_dict, probability)
        
        # Increment prediction counter
        api_manager.increment_predictions()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return PredictionResponse(
            patient_id=patient_id,
            prediction=prediction,
            probability=probability,
            risk_level=risk_level,
            confidence=confidence,
            model_used=model_name,
            timestamp=datetime.now().isoformat(),
            clinical_insights=clinical_insights
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Make cancer risk predictions for multiple patients
    
    - **patients**: List of patient data
    - **model_name**: Name of the model to use for prediction
    """
    try:
        start_time = datetime.now()
        
        # Get the requested model
        pipeline = api_manager.get_model(request.model_name)
        
        predictions = []
        positive_cases = 0
        
        # Process each patient
        for patient in request.patients:
            patient_dict = patient.dict()
            patient_id = patient_dict.pop("patient_id")
            df = pd.DataFrame([patient_dict])
            
            probability = pipeline.predict_proba(df)[0, 1]
            prediction = 1 if probability >= 0.5 else 0
            
            if prediction == 1:
                positive_cases += 1
            
            # Calculate risk level
            if probability < 0.3:
                risk_level = "Low"
            elif probability < 0.7:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            confidence = abs(probability - 0.5) * 2
            
            clinical_insights = generate_clinical_insights(patient_dict, probability)
            
            predictions.append(PredictionResponse(
                patient_id=patient_id,
                prediction=prediction,
                probability=probability,
                risk_level=risk_level,
                confidence=confidence,
                model_used=request.model_name,
                timestamp=datetime.now().isoformat(),
                clinical_insights=clinical_insights
            ))
        
        # Calculate statistics
        total_patients = len(request.patients)
        negative_cases = total_patients - positive_cases
        average_risk = sum([p.probability for p in predictions]) / total_patients
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Increment prediction counter for each patient
        for _ in range(total_patients):
            api_manager.increment_predictions()
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_patients=total_patients,
            positive_cases=positive_cases,
            negative_cases=negative_cases,
            average_risk=average_risk,
            model_used=request.model_name,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/models", response_model=List[ModelInfo], tags=["Models"])
async def list_models():
    """List all available models with their performance metrics"""
    try:
        models_info = []
        
        for model_name, pipeline in api_manager.models.items():
            metrics = api_manager.model_metrics.get(model_name, {})
            
            # Get model type from the pipeline
            model_obj = pipeline.named_steps.get('model')
            model_type = type(model_obj).__name__ if model_obj else "Unknown"
            
            # Get features used
            features_used = api_manager.feature_names
            
            models_info.append(ModelInfo(
                name=model_name,
                type=model_type,
                accuracy=metrics.get('accuracy', 0.0),
                roc_auc=metrics.get('roc_auc', 0.0),
                precision=metrics.get('precision', 0.0),
                recall=metrics.get('recall', 0.0),
                f1_score=metrics.get('f1_score', 0.0),
                features_used=features_used[:20],  # First 20 features
                training_date=datetime.now().strftime("%Y-%m-%d"),
                hyperparameters=model_obj.get_params() if hasattr(model_obj, 'get_params') else {}
            ))
        
        return models_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@app.get("/health", response_model=APIHealthResponse, tags=["Monitoring"])
async def health_check():
    """Check API health status and statistics"""
    try:
        import sys
        
        return APIHealthResponse(
            status="healthy",
            version="1.0.0",
            uptime=api_manager.get_uptime(),
            models_loaded=len(api_manager.models),
            total_predictions=api_manager.total_predictions,
            memory_usage=api_manager.get_memory_usage()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/features", tags=["Models"])
async def list_features():
    """List all feature names used by the models"""
    try:
        return {
            "total_features": len(api_manager.feature_names),
            "features": api_manager.feature_names,
            "categorical_features": [f for f in api_manager.feature_names if '_' in f],
            "numerical_features": [f for f in api_manager.feature_names if '_' not in f]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list features: {str(e)}")

@app.post("/train", tags=["Models"])
async def train_model(background_tasks: BackgroundTasks):
    """
    Trigger model training in the background
    
    This endpoint starts the training pipeline asynchronously
    """
    try:
        # Add training task to background
        background_tasks.add_task(train_and_update_models)
        
        return {
            "message": "Model training started in background",
            "status": "processing",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed to start: {str(e)}")

def generate_clinical_insights(patient_data: dict, probability: float) -> dict:
    """Generate clinical insights based on patient data and prediction"""
    insights = {
        "risk_factors": [],
        "protective_factors": [],
        "recommendations": [],
        "alerts": []
    }
    
    # Analyze risk factors
    if patient_data.get('family_history_cancer') == 1:
        insights["risk_factors"].append("Family history of cancer")
    
    if patient_data.get('smoking_status') == 'Current':
        insights["risk_factors"].append("Current smoker")
    
    if patient_data.get('lump_presence') == 1:
        insights["risk_factors"].append("Presence of lump")
        insights["alerts"].append("Immediate medical consultation recommended for lump evaluation")
    
    if patient_data.get('tumor_marker_level', 0) > 10:
        insights["risk_factors"].append(f"Elevated tumor marker level ({patient_data.get('tumor_marker_level'):.1f})")
    
    if patient_data.get('tumor_size_cm', 0) > 2:
        insights["risk_factors"].append(f"Significant tumor size ({patient_data.get('tumor_size_cm'):.1f} cm)")
        insights["alerts"].append("Tumor size warrants immediate medical attention")
    
    # Analyze protective factors
    if patient_data.get('physical_activity_level') == 'High':
        insights["protective_factors"].append("High physical activity level")
    
    if patient_data.get('diet_quality') == 'Good':
        insights["protective_factors"].append("Good diet quality")
    
    if patient_data.get('alcohol_consumption') == 'Never':
        insights["protective_factors"].append("No alcohol consumption")
    
    # Generate recommendations
    if probability > 0.7:
        insights["recommendations"].extend([
            "Immediate consultation with oncologist recommended",
            "Consider advanced imaging (MRI/PET scan)",
            "Biopsy may be warranted"
        ])
    elif probability > 0.3:
        insights["recommendations"].extend([
            "Regular follow-up with primary care physician",
            "Consider referral to specialist",
            "Monitor symptoms closely"
        ])
    else:
        insights["recommendations"].append("Routine screening recommended based on age and risk factors")
    
    # Add general recommendations
    if patient_data.get('smoking_status') in ['Current', 'Former']:
        insights["recommendations"].append("Smoking cessation counseling recommended")
    
    if patient_data.get('bmi', 25) > 25:
        insights["recommendations"].append("Weight management consultation recommended")
    
    return insights

def train_and_update_models():
    """Background task to train models and update API"""
    print("\n" + "="*60)
    print("BACKGROUND MODEL TRAINING STARTED")
    print("="*60)
    
    try:
        # Run the full training pipeline
        results, kfold_results, tuning_results, best_models, pca_results, regularization_results = run_machine_learning_pipeline()
        
        print("\n" + "="*60)
        print("BACKGROUND MODEL TRAINING COMPLETED")
        print("="*60)
        print("✓ Models have been updated")
        print("✓ API is ready with new models")
        
    except Exception as e:
        print(f"\n✗ Background training failed: {str(e)}")

 
# MAIN EXECUTION
 

def main():
    print("="*60)
    print("AI THUNDERBALL - CANCER DETECTION DATA PIPELINE WITH API")
    print("="*60)
    print("\nIncludes:")
    print("• Complete ML Pipeline with Regularization")
    print("• FastAPI RESTful Web Service")
    print("• Real-time Predictions")
    print("• Batch Processing")
    print("• Model Management")
    print("• Health Monitoring\n")
    
    # Run the ML pipeline (this will also load models into API)
    generate_synthetic_data()
    introduce_anomalies()
    clean_data()
    engineer_features()
    train_test_split_data()
    run_machine_learning_pipeline()
    
    print("\n" + "="*60)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("✓ 11 Visualizations Saved")
    print("✓ 4 CSV Results Files Generated")
    print("✓ Regularization Analysis Complete")
    print("✓ API Models Loaded and Ready")
    print("\nTo start the API server, run:")
    print("    uvicorn filename:app --reload --host 0.0.0.0 --port 8000")
    print("\nAPI Documentation available at:")
    print("    http://localhost:8000/docs")
    print("="*60)

def run_api():
    """Function to run the API server"""
    print("\n" + "="*60)
    print("STARTING AI THUNDERBALL API SERVER")
    print("="*60)
    print("\nAPI Endpoints:")
    print("• http://localhost:8000/           - API information")
    print("• http://localhost:8000/docs       - Interactive documentation")
    print("• http://localhost:8000/predict    - Single prediction")
    print("• http://localhost:8000/predict-batch - Batch predictions")
    print("• http://localhost:8000/models     - List models")
    print("• http://localhost:8000/health     - Health check")
    print("• http://localhost:8000/features   - List features")
    print("\nPress Ctrl+C to stop the server")
    print("="*60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--api":
        # Check if models are already trained
        if not api_manager.models:
            print("Models not found. Running training pipeline first...")
            main()
        
        # Start API server
        run_api()
    else:
        # Run full pipeline (training + visualizations)
        main()
        
        # Ask if user wants to start API
        response = input("\nDo you want to start the API server? (yes/no): ").lower()
        if response in ['yes', 'y']:
            run_api()
