import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from ctgan import CTGAN
from sklearn.model_selection import (
    train_test_split, KFold, cross_val_score,
    GridSearchCV, RandomizedSearchCV, StratifiedKFold
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    auc, precision_recall_curve,
    classification_report, average_precision_score
)
from sklearn.model_selection import validation_curve
import joblib
import json
from contextlib import asynccontextmanager
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# FastAPI imports for API endpoints
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from datetime import datetime


# ─────────────────────────────────────────────
# DATA PIPELINE FUNCTIONS
# ─────────────────────────────────────────────

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

    if 'patient_id' in df.columns:
        df['patient_id'] = range(1000, 1000 + len(df))
        
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

    df_engineered['platelet_lymphocyte_ratio'] = df_engineered['platelet_count'] / (
        df_engineered['wbc_count'] * 0.3 + 1e-10)

    def get_smoking_score(status):
        return {'Never': 0, 'Former': 1, 'Current': 2}.get(status, 0)

    def get_alcohol_score(consumption):
        return {'Never': 0, 'Moderate': 1, 'High': 2}.get(consumption, 0)

    def get_activity_score(activity):
        return {'High': 0, 'Medium': 1, 'Low': 2}.get(activity, 0)

    def get_diet_score(diet):
        return {'Good': 0, 'Average': 1, 'Poor': 2}.get(diet, 0)

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

        if row['unexplained_weight_loss'] == 1: score += 3
        if row['lump_presence'] == 1:           score += 4
        if row['persistent_fatigue'] == 1:      score += 2
        if row['chronic_pain'] == 1:            score += 2
        if row['abnormal_bleeding'] == 1:       score += 3
        if row['persistent_cough'] == 1:        score += 2

        score += row['diabetes'] * 1 + row['hypertension'] * 1 + row['cardiac_disease'] * 2

        if row['lifestyle_risk_score'] > 6:   score += 3
        elif row['lifestyle_risk_score'] > 4: score += 2
        elif row['lifestyle_risk_score'] > 2: score += 1

        if row['tumor_marker_level'] > 50:   score += 3
        elif row['tumor_marker_level'] > 25: score += 2
        elif row['tumor_marker_level'] > 10: score += 1

        if row['tumor_size_cm'] > 5:   score += 4
        elif row['tumor_size_cm'] > 2: score += 2
        elif row['tumor_size_cm'] > 0.5: score += 1

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
    print(f"\n Saved engineered dataset to: '{output_file}'")
    print(f" Dataset shape: {df_engineered.shape}")

    return df_engineered


def train_test_split_data(input_file="AI_Thunderball_FeatureEngineeredDataset.csv"):
    print("Splitting data into train and test sets...")

    df = pd.read_csv(input_file)
    y = df["cancer"]
    X = df.drop(columns=["cancer", "patient_id"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    X_train.to_csv("AI_Thunderball_X_train.csv", index=False)
    X_test.to_csv("AI_Thunderball_X_test.csv", index=False)
    y_train.to_csv("AI_Thunderball_y_train.csv", index=False)
    y_test.to_csv("AI_Thunderball_y_test.csv", index=False)

    print(" Train-Test split completed")
    print(f" Training samples: {X_train.shape[0]}")
    print(f" Testing samples : {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# MODEL EVALUATION HELPERS
# ─────────────────────────────────────────────

def evaluate_model(name, model, X_train, y_train, X_test, y_test, preprocessor):
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    return {
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


def evaluate_model_kfold(name, model, X, y, preprocessor, n_folds=5):
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    cv_accuracy  = cross_val_score(pipeline, X, y, cv=kfold, scoring='accuracy')
    cv_precision = cross_val_score(pipeline, X, y, cv=kfold, scoring='precision')
    cv_recall    = cross_val_score(pipeline, X, y, cv=kfold, scoring='recall')
    cv_f1        = cross_val_score(pipeline, X, y, cv=kfold, scoring='f1')
    cv_roc_auc   = cross_val_score(pipeline, X, y, cv=kfold, scoring='roc_auc')

    return {
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
    pipeline   = result['pipeline']
    model      = pipeline.named_steps["model"]
    model_name = result['Model']

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices     = np.argsort(importances)[-top_n:][::-1]
        return [(feature_names[i], importances[i]) for i in indices], 'Importance'

    elif hasattr(model, 'coef_'):
        coefficients = np.abs(model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_)
        indices      = np.argsort(coefficients)[-top_n:][::-1]
        return [(feature_names[i], coefficients[i]) for i in indices], 'Absolute Coefficient'

    else:
        print(f"Computing permutation importance for {model_name}...")
        perm_importance = permutation_importance(
            pipeline, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
        )
        importances = perm_importance.importances_mean
        indices     = np.argsort(importances)[-top_n:][::-1]
        return [(feature_names[i], importances[i]) for i in indices], 'Permutation Importance'


# ─────────────────────────────────────────────
# PCA ANALYSIS
# ─────────────────────────────────────────────

def pca_analysis(X_train, y_train, X_test, y_test, preprocessor, variance_threshold=0.95):
    print("\n" + "="*60)
    print("PCA - DIMENSIONALITY REDUCTION ANALYSIS")
    print("="*60)

    print("\n[1/6] Preprocessing data for PCA...")

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed  = preprocessor.transform(X_test)

    if hasattr(X_train_processed, 'toarray'):
        X_train_processed = X_train_processed.toarray()
        X_test_processed  = X_test_processed.toarray()

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
    print(f"     Dimensionality reduction: {n_features_original} → {n_components_optimal} "
          f"({(1-n_components_optimal/n_features_original)*100:.1f}% reduction)")

    print("\n[3/6] Applying PCA with optimal components...")

    pca_optimal = PCA(n_components=n_components_optimal)
    X_train_pca = pca_optimal.fit_transform(X_train_processed)
    X_test_pca  = pca_optimal.transform(X_test_processed)

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
    ax2.fill_between(range(1, len(cumulative_variance) + 1), cumulative_variance * 100, alpha=0.3, color='#2ecc71')
    ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100, 'g-', linewidth=2)
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
    print("      Saved: 09_pca_analysis.png")
    plt.show()

    print("\n[5/6] Comparing model performance with vs without PCA...")

    pca_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    comparison_results = []

    for name, model in pca_models.items():
        model_no_pca = model.__class__(**model.get_params())
        model_no_pca.fit(X_train_processed, y_train)
        pred_no_pca = model_no_pca.predict(X_test_processed)
        prob_no_pca = model_no_pca.predict_proba(X_test_processed)[:, 1]
        acc_no_pca  = accuracy_score(y_test, pred_no_pca)
        roc_no_pca  = roc_auc_score(y_test, prob_no_pca)

        model_pca = model.__class__(**model.get_params())
        model_pca.fit(X_train_pca, y_train)
        pred_pca = model_pca.predict(X_test_pca)
        prob_pca = model_pca.predict_proba(X_test_pca)[:, 1]
        acc_pca  = accuracy_score(y_test, pred_pca)
        roc_pca  = roc_auc_score(y_test, prob_pca)

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

    pca_model_names = [r['Model'] for r in comparison_results]
    x     = np.arange(len(pca_model_names))
    width = 0.35

    ax1 = axes[0]
    acc_no_pca_vals = [r['Accuracy_NoPCA'] for r in comparison_results]
    acc_pca_vals    = [r['Accuracy_PCA']   for r in comparison_results]
    bars1 = ax1.bar(x - width/2, acc_no_pca_vals, width, label='Without PCA', alpha=0.8, color='#3498db')
    bars2 = ax1.bar(x + width/2, acc_pca_vals,    width, label='With PCA',    alpha=0.8, color='#e74c3c')
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('Accuracy Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(pca_model_names, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim([0.5, 1.0])
    for bar in list(bars1) + list(bars2):     
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                 f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    ax2 = axes[1]
    roc_no_pca_vals = [r['ROC_AUC_NoPCA'] for r in comparison_results]
    roc_pca_vals    = [r['ROC_AUC_PCA']   for r in comparison_results]
    bars3 = ax2.bar(x - width/2, roc_no_pca_vals, width, label='Without PCA', alpha=0.8, color='#3498db')
    bars4 = ax2.bar(x + width/2, roc_pca_vals,    width, label='With PCA',    alpha=0.8, color='#e74c3c')
    ax2.set_ylabel('ROC-AUC', fontweight='bold')
    ax2.set_title('ROC-AUC Comparison', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(pca_model_names, rotation=45, ha='right')
    ax2.legend()
    ax2.set_ylim([0.5, 1.0])
    for bar in list(bars3) + list(bars4):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                 f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig("10_pca_model_comparison.png", dpi=300, bbox_inches='tight')
    print("      Saved: 10_pca_model_comparison.png")
    plt.show()

    results_df = pd.DataFrame(comparison_results)
    results_df.to_csv("pca_comparison_results.csv", index=False)
    print("      Saved: pca_comparison_results.csv")

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


# ─────────────────────────────────────────────
# HYPERPARAMETER TUNING
# ─────────────────────────────────────────────

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

    base_model_configs = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tuning_results = []
    best_models    = {}

    for name, model in base_model_configs.items():
        print(f"\n{'─'*50}")
        print(f"Tuning: {name}")
        print(f"{'─'*50}")

        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

        print("  [1/3] Evaluating baseline model...")
        pipeline.fit(X_train, y_train)
        baseline_pred  = pipeline.predict(X_test)
        baseline_prob  = pipeline.predict_proba(X_test)[:, 1]
        baseline_roc   = roc_auc_score(y_test, baseline_prob)
        baseline_acc   = accuracy_score(y_test, baseline_pred)
        baseline_f1    = f1_score(y_test, baseline_pred)
        print(f"       Baseline ROC-AUC: {baseline_roc:.4f}")
        print(f"       Baseline Accuracy: {baseline_acc:.4f}")

        param_grid      = param_grids[name]
        n_combinations  = 1
        for values in param_grid.values():
            n_combinations *= len(values)

        print(f"  [2/3] Searching {n_combinations} parameter combinations...")

        if n_combinations > 100:
            search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=50,
                                        cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0, random_state=42)
        else:
            search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv,
                                  scoring='roc_auc', n_jobs=-1, verbose=0)

        search.fit(X_train, y_train)

        print("  [3/3] Evaluating tuned model...")
        tuned_pred = search.predict(X_test)
        tuned_prob = search.predict_proba(X_test)[:, 1]
        tuned_roc  = roc_auc_score(y_test, tuned_prob)
        tuned_acc  = accuracy_score(y_test, tuned_pred)
        tuned_f1   = f1_score(y_test, tuned_pred)

        improvement_roc = tuned_roc - baseline_roc
        improvement_acc = tuned_acc - baseline_acc
        improvement_f1  = tuned_f1  - baseline_f1

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
    print(f"\n🏆 Best Model: {best_result['Model']} with ROC-AUC = {best_result['Tuned_ROC_AUC']:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Hyperparameter Tuning Results: Baseline vs Tuned", fontsize=14, fontweight='bold')

    model_names = [r['Model'] for r in tuning_results]
    x     = np.arange(len(model_names))
    width = 0.35

    ax1 = axes[0]
    baseline_rocs = [r['Baseline_ROC_AUC'] for r in tuning_results]
    tuned_rocs    = [r['Tuned_ROC_AUC']    for r in tuning_results]
    bars1 = ax1.bar(x - width/2, baseline_rocs, width, label='Baseline', alpha=0.8, color='#3498db')
    bars2 = ax1.bar(x + width/2, tuned_rocs,    width, label='Tuned',    alpha=0.8, color='#2ecc71')
    ax1.set_ylabel('ROC-AUC Score', fontweight='bold')
    ax1.set_title('ROC-AUC: Baseline vs Tuned', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim([0.4, 1.0])
    for bar in list(bars1) + list(bars2):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                 f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    ax2 = axes[1]
    baseline_accs = [r['Baseline_Accuracy'] for r in tuning_results]
    tuned_accs    = [r['Tuned_Accuracy']    for r in tuning_results]
    bars3 = ax2.bar(x - width/2, baseline_accs, width, label='Baseline', alpha=0.8, color='#3498db')
    bars4 = ax2.bar(x + width/2, tuned_accs,    width, label='Tuned',    alpha=0.8, color='#2ecc71')
    ax2.set_ylabel('Accuracy', fontweight='bold')
    ax2.set_title('Accuracy: Baseline vs Tuned', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.legend()
    ax2.set_ylim([0.4, 1.0])
    for bar in list(bars3) + list(bars4):  
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                 f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    ax3 = axes[2]
    improvements = [r['ROC_AUC_Improvement'] for r in tuning_results]
    colors       = ['#2ecc71' if imp >= 0 else '#e74c3c' for imp in improvements]
    bars5        = ax3.bar(model_names, improvements, color=colors, alpha=0.8)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_ylabel('ROC-AUC Improvement', fontweight='bold')
    ax3.set_title('Performance Improvement After Tuning', fontweight='bold')
    ax3.set_xticklabels(model_names, rotation=45, ha='right')
    for bar, imp in zip(bars5, improvements):
        imp_label = f"+{imp:.4f}" if imp >= 0 else f"{imp:.4f}"
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                 imp_label, ha='center', va='bottom' if imp >= 0 else 'top', fontsize=9)

    plt.tight_layout()
    plt.savefig("08_hyperparameter_tuning_comparison.png", dpi=300, bbox_inches='tight')
    print("\n Saved: 08_hyperparameter_tuning_comparison.png")
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
    print(" Saved: hyperparameter_tuning_results.csv")

    return tuning_results, best_models


# ─────────────────────────────────────────────
# REGULARIZATION ANALYSIS
# ─────────────────────────────────────────────

def regularization_analysis(X_train, y_train, X_test, y_test, preprocessor):
    """
    Perform regularization analysis to show the effects of L1, L2, and
    ElasticNet regularization on model performance and overfitting prevention.
    """
    print("\n" + "="*60)
    print("REGULARIZATION ANALYSIS")
    print("="*60)
    print("\nRegularization prevents overfitting by adding constraints to models.")
    print("L1 (Lasso): Encourages sparse solutions (feature selection)")
    print("L2 (Ridge): Shrinks coefficients evenly")
    print("ElasticNet: Combines L1 and L2")

    print("\n[1/6] Logistic Regression with different regularization types...")

    logistic_models = {
        "No Regularization": LogisticRegression(penalty=None, max_iter=1000, solver='saga'),
        "L1 Regularization": LogisticRegression(penalty='l1', C=1.0, max_iter=1000, solver='liblinear'),
        "L2 Regularization": LogisticRegression(penalty='l2', C=1.0, max_iter=1000, solver='lbfgs'),
        "ElasticNet": LogisticRegression(penalty='elasticnet', l1_ratio=0.5, C=1.0, max_iter=1000, solver='saga')
    }

    logistic_results = []

    for name, model in logistic_models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)

        y_pred      = pipeline.predict(X_test)
        y_prob      = pipeline.predict_proba(X_test)[:, 1]
        train_score = pipeline.score(X_train, y_train)
        test_score  = pipeline.score(X_test, y_test)

        coef = pipeline.named_steps['model'].coef_
        if coef is not None:
            if len(coef.shape) > 1:
                coef = coef[0]
            non_zero_coef = int(np.sum(np.abs(coef) > 1e-5))
        else:
            non_zero_coef = None

        logistic_results.append({
            "Regularization": name,
            "Train_Accuracy": train_score,
            "Test_Accuracy": test_score,
            "Accuracy_Diff": train_score - test_score,
            "ROC_AUC": roc_auc_score(y_test, y_prob),
            "Non_Zero_Coefficients": non_zero_coef,
            "model": pipeline
        })

        print(f"  {name}:")
        print(f"    Train Accuracy: {train_score:.4f}")
        print(f"    Test Accuracy: {test_score:.4f}")
        print(f"    Overfitting (Train-Test): {(train_score-test_score):.4f}")
        if non_zero_coef is not None:
            print(f"    Non-zero coefficients: {non_zero_coef}")

    print("\n[2/6] Validation curve for regularization strength...")

    val_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(penalty='l2', max_iter=1000, solver='lbfgs'))
    ])

    param_range  = np.logspace(-3, 3, 10)
    train_scores, test_scores = validation_curve(
        val_pipeline, X_train, y_train,
        param_name="model__C",
        param_range=param_range,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std  = np.std(train_scores,  axis=1)
    test_mean  = np.mean(test_scores,  axis=1)
    test_std   = np.std(test_scores,   axis=1)

    print("\n[3/6] SVM with different regularization strengths...")

    svm_c_values = [0.001, 0.01, 0.1, 1, 10, 100]
    svm_results  = []

    for C in svm_c_values:
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", SVC(kernel='rbf', C=C, probability=True, random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        train_score = pipeline.score(X_train, y_train)
        test_score  = pipeline.score(X_test, y_test)
        svm_results.append({
            "C": C,
            "Train_Accuracy": train_score,
            "Test_Accuracy": test_score,
            "Accuracy_Diff": train_score - test_score
        })

    print(f"  C values analyzed: {svm_c_values}")

    print("\n[4/6] Neural Network with L2 regularization...")

    nn_alphas  = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    nn_results = []

    for alpha in nn_alphas:
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", MLPClassifier(hidden_layer_sizes=(50, 25), alpha=alpha,
                                    max_iter=1000, random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        train_score = pipeline.score(X_train, y_train)
        test_score  = pipeline.score(X_test, y_test)
        nn_results.append({
            "Alpha": alpha,
            "Train_Accuracy": train_score,
            "Test_Accuracy": test_score,
            "Accuracy_Diff": train_score - test_score
        })

    print(f"  Alpha values analyzed: {nn_alphas}")

    print("\n[5/6] Creating regularization visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Regularization Analysis: Preventing Overfitting", fontsize=16, fontweight='bold')

    ax1 = axes[0, 0]
    reg_types = [r["Regularization"] for r in logistic_results]
    train_acc = [r["Train_Accuracy"] for r in logistic_results]
    test_acc  = [r["Test_Accuracy"]  for r in logistic_results]
    x         = np.arange(len(reg_types))
    width     = 0.35
    bars1 = ax1.bar(x - width/2, train_acc, width, label='Training Accuracy', alpha=0.8, color='#3498db')
    bars2 = ax1.bar(x + width/2, test_acc,  width, label='Testing Accuracy',  alpha=0.8, color='#2ecc71')
    ax1.set_xlabel('Regularization Type', fontweight='bold')
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('Logistic Regression: Effect of Regularization Type', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(reg_types, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim([0.5, 1.0])
    ax1.grid(True, alpha=0.3)
    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    ax2 = axes[0, 1]
    ax2.semilogx(param_range, train_mean, label="Training Accuracy", color="darkorange", lw=2)
    ax2.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2, color="darkorange")
    ax2.semilogx(param_range, test_mean, label="Cross-validation Accuracy", color="navy", lw=2)
    ax2.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.2, color="navy")
    optimal_idx = np.argmax(test_mean)
    optimal_C   = param_range[optimal_idx]
    ax2.axvline(x=optimal_C, color='red', linestyle='--', linewidth=2,
                label=f'Optimal C = {optimal_C:.3f}')
    ax2.set_xlabel("Regularization Strength (C)", fontweight='bold')
    ax2.set_ylabel("Accuracy", fontweight='bold')
    ax2.set_title("Validation Curve: Effect of Regularization Strength", fontweight='bold')
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    svm_c    = [r["C"]              for r in svm_results]
    svm_train = [r["Train_Accuracy"] for r in svm_results]
    svm_test  = [r["Test_Accuracy"]  for r in svm_results]
    ax3.semilogx(svm_c, svm_train, 'o-', label='Training Accuracy', linewidth=2, markersize=8, color='#3498db')
    ax3.semilogx(svm_c, svm_test,  's-', label='Testing Accuracy',  linewidth=2, markersize=8, color='#e74c3c')
    optimal_svm_idx = np.argmax(svm_test)
    optimal_svm_C   = svm_c[optimal_svm_idx]
    ax3.axvline(x=optimal_svm_C, color='green', linestyle='--', linewidth=2,
                label=f'Optimal C = {optimal_svm_C}')
    ax3.set_xlabel("Regularization Strength (C)", fontweight='bold')
    ax3.set_ylabel("Accuracy", fontweight='bold')
    ax3.set_title("SVM: Effect of Regularization Strength", fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    nn_alpha = [r["Alpha"]          for r in nn_results]
    nn_train = [r["Train_Accuracy"] for r in nn_results]
    nn_test  = [r["Test_Accuracy"]  for r in nn_results]
    ax4.semilogx(nn_alpha, nn_train, 'o-', label='Training Accuracy', linewidth=2, markersize=8, color='#3498db')
    ax4.semilogx(nn_alpha, nn_test,  's-', label='Testing Accuracy',  linewidth=2, markersize=8, color='#e74c3c')
    optimal_nn_idx   = np.argmax(nn_test)
    optimal_nn_alpha = nn_alpha[optimal_nn_idx]
    ax4.axvline(x=optimal_nn_alpha, color='green', linestyle='--', linewidth=2,
                label=f'Optimal α = {optimal_nn_alpha}')
    ax4.set_xlabel("Regularization Strength (Alpha)", fontweight='bold')
    ax4.set_ylabel("Accuracy", fontweight='bold')
    ax4.set_title("Neural Network: Effect of L2 Regularization", fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("11_regularization_analysis.png", dpi=300, bbox_inches='tight')
    print(" Saved: 11_regularization_analysis.png")
    plt.show()

    print("\n[6/6] Coefficient analysis for L1 vs L2 regularization...")

    feature_names = get_feature_names(preprocessor)

    pipeline_l1 = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(penalty='l1', C=1.0, max_iter=1000, solver='liblinear'))
    ])
    pipeline_l2 = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(penalty='l2', C=1.0, max_iter=1000, solver='lbfgs'))
    ])

    pipeline_l1.fit(X_train, y_train)
    pipeline_l2.fit(X_train, y_train)

    coef_l1 = pipeline_l1.named_steps['model'].coef_[0]
    coef_l2 = pipeline_l2.named_steps['model'].coef_[0]

    top_n     = 15
    l1_indices = np.argsort(np.abs(coef_l1))[-top_n:][::-1]
    l2_indices = np.argsort(np.abs(coef_l2))[-top_n:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Feature Coefficients: L1 vs L2 Regularization", fontsize=16, fontweight='bold')

    ax1 = axes[0]
    l1_features = [feature_names[i] for i in l1_indices]
    l1_values   = coef_l1[l1_indices]
    colors1     = ['#e74c3c' if val < 0 else '#2ecc71' for val in l1_values]
    bars1 = ax1.barh(range(len(l1_features)), l1_values, color=colors1, alpha=0.8)
    ax1.set_yticks(range(len(l1_features)))
    ax1.set_yticklabels(l1_features)
    ax1.set_xlabel('Coefficient Value', fontweight='bold')
    ax1.set_title('L1 Regularization (Sparse Solution)', fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax1.invert_yaxis()
    for i, (bar, val) in enumerate(zip(bars1, l1_values)):
        ax1.text(val, i, f' {val:.3f}', va='center', fontsize=9, color='black', fontweight='bold')

    ax2 = axes[1]
    l2_features = [feature_names[i] for i in l2_indices]
    l2_values   = coef_l2[l2_indices]
    colors2     = ['#e74c3c' if val < 0 else '#2ecc71' for val in l2_values]
    bars2 = ax2.barh(range(len(l2_features)), l2_values, color=colors2, alpha=0.8)
    ax2.set_yticks(range(len(l2_features)))
    ax2.set_yticklabels(l2_features)
    ax2.set_xlabel('Coefficient Value', fontweight='bold')
    ax2.set_title('L2 Regularization (Dense Solution)', fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.invert_yaxis()
    for i, (bar, val) in enumerate(zip(bars2, l2_values)):
        ax2.text(val, i, f' {val:.3f}', va='center', fontsize=9, color='black', fontweight='bold')

    plt.tight_layout()
    plt.savefig("12_regularization_coefficients.png", dpi=300, bbox_inches='tight')
    print(" Saved: 12_regularization_coefficients.png")
    plt.show()

    print("\n" + "="*60)
    print("REGULARIZATION ANALYSIS SUMMARY")
    print("="*60)
    print(f"\nLogistic Regression Results:")
    print(f"{'Regularization':<20} {'Train Acc':<12} {'Test Acc':<12} {'Overfitting':<12} {'Non-Zero Coef':<15}")
    print("-" * 75)
    for r in logistic_results:
        print(f"{r['Regularization']:<20} {r['Train_Accuracy']:<12.4f} {r['Test_Accuracy']:<12.4f} "
              f"{r['Accuracy_Diff']:<12.4f} {str(r['Non_Zero_Coefficients']):<15}")

    print(f"\nOptimal Parameters:")
    print(f"  Logistic Regression C: {optimal_C:.3f}")
    print(f"  SVM C: {optimal_svm_C}")
    print(f"  Neural Network Alpha: {optimal_nn_alpha}")

    print(f"\nFeature Selection (L1 vs L2):")
    print(f"  L1 regularization kept {np.sum(np.abs(coef_l1) > 1e-5)} non-zero coefficients")
    print(f"  L2 regularization kept {len(coef_l2)} non-zero coefficients")
    print(f"  L1 sparsity: {(1 - np.sum(np.abs(coef_l1) > 1e-5) / len(coef_l1)) * 100:.1f}%")

    reg_results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'model'} for r in logistic_results])
    reg_results_df.to_csv("regularization_results.csv", index=False)
    print("\n Saved: regularization_results.csv")

    return {
        "logistic_results": logistic_results,
        "svm_results": svm_results,
        "nn_results": nn_results,
        "optimal_C": optimal_C,
        "optimal_svm_C": optimal_svm_C,
        "optimal_nn_alpha": optimal_nn_alpha,
        "coef_l1": coef_l1,
        "coef_l2": coef_l2,
        "feature_names": feature_names
    }


# ─────────────────────────────────────────────
# MACHINE LEARNING PIPELINE
# ─────────────────────────────────────────────

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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("scaler", StandardScaler())]), numerical_cols),
            ("cat", Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols)
        ]
    )

    model_configs = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    results = []
    for name, model in model_configs.items():
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

    for name, model in model_configs.items():
        print(f"\nEvaluating {name} with 5-Fold CV...")
        kfold_res = evaluate_model_kfold(name, clone(model), X_combined, y_combined, preprocessor, n_folds=5)
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

    # --- Plot 1: Model Performance Metrics ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Performance Metrics Comparison", fontsize=16, fontweight='bold')

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    for idx, metric in enumerate(metrics):
        ax     = axes[idx // 2, idx % 2]
        values = [r[metric] for r in results]
        colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
        bars   = ax.bar(range(len(results)), values, color=colors)
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
    print("\n Saved: 01_model_performance_comparison.png")
    plt.show()

    # --- Plot 2: ROC-AUC ranking ---
    fig, ax = plt.subplots(figsize=(10, 6))
    roc_auc_scores  = [r['ROC AUC'] for r in results]
    sorted_indices  = np.argsort(roc_auc_scores)[::-1]
    sorted_names    = [results[i]['Model'] for i in sorted_indices]
    sorted_scores   = [roc_auc_scores[i]   for i in sorted_indices]
    colors          = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_names)))
    bars = ax.barh(sorted_names, sorted_scores, color=colors)
    ax.set_xlabel('ROC-AUC Score', fontweight='bold')
    ax.set_title('Model Ranking by ROC-AUC Score', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
        ax.text(score - 0.05, bar.get_y() + bar.get_height()/2,
                f'{score:.4f}', ha='right', va='center', fontweight='bold', color='white')
    plt.tight_layout()
    plt.savefig("02_model_ranking_roc_auc.png", dpi=300, bbox_inches='tight')
    print(" Saved: 02_model_ranking_roc_auc.png")
    plt.show()

    # --- Plot 3: Confusion Matrices ---
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
    print(" Saved: 03_confusion_matrices.png")
    plt.show()

    # --- Plot 4: Correlation Heatmap ---
    fig, ax = plt.subplots(figsize=(12, 10))
    numerical_data   = X_train.select_dtypes(include=["int64", "float64"])
    correlation_matrix = numerical_data.corr()
    sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, square=True,
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title("Feature Correlation Heatmap (Numerical Features)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("04_correlation_heatmap.png", dpi=300, bbox_inches='tight')
    print(" Saved: 04_correlation_heatmap.png")
    plt.show()

    # --- Plot 5: Feature Importances ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Top 10 Features by Importance/Coefficients (All Models)", fontsize=16, fontweight='bold')
    axes = axes.flatten()

    for plot_idx, result in enumerate(results):
        if plot_idx >= 4:
            break
        ax = axes[plot_idx]
        try:
            top_features, ylabel = get_top_features(result, feature_names, X_test, y_test, top_n=10)
            if top_features:
                features, values = zip(*top_features)
                colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
                ax.barh(range(len(features)), values, color=colors)
                ax.set_yticks(range(len(features)))
                ax.set_yticklabels(features)
                ax.set_xlabel(ylabel, fontweight='bold')
                ax.set_title(f"{result['Model']}", fontweight='bold')
                ax.invert_yaxis()
                for i, val in enumerate(values):
                    ax.text(val, i, f' {val:.4f}', va='center', fontsize=9)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error computing importance\nfor {result["Model"]}:\n{str(e)}',
                    ha='center', va='center', fontsize=10, wrap=True)
            ax.set_title(f"{result['Model']}", fontweight='bold')

    plt.tight_layout()
    plt.savefig("05_top_features_importance.png", dpi=300, bbox_inches='tight')
    print(" Saved: 05_top_features_importance.png")
    plt.show()

    # --- Plot 6: Performance vs Confusion Matrix Metrics ---
    fig, ax = plt.subplots(figsize=(12, 6))
    model_names = [r['Model'] for r in results]
    x     = np.arange(len(model_names))
    width = 0.2

    tn = [r['Confusion Matrix'][0, 0] for r in results]
    fp = [r['Confusion Matrix'][0, 1] for r in results]
    fn = [r['Confusion Matrix'][1, 0] for r in results]
    tp = [r['Confusion Matrix'][1, 1] for r in results]

    fpr_vals     = [fp[i] / (fp[i] + tn[i]) if (fp[i] + tn[i]) > 0 else 0 for i in range(len(results))]
    sensitivity  = [tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0 for i in range(len(results))]
    specificity  = [tn[i] / (tn[i] + fp[i]) if (tn[i] + fp[i]) > 0 else 0 for i in range(len(results))]

    ax.bar(x - 1.5*width, sensitivity,                         width, label='Sensitivity (TPR)', alpha=0.8)
    ax.bar(x - 0.5*width, specificity,                         width, label='Specificity',        alpha=0.8)
    ax.bar(x + 0.5*width, fpr_vals,                            width, label='False Positive Rate', alpha=0.8)
    ax.bar(x + 1.5*width, [r['Accuracy'] for r in results],   width, label='Accuracy',            alpha=0.8)

    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Model Performance vs Confusion Matrix Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig("06_performance_vs_matrix_metrics.png", dpi=300, bbox_inches='tight')
    print(" Saved: 06_performance_vs_matrix_metrics.png")
    plt.show()

    # --- Plot 7: ROC-AUC Curves ---
    fig, ax = plt.subplots(figsize=(10, 8))
    for result in results:
        fpr_curve, tpr_curve, _ = roc_curve(y_test, result['y_prob'])
        ax.plot(fpr_curve, tpr_curve,
                label=f"{result['Model']} (AUC = {result['ROC AUC']:.4f})", linewidth=2)
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
    print(" Saved: 07_roc_auc_curves.png")
    plt.show()

    # --- PCA, Hyperparameter Tuning, Regularization ---
    pca_results, X_train_pca, X_test_pca = pca_analysis(X_train, y_train, X_test, y_test, preprocessor)
    tuning_results, best_models          = hyperparameter_tuning(X_train, y_train, X_test, y_test, preprocessor)
    regularization_results               = regularization_analysis(X_train, y_train, X_test, y_test, preprocessor)

    # --- Save models for API ---
    print("\n" + "="*60)
    print("SAVING MODELS FOR API DEPLOYMENT")
    print("="*60)

    joblib.dump(preprocessor, 'api_preprocessor.pkl')
    print(" Saved: api_preprocessor.pkl")

    for name, mdl in best_models.items():
        joblib.dump(mdl, f'api_model_{name.replace(" ", "_").lower()}.pkl')
        print(f" Saved: api_model_{name.replace(' ', '_').lower()}.pkl")

    best_model_name = max(tuning_results, key=lambda x: x['Tuned_ROC_AUC'])['Model']
    joblib.dump(best_models[best_model_name], 'api_best_model.pkl')
    print(f" Saved best model ({best_model_name}): api_best_model.pkl")

    with open('api_feature_names.json', 'w') as fh:
        json.dump(feature_names, fh)
    print(" Saved: api_feature_names.json")

    model_metadata_dict = {
        "models_available": list(best_models.keys()),
        "best_model": best_model_name,
        "feature_names": feature_names,
        "categorical_columns": list(categorical_cols),
        "numerical_columns": list(numerical_cols),
        "training_date": datetime.now().isoformat(),
        "performance": {r['Model']: {
            "accuracy": r['Tuned_Accuracy'],
            "roc_auc": r['Tuned_ROC_AUC'],
            "f1_score": r['Tuned_F1']
        } for r in tuning_results}
    }

    with open('api_model_metadata.json', 'w') as fh:
        json.dump(model_metadata_dict, fh, indent=2)
    print(" Saved: api_model_metadata.json")

    print("\n" + "="*50)
    print(" All visualizations saved!")
    print(" All models saved for API deployment!")
    print("="*50)

    return results, kfold_results, tuning_results, best_models, pca_results, regularization_results


# ─────────────────────────────────────────────
# GRADIENT BOOSTING ANALYSIS
# ─────────────────────────────────────────────

def analyze_gradient_boosting_model():
    print("="*70)
    print("GRADIENT BOOSTING MODEL - COMPREHENSIVE ANALYSIS")
    print("="*70)

    print("\n[1/9] Loading data and model...")

    try:
        model = joblib.load('api_model_gradient_boosting.pkl')
        print(" Model loaded: Gradient Boosting")
    except Exception:
        try:
            model = joblib.load('api_best_model.pkl')
            print(" Model loaded: Best model (should be Gradient Boosting)")
        except Exception:
            print(" Could not load model. Please ensure model files exist.")
            return

    try:
        X      = pd.read_csv("AI_Thunderball_X_test.csv")
        y_true = pd.read_csv("AI_Thunderball_y_test.csv").values.ravel()
        # Build a lightweight df for the predictions table
        df = X.copy()
        df["cancer"] = y_true
        df["patient_id"] = range(1000, 1000 + len(y_true))
        print(f" Test data loaded: {len(y_true)} held-out samples "
              f"(use test split to get honest metrics)")
    except FileNotFoundError:
        print("  WARNING: Test split files not found. Falling back to full dataset.")
        print("  Metrics will be OPTIMISTIC — model has seen training rows before.")
        try:
            df     = pd.read_csv("AI_Thunderball_FeatureEngineeredDataset.csv")
            y_true = df["cancer"].values
            X      = df.drop(columns=["cancer", "patient_id"])
        except Exception:
            print(" Could not load dataset.")
            return

    print(f"  Features: {X.shape[1]}")
    print(f"  Positive cases: {y_true.sum()} ({y_true.mean()*100:.1f}%)")
    print(f"  Negative cases: {(~y_true.astype(bool)).sum()} ({(1-y_true.mean())*100:.1f}%)")

    print("\n[2/9] Generating predictions...")

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    predictions_df = df.copy()
    predictions_df['predicted_cancer']  = y_pred
    predictions_df['cancer_probability'] = y_prob
    predictions_df['prediction_correct'] = (y_pred == y_true)
    predictions_df['probability_rounded'] = np.round(y_prob * 100, 2)

    print("\n[3/9] Confusion Matrix Analysis...")

    cm            = confusion_matrix(y_true, y_pred)
    cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp

    accuracy        = (tp + tn) / total
    sensitivity_val = tp / (tp + fn)
    specificity_val = tn / (tn + fp)
    precision_val   = tp / (tp + fp)
    f1_score_val    = 2 * (precision_val * sensitivity_val) / (precision_val + sensitivity_val)

    print(f"\n  Confusion Matrix (Raw Counts):")
    print(f"  {'':<15} {'Predicted Negative':<20} {'Predicted Positive':<20}")
    print(f"  {'Actual Negative':<15} {tn:<20} {fp:<20}")
    print(f"  {'Actual Positive':<15} {fn:<20} {tp:<20}")

    print(f"\n  Normalized Confusion Matrix:")
    print(f"  {'':<15} {'Predicted Negative':<20} {'Predicted Positive':<20}")
    print(f"  {'Actual Negative':<15} {cm_normalized[0,0]:.3f} ({cm_normalized[0,0]*100:.1f}%)          "
          f"{cm_normalized[0,1]:.3f} ({cm_normalized[0,1]*100:.1f}%)")
    print(f"  {'Actual Positive':<15} {cm_normalized[1,0]:.3f} ({cm_normalized[1,0]*100:.1f}%)          "
          f"{cm_normalized[1,1]:.3f} ({cm_normalized[1,1]*100:.1f}%)")

    print(f"\n  Performance Metrics:")
    print(f"  Accuracy    : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Sensitivity : {sensitivity_val:.4f} ({sensitivity_val*100:.2f}%) - True Positive Rate")
    print(f"  Specificity : {specificity_val:.4f} ({specificity_val*100:.2f}%) - True Negative Rate")
    print(f"  Precision   : {precision_val:.4f} ({precision_val*100:.2f}%)")
    print(f"  F1-Score    : {f1_score_val:.4f} ({f1_score_val*100:.2f}%)")

    print("\n[4/9] ROC-AUC Curve Analysis...")

    fpr, tpr, thresholds_roc = roc_curve(y_true, y_prob)
    roc_auc_val    = auc(fpr, tpr)
    youden_j       = tpr - fpr
    optimal_idx    = np.argmax(youden_j)
    optimal_threshold_roc = thresholds_roc[optimal_idx]

    print(f"  ROC-AUC Score: {roc_auc_val:.4f}")
    print(f"  Optimal threshold (Youden's index): {optimal_threshold_roc:.4f}")
    print(f"    At {optimal_threshold_roc:.4f}: TPR={tpr[optimal_idx]:.4f}, FPR={fpr[optimal_idx]:.4f}")

    print("\n[5/9] Precision-Recall Curve Analysis...")

    precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_true, y_prob)
    avg_precision  = average_precision_score(y_true, y_prob)
    f1_scores      = 2 * (precision_vals[:-1] * recall_vals[:-1]) / (
                         precision_vals[:-1] + recall_vals[:-1] + 1e-10)
    best_f1_idx    = np.argmax(f1_scores)
    best_threshold_pr = thresholds_pr[best_f1_idx]

    print(f"  Average Precision: {avg_precision:.4f}")
    print(f"  Best F1 threshold: {best_threshold_pr:.4f}")
    print(f"    At {best_threshold_pr:.4f}: Precision={precision_vals[best_f1_idx]:.4f}, "
          f"Recall={recall_vals[best_f1_idx]:.4f}, F1={f1_scores[best_f1_idx]:.4f}")

    print("\n[6/9] Probability Distribution Analysis...")

    prob_negative = y_prob[y_true == 0]
    prob_positive = y_prob[y_true == 1]

    print(f"\n  Probability Statistics:")
    print(f"  {'Class':<15} {'Count':<10} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print(f"  {'-'*60}")
    print(f"  {'Negative (0)':<15} {len(prob_negative):<10} {prob_negative.mean():<10.4f} "
          f"{prob_negative.std():<10.4f} {prob_negative.min():<10.4f} {prob_negative.max():<10.4f}")
    print(f"  {'Positive (1)':<15} {len(prob_positive):<10} {prob_positive.mean():<10.4f} "
          f"{prob_positive.std():<10.4f} {prob_positive.min():<10.4f} {prob_positive.max():<10.4f}")

    print("\n[7/9] Creating visualizations...")

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("Gradient Boosting Model - Key Performance Visualizations",
                 fontsize=16, fontweight='bold')

    ax1 = plt.subplot(2, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'],
                cbar_kws={'label': 'Count'})
    ax1.set_title('Confusion Matrix (Raw Counts)', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Predicted Label', fontweight='bold')
    ax1.set_ylabel('True Label', fontweight='bold')
    ax1.text(0.5, -0.15,
             f'Accuracy: {accuracy:.3f} | Precision: {precision_val:.3f} | '
             f'Recall: {sensitivity_val:.3f} | F1: {f1_score_val:.3f}',
             transform=ax1.transAxes, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(fpr, tpr, color='darkorange', lw=3,
             label=f'ROC curve (AUC = {roc_auc_val:.4f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC = 0.5)')
    ax2.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', s=150,
                label=f'Optimal threshold = {optimal_threshold_roc:.3f}', zorder=5)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate (1 - Specificity)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('True Positive Rate (Sensitivity)', fontweight='bold', fontsize=12)
    ax2.set_title('ROC-AUC Curve', fontweight='bold', fontsize=14)
    ax2.legend(loc="lower right", fontsize=10)
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(recall_vals, precision_vals, color='green', lw=3,
             label=f'PR curve (AP = {avg_precision:.4f})')
    ax3.axhline(y=y_true.mean(), color='gray', linestyle='--', lw=2,
                label=f'Baseline (Prevalence = {y_true.mean():.3f})')
    ax3.scatter(recall_vals[best_f1_idx], precision_vals[best_f1_idx],
                marker='o', color='red', s=150,
                label=f'Best F1 threshold = {best_threshold_pr:.3f}', zorder=5)
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('Recall (Sensitivity)', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Precision', fontweight='bold', fontsize=12)
    ax3.set_title('Precision-Recall Curve', fontweight='bold', fontsize=14)
    ax3.legend(loc="lower left", fontsize=10)
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(2, 2, 4)
    ax4.hist([prob_negative, prob_positive], bins=20, alpha=0.7,
             label=['Actual Negative (Class 0)', 'Actual Positive (Class 1)'],
             color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=1)
    ax4.axvline(x=0.5, color='green', linestyle='--', linewidth=2.5,
                label='Default threshold (0.5)')
    ax4.axvline(x=optimal_threshold_roc, color='orange', linestyle='--', linewidth=2.5,
                label=f'Optimal ROC threshold ({optimal_threshold_roc:.3f})')
    ax4.axvline(x=best_threshold_pr, color='purple', linestyle='--', linewidth=2.5,
                label=f'Best F1 threshold ({best_threshold_pr:.3f})')
    ax4.set_xlabel('Predicted Probability of Cancer', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Frequency', fontweight='bold', fontsize=12)
    ax4.set_title('Probability Distribution by Actual Class', fontweight='bold', fontsize=14)
    ax4.legend(loc='upper center', fontsize=9, ncol=2)
    ax4.grid(True, alpha=0.3)
    stats_text = (f"Class 0: μ={prob_negative.mean():.3f}, σ={prob_negative.std():.3f}\n"
                  f"Class 1: μ={prob_positive.mean():.3f}, σ={prob_positive.std():.3f}")
    ax4.text(0.98, 0.98, stats_text, transform=ax4.transAxes, ha='right', va='top',
             fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig("gradient_boosting_key_visualizations.png", dpi=300, bbox_inches='tight')
    print(" Saved: gradient_boosting_key_visualizations.png")
    plt.show()

    print("\n[8/9] Performance Metrics Table...")
    print("\n" + "="*70)
    print("PERFORMANCE METRICS SUMMARY")
    print("="*70)

    metrics_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall (Sensitivity)',
                   'Specificity', 'F1-Score', 'ROC-AUC', 'Avg Precision'],
        'Value': [accuracy, precision_val, sensitivity_val,
                  specificity_val, f1_score_val, roc_auc_val, avg_precision],
        'Formula': [
            '(TP+TN)/(TP+TN+FP+FN)', 'TP/(TP+FP)', 'TP/(TP+FN)',
            'TN/(TN+FP)', '2*(P*R)/(P+R)', 'Area under ROC curve', 'Area under PR curve'
        ]
    }

    metrics_df = pd.DataFrame(metrics_data)
    metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f'{x:.4f}')
    print("\n", metrics_df.to_string(index=False))
    metrics_df.to_csv("gradient_boosting_performance_metrics.csv", index=False)
    print("\n Saved: gradient_boosting_performance_metrics.csv")

    print("\n[9/9] Threshold Comparison Table...")
    print("\n" + "="*70)
    print("THRESHOLD COMPARISON")
    print("="*70)

    thresholds_to_test = sorted(set([0.3, 0.4, 0.5, 0.6, 0.7,
                                     optimal_threshold_roc, best_threshold_pr]))
    threshold_results  = []

    for thresh in thresholds_to_test:
        y_pred_thresh = (y_prob >= thresh).astype(int)
        acc   = accuracy_score(y_true, y_pred_thresh)
        prec  = precision_score(y_true, y_pred_thresh, zero_division=0)
        rec   = recall_score(y_true, y_pred_thresh, zero_division=0)
        f1    = f1_score(y_true, y_pred_thresh, zero_division=0)
        pos_predictions = int(y_pred_thresh.sum())
        threshold_results.append({
            'Threshold': f'{thresh:.3f}',
            'Accuracy': f'{acc:.4f}',
            'Precision': f'{prec:.4f}',
            'Recall': f'{rec:.4f}',
            'F1-Score': f'{f1:.4f}',
            'Pos Predictions': pos_predictions,
            '% Pos Pred': f'{pos_predictions/len(y_true)*100:.1f}%'
        })

    threshold_df = pd.DataFrame(threshold_results)
    print("\n", threshold_df.to_string(index=False))
    threshold_df.to_csv("gradient_boosting_threshold_comparison.csv", index=False)
    print("\n Saved: gradient_boosting_threshold_comparison.csv")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)

    return predictions_df, {
        'confusion_matrix': cm,
        'roc_auc': roc_auc_val,
        'fpr': fpr,
        'tpr': tpr,
        'precision_vals': precision_vals,
        'recall_vals': recall_vals,
        'avg_precision': avg_precision,
        'y_prob': y_prob,
        'y_pred': y_pred,
        'y_true': y_true,
        'accuracy': accuracy,
        'precision': precision_val,
        'sensitivity': sensitivity_val,
        'specificity': specificity_val,
        'f1_score': f1_score_val
    }


# ─────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────

def generate_simple_prediction_report(output_file="final_prediction_report.csv"):
    print("\n" + "="*60)
    print("GENERATING SIMPLE PREDICTION REPORT")
    print("="*60)

    try:
        print("\n[1/3] Loading data and model...")
        df = pd.read_csv("AI_Thunderball_FeatureEngineeredDataset.csv")
        print(f" Loaded dataset: {df.shape[0]} rows")

        if 'patient_id' not in df.columns:
            df['patient_id'] = range(1000, 1000 + len(df))

        try:
            model = joblib.load('api_model_gradient_boosting.pkl')
            print(" Loaded model: Gradient Boosting")
        except Exception:
            model = joblib.load('api_best_model.pkl')
            print(" Loaded model: Best Model")

        print("\n[2/3] Generating predictions...")
        y_true      = df["cancer"]
        patient_ids = df["patient_id"]
        X           = df.drop(columns=["cancer", "patient_id"])

        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        print(f" Predictions generated for {len(y_pred)} patients")

        print("\n[3/3] Creating report...")
        report_df = pd.DataFrame({
            'Patient_No': patient_ids,
            'Actual': y_true,
            'Predicted': y_pred,
            'Probability': np.round(y_prob, 4),
            'Correct': (y_pred == y_true).astype(int)
        })
        report_df = report_df.sort_values('Patient_No').reset_index(drop=True)
        report_df.to_csv(output_file, index=False)

        print(f"\n Report saved: {output_file}")
        print(f"  Shape: {report_df.shape[0]} rows × {report_df.shape[1]} columns")
        print(f"  Patient IDs: {report_df['Patient_No'].min()} to {report_df['Patient_No'].max()}")

        print("\n" + "="*60)
        print("SAMPLE OF REPORT (First 10 rows)")
        print("="*60)
        print("\n", report_df.head(10).to_string())

        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        print(f"Total Patients: {len(report_df)}")
        print(f"Actual Cancer Cases: {report_df['Actual'].sum()} ({report_df['Actual'].mean()*100:.1f}%)")
        print(f"Predicted Cancer Cases: {report_df['Predicted'].sum()} ({report_df['Predicted'].mean()*100:.1f}%)")
        print(f"Correct Predictions: {report_df['Correct'].sum()} ({report_df['Correct'].mean()*100:.1f}%)")

        tn_val = int(((report_df['Actual'] == 0) & (report_df['Predicted'] == 0)).sum())
        fp_val = int(((report_df['Actual'] == 0) & (report_df['Predicted'] == 1)).sum())
        fn_val = int(((report_df['Actual'] == 1) & (report_df['Predicted'] == 0)).sum())
        tp_val = int(((report_df['Actual'] == 1) & (report_df['Predicted'] == 1)).sum())

        print("\nConfusion Matrix:")
        print(f"  True Negatives:  {tn_val}")
        print(f"  False Positives: {fp_val}")
        print(f"  False Negatives: {fn_val}")
        print(f"  True Positives:  {tp_val}")

        return report_df

    except Exception as e:
        print(f"\n Error generating report: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def generate_simple_prediction_report_with_custom_id(output_file="final_prediction_report.csv",
                                                     id_column="patient_id"):
    print("\n" + "="*60)
    print(f"GENERATING SIMPLE PREDICTION REPORT (using '{id_column}')")
    print("="*60)

    try:
        df = pd.read_csv("AI_Thunderball_FeatureEngineeredDataset.csv")
        print(f" Loaded dataset: {df.shape[0]} rows")

        if id_column not in df.columns:
            print(f"  Column '{id_column}' not found. Available columns: {list(df.columns)}")
            for pid in ['patient_id', 'Patient_ID', 'id', 'ID', 'patientid']:
                if pid in df.columns:
                    id_column = pid
                    print(f" Using '{id_column}' as patient ID column")
                    break
            else:
                print(" No patient ID column found, creating sequential IDs")
                df['patient_id'] = range(1000, 1000 + len(df))
                id_column = 'patient_id'

        try:
            model = joblib.load('api_model_gradient_boosting.pkl')
            print(" Loaded model: Gradient Boosting")
        except Exception:
            model = joblib.load('api_best_model.pkl')
            print(" Loaded model: Best Model")

        y_true      = df["cancer"]
        patient_ids = df[id_column]

        cols_to_drop = ["cancer"]
        if id_column in df.columns:
            cols_to_drop.append(id_column)
        X = df.drop(columns=cols_to_drop)

        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        report_df = pd.DataFrame({
            'Patient_No': patient_ids,
            'Actual': y_true,
            'Predicted': y_pred,
            'Probability': np.round(y_prob, 4),
            'Correct': (y_pred == y_true).astype(int)
        })
        report_df = report_df.sort_values('Patient_No').reset_index(drop=True)
        report_df.to_csv(output_file, index=False)

        print(f"\n Report saved: {output_file}")
        print(f"  Shape: {report_df.shape[0]} rows × {report_df.shape[1]} columns")
        print(f"  Patient IDs: {report_df['Patient_No'].min()} to {report_df['Patient_No'].max()}")

        print("\n" + "="*60)
        print("SAMPLE OF REPORT (First 10 rows)")
        print("="*60)
        print("\n", report_df.head(10).to_string())

        return report_df

    except Exception as e:
        print(f"\n Error generating report: {str(e)}")
        return None


# ─────────────────────────────────────────────
# FASTAPI PYDANTIC MODELS
# ─────────────────────────────────────────────

class PatientData(BaseModel):
    age: int
    gender: str
    blood_group: str
    ethnicity: str
    height_cm: int
    weight_kg: float
    smoking_status: str
    alcohol_consumption: str
    family_history_cancer: int
    physical_activity_level: str
    diet_quality: str
    diabetes: int
    hypertension: int
    asthma: int
    cardiac_disease: int
    prior_radiation_exposure: int
    unexplained_weight_loss: int
    persistent_fatigue: int
    chronic_pain: int
    abnormal_bleeding: int
    persistent_cough: int
    lump_presence: int
    hemoglobin_level: float
    wbc_count: int
    platelet_count: int
    tumor_marker_level: float
    imaging_abnormality: int
    tumor_size_cm: float

class BatchPatientData(BaseModel):
    patients: List[PatientData]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    cancer_risk: str
    model_used: str
    timestamp: str

class BatchPredictionResponse(BaseModel):
    predictions: List[dict]
    model_used: str
    timestamp: str
    total_patients: int
    positive_cases: int
    positive_percentage: float

class ModelInfo(BaseModel):
    name: str
    type: str
    accuracy: float
    roc_auc: float
    f1_score: float
    parameters: dict

class HealthCheck(BaseModel):
    status: str
    timestamp: str
    models_loaded: List[str]
    api_version: str

# Global API state
api_models: Dict[str, Any] = {}
api_preprocessor            = None
api_feature_names: List[str] = []
api_model_metadata: Dict     = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, release on shutdown."""
    global api_models, api_preprocessor, api_feature_names, api_model_metadata

    print("\n" + "="*60)
    print("LOADING MODELS FOR API")
    print("="*60)

    try:
        api_preprocessor = joblib.load('api_preprocessor.pkl')
        print(" Preprocessor loaded")

        model_files = {
            "logistic_regression": "api_model_logistic_regression.pkl",
            "random_forest": "api_model_random_forest.pkl",
            "svm": "api_model_svm.pkl",
            "gradient_boosting": "api_model_gradient_boosting.pkl",
            "best_model": "api_best_model.pkl"
        }

        for model_name, file_path in model_files.items():
            try:
                api_models[model_name] = joblib.load(file_path)
                print(f" {model_name} loaded")
            except FileNotFoundError:
                print(f" {model_name} not found, skipping...")

        with open('api_feature_names.json', 'r') as fh:
            api_feature_names = json.load(fh)
        print(f" Feature names loaded ({len(api_feature_names)} features)")

        with open('api_model_metadata.json', 'r') as fh:
            api_model_metadata = json.load(fh)
        print(" Model metadata loaded")

        print("="*60)
        print("API READY FOR REQUESTS")
        print("="*60)

    except Exception as e:
        print(f" Error loading models: {e}")
        raise e

    yield  # app runs here

    api_models.clear()
    print("API shutdown: models unloaded.")


app = FastAPI(
    title="AI Thunderball Cancer Detection API",
    description="API for cancer risk prediction using machine learning models",
    version="1.0.0",
    lifespan=lifespan
)


# ─────────────────────────────────────────────
# FEATURE ENGINEERING HELPER (shared by /predict endpoints)
# ─────────────────────────────────────────────

def _apply_feature_engineering(patient_df: pd.DataFrame) -> pd.DataFrame:
    patient_df = patient_df.copy()

    patient_df['bmi'] = patient_df['weight_kg'] / ((patient_df['height_cm'] / 100) ** 2)

    def get_age_category(age):
        if age <= 30:   return 'Young'
        if age <= 50:   return 'Middle-aged'
        if age <= 70:   return 'Senior'
        return 'Elderly'

    patient_df['age_category'] = patient_df['age'].apply(get_age_category)
    patient_df['platelet_lymphocyte_ratio'] = patient_df['platelet_count'] / (
        patient_df['wbc_count'] * 0.3 + 1e-10)

    smoking_map  = {'Never': 0, 'Former': 1, 'Current': 2}
    alcohol_map  = {'Never': 0, 'Moderate': 1, 'High': 2}
    activity_map = {'High': 0, 'Medium': 1, 'Low': 2}
    diet_map     = {'Good': 0, 'Average': 1, 'Poor': 2}

    patient_df['lifestyle_risk_score'] = (
        patient_df['smoking_status'].map(smoking_map).fillna(0) +
        patient_df['alcohol_consumption'].map(alcohol_map).fillna(0) +
        patient_df['physical_activity_level'].map(activity_map).fillna(0) +
        patient_df['diet_quality'].map(diet_map).fillna(0)
    )

    def calculate_clinical_risk(row):
        score = 0
        if row['age'] > 50:   score += 3
        elif row['age'] > 40: score += 2
        elif row['age'] > 30: score += 1

        if row['family_history_cancer'] == 1: score += 4

        if row['unexplained_weight_loss'] == 1: score += 3
        if row['lump_presence'] == 1:           score += 4
        if row['persistent_fatigue'] == 1:      score += 2
        if row['chronic_pain'] == 1:            score += 2
        if row['abnormal_bleeding'] == 1:       score += 3
        if row['persistent_cough'] == 1:        score += 2

        score += row['diabetes'] + row['hypertension'] + row['cardiac_disease'] * 2

        if row['lifestyle_risk_score'] > 6:   score += 3
        elif row['lifestyle_risk_score'] > 4: score += 2
        elif row['lifestyle_risk_score'] > 2: score += 1

        if row['tumor_marker_level'] > 50:   score += 3
        elif row['tumor_marker_level'] > 25: score += 2
        elif row['tumor_marker_level'] > 10: score += 1

        if row['tumor_size_cm'] > 5:    score += 4
        elif row['tumor_size_cm'] > 2:  score += 2
        elif row['tumor_size_cm'] > 0.5: score += 1

        if row['imaging_abnormality'] == 1: score += 3

        if row['hemoglobin_level'] < 12 and row['gender'] == 'Female':    score += 2
        elif row['hemoglobin_level'] < 13.5 and row['gender'] == 'Male':  score += 2

        if row['platelet_count'] > 400000 or row['platelet_count'] < 150000: score += 1

        return score

    patient_df['clinical_risk_score'] = patient_df.apply(calculate_clinical_risk, axis=1)
    return patient_df


# ─────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to AI Thunderball Cancer Detection API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict (POST) - Single patient prediction",
            "predict_batch": "/predict/batch (POST) - Batch predictions",
            "models": "/models (GET) - List available models",
            "health": "/health (GET) - API health check"
        },
        "documentation": "/docs"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(patient: PatientData, model_name: str = "best_model"):
    if model_name not in api_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {list(api_models.keys())}"
        )

    try:
        patient_df = pd.DataFrame([patient.model_dump()])
        patient_df = _apply_feature_engineering(patient_df)

        model      = api_models[model_name]
        prediction = int(model.predict(patient_df)[0])
        probability = float(model.predict_proba(patient_df)[0, 1])

        risk_level = "Low" if probability < 0.3 else "Medium" if probability < 0.7 else "High"

        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            cancer_risk=risk_level,
            model_used=model_name,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(batch_data: BatchPatientData, model_name: str = "best_model"):
    if model_name not in api_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {list(api_models.keys())}"
        )

    try:
        patient_df   = pd.DataFrame([p.model_dump() for p in batch_data.patients])
        patient_df   = _apply_feature_engineering(patient_df)

        model        = api_models[model_name]
        predictions  = model.predict(patient_df)
        probabilities = model.predict_proba(patient_df)[:, 1]

        prediction_results = []
        positive_cases     = 0

        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            risk_level = "Low" if prob < 0.3 else "Medium" if prob < 0.7 else "High"
            prediction_results.append({
                "patient_id": i + 1,
                "prediction": int(pred),
                "probability": float(prob),
                "cancer_risk": risk_level
            })
            if pred == 1:
                positive_cases += 1

        total_patients     = len(predictions)
        positive_percentage = (positive_cases / total_patients * 100) if total_patients > 0 else 0

        return BatchPredictionResponse(
            predictions=prediction_results,
            model_used=model_name,
            timestamp=datetime.now().isoformat(),
            total_patients=total_patients,
            positive_cases=positive_cases,
            positive_percentage=round(positive_percentage, 2)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/models", response_model=List[ModelInfo], tags=["Models"])
async def get_models():
    try:
        models_info = []
        for model_name, performance in api_model_metadata.get("performance", {}).items():
            model_key = model_name.replace(" ", "_").lower()
            params    = {}
            if model_key in api_models and hasattr(api_models[model_key], 'get_params'):
                params = api_models[model_key].get_params()
            models_info.append(ModelInfo(
                name=model_name,
                type=model_name,
                accuracy=performance.get("accuracy", 0.0),
                roc_auc=performance.get("roc_auc", 0.0),
                f1_score=performance.get("f1_score", 0.0),
                parameters=params
            ))
        return models_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {str(e)}")


@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=list(api_models.keys()),
        api_version="1.0.0"
    )


@app.get("/features", tags=["Information"])
async def get_features():
    try:
        return {
            "total_features": len(api_feature_names),
            "features": api_feature_names,
            "categorical_columns": api_model_metadata.get("categorical_columns", []),
            "numerical_columns": api_model_metadata.get("numerical_columns", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving features: {str(e)}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def run_full_pipeline():
    print("="*60)
    print("AI THUNDERBALL - CANCER DETECTION DATA PIPELINE")
    print("="*60)

    generate_synthetic_data()
    introduce_anomalies()
    clean_data()
    engineer_features()
    train_test_split_data()
    run_machine_learning_pipeline()

    print("\n" + "="*60)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    run_full_pipeline()

    predictions_df, metrics = analyze_gradient_boosting_model()
    report_df = generate_simple_prediction_report("final_prediction_report.csv")

    print("\n" + "="*60)
    print("STARTING FASTAPI SERVER")
    print("="*60)
    print("\nAPI available at:")
    print("  http://localhost:8000")
    print("  http://localhost:8000/docs  (Swagger UI)")
    print("  http://localhost:8000/redoc (ReDoc UI)")
    print("="*60)

    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
