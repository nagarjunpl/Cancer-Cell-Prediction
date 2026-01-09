import pandas as pd
import numpy as np
from ctgan import CTGAN
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, auc, make_scorer
)

def generate_synthetic_data():
    print(" Generating synthetic patient data... \n")
    rows = 1300
    # Data structure
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
    # Generate synthetic data 
    discrete_cols = [col for col in df.columns if df[col].dtype == 'object' or df[col].nunique() <= 5]
    
    ctgan = CTGAN(epochs=200, batch_size=500, verbose=True)
    ctgan.fit(df, discrete_columns=discrete_cols)
    
    synthetic_data = ctgan.sample(1200)
    
    # Save only synthetic data
    synthetic_data.to_csv("AI_Thunderball_Dataset.csv", index=False)
    
    print(f"\n Generated: {synthetic_data.shape[0]} records")
    return synthetic_data

def introduce_anomalies(input_file="AI_Thunderball_Dataset.csv", output_file="AI_Thunderball_RawDataset.csv"):
    print("\n" + "="*50)
    print("Introducing anomalies...")
    
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f" Error: '{input_file}' not found")
        return None
    
    print(f" Input: {df.shape}")
    
    # 1. NULL values
    null_idx = df.sample(frac=0.03, random_state=42).index
    df.loc[null_idx, ["wbc_count", "hemoglobin_level", "platelet_count"]] = np.nan
    print(f"1️. NULLs: {df[['wbc_count', 'hemoglobin_level', 'platelet_count']].isna().sum().sum()}")
    
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
    
    # Save
    df.to_csv(output_file, index=False)
    
    print(f"\nFinal: {df.shape}")
    print(f"Saved: '{output_file}'")
    
    return df

def clean_data(input_file="AI_Thunderball_RawDataset.csv", output_file="AI_Thunderball_CleanedDataset.csv"):
    print("\n" + "="*50)
    print("Cleaning data...")
    
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f" Error: '{input_file}' not found")
        return None
    
    print(f" Input: {df.shape}")
    
    original_rows = len(df)
    
    # 1. Handle NULL values
    df = df.fillna(df.median(numeric_only=True)).fillna(df.mode().iloc[0])
    print(f"1️. NULLs filled")
    
    # 2. Remove duplicates
    dup_count = df.duplicated().sum()
    if len(df) - dup_count >= 1200:
        df.drop_duplicates(inplace=True)
        print(f"2️. Duplicates removed: {dup_count}")
    else:
        # keep 1200 rows
        to_remove = len(df) - 1200
        duplicates = df[df.duplicated(keep='first')]
        if len(duplicates) >= to_remove:
            df = df.drop_duplicates(keep='first')
            df = df.drop(duplicates.index[:to_remove], errors="ignore")
            print(f"2️. Partial duplicates removed")
    
    # 3. Remove outliers
    before_outliers = len(df)
    df = df[(df["age"] >= 18) & (df["age"] <= 90)]
    df = df[(df["tumor_size_cm"] >= 0.5) & (df["tumor_size_cm"] <= 10)]
    outliers_removed = before_outliers - len(df)
    print(f"3️. Outliers removed: {outliers_removed}")
    
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
    
    # 6. Ensure data types
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
    
    # Reset
    df.reset_index(drop=True, inplace=True)
    
    # Save
    df.to_csv(output_file, index=False)
    
    # Summary
    print(f"\nFinal dataset: {df.shape}")
    print(f"  Rows preserved: {len(df)} / {original_rows}")
    print(f"  Age range: {df['age'].min()}-{df['age'].max()}")
    print(f"\n Saved: '{output_file}' with {len(df)} rows")
    
    return df

def engineer_features(input_file="AI_Thunderball_CleanedDataset.csv", 
                      output_file="AI_Thunderball_FeatureEngineeredDataset.csv"):
    print("\n" + "="*50)
    print("Feature Engineering...")
    
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f" Error: '{input_file}' not found")
        return None
    
    # Make a copy to preserve original columns
    df_engineered = df.copy()
    
    # 1. BMI (Body Mass Index)
    df_engineered['bmi'] = df_engineered['weight_kg'] / ((df_engineered['height_cm'] / 100) ** 2)
    
    # 2. Age Categories
    df_engineered['age_category'] = pd.cut(df_engineered['age'],
                                           bins=[0, 30, 50, 70, 100],
                                           labels=['Young', 'Middle-aged', 'Senior', 'Elderly'])
    
    # 3. Blood Cell Ratios
    df_engineered['platelet_lymphocyte_ratio'] = df_engineered['platelet_count'] / (df_engineered['wbc_count'] * 0.3 + 1e-10)
    
    # Define mapping functions
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
    
    # Calculate lifestyle risk score directly
    df_engineered['lifestyle_risk_score'] = (
        df_engineered['smoking_status'].apply(get_smoking_score) +
        df_engineered['alcohol_consumption'].apply(get_alcohol_score) +
        df_engineered['physical_activity_level'].apply(get_activity_score) +
        df_engineered['diet_quality'].apply(get_diet_score)
    )
    
    # 5. Clinical Risk Score (Cancer-specific)
    def calculate_clinical_risk(row):
        score = 0
        
        # Age factor
        if row['age'] > 50:
            score += 3
        elif row['age'] > 40:
            score += 2
        elif row['age'] > 30:
            score += 1
        
        # Family history
        if row['family_history_cancer'] == 1:
            score += 4
        
        # Symptoms (weight them based on clinical importance)
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
        
        # Comorbidities
        comorbidity_score = (row['diabetes'] * 1 + row['hypertension'] * 1 + row['cardiac_disease'] * 2)
        score += comorbidity_score
        
        # Lifestyle
        if row['lifestyle_risk_score'] > 6:
            score += 3
        elif row['lifestyle_risk_score'] > 4:
            score += 2
        elif row['lifestyle_risk_score'] > 2:
            score += 1
        
        # Lab values
        if row['tumor_marker_level'] > 50:
            score += 3
        elif row['tumor_marker_level'] > 25:
            score += 2
        elif row['tumor_marker_level'] > 10:
            score += 1
        
        # Tumor characteristics
        if row['tumor_size_cm'] > 5:
            score += 4
        elif row['tumor_size_cm'] > 2:
            score += 2
        elif row['tumor_size_cm'] > 0.5:
            score += 1
        
        if row['imaging_abnormality'] == 1:
            score += 3
        
        # Blood abnormalities
        if row['hemoglobin_level'] < 12 and row['gender'] == 'Female':
            score += 2
        elif row['hemoglobin_level'] < 13.5 and row['gender'] == 'Male':
            score += 2
        
        # Platelet count abnormalities
        if row['platelet_count'] > 400000 or row['platelet_count'] < 150000:
            score += 1
        
        return score
    
    df_engineered['clinical_risk_score'] = df_engineered.apply(calculate_clinical_risk, axis=1)
    
    # Save the engineered dataset
    df_engineered.to_csv(output_file, index=False)
    
    # Show some statistics
    print("\nFeature Engineering Statistics:")
    print(f"  BMI range: {df_engineered['bmi'].min():.1f} - {df_engineered['bmi'].max():.1f}")
    print(f"  Clinical Risk Score range: {df_engineered['clinical_risk_score'].min()} - {df_engineered['clinical_risk_score'].max()}")
    print(f"  Lifestyle Risk Score range: {df_engineered['lifestyle_risk_score'].min()} - {df_engineered['lifestyle_risk_score'].max()}")
    print(f"\n✓ Saved engineered dataset to: '{output_file}'")
    print(f"✓ Dataset shape: {df_engineered.shape}")
    
    return df_engineered

def train_test_split_data(input_file="AI_Thunderball_FeatureEngineeredDataset.csv"):
    print("\n" + "="*50)
    print("Splitting data into train and test sets...")

    df = pd.read_csv(input_file)

    # Target variable
    y = df["cancer"]

    # Drop target & non-useful ID column
    X = df.drop(columns=["cancer", "patient_id"])

    # 80% Train, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Save splits
    X_train.to_csv("AI_Thunderball_X_train.csv", index=False)
    X_test.to_csv("AI_Thunderball_X_test.csv", index=False)
    y_train.to_csv("AI_Thunderball_y_train.csv", index=False)
    y_test.to_csv("AI_Thunderball_y_test.csv", index=False)

    print(" Train-Test split completed")
    print(f" Training samples: {X_train.shape[0]}")
    print(f" Testing samples : {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test

def train_and_evaluate_models_kfold(X_train, y_train, X_test, y_test, n_splits=5):
    print("\n" + "="*50)
    print(f"Training and evaluating models with {n_splits}-Fold Cross Validation...")
    
    # Identify feature types
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
    numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    
    # Preprocessing pipeline
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
    
    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    
    # scoring dictionary 
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    # Results storage
    results = []
    
    # K-Fold Cross Validation
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\n{'='*30}")
        print(f"Model: {name}")
        print('='*30)
        
        # Create pipeline
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        
        # Perform K-Fold Cross Validation
        cv_results = cross_validate(
            pipeline, X_train, y_train,
            cv=kfold,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1
        )
        
        # Calculate mean and std for each metric
        cv_metrics = {}
        for metric in scoring.keys():
            scores = cv_results[f'test_{metric}']
            cv_metrics[f'CV_{metric}_mean'] = np.mean(scores)
            cv_metrics[f'CV_{metric}_std'] = np.std(scores)
            print(f"  {metric.upper()}: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
        
        # Train on full training set
        pipeline.fit(X_train, y_train)
        
        # Test set evaluation
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        test_metrics = {
            "Test_Accuracy": accuracy_score(y_test, y_pred),
            "Test_Precision": precision_score(y_test, y_pred),
            "Test_Recall": recall_score(y_test, y_pred),
            "Test_F1": f1_score(y_test, y_pred),
            "Confusion_Matrix": confusion_matrix(y_test, y_pred)
        }
        
        # Calculate ROC AUC if probabilities are available
        if y_prob is not None:
            test_metrics["Test_ROC_AUC"] = roc_auc_score(y_test, y_prob)
        else:
            test_metrics["Test_ROC_AUC"] = 0.5  # Default value for classifiers without probability
        
        print(f"\n  Test Set Performance:")
        print(f"  Accuracy : {test_metrics['Test_Accuracy']:.4f}")
        print(f"  Precision: {test_metrics['Test_Precision']:.4f}")
        print(f"  Recall   : {test_metrics['Test_Recall']:.4f}")
        print(f"  F1 Score : {test_metrics['Test_F1']:.4f}")
        print(f"  ROC AUC  : {test_metrics['Test_ROC_AUC']:.4f}")
        
        # Store results
        model_result = {
            "Model": name,
            "Pipeline": pipeline,
            "CV_Results": cv_results,
            "CV_Metrics": cv_metrics,
            "Test_Metrics": test_metrics,
            "y_pred": y_pred,
            "y_prob": y_prob
        }
        
        results.append(model_result)
    
    return results, preprocessor

def visualize_kfold_results(results, X_train):
    print("\n" + "="*50)
    print("Creating K-Fold Cross Validation Visualizations...")
    
    # Fig 1: K-Fold CV Performance Comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"K-Fold Cross Validation Performance ({len(results[0]['CV_Results']['test_accuracy'])}-Fold)", 
                 fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    
    for idx, metric in enumerate(metrics):
        ax = axes[positions[idx]]
        model_names = [r['Model'] for r in results]
        means = [r['CV_Metrics'][f'CV_{metric}_mean'] for r in results]
        stds = [r['CV_Metrics'][f'CV_{metric}_std'] for r in results]
        
        x = np.arange(len(model_names))
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, 
                      color=plt.cm.Set2(np.linspace(0, 1, len(model_names))))
        
        ax.set_ylabel(f'{metric.upper()}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.02, f'{mean:.3f}\n±{std:.3f}', 
                    ha='center', va='bottom', fontsize=8)
    
    # Fig 1: K-Fold Learning Curves
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig("08_kfold_cv_performance.png", dpi=300, bbox_inches='tight')
    print(" Saved: 08_kfold_cv_performance.png")
    plt.show()
    
    # Fig 2: Train-Test Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Train (CV) vs Test Set Performance Comparison", fontsize=16, fontweight='bold')
    
    comparison_metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    for idx, metric in enumerate(comparison_metrics):
        ax = axes[idx // 2, idx % 2]
        model_names = [r['Model'] for r in results]
        
        cv_means = [r['CV_Metrics'][f'CV_{metric}_mean'] for r in results]
        
        # Fix: Use correct key names for test metrics
        test_scores = []
        for r in results:
            if metric == 'accuracy':
                test_scores.append(r['Test_Metrics']['Test_Accuracy'])
            elif metric == 'precision':
                test_scores.append(r['Test_Metrics']['Test_Precision'])
            elif metric == 'recall':
                test_scores.append(r['Test_Metrics']['Test_Recall'])
            elif metric == 'f1':
                test_scores.append(r['Test_Metrics']['Test_F1'])
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, cv_means, width, label=f'CV Mean ({metric})', 
                       alpha=0.8, color='skyblue')
        bars2 = ax.bar(x + width/2, test_scores, width, label=f'Test ({metric})', 
                       alpha=0.8, color='lightcoral')
        
        ax.set_ylabel(f'{metric.upper()} Score', fontweight='bold')
        ax.set_title(f'{metric.upper()}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig("09_train_test_comparison.png", dpi=300, bbox_inches='tight')
    print(" Saved: 09_train_test_comparison.png")
    plt.show()
    
    # Fig 3: Model Stability
    fig, ax = plt.subplots(figsize=(12, 6))
    
    model_names = [r['Model'] for r in results]
    cv_stds = [r['CV_Metrics']['CV_accuracy_std'] for r in results]
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(model_names)))
    bars = ax.bar(model_names, cv_stds, color=colors, alpha=0.8)
    
    ax.set_ylabel('Standard Deviation (Lower = More Stable)', fontweight='bold')
    ax.set_title('Model Stability Across K-Folds (Accuracy)', fontsize=14, fontweight='bold')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, std in zip(bars, cv_stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{std:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig("10_model_stability.png", dpi=300, bbox_inches='tight')
    print(" Saved: 10_model_stability.png")
    plt.show()
    
    # Fig 4: Detailed K-Fold Results per Model
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Detailed K-Fold Results per Model", fontsize=16, fontweight='bold')
    
    for idx, result in enumerate(results):
        ax = axes[idx // 2, idx % 2]
        model_name = result['Model']
        cv_results = result['CV_Results']
        
        # Get fold scores
        fold_accuracies = cv_results['test_accuracy']
        fold_precisions = cv_results['test_precision']
        fold_recalls = cv_results['test_recall']
        fold_f1s = cv_results['test_f1']
        
        # Plot
        x = np.arange(1, len(fold_accuracies) + 1)
        ax.plot(x, fold_accuracies, 'o-', label='Accuracy', linewidth=2, markersize=8)
        ax.plot(x, fold_precisions, 's-', label='Precision', linewidth=2, markersize=8)
        ax.plot(x, fold_recalls, '^-', label='Recall', linewidth=2, markersize=8)
        ax.plot(x, fold_f1s, 'd-', label='F1', linewidth=2, markersize=8)
        
        ax.set_xlabel('Fold Number', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title(f"{model_name} - Fold-by-Fold Performance", fontweight='bold')
        ax.set_xticks(x)
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add mean lines
        for metric, scores, marker in [('Accuracy', fold_accuracies, 'o'),
                                       ('Precision', fold_precisions, 's'),
                                       ('Recall', fold_recalls, '^'),
                                       ('F1', fold_f1s, 'd')]:
            mean_score = np.mean(scores)
            ax.axhline(y=mean_score, color='gray', linestyle='--', alpha=0.5)
            ax.text(len(scores) + 0.1, mean_score, f'{mean_score:.3f}', 
                    va='center', ha='left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig("11_detailed_kfold_results.png", dpi=300, bbox_inches='tight')
    print(" Saved: 11_detailed_kfold_results.png")
    plt.show()
    
    # Fig 5: Performance Summary Heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data for heatmap
    metrics_list = ['CV Accuracy', 'CV Precision', 'CV Recall', 'CV F1', 'CV ROC-AUC',
                    'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1', 'Test ROC-AUC']
    model_names = [r['Model'] for r in results]
    
    heatmap_data = []
    for result in results:
        row = [
            result['CV_Metrics']['CV_accuracy_mean'],
            result['CV_Metrics']['CV_precision_mean'],
            result['CV_Metrics']['CV_recall_mean'],
            result['CV_Metrics']['CV_f1_mean'],
            result['CV_Metrics']['CV_roc_auc_mean'],
            result['Test_Metrics']['Test_Accuracy'],
            result['Test_Metrics']['Test_Precision'],
            result['Test_Metrics']['Test_Recall'],
            result['Test_Metrics']['Test_F1'],
            result['Test_Metrics']['Test_ROC_AUC']
        ]
        heatmap_data.append(row)
    
    heatmap_data = np.array(heatmap_data).T  # Transpose for better visualization

    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Add text annotations
    for i in range(len(metrics_list)):
        for j in range(len(model_names)):
            text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    # Set labels
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_yticks(np.arange(len(metrics_list)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_yticklabels(metrics_list)
    
    ax.set_title("Comprehensive Performance Summary Heatmap", fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Score', rotation=-90, va="bottom")
    
    plt.tight_layout()
    plt.savefig("12_performance_summary_heatmap.png", dpi=300, bbox_inches='tight')
    print(" Saved: 12_performance_summary_heatmap.png")
    plt.show()
    
    print("\n" + "="*50)
    print(" All K-Fold visualizations have been saved successfully!")
    print("="*50)

def create_original_visualizations(results, X_train, preprocessor):
    print("\n" + "="*50)
    print("Creating original visualizations...")
    
    # Extract Feature Names 
    def get_feature_names(preprocessor):
        feature_names = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == "num":
                feature_names.extend(columns)
            elif name == "cat":
                # Get encoded feature names from OneHotEncoder
                encoder = transformer.named_steps["onehot"]
                feature_names.extend(encoder.get_feature_names_out(columns))
        return feature_names
    
    # Get Feature Importance for All Models
    def get_top_features(result, feature_names, top_n=10):
        
        pipeline = result['Pipeline']
        model = pipeline.named_steps["model"]
        model_name = result['Model']
        
        if hasattr(model, "feature_importances_"):
            # Tree-based models (Random Forest, Gradient Boosting)
            importances = model.feature_importances_
            indices = np.argsort(importances)[-top_n:][::-1]
            top_features = [(feature_names[i], importances[i]) for i in indices]
            ylabel = 'Importance'
            return top_features, ylabel
        
        elif hasattr(model, 'coef_'):
            # Linear models (Logistic Regression)
            if len(model.coef_.shape) > 1:
                coefficients = np.abs(model.coef_[0])
            else:
                coefficients = np.abs(model.coef_)
            
            indices = np.argsort(coefficients)[-top_n:][::-1]
            top_features = [(feature_names[i], coefficients[i]) for i in indices]
            ylabel = 'Absolute Coefficient'
            return top_features, ylabel
        
        else:
            # For models without built-in feature importance (like RBF SVM)
            # Use permutation importance
            print(f"Computing permutation importance for {model_name}...")
            # Load test data for permutation importance
            X_test = pd.read_csv("AI_Thunderball_X_test.csv")
            y_test = pd.read_csv("AI_Thunderball_y_test.csv").values.ravel()
            
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
    
    # Load test data for visualization
    X_test = pd.read_csv("AI_Thunderball_X_test.csv")
    y_test = pd.read_csv("AI_Thunderball_y_test.csv").values.ravel()
    
    # Fig 1: Model Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Performance Metrics Comparison (Test Set)", fontsize=16, fontweight='bold')
    
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        values = []
        for r in results:
            if metric == "Accuracy":
                values.append(r['Test_Metrics']['Test_Accuracy'])
            elif metric == "Precision":
                values.append(r['Test_Metrics']['Test_Precision'])
            elif metric == "Recall":
                values.append(r['Test_Metrics']['Test_Recall'])
            elif metric == "F1 Score":
                values.append(r['Test_Metrics']['Test_F1'])
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
        bars = ax.bar(range(len(results)), values, color=colors)
        ax.set_ylabel(metric, fontweight='bold')
        ax.set_xticks(range(len(results)))
        ax.set_xticklabels([r['Model'] for r in results], rotation=45, ha='right')
        ax.set_ylim([0, 1])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig("01_model_performance_comparison.png", dpi=300, bbox_inches='tight')
    print(" Saved: 01_model_performance_comparison.png")
    plt.show()
    
    # Fig 2: Model Ranking by ROC-AUC
    fig, ax = plt.subplots(figsize=(10, 6))
    roc_auc_scores = [r['Test_Metrics']['Test_ROC_AUC'] for r in results]
    sorted_indices = np.argsort(roc_auc_scores)[::-1]
    sorted_names = [results[i]['Model'] for i in sorted_indices]
    sorted_scores = [roc_auc_scores[i] for i in sorted_indices]
    
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_names)))
    bars = ax.barh(sorted_names, sorted_scores, color=colors)
    ax.set_xlabel('ROC-AUC Score', fontweight='bold')
    ax.set_title('Model Ranking by ROC-AUC Score (Test Set)', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    
    for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
        ax.text(score - 0.05, bar.get_y() + bar.get_height()/2, f'{score:.4f}',
                ha='right', va='center', fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig("02_model_ranking_roc_auc.png", dpi=300, bbox_inches='tight')
    print(" Saved: 02_model_ranking_roc_auc.png")
    plt.show()
    
    # Figure 3: Confusion Matrices
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Confusion Matrices for All Models", fontsize=16, fontweight='bold')
    
    for idx, result in enumerate(results):
        ax = axes[idx // 2, idx % 2]
        cm = result['Test_Metrics']['Confusion_Matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    cbar_kws={'label': 'Count'}, annot_kws={'size': 12})
        ax.set_title(f"{result['Model']}", fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig("03_confusion_matrices.png", dpi=300, bbox_inches='tight')
    print(" Saved: 03_confusion_matrices.png")
    plt.show()
    
    # Fig 4: Correlation Heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    # Only use numerical columns for correlation
    numerical_data = X_train.select_dtypes(include=["int64", "float64"])
    correlation_matrix = numerical_data.corr()
    sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, square=True, 
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title("Feature Correlation Heatmap (Numerical Features)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("04_correlation_heatmap.png", dpi=300, bbox_inches='tight')
    print(" Saved: 04_correlation_heatmap.png")
    plt.show()
    
    # Fig 5: Top Features for All Models
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Top 10 Features by Importance/Coefficients (All Models)", fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    feature_names = get_feature_names(preprocessor)
    
    for plot_idx, result in enumerate(results):
        if plot_idx >= 4:
            break
        
        ax = axes[plot_idx]
        model_name = result['Model']
        
        try:
            top_features, ylabel = get_top_features(result, feature_names, top_n=10)
            
            if top_features:
                features, values = zip(*top_features)
                colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
                ax.barh(range(len(features)), values, color=colors)
                ax.set_yticks(range(len(features)))
                ax.set_yticklabels(features)
                ax.set_xlabel(ylabel, fontweight='bold')
                ax.set_title(f"{model_name}", fontweight='bold')
                ax.invert_yaxis()
                
                # Add value labels
                for i, val in enumerate(values):
                    ax.text(val, i, f' {val:.4f}', va='center', fontsize=9)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error computing importance\nfor {model_name}:\n{str(e)}', 
                    ha='center', va='center', fontsize=10, wrap=True)
            ax.set_title(f"{model_name}", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("05_top_features_importance.png", dpi=300, bbox_inches='tight')
    print(" Saved: 05_top_features_importance.png")
    plt.show()
    
    # Figure 6: Model Performance vs Confusion Matrix Metrics
    fig, ax = plt.subplots(figsize=(12, 6))
    model_names = [r['Model'] for r in results]
    x = np.arange(len(model_names))
    width = 0.2
    
    # Calculate True Positive Rate and False Positive Rate
    cm = [r['Test_Metrics']['Confusion_Matrix'] for r in results]
    tn = [c[0, 0] for c in cm]
    fp = [c[0, 1] for c in cm]
    fn = [c[1, 0] for c in cm]
    tp = [c[1, 1] for c in cm]
    
    # Calculate rates
    fpr = [fp[i] / (fp[i] + tn[i]) if (fp[i] + tn[i]) > 0 else 0 for i in range(len(results))]
    tpr = [tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0 for i in range(len(results))]
    specificity = [tn[i] / (tn[i] + fp[i]) if (tn[i] + fp[i]) > 0 else 0 for i in range(len(results))]
    sensitivity = [tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0 for i in range(len(results))]
    
    ax.bar(x - 1.5*width, sensitivity, width, label='Sensitivity (TPR)', alpha=0.8)
    ax.bar(x - 0.5*width, specificity, width, label='Specificity', alpha=0.8)
    ax.bar(x + 0.5*width, fpr, width, label='False Positive Rate', alpha=0.8)
    ax.bar(x + 1.5*width, [r['Test_Metrics']['Test_Accuracy'] for r in results], width, label='Accuracy', alpha=0.8)
    
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
    
    # Figure 7: ROC-AUC Curves
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for result in results:
        if result['y_prob'] is not None:
            y_prob = result['y_prob']
            fpr_curve, tpr_curve, _ = roc_curve(y_test, y_prob)
            roc_auc = result['Test_Metrics']['Test_ROC_AUC']
            ax.plot(fpr_curve, tpr_curve, label=f"{result['Model']} (AUC = {roc_auc:.4f})", linewidth=2)
    
    # Plot random classifier
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
    
    print("\n" + "="*50)
    print(" All original visualizations have been saved successfully!")
    print("="*50)

def main():
    print("="*60)
    print("AI THUNDERBALL: COMPLETE DATA PIPELINE WITH K-FOLD CV")
    print("="*60)
    
    # Step 1: Generate synthetic data
    print("\n Generating synthetic data...")
    generate_synthetic_data()
    
    # Step 2: Introduce anomalies
    print("\n Introducing anomalies...")
    introduce_anomalies()
    
    # Step 3: Clean data
    print("\n Cleaning data...")
    clean_data()
    
    # Step 4: Feature engineering
    print("\n Engineering features...")
    engineer_features()
    
    # Step 5: Train-test split
    print("\n Splitting data...")
    train_test_split_data()
    
    # Step 6: Load data for K-Fold
    print("\n Loading data for K-Fold Cross Validation...")
    X_train = pd.read_csv("AI_Thunderball_X_train.csv")
    X_test = pd.read_csv("AI_Thunderball_X_test.csv")
    y_train = pd.read_csv("AI_Thunderball_y_train.csv").values.ravel()
    y_test = pd.read_csv("AI_Thunderball_y_test.csv").values.ravel()
    
    print(f"  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    
    # Step 7: Train and evaluate models with K-Fold
    print("\n Training models with K-Fold Cross Validation...")
    results, preprocessor = train_and_evaluate_models_kfold(X_train, y_train, X_test, y_test, n_splits=5)
    
    # Step 8: Create visualizations
    create_original_visualizations(results, X_train, preprocessor)
    visualize_kfold_results(results, X_train)
    

    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated Files:")
    print("1. AI_Thunderball_Dataset.csv - Original synthetic data")
    print("2. AI_Thunderball_RawDataset.csv - Data with anomalies")
    print("3. AI_Thunderball_CleanedDataset.csv - Cleaned data")
    print("4. AI_Thunderball_FeatureEngineeredDataset.csv - Data with engineered features")
    print("5. AI_Thunderball_X_train.csv / X_test.csv - Train/test features")
    print("6. AI_Thunderball_y_train.csv / y_test.csv - Train/test labels")
    print("7. 01-07_*.png - Original model evaluation visualizations")
    print("8. 08-12_*.png - K-Fold Cross Validation visualizations")

    for result in results:
        print(f"\n{result['Model']}:")
        print(f"  CV Accuracy: {result['CV_Metrics']['CV_accuracy_mean']:.4f} (±{result['CV_Metrics']['CV_accuracy_std']:.4f})")
        print(f"  Test Accuracy: {result['Test_Metrics']['Test_Accuracy']:.4f}")
        print(f"  CV ROC-AUC: {result['CV_Metrics']['CV_roc_auc_mean']:.4f} (±{result['CV_Metrics']['CV_roc_auc_std']:.4f})")
        print(f"  Test ROC-AUC: {result['Test_Metrics']['Test_ROC_AUC']:.4f}")

main()
