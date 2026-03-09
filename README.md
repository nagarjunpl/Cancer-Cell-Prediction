# Cancer Cell Prediction Using Machine Learning

An end-to-end Machine Learning ecosystem designed to predict cancer risk through synthetic data generation, advanced feature engineering, and high-performance model deployment.

---

# 📖 Project Overview
**AI Thunderball** is a clinical decision-support framework that bridges the gap between raw medical data and actionable insights. Developed as part of the **XPro Project in collaboration with Federation University**, this system automates the path from data synthesis to a production-ready FastAPI service.

---

# ⚠️ Problem Statement
Early detection of cancer is often hindered by fragmented data, privacy restrictions (HIPAA/GDPR) preventing the sharing of real patient records, and the high dimensionality of clinical biomarkers.

There is a critical need for a system that can:

- Generate privacy-preserving, realistic medical data for research.
- Identify non-linear correlations between lifestyle habits and clinical symptoms.
- Provide a scalable interface for healthcare providers to assess patient risk instantly.

---

# 🎯 Project Objectives

### Data Synthesis
Use **Generative AI (CTGAN)** to create high-fidelity, anonymized patient datasets.

### Anomaly Simulation
Test the pipeline by introducing and cleaning synthetic null values, outliers, and duplicates.

### Clinical Scoring
Engineer domain-specific features such as:

- Clinical Risk Score
- Lifestyle Risk Score

### Model Optimization
Use **PCA for dimensionality reduction** and **Hyperparameter Tuning** for maximum ROC-AUC.

### Deployment
Develop a **RESTful API** capable of both single and batch predictions.

---

# 💡 The Solution: AI-Powered Pipeline
The project utilizes a **multi-stage pipeline** ensuring data integrity and model reliability.

By combining **Gradient Boosting with advanced preprocessing**, the system achieves high sensitivity, which is critical in oncology to minimize **False Negatives**.

---

# 🛠️ Technologies Used

**Language**
- Python 3.x

**Generative AI**
- CTGAN (Conditional Tabular GAN)

**Data Processing**
- Pandas
- NumPy

**Machine Learning**
- Scikit-learn
- Joblib

**Visualization**
- Matplotlib
- Seaborn

**API Framework**
- FastAPI
- Pydantic
- Uvicorn

---

# 🧠 Machine Learning Algorithms

The system benchmarks four models:

### Gradient Boosting Classifier (Primary Model)
Chosen for handling complex non-linear feature relationships effectively.

### Random Forest
Used for ensemble stability and feature importance analysis.

### Support Vector Machine (SVM)
Optimized with RBF kernel for high-dimensional classification.

### Logistic Regression
Used as a baseline model with **L1/L2 regularization analysis**.

---

# 🔄 Project Workflow

## 1. Data Generation & Cleaning

**Data Synthesis**
- 1300 synthetic patient records generated using CTGAN

**Anomaly Injection**
- Introduced 3% NULL values and extreme outliers (e.g., Age = 150)

**Data Cleaning**
- Median and Mode imputation
- Z-score outlier removal

---

## 2. Feature Engineering

**Medical Metrics**
- BMI
- Platelet-to-Lymphocyte Ratio

**Risk Scoring**
Custom scoring based on symptoms such as:

- Lump presence
- Unexplained weight loss
- Family history

These are aggregated into a **Clinical Risk Score**.

---

## 3. Model Analysis

**PCA (Principal Component Analysis)**
- Reduces feature dimensions
- Retains ~95% data variance

**Validation**
- 5-Fold Stratified Cross Validation to ensure strong generalization.

---

## 4. Deployment

The system deploys the trained model through a **FastAPI server**.

Loaded models:
- `api_best_model.pkl`
- `api_preprocessor.pkl`

API Endpoints:

| Endpoint | Description |
|--------|-------------|
| `/predict` | Predict cancer risk for a single patient |
| `/predict/batch` | Predict risk for multiple patients |
| `/models` | View model performance metadata |
| `/health` | Check API server status |

---

# 📊 Results & Performance

The **Gradient Boosting model** performed best.

**Key Metrics**

- High **ROC-AUC Score**
- High **Sensitivity (Recall)** to detect potential cancer cases
- PCA reduced **70% of features** while maintaining **>90% accuracy**

