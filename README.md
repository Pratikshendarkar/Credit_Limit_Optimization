# Credit Limit Optimization Model

A machine learning pipeline designed to predict credit card default risk and identify customers eligible for credit limit increases using advanced ensemble methods.

## 📋 Project Overview

This project implements a dual-model approach to optimize credit card portfolio management through:
1. **Default Risk Prediction** - Identify high-risk customers likely to default
2. **Credit Limit Increase Prediction** - Identify qualified customers for limit increases

The solution combines Random Forest and XGBoost models with ensemble averaging for robust predictions across a customer base of 7,298 accounts.

---

## 🎯 Business Objectives

- **Risk Mitigation**: Proactively identify customers at high default risk to enable early intervention
- **Revenue Optimization**: Flag customers eligible for credit limit increases to drive incremental revenue
- **Customer Segmentation**: Create actionable customer segments based on risk and eligibility profiles
- **Operational Efficiency**: Provide data-driven recommendations for credit management decisions

---

## 📊 Dataset

**Size**: 7,298 customers  
**Features**: 14 original features engineered to 26 total features  
**Data Quality**: No missing values, no duplicates

### Original Features:
- **Demographics**: Customer_Age, Gender, Dependent_count, Education_Level, Marital_Status
- **Financial Profile**: Income_Category, Card_Category, Credit_Limit, Avg_Utilization_Ratio
- **Behavioral**: Months_on_book, Pay_on_time, Attrition_Flag
- **Target Variables**: Default Risk, Credit Limit Increase Eligibility

### Target Variable Distribution:
| Target | Label | Count | Percentage |
|--------|-------|-------|-----------|
| Default Risk | Non-Default | 6,247 | 85.60% |
| Default Risk | Default | 1,051 | 14.40% |
| Increase Eligibility | Not Eligible | 6,270 | 85.91% |
| Increase Eligibility | Eligible | 1,028 | 14.09% |

---

## 🔧 Methodology

### 1. Feature Engineering
Created 17 new features to capture interaction effects and domain-specific insights:

| Feature Category | New Features |
|------------------|-------------|
| **Risk Metrics** | Utilization_Age_Risk, Credit_Efficiency, Payment_Utilization_Risk, High_Utilization, Low_Utilization |
| **Composite Scores** | Payment_Score, Age_Income_Score, Tenure_Income_Score |
| **Tenure & Age** | Tenure_Years, Tenure_Squared, Age_Squared |
| **Categorical Encoding** | Income_Level, Has_Dependents, Premium_Card |
| **Ratio Features** | Credit_Income_Ratio |

**Rationale**: Engineered features capture non-linear relationships and interaction effects that improve model interpretability and predictive power.

### 2. Data Preprocessing
- **Train-Test Split**: 80-20 split (5,838 training / 1,460 test samples)
- **Scaling**: StandardScaler applied to normalize feature distributions
- **Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique) applied to Default Risk training data
  - Original ratio: 14.4% default
  - After SMOTE: 50-50 balanced for fair model training

### 3. Model Development

#### Default Risk Prediction Models:

**Random Forest**
```
Accuracy:  87.81%
F1-Score:  0.6009
ROC-AUC:   0.8700
Precision: 94% (Non-Default), 57% (Default)
Recall:    92% (Non-Default), 64% (Default)
```

**XGBoost** (Selected as Primary Model)
```
Accuracy:  88.90%
F1-Score:  0.5846
ROC-AUC:   0.8661
Precision: 93% (Non-Default), 63% (Default)
Recall:    95% (Non-Default), 54% (Default)
```

#### Credit Limit Increase Prediction Models:

**Random Forest**
```
Accuracy:  83.29%
F1-Score:  0.0543
ROC-AUC:   0.5481
Precision: 86% (Not Eligible), 13% (Eligible)
Recall:    96% (Not Eligible), 3% (Eligible)
```

**XGBoost** (Selected as Primary Model)
```
Accuracy:  79.79%
F1-Score:  0.1194
ROC-AUC:   0.5428
Precision: 86% (Not Eligible), 16% (Eligible)
Recall:    91% (Not Eligible), 10% (Eligible)
```

### 4. Ensemble Approach
Final predictions use ensemble averaging:
```
Final_Score = (Random_Forest_Probability + XGBoost_Probability) / 2
```

This combines the strengths of both models for more robust and stable predictions.

---

## 📈 Key Results

### Default Risk Segmentation:

| Risk Category | Count | Percentage |
|---------------|-------|-----------|
| **High Risk** (>0.6) | Customers flagged for immediate intervention | ~10-12% |
| **Medium Risk** (0.3-0.6) | Customers requiring monitoring | ~20-25% |
| **Low Risk** (<0.3) | Healthy customers | ~65-70% |

### Credit Limit Increase Eligibility:

| Eligibility Level | Recommendation | Percentage |
|------------------|-----------------|-----------|
| **Strong** (>0.7) | Recommend immediate increase | ~8-12% |
| **Moderate** (0.4-0.7) | Consider increase based on risk | ~15-25% |
| **Not Recommended** (<0.4) | Hold or decline increase | ~65-75% |

**Key Finding**: The low AUC scores (0.54-0.55) for increase eligibility prediction indicate that this is a challenging classification problem. The majority of customers (>65%) fall into the "Not Recommended" category, suggesting that credit limit increases are selective and driven by specific behavioral and financial criteria captured by the engineered features.

---

## 🔍 Feature Importance Analysis

### Default Risk Prediction - Top Features:

**Random Forest Model:**
1. Avg_Utilization_Ratio (18.5%)
2. Credit_Efficiency (13.0%)
3. Utilization_Age_Risk (8.1%)
4. Credit_Limit (5.2%)
5. High_Utilization (5.1%)

**XGBoost Model:**
1. High_Utilization (18.3%)
2. Pay_on_Time (12.2%)
3. Income_Level (11.8%)
4. Payment_Score (11.3%)
5. Avg_Utilization_Ratio (7.0%)

### Credit Limit Increase Prediction - Top Features:

**Both Models Highlight:**
1. Credit_Income_Ratio (primary driver)
2. Months_on_book / Tenure features
3. Payment_Score and Pay_on_Time
4. Age-based features
5. Income_Category

**Insight**: Utilization patterns and payment behavior are critical predictors of default risk, while tenure, income, and responsible payment history drive credit limit increase eligibility.

---

## 💡 Key Insights

### 1. Utilization is a Strong Default Predictor
High credit utilization ratios, especially when combined with age and income, show strong correlation with default risk. Customers with utilization >50% warrant closer monitoring.

### 2. Payment History Matters
Customers with consistent on-time payments demonstrate lower default risk regardless of other factors. This is a reliable behavioral signal.

### 3. Income-to-Credit Ratio is Decisive
The Credit_Income_Ratio is the strongest predictor for credit limit increases. Customers with healthy income relative to existing credit limits are better candidates for increases.

### 4. Tenure Indicates Loyalty
Longer customer tenure (years_on_book) correlates with both lower default risk and higher eligibility for increases, suggesting improved customer stability over time.

### 5. Model Disagreement Areas
Random Forest and XGBoost show different feature importance rankings, suggesting complementary insights. Ensemble approach leverages both perspectives.

---

## 📁 Output Files Generated

### Predictions & Segmentation:
- **Customer_Predictions.csv** - All 7,298 customers with probability scores and recommendations
- **High_Default_Risk_Customers.csv** - Top 20 customers flagged for risk intervention
- **High_Increase_Eligible_Customers.csv** - Top 20 customers recommended for limit increases

### Visualizations:
- **01_Feature_Importance_Comparison.png** - Feature rankings across models
- **02_Model_Performance_Metrics.png** - Accuracy, F1, ROC-AUC comparison
- **03_Confusion_Matrices.png** - Prediction accuracy by class
- **04_ROC_Curves.png** - Model discrimination ability
- **05_Feature_Correlation_Heatmap.png** - Inter-feature relationships
- **06_Customer_Segmentation.png** - Distribution across risk/eligibility segments
- **07_Age_Income_Distribution.png** - Demographic patterns in predictions
- **08_Probability_Distributions.png** - Prediction confidence distributions

<img width="1164" height="884" alt="image" src="https://github.com/user-attachments/assets/acae4fa9-581e-43e2-8318-a79748b3fbb0" />
<img width="1173" height="580" alt="image" src="https://github.com/user-attachments/assets/4d61dfc2-9072-4849-aa05-b532a645174c" />
<img width="1157" height="836" alt="image" src="https://github.com/user-attachments/assets/fc626491-e0f4-4983-9699-c874e83ae4a1" />



## 🚀 Implementation Guide

### Prerequisites:
```python
pandas, scikit-learn, xgboost, matplotlib, seaborn, numpy
```

### Quick Start:

1. **Load Data**: Mount Google Drive and load customer dataset
2. **Run Pipeline**: Execute notebook cells sequentially
3. **Review Results**: Check CSV outputs and visualizations
4. **Deploy Predictions**: Use Customer_Predictions.csv for operational decisions

### Using Predictions:

```python
# Load predictions
predictions = pd.read_csv('Customer_Predictions.csv')

# Filter high-risk customers
high_risk = predictions[predictions['Default_Risk_Ensemble'] > 0.6]

# Filter increase-eligible customers
eligible = predictions[predictions['Increase_Recommendation'] == 'Strong']

# Export for campaign
eligible.to_csv('credit_increase_campaign.csv')
```

---

## ⚠️ Model Limitations & Considerations

1. **Class Imbalance**: Default and increase eligibility are minority classes (~14%). While SMOTE addresses training imbalance, real-world distribution should be considered for threshold tuning.

2. **Increase Eligibility Challenge**: Lower AUC (0.54-0.55) suggests increase eligibility is harder to predict. Consider qualitative business rules as supplementary criteria.

3. **Threshold Sensitivity**: Predictions are probability scores. Operational thresholds (0.3, 0.6 for default; 0.4, 0.7 for increase) can be tuned based on business cost/benefit analysis.

4. **Recency Bias**: Model trained on historical data. Customer behavior may change; periodic model retraining recommended (quarterly or semi-annually).

5. **External Factors**: Model doesn't capture macroeconomic conditions, market rates, or competitive pressures that may affect default risk.

---

## 📊 Performance Summary

| Metric | Default Risk | Increase Eligibility |
|--------|--------------|-------------------|
| **Best Model** | XGBoost | XGBoost |
| **Accuracy** | 88.90% | 79.79% |
| **ROC-AUC** | 0.8661 | 0.5428 |
| **F1-Score** | 0.5846 | 0.1194 |
| **Use Case Strength** | Risk identification | Revenue opportunity |

---




---

## 📚 References

- **SMOTE**: Chawla et al., 2002. "SMOTE: Synthetic Minority Over-sampling Technique"
- **Random Forest**: Breiman, 2001. "Random Forests"
- **XGBoost**: Chen & Guestrin, 2016. "XGBoost: A Scalable Tree Boosting System"
- **ROC-AUC**: Fawcett, 2006. "An Introduction to ROC Analysis"
