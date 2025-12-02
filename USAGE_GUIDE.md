# Quick Start Guide: Using the Enhanced Validation Tools

This guide demonstrates how to use the new data quality, validation, and biological validation modules.

## Table of Contents
1. Data Quality Assessment
2. Cross-Validation
3. Biological Validation
4. Complete Example

## Installation

First, ensure you have all required packages:

```bash
pip install -r requirements.txt
```

## 1. Data Quality Assessment

### Basic Quality Check

```python
from Utils.data_quality import validate_data_quality, plot_data_quality_summary
import pandas as pd
import numpy as np

# Load your data
X_train = ...  # Your feature matrix
y_train = ...  # Your labels
feature_names = ...  # List of gene names

# Run comprehensive quality check
report = validate_data_quality(X_train, y_train, feature_names, verbose=True)

# Visualize quality metrics
plot_data_quality_summary(X_train, y_train, feature_names)
```

### Outlier Detection

```python
from Utils.data_quality import detect_outliers

# Detect outliers using Z-score method
outlier_indices, outlier_scores = detect_outliers(X_train, threshold=3, method='zscore')

print(f"Found {len(outlier_indices)} outlier samples")
print(f"Outlier indices: {outlier_indices}")

# Option to remove outliers
X_train_clean = np.delete(X_train, outlier_indices, axis=0)
y_train_clean = np.delete(y_train, outlier_indices, axis=0)
```

### Feature Filtering

```python
from Utils.data_quality import filter_low_variance_genes

# Remove low-variance genes
X_filtered, selected_indices, selected_names = filter_low_variance_genes(
    X_train, 
    feature_names, 
    threshold=0.01, 
    method='variance'
)

print(f"Retained {len(selected_indices)} high-variance genes")
```

### Batch Effect Detection

```python
from Utils.data_quality import check_batch_effects

# Visualize potential batch effects
X_pca, explained_variance = check_batch_effects(
    X_train, 
    labels=y_train,
    n_components=2,
    plot=True
)
```

### Compare Normalization Methods

```python
from Utils.data_quality import compare_normalization_methods

# Compare different scaling methods
norm_results = compare_normalization_methods(
    X_train, 
    X_val,
    method_names=['standard', 'robust', 'quantile']
)

# Use the best method
X_train_scaled = norm_results['robust']['train']
X_val_scaled = norm_results['robust']['val']
scaler = norm_results['robust']['scaler']
```

## 2. Cross-Validation

### Basic Cross-Validation

```python
from Utils.validation import cross_validate_model, print_cv_results
from sklearn.ensemble import RandomForestClassifier

# Initialize model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform cross-validation
cv_results = cross_validate_model(
    model, X_train_scaled, y_train,
    cv=5,
    scoring={
        'accuracy': 'accuracy',
        'f1_macro': 'f1_macro',
        'precision_macro': 'precision_macro'
    }
)

# Print results
print_cv_results(cv_results, model_name="Random Forest")
```

### Compare Multiple Models

```python
from Utils.validation import compare_models_cv, plot_cv_comparison
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb

# Define models
models_dict = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42)
}

# Compare models
comparison_df = compare_models_cv(models_dict, X_train_scaled, y_train, cv=5)
print(comparison_df)

# Store CV results for plotting
cv_results_dict = {}
for name, model in models_dict.items():
    cv_res = cross_validate_model(model, X_train_scaled, y_train, cv=5)
    cv_results_dict[name] = cv_res

# Plot comparison
plot_cv_comparison(cv_results_dict, metric='accuracy')
```

### Evaluation with Confidence Intervals

```python
from Utils.validation import evaluate_with_confidence, print_evaluation_results

# Train model
model.fit(X_train_scaled, y_train)

# Evaluate with bootstrap confidence intervals
results = evaluate_with_confidence(
    model, 
    X_val_scaled, 
    y_val,
    class_names=['LumA', 'LumB', 'Her2', 'Basal', 'Normal'],
    n_bootstrap=1000
)

# Print results
print_evaluation_results(results, model_name="Random Forest")
```

### Statistical Model Comparison

```python
from Utils.validation import statistical_comparison_test

# Get CV scores for two models
scores_rf = cv_results_rf['metrics']['accuracy']['test_scores']
scores_xgb = cv_results_xgb['metrics']['accuracy']['test_scores']

# Statistical test
test_result = statistical_comparison_test(scores_rf, scores_xgb, test='paired_t')

print(f"Test: {test_result['test']}")
print(f"p-value: {test_result['p_value']:.4f}")
print(f"Significant difference: {test_result['significant']}")
```

## 3. Biological Validation

### Initialize Validator

```python
from Utils.biological_validation import BiologicalValidator

validator = BiologicalValidator()
```

### Check PAM50 Overlap

```python
# Your identified top genes
top_genes = ['ERBB2', 'GRB7', 'STARD3', 'FOXC1', 'NAT1', 'ESR1']

# Check overlap with PAM50
overlap = validator.check_pam50_overlap(top_genes)

print(f"PAM50 overlap: {overlap['overlap_count']}/{overlap['total_input_genes']}")
print(f"PAM50 genes: {overlap['overlap_genes']}")
print(f"Novel genes: {overlap['novel_genes']}")
```

### Validate Subtype-Specific Markers

```python
# Validate genes for a specific subtype
her2_genes = ['ERBB2', 'GRB7', 'STARD3', 'PGAP3']

validation = validator.validate_subtype_markers(her2_genes, 'HER2-enriched')

print(f"Validated markers: {validation['validated_count']} "
      f"({validation['validated_percentage']:.1f}%)")
print(f"Validated: {validation['validated_genes']}")
```

### Check Cancer Gene Relevance

```python
# Check if genes are known oncogenes or tumor suppressors
relevance = validator.check_cancer_relevance(top_genes)

print(f"Oncogenes: {relevance['oncogenes']}")
print(f"Tumor suppressors: {relevance['tumor_suppressors']}")
print(f"Novel genes: {relevance['novel_genes']}")
```

### Comprehensive Gene Validation

```python
# Validate all genes in your list
validation_df = validator.validate_gene_list(top_genes, verbose=True)

print(validation_df)

# Filter high-confidence genes
high_confidence = validation_df[
    validation_df['biological_evidence'].str.contains('Strong')
]
print(f"\nHigh-confidence genes: {len(high_confidence)}")
print(high_confidence['gene'].tolist())
```

### Generate Validation Report

```python
# Validate genes for all subtypes
top_genes_by_subtype = {
    'Basal': ['FOXC1', 'KRT5', 'KRT17', 'EGFR'],
    'Her2': ['ERBB2', 'GRB7', 'STARD3', 'PGAP3'],
    'LumA': ['ESR1', 'PGR', 'NAT1', 'BCL2'],
    'LumB': ['ESR1', 'MKI67', 'CCNB1', 'MYBL2'],
    'Normal': ['SFRP1', 'CXXC5', 'MIA']
}

# Generate comprehensive report
report = validator.generate_validation_report(
    top_genes_by_subtype,
    save_path='biological_validation_report.txt'
)

print(report)
```

### Get Pathway Analysis Suggestions

```python
from Utils.biological_validation import suggest_pathway_analysis_tools

# Print recommendations for pathway analysis
suggest_pathway_analysis_tools()
```

### Validation Checklist

```python
from Utils.biological_validation import create_validation_checklist

# Get validation checklist
checklist = create_validation_checklist()
print(checklist)
```

## 4. Complete Example: End-to-End Pipeline

Here's a complete example combining all tools:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Import custom modules
from Utils.data_quality import (validate_data_quality, detect_outliers,
                                filter_low_variance_genes, check_batch_effects,
                                compare_normalization_methods)
from Utils.validation import (cross_validate_model, print_cv_results,
                              evaluate_with_confidence, print_evaluation_results)
from Utils.biological_validation import BiologicalValidator

# ===========================
# 1. LOAD AND PREPARE DATA
# ===========================

# Load your dataset
df = pd.read_csv("path/to/your/dataset.csv")

# Separate features and labels
feature_cols = [c for c in df.columns if c != 'PAM50']
X = df[feature_cols].values
y = df['PAM50'].values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# ===========================
# 2. DATA QUALITY ASSESSMENT
# ===========================

print("\n" + "=" * 70)
print("STEP 1: DATA QUALITY ASSESSMENT")
print("=" * 70)

# Quality check
quality_report = validate_data_quality(X_train, y_train, feature_cols, verbose=True)

# Detect outliers
outlier_indices, outlier_scores = detect_outliers(X_train, threshold=3)
print(f"\nDetected {len(outlier_indices)} outlier samples")

# Option: Remove outliers (be cautious!)
if len(outlier_indices) < len(X_train) * 0.05:  # Only if < 5% outliers
    X_train = np.delete(X_train, outlier_indices, axis=0)
    y_train = np.delete(y_train, outlier_indices)
    print(f"Removed outliers. New training size: {len(X_train)}")

# Filter low-variance genes
X_train_filt, selected_idx, selected_names = filter_low_variance_genes(
    X_train, feature_cols, threshold=0.01
)
X_val_filt = X_val[:, selected_idx]
X_test_filt = X_test[:, selected_idx]

# Check batch effects
X_pca, variance = check_batch_effects(X_train_filt, y_train, plot=True)

# ===========================
# 3. NORMALIZATION
# ===========================

print("\n" + "=" * 70)
print("STEP 2: DATA NORMALIZATION")
print("=" * 70)

# Compare normalization methods
norm_results = compare_normalization_methods(X_train_filt, X_val_filt)

# Use robust scaler (less sensitive to outliers)
scaler = norm_results['robust']['scaler']
X_train_scaled = norm_results['robust']['train']
X_val_scaled = norm_results['robust']['val']
X_test_scaled = scaler.transform(X_test_filt)

# ===========================
# 4. MODEL TRAINING & VALIDATION
# ===========================

print("\n" + "=" * 70)
print("STEP 3: MODEL TRAINING & CROSS-VALIDATION")
print("=" * 70)

# Initialize model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Cross-validation
cv_results = cross_validate_model(
    model, X_train_scaled, y_train,
    cv=5,
    scoring={
        'accuracy': 'accuracy',
        'f1_macro': 'f1_macro',
        'precision_macro': 'precision_macro'
    }
)
print_cv_results(cv_results, model_name="Random Forest")

# Train on full training set
model.fit(X_train_scaled, y_train)

# Evaluate on validation set with confidence intervals
print("\n" + "=" * 70)
print("STEP 4: VALIDATION SET EVALUATION")
print("=" * 70)

eval_results = evaluate_with_confidence(
    model, X_val_scaled, y_val,
    class_names=class_names,
    n_bootstrap=1000
)
print_evaluation_results(eval_results, model_name="Random Forest")

# ===========================
# 5. BIOLOGICAL VALIDATION
# ===========================

print("\n" + "=" * 70)
print("STEP 5: BIOLOGICAL VALIDATION")
print("=" * 70)

# Get feature importances
importances = model.feature_importances_
top_n = 20
top_indices = np.argsort(importances)[::-1][:top_n]
top_genes = [selected_names[i] for i in top_indices]

print(f"Top {top_n} genes by importance:")
for i, (idx, gene) in enumerate(zip(top_indices, top_genes)):
    print(f"{i+1}. {gene}: {importances[idx]:.4f}")

# Initialize biological validator
validator = BiologicalValidator()

# Validate genes
validation_df = validator.validate_gene_list(top_genes, verbose=True)
print("\n", validation_df)

# Check PAM50 overlap
overlap = validator.check_pam50_overlap(top_genes)
print(f"\nPAM50 overlap: {overlap['overlap_count']}/{overlap['total_input_genes']} "
      f"({overlap['overlap_percentage']:.1f}%)")

# If you have top genes per subtype, generate full report
# top_genes_by_subtype = {...}  # Extract from SHAP analysis
# report = validator.generate_validation_report(
#     top_genes_by_subtype,
#     save_path='validation_report.txt'
# )

# ===========================
# 6. FINAL TEST SET EVALUATION
# ===========================

print("\n" + "=" * 70)
print("STEP 6: FINAL TEST SET EVALUATION")
print("=" * 70)

test_results = evaluate_with_confidence(
    model, X_test_scaled, y_test,
    class_names=class_names,
    n_bootstrap=1000
)
print_evaluation_results(test_results, model_name="Random Forest (Test Set)")

print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)
```

## Next Steps

After running this pipeline:

1. **Generate SHAP explanations** for your trained model
2. **Extract top genes per subtype** from SHAP analysis
3. **Run pathway enrichment** using Enrichr or GSEA
4. **Validate on external dataset** (METABRIC, GEO)
5. **Consult with clinicians** about biological plausibility
6. **Document findings** in your research paper

## Additional Resources

- See `ANALYSIS_REPORT.md` for comprehensive recommendations
- Check `biological_validation_report.txt` for validation results
- Use pathway analysis tools (Enrichr, GSEA) for functional validation
- Validate findings against cBioPortal and PubMed literature

## Troubleshooting

### Issue: "Too many outliers detected"
- Solution: Adjust threshold or use 'iqr' method instead of 'zscore'

### Issue: "Low PAM50 overlap"
- Solution: This is okay! You may be discovering novel markers
- Action: Perform pathway enrichment to validate biological relevance

### Issue: "High variance in CV scores"
- Solution: Use more CV folds or nested CV
- Check for data quality issues or class imbalance

### Issue: "Model overfitting"
- Solution: Increase regularization, reduce features, or use simpler model
- Check overfit_gap in CV results

For more help, refer to the comprehensive analysis report or raise an issue on GitHub.
