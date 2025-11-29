# Breast Cancer PAM50 Subtype Classification with Explainable AI

A comprehensive machine learning pipeline for classifying breast cancer molecular subtypes (PAM50) using gene expression data. This project compares interpretable (white-box) and non-interpretable (black-box) models, then applies Explainable AI techniques (SHAP, LIME, DeepSHAP) to identify key genes associated with each molecular subtype.

## Overview

Breast cancer is a heterogeneous disease with distinct molecular subtypes that have different prognoses and treatment responses. The PAM50 gene signature classifies breast tumors into five intrinsic subtypes:
- **Luminal A (LumA)** - Best prognosis, hormone receptor positive
- **Luminal B (LumB)** - Hormone receptor positive, higher proliferation
- **HER2-enriched** - HER2 overexpression
- **Basal-like** - Triple-negative, aggressive
- **Normal-like** - Similar to normal breast tissue

This project uses RNA-seq gene expression data from TCGA (The Cancer Genome Atlas) to train classification models and explain their predictions.

## Project Structure

```
breast-cancer-pam50-xai-classification/
├── Black-Box Models/           # Non-interpretable models
│   ├── RandomForest.ipynb      # Random Forest classifier
│   ├── SVM.ipynb               # Support Vector Machine
│   ├── NeuralNetwork.ipynb     # Deep Neural Network
│   ├── XGBoost.ipynb           # XGBoost classifier
│   └── Saved Model/            # Trained model artifacts
├── White-Box Model/            # Interpretable models
│   ├── LogisticRegression.ipynb  # L1-regularized Logistic Regression
│   ├── ElasticNet.ipynb        # ElasticNet Logistic Regression
│   └── Saved Model/            # Trained model artifacts
├── Utils/                      # Utility modules
│   ├── model.py                # Model training functions
│   ├── explainer.py            # XAI explanation functions
│   └── visualization.py        # Plotting functions
├── EDA.ipynb                   # Exploratory Data Analysis
├── requirements.txt            # Python dependencies
└── setup.py                    # Package setup
```

## Models Implemented

### White-Box (Interpretable) Models
- **Lasso Logistic Regression**: L1-regularized for automatic feature selection
- **ElasticNet Logistic Regression**: Combined L1+L2 regularization

### Black-Box Models
- **Random Forest**: Ensemble of decision trees
- **Support Vector Machine (SVM)**: RBF kernel with probability estimates
- **Neural Network**: Multi-layer perceptron with BatchNorm and Dropout
- **XGBoost**: Gradient boosting with optimized hyperparameters

## Explainability Methods

- **SHAP (SHapley Additive exPlanations)**
  - TreeExplainer for tree-based models
  - LinearExplainer for linear models
  - KernelExplainer for model-agnostic explanations
  - DeepExplainer/GradientExplainer for neural networks
- **Integrated Gradients**: Attribution method for neural networks
- **Permutation Importance**: Model-agnostic feature importance
- **EBM (Explainable Boosting Machine)**: Inherently interpretable model

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/breast-cancer-pam50-xai-classification.git
cd breast-cancer-pam50-xai-classification

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. **Prepare the dataset**: Place your TCGA breast cancer gene expression dataset in CSV format
2. **Run EDA**: Open `EDA.ipynb` to explore the dataset
3. **Train models**: Run notebooks in `White-Box Model/` or `Black-Box Models/`
4. **Analyze explanations**: Each notebook includes SHAP analysis sections

### Quick Start Example

```python
from Utils import model, explainer, visualization as viz

# Load and preprocess data
df = model.load_dataset(file_path="path/to/dataset.csv")
X_train_s, X_val_s, _, scaler = model.preprocess_data(X_train, X_val)

# Train a model
rf_model = model.train_random_forest(X_train_s, y_train)
model.evaluate(rf_model, X_val_s, y_val, class_names, "Random Forest")

# Compute SHAP explanations
exp, shap_values = explainer.compute_shap_tree(rf_model, X_val_s)

# Visualize feature importance
viz.plot_beeswarm(shap_values, X_val_s, feature_names, class_names)
```

## Dataset

The project uses gene expression data from TCGA-BRCA with:
- **1,218 samples** (956 after removing missing labels)
- **20,530 gene features**
- **5 PAM50 subtypes**: LumA (434), LumB (194), Basal (142), Normal (119), Her2 (67)

## Results

Model performance on validation set (typical results):
| Model | Accuracy |
|-------|----------|
| Random Forest | ~90% |
| SVM (RBF) | ~90% |
| XGBoost | ~91% |
| Neural Network | ~88% |
| Logistic Regression (Lasso) | ~88% |

## Key Findings

Through SHAP analysis, we identified genes most important for each subtype classification:
- **Basal**: FOXC1, CXorf61, HORMAD1
- **HER2**: ERBB2, GRB7, STARD3
- **Luminal A**: NAT1, SLC40A1, EPS8L3
- **Luminal B**: TEX19, TIMELESS, FADS1
- **Normal**: OXTR, GLB1L3, RNF186

## Requirements

- Python 3.8+
- scikit-learn
- xgboost
- shap
- torch (for Neural Network)
- interpret (for EBM)
- pandas, numpy, matplotlib, seaborn

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TCGA Research Network for the breast cancer genomic data
- SHAP library by Scott Lundberg
- InterpretML by Microsoft Research
