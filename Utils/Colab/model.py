import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from interpret.glassbox import ExplainableBoostingClassifier

def load_dataset(ide="manual", file_path: str = None):
  if ide == "local":
    df = pd.read_csv("G:\My Drive\ITS\Tugas\Semester_5\Biomedical Engineering\Final Project Req\Dataset.csv")
  elif ide == "colab":
    from google.colab import drive
    drive.mount('/content/drive')
    if file_path is None:
        file_path = "/content/drive/MyDrive/ITS/Tugas/Semester_5/Biomedical Engineering/Final Project Req/Dataset.csv"
    df = pd.read_csv(file_path)
  else:
    if file_path is None:
        raise ValueError("file_path must be provided for manual ide.")
    df = pd.read_csv(file_path)

  return df

def preprocess_data(X_train, X_val, X_test=None):
    """
    Melakukan Standard Scaling pada data.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def train_lasso(X_train, y_train, C=1.0, random_state=42):
    """
    Melatih model White-Box (Lasso Logistic Regression).
    """
    print(f"Training Lasso (White-Box) dengan C={C}...")
    model = LogisticRegression(
        penalty='l1',
        C=C,
        solver='liblinear',
        multi_class='ovr',
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Melatih model Black-Box (Random Forest).
    """
    print(f"Training Random Forest (Black-Box)...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X, y, class_names=None, model_name="Model"):
    """
    Menampilkan metrik evaluasi standar.
    """
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"--- Evaluasi {model_name} ---")
    print(f"Akurasi: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=class_names))
    return acc

def train_ebm(X_train, y_train, random_state=42):
    """
    Melatih model Glass-Box (Explainable Boosting Machine).
    EBM secara otomatis menangani interaksi antar fitur.
    """
    print(f"Training EBM (Glass-Box)...")
    # n_jobs=-1 menggunakan semua core CPU
    ebm = ExplainableBoostingClassifier(random_state=random_state, n_jobs=-1)
    ebm.fit(X_train, y_train)
    print("Training Selesai.")
    return ebm