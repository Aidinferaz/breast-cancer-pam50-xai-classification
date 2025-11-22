import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from interpret.glassbox import ExplainableBoostingClassifier  # Pastikan library 'interpret' terinstall

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
    Melakukan Standard Scaling pada data fitur.
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


def train_svm(X_train, y_train, C=1.0, kernel='rbf', gamma='scale', random_state=42):
    """
    Melatih model SVM (Support Vector Machine).
    probability=True agar kompatibel dengan SHAP.
    """
    print(f"Training SVM ({kernel} kernel)...")
    model = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        probability=True,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


# --- BAGIAN YANG DIPERBAIKI ---
def train_ebm(X_train, y_train, interactions=0, random_state=42):
    """
    Melatih model Glass-Box (Explainable Boosting Machine).
    Parameter 'interactions' ditambahkan untuk mengontrol pencarian interaksi.
    - interactions=0: Cepat (hanya efek main)
    - interactions=10: Mencari 10 interaksi terkuat (lebih lambat)
    """
    print(f"Training EBM (Glass-Box)...")
    print(f"Note: Interaksi diset ke {interactions}")

    ebm = ExplainableBoostingClassifier(
        interactions=interactions,
        random_state=random_state,
        n_jobs=-1
    )
    ebm.fit(X_train, y_train)
    print("Training Selesai.")
    return ebm


def evaluate(model, X, y, class_names=None, model_name="Model"):
    """
    Menampilkan akurasi dan classification report.
    """
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"\n--- Evaluasi Performa: {model_name} ---")
    print(f"Akurasi: {acc:.4f}")
    print("\nLaporan Klasifikasi:")
    print(classification_report(y, y_pred, target_names=class_names))
    return acc

def get_top_features_from_lasso(lasso_model, feature_names, top_n=200):
    """
    Mengambil N fitur terpenting berdasarkan koefisien Lasso.
    Untuk multi-class, kita ambil nilai absolut maksimum di semua kelas.
    """
    # 1. Ambil Koefisien (n_classes, n_features)
    # Kita ambil nilai absolut karena koefisien negatif (-0.5) sama pentingnya dengan positif (0.5)
    abs_coefs = np.abs(lasso_model.coef_)

    # 2. Agregasi: Ambil nilai max per fitur di semua kelas
    # Artinya: "Apakah gen ini penting untuk setidaknya SATU jenis kanker?"
    importances = np.max(abs_coefs, axis=0)

    # 3. Urutkan dari yang terbesar
    # argsort mengembalikan indeks, [::-1] membaliknya jadi descending
    sorted_indices = np.argsort(importances)[::-1]

    # 4. Ambil Top N
    top_indices = sorted_indices[:top_n]
    top_feature_names = [feature_names[i] for i in top_indices]

    print(f"✅ Berhasil menyeleksi {top_n} fitur terbaik dari {len(feature_names)} fitur awal.")
    return top_indices, top_feature_names