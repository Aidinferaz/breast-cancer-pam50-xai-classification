import shap
import numpy as np


def compute_shap_linear(model, X_train, X_target, n_background=50):
    """
    Menghitung SHAP untuk model Linear (Lasso).
    """
    print(f"--- Menghitung SHAP (Linear)... ---")
    background = shap.sample(X_train, n_background)
    explainer = shap.LinearExplainer(model, background)
    shap_values = explainer.shap_values(X_target)
    print("✅ Selesai.")
    return explainer, shap_values


def compute_shap_tree(model, X_target):
    """
    Menghitung SHAP untuk model Tree (Random Forest).
    """
    print(f"--- Menghitung SHAP (Tree)... ---")
    # TreeExplainer tidak butuh background data sebesar Linear
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_target)
    print("✅ Selesai.")
    return explainer, shap_values


def fix_shap_dimensions(shap_values, X_data, class_idx):
    """
    UTILITY: Memperbaiki dimensi matriks SHAP (Transpose otomatis).
    Ini fungsi krusial agar visualisasi tidak error.
    """
    n_samples = X_data.shape[0]
    n_features = X_data.shape[1]
    shap_matrix = None

    # 1. Handle List (Output standar multiclass)
    if isinstance(shap_values, list):
        try:
            candidate = shap_values[class_idx]
            if candidate.shape == (n_samples, n_features):
                shap_matrix = candidate
            elif candidate.shape == (n_features, n_samples):
                shap_matrix = candidate.T
        except IndexError:
            print(f"❌ Error: Index kelas {class_idx} tidak ditemukan.")

    # 2. Handle Numpy Array 3D (Output TreeExplainer / versi baru)
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        # Cek axis mana yang jumlahnya sama dengan sample
        if shap_values.shape[0] == n_samples:
            shap_matrix = shap_values[:, :, class_idx]
        elif shap_values.shape[0] == len(shap_values):  # Jika terbalik (Class, Sample, Feature)
            shap_matrix = shap_values[class_idx, :, :]

    # 3. Handle Array 2D
    else:
        shap_matrix = shap_values

    # Final Check & Force Transpose
    if shap_matrix is not None:
        if shap_matrix.shape != (n_samples, n_features):
            if shap_matrix.shape == (n_features, n_samples):
                shap_matrix = shap_matrix.T

    return shap_matrix