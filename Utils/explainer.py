import shap
import numpy as np


# ==========================================
# 1. SHAP COMPUTATION
# ==========================================

def compute_shap_linear(model, X_train, X_target, n_background=50):
    """
    Menghitung SHAP values untuk model Linear (Lasso/Logistic Regression).
    """
    print(f"--- Menghitung SHAP (Linear)... ---")
    background = shap.sample(X_train, n_background)
    explainer = shap.LinearExplainer(model, background)
    shap_values = explainer.shap_values(X_target)
    print("✅ Perhitungan SHAP Linear selesai.")
    return explainer, shap_values


def compute_shap_tree(model, X_target):
    """
    Menghitung SHAP values untuk model Tree (Random Forest/XGBoost).
    """
    print(f"--- Menghitung SHAP (Tree)... ---")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_target)
    print("✅ Perhitungan SHAP Tree selesai.")
    return explainer, shap_values


def compute_shap_general(model, X_train, X_target, n_background=5, n_samples=5, nsamples_calc=100):
    """
    [REVISI] Menghitung SHAP untuk SVM (KernelExplainer) dengan penanganan format Array/List.
    """
    print(f"--- Menghitung SHAP (Kernel/SVM) Mode Hemat RAM ---")
    print(f"Target: {n_samples} sampel. Limit permutasi: {nsamples_calc}.")

    # 1. Background Data Kecil
    background = shap.sample(X_train, n_background)

    # 2. Inisialisasi Explainer
    explainer = shap.KernelExplainer(model.predict_proba, background, link="logit")

    # 3. Batasi Data Target
    X_target_sub = X_target[:n_samples] if len(X_target) > n_samples else X_target

    # 4. Hitung SHAP Values (Looping)
    shap_values_list = []
    for i in range(len(X_target_sub)):
        print(f"⏳ Memproses sampel {i + 1}/{len(X_target_sub)}...")
        try:
            # nsamples dibatasi agar RAM tidak meledak
            sv = explainer.shap_values(X_target_sub[i:i + 1], nsamples=nsamples_calc)
            shap_values_list.append(sv)
        except MemoryError:
            print(f"❌ Skip sampel {i} (Memory Full).")
            continue

    if not shap_values_list:
        print("Gagal menghitung SHAP.")
        return explainer, None

    # 5. [FIX UTAMA] Rekonstruksi Output (Handle List vs Array)
    first_sv = shap_values_list[0]
    final_shap_values = []

    # Kasus A: Output adalah List (Biasanya model Linear/Tree multiclass lama)
    # Format: [Array(1, F), Array(1, F), ...] sebanyak jumlah kelas
    if isinstance(first_sv, list):
        n_classes = len(first_sv)
        for c in range(n_classes):
            # Gabungkan semua sampel untuk kelas c
            class_shap = np.concatenate([s[c] for s in shap_values_list], axis=0)
            final_shap_values.append(class_shap)

    # Kasus B: Output adalah Numpy Array (Biasanya KernelExplainer/SVM multiclass)
    # Format: Array(1, F, C) -> (Sample, Feature, Class)
    elif isinstance(first_sv, np.ndarray):
        if first_sv.ndim == 3:
            # Kita punya dimensi ke-3 sebagai kelas
            n_classes = first_sv.shape[2]

            # Gabungkan semua sampel dulu menjadi (Total_N, F, C)
            full_array = np.concatenate(shap_values_list, axis=0)

            # Pecah menjadi List of Arrays [(N, F), (N, F), ...] agar konsisten
            for c in range(n_classes):
                final_shap_values.append(full_array[:, :, c])
        else:
            # Jika cuma 2D (Binary/Regression), jadikan list tunggal
            full_array = np.concatenate(shap_values_list, axis=0)
            final_shap_values.append(full_array)

    print(f"✅ Selesai. Terdeteksi {len(final_shap_values)} kelas.")
    return explainer, final_shap_values


# ==========================================
# 2. UTILITY (DIMENSION FIXER)
# ==========================================

def fix_shap_dimensions(shap_values, X_data, class_idx):
    """
    UTILITY: Memperbaiki dimensi matriks SHAP secara otomatis.
    """
    n_samples = X_data.shape[0]
    n_features = X_data.shape[1]
    shap_matrix = None

    # 1. Handle List (Output standar setelah diproses compute_shap_general)
    if isinstance(shap_values, list):
        try:
            candidate = shap_values[class_idx]
            if candidate.shape == (n_samples, n_features):
                shap_matrix = candidate
            elif candidate.shape == (n_features, n_samples):
                shap_matrix = candidate.T
        except IndexError:
            print(f"❌ Error: Index kelas {class_idx} tidak ditemukan (Total kelas: {len(shap_values)}).")

    # 2. Handle Numpy Array 3D Langsung (Jaga-jaga)
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        if shap_values.shape[0] == n_samples:
            shap_matrix = shap_values[:, :, class_idx]
        elif shap_values.shape[0] == len(shap_values):
            shap_matrix = shap_values[class_idx, :, :]

    # 3. Handle Array 2D
    else:
        shap_matrix = shap_values

    # Final Check & Force Transpose
    if shap_matrix is not None:
        if shap_matrix.shape != (n_samples, n_features):
            if shap_matrix.shape == (n_features, n_samples):
                shap_matrix = shap_matrix.T
            else:
                print(f"⚠️ Warning: Dimensi SHAP {shap_matrix.shape} tidak cocok dengan Data {X_data.shape}")
                return None

    return shap_matrix


# ==========================================
# 3. EBM EXPLANATION
# ==========================================

def extract_ebm_explanation(ebm_model, X_val, y_val=None):
    print("--- Mengekstrak Penjelasan EBM... ---")
    ebm_global = ebm_model.explain_global()
    ebm_local = ebm_model.explain_local(X_val, y_val)
    print("✅ Ekstraksi EBM Selesai.")
    return ebm_global, ebm_local