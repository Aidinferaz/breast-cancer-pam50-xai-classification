import shap
import numpy as np
import torch


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

# ==========================================
# 4. Neural Network EXPLANATION
# ==========================================

def compute_shap_neural_network(model_wrapper, X_train, X_target, n_background=50, n_samples=10):
    """
    Menghitung SHAP values untuk Neural Network menggunakan DeepExplainer atau GradientExplainer.
    
    Parameters:
    -----------
    model_wrapper : NeuralNetworkWrapper, Model wrapper dengan API predict_proba
    X_train : array-like, Data training untuk background
    X_target : array-like, Data yang akan dijelaskan
    n_background : int, Jumlah sampel background
    n_samples : int, Jumlah sampel target yang akan diproses
    
    Returns:
    --------
    explainer : SHAP explainer object
    shap_values : list of arrays, SHAP values untuk setiap kelas
    """
    print(f"--- Menghitung SHAP (Neural Network)... ---")
    print(f"Background samples: {n_background}, Target samples: {n_samples}")
    
    device = model_wrapper.device
    model = model_wrapper.model_raw
    model.eval()
    
    # Prepare background data
    background_indices = np.random.choice(len(X_train), min(n_background, len(X_train)), replace=False)
    background = X_train[background_indices]
    background_tensor = torch.FloatTensor(background).to(device)
    
    # Prepare target data
    X_target_sub = X_target[:n_samples] if len(X_target) > n_samples else X_target
    target_tensor = torch.FloatTensor(X_target_sub).to(device)
    
    try:
        # Coba gunakan DeepExplainer (lebih cepat untuk Neural Network)
        print("Mencoba DeepExplainer...")
        explainer = shap.DeepExplainer(model, background_tensor)
        shap_values_raw = explainer.shap_values(target_tensor)
        print("✅ DeepExplainer berhasil.")
        
    except Exception as e:
        print(f"⚠️ DeepExplainer gagal ({e}), beralih ke GradientExplainer...")
        try:
            explainer = shap.GradientExplainer(model, background_tensor)
            shap_values_raw = explainer.shap_values(target_tensor)
            print("✅ GradientExplainer berhasil.")
            
        except Exception as e2:
            print(f"⚠️ GradientExplainer gagal ({e2}), beralih ke KernelExplainer...")
            # Fallback ke KernelExplainer (lebih lambat tapi lebih robust)
            explainer = shap.KernelExplainer(model_wrapper.predict_proba, background)
            
            shap_values_list = []
            for i in range(len(X_target_sub)):
                print(f"⏳ Memproses sampel {i + 1}/{len(X_target_sub)}...")
                sv = explainer.shap_values(X_target_sub[i:i + 1], nsamples=100)
                shap_values_list.append(sv)
            
            # Rekonstruksi output
            first_sv = shap_values_list[0]
            shap_values_raw = []
            if isinstance(first_sv, list):
                n_classes = len(first_sv)
                for c in range(n_classes):
                    class_shap = np.concatenate([s[c] for s in shap_values_list], axis=0)
                    shap_values_raw.append(class_shap)
            else:
                shap_values_raw = np.concatenate(shap_values_list, axis=0)
            
            print("✅ KernelExplainer berhasil.")
    
    # Format output menjadi list konsisten
    final_shap_values = _format_nn_shap_values(shap_values_raw, X_target_sub)
    
    print(f"✅ Perhitungan SHAP Neural Network selesai. Kelas terdeteksi: {len(final_shap_values)}")
    return explainer, final_shap_values


def _format_nn_shap_values(shap_values_raw, X_target):
    """
    Helper function untuk memformat SHAP values dari Neural Network.
    """
    n_samples = X_target.shape[0]
    n_features = X_target.shape[1]
    
    # Jika sudah list, kembalikan langsung
    if isinstance(shap_values_raw, list):
        formatted = []
        for sv in shap_values_raw:
            if isinstance(sv, torch.Tensor):
                sv = sv.cpu().numpy()
            formatted.append(sv)
        return formatted
    
    # Jika tensor, konversi ke numpy
    if isinstance(shap_values_raw, torch.Tensor):
        shap_values_raw = shap_values_raw.cpu().numpy()
    
    # Handle berbagai format dimensi
    if isinstance(shap_values_raw, np.ndarray):
        if shap_values_raw.ndim == 3:
            # Format (N, F, C) -> pecah jadi list
            n_classes = shap_values_raw.shape[2]
            return [shap_values_raw[:, :, c] for c in range(n_classes)]
        elif shap_values_raw.ndim == 2:
            # Format (N, F) -> wrap dalam list
            return [shap_values_raw]
    
    return shap_values_raw


def compute_shap_integrated_gradients(model_wrapper, X_target, baseline=None, n_steps=50):
    """
    Menghitung Integrated Gradients sebagai alternatif SHAP untuk Neural Network.
    Integrated Gradients adalah metode atribusi yang lebih cepat.
    
    Parameters:
    -----------
    model_wrapper : NeuralNetworkWrapper
    X_target : array-like, Data yang akan dijelaskan
    baseline : array-like, Baseline untuk IG (default: zeros)
    n_steps : int, Jumlah langkah interpolasi
    
    Returns:
    --------
    attributions : array, Integrated Gradients attributions
    """
    print(f"--- Menghitung Integrated Gradients... ---")
    
    device = model_wrapper.device
    model = model_wrapper.model_raw
    model.eval()
    
    X_tensor = torch.FloatTensor(X_target).to(device)
    X_tensor.requires_grad = True
    
    if baseline is None:
        baseline = torch.zeros_like(X_tensor).to(device)
    else:
        baseline = torch.FloatTensor(baseline).to(device)
    
    # Interpolasi antara baseline dan input
    scaled_inputs = [baseline + (float(i) / n_steps) * (X_tensor - baseline) for i in range(n_steps + 1)]
    
    # Hitung gradien untuk setiap step
    gradients = []
    for scaled_input in scaled_inputs:
        scaled_input.requires_grad = True
        output = model(scaled_input)
        
        # Untuk multiclass, ambil kelas dengan probabilitas tertinggi
        target_class = output.argmax(dim=1)
        selected_output = output.gather(1, target_class.unsqueeze(1)).sum()
        
        model.zero_grad()
        selected_output.backward(retain_graph=True)
        gradients.append(scaled_input.grad.detach().clone())
    
    # Hitung rata-rata gradien
    avg_gradients = torch.stack(gradients).mean(dim=0)
    
    # Integrated Gradients = (input - baseline) * avg_gradients
    integrated_gradients = (X_tensor - baseline).detach() * avg_gradients
    
    print("✅ Perhitungan Integrated Gradients selesai.")
    return integrated_gradients.cpu().numpy()