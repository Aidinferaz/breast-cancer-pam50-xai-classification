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
    
    PENTING: X_data harus memiliki jumlah sampel yang SAMA dengan yang digunakan
    saat menghitung SHAP values (n_samples parameter di compute_shap_general).
    """
    n_samples = X_data.shape[0]
    n_features = X_data.shape[1]
    shap_matrix = None

    # 1. Handle List (Output standar setelah diproses compute_shap_general)
    if isinstance(shap_values, list):
        try:
            candidate = shap_values[class_idx]
            shap_n_samples = candidate.shape[0]
            shap_n_features = candidate.shape[1] if len(candidate.shape) > 1 else candidate.shape[0]
            
            if candidate.shape == (n_samples, n_features):
                shap_matrix = candidate
            elif candidate.shape == (n_features, n_samples):
                shap_matrix = candidate.T
            elif shap_n_samples != n_samples:
                # Dimensi tidak cocok - berikan pesan error yang jelas
                print(f"❌ Error: SHAP dihitung untuk {shap_n_samples} sampel, tapi X_data memiliki {n_samples} sampel.")
                print(f"   Pastikan X_data[:n] menggunakan n yang sama dengan n_samples di compute_shap_general().")
                return None
        except IndexError:
            print(f"❌ Error: Index kelas {class_idx} tidak ditemukan (Total kelas: {len(shap_values)}).")
            return None

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
                print(f"   Pastikan jumlah sampel di X_data sama dengan n_samples saat compute SHAP.")
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


# ==========================================
# 5. XGBOOST EXPLANATION
# ==========================================

def compute_shap_xgboost(model, X_target, X_train=None, use_tree_explainer=True):
    """
    Menghitung SHAP values khusus untuk XGBoost dengan optimasi.
    
    Parameters:
    -----------
    model : XGBoost model (Classifier atau Regressor)
    X_target : array-like, Data yang akan dijelaskan (DataFrame atau numpy array)
    X_train : array-like, Data training (opsional, untuk background)
    use_tree_explainer : bool, Gunakan TreeExplainer (lebih cepat) atau KernelExplainer
    
    Returns:
    --------
    explainer : SHAP explainer object
    shap_values : array/list, SHAP values
    """
    print(f"--- Menghitung SHAP (XGBoost)... ---")
    
    # Konversi ke numpy array jika DataFrame
    if hasattr(X_target, 'values'):
        X_target = X_target.values
    
    # Pastikan data numerik
    X_target = np.asarray(X_target, dtype=np.float64)
    
    if X_train is not None:
        if hasattr(X_train, 'values'):
            X_train = X_train.values
        X_train = np.asarray(X_train, dtype=np.float64)
    
    if use_tree_explainer:
        try:
            # Coba TreeExplainer langsung
            tree_explainer = shap.TreeExplainer(model)
            shap_values = tree_explainer.shap_values(X_target)
            print("✅ Perhitungan SHAP XGBoost selesai (TreeExplainer).")
            return tree_explainer, shap_values
        except ValueError as e:
            if "could not convert string to float" in str(e):
                # XGBoost multiclass base_score compatibility issue
                print("⚠️ TreeExplainer tidak kompatibel, menggunakan predict_proba wrapper...")
                if X_train is None:
                    # Gunakan X_target sebagai background jika X_train tidak tersedia
                    background = shap.sample(X_target, min(100, len(X_target)))
                else:
                    background = shap.sample(X_train, min(100, len(X_train)))
                
                kernel_explainer = shap.KernelExplainer(model.predict_proba, background)
                shap_values_raw = kernel_explainer.shap_values(X_target, nsamples=100)
                
                # Konversi ke format list [class0, class1, ...] untuk konsistensi
                shap_values = _format_shap_to_list(shap_values_raw)
                
                print("✅ Perhitungan SHAP XGBoost selesai (KernelExplainer fallback).")
                return kernel_explainer, shap_values
            else:
                raise e
    else:
        if X_train is None:
            raise ValueError("X_train diperlukan untuk KernelExplainer")
        background = shap.sample(X_train, min(100, len(X_train)))
        kernel_explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values_raw = kernel_explainer.shap_values(X_target)
        shap_values = _format_shap_to_list(shap_values_raw)
        print("✅ Perhitungan SHAP XGBoost selesai (KernelExplainer).")
        return kernel_explainer, shap_values


def _format_shap_to_list(shap_values):
    """
    Mengkonversi SHAP values ke format list [array_class0, array_class1, ...].
    """
    if isinstance(shap_values, list):
        return shap_values
    
    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            # Format (N, F, C) -> list of (N, F)
            n_classes = shap_values.shape[2]
            return [shap_values[:, :, c] for c in range(n_classes)]
        elif shap_values.ndim == 2:
            # Binary atau regression, wrap dalam list
            return [shap_values]
    
    return shap_values


def compute_shap_xgboost_interactions(model, X_target, max_samples=100):
    """
    Menghitung SHAP interaction values untuk XGBoost.
    Menunjukkan bagaimana fitur berinteraksi satu sama lain.
    
    Parameters:
    -----------
    model : XGBoost model
    X_target : array-like, Data yang akan dijelaskan
    max_samples : int, Maksimum sampel (interaction values mahal secara komputasi)
    
    Returns:
    --------
    explainer : SHAP TreeExplainer
    shap_interaction_values : array, Shape (n_samples, n_features, n_features) atau list untuk multiclass
    """
    print(f"--- Menghitung SHAP Interaction Values (XGBoost)... ---")
    print(f"⚠️ Proses ini membutuhkan waktu lebih lama...")
    
    # Batasi jumlah sampel
    X_subset = X_target[:max_samples] if len(X_target) > max_samples else X_target
    
    try:
        tree_explainer = shap.TreeExplainer(model)
        shap_interaction_values = tree_explainer.shap_interaction_values(X_subset)
        print(f"✅ Perhitungan SHAP Interaction selesai. Shape: {np.array(shap_interaction_values).shape}")
        return tree_explainer, shap_interaction_values
    except ValueError as e:
        if "could not convert string to float" in str(e):
            print("❌ SHAP Interaction tidak tersedia untuk model XGBoost multiclass ini.")
            print("   Gunakan compute_shap_xgboost() untuk SHAP values biasa.")
            return None, None
        else:
            raise e


def get_xgboost_feature_importance(model, importance_type='weight'):
    """
    Mengambil feature importance bawaan XGBoost.
    
    Parameters:
    -----------
    model : XGBoost model
    importance_type : str, Tipe importance:
        - 'weight': Jumlah kemunculan fitur di semua tree
        - 'gain': Rata-rata gain saat fitur digunakan untuk split
        - 'cover': Rata-rata coverage saat fitur digunakan
        - 'total_gain': Total gain
        - 'total_cover': Total coverage
    
    Returns:
    --------
    importance_dict : dict, {feature_name: importance_score}
    """
    print(f"--- Mengambil Feature Importance XGBoost ({importance_type})... ---")
    
    try:
        # Untuk XGBClassifier/XGBRegressor
        importance_dict = model.get_booster().get_score(importance_type=importance_type)
    except AttributeError:
        # Untuk Booster langsung
        importance_dict = model.get_score(importance_type=importance_type)
    
    print(f"✅ Ditemukan {len(importance_dict)} fitur dengan importance > 0")
    return importance_dict


def compare_xgboost_importance_methods(model, X_target, feature_names):
    """
    Membandingkan berbagai metode feature importance untuk XGBoost.
    
    Parameters:
    -----------
    model : XGBoost model
    X_target : array-like, Data target
    feature_names : list, Nama fitur
    
    Returns:
    --------
    comparison_df : dict, Perbandingan importance dari berbagai metode
    """
    print("--- Membandingkan Metode Feature Importance XGBoost... ---")
    
    results = {}
    
    # 1. Native XGBoost Importance (Gain)
    gain_importance = get_xgboost_feature_importance(model, 'gain')
    results['xgb_gain'] = {f: gain_importance.get(f, 0) for f in feature_names}
    
    # 2. Native XGBoost Importance (Weight)
    weight_importance = get_xgboost_feature_importance(model, 'weight')
    results['xgb_weight'] = {f: weight_importance.get(f, 0) for f in feature_names}
    
    # 3. SHAP Values
    _, shap_values = compute_shap_xgboost(model, X_target)
    
    if isinstance(shap_values, list):
        # Multiclass: rata-rata absolute SHAP across classes
        mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_shap = np.abs(shap_values).mean(axis=0)
    
    results['shap'] = {feature_names[i]: mean_shap[i] for i in range(len(feature_names))}
    
    print("✅ Perbandingan selesai.")
    return results


def extract_xgboost_rules(model, feature_names, max_trees=5, max_depth=3):
    """
    Mengekstrak aturan (rules) dari XGBoost trees untuk interpretabilitas.
    
    Parameters:
    -----------
    model : XGBoost model
    feature_names : list, Nama fitur
    max_trees : int, Jumlah tree yang diekstrak
    max_depth : int, Kedalaman maksimum rule
    
    Returns:
    --------
    rules : list of str, Aturan-aturan yang diekstrak
    """
    print(f"--- Mengekstrak Rules dari XGBoost (max {max_trees} trees)... ---")
    
    try:
        booster = model.get_booster()
    except AttributeError:
        booster = model
    
    # Dump trees ke format text
    trees_dump = booster.get_dump(with_stats=True)
    
    rules = []
    for i, tree in enumerate(trees_dump[:max_trees]):
        rules.append(f"\n=== Tree {i} ===")
        # Parse tree structure
        lines = tree.split('\n')
        for line in lines[:max_depth * 3]:  # Approximate depth limit
            if line.strip():
                # Replace feature indices with names if possible
                for j, fname in enumerate(feature_names):
                    line = line.replace(f'f{j}', fname)
                rules.append(line)
    
    print(f"✅ Ekstraksi selesai. Total {len(rules)} baris rules.")
    return rules