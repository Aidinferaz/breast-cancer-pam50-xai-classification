import shap
import numpy as np
import torch


# ==========================================
# 1. SHAP COMPUTATION
# ==========================================

def compute_shap_linear(model, X_train, X_target, n_background=50):
    """Compute SHAP values for linear models (Lasso/Logistic Regression)."""
    print("--- Computing SHAP (Linear)... ---")
    background = shap.sample(X_train, n_background)
    explainer = shap.LinearExplainer(model, background)
    shap_values = explainer.shap_values(X_target)
    print("✅ Linear SHAP computation complete.")
    return explainer, shap_values


def compute_shap_tree(model, X_target):
    """Compute SHAP values for tree-based models (Random Forest/XGBoost)."""
    print("--- Computing SHAP (Tree)... ---")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_target)
    print("✅ Tree SHAP computation complete.")
    return explainer, shap_values


def compute_shap_general(model, X_train, X_target, n_background=5, n_samples=5, nsamples_calc=100):
    """Compute SHAP for SVM using KernelExplainer with memory-efficient processing."""
    print("--- Computing SHAP (Kernel/SVM) Memory-Efficient Mode ---")
    print(f"Target: {n_samples} samples. Permutation limit: {nsamples_calc}.")

    background = shap.sample(X_train, n_background)
    explainer = shap.KernelExplainer(model.predict_proba, background, link="logit")
    X_target_sub = X_target[:n_samples] if len(X_target) > n_samples else X_target

    shap_values_list = []
    for i in range(len(X_target_sub)):
        print(f"⏳ Processing sample {i + 1}/{len(X_target_sub)}...")
        try:
            sv = explainer.shap_values(X_target_sub[i:i + 1], nsamples=nsamples_calc)
            shap_values_list.append(sv)
        except MemoryError:
            print(f"❌ Skipping sample {i} (Out of memory).")
            continue

    if not shap_values_list:
        print("Failed to compute SHAP values.")
        return explainer, None

    # Reconstruct output (handle List vs Array formats)
    first_sv = shap_values_list[0]
    final_shap_values = []

    if isinstance(first_sv, list):
        n_classes = len(first_sv)
        for c in range(n_classes):
            class_shap = np.concatenate([s[c] for s in shap_values_list], axis=0)
            final_shap_values.append(class_shap)
    elif isinstance(first_sv, np.ndarray):
        if first_sv.ndim == 3:
            n_classes = first_sv.shape[2]
            full_array = np.concatenate(shap_values_list, axis=0)
            for c in range(n_classes):
                final_shap_values.append(full_array[:, :, c])
        else:
            full_array = np.concatenate(shap_values_list, axis=0)
            final_shap_values.append(full_array)

    print(f"✅ Complete. Detected {len(final_shap_values)} classes.")
    return explainer, final_shap_values


# ==========================================
# 2. UTILITY (DIMENSION FIXER)
# ==========================================

def fix_shap_dimensions(shap_values, X_data, class_idx):
    """
    Fix SHAP matrix dimensions automatically.
    X_data must have the same number of samples used when computing SHAP values.
    """
    n_samples = X_data.shape[0]
    n_features = X_data.shape[1]
    shap_matrix = None

    if isinstance(shap_values, list):
        try:
            candidate = shap_values[class_idx]
            shap_n_samples = candidate.shape[0]
            
            if candidate.shape == (n_samples, n_features):
                shap_matrix = candidate
            elif candidate.shape == (n_features, n_samples):
                shap_matrix = candidate.T
            elif shap_n_samples != n_samples:
                print(f"❌ Error: SHAP computed for {shap_n_samples} samples, but X_data has {n_samples}.")
                return None
        except IndexError:
            print(f"❌ Error: Class index {class_idx} not found (Total classes: {len(shap_values)}).")
            return None
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        if shap_values.shape[0] == n_samples:
            shap_matrix = shap_values[:, :, class_idx]
        elif shap_values.shape[0] == len(shap_values):
            shap_matrix = shap_values[class_idx, :, :]
    else:
        shap_matrix = shap_values

    if shap_matrix is not None:
        if shap_matrix.shape != (n_samples, n_features):
            if shap_matrix.shape == (n_features, n_samples):
                shap_matrix = shap_matrix.T
            else:
                print(f"⚠️ Warning: SHAP shape {shap_matrix.shape} doesn't match data {X_data.shape}")
                return None

    return shap_matrix


# ==========================================
# 3. EBM EXPLANATION
# ==========================================

def extract_ebm_explanation(ebm_model, X_val, y_val=None):
    """Extract global and local explanations from EBM model."""
    print("--- Extracting EBM Explanations... ---")
    ebm_global = ebm_model.explain_global()
    ebm_local = ebm_model.explain_local(X_val, y_val)
    print("✅ EBM extraction complete.")
    return ebm_global, ebm_local

# ==========================================
# 4. Neural Network EXPLANATION
# ==========================================

def compute_shap_neural_network(model_wrapper, X_train, X_target, n_background=50, n_samples=10):
    """Compute SHAP values for Neural Network using DeepExplainer or GradientExplainer."""
    print("--- Computing SHAP (Neural Network)... ---")
    print(f"Background samples: {n_background}, Target samples: {n_samples}")
    
    device = model_wrapper.device
    model = model_wrapper.model_raw
    model.eval()
    
    background_indices = np.random.choice(len(X_train), min(n_background, len(X_train)), replace=False)
    background = X_train[background_indices]
    background_tensor = torch.FloatTensor(background).to(device)
    
    X_target_sub = X_target[:n_samples] if len(X_target) > n_samples else X_target
    target_tensor = torch.FloatTensor(X_target_sub).to(device)
    
    try:
        print("Trying DeepExplainer...")
        explainer = shap.DeepExplainer(model, background_tensor)
        shap_values_raw = explainer.shap_values(target_tensor)
        print("✅ DeepExplainer succeeded.")
        
    except Exception as e:
        print(f"⚠️ DeepExplainer failed ({e}), switching to GradientExplainer...")
        try:
            explainer = shap.GradientExplainer(model, background_tensor)
            shap_values_raw = explainer.shap_values(target_tensor)
            print("✅ GradientExplainer succeeded.")
            
        except Exception as e2:
            print(f"⚠️ GradientExplainer failed ({e2}), switching to KernelExplainer...")
            explainer = shap.KernelExplainer(model_wrapper.predict_proba, background)
            
            shap_values_list = []
            for i in range(len(X_target_sub)):
                print(f"⏳ Processing sample {i + 1}/{len(X_target_sub)}...")
                sv = explainer.shap_values(X_target_sub[i:i + 1], nsamples=100)
                shap_values_list.append(sv)
            
            first_sv = shap_values_list[0]
            shap_values_raw = []
            if isinstance(first_sv, list):
                n_classes = len(first_sv)
                for c in range(n_classes):
                    class_shap = np.concatenate([s[c] for s in shap_values_list], axis=0)
                    shap_values_raw.append(class_shap)
            else:
                shap_values_raw = np.concatenate(shap_values_list, axis=0)
            
            print("✅ KernelExplainer succeeded.")
    
    final_shap_values = _format_nn_shap_values(shap_values_raw, X_target_sub)
    
    print(f"✅ Neural Network SHAP complete. Classes detected: {len(final_shap_values)}")
    return explainer, final_shap_values


def _format_nn_shap_values(shap_values_raw, X_target):
    """Format Neural Network SHAP values to consistent list format."""
    if isinstance(shap_values_raw, list):
        formatted = []
        for sv in shap_values_raw:
            if isinstance(sv, torch.Tensor):
                sv = sv.cpu().numpy()
            formatted.append(sv)
        return formatted
    
    if isinstance(shap_values_raw, torch.Tensor):
        shap_values_raw = shap_values_raw.cpu().numpy()
    
    if isinstance(shap_values_raw, np.ndarray):
        if shap_values_raw.ndim == 3:
            n_classes = shap_values_raw.shape[2]
            return [shap_values_raw[:, :, c] for c in range(n_classes)]
        elif shap_values_raw.ndim == 2:
            return [shap_values_raw]
    
    return shap_values_raw


def compute_shap_integrated_gradients(model_wrapper, X_target, baseline=None, n_steps=50):
    """Compute Integrated Gradients as an alternative to SHAP for Neural Networks."""
    print("--- Computing Integrated Gradients... ---")
    
    device = model_wrapper.device
    model = model_wrapper.model_raw
    model.eval()
    
    X_tensor = torch.FloatTensor(X_target).to(device)
    X_tensor.requires_grad = True
    
    if baseline is None:
        baseline = torch.zeros_like(X_tensor).to(device)
    else:
        baseline = torch.FloatTensor(baseline).to(device)
    
    scaled_inputs = [baseline + (float(i) / n_steps) * (X_tensor - baseline) for i in range(n_steps + 1)]
    
    gradients = []
    for scaled_input in scaled_inputs:
        scaled_input.requires_grad = True
        output = model(scaled_input)
        target_class = output.argmax(dim=1)
        selected_output = output.gather(1, target_class.unsqueeze(1)).sum()
        model.zero_grad()
        selected_output.backward(retain_graph=True)
        gradients.append(scaled_input.grad.detach().clone())
    
    avg_gradients = torch.stack(gradients).mean(dim=0)
    integrated_gradients = (X_tensor - baseline).detach() * avg_gradients
    
    print("✅ Integrated Gradients computation complete.")
    return integrated_gradients.cpu().numpy()


# ==========================================
# 5. XGBOOST EXPLANATION
# ==========================================

def compute_shap_xgboost(model, X_target, X_train=None, use_tree_explainer=True):
    """Compute SHAP values for XGBoost with optimization."""
    print("--- Computing SHAP (XGBoost)... ---")
    
    if hasattr(X_target, 'values'):
        X_target = X_target.values
    X_target = np.asarray(X_target, dtype=np.float64)
    
    if X_train is not None:
        if hasattr(X_train, 'values'):
            X_train = X_train.values
        X_train = np.asarray(X_train, dtype=np.float64)
    
    if use_tree_explainer:
        try:
            tree_explainer = shap.TreeExplainer(model)
            shap_values = tree_explainer.shap_values(X_target)
            print("✅ XGBoost SHAP complete (TreeExplainer).")
            return tree_explainer, shap_values
        except ValueError as e:
            if "could not convert string to float" in str(e):
                print("⚠️ TreeExplainer incompatible, using predict_proba wrapper...")
                if X_train is None:
                    background = shap.sample(X_target, min(100, len(X_target)))
                else:
                    background = shap.sample(X_train, min(100, len(X_train)))
                
                kernel_explainer = shap.KernelExplainer(model.predict_proba, background)
                shap_values_raw = kernel_explainer.shap_values(X_target, nsamples=100)
                shap_values = _format_shap_to_list(shap_values_raw)
                
                print("✅ XGBoost SHAP complete (KernelExplainer fallback).")
                return kernel_explainer, shap_values
            else:
                raise e
    else:
        if X_train is None:
            raise ValueError("X_train required for KernelExplainer")
        background = shap.sample(X_train, min(100, len(X_train)))
        kernel_explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values_raw = kernel_explainer.shap_values(X_target)
        shap_values = _format_shap_to_list(shap_values_raw)
        print("✅ XGBoost SHAP complete (KernelExplainer).")
        return kernel_explainer, shap_values


def _format_shap_to_list(shap_values):
    """Convert SHAP values to list format [array_class0, array_class1, ...]."""
    if isinstance(shap_values, list):
        return shap_values
    
    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            n_classes = shap_values.shape[2]
            return [shap_values[:, :, c] for c in range(n_classes)]
        elif shap_values.ndim == 2:
            return [shap_values]
    
    return shap_values


def compute_shap_xgboost_interactions(model, X_target, max_samples=100):
    """Compute SHAP interaction values for XGBoost (computationally expensive)."""
    print("--- Computing SHAP Interaction Values (XGBoost)... ---")
    print("⚠️ This process may take a while...")
    
    X_subset = X_target[:max_samples] if len(X_target) > max_samples else X_target
    
    try:
        tree_explainer = shap.TreeExplainer(model)
        shap_interaction_values = tree_explainer.shap_interaction_values(X_subset)
        print(f"✅ SHAP Interaction complete. Shape: {np.array(shap_interaction_values).shape}")
        return tree_explainer, shap_interaction_values
    except ValueError as e:
        if "could not convert string to float" in str(e):
            print("❌ SHAP Interaction not available for this XGBoost multiclass model.")
            return None, None
        else:
            raise e


def get_xgboost_feature_importance(model, importance_type='weight'):
    """Get native XGBoost feature importance."""
    print(f"--- Getting XGBoost Feature Importance ({importance_type})... ---")
    
    try:
        importance_dict = model.get_booster().get_score(importance_type=importance_type)
    except AttributeError:
        importance_dict = model.get_score(importance_type=importance_type)
    
    print(f"✅ Found {len(importance_dict)} features with importance > 0")
    return importance_dict


def compare_xgboost_importance_methods(model, X_target, feature_names):
    """Compare different feature importance methods for XGBoost."""
    print("--- Comparing XGBoost Feature Importance Methods... ---")
    
    results = {}
    
    gain_importance = get_xgboost_feature_importance(model, 'gain')
    results['xgb_gain'] = {f: gain_importance.get(f, 0) for f in feature_names}
    
    weight_importance = get_xgboost_feature_importance(model, 'weight')
    results['xgb_weight'] = {f: weight_importance.get(f, 0) for f in feature_names}
    
    _, shap_values = compute_shap_xgboost(model, X_target)
    
    if isinstance(shap_values, list):
        mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_shap = np.abs(shap_values).mean(axis=0)
    
    results['shap'] = {feature_names[i]: mean_shap[i] for i in range(len(feature_names))}
    
    print("✅ Comparison complete.")
    return results


def extract_xgboost_rules(model, feature_names, max_trees=5, max_depth=3):
    """Extract rules from XGBoost trees for interpretability."""
    print(f"--- Extracting Rules from XGBoost (max {max_trees} trees)... ---")
    
    try:
        booster = model.get_booster()
    except AttributeError:
        booster = model
    
    trees_dump = booster.get_dump(with_stats=True)
    
    rules = []
    for i, tree in enumerate(trees_dump[:max_trees]):
        rules.append(f"\n=== Tree {i} ===")
        lines = tree.split('\n')
        for line in lines[:max_depth * 3]:
            if line.strip():
                for j, fname in enumerate(feature_names):
                    line = line.replace(f'f{j}', fname)
                rules.append(line)
    
    print(f"✅ Extraction complete. Total {len(rules)} rule lines.")
    return rules