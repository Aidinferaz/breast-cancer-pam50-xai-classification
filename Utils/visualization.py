import shap
import matplotlib.pyplot as plt
from Utils import explainer as explainers
from interpret import show
import numpy as np

def plot_beeswarm(shap_values, X_data, feature_names, class_names, target_class='all', max_display=15):
    """Create Global Summary Plot (Beeswarm)."""
    indices_to_plot = []
    if target_class == 'all':
        indices_to_plot = range(len(class_names))
    elif isinstance(target_class, int):
        indices_to_plot = [target_class]
    elif isinstance(target_class, str):
        if target_class in class_names:
            indices_to_plot = [list(class_names).index(target_class)]
        else:
            print(f"❌ Class '{target_class}' not found.")
            return

    for idx in indices_to_plot:
        label = class_names[idx]
        matrix = explainers.fix_shap_dimensions(shap_values, X_data, idx)

        if matrix is not None:
            plt.figure(figsize=(10, 6))
            print(f"Generating Summary Plot for: {label}...")
            shap.summary_plot(matrix, X_data, feature_names=feature_names, show=False, max_display=max_display)
            plt.title(f"Feature Importance: {label}", fontsize=14)
            plt.show()
        else:
            print(f"⚠️ Failed to process dimensions for class {label}")


def plot_waterfall(shap_values, explainer, X_data, feature_names, class_names, sample_idx=0, class_idx=0,
                   max_display=10):
    """Create Local Waterfall Plot for a single patient."""
    shap_matrix = explainers.fix_shap_dimensions(shap_values, X_data, class_idx)

    if shap_matrix is None:
        return

    if hasattr(explainer.expected_value, '__iter__'):
        base_val = explainer.expected_value[class_idx]
    else:
        base_val = explainer.expected_value

    try:
        explanation = shap.Explanation(
            values=shap_matrix[sample_idx],
            base_values=base_val,
            data=X_data[sample_idx],
            feature_names=feature_names
        )

        target_label = class_names[class_idx]
        print(f"\n--- Local Analysis Patient #{sample_idx} (Target Class: {target_label}) ---")

        shap.plots.waterfall(explanation, max_display=max_display)

    except Exception as e:
        print(f"❌ Failed to create waterfall plot: {e}")

def plot_ebm_dashboard(ebm_global, title="EBM Global Explanation"):
    """Display Interactive EBM Dashboard (works in Jupyter Notebook)."""
    print(f"Opening Dashboard: {title}...")
    show(ebm_global)

def plot_ebm_feature_curve(ebm_global, feature_index=0):
    """Display shape function curve for a specific feature."""
    data = ebm_global.visualize(feature_index)
    return data

def plot_nn_feature_importance(shap_values, feature_names, class_names, top_n=20):
    """Create bar plot for Neural Network feature importance."""
    print("Generating Feature Importance Plot for Neural Network...")
    
    if isinstance(shap_values, list):
        all_shap = np.stack([np.abs(sv).mean(axis=0) for sv in shap_values])
        mean_importance = all_shap.mean(axis=0)
    else:
        mean_importance = np.abs(shap_values).mean(axis=0)
    
    sorted_idx = np.argsort(mean_importance)[::-1][:top_n]
    sorted_importance = mean_importance[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(sorted_idx)))
    plt.barh(range(len(sorted_idx)), sorted_importance[::-1], color=colors[::-1])
    plt.yticks(range(len(sorted_idx)), sorted_names[::-1])
    plt.xlabel('Mean |SHAP Value|')
    plt.title(f'Top {top_n} Feature Importance (Neural Network)', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_nn_beeswarm(shap_values, X_data, feature_names, class_names, target_class='all', max_display=15):
    """Create Global Summary Plot (Beeswarm) for Neural Network."""
    indices_to_plot = []
    if target_class == 'all':
        indices_to_plot = range(len(class_names))
    elif isinstance(target_class, int):
        indices_to_plot = [target_class]
    elif isinstance(target_class, str):
        if target_class in class_names:
            indices_to_plot = [list(class_names).index(target_class)]
        else:
            print(f"❌ Class '{target_class}' not found.")
            return

    for idx in indices_to_plot:
        label = class_names[idx]
        matrix = explainers.fix_shap_dimensions(shap_values, X_data, idx)

        if matrix is not None:
            plt.figure(figsize=(10, 6))
            print(f"Generating Summary Plot for: {label} (Neural Network)...")
            shap.summary_plot(matrix, X_data, feature_names=feature_names, show=False, max_display=max_display)
            plt.title(f"Neural Network Feature Importance: {label}", fontsize=14)
            plt.show()
        else:
            print(f"⚠️ Failed to process dimensions for class {label}")


def plot_nn_waterfall(shap_values, explainer, X_data, feature_names, class_names, 
                      sample_idx=0, class_idx=0, max_display=10):
    """Create Local Waterfall Plot for Neural Network."""
    shap_matrix = explainers.fix_shap_dimensions(shap_values, X_data, class_idx)

    if shap_matrix is None:
        return

    if hasattr(explainer, 'expected_value'):
        if hasattr(explainer.expected_value, '__iter__'):
            base_val = explainer.expected_value[class_idx]
        else:
            base_val = explainer.expected_value
    else:
        base_val = 0

    try:
        explanation = shap.Explanation(
            values=shap_matrix[sample_idx],
            base_values=base_val,
            data=X_data[sample_idx],
            feature_names=feature_names
        )

        target_label = class_names[class_idx]
        print(f"\n--- Local Analysis Patient #{sample_idx} (Neural Network - Class: {target_label}) ---")

        shap.plots.waterfall(explanation, max_display=max_display)

    except Exception as e:
        print(f"❌ Failed to create waterfall plot: {e}")


def plot_nn_force(shap_values, explainer, X_data, feature_names, class_names,
                  sample_idx=0, class_idx=0):
    """Create Force Plot for Neural Network."""
    shap_matrix = explainers.fix_shap_dimensions(shap_values, X_data, class_idx)

    if shap_matrix is None:
        return

    if hasattr(explainer, 'expected_value'):
        if hasattr(explainer.expected_value, '__iter__'):
            base_val = explainer.expected_value[class_idx]
        else:
            base_val = explainer.expected_value
    else:
        base_val = 0

    target_label = class_names[class_idx]
    print(f"\n--- Force Plot Patient #{sample_idx} (Neural Network - Class: {target_label}) ---")

    shap.initjs()
    force_plot = shap.force_plot(
        base_val,
        shap_matrix[sample_idx],
        X_data[sample_idx],
        feature_names=feature_names
    )
    return force_plot


def plot_integrated_gradients(ig_values, feature_names, sample_idx=0, top_n=15):
    """Visualize Integrated Gradients for Neural Network."""
    print(f"\n--- Integrated Gradients for Sample #{sample_idx} ---")
    
    sample_ig = ig_values[sample_idx]
    
    sorted_idx = np.argsort(np.abs(sample_ig))[::-1][:top_n]
    sorted_ig = sample_ig[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]
    
    plt.figure(figsize=(10, 6))
    colors = ['red' if x > 0 else 'blue' for x in sorted_ig]
    plt.barh(range(len(sorted_idx)), sorted_ig[::-1], color=colors[::-1])
    plt.yticks(range(len(sorted_idx)), sorted_names[::-1])
    plt.xlabel('Attribution Score')
    plt.title(f'Integrated Gradients - Sample #{sample_idx}', fontsize=14)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.show()


# ==========================================
# XGBOOST VISUALIZATION
# ==========================================

def plot_xgboost_beeswarm(shap_values, X_data, feature_names, class_names, target_class='all', max_display=15):
    """Create Global Summary Plot (Beeswarm) for XGBoost."""
    indices_to_plot = []
    if target_class == 'all':
        indices_to_plot = range(len(class_names))
    elif isinstance(target_class, int):
        indices_to_plot = [target_class]
    elif isinstance(target_class, str):
        if target_class in class_names:
            indices_to_plot = [list(class_names).index(target_class)]
        else:
            print(f"❌ Class '{target_class}' not found.")
            return

    for idx in indices_to_plot:
        label = class_names[idx]
        matrix = explainers.fix_shap_dimensions(shap_values, X_data, idx)

        if matrix is not None:
            plt.figure(figsize=(10, 6))
            print(f"Generating Summary Plot for: {label} (XGBoost)...")
            shap.summary_plot(matrix, X_data, feature_names=feature_names, show=False, max_display=max_display)
            plt.title(f"XGBoost Feature Importance: {label}", fontsize=14)
            plt.show()
        else:
            print(f"⚠️ Failed to process dimensions for class {label}")


def plot_xgboost_bar(shap_values, X_data, feature_names, class_names, class_idx=0, max_display=15):
    """Create Bar Plot for XGBoost feature importance based on SHAP."""
    label = class_names[class_idx]
    matrix = explainers.fix_shap_dimensions(shap_values, X_data, class_idx)
    
    if matrix is None:
        print(f"⚠️ Failed to process dimensions for class {label}")
        return
    
    mean_abs_shap = np.abs(matrix).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1][:max_display]
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(sorted_idx)))[::-1]
    plt.barh(range(len(sorted_idx)), mean_abs_shap[sorted_idx][::-1], color=colors)
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx][::-1])
    plt.xlabel('Mean |SHAP Value|')
    plt.title(f'XGBoost Feature Importance (SHAP): {label}', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_xgboost_waterfall(shap_values, explainer, X_data, feature_names, class_names, 
                           sample_idx=0, class_idx=0, max_display=10):
    """Create Local Waterfall Plot for XGBoost."""
    shap_matrix = explainers.fix_shap_dimensions(shap_values, X_data, class_idx)

    if shap_matrix is None:
        return

    if hasattr(explainer, 'expected_value'):
        if hasattr(explainer.expected_value, '__iter__'):
            base_val = explainer.expected_value[class_idx]
        else:
            base_val = explainer.expected_value
    else:
        base_val = 0

    try:
        explanation = shap.Explanation(
            values=shap_matrix[sample_idx],
            base_values=base_val,
            data=X_data[sample_idx],
            feature_names=feature_names
        )

        target_label = class_names[class_idx]
        print(f"\n--- Local Analysis Patient #{sample_idx} (XGBoost - Class: {target_label}) ---")

        shap.plots.waterfall(explanation, max_display=max_display)

    except Exception as e:
        print(f"❌ Failed to create waterfall plot: {e}")


def plot_xgboost_force(shap_values, explainer, X_data, feature_names, class_names,
                       sample_idx=0, class_idx=0):
    """Create Force Plot for XGBoost."""
    shap_matrix = explainers.fix_shap_dimensions(shap_values, X_data, class_idx)

    if shap_matrix is None:
        return

    if hasattr(explainer, 'expected_value'):
        if hasattr(explainer.expected_value, '__iter__'):
            base_val = explainer.expected_value[class_idx]
        else:
            base_val = explainer.expected_value
    else:
        base_val = 0

    target_label = class_names[class_idx]
    print(f"\n--- Force Plot Patient #{sample_idx} (XGBoost - Class: {target_label}) ---")

    shap.initjs()
    force_plot = shap.force_plot(
        base_val,
        shap_matrix[sample_idx],
        X_data[sample_idx],
        feature_names=feature_names
    )
    return force_plot


def plot_xgboost_dependence(shap_values, X_data, feature_names, class_names, 
                            feature_idx, interaction_idx=None, class_idx=0):
    """Create Dependence Plot for XGBoost showing how SHAP value changes with feature value."""
    shap_matrix = explainers.fix_shap_dimensions(shap_values, X_data, class_idx)

    if shap_matrix is None:
        return

    if isinstance(feature_idx, str):
        feature_idx = list(feature_names).index(feature_idx)
    
    if isinstance(interaction_idx, str) and interaction_idx != 'auto':
        interaction_idx = list(feature_names).index(interaction_idx)

    target_label = class_names[class_idx]
    feature_name = feature_names[feature_idx]
    
    plt.figure(figsize=(10, 6))
    print(f"Generating Dependence Plot for: {feature_name} (Class: {target_label})...")
    
    shap.dependence_plot(
        feature_idx, 
        shap_matrix, 
        X_data, 
        feature_names=feature_names,
        interaction_index=interaction_idx,
        show=False
    )
    plt.title(f'XGBoost Dependence Plot: {feature_name} ({target_label})', fontsize=14)
    plt.show()


def plot_xgboost_interaction_heatmap(shap_interaction_values, feature_names, class_names,
                                     class_idx=0, top_n=10):
    """Create Heatmap for SHAP Interaction Values."""
    print(f"Generating Interaction Heatmap for class: {class_names[class_idx]}...")
    
    if isinstance(shap_interaction_values, list):
        interaction_matrix = shap_interaction_values[class_idx]
    else:
        interaction_matrix = shap_interaction_values
    
    mean_interaction = np.abs(interaction_matrix).mean(axis=0)
    total_interaction = mean_interaction.sum(axis=1)
    top_indices = np.argsort(total_interaction)[::-1][:top_n]
    
    subset_matrix = mean_interaction[np.ix_(top_indices, top_indices)]
    subset_names = [feature_names[i] for i in top_indices]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(subset_matrix, cmap='Reds', aspect='auto')
    plt.colorbar(label='Mean |SHAP Interaction|')
    plt.xticks(range(len(subset_names)), subset_names, rotation=45, ha='right')
    plt.yticks(range(len(subset_names)), subset_names)
    plt.title(f'XGBoost Feature Interaction Heatmap: {class_names[class_idx]}', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_xgboost_importance_comparison(comparison_results, feature_names, top_n=15):
    """Create comparison plot for different feature importance methods."""
    print("Generating Comparison Plot for Feature Importance...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    methods = ['xgb_gain', 'xgb_weight', 'shap']
    titles = ['XGBoost Gain', 'XGBoost Weight', 'SHAP Values']
    colors = ['steelblue', 'forestgreen', 'coral']
    
    for ax, method, title, color in zip(axes, methods, titles, colors):
        importance = comparison_results[method]
        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        names = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        
        if max(values) > 0:
            values = [v / max(values) for v in values]
        
        ax.barh(range(len(names)), values[::-1], color=color)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names[::-1])
        ax.set_xlabel('Normalized Importance')
        ax.set_title(title, fontsize=12)
    
    plt.suptitle('XGBoost Feature Importance: Method Comparison', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_xgboost_decision_plot(shap_values, explainer, X_data, feature_names, class_names,
                               sample_indices=None, class_idx=0, max_display=15):
    """Create Decision Plot for XGBoost showing prediction path from base value."""
    shap_matrix = explainers.fix_shap_dimensions(shap_values, X_data, class_idx)

    if shap_matrix is None:
        return

    if hasattr(explainer, 'expected_value'):
        if hasattr(explainer.expected_value, '__iter__'):
            base_val = explainer.expected_value[class_idx]
        else:
            base_val = explainer.expected_value
    else:
        base_val = 0

    if sample_indices is None:
        sample_indices = list(range(min(10, len(X_data))))
    
    target_label = class_names[class_idx]
    print(f"\n--- Decision Plot (XGBoost - Class: {target_label}) ---")

    plt.figure(figsize=(10, 8))
    shap.decision_plot(
        base_val,
        shap_matrix[sample_indices],
        X_data[sample_indices],
        feature_names=feature_names,
        show=False
    )
    plt.title(f'XGBoost Decision Plot: {target_label}', fontsize=14)
    plt.show()


def plot_xgboost_tree(model, tree_index=0, figsize=(20, 10)):
    """Visualize XGBoost tree structure."""
    try:
        import xgboost as xgb
        
        print(f"Generating Tree Plot for tree #{tree_index}...")
        
        fig, ax = plt.subplots(figsize=figsize)
        xgb.plot_tree(model, num_trees=tree_index, ax=ax)
        plt.title(f'XGBoost Tree #{tree_index}', fontsize=14)
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("❌ xgboost library required for plot_xgboost_tree")
    except Exception as e:
        print(f"❌ Failed to create tree plot: {e}")