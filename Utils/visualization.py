import shap
import matplotlib.pyplot as plt
from Utils import explainer as explainers
from interpret import show
import numpy as np

def plot_beeswarm(shap_values, X_data, feature_names, class_names, target_class='all', max_display=15):
    """
    Membuat Global Summary Plot (Beeswarm).
    """
    # Tentukan index kelas
    indices_to_plot = []
    if target_class == 'all':
        indices_to_plot = range(len(class_names))
    elif isinstance(target_class, int):
        indices_to_plot = [target_class]
    elif isinstance(target_class, str):
        if target_class in class_names:
            indices_to_plot = [list(class_names).index(target_class)]
        else:
            print(f"❌ Kelas '{target_class}' tidak ditemukan.")
            return

    for idx in indices_to_plot:
        label = class_names[idx]
        # Panggil fungsi fix dimensions dari explainers.py
        matrix = explainers.fix_shap_dimensions(shap_values, X_data, idx)

        if matrix is not None:
            plt.figure(figsize=(10, 6))
            print(f"Generasi Plot Summary untuk: {label}...")
            shap.summary_plot(matrix, X_data, feature_names=feature_names, show=False, max_display=max_display)
            plt.title(f"Feature Importance: {label}", fontsize=14)
            plt.show()
        else:
            print(f"⚠️ Gagal memproses dimensi untuk kelas {label}")


def plot_waterfall(shap_values, explainer, X_data, feature_names, class_names, sample_idx=0, class_idx=0,
                   max_display=10):
    """
    Membuat Local Waterfall Plot untuk satu pasien.
    """
    # 1. Ambil Matrix SHAP yang benar dari explainers.py
    shap_matrix = explainers.fix_shap_dimensions(shap_values, X_data, class_idx)

    if shap_matrix is None:
        return

    # 2. Ambil Base Value (Expected Value)
    if hasattr(explainer.expected_value, '__iter__'):
        base_val = explainer.expected_value[class_idx]
    else:
        base_val = explainer.expected_value

    # 3. Buat Objek Explanation
    try:
        explanation = shap.Explanation(
            values=shap_matrix[sample_idx],
            base_values=base_val,
            data=X_data[sample_idx],
            feature_names=feature_names
        )

        target_label = class_names[class_idx]
        print(f"\n--- Analisis Lokal Pasien #{sample_idx} (Kelas Target: {target_label}) ---")

        shap.plots.waterfall(explanation, max_display=max_display)

    except Exception as e:
        print(f"❌ Gagal membuat plot waterfall: {e}")
def plot_ebm_dashboard(ebm_global, title="EBM Global Explanation"):
    """
    Menampilkan Dashboard Interaktif EBM.
    Fitur ini akan membuka widget interaktif di Jupyter Notebook.
    """
    print(f"Membuka Dashboard: {title}...")
    # show() dari library interpret akan merender widget
    show(ebm_global)

def plot_ebm_feature_curve(ebm_global, feature_index=0):
    """
    Menampilkan grafik kurva fungsi (shape function) untuk fitur tertentu.
    Ini menunjukkan bagaimana risiko berubah seiring nilai gen naik/turun.
    """
    # Mengambil data grafik dari objek penjelasan
    data = ebm_global.visualize(feature_index)
    return data # Di notebook, objek ini akan otomatis ter-render

def plot_nn_feature_importance(shap_values, feature_names, class_names, top_n=20):
    """
    Membuat bar plot untuk feature importance dari Neural Network.
    Mengagregasi SHAP values dari semua kelas.
    
    Parameters:
    -----------
    shap_values : list of arrays, SHAP values per kelas
    feature_names : list, Nama fitur
    class_names : list, Nama kelas
    top_n : int, Jumlah fitur teratas yang ditampilkan
    """
    print(f"Generasi Feature Importance Plot untuk Neural Network...")
    
    # Agregasi: rata-rata absolute SHAP value per fitur (across all classes)
    if isinstance(shap_values, list):
        all_shap = np.stack([np.abs(sv).mean(axis=0) for sv in shap_values])
        mean_importance = all_shap.mean(axis=0)
    else:
        mean_importance = np.abs(shap_values).mean(axis=0)
    
    # Sort dan ambil top N
    sorted_idx = np.argsort(mean_importance)[::-1][:top_n]
    sorted_importance = mean_importance[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]
    
    # Plot
    plt.figure(figsize=(10, 8))
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(sorted_idx)))
    plt.barh(range(len(sorted_idx)), sorted_importance[::-1], color=colors[::-1])
    plt.yticks(range(len(sorted_idx)), sorted_names[::-1])
    plt.xlabel('Mean |SHAP Value|')
    plt.title(f'Top {top_n} Feature Importance (Neural Network)', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_nn_beeswarm(shap_values, X_data, feature_names, class_names, target_class='all', max_display=15):
    """
    Membuat Global Summary Plot (Beeswarm) untuk Neural Network.
    
    Parameters:
    -----------
    shap_values : list of arrays, SHAP values dari Neural Network
    X_data : array-like, Data fitur
    feature_names : list, Nama fitur
    class_names : list, Nama kelas
    target_class : str/int, Kelas target ('all' untuk semua kelas)
    max_display : int, Jumlah fitur maksimum yang ditampilkan
    """
    indices_to_plot = []
    if target_class == 'all':
        indices_to_plot = range(len(class_names))
    elif isinstance(target_class, int):
        indices_to_plot = [target_class]
    elif isinstance(target_class, str):
        if target_class in class_names:
            indices_to_plot = [list(class_names).index(target_class)]
        else:
            print(f"❌ Kelas '{target_class}' tidak ditemukan.")
            return

    for idx in indices_to_plot:
        label = class_names[idx]
        matrix = explainers.fix_shap_dimensions(shap_values, X_data, idx)

        if matrix is not None:
            plt.figure(figsize=(10, 6))
            print(f"Generasi Plot Summary untuk: {label} (Neural Network)...")
            shap.summary_plot(matrix, X_data, feature_names=feature_names, show=False, max_display=max_display)
            plt.title(f"Neural Network Feature Importance: {label}", fontsize=14)
            plt.show()
        else:
            print(f"⚠️ Gagal memproses dimensi untuk kelas {label}")


def plot_nn_waterfall(shap_values, explainer, X_data, feature_names, class_names, 
                      sample_idx=0, class_idx=0, max_display=10):
    """
    Membuat Local Waterfall Plot untuk Neural Network.
    
    Parameters:
    -----------
    shap_values : list of arrays, SHAP values
    explainer : SHAP explainer object
    X_data : array-like, Data fitur
    feature_names : list, Nama fitur
    class_names : list, Nama kelas
    sample_idx : int, Index sampel yang akan dijelaskan
    class_idx : int, Index kelas target
    max_display : int, Jumlah fitur maksimum
    """
    shap_matrix = explainers.fix_shap_dimensions(shap_values, X_data, class_idx)

    if shap_matrix is None:
        return

    # Ambil Base Value
    if hasattr(explainer, 'expected_value'):
        if hasattr(explainer.expected_value, '__iter__'):
            base_val = explainer.expected_value[class_idx]
        else:
            base_val = explainer.expected_value
    else:
        # Untuk DeepExplainer, expected_value mungkin tidak ada
        base_val = 0

    try:
        explanation = shap.Explanation(
            values=shap_matrix[sample_idx],
            base_values=base_val,
            data=X_data[sample_idx],
            feature_names=feature_names
        )

        target_label = class_names[class_idx]
        print(f"\n--- Analisis Lokal Pasien #{sample_idx} (Neural Network - Kelas: {target_label}) ---")

        shap.plots.waterfall(explanation, max_display=max_display)

    except Exception as e:
        print(f"❌ Gagal membuat plot waterfall: {e}")


def plot_nn_force(shap_values, explainer, X_data, feature_names, class_names,
                  sample_idx=0, class_idx=0):
    """
    Membuat Force Plot untuk Neural Network (visualisasi alternatif).
    
    Parameters:
    -----------
    shap_values : list of arrays, SHAP values
    explainer : SHAP explainer object
    X_data : array-like, Data fitur
    feature_names : list, Nama fitur
    class_names : list, Nama kelas
    sample_idx : int, Index sampel
    class_idx : int, Index kelas target
    """
    shap_matrix = explainers.fix_shap_dimensions(shap_values, X_data, class_idx)

    if shap_matrix is None:
        return

    # Ambil Base Value
    if hasattr(explainer, 'expected_value'):
        if hasattr(explainer.expected_value, '__iter__'):
            base_val = explainer.expected_value[class_idx]
        else:
            base_val = explainer.expected_value
    else:
        base_val = 0

    target_label = class_names[class_idx]
    print(f"\n--- Force Plot Pasien #{sample_idx} (Neural Network - Kelas: {target_label}) ---")

    shap.initjs()
    force_plot = shap.force_plot(
        base_val,
        shap_matrix[sample_idx],
        X_data[sample_idx],
        feature_names=feature_names
    )
    return force_plot


def plot_integrated_gradients(ig_values, feature_names, sample_idx=0, top_n=15):
    """
    Visualisasi Integrated Gradients untuk Neural Network.
    
    Parameters:
    -----------
    ig_values : array, Integrated Gradients attributions
    feature_names : list, Nama fitur
    sample_idx : int, Index sampel
    top_n : int, Jumlah fitur teratas
    """
    print(f"\n--- Integrated Gradients untuk Sampel #{sample_idx} ---")
    
    sample_ig = ig_values[sample_idx]
    
    # Sort berdasarkan absolute value
    sorted_idx = np.argsort(np.abs(sample_ig))[::-1][:top_n]
    sorted_ig = sample_ig[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]
    
    # Plot
    plt.figure(figsize=(10, 6))
    colors = ['red' if x > 0 else 'blue' for x in sorted_ig]
    plt.barh(range(len(sorted_idx)), sorted_ig[::-1], color=colors[::-1])
    plt.yticks(range(len(sorted_idx)), sorted_names[::-1])
    plt.xlabel('Attribution Score')
    plt.title(f'Integrated Gradients - Sample #{sample_idx}', fontsize=14)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.show()