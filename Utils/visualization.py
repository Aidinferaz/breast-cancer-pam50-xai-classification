import shap
import matplotlib.pyplot as plt
import explainer  # Import modul explainers yang kita buat di atas


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