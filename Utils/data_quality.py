"""
Data Quality Assessment Module

This module provides comprehensive data quality checks for gene expression data,
including outlier detection, variance filtering, and quality metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def validate_data_quality(X, y=None, feature_names=None, verbose=True):
    """
    Comprehensive data quality checks.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    y : array-like, shape (n_samples,), optional
        Target labels
    feature_names : list, optional
        Feature names
    verbose : bool, default=True
        Print detailed report
    
    Returns:
    --------
    report : dict
        Dictionary containing quality metrics
    """
    report = {}
    
    # Convert to numpy array if needed
    if isinstance(X, pd.DataFrame):
        if feature_names is None:
            feature_names = X.columns.tolist()
        X = X.values
    
    n_samples, n_features = X.shape
    report['n_samples'] = n_samples
    report['n_features'] = n_features
    
    # Check for missing values
    n_missing = np.isnan(X).sum()
    report['missing_values'] = int(n_missing)
    report['missing_percentage'] = float((n_missing / X.size) * 100)
    
    # Check for infinite values
    n_inf = np.isinf(X).sum()
    report['infinite_values'] = int(n_inf)
    
    # Check for constant features (zero variance)
    variances = np.var(X, axis=0)
    n_constant = np.sum(variances == 0)
    report['constant_features'] = int(n_constant)
    
    # Check for low-variance features
    n_low_var = np.sum(variances < 0.01)
    report['low_variance_features'] = int(n_low_var)
    
    # Check for outliers (samples with extreme Z-scores)
    with np.errstate(divide='ignore', invalid='ignore'):
        z_scores = np.abs(stats.zscore(X, axis=0, nan_policy='omit'))
        z_scores = np.nan_to_num(z_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    outlier_samples = np.any(z_scores > 3, axis=1)
    report['outlier_samples'] = int(np.sum(outlier_samples))
    report['outlier_percentage'] = float((np.sum(outlier_samples) / n_samples) * 100)
    
    # Check for highly correlated features (if not too many features)
    if n_features <= 5000:
        try:
            corr_matrix = np.corrcoef(X.T)
            # Set diagonal to 0 to exclude self-correlation
            np.fill_diagonal(corr_matrix, 0)
            high_corr_pairs = np.sum(np.abs(corr_matrix) > 0.95) // 2
            report['highly_correlated_pairs'] = int(high_corr_pairs)
        except:
            report['highly_correlated_pairs'] = 'Not computed (too large)'
    else:
        report['highly_correlated_pairs'] = 'Not computed (too many features)'
    
    # Class balance check (if y is provided)
    if y is not None:
        unique, counts = np.unique(y, return_counts=True)
        class_balance = dict(zip([int(u) for u in unique], [int(c) for c in counts]))
        report['class_balance'] = class_balance
        
        # Check for class imbalance
        min_samples = counts.min()
        max_samples = counts.max()
        imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
        report['class_imbalance_ratio'] = float(imbalance_ratio)
    
    # Feature statistics
    report['feature_stats'] = {
        'mean_variance': float(np.mean(variances)),
        'median_variance': float(np.median(variances)),
        'min_variance': float(np.min(variances)),
        'max_variance': float(np.max(variances))
    }
    
    if verbose:
        print("=" * 70)
        print("DATA QUALITY REPORT")
        print("=" * 70)
        print(f"\nDataset Shape: {n_samples} samples × {n_features} features")
        print(f"\nMissing Values: {report['missing_values']} ({report['missing_percentage']:.2f}%)")
        print(f"Infinite Values: {report['infinite_values']}")
        print(f"Constant Features: {report['constant_features']}")
        print(f"Low Variance Features (< 0.01): {report['low_variance_features']}")
        print(f"\nOutlier Samples (Z-score > 3): {report['outlier_samples']} ({report['outlier_percentage']:.2f}%)")
        print(f"Highly Correlated Feature Pairs (|r| > 0.95): {report['highly_correlated_pairs']}")
        
        if y is not None:
            print(f"\nClass Balance:")
            for class_id, count in report['class_balance'].items():
                print(f"  Class {class_id}: {count} samples")
            print(f"Class Imbalance Ratio (max/min): {report['class_imbalance_ratio']:.2f}")
        
        print(f"\nFeature Variance Statistics:")
        print(f"  Mean: {report['feature_stats']['mean_variance']:.4f}")
        print(f"  Median: {report['feature_stats']['median_variance']:.4f}")
        print(f"  Range: [{report['feature_stats']['min_variance']:.4f}, {report['feature_stats']['max_variance']:.4f}]")
        print("=" * 70)
    
    return report


def detect_outliers(X, threshold=3, method='zscore'):
    """
    Detect outlier samples using specified method.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    threshold : float, default=3
        Threshold for outlier detection
    method : str, default='zscore'
        Method to use: 'zscore', 'iqr', 'isolation_forest'
    
    Returns:
    --------
    outlier_indices : array
        Indices of outlier samples
    outlier_scores : array
        Outlier scores for each sample
    """
    if method == 'zscore':
        # Z-score method: flag samples with any feature having |Z| > threshold
        with np.errstate(divide='ignore', invalid='ignore'):
            z_scores = np.abs(stats.zscore(X, axis=0, nan_policy='omit'))
            z_scores = np.nan_to_num(z_scores, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Max Z-score per sample
        outlier_scores = np.max(z_scores, axis=1)
        outlier_mask = outlier_scores > threshold
        outlier_indices = np.where(outlier_mask)[0]
        
    elif method == 'iqr':
        # IQR method: flag samples outside Q1 - 1.5*IQR or Q3 + 1.5*IQR
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outlier_mask = np.any((X < lower_bound) | (X > upper_bound), axis=1)
        outlier_indices = np.where(outlier_mask)[0]
        
        # Score based on number of features outside bounds
        outlier_scores = np.sum((X < lower_bound) | (X > upper_bound), axis=1)
        
    elif method == 'isolation_forest':
        from sklearn.ensemble import IsolationForest
        
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_pred = iso_forest.fit_predict(X)
        outlier_scores = -iso_forest.score_samples(X)  # Higher score = more outlier
        outlier_indices = np.where(outlier_pred == -1)[0]
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return outlier_indices, outlier_scores


def filter_low_variance_genes(X, feature_names=None, threshold=0.01, method='variance'):
    """
    Remove genes with low variance.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    feature_names : list, optional
        Feature names
    threshold : float, default=0.01
        Variance threshold
    method : str, default='variance'
        Method: 'variance', 'mad' (median absolute deviation), 'iqr'
    
    Returns:
    --------
    X_filtered : array
        Filtered feature matrix
    selected_indices : array
        Indices of selected features
    selected_names : list or None
        Names of selected features (if feature_names provided)
    """
    if method == 'variance':
        scores = np.var(X, axis=0)
    elif method == 'mad':
        medians = np.median(X, axis=0)
        scores = np.median(np.abs(X - medians), axis=0)
    elif method == 'iqr':
        scores = stats.iqr(X, axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    selected_mask = scores > threshold
    selected_indices = np.where(selected_mask)[0]
    
    X_filtered = X[:, selected_indices]
    
    selected_names = None
    if feature_names is not None:
        selected_names = [feature_names[i] for i in selected_indices]
    
    print(f"Filtered {len(selected_indices)}/{X.shape[1]} features using {method} > {threshold}")
    
    return X_filtered, selected_indices, selected_names


def check_batch_effects(X, labels=None, batch_labels=None, n_components=2, plot=True):
    """
    Visualize potential batch effects using PCA.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    labels : array-like, optional
        Class labels for coloring
    batch_labels : array-like, optional
        Batch labels for marker shapes
    n_components : int, default=2
        Number of PCA components
    plot : bool, default=True
        Whether to create visualization
    
    Returns:
    --------
    X_pca : array
        PCA-transformed data
    explained_variance : array
        Explained variance ratios
    """
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    explained_variance = pca.explained_variance_ratio_
    
    if plot and n_components >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                
                if batch_labels is not None:
                    # Different markers for batches
                    unique_batches = np.unique(batch_labels[mask])
                    markers = ['o', 's', '^', 'v', '<', '>', 'd', 'p', '*', 'h']
                    
                    for j, batch in enumerate(unique_batches):
                        batch_mask = mask & (batch_labels == batch)
                        ax.scatter(
                            X_pca[batch_mask, 0], X_pca[batch_mask, 1],
                            c=[colors[i]], marker=markers[j % len(markers)],
                            label=f"Class {label}, Batch {batch}",
                            alpha=0.6, s=50
                        )
                else:
                    ax.scatter(
                        X_pca[mask, 0], X_pca[mask, 1],
                        c=[colors[i]], label=f"Class {label}",
                        alpha=0.6, s=50
                    )
        else:
            ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=50)
        
        ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}% variance)')
        ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}% variance)')
        ax.set_title('PCA Visualization - Check for Batch Effects')
        
        if labels is not None:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
    
    print(f"\nPCA Explained Variance:")
    for i, var in enumerate(explained_variance):
        print(f"  PC{i+1}: {var*100:.2f}%")
    print(f"  Total: {sum(explained_variance)*100:.2f}%")
    
    return X_pca, explained_variance


def plot_data_quality_summary(X, y=None, feature_names=None, figsize=(15, 10)):
    """
    Create comprehensive data quality visualization.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    y : array-like, optional
        Target labels
    feature_names : list, optional
        Feature names
    figsize : tuple, default=(15, 10)
        Figure size
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # 1. Feature variance distribution
    variances = np.var(X, axis=0)
    axes[0, 0].hist(variances, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Variance')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Feature Variance Distribution')
    axes[0, 0].axvline(0.01, color='red', linestyle='--', label='Low var threshold')
    axes[0, 0].legend()
    
    # 2. Sample correlation heatmap (first 50 samples)
    n_samples_to_show = min(50, X.shape[0])
    sample_corr = np.corrcoef(X[:n_samples_to_show])
    im = axes[0, 1].imshow(sample_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[0, 1].set_title(f'Sample Correlation (first {n_samples_to_show} samples)')
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('Sample Index')
    plt.colorbar(im, ax=axes[0, 1])
    
    # 3. Missing values per feature
    missing_per_feature = np.isnan(X).sum(axis=0)
    if missing_per_feature.sum() > 0:
        top_missing_idx = np.argsort(missing_per_feature)[-20:]
        axes[0, 2].barh(range(len(top_missing_idx)), missing_per_feature[top_missing_idx])
        axes[0, 2].set_xlabel('Missing Count')
        axes[0, 2].set_title('Top 20 Features with Missing Values')
    else:
        axes[0, 2].text(0.5, 0.5, 'No Missing Values', ha='center', va='center')
        axes[0, 2].set_title('Missing Values')
    
    # 4. Outlier detection
    z_scores = np.abs(stats.zscore(X, axis=0, nan_policy='omit'))
    z_scores = np.nan_to_num(z_scores, nan=0.0)
    max_z_per_sample = np.max(z_scores, axis=1)
    axes[1, 0].hist(max_z_per_sample, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(3, color='red', linestyle='--', label='Outlier threshold')
    axes[1, 0].set_xlabel('Max Z-score per Sample')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Outlier Detection (Z-score)')
    axes[1, 0].legend()
    
    # 5. PCA visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
    
    if y is not None:
        unique_labels = np.unique(y)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        for i, label in enumerate(unique_labels):
            mask = y == label
            axes[1, 1].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                             c=[colors[i]], label=f'Class {label}', alpha=0.6, s=30)
        axes[1, 1].legend()
    else:
        axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=30)
    
    axes[1, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[1, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[1, 1].set_title('PCA Visualization')
    
    # 6. Class distribution (if y provided)
    if y is not None:
        unique, counts = np.unique(y, return_counts=True)
        axes[1, 2].bar(unique, counts, edgecolor='black', alpha=0.7)
        axes[1, 2].set_xlabel('Class')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title('Class Distribution')
        axes[1, 2].set_xticks(unique)
    else:
        axes[1, 2].text(0.5, 0.5, 'No Labels Provided', ha='center', va='center')
        axes[1, 2].set_title('Class Distribution')
    
    plt.tight_layout()
    plt.show()


def compare_normalization_methods(X_train, X_val, method_names=None):
    """
    Compare different normalization methods on the same data.
    
    Parameters:
    -----------
    X_train : array-like
        Training data
    X_val : array-like
        Validation data
    method_names : list, optional
        List of methods to compare: 'standard', 'robust', 'quantile', 'minmax'
    
    Returns:
    --------
    results : dict
        Dictionary with normalized data for each method
    """
    from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, MinMaxScaler
    
    if method_names is None:
        method_names = ['standard', 'robust', 'quantile']
    
    results = {}
    
    scalers = {
        'standard': StandardScaler(),
        'robust': RobustScaler(),
        'quantile': QuantileTransformer(output_distribution='normal'),
        'minmax': MinMaxScaler()
    }
    
    for method in method_names:
        if method not in scalers:
            print(f"Warning: Unknown method '{method}', skipping...")
            continue
        
        scaler = scalers[method]
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        results[method] = {
            'train': X_train_scaled,
            'val': X_val_scaled,
            'scaler': scaler
        }
        
        print(f"\n{method.upper()} Normalization:")
        print(f"  Train - Mean: {X_train_scaled.mean():.6f}, Std: {X_train_scaled.std():.6f}")
        print(f"  Train - Range: [{X_train_scaled.min():.4f}, {X_train_scaled.max():.4f}]")
        print(f"  Val - Mean: {X_val_scaled.mean():.6f}, Std: {X_val_scaled.std():.6f}")
        print(f"  Val - Range: [{X_val_scaled.min():.4f}, {X_val_scaled.max():.4f}]")
    
    return results
