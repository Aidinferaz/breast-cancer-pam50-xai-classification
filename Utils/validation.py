"""
Validation Module for Model Evaluation

This module provides cross-validation, external validation, and model comparison tools.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             make_scorer)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def cross_validate_model(model, X, y, cv=5, scoring=None, return_train_score=True):
    """
    Perform stratified k-fold cross-validation with comprehensive metrics.
    
    Parameters:
    -----------
    model : estimator
        Scikit-learn compatible model
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    y : array-like, shape (n_samples,)
        Target labels
    cv : int, default=5
        Number of folds
    scoring : dict or list, optional
        Scoring metrics to use
    return_train_score : bool, default=True
        Whether to return training scores
    
    Returns:
    --------
    results : dict
        Dictionary containing cross-validation results
    """
    if scoring is None:
        # Default scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision_macro': 'precision_macro',
            'recall_macro': 'recall_macro',
            'f1_macro': 'f1_macro'
        }
    
    # Create stratified k-fold
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Perform cross-validation
    cv_results = cross_validate(
        model, X, y, cv=skf,
        scoring=scoring,
        return_train_score=return_train_score,
        n_jobs=-1
    )
    
    # Summarize results
    results = {
        'cv_folds': cv,
        'metrics': {}
    }
    
    for metric_name in scoring.keys():
        test_key = f'test_{metric_name}'
        train_key = f'train_{metric_name}'
        
        test_scores = cv_results[test_key]
        
        results['metrics'][metric_name] = {
            'test_scores': test_scores,
            'test_mean': np.mean(test_scores),
            'test_std': np.std(test_scores),
            'test_min': np.min(test_scores),
            'test_max': np.max(test_scores),
            'confidence_interval_95': (
                np.mean(test_scores) - 1.96 * np.std(test_scores),
                np.mean(test_scores) + 1.96 * np.std(test_scores)
            )
        }
        
        if return_train_score and train_key in cv_results:
            train_scores = cv_results[train_key]
            results['metrics'][metric_name]['train_scores'] = train_scores
            results['metrics'][metric_name]['train_mean'] = np.mean(train_scores)
            results['metrics'][metric_name]['train_std'] = np.std(train_scores)
            results['metrics'][metric_name]['overfit_gap'] = np.mean(train_scores) - np.mean(test_scores)
    
    return results


def print_cv_results(cv_results, model_name="Model"):
    """Print cross-validation results in a readable format."""
    print("=" * 70)
    print(f"CROSS-VALIDATION RESULTS: {model_name}")
    print("=" * 70)
    print(f"Number of folds: {cv_results['cv_folds']}")
    print()
    
    for metric_name, metric_data in cv_results['metrics'].items():
        print(f"{metric_name.upper()}:")
        print(f"  Test Mean:  {metric_data['test_mean']:.4f} ± {metric_data['test_std']:.4f}")
        print(f"  Test Range: [{metric_data['test_min']:.4f}, {metric_data['test_max']:.4f}]")
        print(f"  95% CI:     [{metric_data['confidence_interval_95'][0]:.4f}, "
              f"{metric_data['confidence_interval_95'][1]:.4f}]")
        
        if 'train_mean' in metric_data:
            print(f"  Train Mean: {metric_data['train_mean']:.4f}")
            print(f"  Overfit Gap: {metric_data['overfit_gap']:.4f}")
        print()
    
    print("=" * 70)


def compare_models_cv(models_dict, X, y, cv=5, scoring='accuracy'):
    """
    Compare multiple models using cross-validation.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of {model_name: model_instance}
    X : array-like
        Feature matrix
    y : array-like
        Target labels
    cv : int, default=5
        Number of folds
    scoring : str, default='accuracy'
        Scoring metric
    
    Returns:
    --------
    comparison_df : DataFrame
        Comparison results
    """
    results = []
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    for model_name, model in models_dict.items():
        print(f"Evaluating {model_name}...")
        scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
        
        results.append({
            'Model': model_name,
            'Mean Score': np.mean(scores),
            'Std Dev': np.std(scores),
            'Min Score': np.min(scores),
            'Max Score': np.max(scores),
            '95% CI Lower': np.mean(scores) - 1.96 * np.std(scores),
            '95% CI Upper': np.mean(scores) + 1.96 * np.std(scores)
        })
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('Mean Score', ascending=False)
    
    return comparison_df


def statistical_comparison_test(scores1, scores2, test='paired_t'):
    """
    Perform statistical test to compare two models.
    
    Parameters:
    -----------
    scores1 : array-like
        CV scores for model 1
    scores2 : array-like
        CV scores for model 2
    test : str, default='paired_t'
        Test to use: 'paired_t', 'wilcoxon'
    
    Returns:
    --------
    result : dict
        Test results
    """
    if test == 'paired_t':
        statistic, p_value = stats.ttest_rel(scores1, scores2)
        test_name = "Paired t-test"
    elif test == 'wilcoxon':
        statistic, p_value = stats.wilcoxon(scores1, scores2)
        test_name = "Wilcoxon signed-rank test"
    else:
        raise ValueError(f"Unknown test: {test}")
    
    result = {
        'test': test_name,
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'mean_diff': np.mean(scores1) - np.mean(scores2)
    }
    
    return result


def bootstrap_confidence_interval(y_true, y_pred, metric='accuracy', n_bootstrap=1000, alpha=0.05):
    """
    Compute bootstrap confidence interval for a metric.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    metric : str or callable
        Metric to compute
    n_bootstrap : int, default=1000
        Number of bootstrap samples
    alpha : float, default=0.05
        Significance level for CI
    
    Returns:
    --------
    ci : dict
        Confidence interval results
    """
    n_samples = len(y_true)
    
    # Metric function
    if metric == 'accuracy':
        metric_func = accuracy_score
    elif metric == 'f1_macro':
        metric_func = lambda y_t, y_p: f1_score(y_t, y_p, average='macro')
    elif metric == 'precision_macro':
        metric_func = lambda y_t, y_p: precision_score(y_t, y_p, average='macro')
    elif metric == 'recall_macro':
        metric_func = lambda y_t, y_p: recall_score(y_t, y_p, average='macro')
    elif callable(metric):
        metric_func = metric
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Bootstrap sampling
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Compute metric
        score = metric_func(y_true_boot, y_pred_boot)
        bootstrap_scores.append(score)
    
    bootstrap_scores = np.array(bootstrap_scores)
    
    # Compute confidence interval
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci = {
        'mean': np.mean(bootstrap_scores),
        'std': np.std(bootstrap_scores),
        'lower': np.percentile(bootstrap_scores, lower_percentile),
        'upper': np.percentile(bootstrap_scores, upper_percentile),
        'bootstrap_scores': bootstrap_scores
    }
    
    return ci


def evaluate_with_confidence(model, X, y, class_names=None, n_bootstrap=1000):
    """
    Evaluate model with bootstrap confidence intervals.
    
    Parameters:
    -----------
    model : estimator
        Trained model
    X : array-like
        Feature matrix
    y : array-like
        True labels
    class_names : list, optional
        Class names
    n_bootstrap : int, default=1000
        Number of bootstrap samples
    
    Returns:
    --------
    results : dict
        Evaluation results with confidence intervals
    """
    # Get predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
    
    # Compute metrics with confidence intervals
    results = {}
    
    # Accuracy
    acc_ci = bootstrap_confidence_interval(y, y_pred, metric='accuracy', n_bootstrap=n_bootstrap)
    results['accuracy'] = acc_ci
    
    # F1 score
    f1_ci = bootstrap_confidence_interval(y, y_pred, metric='f1_macro', n_bootstrap=n_bootstrap)
    results['f1_macro'] = f1_ci
    
    # Precision
    prec_ci = bootstrap_confidence_interval(y, y_pred, metric='precision_macro', n_bootstrap=n_bootstrap)
    results['precision_macro'] = prec_ci
    
    # Recall
    rec_ci = bootstrap_confidence_interval(y, y_pred, metric='recall_macro', n_bootstrap=n_bootstrap)
    results['recall_macro'] = rec_ci
    
    # Confusion matrix
    results['confusion_matrix'] = confusion_matrix(y, y_pred)
    
    # Classification report
    results['classification_report'] = classification_report(y, y_pred, target_names=class_names)
    
    # Per-class metrics (if multi-class)
    if y_prob is not None and len(np.unique(y)) > 2:
        from sklearn.preprocessing import label_binarize
        n_classes = len(np.unique(y))
        y_bin = label_binarize(y, classes=range(n_classes))
        
        per_class_auc = {}
        for i in range(n_classes):
            try:
                auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
                per_class_auc[i] = auc
            except:
                per_class_auc[i] = np.nan
        
        results['per_class_auc'] = per_class_auc
    
    return results


def print_evaluation_results(results, model_name="Model"):
    """Print evaluation results with confidence intervals."""
    print("=" * 70)
    print(f"EVALUATION RESULTS: {model_name}")
    print("=" * 70)
    
    metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    for metric in metrics:
        if metric in results:
            data = results[metric]
            print(f"\n{metric.upper()}:")
            print(f"  Score: {data['mean']:.4f}")
            print(f"  95% CI: [{data['lower']:.4f}, {data['upper']:.4f}]")
            print(f"  Std: {data['std']:.4f}")
    
    if 'confusion_matrix' in results:
        print("\nCONFUSION MATRIX:")
        print(results['confusion_matrix'])
    
    if 'classification_report' in results:
        print("\nCLASSIFICATION REPORT:")
        print(results['classification_report'])
    
    if 'per_class_auc' in results:
        print("\nPER-CLASS AUC:")
        for class_idx, auc in results['per_class_auc'].items():
            print(f"  Class {class_idx}: {auc:.4f}")
    
    print("=" * 70)


def plot_cv_comparison(cv_results_dict, metric='accuracy', figsize=(10, 6)):
    """
    Plot cross-validation comparison for multiple models.
    
    Parameters:
    -----------
    cv_results_dict : dict
        Dictionary of {model_name: cv_results}
    metric : str, default='accuracy'
        Metric to plot
    figsize : tuple
        Figure size
    """
    models = list(cv_results_dict.keys())
    means = []
    stds = []
    
    for model_name in models:
        cv_res = cv_results_dict[model_name]
        if metric in cv_res['metrics']:
            means.append(cv_res['metrics'][metric]['test_mean'])
            stds.append(cv_res['metrics'][metric]['test_std'])
        else:
            means.append(0)
            stds.append(0)
    
    x = np.arange(len(models))
    
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, edgecolor='black')
    
    # Color bars by performance
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(models)))
    sorted_indices = np.argsort(means)
    for i, bar in enumerate(bars):
        bar.set_color(colors[np.where(sorted_indices == i)[0][0]])
    
    ax.set_xlabel('Model')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Cross-Validation {metric.replace("_", " ").title()} Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names=None, normalize=False, figsize=(8, 6)):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    cm : array-like
        Confusion matrix
    class_names : list, optional
        Class names
    normalize : bool, default=False
        Whether to normalize
    figsize : tuple
        Figure size
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    if class_names is not None:
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
    else:
        ax.set_xticks(np.arange(cm.shape[0]))
        ax.set_yticks(np.arange(cm.shape[1]))
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    title = 'Normalized Confusion Matrix' if normalize else 'Confusion Matrix'
    ax.set_title(title)
    
    plt.tight_layout()
    plt.show()


def nested_cross_validation(model, X, y, param_grid, cv_outer=5, cv_inner=3, scoring='accuracy'):
    """
    Perform nested cross-validation for unbiased performance estimation.
    
    Parameters:
    -----------
    model : estimator
        Model to evaluate
    X : array-like
        Feature matrix
    y : array-like
        Target labels
    param_grid : dict
        Parameter grid for hyperparameter tuning
    cv_outer : int, default=5
        Number of outer folds
    cv_inner : int, default=3
        Number of inner folds
    scoring : str, default='accuracy'
        Scoring metric
    
    Returns:
    --------
    results : dict
        Nested CV results
    """
    from sklearn.model_selection import GridSearchCV
    
    outer_cv = StratifiedKFold(n_splits=cv_outer, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=cv_inner, shuffle=True, random_state=42)
    
    outer_scores = []
    best_params_list = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Inner CV for hyperparameter tuning
        grid_search = GridSearchCV(
            model, param_grid, cv=inner_cv,
            scoring=scoring, n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        # Test on outer fold
        score = grid_search.score(X_test, y_test)
        outer_scores.append(score)
        best_params_list.append(grid_search.best_params_)
        
        print(f"Outer fold {fold_idx + 1}/{cv_outer}: Score = {score:.4f}, "
              f"Best params = {grid_search.best_params_}")
    
    results = {
        'outer_scores': outer_scores,
        'mean_score': np.mean(outer_scores),
        'std_score': np.std(outer_scores),
        'best_params_per_fold': best_params_list,
        'confidence_interval_95': (
            np.mean(outer_scores) - 1.96 * np.std(outer_scores),
            np.mean(outer_scores) + 1.96 * np.std(outer_scores)
        )
    }
    
    return results
