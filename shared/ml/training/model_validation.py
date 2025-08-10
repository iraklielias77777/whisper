"""
Model Validation and Testing Framework for User Whisperer Platform
Comprehensive validation including cross-validation, temporal validation, fairness, and interpretability
"""

import logging
import json
import os
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
import pandas as pd
from collections import defaultdict

# ML and validation imports
try:
    from sklearn.model_selection import cross_val_score, GridSearchCV, TimeSeriesSplit, StratifiedKFold
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, roc_curve, precision_recall_curve,
        confusion_matrix, classification_report,
        make_scorer, mean_squared_error, mean_absolute_error
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Validation capabilities will be limited.")

# Interpretability imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Model interpretability will be limited.")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logging.warning("LIME not available. Model interpretability will be limited.")

# Statistical imports
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Container for validation results"""
    validation_type: str
    model_name: str
    metrics: Dict[str, float]
    scores: Dict[str, List[float]]
    metadata: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'validation_type': self.validation_type,
            'model_name': self.model_name,
            'metrics': self.metrics,
            'scores': self.scores,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class FairnessMetrics:
    """Container for fairness metrics"""
    group_accuracy: Dict[str, float]
    group_precision: Dict[str, float]
    group_recall: Dict[str, float]
    demographic_parity: float
    equalized_odds: float
    statistical_parity_difference: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'group_accuracy': self.group_accuracy,
            'group_precision': self.group_precision,
            'group_recall': self.group_recall,
            'demographic_parity': self.demographic_parity,
            'equalized_odds': self.equalized_odds,
            'statistical_parity_difference': self.statistical_parity_difference
        }

class ModelValidator:
    """
    Comprehensive model validation framework
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.validation_results = {}
        self.cv_folds = self.config.get('cv_folds', 5)
        self.random_state = self.config.get('random_state', 42)
        
        logger.info("Initialized ModelValidator")
    
    def validate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        feature_names: Optional[List[str]] = None,
        sensitive_features: Optional[np.ndarray] = None
    ) -> Dict[str, ValidationResult]:
        """Run comprehensive validation suite"""
        
        logger.info(f"Starting comprehensive validation for {model_name}")
        
        results = {}
        
        try:
            # Cross-validation
            cv_result = self.cross_validate(model, X, y, model_name)
            results['cross_validation'] = cv_result
            
            # Temporal validation
            temporal_result = self.temporal_validate(model, X, y, model_name)
            results['temporal_validation'] = temporal_result
            
            # Adversarial validation
            adversarial_result = self.adversarial_validate(model, X, y, model_name)
            results['adversarial_validation'] = adversarial_result
            
            # Fairness validation
            if sensitive_features is not None:
                fairness_result = self.fairness_validate(model, X, y, sensitive_features, model_name)
                results['fairness_validation'] = fairness_result
            
            # Interpretability analysis
            interpretability_result = self.interpretability_analysis(
                model, X, y, model_name, feature_names
            )
            results['interpretability'] = interpretability_result
            
            # Stability validation
            stability_result = self.stability_validate(model, X, y, model_name)
            results['stability_validation'] = stability_result
            
            # Store results
            self.validation_results[model_name] = results
            
            logger.info(f"Comprehensive validation completed for {model_name}")
            
        except Exception as e:
            logger.error(f"Validation failed for {model_name}: {e}")
            results['error'] = str(e)
        
        return results
    
    def cross_validate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        cv_folds: Optional[int] = None
    ) -> ValidationResult:
        """Perform stratified cross-validation"""
        
        cv_folds = cv_folds or self.cv_folds
        
        logger.info(f"Running {cv_folds}-fold cross-validation for {model_name}")
        
        try:
            if not SKLEARN_AVAILABLE:
                # Manual cross-validation
                return self._manual_cross_validate(model, X, y, model_name, cv_folds)
            
            # Use stratified k-fold for classification
            if len(np.unique(y)) <= 10:  # Classification
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            else:  # Regression
                cv = cv_folds
            
            # Define scoring metrics
            scoring = {
                'accuracy': make_scorer(accuracy_score),
                'precision': make_scorer(precision_score, average='binary', zero_division=0),
                'recall': make_scorer(recall_score, average='binary', zero_division=0),
                'f1': make_scorer(f1_score, average='binary', zero_division=0)
            }
            
            # Add AUC for binary classification
            if len(np.unique(y)) == 2:
                scoring['auc'] = make_scorer(roc_auc_score)
            
            results = {}
            scores_dict = {}
            
            for metric_name, scorer in scoring.items():
                try:
                    scores = cross_val_score(
                        model, X, y,
                        cv=cv,
                        scoring=scorer,
                        n_jobs=-1
                    )
                    
                    results[f'{metric_name}_mean'] = scores.mean()
                    results[f'{metric_name}_std'] = scores.std()
                    results[f'{metric_name}_min'] = scores.min()
                    results[f'{metric_name}_max'] = scores.max()
                    
                    scores_dict[metric_name] = scores.tolist()
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate {metric_name} in CV: {e}")
                    results[f'{metric_name}_mean'] = 0.0
                    results[f'{metric_name}_std'] = 0.0
                    scores_dict[metric_name] = []
            
            return ValidationResult(
                validation_type='cross_validation',
                model_name=model_name,
                metrics=results,
                scores=scores_dict,
                metadata={'cv_folds': cv_folds, 'samples': len(X)},
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            return ValidationResult(
                validation_type='cross_validation',
                model_name=model_name,
                metrics={'error': str(e)},
                scores={},
                metadata={},
                timestamp=datetime.now()
            )
    
    def _manual_cross_validate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        cv_folds: int
    ) -> ValidationResult:
        """Manual cross-validation when sklearn is not available"""
        
        n_samples = len(X)
        fold_size = n_samples // cv_folds
        
        accuracies = []
        
        for fold in range(cv_folds):
            # Create train/validation split
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < cv_folds - 1 else n_samples
            
            # Validation set
            X_val = X[start_idx:end_idx]
            y_val = y[start_idx:end_idx]
            
            # Training set
            X_train = np.concatenate([X[:start_idx], X[end_idx:]])
            y_train = np.concatenate([y[:start_idx], y[end_idx:]])
            
            # Train and evaluate
            try:
                if hasattr(model, 'fit'):
                    model.fit(X_train, y_train)
                
                y_pred = model.predict(X_val)
                accuracy = np.mean(y_val == y_pred)
                accuracies.append(accuracy)
                
            except Exception as e:
                logger.warning(f"Manual CV fold {fold} failed: {e}")
                accuracies.append(0.0)
        
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        
        return ValidationResult(
            validation_type='cross_validation',
            model_name=model_name,
            metrics={
                'accuracy_mean': mean_accuracy,
                'accuracy_std': std_accuracy
            },
            scores={'accuracy': accuracies},
            metadata={'cv_folds': cv_folds, 'manual_cv': True},
            timestamp=datetime.now()
        )
    
    def temporal_validate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str
    ) -> ValidationResult:
        """Validate model performance across different time periods"""
        
        logger.info(f"Running temporal validation for {model_name}")
        
        try:
            # Assume data is ordered chronologically
            n_samples = len(X)
            
            # Define time periods (recent, mid, old)
            splits = {
                'recent': (int(n_samples * 0.8), n_samples),
                'mid': (int(n_samples * 0.4), int(n_samples * 0.8)),
                'old': (0, int(n_samples * 0.4))
            }
            
            results = {}
            scores_dict = {}
            
            for period, (start, end) in splits.items():
                if start >= end:
                    continue
                
                X_period = X[start:end]
                y_period = y[start:end]
                
                if len(X_period) == 0:
                    continue
                
                try:
                    y_pred = model.predict(X_period)
                    
                    # Calculate metrics for this period
                    accuracy = np.mean(y_period == y_pred)
                    
                    # Calculate additional metrics if possible
                    metrics = self._calculate_detailed_metrics(y_period, y_pred)
                    
                    results[f'{period}_accuracy'] = accuracy
                    results[f'{period}_samples'] = len(X_period)
                    
                    for metric, value in metrics.items():
                        results[f'{period}_{metric}'] = value
                    
                    scores_dict[period] = [accuracy]
                    
                except Exception as e:
                    logger.warning(f"Temporal validation failed for period {period}: {e}")
                    results[f'{period}_accuracy'] = 0.0
                    results[f'{period}_samples'] = len(X_period)
            
            # Calculate temporal stability
            period_accuracies = [
                results.get(f'{period}_accuracy', 0) 
                for period in ['old', 'mid', 'recent'] 
                if f'{period}_accuracy' in results
            ]
            
            if len(period_accuracies) > 1:
                temporal_stability = 1.0 - np.std(period_accuracies)
                results['temporal_stability'] = max(0.0, temporal_stability)
            
            return ValidationResult(
                validation_type='temporal_validation',
                model_name=model_name,
                metrics=results,
                scores=scores_dict,
                metadata={'periods': list(splits.keys())},
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Temporal validation failed: {e}")
            return ValidationResult(
                validation_type='temporal_validation',
                model_name=model_name,
                metrics={'error': str(e)},
                scores={},
                metadata={},
                timestamp=datetime.now()
            )
    
    def adversarial_validate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str
    ) -> ValidationResult:
        """Test model robustness with adversarial examples"""
        
        logger.info(f"Running adversarial validation for {model_name}")
        
        try:
            results = {}
            scores_dict = {}
            
            # Test with different noise levels
            noise_levels = [0.01, 0.05, 0.1, 0.2]
            
            # Get baseline performance
            try:
                y_pred_clean = model.predict(X)
                baseline_accuracy = np.mean(y == y_pred_clean)
                results['baseline_accuracy'] = baseline_accuracy
            except Exception as e:
                logger.warning(f"Failed to get baseline accuracy: {e}")
                baseline_accuracy = 0.0
                results['baseline_accuracy'] = 0.0
            
            noise_accuracies = []
            
            for noise_level in noise_levels:
                try:
                    # Add Gaussian noise to features
                    noise = np.random.normal(0, noise_level, X.shape)
                    X_noisy = X + noise
                    
                    # Predict on noisy data
                    y_pred_noisy = model.predict(X_noisy)
                    
                    # Calculate accuracy
                    noisy_accuracy = np.mean(y == y_pred_noisy)
                    
                    # Calculate performance degradation
                    degradation = baseline_accuracy - noisy_accuracy
                    
                    results[f'noise_{noise_level}_accuracy'] = noisy_accuracy
                    results[f'noise_{noise_level}_degradation'] = degradation
                    
                    noise_accuracies.append(noisy_accuracy)
                    
                except Exception as e:
                    logger.warning(f"Adversarial test failed for noise {noise_level}: {e}")
                    results[f'noise_{noise_level}_accuracy'] = 0.0
                    results[f'noise_{noise_level}_degradation'] = 1.0
                    noise_accuracies.append(0.0)
            
            # Calculate robustness score
            if noise_accuracies and baseline_accuracy > 0:
                avg_noisy_accuracy = np.mean(noise_accuracies)
                robustness_score = avg_noisy_accuracy / baseline_accuracy
                results['robustness_score'] = robustness_score
            else:
                results['robustness_score'] = 0.0
            
            scores_dict['noise_accuracies'] = noise_accuracies
            scores_dict['noise_levels'] = noise_levels
            
            # Test with feature perturbations
            try:
                perturbation_accuracies = self._test_feature_perturbations(model, X, y)
                results.update(perturbation_accuracies)
            except Exception as e:
                logger.warning(f"Feature perturbation test failed: {e}")
            
            return ValidationResult(
                validation_type='adversarial_validation',
                model_name=model_name,
                metrics=results,
                scores=scores_dict,
                metadata={'noise_levels': noise_levels},
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Adversarial validation failed: {e}")
            return ValidationResult(
                validation_type='adversarial_validation',
                model_name=model_name,
                metrics={'error': str(e)},
                scores={},
                metadata={},
                timestamp=datetime.now()
            )
    
    def _test_feature_perturbations(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """Test robustness to individual feature perturbations"""
        
        results = {}
        
        try:
            # Get baseline predictions
            y_pred_baseline = model.predict(X)
            baseline_accuracy = np.mean(y == y_pred_baseline)
            
            n_features = X.shape[1]
            feature_sensitivities = []
            
            # Test perturbation of each feature
            for feature_idx in range(min(n_features, 20)):  # Limit to first 20 features
                X_perturbed = X.copy()
                
                # Add noise to specific feature
                feature_std = np.std(X[:, feature_idx])
                noise = np.random.normal(0, feature_std * 0.1, X.shape[0])
                X_perturbed[:, feature_idx] += noise
                
                # Get predictions
                y_pred_perturbed = model.predict(X_perturbed)
                perturbed_accuracy = np.mean(y == y_pred_perturbed)
                
                # Calculate sensitivity
                sensitivity = baseline_accuracy - perturbed_accuracy
                feature_sensitivities.append(sensitivity)
            
            if feature_sensitivities:
                results['max_feature_sensitivity'] = max(feature_sensitivities)
                results['avg_feature_sensitivity'] = np.mean(feature_sensitivities)
                results['feature_sensitivity_std'] = np.std(feature_sensitivities)
            
        except Exception as e:
            logger.warning(f"Feature perturbation test failed: {e}")
        
        return results
    
    def fairness_validate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        sensitive_features: np.ndarray,
        model_name: str
    ) -> ValidationResult:
        """Validate model fairness across different groups"""
        
        logger.info(f"Running fairness validation for {model_name}")
        
        try:
            # Get predictions
            y_pred = model.predict(X)
            
            # Initialize results
            results = {}
            scores_dict = {}
            
            # Get unique groups
            unique_groups = np.unique(sensitive_features)
            
            # Calculate metrics for each group
            group_metrics = {}
            
            for group in unique_groups:
                mask = sensitive_features == group
                
                if np.sum(mask) == 0:
                    continue
                
                y_true_group = y[mask]
                y_pred_group = y_pred[mask]
                
                group_size = len(y_true_group)
                
                if group_size == 0:
                    continue
                
                # Calculate group-specific metrics
                group_accuracy = np.mean(y_true_group == y_pred_group)
                
                # Calculate precision and recall
                tp = np.sum((y_true_group == 1) & (y_pred_group == 1))
                fp = np.sum((y_true_group == 0) & (y_pred_group == 1))
                fn = np.sum((y_true_group == 1) & (y_pred_group == 0))
                
                group_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                group_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                
                # Positive prediction rate
                positive_rate = np.mean(y_pred_group == 1)
                
                group_metrics[str(group)] = {
                    'accuracy': group_accuracy,
                    'precision': group_precision,
                    'recall': group_recall,
                    'positive_rate': positive_rate,
                    'size': group_size
                }
                
                # Store in results
                results[f'group_{group}_accuracy'] = group_accuracy
                results[f'group_{group}_precision'] = group_precision
                results[f'group_{group}_recall'] = group_recall
                results[f'group_{group}_positive_rate'] = positive_rate
                results[f'group_{group}_size'] = group_size
            
            # Calculate fairness metrics
            fairness_metrics = self._calculate_fairness_metrics(group_metrics)
            results.update(fairness_metrics)
            
            scores_dict['group_metrics'] = group_metrics
            
            return ValidationResult(
                validation_type='fairness_validation',
                model_name=model_name,
                metrics=results,
                scores=scores_dict,
                metadata={'groups': unique_groups.tolist()},
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Fairness validation failed: {e}")
            return ValidationResult(
                validation_type='fairness_validation',
                model_name=model_name,
                metrics={'error': str(e)},
                scores={},
                metadata={},
                timestamp=datetime.now()
            )
    
    def _calculate_fairness_metrics(self, group_metrics: Dict) -> Dict[str, float]:
        """Calculate fairness metrics from group metrics"""
        
        fairness_results = {}
        
        try:
            if len(group_metrics) < 2:
                return fairness_results
            
            # Extract metrics by group
            group_accuracies = [metrics['accuracy'] for metrics in group_metrics.values()]
            group_positive_rates = [metrics['positive_rate'] for metrics in group_metrics.values()]
            group_precisions = [metrics['precision'] for metrics in group_metrics.values()]
            group_recalls = [metrics['recall'] for metrics in group_metrics.values()]
            
            # Demographic parity (difference in positive rates)
            max_positive_rate = max(group_positive_rates)
            min_positive_rate = min(group_positive_rates)
            demographic_parity_diff = max_positive_rate - min_positive_rate
            
            # Equalized odds (difference in TPR and FPR)
            max_recall = max(group_recalls)
            min_recall = min(group_recalls)
            equalized_odds_diff = max_recall - min_recall
            
            # Overall fairness score (lower is better)
            fairness_score = (demographic_parity_diff + equalized_odds_diff) / 2
            
            fairness_results.update({
                'demographic_parity_difference': demographic_parity_diff,
                'equalized_odds_difference': equalized_odds_diff,
                'fairness_score': fairness_score,
                'max_group_accuracy': max(group_accuracies),
                'min_group_accuracy': min(group_accuracies),
                'accuracy_difference': max(group_accuracies) - min(group_accuracies)
            })
            
        except Exception as e:
            logger.warning(f"Fairness metrics calculation failed: {e}")
        
        return fairness_results
    
    def interpretability_analysis(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        feature_names: Optional[List[str]] = None,
        sample_size: int = 100
    ) -> ValidationResult:
        """Analyze model interpretability using SHAP and LIME"""
        
        logger.info(f"Running interpretability analysis for {model_name}")
        
        results = {}
        scores_dict = {}
        
        # Limit sample size for computational efficiency
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
            y_sample = y[indices]
        else:
            X_sample = X
            y_sample = y
        
        # SHAP analysis
        if SHAP_AVAILABLE:
            try:
                shap_results = self._run_shap_analysis(model, X_sample, feature_names)
                results.update(shap_results['metrics'])
                scores_dict.update(shap_results['scores'])
            except Exception as e:
                logger.warning(f"SHAP analysis failed: {e}")
                results['shap_error'] = str(e)
        else:
            results['shap_available'] = False
        
        # LIME analysis
        if LIME_AVAILABLE:
            try:
                lime_results = self._run_lime_analysis(model, X_sample, y_sample, feature_names)
                results.update(lime_results['metrics'])
                scores_dict.update(lime_results['scores'])
            except Exception as e:
                logger.warning(f"LIME analysis failed: {e}")
                results['lime_error'] = str(e)
        else:
            results['lime_available'] = False
        
        # Feature importance analysis
        try:
            feature_importance_results = self._analyze_feature_importance(model, feature_names)
            results.update(feature_importance_results)
        except Exception as e:
            logger.warning(f"Feature importance analysis failed: {e}")
        
        return ValidationResult(
            validation_type='interpretability',
            model_name=model_name,
            metrics=results,
            scores=scores_dict,
            metadata={'sample_size': len(X_sample)},
            timestamp=datetime.now()
        )
    
    def _run_shap_analysis(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run SHAP analysis"""
        
        results = {'metrics': {}, 'scores': {}}
        
        try:
            # Choose appropriate explainer
            if hasattr(model, 'predict_proba'):
                explainer = shap.Explainer(model.predict_proba, X)
            else:
                explainer = shap.Explainer(model.predict, X)
            
            # Calculate SHAP values
            shap_values = explainer(X)
            
            # Extract feature importance
            if hasattr(shap_values, 'values'):
                importance_scores = np.abs(shap_values.values).mean(axis=0)
                
                results['metrics']['shap_feature_count'] = len(importance_scores)
                results['metrics']['shap_max_importance'] = float(np.max(importance_scores))
                results['metrics']['shap_mean_importance'] = float(np.mean(importance_scores))
                results['metrics']['shap_std_importance'] = float(np.std(importance_scores))
                
                # Top features
                if feature_names and len(feature_names) == len(importance_scores):
                    top_indices = np.argsort(importance_scores)[-10:][::-1]
                    top_features = [(feature_names[i], float(importance_scores[i])) for i in top_indices]
                    results['scores']['shap_top_features'] = top_features
                else:
                    results['scores']['shap_importance_scores'] = importance_scores.tolist()
            
            results['metrics']['shap_analysis_success'] = True
            
        except Exception as e:
            logger.warning(f"SHAP analysis failed: {e}")
            results['metrics']['shap_analysis_success'] = False
            results['metrics']['shap_error'] = str(e)
        
        return results
    
    def _run_lime_analysis(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run LIME analysis"""
        
        results = {'metrics': {}, 'scores': {}}
        
        try:
            # Determine mode
            mode = 'classification' if len(np.unique(y)) <= 10 else 'regression'
            
            # Create LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X,
                mode=mode,
                feature_names=feature_names,
                discretize_continuous=True
            )
            
            # Explain a few instances
            explanations = []
            lime_importances = []
            
            n_explain = min(5, len(X))
            
            for i in range(n_explain):
                try:
                    if hasattr(model, 'predict_proba'):
                        exp = explainer.explain_instance(
                            X[i],
                            model.predict_proba,
                            num_features=min(10, X.shape[1])
                        )
                    else:
                        exp = explainer.explain_instance(
                            X[i],
                            model.predict,
                            num_features=min(10, X.shape[1])
                        )
                    
                    # Extract explanation
                    explanation = exp.as_list()
                    explanations.append(explanation)
                    
                    # Extract feature importances
                    feature_importance = {feat: imp for feat, imp in explanation}
                    lime_importances.append(feature_importance)
                    
                except Exception as e:
                    logger.warning(f"LIME explanation failed for instance {i}: {e}")
                    continue
            
            if explanations:
                results['metrics']['lime_explanations_count'] = len(explanations)
                results['metrics']['lime_avg_features_per_explanation'] = np.mean([len(exp) for exp in explanations])
                results['scores']['lime_explanations'] = explanations[:3]  # Store first 3
                
                # Aggregate feature importances
                if lime_importances:
                    all_features = set()
                    for imp_dict in lime_importances:
                        all_features.update(imp_dict.keys())
                    
                    avg_importances = {}
                    for feature in all_features:
                        importances = [imp_dict.get(feature, 0) for imp_dict in lime_importances]
                        avg_importances[feature] = np.mean(importances)
                    
                    # Sort by absolute importance
                    sorted_features = sorted(avg_importances.items(), key=lambda x: abs(x[1]), reverse=True)
                    results['scores']['lime_feature_importance'] = sorted_features[:10]
            
            results['metrics']['lime_analysis_success'] = True
            
        except Exception as e:
            logger.warning(f"LIME analysis failed: {e}")
            results['metrics']['lime_analysis_success'] = False
            results['metrics']['lime_error'] = str(e)
        
        return results
    
    def _analyze_feature_importance(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze feature importance from model"""
        
        results = {}
        
        try:
            # Try different methods to get feature importance
            importance_scores = None
            
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importance_scores = model.feature_importances_
                results['importance_method'] = 'tree_based'
                
            elif hasattr(model, 'coef_'):
                # Linear models
                coef = model.coef_
                if len(coef.shape) > 1:
                    coef = coef[0]  # For binary classification
                importance_scores = np.abs(coef)
                results['importance_method'] = 'linear_coefficients'
                
            elif hasattr(model, 'get_feature_importance'):
                # Custom method
                importance_scores = model.get_feature_importance()
                results['importance_method'] = 'custom'
            
            if importance_scores is not None:
                results['has_feature_importance'] = True
                results['importance_max'] = float(np.max(importance_scores))
                results['importance_mean'] = float(np.mean(importance_scores))
                results['importance_std'] = float(np.std(importance_scores))
                
                # Calculate feature importance concentration (Gini coefficient)
                sorted_importance = np.sort(importance_scores)
                n = len(sorted_importance)
                cumulative = np.cumsum(sorted_importance)
                gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
                results['importance_concentration'] = float(gini)
                
            else:
                results['has_feature_importance'] = False
                
        except Exception as e:
            logger.warning(f"Feature importance analysis failed: {e}")
            results['feature_importance_error'] = str(e)
        
        return results
    
    def stability_validate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        n_iterations: int = 10
    ) -> ValidationResult:
        """Validate model stability across multiple training runs"""
        
        logger.info(f"Running stability validation for {model_name}")
        
        try:
            results = {}
            scores_dict = {}
            
            # Multiple training runs with different random seeds
            accuracies = []
            predictions_list = []
            
            for iteration in range(n_iterations):
                try:
                    # Create bootstrap sample
                    n_samples = len(X)
                    indices = np.random.choice(n_samples, n_samples, replace=True)
                    X_bootstrap = X[indices]
                    y_bootstrap = y[indices]
                    
                    # Train model on bootstrap sample
                    if hasattr(model, 'fit'):
                        model_copy = self._copy_model(model)
                        model_copy.fit(X_bootstrap, y_bootstrap)
                    else:
                        model_copy = model
                    
                    # Evaluate on original data
                    y_pred = model_copy.predict(X)
                    accuracy = np.mean(y == y_pred)
                    
                    accuracies.append(accuracy)
                    predictions_list.append(y_pred)
                    
                except Exception as e:
                    logger.warning(f"Stability iteration {iteration} failed: {e}")
                    continue
            
            if accuracies:
                # Calculate stability metrics
                results['stability_mean_accuracy'] = np.mean(accuracies)
                results['stability_std_accuracy'] = np.std(accuracies)
                results['stability_min_accuracy'] = np.min(accuracies)
                results['stability_max_accuracy'] = np.max(accuracies)
                results['stability_coefficient_variation'] = np.std(accuracies) / np.mean(accuracies) if np.mean(accuracies) > 0 else 1.0
                
                scores_dict['stability_accuracies'] = accuracies
                
                # Calculate prediction consistency
                if len(predictions_list) > 1:
                    consistency_scores = []
                    for i in range(len(predictions_list)):
                        for j in range(i + 1, len(predictions_list)):
                            consistency = np.mean(predictions_list[i] == predictions_list[j])
                            consistency_scores.append(consistency)
                    
                    results['prediction_consistency'] = np.mean(consistency_scores)
                    scores_dict['consistency_scores'] = consistency_scores
            
            return ValidationResult(
                validation_type='stability_validation',
                model_name=model_name,
                metrics=results,
                scores=scores_dict,
                metadata={'n_iterations': n_iterations},
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Stability validation failed: {e}")
            return ValidationResult(
                validation_type='stability_validation',
                model_name=model_name,
                metrics={'error': str(e)},
                scores={},
                metadata={},
                timestamp=datetime.now()
            )
    
    def _copy_model(self, model: Any) -> Any:
        """Create a copy of the model for stability testing"""
        
        try:
            # Try sklearn clone
            if SKLEARN_AVAILABLE:
                from sklearn.base import clone
                return clone(model)
        except:
            pass
        
        try:
            # Try deepcopy
            import copy
            return copy.deepcopy(model)
        except:
            pass
        
        # Return original model if copying fails
        return model
    
    def _calculate_detailed_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate detailed metrics"""
        
        metrics = {}
        
        try:
            # Basic metrics
            metrics['accuracy'] = np.mean(y_true == y_pred)
            
            # For binary classification
            if len(np.unique(y_true)) == 2:
                tp = np.sum((y_true == 1) & (y_pred == 1))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                fn = np.sum((y_true == 1) & (y_pred == 0))
                tn = np.sum((y_true == 0) & (y_pred == 0))
                
                metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0.0
                
                # Balanced accuracy
                sensitivity = metrics['recall']
                specificity = metrics['specificity']
                metrics['balanced_accuracy'] = (sensitivity + specificity) / 2
        
        except Exception as e:
            logger.warning(f"Detailed metrics calculation failed: {e}")
        
        return metrics
    
    def generate_validation_report(
        self,
        model_name: str,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        
        if model_name not in self.validation_results:
            raise ValueError(f"No validation results found for model {model_name}")
        
        results = self.validation_results[model_name]
        
        # Compile report
        report = {
            'model_name': model_name,
            'validation_timestamp': datetime.now().isoformat(),
            'validation_summary': {},
            'detailed_results': results
        }
        
        # Summary metrics
        summary = {}
        
        for validation_type, validation_result in results.items():
            if isinstance(validation_result, ValidationResult):
                summary[validation_type] = {
                    'status': 'success',
                    'key_metrics': validation_result.metrics
                }
            else:
                summary[validation_type] = {
                    'status': 'error',
                    'error': str(validation_result)
                }
        
        report['validation_summary'] = summary
        
        # Save report if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Validation report saved to {output_path}")
        
        return report
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results"""
        
        summary = {
            'models_validated': len(self.validation_results),
            'validation_types': set(),
            'overall_status': 'success'
        }
        
        for model_name, results in self.validation_results.items():
            for validation_type in results.keys():
                summary['validation_types'].add(validation_type)
        
        summary['validation_types'] = list(summary['validation_types'])
        
        return summary


class FairnessValidator:
    """Specialized validator for fairness and bias detection"""
    
    def __init__(self, sensitive_attributes: List[str]):
        self.sensitive_attributes = sensitive_attributes
        
    def validate_fairness(
        self,
        model: Any,
        X: pd.DataFrame,
        y: np.ndarray,
        sensitive_features: pd.DataFrame
    ) -> FairnessMetrics:
        """Comprehensive fairness validation"""
        
        # Implementation would go here
        # For now, return placeholder
        return FairnessMetrics(
            group_accuracy={},
            group_precision={},
            group_recall={},
            demographic_parity=0.0,
            equalized_odds=0.0,
            statistical_parity_difference=0.0
        )


class InterpretabilityAnalyzer:
    """Specialized analyzer for model interpretability"""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
    
    def analyze_global_importance(self, model: Any, X: np.ndarray) -> Dict[str, Any]:
        """Analyze global feature importance"""
        
        # Implementation would go here
        # For now, return placeholder
        return {
            'global_importance': {},
            'feature_interactions': {},
            'importance_stability': 0.0
        }
