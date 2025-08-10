# User Whisperer ML Package
"""
Machine Learning components for the User Whisperer platform.
Provides predictive models, feature engineering, and optimization algorithms.
"""

__version__ = "1.0.0"

# Core ML model imports
try:
    from .models.churn_prediction import ChurnPredictionModel
    from .models.content_optimization import (
        ThompsonSamplingBandit,
        NeuralBandit,
        Action,
        Context
    )
    from .models.timing_optimization import TimingOptimizationModel
except ImportError as e:
    print(f"Warning: ML models not available: {e}")

# Feature engineering imports
try:
    from .feature_engineering.feature_pipeline import (
        FeatureEngineeringPipeline,
        TemporalFeatureExtractor,
        BehavioralFeatureExtractor,
        EngagementFeatureExtractor,
        MonetizationFeatureExtractor,
        SessionFeatureExtractor,
        ContentFeatureExtractor,
        DeviceFeatureExtractor,
        GeographicFeatureExtractor
    )
except ImportError as e:
    print(f"Warning: Feature engineering not available: {e}")

# Model management
try:
    from .model_manager import (
        ModelManager,
        ModelRegistry,
        ModelVersion,
        ModelMetrics,
        get_model_manager,
        initialize_model_manager
    )
except ImportError as e:
    print(f"Warning: Model manager not available: {e}")

# Training utilities
try:
    from .training.trainer import (
        ModelTrainer,
        TrainingConfig,
        TrainingMetrics,
        ValidationStrategy
    )
    from .training.data_loader import (
        MLDataLoader,
        DataBatch,
        DataSplit
    )
except ImportError as e:
    print(f"Warning: Training utilities not available: {e}")

# Inference engine
try:
    from .inference.inference_engine import (
        InferenceEngine,
        PredictionRequest,
        PredictionResponse,
        BatchPredictionRequest
    )
except ImportError as e:
    print(f"Warning: Inference engine not available: {e}")

# Model types and interfaces
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

class ModelType(Enum):
    CHURN_PREDICTION = "churn_prediction"
    CONTENT_OPTIMIZATION = "content_optimization"
    TIMING_OPTIMIZATION = "timing_optimization"
    ENGAGEMENT_SCORING = "engagement_scoring"
    LTV_PREDICTION = "ltv_prediction"
    FEATURE_EXTRACTION = "feature_extraction"

class PredictionType(Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    RANKING = "ranking"
    RECOMMENDATION = "recommendation"
    TIME_SERIES = "time_series"

@dataclass
class ModelConfig:
    model_type: ModelType
    prediction_type: PredictionType
    version: str
    parameters: Dict[str, Any]
    training_config: Dict[str, Any]
    feature_config: Dict[str, Any]
    preprocessing_config: Dict[str, Any]

@dataclass
class PredictionResult:
    prediction: Union[float, int, List[float], np.ndarray]
    confidence: float
    model_version: str
    features_used: List[str]
    processing_time_ms: float
    metadata: Dict[str, Any]

@dataclass
class ModelPerformance:
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    custom_metrics: Optional[Dict[str, float]] = None

# Utility functions
def create_prediction_response(
    success: bool,
    result: Optional[PredictionResult] = None,
    error: Optional[str] = None
) -> Dict[str, Any]:
    """Create standardized prediction response"""
    import datetime
    return {
        'success': success,
        'result': result,
        'error': error,
        'timestamp': datetime.datetime.utcnow().isoformat()
    }

def validate_features(
    features: Dict[str, Any],
    required_features: List[str]
) -> Tuple[bool, List[str]]:
    """Validate input features"""
    missing_features = [
        feat for feat in required_features
        if feat not in features
    ]
    
    return len(missing_features) == 0, missing_features

def normalize_features(
    features: Dict[str, Any],
    feature_ranges: Dict[str, Tuple[float, float]]
) -> Dict[str, Any]:
    """Normalize features to 0-1 range"""
    normalized = {}
    
    for key, value in features.items():
        if key in feature_ranges:
            min_val, max_val = feature_ranges[key]
            if max_val > min_val:
                normalized[key] = (value - min_val) / (max_val - min_val)
            else:
                normalized[key] = value
        else:
            normalized[key] = value
    
    return normalized

def calculate_feature_importance(
    model: Any,
    feature_names: List[str]
) -> Dict[str, float]:
    """Calculate feature importance from model"""
    importance_dict = {}
    
    try:
        # Try different methods to get feature importance
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            importances = np.abs(model.coef_)
            if len(importances.shape) > 1:
                importances = importances[0]
        else:
            # Default: equal importance
            importances = np.ones(len(feature_names)) / len(feature_names)
        
        for i, name in enumerate(feature_names):
            if i < len(importances):
                importance_dict[name] = float(importances[i])
    
    except Exception:
        # Fallback to equal importance
        for name in feature_names:
            importance_dict[name] = 1.0 / len(feature_names)
    
    return importance_dict

# Constants
DEFAULT_MODEL_CONFIGS = {
    ModelType.CHURN_PREDICTION: {
        'sequence_length': 30,
        'feature_dim': 150,
        'hidden_units': [128, 64],
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'batch_size': 256,
        'epochs': 50
    },
    ModelType.CONTENT_OPTIMIZATION: {
        'n_actions': 100,
        'feature_dim': 200,
        'alpha': 1.0,
        'beta': 1.0,
        'epsilon': 0.1,
        'learning_rate': 0.001
    },
    ModelType.TIMING_OPTIMIZATION: {
        'seasonality_mode': 'multiplicative',
        'changepoint_prior_scale': 0.05,
        'interval_width': 0.95,
        'daily_seasonality': True,
        'weekly_seasonality': True
    }
}

FEATURE_GROUPS = {
    'temporal': [
        'hour_of_day', 'day_of_week', 'day_of_month', 'week_of_year',
        'is_weekend', 'is_business_hours', 'hour_sin', 'hour_cos',
        'day_sin', 'day_cos', 'days_since_signup', 'hours_since_last_activity'
    ],
    'behavioral': [
        'total_events', 'unique_event_types', 'events_per_hour',
        'avg_inter_event_time', 'std_inter_event_time', 'has_error_pattern',
        'has_purchase_intent'
    ],
    'engagement': [
        'engagement_score', 'lifecycle_stage_encoded', 'engagement_event_count',
        'engagement_event_ratio', 'avg_session_length', 'max_session_length',
        'total_sessions', 'features_adopted', 'feature_adoption_rate'
    ],
    'monetization': [
        'is_paid', 'ltv_prediction', 'upgrade_probability',
        'days_until_trial_end', 'is_in_trial', 'total_purchases',
        'total_revenue', 'avg_purchase_value', 'pricing_page_views'
    ]
}

# Model performance thresholds
PERFORMANCE_THRESHOLDS = {
    ModelType.CHURN_PREDICTION: {
        'auc': 0.85,
        'precision': 0.75,
        'recall': 0.70,
        'f1_score': 0.72
    },
    ModelType.CONTENT_OPTIMIZATION: {
        'click_through_rate': 0.15,
        'conversion_rate': 0.05,
        'revenue_per_user': 50.0
    },
    ModelType.TIMING_OPTIMIZATION: {
        'accuracy': 0.60,
        'mae': 0.2,
        'engagement_lift': 0.15
    }
}

__all__ = [
    # Version
    '__version__',
    
    # Core models
    'ChurnPredictionModel',
    'ThompsonSamplingBandit',
    'NeuralBandit',
    'TimingOptimizationModel',
    
    # Feature engineering
    'FeatureEngineeringPipeline',
    'TemporalFeatureExtractor',
    'BehavioralFeatureExtractor',
    'EngagementFeatureExtractor',
    'MonetizationFeatureExtractor',
    
    # Model management
    'ModelManager',
    'ModelRegistry',
    'ModelVersion',
    'ModelMetrics',
    'get_model_manager',
    'initialize_model_manager',
    
    # Training
    'ModelTrainer',
    'TrainingConfig',
    'TrainingMetrics',
    'ValidationStrategy',
    'MLDataLoader',
    'DataBatch',
    'DataSplit',
    
    # Inference
    'InferenceEngine',
    'PredictionRequest',
    'PredictionResponse',
    'BatchPredictionRequest',
    
    # Types and enums
    'ModelType',
    'PredictionType',
    'ModelConfig',
    'PredictionResult',
    'ModelPerformance',
    'Action',
    'Context',
    
    # Utility functions
    'create_prediction_response',
    'validate_features',
    'normalize_features',
    'calculate_feature_importance',
    
    # Constants
    'DEFAULT_MODEL_CONFIGS',
    'FEATURE_GROUPS',
    'PERFORMANCE_THRESHOLDS'
]
