# ML Models Package for User Whisperer
"""
Machine learning models for prediction and optimization.
Provides churn prediction, content optimization, and timing models.
"""

__version__ = "1.0.0"

# Core ML models
try:
    from .churn_prediction import ChurnPredictionModel
except ImportError as e:
    print(f"Warning: Churn prediction model not available: {e}")

try:
    from .content_optimization import (
        ThompsonSamplingBandit,
        NeuralBandit,
        Action,
        Context
    )
except ImportError as e:
    print(f"Warning: Content optimization models not available: {e}")

try:
    from .timing_optimization import (
        TimingOptimizationModel,
        TimingPrediction
    )
except ImportError as e:
    print(f"Warning: Timing optimization model not available: {e}")

# Model utilities and interfaces
try:
    from .base_model import (
        BaseMLModel,
        ModelConfig,
        ModelMetrics,
        PredictionResult
    )
except ImportError as e:
    print(f"Warning: Base model utilities not available: {e}")

__all__ = [
    # Version
    '__version__',
    
    # Core models
    'ChurnPredictionModel',
    'ThompsonSamplingBandit',
    'NeuralBandit',
    'TimingOptimizationModel',
    
    # Model data structures
    'Action',
    'Context',
    'TimingPrediction',
    
    # Base utilities
    'BaseMLModel',
    'ModelConfig',
    'ModelMetrics',
    'PredictionResult'
]
