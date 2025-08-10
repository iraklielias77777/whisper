# Feature Engineering Package for User Whisperer ML
"""
Feature engineering components for machine learning models.
Provides comprehensive feature extraction and preprocessing capabilities.
"""

__version__ = "1.0.0"

# Main feature engineering pipeline
try:
    from .feature_pipeline import (
        FeatureEngineeringPipeline,
        FeatureConfig,
        BaseFeatureExtractor
    )
except ImportError as e:
    print(f"Warning: Feature pipeline not available: {e}")

# Individual feature extractors
try:
    from .feature_pipeline import (
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
    print(f"Warning: Feature extractors not available: {e}")

# Feature processing utilities
try:
    from .feature_utils import (
        FeatureValidator,
        FeatureSelector,
        FeatureTransformer,
        FeatureEncoder
    )
except ImportError as e:
    print(f"Warning: Feature utilities not available: {e}")

__all__ = [
    # Version
    '__version__',
    
    # Main pipeline
    'FeatureEngineeringPipeline',
    'FeatureConfig',
    'BaseFeatureExtractor',
    
    # Feature extractors
    'TemporalFeatureExtractor',
    'BehavioralFeatureExtractor',
    'EngagementFeatureExtractor',
    'MonetizationFeatureExtractor',
    'SessionFeatureExtractor',
    'ContentFeatureExtractor',
    'DeviceFeatureExtractor',
    'GeographicFeatureExtractor',
    
    # Utilities
    'FeatureValidator',
    'FeatureSelector',
    'FeatureTransformer',
    'FeatureEncoder'
]
