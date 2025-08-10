# ML Training Package for User Whisperer
"""
Model training, validation, and deployment infrastructure.
Provides end-to-end training pipelines and online learning capabilities.
"""

__version__ = "1.0.0"

# Training pipeline
try:
    from .training_pipeline import (
        ModelTrainingPipeline,
        TrainingConfig,
        TrainingMetrics,
        ValidationStrategy
    )
except ImportError as e:
    print(f"Warning: Training pipeline not available: {e}")

# Model validation
try:
    from .model_validation import (
        ModelValidator,
        ValidationResult,
        FairnessValidator,
        InterpretabilityAnalyzer
    )
except ImportError as e:
    print(f"Warning: Model validation not available: {e}")

# Data loading
try:
    from .data_loader import (
        MLDataLoader,
        DataBatch,
        DataSplit,
        FeatureStore
    )
except ImportError as e:
    print(f"Warning: Data loader not available: {e}")

# Model deployment
try:
    from .model_deployment import (
        ModelDeployment,
        DeploymentConfig,
        ModelRegistry,
        VersionManager
    )
except ImportError as e:
    print(f"Warning: Model deployment not available: {e}")

# Online learning
try:
    from .online_learning import (
        OnlineLearningSystem,
        AdaptiveLearningRate,
        OnlineFeatureStore,
        FeedbackProcessor
    )
except ImportError as e:
    print(f"Warning: Online learning not available: {e}")

__all__ = [
    # Version
    '__version__',
    
    # Training pipeline
    'ModelTrainingPipeline',
    'TrainingConfig',
    'TrainingMetrics',
    'ValidationStrategy',
    
    # Model validation
    'ModelValidator',
    'ValidationResult',
    'FairnessValidator',
    'InterpretabilityAnalyzer',
    
    # Data loading
    'MLDataLoader',
    'DataBatch',
    'DataSplit',
    'FeatureStore',
    
    # Model deployment
    'ModelDeployment',
    'DeploymentConfig',
    'ModelRegistry',
    'VersionManager',
    
    # Online learning
    'OnlineLearningSystem',
    'AdaptiveLearningRate',
    'OnlineFeatureStore',
    'FeedbackProcessor'
]
