# User Whisperer AI Orchestration Package
"""
Master AI orchestration system for adaptive, intelligent customer engagement.
Provides self-learning, adaptive decision-making, and comprehensive AI coordination.
"""

__version__ = "1.0.0"

# Core orchestration components
try:
    from .master_controller import (
        MasterAIOrchestrator,
        AISystemConfiguration,
        OrchestrationResult
    )
except ImportError as e:
    print(f"Warning: Master controller not available: {e}")

# Dynamic prompt generation
try:
    from .prompt_engine import (
        DynamicPromptEngine,
        PromptTemplate,
        PromptOptimizationEngine
    )
except ImportError as e:
    print(f"Warning: Prompt engine not available: {e}")

# Learning and adaptation
try:
    from .learning_pipeline import (
        AdaptiveLearningPipeline,
        LearningConfiguration,
        LearningResult
    )
except ImportError as e:
    print(f"Warning: Learning pipeline not available: {e}")

# Template generation
try:
    from .template_generator import (
        DynamicTemplateGenerator,
        CustomerTemplate,
        TemplateVariant
    )
except ImportError as e:
    print(f"Warning: Template generator not available: {e}")

# Real-time analytics
try:
    from .analytics_integration import (
        RealTimeAnalyticsEngine,
        AnalyticsConfiguration,
        AdaptationResult
    )
except ImportError as e:
    print(f"Warning: Analytics integration not available: {e}")

# Feedback and self-correction
try:
    from .feedback_loop import (
        IntelligentFeedbackLoop,
        FeedbackProcessor,
        SelfCorrectionEngine
    )
except ImportError as e:
    print(f"Warning: Feedback loop not available: {e}")

# Scenario management
try:
    from .scenario_manager import (
        AdaptiveScenarioManager,
        ScenarioEvolutionEngine,
        ScenarioResult
    )
except ImportError as e:
    print(f"Warning: Scenario manager not available: {e}")

# Configuration schemas
try:
    from .config_schemas import (
        AISystemConfiguration,
        LearningConfiguration,
        FeedbackConfiguration,
        EvolutionConfiguration,
        CustomerAIConfiguration
    )
except ImportError as e:
    print(f"Warning: Configuration schemas not available: {e}")

__all__ = [
    # Version
    '__version__',
    
    # Core orchestration
    'MasterAIOrchestrator',
    'AISystemConfiguration',
    'OrchestrationResult',
    
    # Prompt generation
    'DynamicPromptEngine',
    'PromptTemplate',
    'PromptOptimizationEngine',
    
    # Learning system
    'AdaptiveLearningPipeline',
    'LearningConfiguration',
    'LearningResult',
    
    # Template generation
    'DynamicTemplateGenerator',
    'CustomerTemplate',
    'TemplateVariant',
    
    # Analytics
    'RealTimeAnalyticsEngine',
    'AnalyticsConfiguration',
    'AdaptationResult',
    
    # Feedback system
    'IntelligentFeedbackLoop',
    'FeedbackProcessor',
    'SelfCorrectionEngine',
    
    # Scenario management
    'AdaptiveScenarioManager',
    'ScenarioEvolutionEngine',
    'ScenarioResult',
    
    # Configuration
    'AISystemConfiguration',
    'LearningConfiguration',
    'FeedbackConfiguration',
    'EvolutionConfiguration',
    'CustomerAIConfiguration'
]
