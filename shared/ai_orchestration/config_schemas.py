"""
Configuration schemas for AI orchestration system
Production-ready configurations with validation and type safety
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

class OperationalMode(Enum):
    """AI operational modes"""
    EXPLORATION = "exploration"
    GUIDED = "guided"
    AUTONOMOUS = "autonomous"
    CONSERVATIVE = "conservative"

class LearningStrategy(Enum):
    """Learning strategy types"""
    ADAPTIVE = "adaptive"
    RULE_BASED = "rule_based"
    ML_DRIVEN = "ml_driven"
    HYBRID = "hybrid"

@dataclass
class AISystemConfiguration:
    """
    Master configuration for AI orchestration system
    Production-ready with comprehensive settings
    """
    
    # Core Learning Parameters
    learning_config: Dict[str, Any] = field(default_factory=lambda: {
        "adaptation_rate": 0.15,  # How quickly system adapts to new patterns
        "confidence_threshold": 0.75,  # Minimum confidence for autonomous decisions
        "exploration_rate": 0.20,  # Balance between exploration and exploitation
        "memory_window": 30,  # Days of context to maintain
        "feedback_weight": 0.35,  # Weight of human feedback in learning
        "performance_baseline": 0.70,  # Minimum acceptable performance
        "evolution_cycles": 100,  # Iterations before strategy evolution
        "risk_tolerance": 0.10,  # Acceptable deviation from proven strategies
        "learning_enabled": True,
        "online_learning": True,
        "batch_learning": True,
        "transfer_learning": True
    })
    
    # Customer Profiling Engine
    profiling_config: Dict[str, Any] = field(default_factory=lambda: {
        "segmentation_depth": 5,  # Levels of customer segmentation
        "behavior_dimensions": 12,  # Number of behavioral dimensions tracked
        "personality_markers": 8,  # Personality trait indicators
        "context_factors": 15,  # Environmental/contextual factors
        "dynamic_weighting": True,  # Enable dynamic weight adjustment
        "profile_decay_rate": 0.05,  # How fast old patterns lose influence
        "minimum_interactions": 10,  # Before creating personalized model
        "real_time_updates": True,
        "cohort_analysis": True,
        "lifecycle_tracking": True
    })
    
    # Content Generation Parameters
    content_config: Dict[str, Any] = field(default_factory=lambda: {
        "creativity_spectrum": {
            "conservative": 0.2,  # Safe, proven approaches
            "balanced": 0.5,      # Mix of safe and creative
            "innovative": 0.8,    # Experimental approaches
            "breakthrough": 0.95  # Completely new strategies
        },
        "tone_adaptation": {
            "formal_threshold": 0.7,
            "casual_threshold": 0.3,
            "technical_depth": "auto",  # Automatically adjust based on user
            "emotional_resonance": True,
            "cultural_sensitivity": True,
            "brand_consistency": True
        },
        "message_optimization": {
            "length_optimization": True,
            "readability_target": "auto",
            "personalization_depth": 3,  # Levels of personalization
            "a_b_testing_enabled": True,
            "multivariate_testing": True,
            "dynamic_variants": True
        },
        "llm_integration": {
            "primary_model": "gpt-4-turbo",
            "fallback_models": ["claude-3-sonnet", "gpt-3.5-turbo"],
            "max_tokens": 4000,
            "temperature": 0.7,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1
        }
    })
    
    # Decision Engine Configuration
    decision_config: Dict[str, Any] = field(default_factory=lambda: {
        "strategy_selection": {
            "mode": "adaptive",  # adaptive, rule-based, ml-driven, hybrid
            "confidence_requirements": {
                "autonomous": 0.85,
                "supervised": 0.65,
                "experimental": 0.45
            },
            "fallback_strategies": ["conservative", "template", "human"],
            "risk_assessment": True,
            "impact_prediction": True,
            "multi_armed_bandits": True,
            "contextual_optimization": True
        },
        "timing_optimization": {
            "predictive_window": 72,  # Hours ahead to predict
            "granularity": 15,  # Minutes
            "timezone_aware": True,
            "behavior_based": True,
            "external_factors": True,  # Holidays, events, etc.
            "send_time_optimization": True,
            "frequency_capping": True
        },
        "channel_selection": {
            "multi_channel": True,
            "channel_fatigue_tracking": True,
            "cost_optimization": True,
            "response_prediction": True,
            "delivery_optimization": True,
            "cross_channel_attribution": True
        }
    })
    
    # Feedback Loop Configuration
    feedback_config: Dict[str, Any] = field(default_factory=lambda: {
        "collection_methods": ["implicit", "explicit", "behavioral", "outcome"],
        "processing_pipeline": {
            "real_time": True,
            "batch_interval": 3600,  # Seconds
            "priority_weighting": True,
            "contradiction_resolution": "weighted_consensus",
            "quality_filtering": True,
            "bias_detection": True
        },
        "learning_triggers": {
            "performance_deviation": 0.15,
            "pattern_emergence": 0.30,
            "user_feedback": "immediate",
            "system_anomaly": "immediate",
            "drift_detection": True,
            "significance_testing": True
        },
        "self_correction": {
            "enabled": True,
            "confidence_threshold": 0.8,
            "rollback_enabled": True,
            "human_escalation": True,
            "automatic_rollback_threshold": 0.3
        }
    })
    
    # Performance and Safety Constraints
    performance_config: Dict[str, Any] = field(default_factory=lambda: {
        "latency_targets": {
            "decision_making": 100,  # milliseconds
            "content_generation": 2000,  # milliseconds
            "learning_updates": 5000,  # milliseconds
            "batch_processing": 300  # seconds
        },
        "throughput_targets": {
            "events_per_second": 100000,
            "decisions_per_second": 10000,
            "concurrent_users": 1000000
        },
        "safety_constraints": {
            "max_performance_drop": 0.15,
            "minimum_confidence": 0.60,
            "bias_detection": True,
            "fairness_constraints": True,
            "explainability_required": True,
            "audit_trail": True
        }
    })
    
    # Model Management
    model_config: Dict[str, Any] = field(default_factory=lambda: {
        "versioning": {
            "enabled": True,
            "max_versions": 10,
            "auto_cleanup": True,
            "retention_days": 90
        },
        "deployment": {
            "canary_enabled": True,
            "canary_percentage": 5,
            "blue_green": True,
            "rollback_enabled": True,
            "health_checks": True
        },
        "monitoring": {
            "drift_detection": True,
            "performance_monitoring": True,
            "bias_monitoring": True,
            "fairness_monitoring": True,
            "explainability_monitoring": True
        }
    })
    
    def validate(self) -> List[str]:
        """Validate configuration and return any errors"""
        errors = []
        
        # Validate learning config
        if not 0 <= self.learning_config.get("adaptation_rate", 0) <= 1:
            errors.append("adaptation_rate must be between 0 and 1")
        
        if not 0 <= self.learning_config.get("confidence_threshold", 0) <= 1:
            errors.append("confidence_threshold must be between 0 and 1")
        
        # Validate performance targets
        if self.performance_config.get("latency_targets", {}).get("decision_making", 0) > 1000:
            errors.append("decision_making latency target should be <= 1000ms for real-time performance")
        
        # Validate safety constraints
        if self.performance_config.get("safety_constraints", {}).get("minimum_confidence", 0) < 0.5:
            errors.append("minimum_confidence should be >= 0.5 for production safety")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "learning_config": self.learning_config,
            "profiling_config": self.profiling_config,
            "content_config": self.content_config,
            "decision_config": self.decision_config,
            "feedback_config": self.feedback_config,
            "performance_config": self.performance_config,
            "model_config": self.model_config
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AISystemConfiguration':
        """Create from dictionary"""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AISystemConfiguration':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

@dataclass
class LearningConfiguration:
    """Configuration for continuous learning system"""
    
    # Learning Parameters
    learning_enabled: bool = True
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.0001
    
    # Data Requirements
    minimum_samples: int = 100
    validation_split: float = 0.2
    test_split: float = 0.1
    
    # Update Triggers
    update_triggers: Dict[str, Any] = field(default_factory=lambda: {
        "performance_degradation": 0.1,  # 10% drop triggers update
        "sample_threshold": 1000,  # Update after N new samples
        "time_based": 86400,  # Update daily (seconds)
        "drift_detection": True,  # Update on distribution drift
        "explicit_feedback": True  # Update on explicit corrections
    })
    
    # Model Management
    model_versioning: bool = True
    max_model_versions: int = 10
    rollback_enabled: bool = True
    a_b_testing_enabled: bool = True
    
    # Performance Monitoring
    metrics_tracked: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1_score",
        "auc_roc", "engagement_rate", "conversion_rate",
        "customer_satisfaction", "response_time"
    ])
    
    # Safety Constraints
    safety_checks: Dict[str, Any] = field(default_factory=lambda: {
        "max_performance_drop": 0.15,  # Maximum allowed degradation
        "minimum_confidence": 0.60,  # Minimum confidence for production
        "bias_detection": True,  # Check for model bias
        "fairness_constraints": True,  # Ensure fairness across segments
        "explainability_required": True  # Require explainable decisions
    })

@dataclass
class FeedbackConfiguration:
    """Configuration for feedback processing"""
    
    # Feedback Collection
    collection_enabled: bool = True
    collection_methods: List[str] = field(default_factory=lambda: [
        "explicit", "implicit", "behavioral", "outcome"
    ])
    
    # Processing Configuration
    real_time_processing: bool = True
    batch_processing_interval: int = 3600  # seconds
    quality_filtering: bool = True
    bias_detection: bool = True
    
    # Learning Integration
    immediate_learning: bool = True
    batch_learning: bool = True
    transfer_learning: bool = True
    
    # Weights and Priorities
    feedback_weights: Dict[str, float] = field(default_factory=lambda: {
        "explicit": 1.0,
        "implicit": 0.7,
        "behavioral": 0.8,
        "outcome": 0.9
    })
    
    # Self-Correction
    self_correction_enabled: bool = True
    correction_threshold: float = 0.3
    automatic_rollback: bool = True
    human_escalation: bool = True

@dataclass
class EvolutionConfiguration:
    """Configuration for strategy evolution using genetic algorithms"""
    
    evolution_enabled: bool = True
    
    # Genetic Algorithm Parameters
    population_size: int = 50
    generations: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elite_size: int = 5
    
    # Selection Strategy
    selection_method: str = "tournament"  # tournament, roulette, rank
    tournament_size: int = 3
    
    # Fitness Evaluation
    fitness_function: str = "weighted_composite"
    fitness_weights: Dict[str, float] = field(default_factory=lambda: {
        "conversion_rate": 0.3,
        "engagement_rate": 0.2,
        "satisfaction_score": 0.2,
        "efficiency": 0.15,
        "innovation": 0.15
    })
    
    # Constraints
    diversity_threshold: float = 0.3
    convergence_threshold: float = 0.95
    max_evolution_time: int = 3600  # seconds

@dataclass
class CustomerAIConfiguration:
    """Customer-specific AI configuration"""
    
    customer_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    
    # Customer Analysis
    customer_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Personalized Configurations
    learning_configuration: Optional[LearningConfiguration] = None
    content_configuration: Dict[str, Any] = field(default_factory=dict)
    decision_configuration: Dict[str, Any] = field(default_factory=dict)
    channel_configuration: Dict[str, Any] = field(default_factory=dict)
    
    # Success Metrics
    success_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Monitoring Configuration
    monitoring_configuration: Dict[str, Any] = field(default_factory=dict)
    
    # Safety and Compliance
    safety_configuration: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "customer_id": self.customer_id,
            "created_at": self.created_at.isoformat(),
            "version": self.version,
            "customer_analysis": self.customer_analysis,
            "learning_configuration": self.learning_configuration.__dict__ if self.learning_configuration else None,
            "content_configuration": self.content_configuration,
            "decision_configuration": self.decision_configuration,
            "channel_configuration": self.channel_configuration,
            "success_metrics": self.success_metrics,
            "monitoring_configuration": self.monitoring_configuration,
            "safety_configuration": self.safety_configuration
        }

# Environment-specific configurations
PRODUCTION_CONFIG = AISystemConfiguration(
    learning_config={
        **AISystemConfiguration().learning_config,
        "adaptation_rate": 0.1,
        "exploration_rate": 0.05,
        "risk_tolerance": 0.05
    },
    performance_config={
        **AISystemConfiguration().performance_config,
        "safety_constraints": {
            "max_performance_drop": 0.1,
            "minimum_confidence": 0.75,
            "bias_detection": True,
            "fairness_constraints": True,
            "explainability_required": True,
            "audit_trail": True
        }
    }
)

STAGING_CONFIG = AISystemConfiguration(
    learning_config={
        **AISystemConfiguration().learning_config,
        "adaptation_rate": 0.2,
        "exploration_rate": 0.15,
        "risk_tolerance": 0.15
    },
    performance_config={
        **AISystemConfiguration().performance_config,
        "safety_constraints": {
            "max_performance_drop": 0.2,
            "minimum_confidence": 0.65,
            "bias_detection": True,
            "fairness_constraints": True,
            "explainability_required": True,
            "audit_trail": True
        }
    }
)

DEVELOPMENT_CONFIG = AISystemConfiguration(
    learning_config={
        **AISystemConfiguration().learning_config,
        "adaptation_rate": 0.3,
        "exploration_rate": 0.3,
        "risk_tolerance": 0.25
    },
    performance_config={
        **AISystemConfiguration().performance_config,
        "safety_constraints": {
            "max_performance_drop": 0.3,
            "minimum_confidence": 0.5,
            "bias_detection": True,
            "fairness_constraints": True,
            "explainability_required": False,
            "audit_trail": False
        }
    }
)
