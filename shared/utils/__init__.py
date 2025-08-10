# User Whisperer Shared Utils - Python Package Init
"""
Shared utilities package for the User Whisperer platform.
Provides data management, caching, storage, and performance optimization components.
"""

__version__ = "1.0.0"

# Core utility imports
from .config import Config
from .logger import Logger

# Database components
try:
    from .database_manager import (
        DatabaseManager,
        DatabaseConfig,
        QueryBuilder,
        get_database_manager,
        initialize_database_manager
    )
except ImportError as e:
    print(f"Warning: Database manager not available: {e}")

# Storage components
try:
    from .storage_manager import (
        StorageManager,
        StorageTier,
        StoragePolicy,
        DataCompressor,
        DataEncryptor,
        get_storage_manager,
        initialize_storage_manager
    )
except ImportError as e:
    print(f"Warning: Storage manager not available: {e}")

# Cache components
try:
    from .cache_manager import (
        CacheManager,
        CacheLevel,
        MemoryCache,
        RedisCache,
        CDNCache,
        CacheWarmer,
        cache_decorator,
        get_cache_manager,
        initialize_cache_manager
    )
except ImportError as e:
    print(f"Warning: Cache manager not available: {e}")

# Stream processing components
try:
    from .stream_processor import (
        EventStreamProcessor,
        StreamConfig,
        StreamAggregator,
        EventRouter,
        WindowAggregator,
        SessionProcessor
    )
except ImportError as e:
    print(f"Warning: Stream processor not available: {e}")

# Complex event processing
try:
    from .complex_event_processor import (
        ComplexEventProcessor,
        PatternType,
        PatternCondition,
        PatternMatch,
        PredefinedPatterns
    )
except ImportError as e:
    print(f"Warning: Complex event processor not available: {e}")

# Data lifecycle management
try:
    from .data_lifecycle_manager import (
        DataLifecycleManager,
        DataLifecycleAction,
        DataClassification,
        RetentionRule,
        LifecycleEvent,
        GDPRComplianceHandler,
        get_lifecycle_manager,
        initialize_lifecycle_manager
    )
except ImportError as e:
    print(f"Warning: Data lifecycle manager not available: {e}")

# Performance optimization
try:
    from .performance_optimizer import (
        PerformanceOptimizer,
        PerformanceMetric,
        OptimizationAction,
        PerformanceThreshold,
        PerformanceAlert,
        get_performance_optimizer,
        initialize_performance_optimizer
    )
except ImportError as e:
    print(f"Warning: Performance optimizer not available: {e}")

# Utility functions
def create_service_response(success: bool, data=None, error: str = None) -> dict:
    """Create standardized service response"""
    import datetime
    return {
        'success': success,
        'data': data,
        'error': error,
        'timestamp': datetime.datetime.utcnow().isoformat()
    }

def generate_request_id() -> str:
    """Generate unique request ID"""
    import uuid
    import time
    return f"req_{int(time.time())}_{str(uuid.uuid4())[:8]}"

def sanitize_user_data(data: dict) -> dict:
    """Sanitize user data for logging/analytics"""
    if not isinstance(data, dict):
        return data
    
    sensitive_fields = {
        'email', 'phone', 'password', 'api_key', 'api_secret',
        'ssn', 'credit_card', 'bank_account', 'address'
    }
    
    sanitized = {}
    for key, value in data.items():
        if key.lower() in sensitive_fields:
            sanitized[key] = '[REDACTED]'
        elif isinstance(value, dict):
            sanitized[key] = sanitize_user_data(value)
        elif isinstance(value, list):
            sanitized[key] = [
                sanitize_user_data(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            sanitized[key] = value
    
    return sanitized

# Constants
SERVICE_NAMES = {
    'EVENT_INGESTION': 'event-ingestion',
    'BEHAVIORAL_ANALYSIS': 'behavioral-analysis',
    'DECISION_ENGINE': 'decision-engine',
    'CONTENT_GENERATION': 'content-generation',
    'CHANNEL_ORCHESTRATION': 'channel-orchestration'
}

EVENT_TYPES = {
    'USER_SIGNUP': 'user_signup',
    'USER_LOGIN': 'user_login',
    'USER_LOGOUT': 'user_logout',
    'PROFILE_UPDATE': 'profile_update',
    'PAGE_VIEW': 'page_view',
    'FEATURE_USED': 'feature_used',
    'CONTENT_VIEWED': 'content_viewed',
    'SEARCH_PERFORMED': 'search_performed',
    'PURCHASE_STARTED': 'purchase_started',
    'PURCHASE_COMPLETED': 'purchase_completed',
    'SUBSCRIPTION_STARTED': 'subscription_started',
    'SUBSCRIPTION_CANCELLED': 'subscription_cancelled',
    'ERROR_OCCURRED': 'error_occurred',
    'SESSION_STARTED': 'session_started',
    'SESSION_ENDED': 'session_ended'
}

LIFECYCLE_STAGES = {
    'NEW': 'new',
    'ONBOARDING': 'onboarding',
    'ACTIVATED': 'activated',
    'ENGAGED': 'engaged',
    'POWER_USER': 'power_user',
    'AT_RISK': 'at_risk',
    'DORMANT': 'dormant',
    'CHURNED': 'churned',
    'REACTIVATED': 'reactivated'
}

CHANNELS = {
    'EMAIL': 'email',
    'SMS': 'sms',
    'PUSH': 'push',
    'WEBHOOK': 'webhook',
    'IN_APP': 'in_app'
}

INTERVENTION_TYPES = {
    'ONBOARDING': 'onboarding',
    'ENGAGEMENT': 'engagement',
    'RETENTION': 'retention',
    'MONETIZATION': 'monetization',
    'REACTIVATION': 'reactivation',
    'SUPPORT': 'support',
    'EDUCATION': 'education',
    'CELEBRATION': 'celebration'
}

# Configuration defaults
DEFAULT_CONFIG = {
    'database': {
        'min_pool_size': 10,
        'max_pool_size': 50,
        'command_timeout': 60
    },
    'cache': {
        'l1_max_size': 1000,
        'l1_ttl': 300,
        'l2_ttl': 3600,
        'cache_warming': True
    },
    'storage': {
        'hot_retention_hours': 24,
        'warm_retention_days': 90,
        'cold_retention_days': 365,
        'compression_enabled': True,
        'encryption_enabled': True
    },
    'stream': {
        'max_messages': 100,
        'ack_deadline': 60,
        'buffer_size': 1000,
        'buffer_ttl_hours': 1
    },
    'performance': {
        'monitoring_enabled': True,
        'alert_thresholds': {
            'response_time_ms': 1000,
            'error_rate': 0.05,
            'cache_hit_rate': 0.8,
            'cpu_usage': 0.8,
            'memory_usage': 0.85
        }
    }
}

__all__ = [
    # Version
    '__version__',
    
    # Core utilities
    'Config',
    'Logger',
    
    # Database
    'DatabaseManager',
    'DatabaseConfig', 
    'QueryBuilder',
    'get_database_manager',
    'initialize_database_manager',
    
    # Storage
    'StorageManager',
    'StorageTier',
    'StoragePolicy',
    'DataCompressor',
    'DataEncryptor',
    'get_storage_manager',
    'initialize_storage_manager',
    
    # Cache
    'CacheManager',
    'CacheLevel',
    'MemoryCache',
    'RedisCache',
    'CDNCache', 
    'CacheWarmer',
    'cache_decorator',
    'get_cache_manager',
    'initialize_cache_manager',
    
    # Stream processing
    'EventStreamProcessor',
    'StreamConfig',
    'StreamAggregator',
    'EventRouter',
    'WindowAggregator',
    'SessionProcessor',
    
    # Complex event processing
    'ComplexEventProcessor',
    'PatternType',
    'PatternCondition',
    'PatternMatch',
    'PredefinedPatterns',
    
    # Data lifecycle
    'DataLifecycleManager',
    'DataLifecycleAction',
    'DataClassification',
    'RetentionRule',
    'LifecycleEvent',
    'GDPRComplianceHandler',
    'get_lifecycle_manager',
    'initialize_lifecycle_manager',
    
    # Performance
    'PerformanceOptimizer',
    'PerformanceMetric',
    'OptimizationAction',
    'PerformanceThreshold',
    'PerformanceAlert',
    'get_performance_optimizer',
    'initialize_performance_optimizer',
    
    # Utility functions
    'create_service_response',
    'generate_request_id',
    'sanitize_user_data',
    
    # Constants
    'SERVICE_NAMES',
    'EVENT_TYPES',
    'LIFECYCLE_STAGES',
    'CHANNELS',
    'INTERVENTION_TYPES',
    'DEFAULT_CONFIG'
]
