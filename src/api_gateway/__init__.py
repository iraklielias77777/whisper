# API Gateway for User Whisperer Platform
"""
Comprehensive API Gateway providing unified access to all User Whisperer services.
Includes rate limiting, authentication, load balancing, and request routing.
"""

__version__ = "1.0.0"

# Core gateway components
try:
    from .gateway import APIGateway, create_app
except ImportError as e:
    print(f"Warning: Core gateway not available: {e}")

# GraphQL API
try:
    from .graphql_api import (
        create_graphql_router,
        Query,
        Mutation,
        Subscription,
        schema
    )
except ImportError as e:
    print(f"Warning: GraphQL API not available: {e}")

# Authentication and authorization
try:
    from .auth import (
        AuthenticationMiddleware,
        AuthorizationMiddleware,
        APIKeyValidator,
        JWTValidator
    )
except ImportError as e:
    print(f"Warning: Auth components not available: {e}")

# Rate limiting
try:
    from .rate_limiter import (
        RateLimiter,
        RedisRateLimiter,
        MemoryRateLimiter,
        RateLimitMiddleware
    )
except ImportError as e:
    print(f"Warning: Rate limiter not available: {e}")

# Load balancing
try:
    from .load_balancer import (
        LoadBalancer,
        RoundRobinBalancer,
        WeightedBalancer,
        HealthChecker
    )
except ImportError as e:
    print(f"Warning: Load balancer not available: {e}")

# Request routing
try:
    from .router import (
        ServiceRouter,
        Route,
        RouteMatch,
        RequestRouter
    )
except ImportError as e:
    print(f"Warning: Router not available: {e}")

__all__ = [
    '__version__',
    
    # Core gateway
    'APIGateway',
    'create_app',
    
    # GraphQL
    'create_graphql_router',
    'Query',
    'Mutation', 
    'Subscription',
    'schema',
    
    # Authentication
    'AuthenticationMiddleware',
    'AuthorizationMiddleware',
    'APIKeyValidator',
    'JWTValidator',
    
    # Rate limiting
    'RateLimiter',
    'RedisRateLimiter',
    'MemoryRateLimiter',
    'RateLimitMiddleware',
    
    # Load balancing
    'LoadBalancer',
    'RoundRobinBalancer',
    'WeightedBalancer',
    'HealthChecker',
    
    # Routing
    'ServiceRouter',
    'Route',
    'RouteMatch',
    'RequestRouter'
]
