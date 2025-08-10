"""
API Gateway for User Whisperer Platform
Comprehensive gateway with rate limiting, load balancing, and service routing
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
import hashlib
import uuid

# FastAPI and middleware imports
try:
    from fastapi import FastAPI, Request, HTTPException, Depends, status, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from fastapi.responses import JSONResponse, PlainTextResponse
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Redis imports
try:
    import aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# HTTP client imports
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Metrics (if Prometheus is available)
if PROMETHEUS_AVAILABLE:
    request_counter = Counter(
        'api_requests_total',
        'Total API requests',
        ['method', 'endpoint', 'status_code', 'service']
    )
    
    request_duration = Histogram(
        'api_request_duration_seconds',
        'API request duration',
        ['method', 'endpoint', 'service']
    )
    
    active_connections = Gauge(
        'api_active_connections',
        'Number of active connections'
    )
    
    rate_limit_counter = Counter(
        'api_rate_limit_exceeded_total',
        'Total rate limit exceeded events',
        ['client_id', 'endpoint']
    )
    
    service_health = Gauge(
        'api_service_health',
        'Service health status (1=healthy, 0=unhealthy)',
        ['service']
    )


class APIGateway:
    """
    Comprehensive API Gateway for User Whisperer platform
    """
    
    def __init__(self, config: Dict[str, Any]):
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required for APIGateway")
        
        self.config = config
        self.app = FastAPI(
            title="User Whisperer API Gateway",
            version="1.0.0",
            description="Unified API for the User Whisperer platform",
            docs_url="/docs" if config.get('enable_docs', True) else None,
            redoc_url="/redoc" if config.get('enable_docs', True) else None,
            openapi_url="/openapi.json" if config.get('enable_docs', True) else None
        )
        
        # Redis connection for rate limiting and caching
        self.redis = None
        
        # Service registry
        self.services = config.get('services', {})
        
        # Rate limiting configuration
        self.rate_limits = config.get('rate_limits', {
            'default': {'requests': 1000, 'window': 3600},  # 1000 requests per hour
            'authenticated': {'requests': 5000, 'window': 3600},  # 5000 requests per hour
            'premium': {'requests': 10000, 'window': 3600}  # 10000 requests per hour
        })
        
        # Load balancing configuration
        self.load_balancer_config = config.get('load_balancer', {
            'strategy': 'round_robin',
            'health_check_interval': 30,
            'unhealthy_threshold': 3,
            'healthy_threshold': 2
        })
        
        # Circuit breaker configuration
        self.circuit_breaker_config = config.get('circuit_breaker', {
            'failure_threshold': 5,
            'recovery_timeout': 60,
            'expected_failure_rate': 0.5
        })
        
        # Service health tracking
        self.service_health_status = {}
        self.service_metrics = {}
        
        # Active connections tracking
        self.active_connections_count = 0
        
        # HTTP client session
        self.http_session = None
        
        # Initialize security
        self.security = HTTPBearer(auto_error=False)
        
        # Setup middleware and routes
        self.setup_middleware()
        self.setup_routes()
        
        logger.info("API Gateway initialized")
    
    def setup_middleware(self):
        """Configure middleware stack"""
        
        # Trusted hosts (production security)
        if self.config.get('trusted_hosts'):
            self.app.add_middleware(
                TrustedHostMiddleware,
                allowed_hosts=self.config['trusted_hosts']
            )
        
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.get('cors_origins', ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["X-Process-Time", "X-Request-ID"]
        )
        
        # Compression
        self.app.add_middleware(
            GZipMiddleware,
            minimum_size=1000
        )
        
        # Custom middleware
        @self.app.middleware("http")
        async def request_middleware(request: Request, call_next):
            # Generate request ID
            request_id = str(uuid.uuid4())
            request.state.request_id = request_id
            
            # Track active connections
            self.active_connections_count += 1
            if PROMETHEUS_AVAILABLE:
                active_connections.set(self.active_connections_count)
            
            start_time = time.time()
            
            try:
                # Add request ID header
                response = await call_next(request)
                
                # Add timing and request ID headers
                process_time = time.time() - start_time
                response.headers["X-Process-Time"] = str(process_time)
                response.headers["X-Request-ID"] = request_id
                
                # Record metrics
                if PROMETHEUS_AVAILABLE:
                    service_name = getattr(request.state, 'service_name', 'unknown')
                    
                    request_counter.labels(
                        method=request.method,
                        endpoint=request.url.path,
                        status_code=response.status_code,
                        service=service_name
                    ).inc()
                    
                    request_duration.labels(
                        method=request.method,
                        endpoint=request.url.path,
                        service=service_name
                    ).observe(process_time)
                
                return response
                
            except Exception as e:
                # Log error
                logger.error(f"Request {request_id} failed: {e}")
                
                # Record error metrics
                if PROMETHEUS_AVAILABLE:
                    request_counter.labels(
                        method=request.method,
                        endpoint=request.url.path,
                        status_code=500,
                        service='gateway'
                    ).inc()
                
                # Return error response
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "Internal server error",
                        "request_id": request_id
                    }
                )
            
            finally:
                # Update active connections
                self.active_connections_count -= 1
                if PROMETHEUS_AVAILABLE:
                    active_connections.set(self.active_connections_count)
        
        @self.app.middleware("http")
        async def rate_limit_middleware(request: Request, call_next):
            # Rate limiting
            client_id = await self.get_client_id(request)
            rate_limit_key = f"rate_limit:{client_id}"
            
            if not await self.check_rate_limit(client_id, request.url.path):
                if PROMETHEUS_AVAILABLE:
                    rate_limit_counter.labels(
                        client_id=client_id,
                        endpoint=request.url.path
                    ).inc()
                
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Rate limit exceeded",
                        "retry_after": 60
                    },
                    headers={"Retry-After": "60"}
                )
            
            return await call_next(request)
    
    def setup_routes(self):
        """Set up API routes"""
        
        # Health check
        @self.app.get("/health", 
                     summary="Health Check", 
                     description="Check API Gateway health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0",
                "services": await self.get_service_health_summary()
            }
        
        # Readiness probe
        @self.app.get("/ready")
        async def readiness_check():
            # Check if all critical services are healthy
            critical_services = ['event_ingestion', 'behavioral_analysis']
            
            for service in critical_services:
                if not await self.is_service_healthy(service):
                    return JSONResponse(
                        status_code=503,
                        content={"status": "not_ready", "unhealthy_services": [service]}
                    )
            
            return {"status": "ready"}
        
        # Metrics endpoint
        if PROMETHEUS_AVAILABLE:
            @self.app.get("/metrics", response_class=PlainTextResponse)
            async def metrics():
                return generate_latest()
        
        # Event tracking endpoints
        @self.app.post("/v1/events/track",
                      summary="Track Event",
                      description="Track a single user event")
        async def track_event(
            event: Dict[str, Any],
            request: Request,
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            await self.verify_authentication(credentials)
            
            request.state.service_name = 'event_ingestion'
            
            return await self.route_to_service(
                'event_ingestion',
                'POST',
                '/events/track',
                event,
                request
            )
        
        # Batch event tracking
        @self.app.post("/v1/events/batch",
                      summary="Track Events Batch",
                      description="Track multiple events in a single request")
        async def track_batch(
            data: Dict[str, Any],
            request: Request,
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            await self.verify_authentication(credentials)
            
            events = data.get('events', [])
            
            # Validate batch size
            max_batch_size = self.config.get('max_batch_size', 1000)
            if len(events) > max_batch_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"Batch size exceeds limit ({max_batch_size})"
                )
            
            request.state.service_name = 'event_ingestion'
            
            return await self.route_to_service(
                'event_ingestion',
                'POST',
                '/events/batch',
                data,
                request
            )
        
        # User identification
        @self.app.post("/v1/users/identify",
                      summary="Identify User",
                      description="Identify and update user information")
        async def identify_user(
            user_data: Dict[str, Any],
            request: Request,
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            await self.verify_authentication(credentials)
            
            request.state.service_name = 'event_ingestion'
            
            return await self.route_to_service(
                'event_ingestion',
                'POST',
                '/users/identify',
                user_data,
                request
            )
        
        # Get user profile
        @self.app.get("/v1/users/{user_id}",
                     summary="Get User Profile",
                     description="Retrieve user profile and behavioral data")
        async def get_user(
            user_id: str,
            request: Request,
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            await self.verify_authentication(credentials)
            
            request.state.service_name = 'behavioral_analysis'
            
            return await self.route_to_service(
                'behavioral_analysis',
                'GET',
                f'/users/{user_id}',
                None,
                request
            )
        
        # Get user behavioral analysis
        @self.app.get("/v1/users/{user_id}/analysis",
                     summary="Get User Analysis",
                     description="Get behavioral analysis for a user")
        async def get_user_analysis(
            user_id: str,
            include_predictions: bool = True,
            request: Request,
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            await self.verify_authentication(credentials)
            
            request.state.service_name = 'behavioral_analysis'
            
            params = {'include_predictions': include_predictions}
            
            return await self.route_to_service(
                'behavioral_analysis',
                'GET',
                f'/users/{user_id}/analysis',
                None,
                request,
                params=params
            )
        
        # Get decisions
        @self.app.get("/v1/decisions",
                     summary="Get Decisions",
                     description="Retrieve intervention decisions")
        async def get_decisions(
            user_id: Optional[str] = None,
            limit: int = 100,
            offset: int = 0,
            request: Request,
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            await self.verify_authentication(credentials)
            
            request.state.service_name = 'decision_engine'
            
            params = {
                'user_id': user_id,
                'limit': min(limit, 1000),  # Cap at 1000
                'offset': offset
            }
            
            return await self.route_to_service(
                'decision_engine',
                'GET',
                '/decisions',
                None,
                request,
                params=params
            )
        
        # Trigger decision
        @self.app.post("/v1/decisions/trigger",
                      summary="Trigger Decision",
                      description="Trigger intervention decision for a user")
        async def trigger_decision(
            decision_request: Dict[str, Any],
            request: Request,
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            await self.verify_authentication(credentials)
            
            request.state.service_name = 'decision_engine'
            
            return await self.route_to_service(
                'decision_engine',
                'POST',
                '/decisions/trigger',
                decision_request,
                request
            )
        
        # Generate content
        @self.app.post("/v1/content/generate",
                      summary="Generate Content",
                      description="Generate personalized content")
        async def generate_content(
            content_request: Dict[str, Any],
            request: Request,
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            await self.verify_authentication(credentials)
            
            request.state.service_name = 'content_generation'
            
            return await self.route_to_service(
                'content_generation',
                'POST',
                '/content/generate',
                content_request,
                request
            )
        
        # Send message
        @self.app.post("/v1/messages/send",
                      summary="Send Message",
                      description="Send message via channel orchestration")
        async def send_message(
            message_request: Dict[str, Any],
            request: Request,
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            await self.verify_authentication(credentials)
            
            request.state.service_name = 'channel_orchestration'
            
            return await self.route_to_service(
                'channel_orchestration',
                'POST',
                '/messages/send',
                message_request,
                request
            )
        
        # Webhook endpoints
        @self.app.post("/webhooks/ga4/{app_id}",
                      summary="GA4 Webhook",
                      description="Process Google Analytics 4 webhook")
        async def ga4_webhook(
            app_id: str,
            data: Dict[str, Any],
            request: Request
        ):
            # Verify webhook authenticity
            if not await self.verify_webhook(request, 'ga4'):
                raise HTTPException(status_code=401, detail="Invalid webhook signature")
            
            # Process webhook
            return await self.process_webhook('ga4', app_id, data, request)
        
        @self.app.post("/webhooks/mixpanel/{app_id}",
                      summary="Mixpanel Webhook", 
                      description="Process Mixpanel webhook")
        async def mixpanel_webhook(
            app_id: str,
            data: Dict[str, Any],
            request: Request
        ):
            # Verify webhook authenticity
            if not await self.verify_webhook(request, 'mixpanel'):
                raise HTTPException(status_code=401, detail="Invalid webhook signature")
            
            # Process webhook
            return await self.process_webhook('mixpanel', app_id, data, request)
        
        # Service management endpoints
        @self.app.get("/admin/services",
                     summary="List Services",
                     description="List all registered services")
        async def list_services(
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            await self.verify_admin_access(credentials)
            
            return {
                "services": list(self.services.keys()),
                "health_status": self.service_health_status
            }
        
        @self.app.get("/admin/services/{service_name}/health",
                     summary="Service Health",
                     description="Get detailed health information for a service")
        async def service_health_detail(
            service_name: str,
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            await self.verify_admin_access(credentials)
            
            if service_name not in self.services:
                raise HTTPException(status_code=404, detail="Service not found")
            
            health_info = await self.get_service_health_detail(service_name)
            
            return {
                "service": service_name,
                "health": health_info
            }
    
    async def verify_authentication(self, credentials: Optional[HTTPAuthorizationCredentials]):
        """Verify API authentication"""
        
        if not credentials:
            raise HTTPException(
                status_code=401,
                detail="Missing authentication credentials"
            )
        
        api_key = credentials.credentials
        
        # Validate API key
        if not await self.validate_api_key(api_key):
            raise HTTPException(
                status_code=401,
                detail="Invalid API key"
            )
    
    async def verify_admin_access(self, credentials: Optional[HTTPAuthorizationCredentials]):
        """Verify admin access"""
        
        await self.verify_authentication(credentials)
        
        # Additional admin validation would go here
        # For now, just ensure authentication
    
    async def validate_api_key(self, api_key: str) -> bool:
        """Validate API key"""
        
        # Check cache first
        cache_key = f"api_key:{hashlib.sha256(api_key.encode()).hexdigest()}"
        
        if self.redis:
            try:
                cached = await self.redis.get(cache_key)
                if cached:
                    return cached.decode() == '1'
            except Exception as e:
                logger.warning(f"Redis cache check failed: {e}")
        
        # Validate against database/service
        is_valid = await self.check_api_key_in_store(api_key)
        
        # Cache result
        if self.redis and is_valid:
            try:
                await self.redis.setex(cache_key, 300, '1')  # Cache for 5 minutes
            except Exception as e:
                logger.warning(f"Redis cache write failed: {e}")
        
        return is_valid
    
    async def check_api_key_in_store(self, api_key: str) -> bool:
        """Check API key in persistent store"""
        
        # This would implement actual database/service check
        # For now, simple validation
        return len(api_key) >= 32 and api_key.startswith('uw_')
    
    async def get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting"""
        
        # Try to get from API key
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            api_key = auth_header[7:]
            return hashlib.sha256(api_key.encode()).hexdigest()[:16]
        
        # Fall back to IP address
        return request.client.host if request.client else 'unknown'
    
    async def check_rate_limit(self, client_id: str, endpoint: str) -> bool:
        """Check rate limit for client"""
        
        if not self.redis:
            return True  # No rate limiting if Redis not available
        
        # Determine rate limit based on client
        rate_limit = self.get_rate_limit_for_client(client_id)
        
        key = f"rate_limit:{client_id}:{endpoint}"
        window = rate_limit['window']
        limit = rate_limit['requests']
        
        try:
            # Use sliding window rate limiting
            now = time.time()
            pipeline = self.redis.pipeline()
            
            # Remove old entries
            pipeline.zremrangebyscore(key, 0, now - window)
            
            # Count current requests
            pipeline.zcard(key)
            
            # Add current request
            pipeline.zadd(key, {str(now): now})
            
            # Set expiry
            pipeline.expire(key, int(window))
            
            results = await pipeline.execute()
            current_count = results[1]
            
            return current_count < limit
            
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            return True  # Allow on error
    
    def get_rate_limit_for_client(self, client_id: str) -> Dict[str, int]:
        """Get rate limit configuration for client"""
        
        # This would implement logic to determine client tier
        # For now, return default
        return self.rate_limits['default']
    
    async def route_to_service(
        self,
        service_name: str,
        method: str,
        path: str,
        data: Optional[Dict] = None,
        request: Request = None,
        params: Optional[Dict] = None
    ) -> Dict:
        """Route request to backend service with load balancing"""
        
        service_config = self.services.get(service_name)
        
        if not service_config:
            raise HTTPException(
                status_code=503,
                detail=f"Service {service_name} not available"
            )
        
        # Get healthy service instance
        service_url = await self.get_healthy_service_url(service_name)
        
        if not service_url:
            raise HTTPException(
                status_code=503,
                detail=f"No healthy instances available for {service_name}"
            )
        
        # Make request to service
        return await self.make_service_request(
            service_url,
            method,
            path,
            data,
            params,
            request
        )
    
    async def get_healthy_service_url(self, service_name: str) -> Optional[str]:
        """Get URL of healthy service instance"""
        
        service_config = self.services[service_name]
        instances = service_config.get('instances', [service_config.get('url')])
        
        # Filter healthy instances
        healthy_instances = []
        
        for instance in instances:
            if isinstance(instance, str):
                instance_url = instance
            else:
                instance_url = instance.get('url')
            
            if await self.is_service_healthy(service_name, instance_url):
                healthy_instances.append(instance_url)
        
        if not healthy_instances:
            return None
        
        # Simple round-robin selection
        # In production, would use more sophisticated load balancing
        import random
        return random.choice(healthy_instances)
    
    async def is_service_healthy(self, service_name: str, instance_url: str = None) -> bool:
        """Check if service instance is healthy"""
        
        health_key = f"{service_name}:{instance_url}" if instance_url else service_name
        
        # Check cached health status
        if health_key in self.service_health_status:
            health_info = self.service_health_status[health_key]
            
            # Check if health check is recent
            if (datetime.now() - health_info['last_check']).total_seconds() < 30:
                return health_info['healthy']
        
        # Perform health check
        try:
            if not self.http_session:
                connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
                timeout = aiohttp.ClientTimeout(total=5)
                self.http_session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout
                )
            
            service_config = self.services[service_name]
            base_url = instance_url or service_config.get('url')
            health_path = service_config.get('health_path', '/health')
            
            health_url = f"{base_url.rstrip('/')}{health_path}"
            
            async with self.http_session.get(health_url) as response:
                is_healthy = response.status == 200
                
                # Update health status
                self.service_health_status[health_key] = {
                    'healthy': is_healthy,
                    'last_check': datetime.now(),
                    'status_code': response.status
                }
                
                # Update Prometheus metrics
                if PROMETHEUS_AVAILABLE:
                    service_health.labels(service=service_name).set(1 if is_healthy else 0)
                
                return is_healthy
        
        except Exception as e:
            logger.warning(f"Health check failed for {service_name}: {e}")
            
            # Update health status
            self.service_health_status[health_key] = {
                'healthy': False,
                'last_check': datetime.now(),
                'error': str(e)
            }
            
            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE:
                service_health.labels(service=service_name).set(0)
            
            return False
    
    async def make_service_request(
        self,
        service_url: str,
        method: str,
        path: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        request: Request = None
    ) -> Dict:
        """Make HTTP request to service"""
        
        if not self.http_session:
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
            timeout = aiohttp.ClientTimeout(total=30)
            self.http_session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
        
        url = f"{service_url.rstrip('/')}{path}"
        
        # Prepare headers
        headers = {
            'Content-Type': 'application/json',
            'X-Gateway-Request': '1'
        }
        
        # Forward request ID if available
        if request and hasattr(request.state, 'request_id'):
            headers['X-Request-ID'] = request.state.request_id
        
        # Forward original headers (selectively)
        if request:
            forwarded_headers = ['User-Agent', 'X-Forwarded-For', 'X-Real-IP']
            for header in forwarded_headers:
                if header in request.headers:
                    headers[header] = request.headers[header]
        
        try:
            async with self.http_session.request(
                method,
                url,
                json=data,
                params=params,
                headers=headers
            ) as response:
                
                if response.status >= 500:
                    # Server error - could retry with different instance
                    logger.error(f"Service error {response.status} from {url}")
                    
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Service temporarily unavailable"
                    )
                
                elif response.status >= 400:
                    # Client error - forward as-is
                    error_data = await response.json() if response.content_type == 'application/json' else {"error": await response.text()}
                    
                    raise HTTPException(
                        status_code=response.status,
                        detail=error_data
                    )
                
                # Success response
                if response.content_type == 'application/json':
                    return await response.json()
                else:
                    return {"response": await response.text()}
        
        except aiohttp.ClientError as e:
            logger.error(f"Service request failed to {url}: {e}")
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable"
            )
    
    async def verify_webhook(self, request: Request, provider: str) -> bool:
        """Verify webhook authenticity"""
        
        if provider == 'ga4':
            # Verify Google signature
            signature = request.headers.get('X-Google-Signature')
            # Implement actual GA4 signature verification
            return True
        
        elif provider == 'mixpanel':
            # Verify Mixpanel signature
            signature = request.headers.get('X-Mixpanel-Signature')
            # Implement actual Mixpanel signature verification
            return True
        
        return False
    
    async def process_webhook(
        self,
        provider: str,
        app_id: str,
        data: Dict,
        request: Request
    ) -> Dict:
        """Process webhook data"""
        
        # Route to event ingestion service
        return await self.route_to_service(
            'event_ingestion',
            'POST',
            f'/webhooks/{provider}',
            {
                'app_id': app_id,
                'provider': provider,
                'data': data
            },
            request
        )
    
    async def get_service_health_summary(self) -> Dict[str, Any]:
        """Get health summary for all services"""
        
        summary = {}
        
        for service_name in self.services:
            is_healthy = await self.is_service_healthy(service_name)
            summary[service_name] = {
                'healthy': is_healthy,
                'status': 'healthy' if is_healthy else 'unhealthy'
            }
        
        return summary
    
    async def get_service_health_detail(self, service_name: str) -> Dict[str, Any]:
        """Get detailed health information for a service"""
        
        service_config = self.services.get(service_name, {})
        instances = service_config.get('instances', [service_config.get('url')])
        
        instance_health = {}
        
        for instance in instances:
            instance_url = instance if isinstance(instance, str) else instance.get('url')
            is_healthy = await self.is_service_healthy(service_name, instance_url)
            
            health_key = f"{service_name}:{instance_url}"
            health_info = self.service_health_status.get(health_key, {})
            
            instance_health[instance_url] = {
                'healthy': is_healthy,
                'last_check': health_info.get('last_check', '').isoformat() if health_info.get('last_check') else None,
                'status_code': health_info.get('status_code'),
                'error': health_info.get('error')
            }
        
        overall_healthy = any(info['healthy'] for info in instance_health.values())
        
        return {
            'overall_healthy': overall_healthy,
            'instances': instance_health,
            'config': service_config
        }
    
    async def start(self):
        """Start the API gateway"""
        
        # Initialize Redis connection
        if REDIS_AVAILABLE and self.config.get('redis_url'):
            try:
                self.redis = await aioredis.from_url(self.config['redis_url'])
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
        
        # Start background health checks
        asyncio.create_task(self.health_check_loop())
        
        logger.info("API Gateway started")
    
    async def stop(self):
        """Stop the API gateway"""
        
        # Close HTTP session
        if self.http_session:
            await self.http_session.close()
        
        # Close Redis connection
        if self.redis:
            await self.redis.close()
        
        logger.info("API Gateway stopped")
    
    async def health_check_loop(self):
        """Background health check loop"""
        
        while True:
            try:
                # Check health of all services
                for service_name in self.services:
                    await self.is_service_healthy(service_name)
                
                # Wait for next check
                await asyncio.sleep(self.load_balancer_config.get('health_check_interval', 30))
                
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(60)  # Wait longer on error


def create_app(config: Dict[str, Any]) -> FastAPI:
    """
    Create FastAPI application with API Gateway
    """
    
    gateway = APIGateway(config)
    
    @gateway.app.on_event("startup")
    async def startup():
        await gateway.start()
    
    @gateway.app.on_event("shutdown")
    async def shutdown():
        await gateway.stop()
    
    return gateway.app
