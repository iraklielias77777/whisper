"""
Base classes for third-party integrations
"""

import abc
import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import hashlib
import hmac
import json

logger = logging.getLogger(__name__)


class BaseIntegration(abc.ABC):
    """
    Abstract base class for third-party integrations
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider_name = self.__class__.__name__.replace('Integration', '').lower()
        self.event_mapping = self.load_event_mapping()
        
    @abc.abstractmethod
    def load_event_mapping(self) -> Dict[str, str]:
        """Load event mapping configuration"""
        pass
    
    @abc.abstractmethod
    async def setup_integration(self, app_id: str) -> Dict[str, Any]:
        """Set up integration for an app"""
        pass
    
    @abc.abstractmethod
    def transform_event(self, external_event: Dict[str, Any]) -> Dict[str, Any]:
        """Transform external event to User Whisperer format"""
        pass
    
    def map_event_type(self, external_event_type: str) -> str:
        """Map external event type to User Whisperer event type"""
        return self.event_mapping.get(external_event_type, external_event_type.lower().replace(' ', '_'))
    
    async def store_event(self, event: Dict[str, Any]):
        """Store transformed event"""
        # This would be implemented by the actual integration
        # to store events in the User Whisperer system
        logger.info(f"Storing event: {event['event_type']}")
    
    def generate_event_id(self) -> str:
        """Generate unique event ID"""
        import uuid
        import time
        return f"ext_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"


class WebhookProcessor(abc.ABC):
    """
    Abstract base class for webhook processors
    """
    
    def __init__(self, integration: BaseIntegration, webhook_secret: Optional[str] = None):
        self.integration = integration
        self.webhook_secret = webhook_secret
        
    @abc.abstractmethod
    def verify_webhook_signature(self, payload: bytes, signature: str) -> bool:
        """Verify webhook signature"""
        pass
    
    @abc.abstractmethod
    async def process_webhook_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming webhook event"""
        pass
    
    def _verify_hmac_signature(
        self, 
        payload: bytes, 
        signature: str, 
        algorithm: str = 'sha256'
    ) -> bool:
        """Verify HMAC signature"""
        
        if not self.webhook_secret:
            logger.warning("No webhook secret configured, skipping signature verification")
            return True
        
        try:
            # Remove algorithm prefix if present
            if signature.startswith(f'{algorithm}='):
                signature = signature[len(f'{algorithm}='):]
            
            # Calculate expected signature
            expected_signature = hmac.new(
                self.webhook_secret.encode(),
                payload,
                getattr(hashlib, algorithm)
            ).hexdigest()
            
            # Compare signatures
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False


class EventBuffer:
    """
    Buffer for batching events before processing
    """
    
    def __init__(
        self, 
        max_size: int = 100, 
        flush_interval: float = 5.0,
        processor: Optional[Callable[[List[Dict]], None]] = None
    ):
        self.max_size = max_size
        self.flush_interval = flush_interval
        self.processor = processor
        self.buffer = []
        self.last_flush = datetime.now()
        self._flush_task = None
        
    async def add_event(self, event: Dict[str, Any]):
        """Add event to buffer"""
        
        self.buffer.append(event)
        
        # Flush if buffer is full
        if len(self.buffer) >= self.max_size:
            await self.flush()
        
        # Start flush timer if not already running
        if self._flush_task is None:
            self._flush_task = asyncio.create_task(self._flush_timer())
    
    async def flush(self):
        """Flush buffer"""
        
        if not self.buffer:
            return
        
        events = self.buffer.copy()
        self.buffer.clear()
        self.last_flush = datetime.now()
        
        if self.processor:
            try:
                if asyncio.iscoroutinefunction(self.processor):
                    await self.processor(events)
                else:
                    self.processor(events)
            except Exception as e:
                logger.error(f"Event processing failed: {e}")
                # Re-add events to buffer for retry
                self.buffer.extend(events)
    
    async def _flush_timer(self):
        """Timer-based flush"""
        
        try:
            while self.buffer:
                await asyncio.sleep(self.flush_interval)
                
                # Check if flush interval has passed
                if (datetime.now() - self.last_flush).total_seconds() >= self.flush_interval:
                    await self.flush()
        
        except asyncio.CancelledError:
            # Flush on cancellation
            await self.flush()
            raise
        
        finally:
            self._flush_task = None


class RateLimiter:
    """
    Simple rate limiter for webhook processing
    """
    
    def __init__(self, max_requests: int = 1000, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []
    
    def is_allowed(self) -> bool:
        """Check if request is allowed"""
        
        now = datetime.now()
        
        # Remove old requests
        cutoff = now.timestamp() - self.window_seconds
        self.requests = [req for req in self.requests if req > cutoff]
        
        # Check limit
        if len(self.requests) >= self.max_requests:
            return False
        
        # Add current request
        self.requests.append(now.timestamp())
        return True
    
    def get_reset_time(self) -> Optional[datetime]:
        """Get time when rate limit resets"""
        
        if not self.requests:
            return None
        
        oldest_request = min(self.requests)
        reset_time = datetime.fromtimestamp(oldest_request + self.window_seconds)
        
        return reset_time if reset_time > datetime.now() else None
