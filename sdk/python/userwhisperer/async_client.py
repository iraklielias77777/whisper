"""
User Whisperer Python SDK - Asynchronous Client
"""

import asyncio
import json
import time
import uuid
import platform
import sys
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Callable
import logging

try:
    import aiohttp
    from aiohttp import ClientTimeout, ClientSession
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from .__version__ import __version__
from .exceptions import (
    UserWhispererError,
    AuthenticationError,
    ValidationError,
    RateLimitError,
    NetworkError,
    ConfigurationError,
    QueueFullError
)

logger = logging.getLogger(__name__)


class AsyncUserWhisperer:
    """
    Asynchronous Python SDK for User Whisperer platform
    """
    
    def __init__(
        self,
        api_key: str,
        app_id: str,
        endpoint: str = "https://api.userwhisperer.ai",
        debug: bool = False,
        batch_size: int = 100,
        flush_interval: float = 5.0,
        timeout: float = 10.0,
        max_retries: int = 3,
        max_queue_size: int = 10000,
        send_automatically: bool = True,
        gzip: bool = True,
        connector_limit: int = 100
    ):
        """
        Initialize Async User Whisperer client
        
        Args:
            api_key: API key for authentication
            app_id: Application ID
            endpoint: API endpoint URL
            debug: Enable debug logging
            batch_size: Number of events to batch before sending
            flush_interval: Interval in seconds to flush events
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            max_queue_size: Maximum size of event queue
            send_automatically: Whether to send events automatically
            gzip: Enable gzip compression
            connector_limit: Maximum number of connections
        """
        if not AIOHTTP_AVAILABLE:
            raise ConfigurationError("aiohttp is required for AsyncUserWhisperer. Install with: pip install userwhisperer[async]")
        
        # Validate required parameters
        if not api_key:
            raise ConfigurationError("api_key is required")
        if not app_id:
            raise ConfigurationError("app_id is required")
        
        self.api_key = api_key
        self.app_id = app_id
        self.endpoint = endpoint.rstrip('/')
        self.debug = debug
        self.batch_size = max(1, batch_size)
        self.flush_interval = max(0.1, flush_interval)
        self.timeout = max(1.0, timeout)
        self.max_retries = max(0, max_retries)
        self.max_queue_size = max(1, max_queue_size)
        self.send_automatically = send_automatically
        self.gzip = gzip
        self.connector_limit = connector_limit
        
        # State
        self.event_queue = asyncio.Queue(maxsize=max_queue_size)
        self.user_id = None
        self.session_id = self._generate_session_id()
        self.anonymous_id = self._generate_anonymous_id()
        
        # HTTP session
        self.session = None
        self.flush_task = None
        self._closed = False
        
        # Setup logging
        if debug:
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        logger.debug(f"AsyncUserWhisperer initialized with app_id: {app_id}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def start(self):
        """Start the async client"""
        
        if self._closed:
            raise UserWhispererError("Client has been closed")
        
        if self.session is None:
            # Create HTTP session
            connector = aiohttp.TCPConnector(
                limit=self.connector_limit,
                limit_per_host=30,
                ttl_dns_cache=300,
                use_dns_cache=True
            )
            
            timeout = ClientTimeout(total=self.timeout)
            
            headers = {
                'User-Agent': f'userwhisperer-python-async/{__version__}',
                'Authorization': f'Bearer {self.api_key}',
                'X-App-ID': self.app_id,
                'Content-Type': 'application/json'
            }
            
            self.session = ClientSession(
                connector=connector,
                timeout=timeout,
                headers=headers,
                raise_for_status=False
            )
            
            logger.debug("HTTP session created")
        
        # Start automatic flushing
        if self.send_automatically and self.flush_task is None:
            self.flush_task = asyncio.create_task(self._flush_worker())
            logger.debug("Flush task started")
    
    async def close(self):
        """Close the async client"""
        
        if self._closed:
            return
        
        self._closed = True
        
        logger.debug("Closing AsyncUserWhisperer client")
        
        # Cancel flush task
        if self.flush_task and not self.flush_task.done():
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass
        
        # Flush remaining events
        await self.flush()
        
        # Close HTTP session
        if self.session:
            await self.session.close()
            self.session = None
        
        logger.debug("AsyncUserWhisperer client closed")
    
    async def track(
        self,
        event_type: str,
        properties: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Track an event asynchronously
        
        Args:
            event_type: Type of event to track
            properties: Event properties
            user_id: User ID (overrides current user)
            timestamp: Event timestamp
            context: Additional context
            
        Returns:
            Event ID
            
        Raises:
            ValidationError: If event data is invalid
            QueueFullError: If event queue is full
        """
        if self._closed:
            raise UserWhispererError("Client has been closed")
        
        if not self._validate_event_type(event_type):
            raise ValidationError(f"Invalid event type: {event_type}")
        
        event_id = self._generate_event_id()
        
        event = {
            "id": event_id,
            "app_id": self.app_id,
            "user_id": user_id or self.user_id or self.anonymous_id,
            "session_id": self.session_id,
            "event_type": event_type,
            "properties": self._sanitize_properties(properties or {}),
            "context": self._merge_context(context),
            "timestamp": self._format_timestamp(timestamp or datetime.now(timezone.utc))
        }
        
        # Add to queue
        try:
            await self.event_queue.put(event)
        except asyncio.QueueFull:
            raise QueueFullError(f"Event queue is full (max size: {self.max_queue_size})")
        
        # Auto-flush if batch size reached
        if self.send_automatically and self.event_queue.qsize() >= self.batch_size:
            asyncio.create_task(self.flush())
        
        logger.debug(f"Tracked event: {event_type} for user: {event['user_id']}")
        return event_id
    
    async def identify(
        self,
        user_id: str,
        traits: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Identify a user
        
        Args:
            user_id: User identifier
            traits: User traits/properties
            timestamp: Identification timestamp
            
        Returns:
            Event ID
        """
        if not user_id:
            raise ValidationError("user_id is required")
        
        # Update current user
        previous_user_id = self.user_id
        self.user_id = user_id
        
        # Track identify event
        identify_traits = dict(traits or {})
        identify_traits.update({
            "$user_id": user_id,
            "$previous_id": previous_user_id
        })
        
        event_id = await self.track(
            "$identify",
            identify_traits,
            user_id=user_id,
            timestamp=timestamp
        )
        
        logger.debug(f"User identified: {user_id}")
        return event_id
    
    async def alias(
        self,
        new_user_id: str,
        previous_user_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Alias a user ID
        
        Args:
            new_user_id: New user identifier
            previous_user_id: Previous user identifier
            timestamp: Alias timestamp
            
        Returns:
            Event ID
        """
        if not new_user_id:
            raise ValidationError("new_user_id is required")
        
        prev_id = previous_user_id or self.user_id or self.anonymous_id
        
        event_id = await self.track(
            "$alias",
            {
                "previous_id": prev_id,
                "new_id": new_user_id
            },
            timestamp=timestamp
        )
        
        # Update current user
        self.user_id = new_user_id
        
        logger.debug(f"User aliased: {prev_id} -> {new_user_id}")
        return event_id
    
    async def group(
        self,
        group_id: str,
        traits: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Associate user with a group
        
        Args:
            group_id: Group identifier
            traits: Group traits/properties
            user_id: User identifier
            timestamp: Group timestamp
            
        Returns:
            Event ID
        """
        if not group_id:
            raise ValidationError("group_id is required")
        
        group_props = dict(traits or {})
        group_props["group_id"] = group_id
        
        event_id = await self.track(
            "$group",
            group_props,
            user_id=user_id,
            timestamp=timestamp
        )
        
        logger.debug(f"User grouped: {group_id}")
        return event_id
    
    async def page(
        self,
        name: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Track a page view
        
        Args:
            name: Page name
            properties: Page properties
            user_id: User identifier
            timestamp: Page view timestamp
            
        Returns:
            Event ID
        """
        page_props = dict(properties or {})
        if name:
            page_props["page_name"] = name
        
        return await self.track(
            "$page_view",
            page_props,
            user_id=user_id,
            timestamp=timestamp
        )
    
    async def screen(
        self,
        name: str,
        properties: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Track a screen view
        
        Args:
            name: Screen name
            properties: Screen properties
            user_id: User identifier
            timestamp: Screen view timestamp
            
        Returns:
            Event ID
        """
        if not name:
            raise ValidationError("screen name is required")
        
        screen_props = dict(properties or {})
        screen_props["screen_name"] = name
        
        return await self.track(
            "$screen_view",
            screen_props,
            user_id=user_id,
            timestamp=timestamp
        )
    
    async def flush(self) -> bool:
        """
        Flush event queue immediately
        
        Returns:
            True if successful, False otherwise
        """
        if self._closed:
            return False
        
        events = []
        
        # Collect events from queue
        try:
            while not self.event_queue.empty() and len(events) < self.batch_size:
                try:
                    event = self.event_queue.get_nowait()
                    events.append(event)
                except asyncio.QueueEmpty:
                    break
        except Exception as e:
            logger.error(f"Error collecting events: {e}")
            return False
        
        if not events:
            return True
        
        return await self._send_batch(events)
    
    async def _flush_worker(self):
        """Background worker for flushing events"""
        
        try:
            while not self._closed:
                await asyncio.sleep(self.flush_interval)
                await self.flush()
        except asyncio.CancelledError:
            logger.debug("Flush worker cancelled")
            raise
        except Exception as e:
            logger.error(f"Flush worker error: {e}")
    
    async def _send_batch(
        self,
        events: List[Dict],
        retry_count: int = 0
    ) -> bool:
        """
        Send batch of events
        
        Args:
            events: List of events to send
            retry_count: Current retry attempt
            
        Returns:
            True if successful, False otherwise
        """
        if not self.session:
            await self.start()
        
        url = f"{self.endpoint}/v1/events/batch"
        
        data = {
            "events": events,
            "sdk_version": __version__,
            "client_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            start_time = time.time()
            
            async with self.session.post(url, json=data) as response:
                response_time = time.time() - start_time
                
                # Handle different response codes
                if response.status == 401:
                    raise AuthenticationError("Invalid API key")
                elif response.status == 429:
                    retry_after = response.headers.get('Retry-After')
                    raise RateLimitError(
                        "Rate limit exceeded",
                        retry_after=int(retry_after) if retry_after else None
                    )
                elif response.status >= 400:
                    error_text = await response.text()
                    raise NetworkError(
                        f"HTTP {response.status}: {error_text}",
                        status_code=response.status,
                        response_body=error_text
                    )
                
                result = await response.json()
                
                # Handle partial failures
                if result.get("failed"):
                    logger.warning(f"Failed events: {result['failed']}")
                
                success_count = len(events) - len(result.get("failed", []))
                logger.debug(f"Sent {success_count}/{len(events)} events in {response_time:.3f}s")
                
                return True
        
        except (RateLimitError, AuthenticationError):
            # Don't retry auth/rate limit errors
            raise
        
        except (aiohttp.ClientError, NetworkError) as e:
            logger.error(f"Failed to send events (attempt {retry_count + 1}): {e}")
            
            if retry_count < self.max_retries:
                # Exponential backoff with jitter
                delay = min(2 ** retry_count + (time.time() % 1), 30)
                await asyncio.sleep(delay)
                return await self._send_batch(events, retry_count + 1)
            
            # Re-queue events for later
            for event in events:
                try:
                    await self.event_queue.put(event)
                except asyncio.QueueFull:
                    logger.warning("Queue full, dropping event")
                    break
            
            return False
    
    def _merge_context(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Merge provided context with default context"""
        
        default_context = {
            "library": {
                "name": "userwhisperer-python-async",
                "version": __version__
            },
            "runtime": {
                "python_version": sys.version,
                "platform": platform.platform(),
                "processor": platform.processor()
            }
        }
        
        if context:
            # Deep merge contexts
            merged = default_context.copy()
            for key, value in context.items():
                if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key] = {**merged[key], **value}
                else:
                    merged[key] = value
            return merged
        
        return default_context
    
    def _sanitize_properties(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize event properties"""
        
        sanitized = {}
        
        for key, value in properties.items():
            # Skip None values and non-serializable types
            if value is None:
                continue
            
            try:
                # Test JSON serialization
                json.dumps(value)
                sanitized[key] = value
            except (TypeError, ValueError):
                logger.warning(f"Skipping non-serializable property: {key}")
        
        return sanitized
    
    def _validate_event_type(self, event_type: str) -> bool:
        """Validate event type"""
        
        if not isinstance(event_type, str):
            return False
        
        if not event_type or len(event_type) > 100:
            return False
        
        # Check pattern (alphanumeric, underscore, dollar)
        import re
        pattern = r'^[a-zA-Z_$][a-zA-Z0-9_]*$'
        return bool(re.match(pattern, event_type))
    
    def _format_timestamp(self, timestamp: datetime) -> str:
        """Format timestamp to ISO string"""
        
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        
        return timestamp.isoformat()
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        return f"evt_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"ses_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    
    def _generate_anonymous_id(self) -> str:
        """Generate anonymous user ID"""
        return f"anon_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    
    def get_user_id(self) -> Optional[str]:
        """Get current user ID"""
        return self.user_id
    
    def get_session_id(self) -> str:
        """Get current session ID"""
        return self.session_id
    
    def get_anonymous_id(self) -> str:
        """Get anonymous ID"""
        return self.anonymous_id
    
    def reset(self):
        """Reset user state"""
        
        self.user_id = None
        self.session_id = self._generate_session_id()
        self.anonymous_id = self._generate_anonymous_id()
        
        logger.debug("User state reset")
    
    def queue_size(self) -> int:
        """Get current queue size"""
        return self.event_queue.qsize()
    
    def is_closed(self) -> bool:
        """Check if client is closed"""
        return self._closed
