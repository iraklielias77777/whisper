"""
User Whisperer Python SDK - Synchronous Client
"""

import json
import uuid
import time
import threading
import queue
import logging
import platform
import sys
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

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


class UserWhisperer:
    """
    Synchronous Python SDK for User Whisperer platform
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
        gzip: bool = True
    ):
        """
        Initialize User Whisperer client
        
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
        """
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
        
        # State
        self.event_queue = queue.Queue(maxsize=max_queue_size)
        self.user_id = None
        self.session_id = self._generate_session_id()
        self.anonymous_id = self._generate_anonymous_id()
        
        # Thread management
        self._stop_flush = threading.Event()
        self._flush_thread = None
        
        # HTTP session with retry strategy
        self._session = self._create_http_session()
        
        # Setup logging
        if debug:
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Start background flush thread
        if send_automatically:
            self.start_flush_thread()
        
        logger.debug(f"UserWhisperer initialized with app_id: {app_id}")
    
    def _create_http_session(self) -> requests.Session:
        """Create HTTP session with retry strategy"""
        
        session = requests.Session()
        
        # Retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=1,
            respect_retry_after_header=True
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Default headers
        session.headers.update({
            'User-Agent': f'userwhisperer-python/{__version__}',
            'Authorization': f'Bearer {self.api_key}',
            'X-App-ID': self.app_id,
            'Content-Type': 'application/json'
        })
        
        if self.gzip:
            session.headers['Accept-Encoding'] = 'gzip, deflate'
        
        return session
    
    def start_flush_thread(self):
        """Start background flush thread"""
        
        if self._flush_thread and self._flush_thread.is_alive():
            return
        
        self._stop_flush.clear()
        self._flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self._flush_thread.start()
        
        logger.debug("Flush thread started")
    
    def stop_flush_thread(self):
        """Stop background flush thread"""
        
        if self._flush_thread and self._flush_thread.is_alive():
            self._stop_flush.set()
            self._flush_thread.join(timeout=5.0)
        
        logger.debug("Flush thread stopped")
    
    def track(
        self,
        event_type: str,
        properties: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Track an event
        
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
            self.event_queue.put_nowait(event)
        except queue.Full:
            raise QueueFullError(f"Event queue is full (max size: {self.max_queue_size})")
        
        # Auto-flush if batch size reached
        if self.send_automatically and self.event_queue.qsize() >= self.batch_size:
            self.flush()
        
        logger.debug(f"Tracked event: {event_type} for user: {event['user_id']}")
        return event_id
    
    def identify(
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
        
        event_id = self.track(
            "$identify",
            identify_traits,
            user_id=user_id,
            timestamp=timestamp
        )
        
        logger.debug(f"User identified: {user_id}")
        return event_id
    
    def alias(
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
        
        event_id = self.track(
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
    
    def group(
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
        
        event_id = self.track(
            "$group",
            group_props,
            user_id=user_id,
            timestamp=timestamp
        )
        
        logger.debug(f"User grouped: {group_id}")
        return event_id
    
    def page(
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
        
        return self.track(
            "$page_view",
            page_props,
            user_id=user_id,
            timestamp=timestamp
        )
    
    def screen(
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
        
        return self.track(
            "$screen_view",
            screen_props,
            user_id=user_id,
            timestamp=timestamp
        )
    
    def flush(self) -> bool:
        """
        Flush event queue immediately
        
        Returns:
            True if successful, False otherwise
        """
        events = []
        
        # Collect events from queue
        try:
            while not self.event_queue.empty() and len(events) < self.batch_size:
                events.append(self.event_queue.get_nowait())
        except queue.Empty:
            pass
        
        if not events:
            return True
        
        return self._send_batch(events)
    
    def _flush_worker(self):
        """Background worker for flushing events"""
        
        while not self._stop_flush.is_set():
            try:
                # Wait for flush interval or stop signal
                if self._stop_flush.wait(timeout=self.flush_interval):
                    break
                
                # Flush events
                self.flush()
                
            except Exception as e:
                logger.error(f"Flush worker error: {e}")
                # Continue running even if flush fails
    
    def _send_batch(
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
        url = urljoin(self.endpoint, "/v1/events/batch")
        
        data = {
            "events": events,
            "sdk_version": __version__,
            "client_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            start_time = time.time()
            
            response = self._session.post(
                url,
                json=data,
                timeout=self.timeout
            )
            
            response_time = time.time() - start_time
            
            # Handle different response codes
            if response.status_code == 401:
                raise AuthenticationError("Invalid API key")
            elif response.status_code == 429:
                retry_after = response.headers.get('Retry-After')
                raise RateLimitError(
                    "Rate limit exceeded",
                    retry_after=int(retry_after) if retry_after else None
                )
            elif response.status_code >= 400:
                raise NetworkError(
                    f"HTTP {response.status_code}: {response.text}",
                    status_code=response.status_code,
                    response_body=response.text
                )
            
            response.raise_for_status()
            
            result = response.json()
            
            # Handle partial failures
            if result.get("failed"):
                logger.warning(f"Failed events: {result['failed']}")
                # Could implement retry logic for failed events
            
            success_count = len(events) - len(result.get("failed", []))
            logger.debug(f"Sent {success_count}/{len(events)} events in {response_time:.3f}s")
            
            return True
            
        except (RateLimitError, AuthenticationError):
            # Don't retry auth/rate limit errors
            raise
            
        except (requests.exceptions.RequestException, NetworkError) as e:
            logger.error(f"Failed to send events (attempt {retry_count + 1}): {e}")
            
            if retry_count < self.max_retries:
                # Exponential backoff with jitter
                delay = min(2 ** retry_count + (time.time() % 1), 30)
                time.sleep(delay)
                return self._send_batch(events, retry_count + 1)
            
            # Re-queue events for later
            for event in events:
                try:
                    self.event_queue.put_nowait(event)
                except queue.Full:
                    logger.warning("Queue full, dropping event")
                    break
            
            return False
    
    def _merge_context(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Merge provided context with default context"""
        
        default_context = {
            "library": {
                "name": "userwhisperer-python",
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
    
    def close(self):
        """
        Close the client and flush remaining events
        """
        logger.debug("Closing UserWhisperer client")
        
        # Stop flush thread
        self.stop_flush_thread()
        
        # Flush remaining events
        self.flush()
        
        # Close HTTP session
        if self._session:
            self._session.close()
        
        logger.debug("UserWhisperer client closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    def __del__(self):
        """Destructor"""
        try:
            self.close()
        except:
            pass  # Ignore errors during cleanup
