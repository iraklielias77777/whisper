"""
Event Stream Processor for User Whisperer Platform
Handles Google Pub/Sub integration and stream processing patterns
"""

import asyncio
from typing import Dict, List, Optional, Callable, Any
from google.cloud import pubsub_v1
from google.api_core import retry
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import uuid
import time

logger = logging.getLogger(__name__)

@dataclass
class StreamConfig:
    project_id: str
    topic_name: str
    subscription_name: str
    max_messages: int = 100
    ack_deadline: int = 60
    max_outstanding_messages: int = 1000
    enable_ordering: bool = True
    enable_exactly_once: bool = True

class EventStreamProcessor:
    """
    Processes events from Google Pub/Sub stream
    """
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.publisher = pubsub_v1.PublisherClient()
        self.subscriber = pubsub_v1.SubscriberClient()
        self.topic_path = self.publisher.topic_path(
            config.project_id,
            config.topic_name
        )
        self.subscription_path = self.subscriber.subscription_path(
            config.project_id,
            config.subscription_name
        )
        self.handlers = {}
        self.is_running = False
        self.message_count = 0
        self.error_count = 0
        self._shutdown_event = asyncio.Event()
        
    async def initialize(self):
        """Initialize stream infrastructure"""
        
        try:
            # Create topic if not exists
            try:
                self.publisher.create_topic(request={"name": self.topic_path})
                logger.info(f"Created topic: {self.topic_path}")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.error(f"Failed to create topic: {e}")
                    raise
                logger.info(f"Topic already exists: {self.topic_path}")
            
            # Create subscription if not exists
            try:
                subscription_config = {
                    "name": self.subscription_path,
                    "topic": self.topic_path,
                    "ack_deadline_seconds": self.config.ack_deadline,
                }
                
                # Add ordering and exactly-once if supported
                if self.config.enable_message_ordering:
                    subscription_config["enable_message_ordering"] = True
                    
                if self.config.enable_exactly_once:
                    subscription_config["enable_exactly_once_delivery"] = True
                
                self.subscriber.create_subscription(request=subscription_config)
                logger.info(f"Created subscription: {self.subscription_path}")
                
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.error(f"Failed to create subscription: {e}")
                    raise
                logger.info(f"Subscription already exists: {self.subscription_path}")
                
        except Exception as e:
            logger.error(f"Stream initialization failed: {e}")
            raise
    
    def register_handler(
        self,
        event_type: str,
        handler: Callable[[Dict], None]
    ):
        """Register event handler"""
        self.handlers[event_type] = handler
        logger.info(f"Registered handler for event type: {event_type}")
    
    def register_pattern_handler(
        self,
        pattern: str,
        handler: Callable[[Dict], None]
    ):
        """Register pattern-based handler (supports regex)"""
        import re
        self.handlers[re.compile(pattern)] = handler
        logger.info(f"Registered pattern handler: {pattern}")
    
    async def publish_event(
        self,
        event_type: str,
        event_data: Dict,
        ordering_key: Optional[str] = None,
        attributes: Optional[Dict[str, str]] = None
    ) -> str:
        """Publish event to stream"""
        
        try:
            message_data = {
                'event_type': event_type,
                'timestamp': datetime.utcnow().isoformat(),
                'data': event_data,
                'message_id': str(uuid.uuid4())
            }
            
            message_bytes = json.dumps(message_data).encode('utf-8')
            
            # Prepare message attributes
            msg_attributes = attributes or {}
            msg_attributes.update({
                'event_type': event_type,
                'timestamp': message_data['timestamp']
            })
            
            # Publish message
            publish_future = self.publisher.publish(
                self.topic_path,
                message_bytes,
                ordering_key=ordering_key or event_data.get('user_id', ''),
                **msg_attributes
            )
            
            # Wait for message ID
            message_id = await asyncio.get_event_loop().run_in_executor(
                None,
                publish_future.result,
                10.0  # 10 second timeout
            )
            
            logger.debug(f"Published event {event_type} with message ID: {message_id}")
            return message_id
            
        except Exception as e:
            logger.error(f"Failed to publish event {event_type}: {e}")
            raise
    
    async def start_processing(self):
        """Start processing events from stream"""
        
        if self.is_running:
            logger.warning("Stream processor is already running")
            return
            
        self.is_running = True
        logger.info(f"Starting event processing from {self.subscription_path}")
        
        try:
            flow_control = pubsub_v1.types.FlowControl(
                max_messages=self.config.max_outstanding_messages,
                max_bytes=1024 * 1024 * 100  # 100MB
            )
            
            def callback(message):
                """Callback for processing messages"""
                try:
                    # Process message asynchronously
                    asyncio.create_task(self.process_message(message))
                except Exception as e:
                    logger.error(f"Error creating message processing task: {e}")
                    message.nack()
            
            # Start streaming pull
            streaming_pull_future = self.subscriber.subscribe(
                self.subscription_path,
                callback=callback,
                flow_control=flow_control
            )
            
            logger.info("Event stream processor started successfully")
            
            # Keep processing until shutdown
            with self.subscriber:
                try:
                    while self.is_running:
                        await asyncio.sleep(1)
                        
                        # Log periodic statistics
                        if self.message_count % 1000 == 0 and self.message_count > 0:
                            logger.info(f"Processed {self.message_count} messages, {self.error_count} errors")
                            
                except KeyboardInterrupt:
                    logger.info("Received shutdown signal")
                finally:
                    await self.stop_processing()
                    streaming_pull_future.cancel()
                    
                    try:
                        streaming_pull_future.result()
                    except Exception:
                        pass  # Expected when cancelling
                        
        except Exception as e:
            logger.error(f"Stream processing failed: {e}")
            raise
    
    async def stop_processing(self):
        """Stop processing events"""
        
        if not self.is_running:
            return
            
        logger.info("Stopping event stream processor...")
        self.is_running = False
        self._shutdown_event.set()
        
        # Wait a moment for in-flight messages to complete
        await asyncio.sleep(2)
        
        logger.info(f"Stream processor stopped. Final stats: {self.message_count} messages processed, {self.error_count} errors")
    
    async def process_message(self, message):
        """Process individual message"""
        
        start_time = time.time()
        
        try:
            # Parse message
            event_data = json.loads(message.data.decode('utf-8'))
            event_type = event_data.get('event_type')
            
            if not event_type:
                logger.warning("Message missing event_type, acknowledging")
                message.ack()
                return
            
            # Find appropriate handler
            handler = self.find_handler(event_type)
            
            if handler:
                # Execute handler
                if asyncio.iscoroutinefunction(handler):
                    await handler(event_data['data'])
                else:
                    # Run sync handler in thread pool
                    await asyncio.get_event_loop().run_in_executor(
                        None, handler, event_data['data']
                    )
                
                # Acknowledge successful processing
                message.ack()
                self.message_count += 1
                
                # Track processing metrics
                processing_time = (time.time() - start_time) * 1000
                await self.track_processing_metrics(
                    event_type,
                    success=True,
                    processing_time_ms=processing_time
                )
                
            else:
                logger.warning(f"No handler found for event type: {event_type}")
                # Acknowledge to prevent redelivery of unhandled events
                message.ack()
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message JSON: {e}")
            message.ack()  # Acknowledge malformed messages to prevent infinite retry
            self.error_count += 1
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            
            # Nack message for retry
            message.nack()
            self.error_count += 1
            
            # Track error metrics
            await self.track_processing_metrics(
                event_data.get('event_type', 'unknown') if 'event_data' in locals() else 'unknown',
                success=False,
                error=str(e)
            )
    
    def find_handler(self, event_type: str) -> Optional[Callable]:
        """Find appropriate handler for event type"""
        
        # Direct match
        if event_type in self.handlers:
            return self.handlers[event_type]
        
        # Pattern match
        import re
        for pattern, handler in self.handlers.items():
            if isinstance(pattern, re.Pattern) and pattern.match(event_type):
                return handler
        
        return None
    
    async def track_processing_metrics(
        self,
        event_type: str,
        success: bool,
        processing_time_ms: Optional[float] = None,
        error: Optional[str] = None
    ):
        """Track event processing metrics"""
        
        # This would integrate with your metrics system
        # For now, just log
        if success:
            logger.debug(f"Successfully processed {event_type} in {processing_time_ms:.2f}ms")
        else:
            logger.error(f"Failed to process {event_type}: {error}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        
        return {
            'messages_processed': self.message_count,
            'errors': self.error_count,
            'handlers_registered': len(self.handlers),
            'is_running': self.is_running,
            'topic_path': self.topic_path,
            'subscription_path': self.subscription_path
        }


class StreamAggregator:
    """
    Aggregates events for batch processing
    """
    
    def __init__(self, batch_size: int = 100, timeout: float = 5.0):
        self.batch_size = batch_size
        self.timeout = timeout
        self.buffer = []
        self.lock = asyncio.Lock()
        self.flush_task = None
        self.total_processed = 0
        
    async def add_event(self, event: Dict):
        """Add event to aggregation buffer"""
        
        async with self.lock:
            self.buffer.append({
                'event': event,
                'added_at': datetime.utcnow()
            })
            
            if len(self.buffer) >= self.batch_size:
                await self.flush()
            elif not self.flush_task or self.flush_task.done():
                self.flush_task = asyncio.create_task(
                    self.auto_flush()
                )
    
    async def auto_flush(self):
        """Auto flush after timeout"""
        
        try:
            await asyncio.sleep(self.timeout)
            await self.flush()
        except asyncio.CancelledError:
            pass  # Expected when manually flushing
    
    async def flush(self):
        """Flush buffer for processing"""
        
        async with self.lock:
            if not self.buffer:
                return
            
            batch = [item['event'] for item in self.buffer]
            self.buffer.clear()
            
            if self.flush_task and not self.flush_task.done():
                self.flush_task.cancel()
                self.flush_task = None
        
        if batch:
            await self.process_batch(batch)
            self.total_processed += len(batch)
            logger.info(f"Processed batch of {len(batch)} events")
    
    async def process_batch(self, batch: List[Dict]):
        """Process aggregated batch - implement in subclasses"""
        
        # Default implementation - override in subclasses
        for event in batch:
            logger.debug(f"Processing event: {event.get('event_type', 'unknown')}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get aggregator statistics"""
        
        async with self.lock:
            buffer_size = len(self.buffer)
        
        return {
            'buffer_size': buffer_size,
            'total_processed': self.total_processed,
            'batch_size': self.batch_size,
            'timeout': self.timeout
        }


class EventRouter:
    """
    Routes events to appropriate processors
    """
    
    def __init__(self):
        self.routes = {}
        self.dead_letter_queue = []
        self.routed_count = 0
        self.dead_letter_count = 0
        
    def add_route(
        self,
        pattern: str,
        processor: Callable[[Dict], Any]
    ):
        """Add routing rule"""
        
        import re
        compiled_pattern = re.compile(pattern)
        self.routes[compiled_pattern] = processor
        logger.info(f"Added route: {pattern}")
    
    def add_exact_route(
        self,
        event_type: str,
        processor: Callable[[Dict], Any]
    ):
        """Add exact match routing rule"""
        
        self.routes[event_type] = processor
        logger.info(f"Added exact route: {event_type}")
    
    async def route_event(self, event: Dict) -> bool:
        """Route event to appropriate processor"""
        
        event_type = event.get('event_type', '')
        
        # Try exact match first
        if event_type in self.routes:
            processor = self.routes[event_type]
            try:
                if asyncio.iscoroutinefunction(processor):
                    await processor(event)
                else:
                    await asyncio.get_event_loop().run_in_executor(
                        None, processor, event
                    )
                
                self.routed_count += 1
                return True
                
            except Exception as e:
                logger.error(f"Processor failed for {event_type}: {e}")
                await self.send_to_dead_letter(event, str(e))
                return False
        
        # Try pattern matches
        import re
        for pattern, processor in self.routes.items():
            if isinstance(pattern, re.Pattern) and pattern.match(event_type):
                try:
                    if asyncio.iscoroutinefunction(processor):
                        await processor(event)
                    else:
                        await asyncio.get_event_loop().run_in_executor(
                            None, processor, event
                        )
                    
                    self.routed_count += 1
                    return True
                    
                except Exception as e:
                    logger.error(f"Processor failed for {event_type}: {e}")
                    await self.send_to_dead_letter(event, str(e))
                    return False
        
        # No matching route
        await self.send_to_dead_letter(event, "No matching route")
        return False
    
    async def send_to_dead_letter(
        self,
        event: Dict,
        reason: str
    ):
        """Send unprocessable event to dead letter queue"""
        
        dead_letter_item = {
            'event': event,
            'reason': reason,
            'timestamp': datetime.utcnow().isoformat(),
            'retry_count': event.get('retry_count', 0)
        }
        
        self.dead_letter_queue.append(dead_letter_item)
        self.dead_letter_count += 1
        
        # Limit dead letter queue size
        if len(self.dead_letter_queue) > 1000:
            self.dead_letter_queue = self.dead_letter_queue[-500:]  # Keep last 500
        
        logger.warning(f"Event sent to dead letter queue: {reason}")
        
        # Could also persist to database here
        # await self.persist_dead_letter(dead_letter_item)
    
    async def get_dead_letter_events(self, limit: int = 100) -> List[Dict]:
        """Get events from dead letter queue"""
        
        return self.dead_letter_queue[-limit:]
    
    async def retry_dead_letter_event(self, event_index: int) -> bool:
        """Retry a specific dead letter event"""
        
        if 0 <= event_index < len(self.dead_letter_queue):
            dead_letter_item = self.dead_letter_queue[event_index]
            event = dead_letter_item['event'].copy()
            event['retry_count'] = event.get('retry_count', 0) + 1
            
            # Attempt to route again
            success = await self.route_event(event)
            
            if success:
                # Remove from dead letter queue
                del self.dead_letter_queue[event_index]
                logger.info(f"Successfully retried dead letter event")
                
            return success
        
        return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        
        return {
            'routes_registered': len(self.routes),
            'events_routed': self.routed_count,
            'dead_letter_count': self.dead_letter_count,
            'dead_letter_queue_size': len(self.dead_letter_queue)
        }


class WindowAggregator:
    """
    Time-window based event aggregation
    """
    
    def __init__(
        self,
        window_size: timedelta,
        slide_interval: Optional[timedelta] = None
    ):
        self.window_size = window_size
        self.slide_interval = slide_interval or window_size
        self.windows = defaultdict(list)
        self.lock = asyncio.Lock()
        self._cleanup_task = None
        
    async def add_event(self, event: Dict):
        """Add event to appropriate window"""
        
        event_time = datetime.fromisoformat(
            event.get('timestamp', datetime.utcnow().isoformat())
        )
        window_key = self.get_window_key(event_time)
        
        async with self.lock:
            self.windows[window_key].append({
                'event': event,
                'window_time': event_time
            })
        
        # Start cleanup task if not running
        if not self._cleanup_task or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(
                self.periodic_cleanup()
            )
    
    def get_window_key(self, timestamp: datetime) -> str:
        """Calculate window key for timestamp"""
        
        # Align to window boundaries
        window_start = timestamp - (timestamp - datetime.min) % self.window_size
        return window_start.isoformat()
    
    async def get_window_results(
        self,
        window_key: str
    ) -> List[Dict]:
        """Get aggregated results for window"""
        
        async with self.lock:
            window_data = self.windows.get(window_key, [])
            return [item['event'] for item in window_data]
    
    async def get_current_window_results(self) -> List[Dict]:
        """Get results for current window"""
        
        current_key = self.get_window_key(datetime.utcnow())
        return await self.get_window_results(current_key)
    
    async def cleanup_old_windows(self, retention: timedelta):
        """Remove old windows past retention period"""
        
        cutoff = datetime.utcnow() - retention
        
        async with self.lock:
            old_keys = [
                key for key in self.windows
                if datetime.fromisoformat(key) < cutoff
            ]
            
            for key in old_keys:
                del self.windows[key]
                
        if old_keys:
            logger.info(f"Cleaned up {len(old_keys)} old windows")
    
    async def periodic_cleanup(self):
        """Periodic cleanup of old windows"""
        
        while True:
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                await self.cleanup_old_windows(timedelta(hours=24))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Window cleanup failed: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get window aggregator statistics"""
        
        async with self.lock:
            total_events = sum(len(events) for events in self.windows.values())
            window_count = len(self.windows)
        
        return {
            'window_count': window_count,
            'total_events': total_events,
            'window_size': str(self.window_size),
            'slide_interval': str(self.slide_interval)
        }


class SessionProcessor:
    """
    Session-based event processing
    """
    
    def __init__(
        self,
        session_timeout: timedelta = timedelta(minutes=30)
    ):
        self.session_timeout = session_timeout
        self.sessions = {}
        self.lock = asyncio.Lock()
        self.completed_sessions = 0
        self._cleanup_task = None
        
    async def process_event(self, event: Dict):
        """Process event within session context"""
        
        user_id = event.get('user_id')
        session_id = event.get('session_id')
        
        if not user_id:
            logger.warning("Event missing user_id, skipping session processing")
            return
        
        async with self.lock:
            # Find or create session
            if session_id and session_id in self.sessions:
                session = self.sessions[session_id]
            else:
                # Look for active session for user
                active_session = None
                for sid, sess in self.sessions.items():
                    if (sess['user_id'] == user_id and 
                        not self.is_session_expired(sess)):
                        active_session = sess
                        break
                
                if active_session:
                    session = active_session
                else:
                    # Create new session
                    session_id = session_id or self.generate_session_id()
                    session = {
                        'id': session_id,
                        'user_id': user_id,
                        'started_at': datetime.utcnow(),
                        'last_activity': datetime.utcnow(),
                        'events': [],
                        'metrics': {
                            'event_count': 0,
                            'event_types': set(),
                            'duration_seconds': 0
                        }
                    }
                    self.sessions[session_id] = session
            
            # Update session
            session['last_activity'] = datetime.utcnow()
            session['events'].append(event)
            
            # Update metrics
            self.update_session_metrics(session, event)
            
            # Check for session completion
            if self.is_session_complete(session):
                await self.finalize_session(session)
        
        # Start cleanup task if not running
        if not self._cleanup_task or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(
                self.periodic_cleanup()
            )
    
    def update_session_metrics(
        self,
        session: Dict,
        event: Dict
    ):
        """Update session metrics"""
        
        metrics = session['metrics']
        
        # Event count
        metrics['event_count'] += 1
        
        # Unique event types
        metrics['event_types'].add(event.get('event_type', 'unknown'))
        
        # Session duration
        duration = (
            session['last_activity'] - session['started_at']
        ).total_seconds()
        metrics['duration_seconds'] = duration
        
        # Add more metrics as needed
        if event.get('event_type') == 'page_view':
            metrics['pages_viewed'] = metrics.get('pages_viewed', 0) + 1
        elif event.get('event_type') == 'error':
            metrics['errors'] = metrics.get('errors', 0) + 1
    
    def is_session_complete(self, session: Dict) -> bool:
        """Check if session is complete"""
        
        # Session timeout
        if self.is_session_expired(session):
            return True
        
        # Explicit session end event
        if session['events']:
            last_event = session['events'][-1]
            if last_event.get('event_type') == 'session_end':
                return True
        
        return False
    
    def is_session_expired(self, session: Dict) -> bool:
        """Check if session has expired"""
        
        idle_time = datetime.utcnow() - session['last_activity']
        return idle_time > self.session_timeout
    
    async def finalize_session(self, session: Dict):
        """Finalize and persist session"""
        
        # Calculate final metrics
        session['ended_at'] = datetime.utcnow()
        session['total_duration'] = (
            session['ended_at'] - session['started_at']
        ).total_seconds()
        
        # Convert set to list for serialization
        session['metrics']['event_types'] = list(session['metrics']['event_types'])
        
        # Persist to database (implement as needed)
        await self.persist_session(session)
        
        # Remove from active sessions
        if session['id'] in self.sessions:
            del self.sessions[session['id']]
        
        self.completed_sessions += 1
        logger.info(f"Finalized session {session['id']} for user {session['user_id']}")
    
    async def persist_session(self, session: Dict):
        """Persist session to database"""
        
        # This would save session data to database
        # For now, just log
        logger.debug(f"Persisting session: {session['id']}")
    
    def generate_session_id(self) -> str:
        """Generate unique session ID"""
        
        return str(uuid.uuid4())
    
    async def periodic_cleanup(self):
        """Periodic cleanup of expired sessions"""
        
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                async with self.lock:
                    expired_sessions = [
                        session for session in self.sessions.values()
                        if self.is_session_expired(session)
                    ]
                    
                    for session in expired_sessions:
                        await self.finalize_session(session)
                        
                if expired_sessions:
                    logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Session cleanup failed: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get session processor statistics"""
        
        async with self.lock:
            active_sessions = len(self.sessions)
        
        return {
            'active_sessions': active_sessions,
            'completed_sessions': self.completed_sessions,
            'session_timeout': str(self.session_timeout)
        } 