"""
Complex Event Processing (CEP) for User Whisperer Platform
Detects behavioral patterns and triggers intelligent interventions
"""

import asyncio
from typing import Dict, List, Optional, Callable, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class PatternType(Enum):
    SEQUENCE = "sequence"
    FREQUENCY = "frequency"
    ABSENCE = "absence"
    THRESHOLD = "threshold"
    CORRELATION = "correlation"

@dataclass
class PatternCondition:
    name: str
    pattern_type: PatternType
    events: List[str]
    timeframe: timedelta
    threshold: Optional[int] = None
    condition_func: Optional[Callable] = None
    metadata: Optional[Dict] = None

@dataclass
class PatternMatch:
    pattern_name: str
    user_id: str
    matched_events: List[Dict]
    match_timestamp: datetime
    confidence_score: float
    metadata: Dict

class ComplexEventProcessor:
    """
    Complex Event Processing (CEP) for pattern detection
    """
    
    def __init__(self, buffer_size: int = 1000, buffer_ttl: timedelta = timedelta(hours=1)):
        self.patterns = {}
        self.event_buffer = defaultdict(lambda: deque(maxlen=buffer_size))
        self.buffer_size = buffer_size
        self.buffer_ttl = buffer_ttl
        self.pattern_matches = []
        self.stats = {
            'events_processed': 0,
            'patterns_matched': 0,
            'patterns_registered': 0
        }
        self._cleanup_task = None
        
    def register_pattern(
        self,
        name: str,
        condition: PatternCondition,
        action: Callable[[str, PatternMatch], Any]
    ):
        """Register event pattern"""
        
        self.patterns[name] = {
            'condition': condition,
            'action': action,
            'match_count': 0,
            'last_match': None
        }
        
        self.stats['patterns_registered'] += 1
        logger.info(f"Registered pattern: {name}")
    
    async def process_event(self, event: Dict):
        """Process event against registered patterns"""
        
        user_id = event.get('user_id')
        if not user_id:
            logger.warning("Event missing user_id, skipping pattern processing")
            return
        
        # Add to buffer with timestamp
        buffer_item = {
            'event': event,
            'timestamp': datetime.utcnow(),
            'processed': False
        }
        
        self.event_buffer[user_id].append(buffer_item)
        self.stats['events_processed'] += 1
        
        # Check patterns for this user
        await self.check_patterns_for_user(user_id, event)
        
        # Start cleanup task if not running
        if not self._cleanup_task or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self.periodic_cleanup())
    
    async def check_patterns_for_user(self, user_id: str, current_event: Dict):
        """Check all patterns for a specific user"""
        
        user_buffer = list(self.event_buffer[user_id])
        
        for pattern_name, pattern_info in self.patterns.items():
            condition = pattern_info['condition']
            
            try:
                match = await self.evaluate_pattern(
                    condition, user_buffer, current_event, user_id
                )
                
                if match:
                    # Execute pattern action
                    await self.execute_pattern_action(
                        pattern_name, pattern_info, match
                    )
                    
            except Exception as e:
                logger.error(f"Error evaluating pattern {pattern_name}: {e}")
    
    async def evaluate_pattern(
        self,
        condition: PatternCondition,
        buffer: List[Dict],
        current_event: Dict,
        user_id: str
    ) -> Optional[PatternMatch]:
        """Evaluate a specific pattern condition"""
        
        # Filter events within timeframe
        cutoff_time = datetime.utcnow() - condition.timeframe
        recent_events = [
            item for item in buffer
            if item['timestamp'] > cutoff_time
        ]
        
        if condition.pattern_type == PatternType.SEQUENCE:
            return await self.check_sequence_pattern(
                condition, recent_events, current_event, user_id
            )
        elif condition.pattern_type == PatternType.FREQUENCY:
            return await self.check_frequency_pattern(
                condition, recent_events, current_event, user_id
            )
        elif condition.pattern_type == PatternType.ABSENCE:
            return await self.check_absence_pattern(
                condition, recent_events, current_event, user_id
            )
        elif condition.pattern_type == PatternType.THRESHOLD:
            return await self.check_threshold_pattern(
                condition, recent_events, current_event, user_id
            )
        elif condition.pattern_type == PatternType.CORRELATION:
            return await self.check_correlation_pattern(
                condition, recent_events, current_event, user_id
            )
        
        return None
    
    async def check_sequence_pattern(
        self,
        condition: PatternCondition,
        buffer: List[Dict],
        current_event: Dict,
        user_id: str
    ) -> Optional[PatternMatch]:
        """Check sequence pattern (events in specific order)"""
        
        if not condition.events:
            return None
        
        # Extract events from buffer
        events = [item['event'] for item in buffer]
        
        # Look for sequence
        sequence_index = 0
        matched_events = []
        
        for event in events:
            if sequence_index < len(condition.events):
                expected_event = condition.events[sequence_index]
                
                if self.event_matches_pattern(event, expected_event):
                    matched_events.append(event)
                    sequence_index += 1
        
        # Check if complete sequence was found
        if sequence_index == len(condition.events):
            return PatternMatch(
                pattern_name=condition.name,
                user_id=user_id,
                matched_events=matched_events,
                match_timestamp=datetime.utcnow(),
                confidence_score=1.0,
                metadata={'sequence_length': len(matched_events)}
            )
        
        return None
    
    async def check_frequency_pattern(
        self,
        condition: PatternCondition,
        buffer: List[Dict],
        current_event: Dict,
        user_id: str
    ) -> Optional[PatternMatch]:
        """Check frequency pattern (event occurs N times)"""
        
        if not condition.events or not condition.threshold:
            return None
        
        target_event = condition.events[0]
        matched_events = []
        
        for item in buffer:
            event = item['event']
            if self.event_matches_pattern(event, target_event):
                matched_events.append(event)
        
        if len(matched_events) >= condition.threshold:
            confidence = min(1.0, len(matched_events) / condition.threshold)
            
            return PatternMatch(
                pattern_name=condition.name,
                user_id=user_id,
                matched_events=matched_events,
                match_timestamp=datetime.utcnow(),
                confidence_score=confidence,
                metadata={'frequency': len(matched_events), 'threshold': condition.threshold}
            )
        
        return None
    
    async def check_absence_pattern(
        self,
        condition: PatternCondition,
        buffer: List[Dict],
        current_event: Dict,
        user_id: str
    ) -> Optional[PatternMatch]:
        """Check absence pattern (expected event didn't occur)"""
        
        if not condition.events:
            return None
        
        target_event = condition.events[0]
        
        # Check if target event is absent
        for item in buffer:
            event = item['event']
            if self.event_matches_pattern(event, target_event):
                return None  # Event was found, no absence
        
        # Event was absent
        return PatternMatch(
            pattern_name=condition.name,
            user_id=user_id,
            matched_events=[],
            match_timestamp=datetime.utcnow(),
            confidence_score=1.0,
            metadata={'absent_event': target_event, 'timeframe': str(condition.timeframe)}
        )
    
    async def check_threshold_pattern(
        self,
        condition: PatternCondition,
        buffer: List[Dict],
        current_event: Dict,
        user_id: str
    ) -> Optional[PatternMatch]:
        """Check threshold pattern (metric exceeds threshold)"""
        
        if not condition.condition_func or not condition.threshold:
            return None
        
        events = [item['event'] for item in buffer]
        
        try:
            # Use custom condition function to calculate metric
            if asyncio.iscoroutinefunction(condition.condition_func):
                metric_value = await condition.condition_func(events, current_event)
            else:
                metric_value = condition.condition_func(events, current_event)
            
            if metric_value >= condition.threshold:
                return PatternMatch(
                    pattern_name=condition.name,
                    user_id=user_id,
                    matched_events=events,
                    match_timestamp=datetime.utcnow(),
                    confidence_score=min(1.0, metric_value / condition.threshold),
                    metadata={'metric_value': metric_value, 'threshold': condition.threshold}
                )
                
        except Exception as e:
            logger.error(f"Error evaluating threshold condition: {e}")
        
        return None
    
    async def check_correlation_pattern(
        self,
        condition: PatternCondition,
        buffer: List[Dict],
        current_event: Dict,
        user_id: str
    ) -> Optional[PatternMatch]:
        """Check correlation pattern (events occur together)"""
        
        if len(condition.events) < 2:
            return None
        
        event_groups = defaultdict(list)
        
        # Group events by type
        for item in buffer:
            event = item['event']
            for pattern_event in condition.events:
                if self.event_matches_pattern(event, pattern_event):
                    event_groups[pattern_event].append(event)
        
        # Check if all event types are present
        if len(event_groups) == len(condition.events):
            # Calculate correlation strength
            min_count = min(len(events) for events in event_groups.values())
            correlation_strength = min_count / max(1, condition.threshold or 1)
            
            if correlation_strength >= 1.0:
                all_matched_events = []
                for events in event_groups.values():
                    all_matched_events.extend(events)
                
                return PatternMatch(
                    pattern_name=condition.name,
                    user_id=user_id,
                    matched_events=all_matched_events,
                    match_timestamp=datetime.utcnow(),
                    confidence_score=min(1.0, correlation_strength),
                    metadata={'correlation_events': len(event_groups)}
                )
        
        return None
    
    def event_matches_pattern(self, event: Dict, pattern: str) -> bool:
        """Check if event matches pattern (supports regex)"""
        
        event_type = event.get('event_type', '')
        
        # Try exact match first
        if event_type == pattern:
            return True
        
        # Try regex match
        try:
            return bool(re.match(pattern, event_type))
        except re.error:
            return False
    
    async def execute_pattern_action(
        self,
        pattern_name: str,
        pattern_info: Dict,
        match: PatternMatch
    ):
        """Execute action for matched pattern"""
        
        try:
            action = pattern_info['action']
            
            if asyncio.iscoroutinefunction(action):
                await action(pattern_name, match)
            else:
                await asyncio.get_event_loop().run_in_executor(
                    None, action, pattern_name, match
                )
            
            # Update statistics
            pattern_info['match_count'] += 1
            pattern_info['last_match'] = datetime.utcnow()
            self.stats['patterns_matched'] += 1
            
            # Store match for analysis
            self.pattern_matches.append(match)
            
            # Limit match history
            if len(self.pattern_matches) > 1000:
                self.pattern_matches = self.pattern_matches[-500:]
            
            logger.info(f"Pattern matched: {pattern_name} for user {match.user_id}")
            
        except Exception as e:
            logger.error(f"Error executing pattern action {pattern_name}: {e}")
    
    async def periodic_cleanup(self):
        """Periodic cleanup of old events"""
        
        while True:
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                await self.cleanup_old_events()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"CEP cleanup failed: {e}")
    
    async def cleanup_old_events(self):
        """Remove old events from buffers"""
        
        cutoff_time = datetime.utcnow() - self.buffer_ttl
        cleaned_users = 0
        
        for user_id, buffer in self.event_buffer.items():
            original_size = len(buffer)
            
            # Remove old events
            while buffer and buffer[0]['timestamp'] < cutoff_time:
                buffer.popleft()
            
            if len(buffer) < original_size:
                cleaned_users += 1
        
        if cleaned_users > 0:
            logger.info(f"Cleaned up old events for {cleaned_users} users")
    
    async def get_user_patterns(self, user_id: str) -> List[PatternMatch]:
        """Get recent pattern matches for user"""
        
        return [
            match for match in self.pattern_matches
            if match.user_id == user_id
        ]
    
    async def get_pattern_stats(self, pattern_name: str) -> Optional[Dict]:
        """Get statistics for specific pattern"""
        
        if pattern_name not in self.patterns:
            return None
        
        pattern_info = self.patterns[pattern_name]
        
        return {
            'name': pattern_name,
            'match_count': pattern_info['match_count'],
            'last_match': pattern_info['last_match'],
            'condition': {
                'type': pattern_info['condition'].pattern_type.value,
                'events': pattern_info['condition'].events,
                'timeframe': str(pattern_info['condition'].timeframe)
            }
        }
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get overall CEP statistics"""
        
        active_users = len(self.event_buffer)
        total_events_buffered = sum(len(buffer) for buffer in self.event_buffer.values())
        
        return {
            **self.stats,
            'active_users': active_users,
            'total_events_buffered': total_events_buffered,
            'recent_matches': len(self.pattern_matches),
            'buffer_size_limit': self.buffer_size,
            'buffer_ttl': str(self.buffer_ttl)
        }


# Pre-defined pattern conditions and actions
class PredefinedPatterns:
    """Collection of pre-defined behavioral patterns"""
    
    @staticmethod
    def rapid_error_pattern() -> PatternCondition:
        """Detect rapid error occurrence pattern"""
        
        return PatternCondition(
            name="rapid_errors",
            pattern_type=PatternType.FREQUENCY,
            events=["error"],
            timeframe=timedelta(minutes=5),
            threshold=5,
            metadata={"severity": "high", "intervention": "immediate"}
        )
    
    @staticmethod
    def purchase_abandonment_pattern() -> PatternCondition:
        """Detect purchase abandonment pattern"""
        
        return PatternCondition(
            name="purchase_abandonment",
            pattern_type=PatternType.SEQUENCE,
            events=["pricing_viewed", "add_to_cart"],
            timeframe=timedelta(minutes=15),
            metadata={"intervention": "retention", "urgency": "medium"}
        )
    
    @staticmethod
    def feature_discovery_pattern() -> PatternCondition:
        """Detect when user discovers new features"""
        
        def feature_discovery_condition(events: List[Dict], current_event: Dict) -> float:
            """Calculate feature discovery score"""
            
            if current_event.get('event_type') != 'feature_used':
                return 0.0
            
            feature_name = current_event.get('properties', {}).get('feature')
            if not feature_name:
                return 0.0
            
            # Check if this is first time using this feature
            for event in events[:-1]:  # Exclude current event
                if (event.get('event_type') == 'feature_used' and
                    event.get('properties', {}).get('feature') == feature_name):
                    return 0.0  # Not first time
            
            return 1.0  # First time using this feature
        
        return PatternCondition(
            name="feature_discovery",
            pattern_type=PatternType.THRESHOLD,
            events=["feature_used"],
            timeframe=timedelta(hours=1),
            threshold=1,
            condition_func=feature_discovery_condition,
            metadata={"intervention": "education", "timing": "immediate"}
        )
    
    @staticmethod
    def engagement_drop_pattern() -> PatternCondition:
        """Detect engagement drop pattern"""
        
        return PatternCondition(
            name="engagement_drop",
            pattern_type=PatternType.ABSENCE,
            events=["session_start"],
            timeframe=timedelta(days=3),
            metadata={"intervention": "re_engagement", "urgency": "medium"}
        )
    
    @staticmethod
    def power_user_pattern() -> PatternCondition:
        """Detect power user behavior"""
        
        def power_user_condition(events: List[Dict], current_event: Dict) -> float:
            """Calculate power user score"""
            
            feature_usage = set()
            session_count = 0
            
            for event in events:
                if event.get('event_type') == 'feature_used':
                    feature_name = event.get('properties', {}).get('feature')
                    if feature_name:
                        feature_usage.add(feature_name)
                elif event.get('event_type') == 'session_start':
                    session_count += 1
            
            # Score based on feature diversity and session frequency
            feature_score = min(1.0, len(feature_usage) / 10)  # 10+ features = max score
            session_score = min(1.0, session_count / 20)  # 20+ sessions = max score
            
            return (feature_score + session_score) / 2
        
        return PatternCondition(
            name="power_user",
            pattern_type=PatternType.THRESHOLD,
            events=["feature_used", "session_start"],
            timeframe=timedelta(days=7),
            threshold=0.7,
            condition_func=power_user_condition,
            metadata={"intervention": "monetization", "timing": "optimal"}
        )


# Example pattern actions
async def rapid_error_action(pattern_name: str, match: PatternMatch):
    """Action for rapid error pattern"""
    
    logger.warning(f"Rapid errors detected for user {match.user_id}")
    
    # This would trigger immediate support intervention
    # await send_support_notification(match.user_id, match.matched_events)
    # await create_support_ticket(match.user_id, "Rapid errors detected")

async def purchase_abandonment_action(pattern_name: str, match: PatternMatch):
    """Action for purchase abandonment pattern"""
    
    logger.info(f"Purchase abandonment detected for user {match.user_id}")
    
    # This would trigger retention campaign
    # await schedule_retention_message(match.user_id, "cart_abandonment")
    # await apply_discount_offer(match.user_id, percentage=10)

async def feature_discovery_action(pattern_name: str, match: PatternMatch):
    """Action for feature discovery pattern"""
    
    logger.info(f"Feature discovery detected for user {match.user_id}")
    
    # This would trigger educational content
    # await send_feature_tips(match.user_id, discovered_feature)
    # await track_feature_adoption(match.user_id, discovered_feature)

async def engagement_drop_action(pattern_name: str, match: PatternMatch):
    """Action for engagement drop pattern"""
    
    logger.info(f"Engagement drop detected for user {match.user_id}")
    
    # This would trigger re-engagement campaign
    # await schedule_re_engagement_message(match.user_id)
    # await create_personalized_content(match.user_id, "win_back")

async def power_user_action(pattern_name: str, match: PatternMatch):
    """Action for power user pattern"""
    
    logger.info(f"Power user behavior detected for user {match.user_id}")
    
    # This would trigger monetization opportunity
    # await schedule_upgrade_campaign(match.user_id)
    # await offer_premium_features(match.user_id) 