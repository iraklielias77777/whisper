"""
GraphQL API for User Whisperer Platform
Comprehensive GraphQL schema with queries, mutations, and subscriptions
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, AsyncGenerator
from datetime import datetime, timedelta
import json

# GraphQL imports
try:
    import strawberry
    from strawberry.fastapi import GraphQLRouter
    from strawberry.types import Info
    from strawberry.subscriptions import GRAPHQL_TRANSPORT_WS_PROTOCOL, GRAPHQL_WS_PROTOCOL
    STRAWBERRY_AVAILABLE = True
except ImportError:
    STRAWBERRY_AVAILABLE = False

# WebSocket support
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

logger = logging.getLogger(__name__)

if not STRAWBERRY_AVAILABLE:
    logger.warning("Strawberry GraphQL not available - GraphQL API disabled")

# GraphQL Types
if STRAWBERRY_AVAILABLE:
    @strawberry.type
    class User:
        id: str
        app_id: str
        external_user_id: str
        email: Optional[str] = None
        name: Optional[str] = None
        first_name: Optional[str] = None
        last_name: Optional[str] = None
        phone: Optional[str] = None
        avatar: Optional[str] = None
        lifecycle_stage: str
        engagement_score: float
        churn_risk_score: float
        ltv_prediction: Optional[float] = None
        created_at: datetime
        updated_at: datetime
        last_active_at: Optional[datetime] = None
        traits: strawberry.scalars.JSON
    
    @strawberry.type
    class Event:
        id: str
        app_id: str
        user_id: str
        session_id: str
        event_type: str
        properties: strawberry.scalars.JSON
        context: strawberry.scalars.JSON
        created_at: datetime
        processed_at: Optional[datetime] = None
        source: Optional[str] = None
    
    @strawberry.type
    class Message:
        id: str
        app_id: str
        user_id: str
        channel: str
        content: str
        template_id: Optional[str] = None
        status: str
        scheduled_at: Optional[datetime] = None
        sent_at: Optional[datetime] = None
        delivered_at: Optional[datetime] = None
        opened_at: Optional[datetime] = None
        clicked_at: Optional[datetime] = None
        error_message: Optional[str] = None
        metadata: strawberry.scalars.JSON
    
    @strawberry.type
    class Decision:
        id: str
        app_id: str
        user_id: str
        intervention_type: str
        should_intervene: bool
        confidence_score: float
        reasoning: Optional[str] = None
        context: strawberry.scalars.JSON
        created_at: datetime
        executed_at: Optional[datetime] = None
        result: Optional[str] = None
    
    @strawberry.type
    class Campaign:
        id: str
        app_id: str
        name: str
        description: Optional[str] = None
        type: str
        status: str
        target_criteria: strawberry.scalars.JSON
        content_template: strawberry.scalars.JSON
        schedule: strawberry.scalars.JSON
        created_at: datetime
        started_at: Optional[datetime] = None
        ended_at: Optional[datetime] = None
        metrics: strawberry.scalars.JSON
    
    @strawberry.type
    class Analytics:
        period_start: datetime
        period_end: datetime
        total_users: int
        active_users: int
        new_users: int
        returning_users: int
        churn_rate: float
        engagement_rate: float
        conversion_rate: float
        avg_session_duration: float
        total_events: int
        total_messages: int
        message_open_rate: float
        message_click_rate: float
        revenue: Optional[float] = None
        ltv_avg: Optional[float] = None
    
    @strawberry.type
    class Cohort:
        id: str
        app_id: str
        name: str
        description: Optional[str] = None
        criteria: strawberry.scalars.JSON
        user_count: int
        created_at: datetime
        updated_at: datetime
        metrics: strawberry.scalars.JSON
    
    @strawberry.type
    class Segment:
        id: str
        app_id: str
        name: str
        description: Optional[str] = None
        type: str
        criteria: strawberry.scalars.JSON
        user_count: int
        created_at: datetime
        updated_at: datetime
        last_computed_at: Optional[datetime] = None
    
    @strawberry.type
    class MLModel:
        id: str
        name: str
        type: str
        version: str
        status: str
        accuracy: Optional[float] = None
        precision: Optional[float] = None
        recall: Optional[float] = None
        f1_score: Optional[float] = None
        created_at: datetime
        trained_at: Optional[datetime] = None
        deployed_at: Optional[datetime] = None
        metadata: strawberry.scalars.JSON
    
    @strawberry.type
    class Prediction:
        id: str
        user_id: str
        model_id: str
        model_type: str
        prediction_type: str
        value: float
        confidence: float
        features: strawberry.scalars.JSON
        created_at: datetime
        expires_at: Optional[datetime] = None
    
    # Input Types
    @strawberry.input
    class UserFilter:
        app_id: Optional[str] = None
        lifecycle_stage: Optional[str] = None
        engagement_score_min: Optional[float] = None
        engagement_score_max: Optional[float] = None
        churn_risk_min: Optional[float] = None
        churn_risk_max: Optional[float] = None
        created_after: Optional[datetime] = None
        created_before: Optional[datetime] = None
        last_active_after: Optional[datetime] = None
        last_active_before: Optional[datetime] = None
    
    @strawberry.input
    class EventFilter:
        app_id: Optional[str] = None
        user_id: Optional[str] = None
        event_type: Optional[str] = None
        created_after: Optional[datetime] = None
        created_before: Optional[datetime] = None
        source: Optional[str] = None
    
    @strawberry.input
    class MessageFilter:
        app_id: Optional[str] = None
        user_id: Optional[str] = None
        channel: Optional[str] = None
        status: Optional[str] = None
        sent_after: Optional[datetime] = None
        sent_before: Optional[datetime] = None
    
    @strawberry.input
    class AnalyticsFilter:
        app_id: str
        period_start: datetime
        period_end: datetime
        segment_id: Optional[str] = None
        cohort_id: Optional[str] = None
    
    @strawberry.input
    class TrackEventInput:
        app_id: str
        user_id: str
        event_type: str
        properties: Optional[strawberry.scalars.JSON] = None
        context: Optional[strawberry.scalars.JSON] = None
        timestamp: Optional[datetime] = None
    
    @strawberry.input
    class IdentifyUserInput:
        app_id: str
        user_id: str
        traits: Optional[strawberry.scalars.JSON] = None
        timestamp: Optional[datetime] = None
    
    @strawberry.input
    class SendMessageInput:
        app_id: str
        user_id: str
        channel: str
        content: str
        template_id: Optional[str] = None
        schedule_at: Optional[datetime] = None
        metadata: Optional[strawberry.scalars.JSON] = None
    
    @strawberry.input
    class CreateCampaignInput:
        app_id: str
        name: str
        description: Optional[str] = None
        type: str
        target_criteria: strawberry.scalars.JSON
        content_template: strawberry.scalars.JSON
        schedule: strawberry.scalars.JSON
    
    @strawberry.input
    class CreateSegmentInput:
        app_id: str
        name: str
        description: Optional[str] = None
        type: str
        criteria: strawberry.scalars.JSON
    
    # Queries
    @strawberry.type
    class Query:
        @strawberry.field
        async def user(self, info: Info, id: str) -> Optional[User]:
            """Get user by ID"""
            
            db = info.context["db"]
            user_data = await db.fetch_user(id)
            
            if user_data:
                return User(**user_data)
            
            return None
        
        @strawberry.field
        async def users(
            self,
            info: Info,
            filter: Optional[UserFilter] = None,
            limit: int = 100,
            offset: int = 0
        ) -> List[User]:
            """Get users with filtering and pagination"""
            
            db = info.context["db"]
            
            # Build filter criteria
            filter_dict = {}
            if filter:
                filter_dict = strawberry.asdict(filter)
                # Remove None values
                filter_dict = {k: v for k, v in filter_dict.items() if v is not None}
            
            users_data = await db.fetch_users(
                filter_dict,
                limit=min(limit, 1000),  # Cap at 1000
                offset=offset
            )
            
            return [User(**u) for u in users_data]
        
        @strawberry.field
        async def events(
            self,
            info: Info,
            filter: Optional[EventFilter] = None,
            limit: int = 100,
            offset: int = 0
        ) -> List[Event]:
            """Get events with filtering and pagination"""
            
            db = info.context["db"]
            
            filter_dict = {}
            if filter:
                filter_dict = strawberry.asdict(filter)
                filter_dict = {k: v for k, v in filter_dict.items() if v is not None}
            
            events_data = await db.fetch_events(
                filter_dict,
                limit=min(limit, 1000),
                offset=offset
            )
            
            return [Event(**e) for e in events_data]
        
        @strawberry.field
        async def messages(
            self,
            info: Info,
            filter: Optional[MessageFilter] = None,
            limit: int = 100,
            offset: int = 0
        ) -> List[Message]:
            """Get messages with filtering and pagination"""
            
            db = info.context["db"]
            
            filter_dict = {}
            if filter:
                filter_dict = strawberry.asdict(filter)
                filter_dict = {k: v for k, v in filter_dict.items() if v is not None}
            
            messages_data = await db.fetch_messages(
                filter_dict,
                limit=min(limit, 1000),
                offset=offset
            )
            
            return [Message(**m) for m in messages_data]
        
        @strawberry.field
        async def decisions(
            self,
            info: Info,
            user_id: Optional[str] = None,
            app_id: Optional[str] = None,
            limit: int = 100,
            offset: int = 0
        ) -> List[Decision]:
            """Get decisions with filtering"""
            
            db = info.context["db"]
            
            filter_dict = {}
            if user_id:
                filter_dict['user_id'] = user_id
            if app_id:
                filter_dict['app_id'] = app_id
            
            decisions_data = await db.fetch_decisions(
                filter_dict,
                limit=min(limit, 1000),
                offset=offset
            )
            
            return [Decision(**d) for d in decisions_data]
        
        @strawberry.field
        async def campaigns(
            self,
            info: Info,
            app_id: str,
            status: Optional[str] = None,
            limit: int = 100,
            offset: int = 0
        ) -> List[Campaign]:
            """Get campaigns for an app"""
            
            db = info.context["db"]
            
            filter_dict = {'app_id': app_id}
            if status:
                filter_dict['status'] = status
            
            campaigns_data = await db.fetch_campaigns(
                filter_dict,
                limit=min(limit, 1000),
                offset=offset
            )
            
            return [Campaign(**c) for c in campaigns_data]
        
        @strawberry.field
        async def analytics(
            self,
            info: Info,
            filter: AnalyticsFilter
        ) -> Analytics:
            """Get analytics data for a time period"""
            
            db = info.context["db"]
            
            filter_dict = strawberry.asdict(filter)
            analytics_data = await db.fetch_analytics(filter_dict)
            
            return Analytics(**analytics_data)
        
        @strawberry.field
        async def cohorts(
            self,
            info: Info,
            app_id: str,
            limit: int = 100,
            offset: int = 0
        ) -> List[Cohort]:
            """Get cohorts for an app"""
            
            db = info.context["db"]
            
            cohorts_data = await db.fetch_cohorts(
                app_id,
                limit=min(limit, 1000),
                offset=offset
            )
            
            return [Cohort(**c) for c in cohorts_data]
        
        @strawberry.field
        async def segments(
            self,
            info: Info,
            app_id: str,
            limit: int = 100,
            offset: int = 0
        ) -> List[Segment]:
            """Get segments for an app"""
            
            db = info.context["db"]
            
            segments_data = await db.fetch_segments(
                app_id,
                limit=min(limit, 1000),
                offset=offset
            )
            
            return [Segment(**s) for s in segments_data]
        
        @strawberry.field
        async def ml_models(
            self,
            info: Info,
            type: Optional[str] = None,
            status: Optional[str] = None,
            limit: int = 100
        ) -> List[MLModel]:
            """Get ML models"""
            
            db = info.context["db"]
            
            filter_dict = {}
            if type:
                filter_dict['type'] = type
            if status:
                filter_dict['status'] = status
            
            models_data = await db.fetch_ml_models(
                filter_dict,
                limit=min(limit, 1000)
            )
            
            return [MLModel(**m) for m in models_data]
        
        @strawberry.field
        async def predictions(
            self,
            info: Info,
            user_id: str,
            model_type: Optional[str] = None,
            limit: int = 100
        ) -> List[Prediction]:
            """Get predictions for a user"""
            
            db = info.context["db"]
            
            filter_dict = {'user_id': user_id}
            if model_type:
                filter_dict['model_type'] = model_type
            
            predictions_data = await db.fetch_predictions(
                filter_dict,
                limit=min(limit, 1000)
            )
            
            return [Prediction(**p) for p in predictions_data]
    
    # Mutations
    @strawberry.type
    class Mutation:
        @strawberry.mutation
        async def track_event(
            self,
            info: Info,
            input: TrackEventInput
        ) -> Event:
            """Track a user event"""
            
            db = info.context["db"]
            event_queue = info.context["event_queue"]
            
            # Convert input to dict
            event_data = strawberry.asdict(input)
            
            # Add metadata
            event_data.update({
                'id': f"gql_{int(datetime.now().timestamp() * 1000)}",
                'session_id': f"gql_session_{event_data['user_id']}",
                'created_at': datetime.utcnow(),
                'source': 'graphql'
            })
            
            # Queue event for processing
            await event_queue.put(event_data)
            
            # Store event
            event_id = await db.store_event(event_data)
            event_data['id'] = event_id
            
            return Event(**event_data)
        
        @strawberry.mutation
        async def identify_user(
            self,
            info: Info,
            input: IdentifyUserInput
        ) -> User:
            """Identify and update user information"""
            
            db = info.context["db"]
            
            # Convert input to dict
            user_data = strawberry.asdict(input)
            
            # Update or create user
            user = await db.upsert_user(user_data)
            
            return User(**user)
        
        @strawberry.mutation
        async def send_message(
            self,
            info: Info,
            input: SendMessageInput
        ) -> Message:
            """Send a message to a user"""
            
            db = info.context["db"]
            message_queue = info.context["message_queue"]
            
            # Convert input to dict
            message_data = strawberry.asdict(input)
            
            # Add metadata
            message_data.update({
                'id': f"gql_msg_{int(datetime.now().timestamp() * 1000)}",
                'status': 'queued' if message_data.get('schedule_at') else 'pending',
                'created_at': datetime.utcnow()
            })
            
            # Queue message for processing
            await message_queue.put(message_data)
            
            # Store message
            message_id = await db.store_message(message_data)
            message_data['id'] = message_id
            
            return Message(**message_data)
        
        @strawberry.mutation
        async def create_campaign(
            self,
            info: Info,
            input: CreateCampaignInput
        ) -> Campaign:
            """Create a new campaign"""
            
            db = info.context["db"]
            
            # Convert input to dict
            campaign_data = strawberry.asdict(input)
            
            # Add metadata
            campaign_data.update({
                'id': f"camp_{int(datetime.now().timestamp() * 1000)}",
                'status': 'draft',
                'created_at': datetime.utcnow(),
                'metrics': {}
            })
            
            # Store campaign
            campaign_id = await db.store_campaign(campaign_data)
            campaign_data['id'] = campaign_id
            
            return Campaign(**campaign_data)
        
        @strawberry.mutation
        async def update_campaign(
            self,
            info: Info,
            id: str,
            status: Optional[str] = None,
            name: Optional[str] = None
        ) -> Campaign:
            """Update a campaign"""
            
            db = info.context["db"]
            
            # Build update data
            update_data = {}
            if status:
                update_data['status'] = status
                if status == 'active':
                    update_data['started_at'] = datetime.utcnow()
                elif status == 'completed':
                    update_data['ended_at'] = datetime.utcnow()
            
            if name:
                update_data['name'] = name
            
            # Update campaign
            campaign = await db.update_campaign(id, update_data)
            
            return Campaign(**campaign)
        
        @strawberry.mutation
        async def create_segment(
            self,
            info: Info,
            input: CreateSegmentInput
        ) -> Segment:
            """Create a new user segment"""
            
            db = info.context["db"]
            
            # Convert input to dict
            segment_data = strawberry.asdict(input)
            
            # Add metadata
            segment_data.update({
                'id': f"seg_{int(datetime.now().timestamp() * 1000)}",
                'user_count': 0,  # Will be computed asynchronously
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow()
            })
            
            # Store segment
            segment_id = await db.store_segment(segment_data)
            segment_data['id'] = segment_id
            
            # Trigger segment computation
            segment_queue = info.context.get("segment_queue")
            if segment_queue:
                await segment_queue.put({
                    'action': 'compute_segment',
                    'segment_id': segment_id
                })
            
            return Segment(**segment_data)
        
        @strawberry.mutation
        async def trigger_decision(
            self,
            info: Info,
            user_id: str,
            intervention_type: Optional[str] = None
        ) -> Decision:
            """Trigger an intervention decision for a user"""
            
            db = info.context["db"]
            decision_queue = info.context["decision_queue"]
            
            # Create decision request
            decision_data = {
                'id': f"dec_{int(datetime.now().timestamp() * 1000)}",
                'user_id': user_id,
                'intervention_type': intervention_type or 'automatic',
                'created_at': datetime.utcnow(),
                'context': {'triggered_via': 'graphql'}
            }
            
            # Queue decision for processing
            await decision_queue.put(decision_data)
            
            # Store decision
            decision_id = await db.store_decision(decision_data)
            decision_data['id'] = decision_id
            
            return Decision(**decision_data)
    
    # Subscriptions
    @strawberry.type
    class Subscription:
        @strawberry.subscription
        async def user_events(
            self,
            info: Info,
            user_id: str
        ) -> AsyncGenerator[Event, None]:
            """Subscribe to real-time events for a user"""
            
            event_stream = info.context["event_stream"]
            
            # Subscribe to user's event stream
            async for event_data in event_stream.subscribe(f"user:{user_id}"):
                yield Event(**event_data)
        
        @strawberry.subscription
        async def app_events(
            self,
            info: Info,
            app_id: str,
            event_types: Optional[List[str]] = None
        ) -> AsyncGenerator[Event, None]:
            """Subscribe to real-time events for an app"""
            
            event_stream = info.context["event_stream"]
            
            # Subscribe to app's event stream
            async for event_data in event_stream.subscribe(f"app:{app_id}"):
                # Filter by event types if specified
                if event_types and event_data.get('event_type') not in event_types:
                    continue
                
                yield Event(**event_data)
        
        @strawberry.subscription
        async def decisions(
            self,
            info: Info,
            user_id: Optional[str] = None,
            app_id: Optional[str] = None
        ) -> AsyncGenerator[Decision, None]:
            """Subscribe to real-time decisions"""
            
            decision_stream = info.context["decision_stream"]
            
            # Determine subscription key
            if user_id:
                subscription_key = f"user:{user_id}:decisions"
            elif app_id:
                subscription_key = f"app:{app_id}:decisions"
            else:
                subscription_key = "decisions"
            
            async for decision_data in decision_stream.subscribe(subscription_key):
                yield Decision(**decision_data)
        
        @strawberry.subscription
        async def messages(
            self,
            info: Info,
            user_id: str
        ) -> AsyncGenerator[Message, None]:
            """Subscribe to real-time messages for a user"""
            
            message_stream = info.context["message_stream"]
            
            async for message_data in message_stream.subscribe(f"user:{user_id}:messages"):
                yield Message(**message_data)
        
        @strawberry.subscription
        async def campaign_updates(
            self,
            info: Info,
            campaign_id: str
        ) -> AsyncGenerator[Campaign, None]:
            """Subscribe to campaign status updates"""
            
            campaign_stream = info.context["campaign_stream"]
            
            async for campaign_data in campaign_stream.subscribe(f"campaign:{campaign_id}"):
                yield Campaign(**campaign_data)
        
        @strawberry.subscription
        async def analytics_updates(
            self,
            info: Info,
            app_id: str,
            interval_minutes: int = 5
        ) -> AsyncGenerator[Analytics, None]:
            """Subscribe to real-time analytics updates"""
            
            analytics_stream = info.context["analytics_stream"]
            
            # Subscribe to analytics updates
            async for analytics_data in analytics_stream.subscribe(
                f"app:{app_id}:analytics",
                interval=interval_minutes * 60
            ):
                yield Analytics(**analytics_data)
    
    # Create GraphQL schema
    schema = strawberry.Schema(
        query=Query,
        mutation=Mutation,
        subscription=Subscription
    )


# Context providers
class DatabaseContext:
    """Mock database context for GraphQL operations"""
    
    async def fetch_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Fetch user by ID"""
        # This would implement actual database query
        return {
            'id': user_id,
            'app_id': 'app_123',
            'external_user_id': f'user_{user_id}',
            'email': f'user{user_id}@example.com',
            'name': f'User {user_id}',
            'lifecycle_stage': 'engaged',
            'engagement_score': 0.75,
            'churn_risk_score': 0.1,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow(),
            'traits': {'source': 'graphql'}
        }
    
    async def fetch_users(
        self,
        filter_dict: Dict[str, Any],
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Fetch users with filtering"""
        # Mock implementation
        users = []
        for i in range(min(limit, 10)):  # Return up to 10 mock users
            users.append({
                'id': f'user_{offset + i}',
                'app_id': filter_dict.get('app_id', 'app_123'),
                'external_user_id': f'user_{offset + i}',
                'email': f'user{offset + i}@example.com',
                'lifecycle_stage': 'engaged',
                'engagement_score': 0.75,
                'churn_risk_score': 0.1,
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow(),
                'traits': {'source': 'graphql'}
            })
        return users
    
    async def fetch_events(
        self,
        filter_dict: Dict[str, Any],
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Fetch events with filtering"""
        # Mock implementation
        events = []
        for i in range(min(limit, 10)):
            events.append({
                'id': f'event_{offset + i}',
                'app_id': filter_dict.get('app_id', 'app_123'),
                'user_id': filter_dict.get('user_id', 'user_123'),
                'session_id': 'session_123',
                'event_type': 'page_viewed',
                'properties': {'page': '/dashboard'},
                'context': {'source': 'web'},
                'created_at': datetime.utcnow(),
                'source': 'graphql'
            })
        return events
    
    async def fetch_messages(
        self,
        filter_dict: Dict[str, Any],
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Fetch messages with filtering"""
        # Mock implementation
        return []
    
    async def fetch_decisions(
        self,
        filter_dict: Dict[str, Any],
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Fetch decisions with filtering"""
        # Mock implementation
        return []
    
    async def fetch_campaigns(
        self,
        filter_dict: Dict[str, Any],
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Fetch campaigns"""
        # Mock implementation
        return []
    
    async def fetch_analytics(self, filter_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch analytics data"""
        # Mock implementation
        return {
            'period_start': filter_dict['period_start'],
            'period_end': filter_dict['period_end'],
            'total_users': 1000,
            'active_users': 750,
            'new_users': 50,
            'returning_users': 700,
            'churn_rate': 0.05,
            'engagement_rate': 0.75,
            'conversion_rate': 0.15,
            'avg_session_duration': 300.0,
            'total_events': 10000,
            'total_messages': 500,
            'message_open_rate': 0.80,
            'message_click_rate': 0.25
        }
    
    async def store_event(self, event_data: Dict[str, Any]) -> str:
        """Store event and return ID"""
        return event_data.get('id', 'new_event_id')
    
    async def upsert_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Upsert user and return updated data"""
        return {
            'id': user_data.get('user_id', 'new_user_id'),
            'app_id': user_data.get('app_id'),
            'external_user_id': user_data.get('user_id'),
            'traits': user_data.get('traits', {}),
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow(),
            'lifecycle_stage': 'new',
            'engagement_score': 0.0,
            'churn_risk_score': 0.0
        }


class MockEventStream:
    """Mock event stream for subscriptions"""
    
    async def subscribe(self, channel: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Subscribe to event stream"""
        
        # Mock implementation - emit periodic events
        counter = 0
        while True:
            await asyncio.sleep(5)  # Emit every 5 seconds
            
            yield {
                'id': f'stream_event_{counter}',
                'app_id': 'app_123',
                'user_id': 'user_123',
                'session_id': 'session_123',
                'event_type': 'mock_event',
                'properties': {'counter': counter},
                'context': {'source': 'stream'},
                'created_at': datetime.utcnow()
            }
            
            counter += 1


def create_graphql_router(context: Dict[str, Any]) -> Optional[GraphQLRouter]:
    """
    Create GraphQL router with context
    """
    
    if not STRAWBERRY_AVAILABLE:
        logger.error("Strawberry GraphQL not available")
        return None
    
    # Provide default context if not provided
    default_context = {
        "db": DatabaseContext(),
        "event_queue": asyncio.Queue(),
        "message_queue": asyncio.Queue(),
        "decision_queue": asyncio.Queue(),
        "segment_queue": asyncio.Queue(),
        "event_stream": MockEventStream(),
        "decision_stream": MockEventStream(),
        "message_stream": MockEventStream(),
        "campaign_stream": MockEventStream(),
        "analytics_stream": MockEventStream()
    }
    
    # Merge provided context with defaults
    merged_context = {**default_context, **context}
    
    return GraphQLRouter(
        schema,
        context_getter=lambda: merged_context,
        subscription_protocols=[
            GRAPHQL_TRANSPORT_WS_PROTOCOL,
            GRAPHQL_WS_PROTOCOL,
        ] if WEBSOCKETS_AVAILABLE else []
    )


# Export schema and types for external use
if STRAWBERRY_AVAILABLE:
    __all__ = [
        'schema',
        'Query',
        'Mutation', 
        'Subscription',
        'User',
        'Event',
        'Message',
        'Decision',
        'Campaign',
        'Analytics',
        'create_graphql_router',
        'DatabaseContext',
        'MockEventStream'
    ]
else:
    __all__ = ['create_graphql_router']
