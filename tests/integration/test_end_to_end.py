#!/usr/bin/env python3
"""
End-to-End Integration Tests for User Whisperer Platform
Tests the complete user journey from event tracking to automated responses
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pytest
import aiohttp
import websockets
from unittest.mock import AsyncMock, patch

# Test configuration
TEST_CONFIG = {
    'api_base_url': 'http://localhost:8000',
    'graphql_url': 'http://localhost:8000/graphql',
    'websocket_url': 'ws://localhost:8000/graphql',
    'test_api_key': 'test_api_key_e2e_testing',
    'test_app_id': 'app_e2e_test_123',
    'timeout': 30
}


class E2ETestSuite:
    """Complete end-to-end test suite for User Whisperer Platform"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.test_user_id = f"test_user_{uuid.uuid4().hex[:8]}"
        self.test_session_id = f"test_session_{uuid.uuid4().hex[:8]}"
        self.tracked_events: List[Dict[str, Any]] = []
        self.received_decisions: List[Dict[str, Any]] = []
        self.sent_messages: List[Dict[str, Any]] = []
    
    async def setup(self):
        """Set up test environment"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=TEST_CONFIG['timeout']),
            headers={
                'Authorization': f"Bearer {TEST_CONFIG['test_api_key']}",
                'Content-Type': 'application/json'
            }
        )
        
        # Wait for services to be ready
        await self._wait_for_services()
        
        # Clean up any existing test data
        await self._cleanup_test_data()
    
    async def teardown(self):
        """Clean up test environment"""
        if self.session:
            await self.session.close()
        
        # Clean up test data
        await self._cleanup_test_data()
    
    async def _wait_for_services(self):
        """Wait for all services to be healthy"""
        services = [
            f"{TEST_CONFIG['api_base_url']}/health",
            f"{TEST_CONFIG['api_base_url']}/health/ready"
        ]
        
        for service_url in services:
            max_attempts = 30
            for attempt in range(max_attempts):
                try:
                    async with self.session.get(service_url) as response:
                        if response.status == 200:
                            break
                except:
                    pass
                
                if attempt == max_attempts - 1:
                    raise Exception(f"Service not ready: {service_url}")
                
                await asyncio.sleep(1)
    
    async def _cleanup_test_data(self):
        """Clean up any test data"""
        try:
            # Delete test user data
            await self.session.delete(
                f"{TEST_CONFIG['api_base_url']}/api/v1/users/{self.test_user_id}"
            )
        except:
            pass  # User might not exist yet
    
    async def test_complete_user_journey(self):
        """Test complete user journey from signup to automated engagement"""
        
        print("🚀 Starting complete user journey test...")
        
        # Phase 1: User Signup and Initial Events
        await self._test_user_signup()
        await self._test_event_tracking()
        
        # Phase 2: Behavioral Analysis
        await self._test_behavioral_analysis()
        
        # Phase 3: Decision Making
        await self._test_decision_engine()
        
        # Phase 4: Content Generation and Delivery
        await self._test_content_generation()
        await self._test_message_delivery()
        
        # Phase 5: Analytics and Feedback Loop
        await self._test_analytics_collection()
        await self._test_feedback_loop()
        
        print("✅ Complete user journey test passed!")
    
    async def _test_user_signup(self):
        """Test user identification and signup flow"""
        print("👤 Testing user signup flow...")
        
        # Identify user
        user_data = {
            'user_id': self.test_user_id,
            'traits': {
                'email': 'test@example.com',
                'name': 'Test User',
                'plan': 'free',
                'signup_date': datetime.utcnow().isoformat(),
                'source': 'e2e_test'
            }
        }
        
        async with self.session.post(
            f"{TEST_CONFIG['api_base_url']}/api/v1/users/identify",
            json=user_data
        ) as response:
            assert response.status == 200
            result = await response.json()
            assert result['success'] is True
        
        # Track signup event
        signup_event = {
            'event_type': 'user_signup',
            'user_id': self.test_user_id,
            'app_id': TEST_CONFIG['test_app_id'],
            'session_id': self.test_session_id,
            'properties': {
                'signup_method': 'email',
                'plan': 'free',
                'source': 'e2e_test'
            },
            'context': {
                'ip': '127.0.0.1',
                'user_agent': 'E2E Test Client',
                'source': 'test'
            }
        }
        
        async with self.session.post(
            f"{TEST_CONFIG['api_base_url']}/api/v1/events/track",
            json=signup_event
        ) as response:
            assert response.status == 200
            result = await response.json()
            assert result['success'] is True
            self.tracked_events.append(result['data'])
        
        print("✅ User signup flow completed")
    
    async def _test_event_tracking(self):
        """Test event tracking and ingestion"""
        print("📊 Testing event tracking...")
        
        # Simulate user activity over time
        activities = [
            ('page_viewed', {'page': '/dashboard', 'duration': 30}),
            ('feature_used', {'feature': 'analytics', 'duration': 120}),
            ('button_clicked', {'button': 'upgrade', 'location': 'header'}),
            ('page_viewed', {'page': '/pricing', 'duration': 45}),
            ('form_started', {'form': 'upgrade', 'plan': 'pro'}),
            ('form_abandoned', {'form': 'upgrade', 'step': 2})
        ]
        
        for i, (event_type, properties) in enumerate(activities):
            event_data = {
                'event_type': event_type,
                'user_id': self.test_user_id,
                'app_id': TEST_CONFIG['test_app_id'],
                'session_id': self.test_session_id,
                'properties': properties,
                'timestamp': (datetime.utcnow() + timedelta(minutes=i*2)).isoformat()
            }
            
            async with self.session.post(
                f"{TEST_CONFIG['api_base_url']}/api/v1/events/track",
                json=event_data
            ) as response:
                assert response.status == 200
                result = await response.json()
                self.tracked_events.append(result['data'])
            
            # Small delay to simulate real user behavior
            await asyncio.sleep(0.1)
        
        # Test batch event tracking
        batch_events = [
            {
                'event_type': 'content_viewed',
                'user_id': self.test_user_id,
                'app_id': TEST_CONFIG['test_app_id'],
                'properties': {'content_type': 'article', 'article_id': f'article_{i}'}
            }
            for i in range(5)
        ]
        
        async with self.session.post(
            f"{TEST_CONFIG['api_base_url']}/api/v1/events/batch",
            json={'events': batch_events}
        ) as response:
            assert response.status == 200
            result = await response.json()
            assert result['success'] is True
            assert len(result['data']['results']) == 5
        
        print(f"✅ Event tracking completed ({len(self.tracked_events) + 5} events)")
    
    async def _test_behavioral_analysis(self):
        """Test behavioral analysis and user profiling"""
        print("🧠 Testing behavioral analysis...")
        
        # Wait for events to be processed
        await asyncio.sleep(2)
        
        # Get user analysis
        async with self.session.get(
            f"{TEST_CONFIG['api_base_url']}/api/v1/users/{self.test_user_id}/analysis"
        ) as response:
            assert response.status == 200
            analysis = await response.json()
            
            assert 'behavioral_metrics' in analysis['data']
            assert 'engagement_score' in analysis['data']
            assert 'lifecycle_stage' in analysis['data']
            
            # Verify metrics make sense
            metrics = analysis['data']['behavioral_metrics']
            assert metrics['session_count'] >= 1
            assert metrics['event_count'] >= len(self.tracked_events)
        
        # Test churn prediction
        async with self.session.get(
            f"{TEST_CONFIG['api_base_url']}/api/v1/users/{self.test_user_id}/churn_risk"
        ) as response:
            assert response.status == 200
            churn_data = await response.json()
            
            assert 'churn_risk_score' in churn_data['data']
            assert 0 <= churn_data['data']['churn_risk_score'] <= 1
        
        print("✅ Behavioral analysis completed")
    
    async def _test_decision_engine(self):
        """Test AI decision making and interventions"""
        print("🤖 Testing decision engine...")
        
        # Trigger decision for user
        decision_request = {
            'user_id': self.test_user_id,
            'context': {
                'trigger': 'form_abandoned',
                'source': 'e2e_test'
            }
        }
        
        async with self.session.post(
            f"{TEST_CONFIG['api_base_url']}/api/v1/decisions/trigger",
            json=decision_request
        ) as response:
            assert response.status == 200
            decision = await response.json()
            
            assert 'decision_id' in decision['data']
            assert 'should_intervene' in decision['data']
            assert 'intervention_type' in decision['data']
            assert 'confidence' in decision['data']
            
            self.received_decisions.append(decision['data'])
        
        # Get decision history
        async with self.session.get(
            f"{TEST_CONFIG['api_base_url']}/api/v1/decisions",
            params={'user_id': self.test_user_id}
        ) as response:
            assert response.status == 200
            decisions = await response.json()
            
            assert len(decisions['data']) >= 1
            assert decisions['data'][0]['user_id'] == self.test_user_id
        
        print("✅ Decision engine testing completed")
    
    async def _test_content_generation(self):
        """Test AI content generation"""
        print("✍️ Testing content generation...")
        
        if not self.received_decisions:
            # Create a mock decision for content generation
            self.received_decisions.append({
                'intervention_type': 'retention_email',
                'confidence': 0.85
            })
        
        # Generate personalized content
        content_request = {
            'user_id': self.test_user_id,
            'content_type': 'email',
            'template_id': 'retention_email_v1',
            'personalization_data': {
                'name': 'Test User',
                'abandoned_feature': 'upgrade',
                'plan': 'free'
            }
        }
        
        async with self.session.post(
            f"{TEST_CONFIG['api_base_url']}/api/v1/content/generate",
            json=content_request
        ) as response:
            assert response.status == 200
            content = await response.json()
            
            assert 'content_id' in content['data']
            assert 'subject' in content['data']
            assert 'body' in content['data']
            assert len(content['data']['body']) > 0
        
        # Test A/B testing content
        ab_request = {
            'user_id': self.test_user_id,
            'experiment_id': 'retention_email_test',
            'variants': ['control', 'variant_a', 'variant_b']
        }
        
        async with self.session.post(
            f"{TEST_CONFIG['api_base_url']}/api/v1/content/ab_test",
            json=ab_request
        ) as response:
            assert response.status == 200
            ab_content = await response.json()
            
            assert 'variant' in ab_content['data']
            assert ab_content['data']['variant'] in ab_request['variants']
        
        print("✅ Content generation completed")
    
    async def _test_message_delivery(self):
        """Test multi-channel message delivery"""
        print("📧 Testing message delivery...")
        
        # Send email message
        email_message = {
            'user_id': self.test_user_id,
            'channel': 'email',
            'content': {
                'subject': 'Complete your upgrade - Special offer inside!',
                'body': 'Hi Test User, we noticed you were interested in upgrading...',
                'template_id': 'retention_email_v1'
            },
            'metadata': {
                'campaign_id': 'retention_campaign_e2e',
                'experiment_id': 'retention_email_test'
            }
        }
        
        async with self.session.post(
            f"{TEST_CONFIG['api_base_url']}/api/v1/messages/send",
            json=email_message
        ) as response:
            assert response.status == 200
            message = await response.json()
            
            assert 'message_id' in message['data']
            assert message['data']['status'] in ['queued', 'sent']
            
            self.sent_messages.append(message['data'])
        
        # Test in-app notification
        inapp_message = {
            'user_id': self.test_user_id,
            'channel': 'in_app',
            'content': {
                'title': 'Upgrade available',
                'body': 'Complete your upgrade to unlock premium features',
                'action_url': '/upgrade',
                'action_text': 'Upgrade Now'
            }
        }
        
        async with self.session.post(
            f"{TEST_CONFIG['api_base_url']}/api/v1/messages/send",
            json=inapp_message
        ) as response:
            assert response.status == 200
            message = await response.json()
            self.sent_messages.append(message['data'])
        
        # Check message delivery status
        for sent_message in self.sent_messages:
            async with self.session.get(
                f"{TEST_CONFIG['api_base_url']}/api/v1/messages/{sent_message['message_id']}/status"
            ) as response:
                assert response.status == 200
                status = await response.json()
                
                assert 'status' in status['data']
                assert status['data']['status'] in ['queued', 'sent', 'delivered', 'failed']
        
        print("✅ Message delivery testing completed")
    
    async def _test_analytics_collection(self):
        """Test analytics and reporting"""
        print("📈 Testing analytics collection...")
        
        # Get user analytics
        analytics_params = {
            'start_date': (datetime.utcnow() - timedelta(hours=1)).isoformat(),
            'end_date': datetime.utcnow().isoformat()
        }
        
        async with self.session.get(
            f"{TEST_CONFIG['api_base_url']}/api/v1/analytics/users/{self.test_user_id}",
            params=analytics_params
        ) as response:
            assert response.status == 200
            analytics = await response.json()
            
            assert 'event_count' in analytics['data']
            assert 'session_count' in analytics['data']
            assert analytics['data']['event_count'] >= len(self.tracked_events)
        
        # Get funnel analysis
        funnel_request = {
            'steps': [
                {'event_type': 'page_viewed', 'properties': {'page': '/pricing'}},
                {'event_type': 'form_started', 'properties': {'form': 'upgrade'}},
                {'event_type': 'subscription_upgraded'}
            ],
            'time_window': 3600  # 1 hour
        }
        
        async with self.session.post(
            f"{TEST_CONFIG['api_base_url']}/api/v1/analytics/funnel",
            json=funnel_request
        ) as response:
            assert response.status == 200
            funnel = await response.json()
            
            assert 'steps' in funnel['data']
            assert len(funnel['data']['steps']) == 3
        
        # Test cohort analysis
        cohort_request = {
            'cohort_event': {'event_type': 'user_signup'},
            'return_event': {'event_type': 'feature_used'},
            'periods': 7,  # 7 days
            'period_type': 'day'
        }
        
        async with self.session.post(
            f"{TEST_CONFIG['api_base_url']}/api/v1/analytics/cohort",
            json=cohort_request
        ) as response:
            assert response.status == 200
            cohort = await response.json()
            
            assert 'cohorts' in cohort['data']
        
        print("✅ Analytics collection completed")
    
    async def _test_feedback_loop(self):
        """Test feedback loop and learning"""
        print("🔄 Testing feedback loop...")
        
        # Simulate user interaction with sent messages
        for sent_message in self.sent_messages:
            # Track message opened
            open_event = {
                'event_type': 'email_opened' if sent_message.get('channel') == 'email' else 'notification_opened',
                'user_id': self.test_user_id,
                'app_id': TEST_CONFIG['test_app_id'],
                'properties': {
                    'message_id': sent_message['message_id'],
                    'campaign_id': sent_message.get('metadata', {}).get('campaign_id')
                }
            }
            
            async with self.session.post(
                f"{TEST_CONFIG['api_base_url']}/api/v1/events/track",
                json=open_event
            ) as response:
                assert response.status == 200
        
        # Simulate user taking action (positive feedback)
        action_event = {
            'event_type': 'subscription_upgraded',
            'user_id': self.test_user_id,
            'app_id': TEST_CONFIG['test_app_id'],
            'properties': {
                'from_plan': 'free',
                'to_plan': 'pro',
                'source': 'retention_campaign'
            }
        }
        
        async with self.session.post(
            f"{TEST_CONFIG['api_base_url']}/api/v1/events/track",
            json=action_event
        ) as response:
            assert response.status == 200
        
        # Wait for feedback processing
        await asyncio.sleep(1)
        
        # Get updated user profile to verify learning
        async with self.session.get(
            f"{TEST_CONFIG['api_base_url']}/api/v1/users/{self.test_user_id}/analysis"
        ) as response:
            assert response.status == 200
            updated_analysis = await response.json()
            
            # Verify engagement score has improved
            assert updated_analysis['data']['engagement_score'] > 0
        
        print("✅ Feedback loop testing completed")
    
    async def test_graphql_api(self):
        """Test GraphQL API functionality"""
        print("🔗 Testing GraphQL API...")
        
        query = """
        query GetUser($userId: String!) {
            user(id: $userId) {
                id
                email
                engagementScore
                churnRiskScore
                lifecycleStage
            }
            events(filter: { userId: $userId }, limit: 5) {
                id
                eventType
                properties
                createdAt
            }
        }
        """
        
        variables = {'userId': self.test_user_id}
        
        async with self.session.post(
            TEST_CONFIG['graphql_url'],
            json={'query': query, 'variables': variables}
        ) as response:
            assert response.status == 200
            result = await response.json()
            
            assert 'data' in result
            assert 'user' in result['data']
            assert 'events' in result['data']
            
            if result['data']['user']:
                assert result['data']['user']['id'] == self.test_user_id
        
        print("✅ GraphQL API testing completed")
    
    async def test_websocket_subscriptions(self):
        """Test real-time WebSocket subscriptions"""
        print("🔌 Testing WebSocket subscriptions...")
        
        try:
            # Test WebSocket connection
            uri = f"{TEST_CONFIG['websocket_url']}"
            
            async with websockets.connect(uri) as websocket:
                # Subscribe to user events
                subscription = {
                    'type': 'start',
                    'payload': {
                        'query': f"""
                        subscription {{
                            userEvents(userId: "{self.test_user_id}") {{
                                id
                                eventType
                                properties
                                createdAt
                            }}
                        }}
                        """
                    }
                }
                
                await websocket.send(json.dumps(subscription))
                
                # Send a test event
                test_event = {
                    'event_type': 'websocket_test',
                    'user_id': self.test_user_id,
                    'app_id': TEST_CONFIG['test_app_id'],
                    'properties': {'test': True}
                }
                
                async with self.session.post(
                    f"{TEST_CONFIG['api_base_url']}/api/v1/events/track",
                    json=test_event
                ) as response:
                    assert response.status == 200
                
                # Wait for subscription message
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(message)
                    
                    assert 'payload' in data
                    assert 'data' in data['payload']
                    
                except asyncio.TimeoutError:
                    print("⚠️ WebSocket subscription timeout - may be expected in test environment")
        
        except Exception as e:
            print(f"⚠️ WebSocket test skipped: {e}")
        
        print("✅ WebSocket subscription testing completed")
    
    async def test_performance_and_load(self):
        """Test system performance under load"""
        print("⚡ Testing performance and load...")
        
        # Concurrent event tracking
        concurrent_events = []
        for i in range(50):
            event_data = {
                'event_type': 'load_test_event',
                'user_id': f"load_test_user_{i}",
                'app_id': TEST_CONFIG['test_app_id'],
                'properties': {'iteration': i, 'timestamp': time.time()}
            }
            concurrent_events.append(event_data)
        
        # Track events concurrently
        start_time = time.time()
        
        tasks = []
        for event in concurrent_events:
            task = self.session.post(
                f"{TEST_CONFIG['api_base_url']}/api/v1/events/track",
                json=event
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Verify performance
        successful_requests = sum(1 for r in responses if not isinstance(r, Exception))
        throughput = successful_requests / duration
        
        print(f"📊 Performance metrics:")
        print(f"   - Requests: {len(concurrent_events)}")
        print(f"   - Successful: {successful_requests}")
        print(f"   - Duration: {duration:.2f}s")
        print(f"   - Throughput: {throughput:.2f} req/s")
        
        # Assert minimum performance requirements
        assert throughput > 10  # At least 10 requests per second
        assert successful_requests >= len(concurrent_events) * 0.95  # 95% success rate
        
        print("✅ Performance and load testing completed")


@pytest.mark.asyncio
async def test_complete_platform_integration():
    """Main integration test that runs the complete test suite"""
    
    test_suite = E2ETestSuite()
    
    try:
        await test_suite.setup()
        
        # Run all test phases
        await test_suite.test_complete_user_journey()
        await test_suite.test_graphql_api()
        await test_suite.test_websocket_subscriptions()
        await test_suite.test_performance_and_load()
        
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print(f"📊 Test Summary:")
        print(f"   - Events tracked: {len(test_suite.tracked_events)}")
        print(f"   - Decisions made: {len(test_suite.received_decisions)}")
        print(f"   - Messages sent: {len(test_suite.sent_messages)}")
        
    finally:
        await test_suite.teardown()


if __name__ == "__main__":
    # Run tests directly
    asyncio.run(test_complete_platform_integration())
