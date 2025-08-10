#!/usr/bin/env python3
"""
User Whisperer Python SDK Usage Examples
Demonstrates various SDK features and best practices
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
import json
import random

# Import the User Whisperer SDK
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../sdk/python'))

from userwhisperer import UserWhisperer, AsyncUserWhisperer


def basic_usage_example():
    """
    Basic SDK usage example with synchronous client
    """
    print("🚀 Basic Usage Example")
    print("=" * 50)
    
    # Initialize the SDK
    uw = UserWhisperer(
        api_key='uw_demo_key_123456789abcdef',
        app_id='demo_app',
        endpoint='https://api.userwhisperer.ai',
        debug=True,
        batch_size=10,
        flush_interval=2.0
    )
    
    try:
        # Identify a user
        print("👤 Identifying user...")
        uw.identify('demo_user_python', {
            'email': 'python-demo@example.com',
            'name': 'Python Demo User',
            'language': 'python',
            'sdk_version': '1.0.0',
            'signup_date': datetime.now(timezone.utc).isoformat()
        })
        
        # Track various events
        print("📊 Tracking events...")
        
        # Page view
        uw.page('Python Demo Page', {
            'url': '/demo/python',
            'source': 'demo_script'
        })
        
        # Custom events
        events_to_track = [
            ('session_started', {'source': 'python_demo'}),
            ('feature_used', {'feature': 'python_sdk', 'version': '1.0.0'}),
            ('button_clicked', {'button': 'demo_button', 'location': 'header'}),
            ('form_submitted', {'form': 'contact', 'success': True}),
            ('purchase_completed', {
                'amount': 99.99,
                'currency': 'USD',
                'product': 'Pro Plan',
                'payment_method': 'credit_card'
            })
        ]
        
        for event_type, properties in events_to_track:
            event_id = uw.track(event_type, properties)
            print(f"  ✅ Tracked: {event_type} (ID: {event_id})")
            time.sleep(0.1)  # Small delay between events
        
        # Group association
        print("🏢 Associating with group...")
        uw.group('python_demo_org', {
            'name': 'Python Demo Organization',
            'plan': 'enterprise',
            'industry': 'Technology',
            'employees': 50
        })
        
        # Alias user
        print("🔄 Creating user alias...")
        uw.alias('python_demo_user_aliased')
        
        # Flush events
        print("🚀 Flushing events...")
        success = uw.flush()
        print(f"  Flush result: {'Success' if success else 'Failed'}")
        
        # Show queue status
        print(f"📊 Queue size: {uw.queue_size()}")
        print(f"👤 Current user: {uw.get_user_id()}")
        print(f"🔗 Session ID: {uw.get_session_id()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        # Clean up
        uw.close()
        print("✅ SDK closed successfully")


async def async_usage_example():
    """
    Asynchronous SDK usage example
    """
    print("\n🚀 Async Usage Example")
    print("=" * 50)
    
    # Initialize async client
    async with AsyncUserWhisperer(
        api_key='uw_demo_key_123456789abcdef',
        app_id='demo_app_async',
        endpoint='https://api.userwhisperer.ai',
        debug=True,
        batch_size=5,
        flush_interval=1.0
    ) as uw:
        
        # Start the client
        await uw.start()
        
        try:
            # Identify user
            print("👤 Identifying async user...")
            await uw.identify('async_demo_user', {
                'email': 'async-demo@example.com',
                'name': 'Async Demo User',
                'client_type': 'async_python',
                'features': ['async_support', 'high_performance']
            })
            
            # Track events concurrently
            print("📊 Tracking events concurrently...")
            
            async def track_user_journey():
                """Simulate a user journey"""
                journey_events = [
                    ('app_opened', {'platform': 'python'}),
                    ('onboarding_started', {'step': 1}),
                    ('onboarding_step_completed', {'step': 1, 'time_spent': 30}),
                    ('onboarding_step_completed', {'step': 2, 'time_spent': 45}),
                    ('onboarding_completed', {'total_time': 75, 'steps_completed': 2}),
                    ('dashboard_viewed', {'first_time': True}),
                    ('feature_discovered', {'feature': 'analytics'}),
                    ('help_accessed', {'section': 'getting_started'})
                ]
                
                for event_type, properties in journey_events:
                    event_id = await uw.track(event_type, properties)
                    print(f"  ✅ Async tracked: {event_type} (ID: {event_id})")
                    await asyncio.sleep(0.2)  # Simulate time between events
            
            async def track_engagement_events():
                """Track engagement events"""
                for i in range(5):
                    await uw.track('content_viewed', {
                        'content_type': 'article',
                        'content_id': f'article_{i+1}',
                        'view_duration': random.randint(30, 300)
                    })
                    print(f"  📖 Content view {i+1} tracked")
                    await asyncio.sleep(0.3)
            
            # Run tasks concurrently
            await asyncio.gather(
                track_user_journey(),
                track_engagement_events()
            )
            
            # Page tracking
            print("📄 Tracking page views...")
            pages = ['/home', '/features', '/pricing', '/contact', '/dashboard']
            
            page_tasks = []
            for page in pages:
                task = uw.page(f'Page {page}', {
                    'path': page,
                    'load_time': random.randint(100, 1000),
                    'user_agent': 'Python Async Demo'
                })
                page_tasks.append(task)
            
            await asyncio.gather(*page_tasks)
            print(f"  ✅ Tracked {len(pages)} page views")
            
            # Group and screen tracking
            await uw.group('async_demo_group', {
                'type': 'development_team',
                'size': 'small'
            })
            
            await uw.screen('Dashboard Screen', {
                'screen_size': '1920x1080',
                'widgets': ['charts', 'metrics', 'alerts']
            })
            
            # Final flush
            print("🚀 Final flush...")
            await uw.flush()
            
            print(f"📊 Final queue size: {uw.queue_size()}")
            
        except Exception as e:
            print(f"❌ Async error: {e}")


def error_handling_example():
    """
    Demonstrate error handling and retry logic
    """
    print("\n🛡️ Error Handling Example")
    print("=" * 50)
    
    # Initialize with invalid endpoint to demonstrate error handling
    uw = UserWhisperer(
        api_key='invalid_key',
        app_id='demo_app',
        endpoint='https://invalid-endpoint.example.com',
        debug=True,
        max_retries=2,
        timeout=5.0
    )
    
    try:
        # This should fail and demonstrate retry logic
        print("🔥 Attempting to track with invalid configuration...")
        
        event_id = uw.track('test_error_event', {
            'test': True,
            'expected_to_fail': True
        })
        
        print(f"Event queued with ID: {event_id}")
        
        # Try to flush (this will fail)
        print("🚀 Attempting to flush (this should fail)...")
        success = uw.flush()
        print(f"Flush result: {'Success' if success else 'Failed (as expected)'}")
        
        # Check queue size (events should be re-queued after failure)
        print(f"📊 Queue size after failed flush: {uw.queue_size()}")
        
    except Exception as e:
        print(f"❌ Expected error caught: {e}")
    
    finally:
        uw.close()


def performance_testing():
    """
    Performance testing with high-volume events
    """
    print("\n⚡ Performance Testing")
    print("=" * 50)
    
    uw = UserWhisperer(
        api_key='uw_demo_key_123456789abcdef',
        app_id='demo_app_perf',
        endpoint='https://api.userwhisperer.ai',
        debug=False,  # Disable debug for performance testing
        batch_size=100,
        flush_interval=1.0,
        max_queue_size=10000
    )
    
    try:
        # Identify test user
        uw.identify('perf_test_user', {
            'test_type': 'performance',
            'start_time': datetime.now(timezone.utc).isoformat()
        })
        
        # Track high volume of events
        num_events = 1000
        print(f"📊 Tracking {num_events} events for performance testing...")
        
        start_time = time.time()
        
        for i in range(num_events):
            uw.track('performance_test_event', {
                'event_number': i,
                'batch': i // 100,
                'timestamp': time.time(),
                'random_data': random.choice(['A', 'B', 'C', 'D', 'E'])
            })
            
            if (i + 1) % 100 == 0:
                print(f"  ✅ Tracked {i + 1} events")
        
        tracking_time = time.time() - start_time
        
        print(f"⏱️ Tracking completed in {tracking_time:.2f} seconds")
        print(f"📈 Rate: {num_events / tracking_time:.0f} events/second")
        print(f"📊 Current queue size: {uw.queue_size()}")
        
        # Final flush
        print("🚀 Final flush...")
        flush_start = time.time()
        uw.flush()
        flush_time = time.time() - flush_start
        
        print(f"⏱️ Flush completed in {flush_time:.2f} seconds")
        print(f"📊 Final queue size: {uw.queue_size()}")
        
    except Exception as e:
        print(f"❌ Performance test error: {e}")
    
    finally:
        uw.close()


def context_manager_example():
    """
    Demonstrate using SDK as context manager
    """
    print("\n🎯 Context Manager Example")
    print("=" * 50)
    
    # Using with statement for automatic cleanup
    with UserWhisperer(
        api_key='uw_demo_key_123456789abcdef',
        app_id='demo_app_context',
        endpoint='https://api.userwhisperer.ai',
        debug=True
    ) as uw:
        
        print("👤 Context manager: Identifying user...")
        uw.identify('context_user', {
            'usage_pattern': 'context_manager',
            'cleanup': 'automatic'
        })
        
        print("📊 Context manager: Tracking events...")
        for i in range(5):
            uw.track('context_event', {
                'iteration': i,
                'in_context': True
            })
        
        print("🚀 Context manager: Manual flush...")
        uw.flush()
        
        # Automatic cleanup happens when exiting the context
        print("✅ Exiting context (automatic cleanup will occur)")


def main():
    """
    Run all examples
    """
    print("🎬 User Whisperer Python SDK Examples")
    print("=" * 60)
    
    try:
        # Run synchronous examples
        basic_usage_example()
        error_handling_example()
        performance_testing()
        context_manager_example()
        
        # Run async example
        asyncio.run(async_usage_example())
        
        print("\n🎉 All examples completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Examples failed with error: {e}")


if __name__ == "__main__":
    main()
