#!/usr/bin/env python3
"""
Mixpanel Integration Example
Demonstrates how to set up and use the Mixpanel integration
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any

# Add the project root to the Python path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.integrations.mixpanel import MixpanelIntegration, MixpanelWebhookProcessor, MixpanelDataExporter


async def setup_mixpanel_integration():
    """
    Example of setting up Mixpanel integration
    """
    print("🚀 Setting up Mixpanel Integration")
    print("=" * 60)
    
    # Configuration for Mixpanel integration
    config = {
        'mixpanel_token': 'your_mixpanel_project_token',
        'mixpanel_api_secret': 'your_mixpanel_api_secret',
        'mixpanel_project_id': 'your_project_id',
        'mixpanel_eu': False,  # Set to True for EU data residency
        'webhook_secret': 'your_webhook_secret'
    }
    
    try:
        # Initialize Mixpanel integration
        print("📊 Initializing Mixpanel integration...")
        mixpanel_integration = MixpanelIntegration(config)
        
        # Set up integration for a demo app
        app_id = 'demo_app_mixpanel'
        print(f"🔧 Setting up integration for app: {app_id}")
        
        integration_result = await mixpanel_integration.setup_integration(app_id)
        
        print("✅ Mixpanel Integration setup completed!")
        print(json.dumps(integration_result, indent=2, default=str))
        
        return mixpanel_integration, integration_result
        
    except Exception as e:
        print(f"❌ Mixpanel integration setup failed: {e}")
        return None, None


async def test_event_transformation():
    """
    Test Mixpanel event transformation
    """
    print("\n🔄 Testing Mixpanel Event Transformation")
    print("=" * 60)
    
    # Sample Mixpanel event
    mixpanel_event = {
        "event": "Purchase",
        "properties": {
            "distinct_id": "user_67890",
            "time": 1642234200,
            "$browser": "Chrome",
            "$browser_version": "96.0.4664.110",
            "$city": "San Francisco",
            "$region": "California", 
            "$country_code": "US",
            "$current_url": "https://example.com/checkout",
            "$device": "Desktop",
            "$os": "Mac OS X",
            "$screen_height": 1080,
            "$screen_width": 1920,
            "value": 99.99,
            "currency": "USD",
            "product_name": "Pro Plan",
            "payment_method": "stripe",
            "utm_source": "google",
            "utm_medium": "cpc",
            "utm_campaign": "q4_promotion"
        }
    }
    
    # Initialize Mixpanel integration (minimal config for transformation test)
    config = {
        'mixpanel_token': 'test_token',
        'mixpanel_api_secret': 'test_secret',
        'mixpanel_project_id': 'test_project'
    }
    
    try:
        mixpanel_integration = MixpanelIntegration(config)
        
        print("📥 Original Mixpanel event:")
        print(json.dumps(mixpanel_event, indent=2))
        
        # Transform the event
        print("\n🔄 Transforming event...")
        transformed_event = mixpanel_integration.transform_event(mixpanel_event)
        
        print("\n📤 Transformed User Whisperer event:")
        print(json.dumps(transformed_event, indent=2, default=str))
        
        # Validate transformation
        print("\n✅ Transformation validation:")
        print(f"  Event Type: {mixpanel_event['event']} → {transformed_event['event_type']}")
        print(f"  User ID: {mixpanel_event['properties']['distinct_id']} → {transformed_event['user_id']}")
        print(f"  Custom Properties: {len(transformed_event['properties'])} extracted")
        print(f"  Context Sections: {len(transformed_event['context'])} sections")
        
        # Show context details
        print("\n📋 Context breakdown:")
        for section, data in transformed_event['context'].items():
            non_empty_fields = sum(1 for v in data.values() if v is not None and v != '')
            print(f"  {section}: {non_empty_fields} fields populated")
        
    except Exception as e:
        print(f"❌ Event transformation failed: {e}")


async def test_webhook_processing():
    """
    Test Mixpanel webhook processing
    """
    print("\n🌐 Testing Mixpanel Webhook Processing")
    print("=" * 60)
    
    # Sample webhook payload
    webhook_payload = {
        "data": {
            "event": "Sign Up",
            "properties": {
                "distinct_id": "user_12345",
                "time": 1642234200,
                "$email": "user@example.com",
                "$name": "John Doe",
                "$browser": "Safari",
                "$city": "New York",
                "$region": "New York",
                "$country_code": "US",
                "signup_method": "email",
                "referrer": "google",
                "plan": "free"
            }
        }
    }
    
    try:
        # Initialize webhook processor
        config = {
            'mixpanel_token': 'test_token',
            'mixpanel_api_secret': 'test_secret',
            'mixpanel_project_id': 'test_project',
            'webhook_secret': 'test_webhook_secret'
        }
        
        mixpanel_integration = MixpanelIntegration(config)
        webhook_processor = MixpanelWebhookProcessor(
            integration=mixpanel_integration,
            webhook_secret='test_webhook_secret'
        )
        
        print("📥 Processing webhook payload...")
        print("📊 Webhook data:")
        print(json.dumps(webhook_payload, indent=2))
        
        # Process the webhook
        result = await webhook_processor.process_webhook_event(webhook_payload)
        
        print("\n✅ Webhook processing result:")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"❌ Webhook processing failed: {e}")


async def test_historical_data_export():
    """
    Test historical data export from Mixpanel
    """
    print("\n📈 Testing Historical Data Export")
    print("=" * 60)
    
    config = {
        'mixpanel_token': 'test_token',
        'mixpanel_api_secret': 'test_secret',
        'mixpanel_project_id': 'test_project'
    }
    
    try:
        mixpanel_integration = MixpanelIntegration(config)
        
        # Define date range for export (last 7 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        print(f"📅 Exporting data from {start_date.date()} to {end_date.date()}")
        
        # Note: This would normally make actual API calls to Mixpanel
        # For demo purposes, we'll simulate the process
        print("⚠️  Note: This is a simulation - actual export requires Mixpanel API credentials")
        
        export_result = await mixpanel_integration.export_historical_data(
            app_id='demo_app',
            start_date=start_date,
            end_date=end_date,
            batch_size=1000
        )
        
        print("✅ Historical export completed:")
        print(json.dumps(export_result, indent=2, default=str))
        
    except Exception as e:
        print(f"❌ Historical export failed: {e}")


async def test_data_exporter():
    """
    Test the MixpanelDataExporter utility
    """
    print("\n🔧 Testing Mixpanel Data Exporter")
    print("=" * 60)
    
    config = {
        'mixpanel_token': 'test_token',
        'mixpanel_api_secret': 'test_secret',
        'mixpanel_project_id': 'test_project'
    }
    
    try:
        mixpanel_integration = MixpanelIntegration(config)
        data_exporter = MixpanelDataExporter(mixpanel_integration)
        
        # Test date range splitting
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        print(f"📅 Testing date range splitting for 30-day period")
        print(f"  Start: {start_date.date()}")
        print(f"  End: {end_date.date()}")
        
        # This would normally export data, but we'll just show the concept
        print("⚠️  Note: Actual export requires API credentials and would process real data")
        
        # Simulate the chunking logic
        chunks = data_exporter._split_date_range(start_date, end_date, max_days=7)
        
        print(f"\n📦 Date range split into {len(chunks)} chunks:")
        for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
            days = (chunk_end - chunk_start).days
            print(f"  Chunk {i}: {chunk_start.date()} to {chunk_end.date()} ({days} days)")
        
    except Exception as e:
        print(f"❌ Data exporter test failed: {e}")


def test_event_mapping():
    """
    Test and display Mixpanel event mapping
    """
    print("\n🗺️ Testing Event Mapping")
    print("=" * 60)
    
    config = {
        'mixpanel_token': 'test_token',
        'mixpanel_api_secret': 'test_secret',
        'mixpanel_project_id': 'test_project'
    }
    
    try:
        mixpanel_integration = MixpanelIntegration(config)
        event_mapping = mixpanel_integration.load_event_mapping()
        
        print(f"📊 Loaded {len(event_mapping)} event mappings:")
        
        # Group mappings by category
        categories = {
            'Special Events': ['$identify', '$create_alias', '$merge', '$track'],
            'Page/Screen Events': ['$pageview', '$screen', 'Page View', 'Screen View'],
            'User Lifecycle': ['Sign Up', 'Login', 'Logout', 'Account Created'],
            'E-commerce': ['Purchase', 'Add to Cart', 'Checkout Started'],
            'Subscription': ['Subscription Started', 'Trial Started'],
            'Content': ['Content Viewed', 'Video Started', 'Article Read'],
            'Marketing': ['Email Opened', 'Push Notification Received'],
            'Social': ['Share', 'Like', 'Comment', 'Follow']
        }
        
        for category, events in categories.items():
            print(f"\n📂 {category}:")
            for event in events:
                if event in event_mapping:
                    mapped = event_mapping[event]
                    print(f"  {event:30} → {mapped}")
        
        # Test mapping function
        print("\n🔄 Testing mapping function:")
        test_events = ['Purchase', 'Sign Up', 'unknown_event', '$identify']
        
        for event in test_events:
            mapped = mixpanel_integration.map_event_type(event)
            print(f"  {event:20} → {mapped}")
        
    except Exception as e:
        print(f"❌ Event mapping test failed: {e}")


async def test_rate_limiting():
    """
    Test rate limiting functionality
    """
    print("\n⏱️ Testing Rate Limiting")
    print("=" * 60)
    
    config = {
        'mixpanel_token': 'test_token',
        'mixpanel_api_secret': 'test_secret',
        'mixpanel_project_id': 'test_project'
    }
    
    try:
        mixpanel_integration = MixpanelIntegration(config)
        rate_limiter = mixpanel_integration.rate_limiter
        
        print(f"📊 Rate limiter configured:")
        print(f"  Max requests: {rate_limiter.max_requests}")
        print(f"  Window: {rate_limiter.window_seconds} seconds")
        
        # Test rate limiting
        print("\n🔄 Testing rate limit behavior:")
        
        # Make several requests quickly
        for i in range(5):
            allowed = rate_limiter.is_allowed()
            print(f"  Request {i+1}: {'✅ Allowed' if allowed else '❌ Rate limited'}")
            
            if not allowed:
                reset_time = rate_limiter.get_reset_time()
                print(f"    Reset at: {reset_time}")
        
        # Show current state
        print(f"\n📈 Current request count: {len(rate_limiter.requests)}")
        
    except Exception as e:
        print(f"❌ Rate limiting test failed: {e}")


async def comprehensive_mixpanel_demo():
    """
    Run comprehensive Mixpanel integration demo
    """
    print("🎬 Comprehensive Mixpanel Integration Demo")
    print("=" * 70)
    
    try:
        # 1. Setup integration
        await setup_mixpanel_integration()
        
        # 2. Test event transformation
        await test_event_transformation()
        
        # 3. Test webhook processing
        await test_webhook_processing()
        
        # 4. Test historical export
        await test_historical_data_export()
        
        # 5. Test data exporter utility
        await test_data_exporter()
        
        # 6. Test event mapping
        test_event_mapping()
        
        # 7. Test rate limiting
        await test_rate_limiting()
        
        print("\n🎉 Mixpanel Integration Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")


def main():
    """
    Main function to run all Mixpanel integration examples
    """
    print("🚀 Mixpanel Integration Examples")
    print("=" * 80)
    
    try:
        # Run the comprehensive demo
        asyncio.run(comprehensive_mixpanel_demo())
        
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")


if __name__ == "__main__":
    main()
