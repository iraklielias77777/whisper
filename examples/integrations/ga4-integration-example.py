#!/usr/bin/env python3
"""
Google Analytics 4 Integration Example
Demonstrates how to set up and use the GA4 integration
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any

# Add the project root to the Python path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.integrations.google_analytics import GA4Integration, GA4WebhookProcessor


async def setup_ga4_integration():
    """
    Example of setting up GA4 integration
    """
    print("🚀 Setting up Google Analytics 4 Integration")
    print("=" * 60)
    
    # Configuration for GA4 integration
    config = {
        'ga4_property_id': '123456789',
        'ga4_measurement_id': 'G-XXXXXXXXXX',
        'ga4_api_secret': 'your_ga4_api_secret_here',
        'gcp_project': 'your-gcp-project',
        'bigquery_dataset': 'user_whisperer_analytics',
        'webhook_api_key': 'your_webhook_api_key',
        'enable_bigquery_export': True
    }
    
    try:
        # Initialize GA4 integration
        print("📊 Initializing GA4 integration...")
        ga4_integration = GA4Integration(config)
        
        # Set up integration for a demo app
        app_id = 'demo_app_ga4'
        print(f"🔧 Setting up integration for app: {app_id}")
        
        integration_result = await ga4_integration.setup_integration(app_id)
        
        print("✅ GA4 Integration setup completed!")
        print(json.dumps(integration_result, indent=2, default=str))
        
        return ga4_integration, integration_result
        
    except Exception as e:
        print(f"❌ GA4 integration setup failed: {e}")
        return None, None


async def test_event_transformation():
    """
    Test GA4 event transformation
    """
    print("\n🔄 Testing GA4 Event Transformation")
    print("=" * 60)
    
    # Sample GA4 event
    ga4_event = {
        "event_name": "page_view",
        "event_timestamp": 1642234200000000,  # Microseconds
        "user_id": "user_12345",
        "user_pseudo_id": "anonymous_12345",
        "ga_session_id": "1642234200",
        "ga_session_number": 1,
        "event_params": [
            {
                "key": "page_title",
                "value": {"string_value": "Home Page"}
            },
            {
                "key": "page_location",
                "value": {"string_value": "https://example.com/"}
            },
            {
                "key": "engagement_time_msec",
                "value": {"int_value": 15000}
            }
        ],
        "device": {
            "category": "desktop",
            "mobile_brand_name": "",
            "mobile_model_name": "",
            "operating_system": "Windows",
            "operating_system_version": "10",
            "language": "en-us"
        },
        "geo": {
            "continent": "Americas",
            "country": "United States",
            "region": "California",
            "city": "San Francisco"
        },
        "traffic_source": {
            "medium": "organic",
            "source": "google",
            "campaign": ""
        },
        "app_info": {
            "version": "1.0.0",
            "install_source": "google-play"
        }
    }
    
    # Initialize GA4 integration (minimal config for transformation test)
    config = {
        'ga4_property_id': '123456789',
        'ga4_measurement_id': 'G-XXXXXXXXXX',
        'ga4_api_secret': 'test_secret',
        'gcp_project': 'test-project'
    }
    
    try:
        ga4_integration = GA4Integration(config)
        
        print("📥 Original GA4 event:")
        print(json.dumps(ga4_event, indent=2))
        
        # Transform the event
        print("\n🔄 Transforming event...")
        transformed_event = ga4_integration.transform_event(ga4_event)
        
        print("\n📤 Transformed User Whisperer event:")
        print(json.dumps(transformed_event, indent=2, default=str))
        
        # Validate transformation
        print("\n✅ Transformation validation:")
        print(f"  Event Type: {ga4_event['event_name']} → {transformed_event['event_type']}")
        print(f"  User ID: {ga4_event.get('user_id', 'N/A')} → {transformed_event['user_id']}")
        print(f"  Properties count: {len(ga4_event['event_params'])} → {len(transformed_event['properties'])}")
        print(f"  Has context: {bool(transformed_event.get('context'))}")
        
    except Exception as e:
        print(f"❌ Event transformation failed: {e}")


async def test_webhook_processing():
    """
    Test GA4 webhook processing
    """
    print("\n🌐 Testing GA4 Webhook Processing")
    print("=" * 60)
    
    # Sample webhook payload
    webhook_payload = {
        "message": {
            "data": "eyJldmVudF9uYW1lIjogInB1cmNoYXNlIiwgImV2ZW50X3RpbWVzdGFtcCI6IDE2NDIyMzQyMDAwMDAwMDAsICJ1c2VyX2lkIjogInVzZXJfNjc4OTAiLCAiZXZlbnRfcGFyYW1zIjogW3sia2V5IjogInZhbHVlIiwgInZhbHVlIjogeyJkb3VibGVfdmFsdWUiOiA5OS45OX19LCB7ImtleSI6ICJjdXJyZW5jeSIsICJ2YWx1ZSI6IHsic3RyaW5nX3ZhbHVlIjogIlVTRCJ9fV19"
        }
    }
    
    try:
        # Initialize webhook processor
        config = {
            'ga4_property_id': '123456789',
            'ga4_measurement_id': 'G-XXXXXXXXXX', 
            'ga4_api_secret': 'test_secret',
            'gcp_project': 'test-project'
        }
        
        ga4_integration = GA4Integration(config)
        webhook_processor = GA4WebhookProcessor(
            integration=ga4_integration,
            webhook_secret='test_webhook_secret'
        )
        
        print("📥 Processing webhook payload...")
        
        # Decode the base64 message data (simulating Pub/Sub)
        import base64
        decoded_data = base64.b64decode(webhook_payload['message']['data'])
        event_data = json.loads(decoded_data.decode('utf-8'))
        
        print("📊 Decoded event data:")
        print(json.dumps(event_data, indent=2))
        
        # Process the webhook
        result = await webhook_processor.process_webhook_event(event_data)
        
        print("\n✅ Webhook processing result:")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"❌ Webhook processing failed: {e}")


async def test_historical_data_sync():
    """
    Test historical data synchronization
    """
    print("\n📈 Testing Historical Data Sync")
    print("=" * 60)
    
    config = {
        'ga4_property_id': '123456789',
        'ga4_measurement_id': 'G-XXXXXXXXXX',
        'ga4_api_secret': 'test_secret',
        'gcp_project': 'test-project',
        'bigquery_project': 'test-project',
        'bigquery_dataset': 'analytics'
    }
    
    try:
        ga4_integration = GA4Integration(config)
        
        # Define date range for sync (last 7 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        print(f"📅 Syncing data from {start_date.date()} to {end_date.date()}")
        
        # Note: This would normally make actual API calls to GA4
        # For demo purposes, we'll simulate the process
        print("⚠️  Note: This is a simulation - actual sync requires GA4 API credentials")
        
        sync_result = await ga4_integration.sync_historical_data(
            app_id='demo_app',
            start_date=start_date,
            end_date=end_date,
            batch_size=1000
        )
        
        print("✅ Historical sync completed:")
        print(json.dumps(sync_result, indent=2, default=str))
        
    except Exception as e:
        print(f"❌ Historical sync failed: {e}")


def generate_cloud_function_example():
    """
    Generate example Cloud Function code
    """
    print("\n☁️ Google Cloud Function Example")
    print("=" * 60)
    
    config = {
        'ga4_property_id': '123456789',
        'ga4_measurement_id': 'G-XXXXXXXXXX',
        'ga4_api_secret': 'test_secret',
        'gcp_project': 'test-project',
        'webhook_api_key': 'test_webhook_key'
    }
    
    ga4_integration = GA4Integration(config)
    
    # Generate Cloud Function code
    app_id = 'demo_app'
    topic_name = 'ga4-events-demo'
    
    function_code = ga4_integration.generate_cloud_function_code(app_id, topic_name)
    
    print("📝 Generated Cloud Function code:")
    print("```python")
    print(function_code[:1000] + "..." if len(function_code) > 1000 else function_code)
    print("```")
    
    # Save to file
    output_file = 'ga4_cloud_function_example.py'
    with open(output_file, 'w') as f:
        f.write(function_code)
    
    print(f"\n💾 Full code saved to: {output_file}")


async def comprehensive_ga4_demo():
    """
    Run comprehensive GA4 integration demo
    """
    print("🎬 Comprehensive GA4 Integration Demo")
    print("=" * 70)
    
    try:
        # 1. Setup integration
        await setup_ga4_integration()
        
        # 2. Test event transformation
        await test_event_transformation()
        
        # 3. Test webhook processing
        await test_webhook_processing()
        
        # 4. Test historical sync
        await test_historical_data_sync()
        
        # 5. Generate Cloud Function example
        generate_cloud_function_example()
        
        print("\n🎉 GA4 Integration Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")


def main():
    """
    Main function to run all GA4 integration examples
    """
    print("🚀 Google Analytics 4 Integration Examples")
    print("=" * 80)
    
    try:
        # Run the comprehensive demo
        asyncio.run(comprehensive_ga4_demo())
        
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")


if __name__ == "__main__":
    main()
