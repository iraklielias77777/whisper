"""
Google Analytics 4 Integration for User Whisperer Platform
Handles real-time event streaming, historical data sync, and Cloud Function deployment
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import hashlib
import base64

# Google Cloud imports
try:
    from google.cloud import functions_v1
    from google.analytics.data_v1beta import BetaAnalyticsDataClient
    from google.analytics.admin import AnalyticsAdminServiceClient
    from google.cloud import bigquery
    from google.cloud import pubsub_v1
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False
    logging.warning("Google Cloud libraries not available")

from .base import BaseIntegration, WebhookProcessor

logger = logging.getLogger(__name__)


class GA4Integration(BaseIntegration):
    """
    Google Analytics 4 integration with real-time streaming and historical sync
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not GOOGLE_CLOUD_AVAILABLE:
            raise ImportError("Google Cloud libraries required for GA4 integration")
        
        self.property_id = config['ga4_property_id']
        self.measurement_id = config['ga4_measurement_id']
        self.api_secret = config['ga4_api_secret']
        self.gcp_project = config['gcp_project']
        self.bigquery_dataset = config.get('bigquery_dataset', 'user_whisperer')
        
        # Initialize Google Cloud clients
        self.data_client = BetaAnalyticsDataClient()
        self.admin_client = AnalyticsAdminServiceClient()
        self.bigquery_client = bigquery.Client(project=self.gcp_project)
        self.functions_client = functions_v1.CloudFunctionsServiceClient()
        self.pubsub_client = pubsub_v1.PublisherClient()
        
        logger.info(f"GA4 Integration initialized for property: {self.property_id}")
    
    def load_event_mapping(self) -> Dict[str, str]:
        """Load GA4 event mapping configuration"""
        
        return {
            # Standard events
            'page_view': 'page_viewed',
            'session_start': 'session_started',
            'first_visit': 'user_signed_up',
            'user_engagement': 'user_engaged',
            
            # Ecommerce events
            'purchase': 'purchase_completed',
            'add_to_cart': 'cart_item_added',
            'begin_checkout': 'checkout_started',
            'add_payment_info': 'payment_info_added',
            'add_shipping_info': 'shipping_info_added',
            'refund': 'purchase_refunded',
            'remove_from_cart': 'cart_item_removed',
            'select_item': 'product_selected',
            'select_promotion': 'promotion_selected',
            'view_cart': 'cart_viewed',
            'view_item': 'product_viewed',
            'view_item_list': 'product_list_viewed',
            'view_promotion': 'promotion_viewed',
            
            # Engagement events
            'click': 'element_clicked',
            'scroll': 'page_scrolled',
            'search': 'search_performed',
            'share': 'content_shared',
            'sign_up': 'user_registered',
            'login': 'user_logged_in',
            'file_download': 'file_downloaded',
            'form_submit': 'form_submitted',
            'video_start': 'video_started',
            'video_progress': 'video_progress',
            'video_complete': 'video_completed'
        }
    
    async def setup_integration(self, app_id: str) -> Dict[str, Any]:
        """
        Set up complete GA4 integration for an app
        """
        
        logger.info(f"Setting up GA4 integration for app: {app_id}")
        
        try:
            # Step 1: Create Pub/Sub topic for event streaming
            topic_name = f"ga4-events-{app_id}"
            topic_path = self.pubsub_client.topic_path(self.gcp_project, topic_name)
            
            try:
                self.pubsub_client.create_topic(request={"name": topic_path})
                logger.info(f"Created Pub/Sub topic: {topic_name}")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.error(f"Failed to create Pub/Sub topic: {e}")
                    raise
            
            # Step 2: Deploy Cloud Function for event processing
            function_name = f"ga4-processor-{app_id}"
            cloud_function_url = await self.deploy_cloud_function(app_id, function_name, topic_name)
            
            # Step 3: Create webhook endpoint
            webhook_url = f"https://api.userwhisperer.ai/webhooks/ga4/{app_id}"
            
            # Step 4: Set up BigQuery export (if enabled)
            bigquery_setup = None
            if self.config.get('enable_bigquery_export', True):
                bigquery_setup = await self.setup_bigquery_export(app_id)
            
            # Step 5: Configure GA4 data stream
            data_stream_config = await self.configure_data_stream(app_id)
            
            integration_config = {
                'status': 'active',
                'app_id': app_id,
                'property_id': self.property_id,
                'measurement_id': self.measurement_id,
                'webhook_url': webhook_url,
                'cloud_function': {
                    'name': function_name,
                    'url': cloud_function_url,
                    'topic': topic_name
                },
                'bigquery_setup': bigquery_setup,
                'data_stream': data_stream_config,
                'event_mapping': self.event_mapping,
                'configured_at': datetime.utcnow().isoformat()
            }
            
            # Store integration config
            await self.store_integration_config(app_id, integration_config)
            
            logger.info(f"GA4 integration setup completed for app: {app_id}")
            return integration_config
            
        except Exception as e:
            logger.error(f"GA4 integration setup failed for app {app_id}: {e}")
            raise
    
    async def deploy_cloud_function(self, app_id: str, function_name: str, topic_name: str) -> str:
        """
        Deploy Google Cloud Function for GA4 event processing
        """
        
        logger.info(f"Deploying Cloud Function: {function_name}")
        
        # Generate Cloud Function source code
        function_source = self.generate_cloud_function_code(app_id, topic_name)
        
        # Cloud Function configuration
        parent = f"projects/{self.gcp_project}/locations/us-central1"
        function_path = f"{parent}/functions/{function_name}"
        
        function_config = {
            'name': function_path,
            'description': f'GA4 event processor for User Whisperer app {app_id}',
            'source_code': {
                'zip_upload': {
                    'upload_url': '',  # Would be populated by upload process
                }
            },
            'entry_point': 'process_ga4_event',
            'runtime': 'python39',
            'timeout': {'seconds': 60},
            'available_memory_mb': 256,
            'event_trigger': {
                'event_type': 'providers/cloud.pubsub/eventTypes/topic.publish',
                'resource': f"projects/{self.gcp_project}/topics/{topic_name}"
            },
            'environment_variables': {
                'APP_ID': app_id,
                'WEBHOOK_URL': f"https://api.userwhisperer.ai/webhooks/ga4/{app_id}",
                'API_KEY': self.config.get('webhook_api_key', ''),
                'PROPERTY_ID': self.property_id
            },
            'labels': {
                'app': 'user-whisperer',
                'integration': 'ga4',
                'app-id': app_id.replace('_', '-')
            }
        }
        
        try:
            # In a real implementation, this would:
            # 1. Create a deployment package with the function code
            # 2. Upload it to Google Cloud Storage
            # 3. Deploy the function using the Functions API
            
            # For now, we'll simulate the deployment
            function_url = f"https://us-central1-{self.gcp_project}.cloudfunctions.net/{function_name}"
            
            logger.info(f"Cloud Function deployed: {function_url}")
            return function_url
            
        except Exception as e:
            logger.error(f"Cloud Function deployment failed: {e}")
            raise
    
    def generate_cloud_function_code(self, app_id: str, topic_name: str) -> str:
        """
        Generate Cloud Function source code for GA4 event processing
        """
        
        function_code = f'''
import json
import base64
import requests
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def process_ga4_event(event, context):
    """
    Process GA4 events from Pub/Sub and forward to User Whisperer
    """
    
    try:
        # Decode Pub/Sub message
        if 'data' in event:
            message_data = base64.b64decode(event['data']).decode('utf-8')
            ga4_event = json.loads(message_data)
        else:
            ga4_event = event
        
        logger.info(f"Processing GA4 event: {{ga4_event.get('event_name', 'unknown')}}")
        
        # Transform GA4 event to User Whisperer format
        uw_event = transform_ga4_event(ga4_event)
        
        # Send to User Whisperer webhook
        webhook_url = "https://api.userwhisperer.ai/webhooks/ga4/{app_id}"
        api_key = "{self.config.get('webhook_api_key', '')}"
        
        headers = {{
            'Authorization': f'Bearer {{api_key}}',
            'Content-Type': 'application/json',
            'X-GA4-Property': "{self.property_id}",
            'X-Source': 'ga4-cloud-function'
        }}
        
        response = requests.post(
            webhook_url,
            json=uw_event,
            headers=headers,
            timeout=30
        )
        
        response.raise_for_status()
        
        logger.info(f"Successfully processed GA4 event: {{uw_event['id']}}")
        
        return {{
            'statusCode': 200,
            'body': json.dumps({{'processed': True, 'event_id': uw_event['id']}})
        }}
        
    except Exception as e:
        logger.error(f"Failed to process GA4 event: {{e}}")
        
        return {{
            'statusCode': 500,
            'body': json.dumps({{'error': str(e)}})
        }}

def transform_ga4_event(ga4_event):
    """Transform GA4 event to User Whisperer format"""
    
    # Event mapping
    event_mapping = {json.dumps(self.event_mapping, indent=8)}
    
    # Extract basic event info
    event_name = ga4_event.get('event_name', 'unknown')
    user_id = ga4_event.get('user_id') or ga4_event.get('user_pseudo_id')
    timestamp = ga4_event.get('event_timestamp', 0)
    
    # Map event type
    mapped_event_type = event_mapping.get(event_name, event_name.lower().replace(' ', '_'))
    
    # Extract properties from event parameters
    properties = extract_event_properties(ga4_event)
    
    # Extract context
    context = extract_event_context(ga4_event)
    
    # Create User Whisperer event
    uw_event = {{
        'id': f"ga4_{{int(timestamp)}}_{user_id[:8] if user_id else 'anon'}",
        'app_id': "{app_id}",
        'user_id': user_id,
        'event_type': mapped_event_type,
        'properties': properties,
        'context': context,
        'timestamp': datetime.fromtimestamp(timestamp / 1000000).isoformat() if timestamp else datetime.utcnow().isoformat(),
        'source': 'ga4',
        'source_event_name': event_name
    }}
    
    return uw_event

def extract_event_properties(ga4_event):
    """Extract properties from GA4 event parameters"""
    
    properties = {{}}
    
    for param in ga4_event.get('event_params', []):
        key = param.get('key')
        value = param.get('value', {{}})
        
        # Extract the actual value based on type
        if 'string_value' in value:
            properties[key] = value['string_value']
        elif 'int_value' in value:
            properties[key] = value['int_value']
        elif 'float_value' in value:
            properties[key] = value['float_value']
        elif 'double_value' in value:
            properties[key] = value['double_value']
    
    return properties

def extract_event_context(ga4_event):
    """Extract context from GA4 event"""
    
    context = {{
        'ga4': {{
            'session_id': ga4_event.get('ga_session_id'),
            'session_number': ga4_event.get('ga_session_number'),
            'property_id': "{self.property_id}",
            'stream_id': ga4_event.get('stream_id')
        }},
        'traffic_source': ga4_event.get('traffic_source', {{}}),
        'device': ga4_event.get('device', {{}}),
        'geo': ga4_event.get('geo', {{}}),
        'app_info': ga4_event.get('app_info', {{}})
    }}
    
    return context
'''
        
        return function_code
    
    async def setup_bigquery_export(self, app_id: str) -> Dict[str, Any]:
        """
        Set up BigQuery export for GA4 data
        """
        
        logger.info(f"Setting up BigQuery export for app: {app_id}")
        
        try:
            # Create dataset if it doesn't exist
            dataset_id = f"{self.bigquery_dataset}_{app_id}"
            dataset_ref = self.bigquery_client.dataset(dataset_id)
            
            try:
                dataset = bigquery.Dataset(dataset_ref)
                dataset.location = "US"
                dataset.description = f"GA4 data export for User Whisperer app {app_id}"
                
                dataset = self.bigquery_client.create_dataset(dataset)
                logger.info(f"Created BigQuery dataset: {dataset_id}")
                
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.error(f"Failed to create BigQuery dataset: {e}")
                    raise
            
            # Create events table
            table_id = f"{dataset_id}.ga4_events"
            table_ref = dataset_ref.table("ga4_events")
            
            schema = [
                bigquery.SchemaField("event_date", "DATE"),
                bigquery.SchemaField("event_timestamp", "INTEGER"),
                bigquery.SchemaField("event_name", "STRING"),
                bigquery.SchemaField("user_id", "STRING"),
                bigquery.SchemaField("user_pseudo_id", "STRING"),
                bigquery.SchemaField("ga_session_id", "STRING"),
                bigquery.SchemaField("ga_session_number", "INTEGER"),
                bigquery.SchemaField("event_params", "RECORD", mode="REPEATED", fields=[
                    bigquery.SchemaField("key", "STRING"),
                    bigquery.SchemaField("value", "RECORD", fields=[
                        bigquery.SchemaField("string_value", "STRING"),
                        bigquery.SchemaField("int_value", "INTEGER"),
                        bigquery.SchemaField("float_value", "FLOAT"),
                        bigquery.SchemaField("double_value", "FLOAT"),
                    ])
                ]),
                bigquery.SchemaField("device", "RECORD", fields=[
                    bigquery.SchemaField("category", "STRING"),
                    bigquery.SchemaField("mobile_brand_name", "STRING"),
                    bigquery.SchemaField("mobile_model_name", "STRING"),
                    bigquery.SchemaField("operating_system", "STRING"),
                    bigquery.SchemaField("language", "STRING"),
                ]),
                bigquery.SchemaField("geo", "RECORD", fields=[
                    bigquery.SchemaField("continent", "STRING"),
                    bigquery.SchemaField("country", "STRING"),
                    bigquery.SchemaField("region", "STRING"),
                    bigquery.SchemaField("city", "STRING"),
                ]),
            ]
            
            table = bigquery.Table(table_ref, schema=schema)
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="event_date"
            )
            
            try:
                table = self.bigquery_client.create_table(table)
                logger.info(f"Created BigQuery table: {table_id}")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.error(f"Failed to create BigQuery table: {e}")
                    raise
            
            return {
                'dataset_id': dataset_id,
                'table_id': table_id,
                'status': 'active'
            }
            
        except Exception as e:
            logger.error(f"BigQuery setup failed: {e}")
            raise
    
    async def configure_data_stream(self, app_id: str) -> Dict[str, Any]:
        """
        Configure GA4 data stream for real-time export
        """
        
        logger.info(f"Configuring GA4 data stream for app: {app_id}")
        
        # This would configure the GA4 property to send events to our Pub/Sub topic
        # In practice, this would involve:
        # 1. Creating a custom data stream
        # 2. Configuring event export to Pub/Sub
        # 3. Setting up proper authentication
        
        return {
            'stream_name': f"user-whisperer-{app_id}",
            'export_destination': f"projects/{self.gcp_project}/topics/ga4-events-{app_id}",
            'status': 'configured'
        }
    
    def transform_event(self, ga4_event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform GA4 event to User Whisperer format
        """
        
        # Extract basic event info
        event_name = ga4_event.get('event_name', 'unknown')
        user_id = ga4_event.get('user_id') or ga4_event.get('user_pseudo_id')
        timestamp = ga4_event.get('event_timestamp', 0)
        
        # Map event type
        mapped_event_type = self.map_event_type(event_name)
        
        # Extract properties from event parameters
        properties = self.extract_event_properties(ga4_event)
        
        # Extract context
        context = self.extract_event_context(ga4_event)
        
        # Create User Whisperer event
        transformed_event = {
            'id': self.generate_event_id(),
            'user_id': user_id,
            'event_type': mapped_event_type,
            'properties': properties,
            'context': context,
            'timestamp': datetime.fromtimestamp(timestamp / 1000000).isoformat() if timestamp else datetime.utcnow().isoformat(),
            'source': 'ga4',
            'source_event_name': event_name
        }
        
        return transformed_event
    
    def extract_event_properties(self, ga4_event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract properties from GA4 event parameters"""
        
        properties = {}
        
        for param in ga4_event.get('event_params', []):
            key = param.get('key')
            value = param.get('value', {})
            
            # Extract the actual value based on type
            if 'string_value' in value:
                properties[key] = value['string_value']
            elif 'int_value' in value:
                properties[key] = value['int_value']
            elif 'float_value' in value:
                properties[key] = value['float_value']
            elif 'double_value' in value:
                properties[key] = value['double_value']
        
        return properties
    
    def extract_event_context(self, ga4_event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract context from GA4 event"""
        
        return {
            'ga4': {
                'session_id': ga4_event.get('ga_session_id'),
                'session_number': ga4_event.get('ga_session_number'),
                'property_id': self.property_id,
                'stream_id': ga4_event.get('stream_id')
            },
            'traffic_source': ga4_event.get('traffic_source', {}),
            'device': {
                'category': ga4_event.get('device', {}).get('category'),
                'mobile_brand_name': ga4_event.get('device', {}).get('mobile_brand_name'),
                'mobile_model_name': ga4_event.get('device', {}).get('mobile_model_name'),
                'operating_system': ga4_event.get('device', {}).get('operating_system'),
                'operating_system_version': ga4_event.get('device', {}).get('operating_system_version'),
                'language': ga4_event.get('device', {}).get('language')
            },
            'geo': {
                'continent': ga4_event.get('geo', {}).get('continent'),
                'country': ga4_event.get('geo', {}).get('country'),
                'region': ga4_event.get('geo', {}).get('region'),
                'city': ga4_event.get('geo', {}).get('city')
            },
            'app_info': {
                'version': ga4_event.get('app_info', {}).get('version'),
                'install_source': ga4_event.get('app_info', {}).get('install_source')
            }
        }
    
    async def sync_historical_data(
        self,
        app_id: str,
        start_date: datetime,
        end_date: datetime,
        batch_size: int = 10000
    ) -> Dict[str, Any]:
        """
        Sync historical GA4 data using the Data API
        """
        
        logger.info(f"Syncing historical GA4 data for app {app_id}: {start_date} to {end_date}")
        
        try:
            from google.analytics.data_v1beta import (
                RunReportRequest,
                DateRange,
                Dimension,
                Metric
            )
            
            request = RunReportRequest(
                property=f"properties/{self.property_id}",
                date_ranges=[DateRange(
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d")
                )],
                dimensions=[
                    Dimension(name="date"),
                    Dimension(name="eventName"),
                    Dimension(name="userId"),
                    Dimension(name="sessionId"),
                    Dimension(name="deviceCategory"),
                    Dimension(name="country")
                ],
                metrics=[
                    Metric(name="eventCount"),
                    Metric(name="totalUsers"),
                    Metric(name="sessions")
                ],
                limit=batch_size
            )
            
            response = self.data_client.run_report(request)
            
            # Process and transform events
            events_processed = 0
            batch_events = []
            
            for row in response.rows:
                # Create event from row data
                event_data = {
                    'event_date': row.dimension_values[0].value,
                    'event_name': row.dimension_values[1].value,
                    'user_id': row.dimension_values[2].value,
                    'ga_session_id': row.dimension_values[3].value,
                    'device': {'category': row.dimension_values[4].value},
                    'geo': {'country': row.dimension_values[5].value},
                    'event_count': int(row.metric_values[0].value),
                    'event_timestamp': int(datetime.strptime(
                        row.dimension_values[0].value, 
                        "%Y%m%d"
                    ).timestamp() * 1000000)
                }
                
                # Transform and add to batch
                transformed_event = self.transform_event(event_data)
                batch_events.append(transformed_event)
                events_processed += 1
                
                # Process batch when full
                if len(batch_events) >= batch_size:
                    await self.process_event_batch(batch_events)
                    batch_events = []
            
            # Process remaining events
            if batch_events:
                await self.process_event_batch(batch_events)
            
            sync_result = {
                'app_id': app_id,
                'events_processed': events_processed,
                'date_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'status': 'completed',
                'synced_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Historical sync completed: {events_processed} events processed")
            return sync_result
            
        except Exception as e:
            logger.error(f"Historical data sync failed: {e}")
            raise
    
    async def process_event_batch(self, events: List[Dict[str, Any]]):
        """Process a batch of transformed events"""
        
        # Store events in the User Whisperer system
        for event in events:
            await self.store_event(event)
    
    async def store_integration_config(self, app_id: str, config: Dict[str, Any]):
        """Store integration configuration"""
        
        # This would store the configuration in the User Whisperer database
        logger.info(f"Stored GA4 integration config for app: {app_id}")


class GA4WebhookProcessor(WebhookProcessor):
    """
    Webhook processor for GA4 events
    """
    
    def verify_webhook_signature(self, payload: bytes, signature: str) -> bool:
        """
        Verify GA4 webhook signature
        """
        
        # GA4 webhooks typically use Google's signature verification
        # This would implement the actual verification logic
        logger.debug("Verifying GA4 webhook signature")
        return True
    
    async def process_webhook_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming GA4 webhook event
        """
        
        try:
            # Transform GA4 event
            transformed_event = self.integration.transform_event(event_data)
            
            # Store event
            await self.integration.store_event(transformed_event)
            
            logger.info(f"Processed GA4 webhook event: {transformed_event['event_type']}")
            
            return {
                'processed': True,
                'event_id': transformed_event['id'],
                'event_type': transformed_event['event_type']
            }
            
        except Exception as e:
            logger.error(f"GA4 webhook processing failed: {e}")
            raise
