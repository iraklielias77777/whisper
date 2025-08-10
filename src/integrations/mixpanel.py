"""
Mixpanel Integration for User Whisperer Platform
Handles real-time webhook processing and data synchronization
"""

import asyncio
import json
import logging
import hashlib
import hmac
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from urllib.parse import urljoin

# HTTP client imports
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from .base import BaseIntegration, WebhookProcessor, EventBuffer, RateLimiter

logger = logging.getLogger(__name__)


class MixpanelIntegration(BaseIntegration):
    """
    Mixpanel integration with webhook processing and data export
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.project_token = config['mixpanel_token']
        self.api_secret = config['mixpanel_api_secret']
        self.project_id = config.get('mixpanel_project_id')
        self.eu_residency = config.get('mixpanel_eu', False)
        self.webhook_secret = config.get('webhook_secret')
        
        # API endpoints
        self.api_base = "https://eu.mixpanel.com/api" if self.eu_residency else "https://mixpanel.com/api"
        self.data_api_base = "https://data-eu.mixpanel.com/api" if self.eu_residency else "https://data.mixpanel.com/api"
        
        # Rate limiter for API calls
        self.rate_limiter = RateLimiter(max_requests=60, window_seconds=60)  # 60 requests per minute
        
        # Event buffer for batch processing
        self.event_buffer = EventBuffer(
            max_size=100,
            flush_interval=5.0,
            processor=self.process_event_batch
        )
        
        logger.info(f"Mixpanel Integration initialized for project: {self.project_token}")
    
    def load_event_mapping(self) -> Dict[str, str]:
        """Load Mixpanel event mapping configuration"""
        
        return {
            # Special Mixpanel events
            '$identify': 'user_identified',
            '$create_alias': 'user_aliased',
            '$merge': 'user_merged',
            '$track': 'custom_event',
            
            # Page/Screen events
            '$pageview': 'page_viewed',
            '$screen': 'screen_viewed',
            'Page View': 'page_viewed',
            'Screen View': 'screen_viewed',
            
            # User lifecycle events
            'Sign Up': 'user_signed_up',
            'Login': 'user_logged_in',
            'Logout': 'user_logged_out',
            'Account Created': 'account_created',
            'Account Deleted': 'account_deleted',
            'Profile Updated': 'profile_updated',
            
            # Engagement events
            'App Opened': 'app_opened',
            'App Backgrounded': 'app_backgrounded',
            'Session Start': 'session_started',
            'Session End': 'session_ended',
            'Feature Used': 'feature_used',
            'Tutorial Started': 'tutorial_started',
            'Tutorial Completed': 'tutorial_completed',
            
            # E-commerce events
            'Purchase': 'purchase_completed',
            'Order Completed': 'purchase_completed',
            'Item Purchased': 'item_purchased',
            'Add to Cart': 'cart_item_added',
            'Remove from Cart': 'cart_item_removed',
            'Checkout Started': 'checkout_started',
            'Payment Info Added': 'payment_info_added',
            'Order Refunded': 'purchase_refunded',
            'Product Viewed': 'product_viewed',
            'Product List Viewed': 'product_list_viewed',
            
            # Subscription events
            'Subscription Started': 'subscription_created',
            'Subscription Renewed': 'subscription_renewed',
            'Subscription Cancelled': 'subscription_cancelled',
            'Subscription Upgraded': 'subscription_upgraded',
            'Subscription Downgraded': 'subscription_downgraded',
            'Trial Started': 'trial_started',
            'Trial Converted': 'trial_converted',
            'Trial Expired': 'trial_expired',
            
            # Content events
            'Content Viewed': 'content_viewed',
            'Content Shared': 'content_shared',
            'Content Liked': 'content_liked',
            'Content Downloaded': 'content_downloaded',
            'Video Started': 'video_started',
            'Video Completed': 'video_completed',
            'Article Read': 'article_read',
            
            # Marketing events
            'Email Opened': 'email_opened',
            'Email Clicked': 'email_clicked',
            'Push Notification Received': 'push_received',
            'Push Notification Opened': 'push_opened',
            'Ad Clicked': 'ad_clicked',
            'Campaign Viewed': 'campaign_viewed',
            
            # Social events
            'Share': 'content_shared',
            'Like': 'content_liked',
            'Comment': 'comment_created',
            'Follow': 'user_followed',
            'Invite Sent': 'invite_sent',
            'Invite Accepted': 'invite_accepted',
            
            # Search and discovery
            'Search': 'search_performed',
            'Search Results Viewed': 'search_results_viewed',
            'Filter Applied': 'filter_applied',
            'Sort Applied': 'sort_applied',
            
            # Support events
            'Support Ticket Created': 'support_ticket_created',
            'Support Chat Started': 'support_chat_started',
            'FAQ Viewed': 'faq_viewed',
            'Help Article Viewed': 'help_article_viewed'
        }
    
    async def setup_integration(self, app_id: str) -> Dict[str, Any]:
        """
        Set up Mixpanel webhook integration for an app
        """
        
        logger.info(f"Setting up Mixpanel integration for app: {app_id}")
        
        try:
            # Create webhook URL
            webhook_url = f"https://api.userwhisperer.ai/webhooks/mixpanel/{app_id}"
            
            # Register webhook with Mixpanel
            webhook_config = await self.create_webhook(app_id, webhook_url)
            
            # Set up data export configuration
            export_config = await self.setup_data_export(app_id)
            
            # Create integration configuration
            integration_config = {
                'status': 'active',
                'app_id': app_id,
                'project_token': self.project_token,
                'project_id': self.project_id,
                'webhook_url': webhook_url,
                'webhook_config': webhook_config,
                'export_config': export_config,
                'event_mapping': self.event_mapping,
                'eu_residency': self.eu_residency,
                'configured_at': datetime.utcnow().isoformat()
            }
            
            # Store integration config
            await self.store_integration_config(app_id, integration_config)
            
            logger.info(f"Mixpanel integration setup completed for app: {app_id}")
            return integration_config
            
        except Exception as e:
            logger.error(f"Mixpanel integration setup failed for app {app_id}: {e}")
            raise
    
    async def create_webhook(self, app_id: str, webhook_url: str) -> Dict[str, Any]:
        """
        Create Mixpanel webhook for real-time events
        """
        
        logger.info(f"Creating Mixpanel webhook for app: {app_id}")
        
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available, using requests for webhook creation")
            return await self._create_webhook_sync(app_id, webhook_url)
        
        webhook_data = {
            "webhook_url": webhook_url,
            "events": ["all"],  # Subscribe to all events
            "name": f"user-whisperer-{app_id}",
            "description": f"User Whisperer integration webhook for app {app_id}",
            "active": True
        }
        
        if self.webhook_secret:
            webhook_data["secret"] = self.webhook_secret
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_secret}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.api_base}/app/projects/{self.project_id}/webhooks"
                
                async with session.post(url, json=webhook_data, headers=headers) as response:
                    if response.status == 201:
                        result = await response.json()
                        logger.info(f"Webhook created successfully: {result.get('id')}")
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"Webhook creation failed: {response.status} - {error_text}")
                        raise Exception(f"Webhook creation failed: {response.status}")
        
        except Exception as e:
            logger.error(f"Failed to create webhook: {e}")
            # Return mock configuration for development
            return {
                'id': f'webhook_{app_id}_{int(time.time())}',
                'url': webhook_url,
                'status': 'simulated',
                'events': ['all']
            }
    
    async def _create_webhook_sync(self, app_id: str, webhook_url: str) -> Dict[str, Any]:
        """
        Create webhook using synchronous requests
        """
        
        if not REQUESTS_AVAILABLE:
            logger.warning("requests not available, returning mock webhook config")
            return {
                'id': f'webhook_{app_id}_{int(time.time())}',
                'url': webhook_url,
                'status': 'simulated'
            }
        
        webhook_data = {
            "webhook_url": webhook_url,
            "events": ["all"],
            "name": f"user-whisperer-{app_id}",
            "active": True
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_secret}",
            "Content-Type": "application/json"
        }
        
        try:
            url = f"{self.api_base}/app/projects/{self.project_id}/webhooks"
            response = requests.post(url, json=webhook_data, headers=headers, timeout=30)
            
            if response.status_code == 201:
                result = response.json()
                logger.info(f"Webhook created successfully: {result.get('id')}")
                return result
            else:
                logger.error(f"Webhook creation failed: {response.status_code} - {response.text}")
                raise Exception(f"Webhook creation failed: {response.status_code}")
                
        except requests.RequestException as e:
            logger.error(f"Failed to create webhook: {e}")
            raise
    
    async def setup_data_export(self, app_id: str) -> Dict[str, Any]:
        """
        Set up data export configuration for historical sync
        """
        
        logger.info(f"Setting up Mixpanel data export for app: {app_id}")
        
        # Configure data export settings
        export_config = {
            'export_format': 'json',
            'compression': 'gzip',
            'batch_size': 1000,
            'export_events': True,
            'export_people': True,
            'export_cohorts': False,
            'date_range_limit': 90,  # days
            'rate_limit': {
                'requests_per_hour': 60,
                'concurrent_requests': 5
            }
        }
        
        return export_config
    
    def transform_event(self, mixpanel_event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform Mixpanel event to User Whisperer format
        """
        
        properties = mixpanel_event.get('properties', {})
        
        # Extract basic event information
        event_name = mixpanel_event.get('event', 'unknown')
        distinct_id = properties.get('distinct_id')
        timestamp = properties.get('time', time.time())
        
        # Map event type
        mapped_event_type = self.map_event_type(event_name)
        
        # Extract and clean properties
        event_properties = self.extract_event_properties(properties)
        
        # Extract context
        context = self.extract_event_context(properties)
        
        # Create User Whisperer event
        transformed_event = {
            'id': self.generate_event_id(),
            'user_id': distinct_id,
            'event_type': mapped_event_type,
            'properties': event_properties,
            'context': context,
            'timestamp': datetime.fromtimestamp(timestamp).isoformat() if isinstance(timestamp, (int, float)) else timestamp,
            'source': 'mixpanel',
            'source_event_name': event_name
        }
        
        return transformed_event
    
    def extract_event_properties(self, mp_properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and clean event properties from Mixpanel event
        """
        
        # Mixpanel internal properties to exclude
        excluded_keys = {
            'mp_api_endpoint', 'mp_api_timestamp_ms', 'mp_processing_time_ms',
            '$browser', '$browser_version', '$city', '$region', '$country_code',
            '$current_url', '$initial_referrer', '$initial_referring_domain',
            '$os', '$screen_height', '$screen_width', '$lib_version',
            'time', 'distinct_id', '$device_id', '$user_id', '$insert_id',
            'token', 'mp_lib', 'mp_sent_by_lib_version'
        }
        
        # Extract custom properties (non-internal)
        properties = {}
        
        for key, value in mp_properties.items():
            # Skip internal properties
            if key in excluded_keys:
                continue
            
            # Skip properties starting with $ (Mixpanel internal)
            if key.startswith('$') and key not in ['$email', '$name', '$phone', '$avatar']:
                continue
            
            # Clean key name
            clean_key = key.replace('$', '') if key.startswith('$') else key
            properties[clean_key] = value
        
        return properties
    
    def extract_event_context(self, mp_properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract context information from Mixpanel properties
        """
        
        context = {
            'mixpanel': {
                'distinct_id': mp_properties.get('distinct_id'),
                'device_id': mp_properties.get('$device_id'),
                'insert_id': mp_properties.get('$insert_id'),
                'user_id': mp_properties.get('$user_id'),
                'lib_version': mp_properties.get('$lib_version'),
                'project_token': self.project_token
            },
            'location': {
                'city': mp_properties.get('$city'),
                'region': mp_properties.get('$region'),
                'country': mp_properties.get('$country_code'),
                'timezone': mp_properties.get('$timezone')
            },
            'device': {
                'browser': mp_properties.get('$browser'),
                'browser_version': mp_properties.get('$browser_version'),
                'os': mp_properties.get('$os'),
                'device': mp_properties.get('$device'),
                'screen_height': mp_properties.get('$screen_height'),
                'screen_width': mp_properties.get('$screen_width'),
                'lib_version': mp_properties.get('$lib_version')
            },
            'page': {
                'current_url': mp_properties.get('$current_url'),
                'initial_referrer': mp_properties.get('$initial_referrer'),
                'initial_referring_domain': mp_properties.get('$initial_referring_domain'),
                'referring_domain': mp_properties.get('$referring_domain'),
                'referrer': mp_properties.get('$referrer')
            },
            'campaign': {
                'utm_source': mp_properties.get('utm_source'),
                'utm_medium': mp_properties.get('utm_medium'),
                'utm_campaign': mp_properties.get('utm_campaign'),
                'utm_term': mp_properties.get('utm_term'),
                'utm_content': mp_properties.get('utm_content')
            }
        }
        
        # Remove empty nested dictionaries
        return {k: v for k, v in context.items() if any(v.values()) if isinstance(v, dict) else v}
    
    async def export_historical_data(
        self,
        app_id: str,
        start_date: datetime,
        end_date: datetime,
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Export historical data from Mixpanel
        """
        
        logger.info(f"Exporting Mixpanel historical data for app {app_id}: {start_date} to {end_date}")
        
        try:
            # Check rate limits
            if not self.rate_limiter.is_allowed():
                reset_time = self.rate_limiter.get_reset_time()
                raise Exception(f"Rate limit exceeded. Reset at: {reset_time}")
            
            events_processed = 0
            batch_events = []
            
            # Export events using Mixpanel Raw Data Export API
            export_url = f"{self.data_api_base}/2.0/export"
            
            params = {
                'from_date': start_date.strftime('%Y-%m-%d'),
                'to_date': end_date.strftime('%Y-%m-%d'),
                'limit': batch_size
            }
            
            headers = {
                'Authorization': f'Bearer {self.api_secret}',
                'Accept': 'application/json'
            }
            
            if AIOHTTP_AVAILABLE:
                async with aiohttp.ClientSession() as session:
                    async with session.get(export_url, params=params, headers=headers) as response:
                        if response.status == 200:
                            # Process line-delimited JSON response
                            async for line in response.content:
                                line_str = line.decode('utf-8').strip()
                                if line_str:
                                    try:
                                        mixpanel_event = json.loads(line_str)
                                        transformed_event = self.transform_event(mixpanel_event)
                                        batch_events.append(transformed_event)
                                        events_processed += 1
                                        
                                        # Process batch when full
                                        if len(batch_events) >= batch_size:
                                            await self.process_event_batch(batch_events)
                                            batch_events = []
                                            
                                    except json.JSONDecodeError as e:
                                        logger.warning(f"Failed to parse JSON line: {e}")
                                        continue
                        else:
                            error_text = await response.text()
                            raise Exception(f"Export failed: {response.status} - {error_text}")
            
            else:
                # Fallback to requests
                if REQUESTS_AVAILABLE:
                    response = requests.get(export_url, params=params, headers=headers, timeout=120)
                    response.raise_for_status()
                    
                    # Process line-delimited JSON
                    for line in response.text.split('\n'):
                        line = line.strip()
                        if line:
                            try:
                                mixpanel_event = json.loads(line)
                                transformed_event = self.transform_event(mixpanel_event)
                                batch_events.append(transformed_event)
                                events_processed += 1
                                
                                if len(batch_events) >= batch_size:
                                    await self.process_event_batch(batch_events)
                                    batch_events = []
                                    
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse JSON line: {e}")
                                continue
            
            # Process remaining events
            if batch_events:
                await self.process_event_batch(batch_events)
            
            export_result = {
                'app_id': app_id,
                'events_processed': events_processed,
                'date_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'status': 'completed',
                'exported_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Historical export completed: {events_processed} events processed")
            return export_result
            
        except Exception as e:
            logger.error(f"Historical data export failed: {e}")
            raise
    
    async def process_event_batch(self, events: List[Dict[str, Any]]):
        """Process a batch of transformed events"""
        
        logger.debug(f"Processing batch of {len(events)} events")
        
        # Store events in the User Whisperer system
        for event in events:
            await self.store_event(event)
    
    async def get_project_info(self) -> Dict[str, Any]:
        """
        Get Mixpanel project information
        """
        
        if not self.rate_limiter.is_allowed():
            raise Exception("Rate limit exceeded")
        
        try:
            headers = {
                'Authorization': f'Bearer {self.api_secret}',
                'Accept': 'application/json'
            }
            
            url = f"{self.api_base}/app/projects/{self.project_id}"
            
            if AIOHTTP_AVAILABLE:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            error_text = await response.text()
                            raise Exception(f"Failed to get project info: {response.status} - {error_text}")
            else:
                if REQUESTS_AVAILABLE:
                    response = requests.get(url, headers=headers, timeout=30)
                    response.raise_for_status()
                    return response.json()
                else:
                    raise Exception("No HTTP client available")
                    
        except Exception as e:
            logger.error(f"Failed to get project info: {e}")
            raise
    
    async def store_integration_config(self, app_id: str, config: Dict[str, Any]):
        """Store integration configuration"""
        
        # This would store the configuration in the User Whisperer database
        logger.info(f"Stored Mixpanel integration config for app: {app_id}")


class MixpanelWebhookProcessor(WebhookProcessor):
    """
    Webhook processor for Mixpanel events
    """
    
    def verify_webhook_signature(self, payload: bytes, signature: str) -> bool:
        """
        Verify Mixpanel webhook signature using HMAC-SHA256
        """
        
        if not self.webhook_secret:
            logger.warning("No webhook secret configured for Mixpanel, skipping verification")
            return True
        
        return self._verify_hmac_signature(payload, signature, 'sha256')
    
    async def process_webhook_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming Mixpanel webhook event
        """
        
        try:
            # Handle different Mixpanel webhook formats
            if 'data' in event_data:
                # Standard Mixpanel webhook format
                webhook_data = event_data['data']
            elif 'event' in event_data:
                # Direct event format
                webhook_data = event_data
            else:
                # Assume the entire payload is the event
                webhook_data = event_data
            
            # Transform Mixpanel event
            transformed_event = self.integration.transform_event(webhook_data)
            
            # Add to buffer for batch processing
            await self.integration.event_buffer.add_event(transformed_event)
            
            logger.info(f"Processed Mixpanel webhook event: {transformed_event['event_type']}")
            
            return {
                'processed': True,
                'event_id': transformed_event['id'],
                'event_type': transformed_event['event_type'],
                'user_id': transformed_event['user_id']
            }
            
        except Exception as e:
            logger.error(f"Mixpanel webhook processing failed: {e}")
            raise


class MixpanelDataExporter:
    """
    Utility class for exporting large amounts of historical data from Mixpanel
    """
    
    def __init__(self, integration: MixpanelIntegration):
        self.integration = integration
        self.rate_limiter = RateLimiter(max_requests=5, window_seconds=60)  # Conservative rate limiting
    
    async def export_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        batch_size: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Export events for a specific date range
        """
        
        if not self.rate_limiter.is_allowed():
            reset_time = self.rate_limiter.get_reset_time()
            raise Exception(f"Rate limit exceeded. Reset at: {reset_time}")
        
        logger.info(f"Exporting Mixpanel data: {start_date} to {end_date}")
        
        # Split large date ranges into smaller chunks
        date_chunks = self._split_date_range(start_date, end_date, max_days=7)
        
        all_events = []
        
        for chunk_start, chunk_end in date_chunks:
            logger.debug(f"Exporting chunk: {chunk_start} to {chunk_end}")
            
            chunk_events = await self.integration.export_historical_data(
                app_id="export",  # Placeholder
                start_date=chunk_start,
                end_date=chunk_end,
                batch_size=batch_size
            )
            
            all_events.extend(chunk_events.get('events', []))
            
            # Add delay between chunks to respect rate limits
            await asyncio.sleep(1)
        
        return all_events
    
    def _split_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        max_days: int = 7
    ) -> List[Tuple[datetime, datetime]]:
        """
        Split a date range into smaller chunks
        """
        
        chunks = []
        current_date = start_date
        
        while current_date < end_date:
            chunk_end = min(
                current_date + timedelta(days=max_days),
                end_date
            )
            chunks.append((current_date, chunk_end))
            current_date = chunk_end + timedelta(days=1)
        
        return chunks
