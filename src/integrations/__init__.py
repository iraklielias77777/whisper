# Third-Party Integrations for User Whisperer Platform
"""
Integration modules for connecting with external analytics and data platforms.
Supports real-time data synchronization and webhook processing.
"""

__version__ = "1.0.0"

# Google Analytics 4 integration
try:
    from .google_analytics import GA4Integration, GA4WebhookProcessor
except ImportError as e:
    print(f"Warning: GA4 integration not available: {e}")

# Mixpanel integration
try:
    from .mixpanel import MixpanelIntegration, MixpanelWebhookProcessor
except ImportError as e:
    print(f"Warning: Mixpanel integration not available: {e}")

# Segment integration
try:
    from .segment import SegmentIntegration
except ImportError as e:
    print(f"Warning: Segment integration not available: {e}")

# Facebook/Meta integration
try:
    from .facebook import FacebookIntegration
except ImportError as e:
    print(f"Warning: Facebook integration not available: {e}")

# Base integration class
from .base import BaseIntegration, WebhookProcessor

__all__ = [
    '__version__',
    'BaseIntegration',
    'WebhookProcessor',
    'GA4Integration',
    'GA4WebhookProcessor',
    'MixpanelIntegration',
    'MixpanelWebhookProcessor',
    'SegmentIntegration',
    'FacebookIntegration'
]
