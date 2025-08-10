"""
User Whisperer Python SDK

A powerful Python library for integrating with the User Whisperer platform.
Track user events, analyze behavior, and deliver personalized experiences.
"""

from .__version__ import __version__
from .client import UserWhisperer
from .async_client import AsyncUserWhisperer
from .exceptions import (
    UserWhispererError,
    AuthenticationError,
    ValidationError,
    RateLimitError,
    NetworkError
)

__all__ = [
    '__version__',
    'UserWhisperer',
    'AsyncUserWhisperer',
    'UserWhispererError',
    'AuthenticationError',
    'ValidationError',
    'RateLimitError',
    'NetworkError'
]
