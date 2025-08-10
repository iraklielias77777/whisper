"""
User Whisperer SDK Exceptions
"""


class UserWhispererError(Exception):
    """Base exception for User Whisperer SDK"""
    pass


class AuthenticationError(UserWhispererError):
    """Raised when authentication fails"""
    pass


class ValidationError(UserWhispererError):
    """Raised when data validation fails"""
    pass


class RateLimitError(UserWhispererError):
    """Raised when rate limit is exceeded"""
    
    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


class NetworkError(UserWhispererError):
    """Raised when network request fails"""
    
    def __init__(self, message, status_code=None, response_body=None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class ConfigurationError(UserWhispererError):
    """Raised when configuration is invalid"""
    pass


class QueueFullError(UserWhispererError):
    """Raised when event queue is full"""
    pass
