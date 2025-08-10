# User Whisperer API Documentation

## Overview

The User Whisperer platform provides a comprehensive set of APIs for tracking user behavior, analyzing engagement patterns, and delivering personalized experiences. This documentation covers both REST and GraphQL APIs.

## Base URL

```
Production: https://api.userwhisperer.ai
Staging: https://staging-api.userwhisperer.ai
```

## Authentication

All API requests require authentication using an API key:

```http
Authorization: Bearer your_api_key_here
```

API keys can be obtained from the User Whisperer dashboard.

## Rate Limiting

API requests are rate-limited based on your subscription plan:

- **Free Tier**: 1,000 requests per hour
- **Pro Tier**: 5,000 requests per hour  
- **Enterprise**: 10,000 requests per hour

Rate limit headers are included in all responses:

```http
X-RateLimit-Limit: 5000
X-RateLimit-Remaining: 4999
X-RateLimit-Reset: 1640995200
```

## REST API

### Event Tracking

#### Track Single Event

```http
POST /v1/events/track
```

**Request Body:**
```json
{
  "user_id": "user_123",
  "event_type": "page_viewed",
  "properties": {
    "page": "/dashboard",
    "source": "web"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Response:**
```json
{
  "event_id": "evt_1642234200_abc123",
  "status": "tracked",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Track Batch Events

```http
POST /v1/events/batch
```

**Request Body:**
```json
{
  "events": [
    {
      "user_id": "user_123",
      "event_type": "page_viewed",
      "properties": {"page": "/dashboard"}
    },
    {
      "user_id": "user_456", 
      "event_type": "button_clicked",
      "properties": {"button": "signup"}
    }
  ]
}
```

**Response:**
```json
{
  "processed": 2,
  "failed": 0,
  "event_ids": ["evt_1642234200_abc123", "evt_1642234201_def456"]
}
```

### User Management

#### Identify User

```http
POST /v1/users/identify
```

**Request Body:**
```json
{
  "user_id": "user_123",
  "traits": {
    "email": "user@example.com",
    "name": "John Doe",
    "plan": "pro"
  }
}
```

#### Get User Profile

```http
GET /v1/users/{user_id}
```

**Response:**
```json
{
  "id": "user_123",
  "email": "user@example.com",
  "name": "John Doe",
  "lifecycle_stage": "engaged",
  "engagement_score": 0.85,
  "churn_risk_score": 0.15,
  "created_at": "2024-01-01T00:00:00Z",
  "last_active_at": "2024-01-15T10:30:00Z"
}
```

### Analytics

#### Get User Analysis

```http
GET /v1/users/{user_id}/analysis?include_predictions=true
```

**Response:**
```json
{
  "user_id": "user_123",
  "behavioral_metrics": {
    "session_frequency": 4.2,
    "avg_session_duration": 420,
    "feature_adoption_rate": 0.75
  },
  "predictions": {
    "churn_risk": 0.15,
    "ltv_prediction": 2450.00,
    "next_best_action": "upgrade_prompt"
  },
  "segments": ["power_users", "mobile_first"]
}
```

### Decisions

#### Get Decisions

```http
GET /v1/decisions?user_id=user_123&limit=10
```

#### Trigger Decision

```http
POST /v1/decisions/trigger
```

**Request Body:**
```json
{
  "user_id": "user_123",
  "intervention_type": "retention_campaign"
}
```

### Content & Messaging

#### Generate Content

```http
POST /v1/content/generate
```

**Request Body:**
```json
{
  "user_id": "user_123",
  "content_type": "email",
  "template_id": "welcome_series_1",
  "personalization_data": {
    "name": "John",
    "last_feature_used": "analytics_dashboard"
  }
}
```

#### Send Message

```http
POST /v1/messages/send
```

**Request Body:**
```json
{
  "user_id": "user_123",
  "channel": "email",
  "content": "Welcome to User Whisperer!",
  "template_id": "welcome_email",
  "schedule_at": "2024-01-15T15:00:00Z"
}
```

## GraphQL API

The GraphQL endpoint is available at `/graphql` with an interactive explorer at `/graphql` (when enabled).

### Example Queries

#### Get User with Events

```graphql
query GetUserWithEvents($userId: String!) {
  user(id: $userId) {
    id
    name
    email
    engagementScore
    churnRiskScore
    lifecycleStage
    lastActiveAt
  }
  
  events(filter: { userId: $userId }, limit: 10) {
    id
    eventType
    properties
    createdAt
  }
}
```

#### Track Event

```graphql
mutation TrackEvent($input: TrackEventInput!) {
  trackEvent(input: $input) {
    id
    eventType
    createdAt
  }
}
```

Variables:
```json
{
  "input": {
    "appId": "app_123",
    "userId": "user_456",
    "eventType": "feature_used",
    "properties": {"feature": "analytics"}
  }
}
```

#### Subscribe to Real-time Events

```graphql
subscription UserEvents($userId: String!) {
  userEvents(userId: $userId) {
    id
    eventType
    properties
    createdAt
  }
}
```

### GraphQL Schema

The complete GraphQL schema includes:

- **Types**: User, Event, Message, Decision, Campaign, Analytics, Cohort, Segment, MLModel, Prediction
- **Queries**: Get users, events, analytics, campaigns, etc.
- **Mutations**: Track events, send messages, create campaigns, etc.
- **Subscriptions**: Real-time updates for events, decisions, messages, analytics

## Webhooks

### Third-party Integration Webhooks

#### Google Analytics 4

```http
POST /webhooks/ga4/{app_id}
```

Receives GA4 events and transforms them into User Whisperer events.

#### Mixpanel

```http
POST /webhooks/mixpanel/{app_id}
```

Processes Mixpanel webhook events for real-time synchronization.

## Error Handling

All APIs return consistent error responses:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid event type provided",
    "details": {
      "field": "event_type",
      "value": "invalid_event"
    },
    "request_id": "req_1642234200_xyz789"
  }
}
```

### Common Error Codes

- `AUTHENTICATION_ERROR` (401): Invalid API key
- `AUTHORIZATION_ERROR` (403): Insufficient permissions
- `VALIDATION_ERROR` (400): Invalid request data
- `RATE_LIMIT_EXCEEDED` (429): Too many requests
- `NOT_FOUND` (404): Resource not found
- `INTERNAL_ERROR` (500): Server error

## SDKs

### JavaScript SDK

```javascript
import UserWhisperer from '@userwhisperer/sdk-js';

const uw = new UserWhisperer({
  apiKey: 'your_api_key',
  appId: 'your_app_id'
});

// Track event
uw.track('page_viewed', {
  page: '/dashboard',
  source: 'web'
});

// Identify user
uw.identify('user_123', {
  email: 'user@example.com',
  name: 'John Doe'
});
```

### Python SDK

```python
from userwhisperer import UserWhisperer

uw = UserWhisperer(
    api_key='your_api_key',
    app_id='your_app_id'
)

# Track event
uw.track('page_viewed', {
    'page': '/dashboard',
    'source': 'web'
}, user_id='user_123')

# Identify user
uw.identify('user_123', {
    'email': 'user@example.com',
    'name': 'John Doe'
})
```

## Best Practices

### Event Tracking

1. **Use consistent event naming**: Use snake_case and descriptive names
2. **Include relevant properties**: Add context that will be useful for analysis
3. **Batch events when possible**: Use batch endpoints for better performance
4. **Handle failures gracefully**: Implement retry logic for failed requests

### Rate Limiting

1. **Implement exponential backoff**: Wait progressively longer between retries
2. **Monitor rate limit headers**: Adjust request frequency based on remaining quota
3. **Use batch endpoints**: Reduce request count by batching operations

### Security

1. **Keep API keys secure**: Never expose API keys in client-side code
2. **Use HTTPS**: Always use secure connections
3. **Rotate keys regularly**: Update API keys periodically
4. **Implement proper authentication**: Verify API keys on server-side

### Performance

1. **Cache responses**: Cache analytics and user data when appropriate
2. **Use GraphQL efficiently**: Request only needed fields
3. **Implement pagination**: Use limit/offset for large datasets
4. **Monitor response times**: Track API performance and optimize accordingly

## Support

For API support, please contact:

- Email: api-support@userwhisperer.ai
- Documentation: https://docs.userwhisperer.ai
- Status Page: https://status.userwhisperer.ai
- GitHub: https://github.com/userwhisperer/api-issues
