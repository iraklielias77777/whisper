# User Whisperer Platform

An autonomous AI-powered customer engagement platform that revolutionizes how subscription applications communicate with their users.

## Overview

The User Whisperer Platform treats each user as a unique individual rather than a member of a segment. Every user has their own:

- **Behavioral Model**: Continuously learning patterns unique to that user
- **Communication Preferences**: Discovered through experimentation and observation  
- **Engagement History**: Detailed tracking of what works for this specific user
- **Optimization Strategy**: Personalized approach to maximize engagement

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Whisperer Platform                      │
├─────────────────────────────────────────────────────────────────┤
│  Client SDK → API Gateway → Event Ingestion → Event Stream      │
│                                   ↓                              │
│  Behavioral Analysis ← → Decision Engine → Content Generator     │
│                                   ↓                              │
│              Channel Orchestrator → External Channels           │
└─────────────────────────────────────────────────────────────────┘
```

## Core Objectives

- 25% improvement in user retention within 30 days
- 20% increase in trial-to-paid conversion  
- 35% reduction in churn through predictive intervention
- 2x industry-average engagement rates
- 90% reduction in manual campaign management

## Technical Requirements

- Process 100,000+ events per second with <50ms latency
- Support 1 million+ concurrent users per client
- 99.95% uptime guarantee  
- Generate personalized content in <2 seconds
- Make intervention decisions in <100ms

## Quick Start

### Prerequisites

- Node.js 20+
- Python 3.11+
- Go 1.21+
- PostgreSQL 15+
- Redis 7+
- Docker & Docker Compose

### Setup

```bash
# Clone repository
git clone https://github.com/userwhisperer/platform.git
cd user-whisperer

# Setup development environment
make setup

# Start local services
make dev

# Run tests
make test

# Deploy to staging
make deploy-staging
```

## Services

### Event Ingestion Service (Node.js/TypeScript)
- High-throughput event processing
- Real-time deduplication  
- Schema validation
- Event enrichment

### Behavioral Analysis Engine (Python)
- User behavior pattern detection
- Churn risk calculation
- Engagement scoring
- Lifecycle classification

### Decision Engine (Python)  
- ML-powered intervention decisions
- Communication fatigue management
- Channel optimization
- Timing optimization

### Content Generation Service (Python)
- LLM-powered personalization
- A/B testing integration
- Compliance checking
- Multi-language support

### Channel Orchestration Service (Go)
- Multi-channel delivery
- Rate limiting
- Delivery tracking
- Failure handling

## Development

### Repository Structure

```
user-whisperer/
├── services/          # Microservices
├── shared/           # Common code
├── infrastructure/   # K8s, Terraform
├── ml-models/       # ML training & serving
├── sdk/             # Client libraries
├── docs/            # Documentation
└── scripts/         # Automation
```

### Technology Stack

- **Languages**: TypeScript, Python, Go
- **Databases**: PostgreSQL, Redis, BigQuery  
- **Message Queue**: Google Pub/Sub
- **ML**: TensorFlow, XGBoost, Scikit-learn
- **LLM**: OpenAI GPT-4, Anthropic Claude
- **Infrastructure**: Kubernetes, Docker, Terraform
- **Monitoring**: Prometheus, Grafana, Jaeger

### Testing Strategy

- **Unit Tests**: 70% - Business logic, transformations
- **Integration Tests**: 25% - Service interactions, DB operations  
- **E2E Tests**: 5% - Critical user journeys

## API Documentation

API documentation is available at `/docs/api/` and served at `http://localhost:3000/docs` in development.

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/JIRA-123-description`)
3. Commit changes (`git commit -am 'feat(scope): description'`)
4. Push branch (`git push origin feature/JIRA-123-description`)
5. Create Pull Request

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**: feat, fix, docs, style, refactor, test, chore

## Security

- All API endpoints require authentication
- Event data is encrypted at rest and in transit
- PII data is automatically anonymized
- GDPR compliance built-in
- SOC 2 Type II certified

## License

Proprietary - Internal Use Only

## Support

- **Documentation**: [Internal Wiki](https://wiki.company.com/user-whisperer)
- **Slack**: #user-whisperer-support
- **On-call**: Use PagerDuty for production issues
```

Now let me create the package.json for the root project:

```json:user-whisperer/package.json
{
  "name": "user-whisperer-platform",
  "version": "1.0.0",
  "description": "Autonomous AI-powered customer engagement platform",
  "main": "index.js",
  "private": true,
  "workspaces": [
    "services/*",
    "sdk/*",
    "shared/*"
  ],
  "scripts": {
    "setup": "scripts/setup.sh",
    "dev": "docker-compose up -d",
    "dev:logs": "docker-compose logs -f",
    "dev:stop": "docker-compose down",
    "dev:clean": "docker-compose down -v",
    "test": "npm run test:unit && npm run test:integration",
    "test:unit": "npm run test --workspaces",
    "test:integration": "scripts/test-integration.sh",
    "test:e2e": "scripts/test-e2e.sh",
    "build": "npm run build --workspaces",
    "build:docker": "scripts/build-docker.sh",
    "lint": "npm run lint --workspaces",
    "lint:fix": "npm run lint:fix --workspaces",
    "typecheck": "npm run typecheck --workspaces",
    "deploy:staging": "scripts/deploy.sh staging",
    "deploy:production": "scripts/deploy.sh production",
    "migrate:dev": "cd services/event-ingestion && npm run migrate",
    "migrate:staging": "kubectl exec -it deployment/postgres -- psql -f migrations/latest.sql",
    "seed:dev": "scripts/seed-dev-data.sh",
    "logs:dev": "kubectl logs -f -l app=user-whisperer -n development",
    "logs:staging": "kubectl logs -f -l app=user-whisperer -n staging",
    "logs:production": "kubectl logs -f -l app=user-whisperer -n production"
  },
  "engines": {
    "node": ">=20.0.0",
    "npm": ">=10.0.0"
  },
  "devDependencies": {
    "@typescript-eslint/eslint-plugin": "^6.0.0",
    "@typescript-eslint/parser": "^6.0.0",
    "eslint": "^8.50.0",
    "eslint-config-prettier": "^9.0.0",
    "eslint-plugin-prettier": "^5.0.0",
    "husky": "^8.0.3",
    "lint-staged": "^14.0.1",
    "prettier": "^3.0.3",
    "typescript": "^5.3.0"
  },
  "lint-staged": {
    "*.{js,ts,jsx,tsx}": [
      "eslint --fix",
      "prettier --write"
    ],
    "*.{json,md,yml,yaml}": [
      "prettier --write"
    ]
  },
  "husky": {
    "hooks": {
      "pre-commit": "lint-staged",
      "commit-msg": "commitlint -E HUSKY_GIT_PARAMS"
    }
  },
  "repository": {
    "type": "git",
    "url": "https://github.com/userwhisperer/platform.git"
  },
  "keywords": [
    "ai",
    "machine-learning",
    "customer-engagement",
    "personalization",
    "automation",
    "saas"
  ],
  "author": "User Whisperer Team",
  "license": "PROPRIETARY"
}
```

Now let me create the Docker Compose configuration for local development:

```yaml:user-whisperer/docker-compose.yml
version: '3.8'

services:
  # Databases
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: userwhisperer_dev
      POSTGRES_USER: uwdev
      POSTGRES_PASSWORD: localdev123
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./infrastructure/sql/init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U uwdev -d userwhisperer_dev"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Message Queue (Local Pub/Sub Emulator)
  pubsub-emulator:
    image: gcr.io/google.com/cloudsdktool/cloud-sdk:latest
    command: >
      sh -c "gcloud beta emulators pubsub start 
             --project=user-whisperer-dev 
             --host-port=0.0.0.0:8085"
    ports:
      - "8085:8085"
    environment:
      PUBSUB_PROJECT_ID: user-whisperer-dev

  # API Gateway (Kong)
  kong:
    image: kong:3.4-alpine
    environment:
      KONG_DATABASE: "off"
      KONG_DECLARATIVE_CONFIG: /kong/kong.yml
      KONG_PROXY_ACCESS_LOG: /dev/stdout
      KONG_ADMIN_ACCESS_LOG: /dev/stdout
      KONG_PROXY_ERROR_LOG: /dev/stderr
      KONG_ADMIN_ERROR_LOG: /dev/stderr
      KONG_ADMIN_LISTEN: "0.0.0.0:8001"
    ports:
      - "8000:8000"
      - "8001:8001"
    volumes:
      - ./infrastructure/kong/kong.yml:/kong/kong.yml
    depends_on:
      - event-ingestion
      - api-server

  # Core Services
  event-ingestion:
    build:
      context: ./services/event-ingestion
      dockerfile: Dockerfile
    environment:
      - NODE_ENV=development
      - POSTGRES_URL=postgresql://uwdev:localdev123@postgres:5432/userwhisperer_dev
      - REDIS_URL=redis://redis:6379
      - PUBSUB_EMULATOR_HOST=pubsub-emulator:8085
      - LOG_LEVEL=debug
    ports:
      - "3001:3001"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      pubsub-emulator:
        condition: service_started
    volumes:
      - ./services/event-ingestion:/app
      - /app/node_modules
    command: npm run dev

  behavioral-analysis:
    build:
      context: ./services/behavioral-analysis
      dockerfile: Dockerfile
    environment:
      - POSTGRES_URL=postgresql://uwdev:localdev123@postgres:5432/userwhisperer_dev
      - REDIS_URL=redis://redis:6379
      - PUBSUB_EMULATOR_HOST=pubsub-emulator:8085
      - LOG_LEVEL=debug
    ports:
      - "3002:3002"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./services/behavioral-analysis:/app
      - ./ml-models:/app/models
    command: python -m uvicorn main:app --host 0.0.0.0 --port 3002 --reload

  decision-engine:
    build:
      context: ./services/decision-engine
      dockerfile: Dockerfile
    environment:
      - POSTGRES_URL=postgresql://uwdev:localdev123@postgres:5432/userwhisperer_dev
      - REDIS_URL=redis://redis:6379
      - PUBSUB_EMULATOR_HOST=pubsub-emulator:8085
      - LOG_LEVEL=debug
    ports:
      - "3003:3003"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./services/decision-engine:/app
      - ./ml-models:/app/models
    command: python -m uvicorn main:app --host 0.0.0.0 --port 3003 --reload

  content-generator:
    build:
      context: ./services/content-generator
      dockerfile: Dockerfile
    environment:
      - POSTGRES_URL=postgresql://uwdev:localdev123@postgres:5432/userwhisperer_dev
      - REDIS_URL=redis://redis:6379
      - PUBSUB_EMULATOR_HOST=pubsub-emulator:8085
      - OPENAI_API_KEY=${OPENAI_API_KEY:-sk-dev-mock}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-sk-ant-dev-mock}
      - LOG_LEVEL=debug
    ports:
      - "3004:3004"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./services/content-generator:/app
    command: python -m uvicorn main:app --host 0.0.0.0 --port 3004 --reload

  channel-orchestrator:
    build:
      context: ./services/channel-orchestrator
      dockerfile: Dockerfile
    environment:
      - POSTGRES_URL=postgresql://uwdev:localdev123@postgres:5432/userwhisperer_dev
      - REDIS_URL=redis://redis:6379
      - PUBSUB_EMULATOR_HOST=pubsub-emulator:8085
      - SENDGRID_API_KEY=${SENDGRID_API_KEY:-SG.dev.mock}
      - TWILIO_ACCOUNT_SID=${TWILIO_ACCOUNT_SID:-ACdev}
      - TWILIO_AUTH_TOKEN=${TWILIO_AUTH_TOKEN:-devtoken}
      - LOG_LEVEL=debug
    ports:
      - "3005:3005"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./services/channel-orchestrator:/app
    command: air -c .air.toml

  api-server:
    build:
      context: ./services/api-server
      dockerfile: Dockerfile
    environment:
      - NODE_ENV=development
      - POSTGRES_URL=postgresql://uwdev:localdev123@postgres:5432/userwhisperer_dev
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL=debug
    ports:
      - "3000:3000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./services/api-server:/app
      - /app/node_modules
    command: npm run dev

  # Monitoring & Observability
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./infrastructure/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3006:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./infrastructure/monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./infrastructure/monitoring/grafana/datasources:/etc/grafana/provisioning/datasources

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"
      - "14268:14268"
    environment:
      - COLLECTOR_OTLP_ENABLED=true

volumes:
  postgres_data:
  redis_data:
  grafana_data:

networks:
  default:
    name: user-whisperer-dev
```

Now let me create the Makefile for build automation:

```makefile:user-whisperer/Makefile
.PHONY: help setup dev dev-stop dev-clean test build deploy docs

# Default target
help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Development Environment
setup: ## Setup development environment
	@echo "Setting up User Whisperer development environment..."
	@scripts/setup.sh

dev: ## Start development environment
	@echo "Starting development services..."
	docker-compose up -d
	@echo "Services started. Access points:"
	@echo "  API Gateway: http://localhost:8000"
	@echo "  API Server: http://localhost:3000"
	@echo "  Grafana: http://localhost:3006 (admin/admin)"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Jaeger: http://localhost:16686"

dev-logs: ## Show development logs
	docker-compose logs -f

dev-stop: ## Stop development environment
	docker-compose down

dev-clean: ## Clean development environment (removes volumes)
	docker-compose down -v
	docker system prune -f

# Testing
test: ## Run all tests
	@echo "Running unit tests..."
	@npm run test:unit
	@echo "Running integration tests..."
	@npm run test:integration

test-unit: ## Run unit tests only
	npm run test:unit

test-integration: ## Run integration tests
	npm run test:integration

test-e2e: ## Run end-to-end tests
	npm run test:e2e

test-load: ## Run load tests
	@scripts/test-load.sh

# Building
build: ## Build all services
	npm run build

build-docker: ## Build Docker images
	@scripts/build-docker.sh

# Code Quality
lint: ## Run linter
	npm run lint

lint-fix: ## Fix linting issues
	npm run lint:fix

typecheck: ## Run TypeScript type checking
	npm run typecheck

format: ## Format code
	prettier --write "**/*.{js,ts,json,md,yml,yaml}"

# Database
migrate: ## Run database migrations
	@scripts/migrate.sh

migrate-create: ## Create new migration
	@read -p "Migration name: " name; \
	cd services/event-ingestion && npm run migrate:create -- $$name

seed: ## Seed development database
	@scripts/seed-dev-data.sh

# Deployment
deploy-staging: ## Deploy to staging
	@scripts/deploy.sh staging

deploy-production: ## Deploy to production
	@scripts/deploy.sh production

# Documentation
docs: ## Generate documentation
	@echo "Generating API documentation..."
	@npm run docs:generate
	@echo "Documentation available at http://localhost:3000/docs"

docs-serve: ## Serve documentation locally
	@cd docs && python -m http.server 8080

# Monitoring
logs-dev: ## Show development logs
	kubectl logs -f -l app=user-whisperer -n development

logs-staging: ## Show staging logs
	kubectl logs -f -l app=user-whisperer -n staging

logs-production: ## Show production logs
	kubectl logs -f -l app=user-whisperer -n production

metrics: ## Open metrics dashboard
	@open http://localhost:3006

# Utilities
shell-postgres: ## Connect to PostgreSQL
	docker-compose exec postgres psql -U uwdev -d userwhisperer_dev

shell-redis: ## Connect to Redis
	docker-compose exec redis redis-cli

clean: ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	@find . -name "node_modules" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@find . -name "dist" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "build" -type d -exec rm -rf {} + 2>/dev/null || true

# Security
security-scan: ## Run security scans
	@scripts/security-scan.sh

vulnerability-check: ## Check for vulnerabilities
	npm audit
	@cd services/behavioral-analysis && safety check
	@cd services/decision-engine && safety check
	@cd services/content-generator && safety check

# Performance
benchmark: ## Run performance benchmarks
	@scripts/benchmark.sh

profile: ## Profile application performance
	@scripts/profile.sh

# Backup & Recovery
backup-dev: ## Backup development database
	@scripts/backup.sh development

backup-staging: ## Backup staging database
	@scripts/backup.sh staging

restore-dev: ## Restore development database
	@scripts/restore.sh development

# Installation helpers
install-tools: ## Install required development tools
	@scripts/install-tools.sh

check-deps: ## Check all dependencies are installed
	@scripts/check-dependencies.sh

update-deps: ## Update all dependencies
	@npm update --workspaces
	@cd services/behavioral-analysis && pip install -r requirements.txt --upgrade
	@cd services/decision-engine && pip install -r requirements.txt --upgrade
	@cd services/content-generator && pip install -r requirements.txt --upgrade

# Environment Management
env-dev: ## Switch to development environment
	@echo "Switching to development environment..."
	@cp .env.development .env

env-staging: ## Switch to staging environment
	@echo "Switching to staging environment..."
	@cp .env.staging .env

env-production: ## Switch to production environment
	@echo "Switching to production environment..."
	@cp .env.production .env
```

Now let me create the environment configuration files:

```bash:user-whisperer/.env.development
# Environment
NODE_ENV=development
LOG_LEVEL=debug
ENVIRONMENT=development

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=userwhisperer_dev
POSTGRES_USER=uwdev
POSTGRES_PASSWORD=localdev123
POSTGRES_URL=postgresql://uwdev:localdev123@localhost:5432/userwhisperer_dev

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_URL=redis://localhost:6379

# Services
API_PORT=3000
INGESTION_PORT=3001
BEHAVIORAL_PORT=3002
DECISION_PORT=3003
CONTENT_PORT=3004
ORCHESTRATOR_PORT=3005

# Message Queue (Local Pub/Sub Emulator)
PUBSUB_EMULATOR_HOST=localhost:8085
PUBSUB_PROJECT_ID=user-whisperer-dev

# External APIs (Development)
OPENAI_API_KEY=sk-dev-mock-key-for-local-testing
ANTHROPIC_API_KEY=sk-ant-dev-mock-key-for-local-testing
SENDGRID_API_KEY=SG.dev.mock-key-for-local-testing
TWILIO_ACCOUNT_SID=ACdev_mock_account_sid
TWILIO_AUTH_TOKEN=dev_mock_auth_token
FIREBASE_PROJECT_ID=uw-dev

# Feature Flags
ENABLE_ML_MODELS=false
ENABLE_LLM_GENERATION=false
USE_MOCK_SERVICES=true
ENABLE_DETAILED_LOGGING=true
ENABLE_PERFORMANCE_MONITORING=true

# Security
JWT_SECRET=dev-jwt-secret-key-not-for-production
API_KEY_SALT=dev-api-key-salt
ENCRYPTION_KEY=dev-encryption-key-32-chars-long

# Rate Limiting
RATE_LIMIT_WINDOW_MS=60000
RATE_LIMIT_MAX_REQUESTS=1000

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3006
JAEGER_PORT=16686

# ML Models
MODEL_SERVING_URL=http://localhost:8501
MODEL_VERSION=dev
FEATURE_STORE_URL=http://localhost:6566

# Content Generation
MAX_CONTENT_LENGTH=1000
CONTENT_GENERATION_TIMEOUT=5000
DEFAULT_LANGUAGE=en

# Channel Configuration
EMAIL_RATE_LIMIT=100
SMS_RATE_LIMIT=50
PUSH_RATE_LIMIT=200

# Development Helpers
AUTO_MIGRATE=true
SEED_DATA=true
MOCK_EXTERNAL_SERVICES=true
DEBUG_SQL=false
PRETTY_LOGS=true
```

Now let me create the shared schemas and utilities. First, the event schemas:

```json:user-whisperer/shared/schemas/event.schema.json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "User Whisperer Event Schema",
  "description": "Schema for all events processed by the User Whisperer platform",
  "type": "object",
  "required": ["event_id", "app_id", "user_id", "event_type", "timestamp"],
  "properties": {
    "event_id": {
      "type": "string",
      "pattern": "^evt_[a-zA-Z0-9]{16}$",
      "description": "Unique event identifier"
    },
    "app_id": {
      "type": "string",
      "pattern": "^app_[a-zA-Z0-9]{16}$",
      "description": "Application identifier"
    },
    "user_id": {
      "type": "string",
      "pattern": "^usr_[a-zA-Z0-9]{16}$",
      "description": "User identifier"
    },
    "session_id": {
      "type": "string",
      "pattern": "^ses_[a-zA-Z0-9]{16}$",
      "description": "Session identifier"
    },
    "event_type": {
      "type": "string",
      "enum": [
        "user_signup",
        "user_login",
        "user_logout",
        "feature_used",
        "page_viewed",
        "button_clicked",
        "form_submitted",
        "error_encountered",
        "trial_started",
        "subscription_upgraded",
        "subscription_downgraded",
        "subscription_cancelled",
        "payment_failed",
        "support_ticket_created",
        "app_crashed",
        "feature_discovery",
        "onboarding_step_completed",
        "tutorial_viewed",
        "settings_changed",
        "data_exported",
        "data_imported",
        "collaboration_invited",
        "collaboration_accepted",
        "notification_sent",
        "notification_opened",
        "notification_clicked",
        "email_sent",
        "email_opened",
        "email_clicked",
        "sms_sent",
        "sms_delivered",
        "push_sent",
        "push_opened"
      ],
      "description": "Type of event that occurred"
    },
    "timestamp": {
      "type": "string",
      "format": "date-time",
      "description": "ISO 8601 timestamp when event occurred"
    },
    "properties": {
      "type": "object",
      "description": "Event-specific properties",
      "additionalProperties": true,
      "properties": {
        "feature": {
          "type": "string",
          "description": "Feature name for feature_used events"
        },
        "page": {
          "type": "string",
          "description": "Page path for page_viewed events"
        },
        "button": {
          "type": "string",
          "description": "Button identifier for button_clicked events"
        },
        "form": {
          "type": "string", 
          "description": "Form identifier for form_submitted events"
        },
        "error_type": {
          "type": "string",
          "description": "Error category for error_encountered events"
        },
        "error_message": {
          "type": "string",
          "description": "Error message for error_encountered events"
        },
        "plan": {
          "type": "string",
          "enum": ["free", "basic", "pro", "enterprise"],
          "description": "Subscription plan"
        },
        "amount": {
          "type": "number",
          "description": "Monetary amount in cents"
        },
        "currency": {
          "type": "string",
          "pattern": "^[A-Z]{3}$",
          "description": "ISO 4217 currency code"
        },
        "duration": {
          "type": "number",
          "description": "Duration in milliseconds"
        },
        "value": {
          "type": "number",
          "description": "Numeric value associated with event"
        },
        "category": {
          "type": "string",
          "description": "Event category"
        },
        "label": {
          "type": "string",
          "description": "Event label"
        }
      }
    },
    "context": {
      "type": "object",
      "description": "Contextual information about the event",
      "properties": {
        "ip": {
          "type": "string",
          "format": "ipv4",
          "description": "User IP address"
        },
        "user_agent": {
          "type": "string",
          "description": "User agent string"
        },
        "referrer": {
          "type": "string",
          "format": "uri",
          "description": "Referrer URL"
        },
        "utm_source": {
          "type": "string",
          "description": "UTM source parameter"
        },
        "utm_medium": {
          "type": "string",
          "description": "UTM medium parameter"
        },
        "utm_campaign": {
          "type": "string",
          "description": "UTM campaign parameter"
        },
        "device_type": {
          "type": "string",
          "enum": ["desktop", "mobile", "tablet"],
          "description": "Device type"
        },
        "browser": {
          "type": "string",
          "description": "Browser name"
        },
        "browser_version": {
          "type": "string",
          "description": "Browser version"
        },
        "os": {
          "type": "string",
          "description": "Operating system"
        },
        "os_version": {
          "type": "string",
          "description": "Operating system version"
        },
        "screen_resolution": {
          "type": "string",
          "pattern": "^\\d+x\\d+$",
          "description": "Screen resolution (e.g., 1920x1080)"
        },
        "timezone": {
          "type": "string",
          "description": "User timezone"
        },
        "locale": {
          "type": "string",
          "pattern": "^[a-z]{2}-[A-Z]{2}$",
          "description": "User locale (e.g., en-US)"
        },
        "app_version": {
          "type": "string",
          "description": "Application version"
        },
        "sdk_version": {
          "type": "string",
          "description": "SDK version"
        }
      }
    },
    "metadata": {
      "type": "object",
      "description": "Internal metadata for processing",
      "properties": {
        "received_at": {
          "type": "string",
          "format": "date-time",
          "description": "When event was received by ingestion service"
        },
        "processed_at": {
          "type": "string",
          "format": "date-time",
          "description": "When event was processed"
        },
        "schema_version": {
          "type": "string",
          "description": "Schema version used for validation"
        },
        "source": {
          "type": "string",
          "enum": ["sdk", "api", "webhook", "batch"],
          "description": "Event source"
        },
        "batch_id": {
          "type": "string",
          "description": "Batch identifier for batch events"
        }
      }
    }
  }
}
```

Now let me create the Protocol Buffers definitions for service communication:

```protobuf:user-whisperer/shared/protos/events.proto
syntax = "proto3";

package userwhisperer.events;

import "google/protobuf/timestamp.proto";
import "google/protobuf/struct.proto";

option go_package = "github.com/userwhisperer/platform/protos/events";

// Event service for processing user events
service EventService {
  // Ingest a single event
  rpc IngestEvent(IngestEventRequest) returns (IngestEventResponse);
  
  // Batch ingest multiple events
  rpc IngestEventBatch(IngestEventBatchRequest) returns (IngestEventBatchResponse);
  
  // Get event by ID
  rpc GetEvent(GetEventRequest) returns (Event);
  
  // Query events
  rpc QueryEvents(QueryEventsRequest) returns (QueryEventsResponse);
}

// Core event structure
message Event {
  string event_id = 1;
  string app_id = 2;
  string user_id = 3;
  string session_id = 4;
  string event_type = 5;
  google.protobuf.Timestamp timestamp = 6;
  google.protobuf.Struct properties = 7;
  EventContext context = 8;
  EventMetadata metadata = 9;
}

// Event context information
message EventContext {
  string ip = 1;
  string user_agent = 2;
  string referrer = 3;
  string utm_source = 4;
  string utm_medium = 5;
  string utm_campaign = 6;
  string device_type = 7;
  string browser = 8;
  string browser_version = 9;
  string os = 10;
  string os_version = 11;
  string screen_resolution = 12;
  string timezone = 13;
  string locale = 14;
  string app_version = 15;
  string sdk_version = 16;
}

// Event metadata
message EventMetadata {
  google.protobuf.Timestamp received_at = 1;
  google.protobuf.Timestamp processed_at = 2;
  string schema_version = 3;
  string source = 4;
  string batch_id = 5;
}

// Request/Response messages
message IngestEventRequest {
  Event event = 1;
}

message IngestEventResponse {
  string event_id = 1;
  bool success = 2;
  string error_message = 3;
}

message IngestEventBatchRequest {
  repeated Event events = 1;
}

message IngestEventBatchResponse {
  repeated IngestEventResponse results = 1;
}

message GetEventRequest {
  string event_id = 1;
}

message QueryEventsRequest {
  string app_id = 1;
  string user_id = 2;
  repeated string event_types = 3;
  google.protobuf.Timestamp start_time = 4;
  google.protobuf.Timestamp end_time = 5;
  int32 limit = 6;
  string cursor = 7;
}

message QueryEventsResponse {
  repeated Event events = 1;
  string next_cursor = 2;
  bool has_more = 3;
}
```

Let me also create the behavioral analysis proto:

```protobuf:user-whisperer/shared/protos/behavioral.proto
syntax = "proto3";

package userwhisperer.behavioral;

import "google/protobuf/timestamp.proto";
import "google/protobuf/struct.proto";

option go_package = "github.com/userwhisperer/platform/protos/behavioral";

// Behavioral analysis service
service BehavioralService {
  // Analyze user behavior from event
  rpc AnalyzeBehavior(AnalyzeBehaviorRequest) returns (AnalyzeBehaviorResponse);
  
  // Get user behavioral profile
  rpc GetUserProfile(GetUserProfileRequest) returns (UserProfile);
  
  // Update user behavioral metrics
  rpc UpdateUserMetrics(UpdateUserMetricsRequest) returns (UpdateUserMetricsResponse);
  
  // Calculate churn risk
  rpc CalculateChurnRisk(CalculateChurnRiskRequest) returns (ChurnRiskResponse);
  
  // Get user lifecycle stage
  rpc GetLifecycleStage(GetLifecycleStageRequest) returns (LifecycleStageResponse);
}

// User behavioral profile
message UserProfile {
  string user_id = 1;
  string app_id = 2;
  LifecycleStage lifecycle_stage = 3;
  BehavioralMetrics metrics = 4;
  repeated BehavioralPattern patterns = 5;
  EngagementScores scores = 6;
  google.protobuf.Timestamp created_at = 7;
  google.protobuf.Timestamp updated_at = 8;
}

// Lifecycle stages
enum LifecycleStage {
  LIFECYCLE_UNKNOWN = 0;
  LIFECYCLE_ONBOARDING = 1;
  LIFECYCLE_ACTIVATED = 2;
  LIFECYCLE_POWER_USER = 3;
  LIFECYCLE_AT_RISK = 4;
  LIFECYCLE_CHURNED = 5;
}

// Behavioral metrics
message BehavioralMetrics {
  int32 session_count = 1;
  double avg_session_duration = 2;
  int32 days_since_last_activity = 3;
  double feature_adoption_rate = 4;
  int32 error_encounter_count = 5;
  double engagement_trend = 6;
  int32 support_ticket_count = 7;
  double conversion_likelihood = 8;
  google.protobuf.Struct custom_metrics = 9;
}

// Behavioral patterns
message BehavioralPattern {
  string pattern_type = 1;
  string pattern_name = 2;
  double confidence = 3;
  google.protobuf.Struct parameters = 4;
  google.protobuf.Timestamp detected_at = 5;
}

// Engagement scores
message EngagementScores {
  double churn_risk = 1;
  double engagement_level = 2;
  double monetization_readiness = 3;
  double feature_satisfaction = 4;
  double support_satisfaction = 5;
  double overall_health = 6;
}

// Request/Response messages
message AnalyzeBehaviorRequest {
  string user_id = 1;
  string app_id = 2;
  google.protobuf.Struct event_data = 3;
}

message AnalyzeBehaviorResponse {
  UserProfile updated_profile = 1;
  repeated string triggered_patterns = 2;
  bool requires_intervention = 3;
}

message GetUserProfileRequest {
  string user_id = 1;
  string app_id = 2;
}

message UpdateUserMetricsRequest {
  string user_id = 1;
  string app_id = 2;
  BehavioralMetrics metrics = 3;
}

message UpdateUserMetricsResponse {
  bool success = 1;
  string error_message = 2;
}

message CalculateChurnRiskRequest {
  string user_id = 1;
  string app_id = 2;
}

message ChurnRiskResponse {
  double churn_risk_score = 1;
  repeated string risk_factors = 2;
  string risk_level = 3; // low, medium, high, critical
}

message GetLifecycleStageRequest {
  string user_id = 1;
  string app_id = 2;
}

message LifecycleStageResponse {
  LifecycleStage current_stage = 1;
  LifecycleStage predicted_next_stage = 2;
  double stage_confidence = 3;
}
```

Now let me create shared utilities. First, a common configuration utility:

```typescript:user-whisperer/shared/utils/config.ts
import { config } from 'dotenv';
import { z } from 'zod';

// Load environment variables
config();

// Define configuration schema
const configSchema = z.object({
  // Environment
  NODE_ENV: z.enum(['development', 'staging', 'production']).default('development'),
  LOG_LEVEL: z.enum(['debug', 'info', 'warn', 'error']).default('info'),
  
  // Database
  POSTGRES_URL: z.string().url(),
  REDIS_URL: z.string().url(),
  
  // Services
  API_PORT: z.coerce.number().default(3000),
  INGESTION_PORT: z.coerce.number().default(3001),
  BEHAVIORAL_PORT: z.coerce.number().default(3002),
  DECISION_PORT: z.coerce.number().default(3003),
  CONTENT_PORT: z.coerce.number().default(3004),
  ORCHESTRATOR_PORT: z.coerce.number().default(3005),
  
  // Message Queue
  PUBSUB_PROJECT_ID: z.string().default('user-whisperer-dev'),
  PUBSUB_EMULATOR_HOST: z.string().optional(),
  
  // External APIs
  OPENAI_API_KEY: z.string().min(1),
  ANTHROPIC_API_KEY: z.string().min(1),
  SENDGRID_API_KEY: z.string().min(1),
  TWILIO_ACCOUNT_SID: z.string().min(1),
  TWILIO_AUTH_TOKEN: z.string().min(1),
  
  // Security
  JWT_SECRET: z.string().min(32),
  API_KEY_SALT: z.string().min(16),
  ENCRYPTION_KEY: z.string().length(32),
  
  // Feature Flags
  ENABLE_ML_MODELS: z.coerce.boolean().default(false),
  ENABLE_LLM_GENERATION: z.coerce.boolean().default(false),
  USE_MOCK_SERVICES: z.coerce.boolean().default(true),
  
  // Rate Limiting
  RATE_LIMIT_WINDOW_MS: z.coerce.number().default(60000),
  RATE_LIMIT_MAX_REQUESTS: z.coerce.number().default(1000),
});

export type Config = z.infer<typeof configSchema>;

class ConfigManager {
  private static instance: ConfigManager;
  private _config: Config;

  private constructor() {
    try {
      this._config = configSchema.parse(process.env);
    } catch (error) {
      if (error instanceof z.ZodError) {
        const missingFields = error.errors.map(e => e.path.join('.')).join(', ');
        throw new Error(`Invalid configuration. Missing or invalid fields: ${missingFields}`);
      }
      throw error;
    }
  }

  public static getInstance(): ConfigManager {
    if (!ConfigManager.instance) {
      ConfigManager.instance = new ConfigManager();
    }
    return ConfigManager.instance;
  }

  public get config(): Config {
    return this._config;
  }

  public isDevelopment(): boolean {
    return this._config.NODE_ENV === 'development';
  }

  public isProduction(): boolean {
    return this._config.NODE_ENV === 'production';
  }

  public isStaging(): boolean {
    return this._config.NODE_ENV === 'staging';
  }

  public getServiceUrl(service: string): string {
    const portMap: Record<string, number> = {
      'api': this._config.API_PORT,
      'ingestion': this._config.INGESTION_PORT,
      'behavioral': this._config.BEHAVIORAL_PORT,
      'decision': this._config.DECISION_PORT,
      'content': this._config.CONTENT_PORT,
      'orchestrator': this._config.ORCHESTRATOR_PORT,
    };

    const port = portMap[service];
    if (!port) {
      throw new Error(`Unknown service: ${service}`);
    }

    return `http://localhost:${port}`;
  }
}

export const getConfig = (): Config => {
  return ConfigManager.getInstance().config;
};

export const configManager = ConfigManager.getInstance();
```

Let me create a logger utility:

```typescript:user-whisperer/shared/utils/logger.ts
import winston from 'winston';
import { getConfig } from './config';

const config = getConfig();

// Define log levels
const levels = {
  error: 0,
  warn: 1,
  info: 2,
  debug: 3,
};

// Define colors for each level
const colors = {
  error: 'red',
  warn: 'yellow',
  info: 'green',
  debug: 'blue',
};

winston.addColors(colors);

// Create format for development (pretty)
const developmentFormat = winston.format.combine(
  winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
  winston.format.errors({ stack: true }),
  winston.format.colorize({ all: true }),
  winston.format.printf(({ timestamp, level, message, service, ...meta }) => {
    const metaStr = Object.keys(meta).length ? JSON.stringify(meta, null, 2) : '';
    const serviceStr = service ? `[${service}]` : '';
    return `${timestamp} ${level} ${serviceStr}: ${message} ${metaStr}`;
  })
);

// Create format for production (JSON)
const productionFormat = winston.format.combine(
  winston.format.timestamp(),
  winston.format.errors({ stack: true }),
  winston.format.json()
);

// Create transports
const transports: winston.transport[] = [
  new winston.transports.Console({
    level: config.LOG_LEVEL,
    format: config.NODE_ENV === 'production' ? productionFormat : developmentFormat,
  }),
];

// Add file transport for production
if (config.NODE_ENV === 'production') {
  transports.push(
    new winston.transports.File({
      filename: 'logs/error.log',
      level: 'error',
      format: productionFormat,
    }),
    new winston.transports.File({
      filename: 'logs/combined.log',
      format: productionFormat,
    })
  );
}

// Create logger instance
const logger = winston.createLogger({
  levels,
  transports,
  exitOnError: false,
});

// Create service-specific logger factory
export const createServiceLogger = (serviceName: string) => {
  return {
    error: (message: string, meta?: any) => logger.error(message, { service: serviceName, ...meta }),
    warn: (message: string, meta?: any) => logger.warn(message, { service: serviceName, ...meta }),
    info: (message: string, meta?: any) => logger.info(message, { service: serviceName, ...meta }),
    debug: (message: string, meta?: any) => logger.debug(message, { service: serviceName, ...meta }),
  };
};

export default logger;
```

Now let me create a database utility:

```typescript:user-whisperer/shared/utils/database.ts
import { Pool, PoolClient } from 'pg';
import { getConfig } from './config';
import { createServiceLogger } from './logger';

const config = getConfig();
const logger = createServiceLogger('database');

class DatabaseManager {
  private static instance: DatabaseManager;
  private pool: Pool;

  private constructor() {
    this.pool = new Pool({
      connectionString: config.POSTGRES_URL,
      max: 20,
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 2000,
      statement_timeout: 30000,
      query_timeout: 30000,
    });

    // Handle pool errors
    this.pool.on('error', (err) => {
      logger.error('Unexpected error on idle client', err);
    });

    // Handle connection events
    this.pool.on('connect', () => {
      logger.debug('New client connected');
    });

    this.pool.on('remove', () => {
      logger.debug('Client removed');
    });
  }

  public static getInstance(): DatabaseManager {
    if (!DatabaseManager.instance) {
      DatabaseManager.instance = new DatabaseManager();
    }
    return DatabaseManager.instance;
  }

  public async getClient(): Promise<PoolClient> {
    return this.pool.connect();
  }

  public async query(text: string, params?: any[]): Promise<any> {
    const start = Date.now();
    try {
      const result = await this.pool.query(text, params);
      const duration = Date.now() - start;
      logger.debug('Executed query', { duration, rows: result.rowCount });
      return result;
    } catch (error) {
      const duration = Date.now() - start;
      logger.error('Query error', { duration, error, query: text });
      throw error;
    }
  }

  public async transaction<T>(callback: (client: PoolClient) => Promise<T>): Promise<T> {
    const client = await this.getClient();
    try {
      await client.query('BEGIN');
      const result = await callback(client);
      await client.query('COMMIT');
      return result;
    } catch (error) {
      await client.query('ROLLBACK');
      throw error;
    } finally {
      client.release();
    }
  }

  public async close(): Promise<void> {
    await this.pool.end();
    logger.info('Database pool closed');
  }

  public async healthCheck(): Promise<boolean> {
    try {
      await this.query('SELECT 1');
      return true;
    } catch (error) {
      logger.error('Database health check failed', error);
      return false;
    }
  }
}

export const db = DatabaseManager.getInstance();

// Helper functions for common operations
export const withTransaction = async <T>(
  callback: (client: PoolClient) => Promise<T>
): Promise<T> => {
  return db.transaction(callback);
};

export const executeQuery = async (text: string, params?: any[]): Promise<any> => {
  return db.query(text, params);
};

export const healthCheck = async (): Promise<boolean> => {
  return db.healthCheck();
};
```

Let me create a Redis utility:

```typescript:user-whisperer/shared/utils/redis.ts
import Redis from 'ioredis';
import { getConfig } from './config';
import { createServiceLogger } from './logger';

const config = getConfig();
const logger = createServiceLogger('redis');

class RedisManager {
  private static instance: RedisManager;
  private client: Redis;
  private subscriber: Redis;
  private publisher: Redis;

  private constructor() {
    const redisConfig = {
      host: new URL(config.REDIS_URL).hostname,
      port: parseInt(new URL(config.REDIS_URL).port) || 6379,
      password: new URL(config.REDIS_URL).password || undefined,
      retryDelayOnFailover: 100,
      maxRetriesPerRequest: 3,
      lazyConnect: true,
    };

    this.client = new Redis(redisConfig);
    this.subscriber = new Redis(redisConfig);
    this.publisher = new Redis(redisConfig);

    // Event handlers
    this.client.on('connect', () => logger.info('Redis client connected'));
    this.client.on('error', (err) => logger.error('Redis client error', err));
    this.client.on('close', () => logger.info('Redis client connection closed'));

    this.subscriber.on('connect', () => logger.info('Redis subscriber connected'));
    this.subscriber.on('error', (err) => logger.error('Redis subscriber error', err));

    this.publisher.on('connect', () => logger.info('Redis publisher connected'));
    this.publisher.on('error', (err) => logger.error('Redis publisher error', err));
  }

  public static getInstance(): RedisManager {
    if (!RedisManager.instance) {
      RedisManager.instance = new RedisManager();
    }
    return RedisManager.instance;
  }

  // Basic operations
  public async get(key: string): Promise<string | null> {
    try {
      return await this.client.get(key);
    } catch (error) {
      logger.error('Redis GET error', { key, error });
      throw error;
    }
  }

  public async set(key: string, value: string, ttlSeconds?: number): Promise<void> {
    try {
      if (ttlSeconds) {
        await this.client.setex(key, ttlSeconds, value);
      } else {
        await this.client.set(key, value);
      }
    } catch (error) {
      logger.error('Redis SET error', { key, error });
      throw error;
    }
  }

  public async del(key: string): Promise<number> {
    try {
      return await this.client.del(key);
    } catch (error) {
      logger.error('Redis DEL error', { key, error });
      throw error;
    }
  }

  public async exists(key: string): Promise<boolean> {
    try {
      const result = await this.client.exists(key);
      return result === 1;
    } catch (error) {
      logger.error('Redis EXISTS error', { key, error });
      throw error;
    }
  }

  // JSON operations
  public async getJson<T>(key: string): Promise<T | null> {
    const value = await this.get(key);
    return value ? JSON.parse(value) : null;
  }

  public async setJson<T>(key: string, value: T, ttlSeconds?: number): Promise<void> {
    await this.set(key, JSON.stringify(value), ttlSeconds);
  }

  // Hash operations
  public async hget(key: string, field: string): Promise<string | null> {
    try {
      return await this.client.hget(key, field);
    } catch (error) {
      logger.error('Redis HGET error', { key, field, error });
      throw error;
    }
  }

  public async hset(key: string, field: string, value: string): Promise<void> {
    try {
      await this.client.hset(key, field, value);
    } catch (error) {
      logger.error('Redis HSET error', { key, field, error });
      throw error;
    }
  }

  public async hgetall(key: string): Promise<Record<string, string>> {
    try {
      return await this.client.hgetall(key);
    } catch (error) {
      logger.error('Redis HGETALL error', { key, error });
      throw error;
    }
  }

  // List operations
  public async lpush(key: string, ...values: string[]): Promise<number> {
    try {
      return await this.client.lpush(key, ...values);
    } catch (error) {
      logger.error('Redis LPUSH error', { key, error });
      throw error;
    }
  }

  public async rpop(key: string): Promise<string | null> {
    try {
      return await this.client.rpop(key);
    } catch (error) {
      logger.error('Redis RPOP error', { key, error });
      throw error;
    }
  }

  public async llen(key: string): Promise<number> {
    try {
      return await this.client.llen(key);
    } catch (error) {
      logger.error('Redis LLEN error', { key, error });
      throw error;
    }
  }

  // Set operations
  public async sadd(key: string, ...members: string[]): Promise<number> {
    try {
      return await this.client.sadd(key, ...members);
    } catch (error) {
      logger.error('Redis SADD error', { key, error });
      throw error;
    }
  }

  public async sismember(key: string, member: string): Promise<boolean> {
    try {
      const result = await this.client.sismember(key, member);
      return result === 1;
    } catch (error) {
      logger.error('Redis SISMEMBER error', { key, member, error });
      throw error;
    }
  }

  // Pub/Sub operations
  public async publish(channel: string, message: string): Promise<number> {
    try {
      return await this.publisher.publish(channel, message);
    } catch (error) {
      logger.error('Redis PUBLISH error', { channel, error });
      throw error;
    }
  }

  public async subscribe(channel: string, callback: (message: string) => void): Promise<void> {
    try {
      await this.subscriber.subscribe(channel);
      this.subscriber.on('message', (receivedChannel, message) => {
        if (receivedChannel === channel) {
          callback(message);
        }
      });
    } catch (error) {
      logger.error('Redis SUBSCRIBE error', { channel, error });
      throw error;
    }
  }

  // Cache operations with TTL
  public async cache<T>(
    key: string,
    fetcher: () => Promise<T>,
    ttlSeconds: number = 3600
  ): Promise<T> {
    // Try to get from cache first
    const cached = await this.getJson<T>(key);
    if (cached !== null) {
      return cached;
    }

    // Fetch and cache
    const value = await fetcher();
    await this.setJson(key, value, ttlSeconds);
    return value;
  }

  // Lock operations
  public async acquireLock(
    lockKey: string,
    ttlSeconds: number = 10,
    maxWaitMs: number = 5000
  ): Promise<boolean> {
    const startTime = Date.now();
    const lockValue = `${Date.now()}-${Math.random()}`;

    while (Date.now() - startTime < maxWaitMs) {
      const result = await this.client.set(lockKey, lockValue, 'EX', ttlSeconds, 'NX');
      if (result === 'OK') {
        return true;
      }
      await new Promise(resolve => setTimeout(resolve, 10));
    }

    return false;
  }

  public async releaseLock(lockKey: string): Promise<void> {
    await this.del(lockKey);
  }

  // Health check
  public async healthCheck(): Promise<boolean> {
    try {
      const result = await this.client.ping();
      return result === 'PONG';
    } catch (error) {
      logger.error('Redis health check failed', error);
      return false;
    }
  }

  // Cleanup
  public async close(): Promise<void> {
    await Promise.all([
      this.client.quit(),
      this.subscriber.quit(),
      this.publisher.quit(),
    ]);
    logger.info('Redis connections closed');
  }
}

export const redis = RedisManager.getInstance();

// Helper functions
export const cache = async <T>(
  key: string,
  fetcher: () => Promise<T>,
  ttlSeconds: number = 3600
): Promise<T> => {
  return redis.cache(key, fetcher, ttlSeconds);
};

export const withLock = async <T>(
  lockKey: string,
  operation: () => Promise<T>,
  ttlSeconds: number = 10
): Promise<T> => {
  const acquired = await redis.acquireLock(lockKey, ttlSeconds);
  if (!acquired) {
    throw new Error(`Failed to acquire lock: ${lockKey}`);
  }

  try {
    return await operation();
  } finally {
    await redis.releaseLock(lockKey);
  }
};
```

Now let me create the Event Ingestion Service (Node.js/TypeScript):

```json:user-whisperer/services/event-ingestion/package.json
{
  "name": "@userwhisperer/event-ingestion",
  "version": "1.0.0",
  "description": "High-throughput event ingestion service",
  "main": "dist/index.js",
  "scripts": {
    "start": "node dist/index.js",
    "dev": "ts-node-dev --respawn --transpile-only src/index.ts",
    "build": "tsc",
    "test": "jest",
    "test:watch": "jest --watch",
    "test:coverage": "jest --coverage",
    "lint": "eslint src/**/*.ts",
    "lint:fix": "eslint src/**/*.ts --fix",
    "typecheck": "tsc --noEmit",
    "migrate": "knex migrate:latest",
    "migrate:create": "knex migrate:make",
    "migrate:rollback": "knex migrate:rollback"
  },
  "dependencies": {
    "@google-cloud/pubsub": "^4.0.7",
    "@userwhisperer/shared": "file:../../shared",
    "express": "^4.18.2",
    "helmet": "^7.1.0",
    "cors": "^2.8.5",
    "compression": "^1.7.4",
    "express-rate-limit": "^7.1.5",
    "express-validator": "^7.0.1",
    "ajv": "^8.12.0",
    "ajv-formats": "^2.1.1",
    "crypto": "^1.0.1",
    "uuid": "^9.0.1",
    "ioredis": "^5.3.2",
    "pg": "^8.11.3",
    "knex": "^3.0.1",
    "winston": "^3.11.0",
    "prom-client": "^15.1.0"
  },
  "devDependencies": {
    "@types/express": "^4.17.21",
    "@types/node": "^20.0.0",
    "@types/uuid": "^9.0.7",
    "@types/compression": "^1.7.5",
    "@types/cors": "^2.8.17",
    "@types/jest": "^29.5.8",
    "jest": "^29.7.0",
    "ts-jest": "^29.1.1",
    "ts-node-dev": "^2.0.0",
    "typescript": "^5.3.0",
    "supertest": "^6.3.3",
    "@types/supertest": "^2.0.16"
  },
  "engines": {
    "node": ">=20.0.0"
  }
}
```

```typescript:user-whisperer/services/event-ingestion/src/index.ts
import express from 'express';
import helmet from 'helmet';
import cors from 'cors';
import compression from 'compression';
import rateLimit from 'express-rate-limit';
import { createServiceLogger } from '@userwhisperer/shared/utils/logger';
import { getConfig } from '@userwhisperer/shared/utils/config';
import { EventIngestionService } from './services/EventIngestionService';
import { HealthService } from './services/HealthService';
import { MetricsService } from './services/MetricsService';
import { eventRoutes } from './routes/events';
import { healthRoutes } from './routes/health';
import { metricsRoutes } from './routes/metrics';
import { errorHandler } from './middleware/errorHandler';
import { requestLogger } from './middleware/requestLogger';

const config = getConfig();
const logger = createServiceLogger('event-ingestion');

class EventIngestionServer {
  private app: express.Application;
  private eventService: EventIngestionService;
  private healthService: HealthService;
  private metricsService: MetricsService;

  constructor() {
    this.app = express();
    this.eventService = new EventIngestionService();
    this.healthService = new HealthService();
    this.metricsService = new MetricsService();
    
    this.setupMiddleware();
    this.setupRoutes();
    this.setupErrorHandling();
  }

  private setupMiddleware(): void {
    // Security
    this.app.use(helmet());
    this.app.use(cors({
      origin: config.NODE_ENV === 'production' ? ['https://app.userwhisperer.com'] : true,
      credentials: true,
    }));

    // Performance
    this.app.use(compression());

    // Rate limiting
    const limiter = rateLimit({
      windowMs: config.RATE_LIMIT_WINDOW_MS,
      max: config.RATE_LIMIT_MAX_REQUESTS,
      standardHeaders: true,
      legacyHeaders: false,
      message: 'Too many requests from this IP',
    });
    this.app.use(limiter);

    // Body parsing
    this.app.use(express.json({ limit: '10mb' }));
    this.app.use(express.urlencoded({ extended: true, limit: '10mb' }));

    // Logging
    this.app.use(requestLogger);
  }

  private setupRoutes(): void {
    // Pass services to routes
    this.app.use('/events', eventRoutes(this.eventService));
    this.app.use('/health', healthRoutes(this.healthService));
    this.app.use('/metrics', metricsRoutes(this.metricsService));

    // Root endpoint
    this.app.get('/', (req, res) => {
      res.json({
        service: 'User Whisperer Event Ingestion',
        version: '1.0.0',
        environment: config.NODE_ENV,
        timestamp: new Date().toISOString(),
      });
    });
  }

  private setupErrorHandling(): void {
    this.app.use(errorHandler);
  }

  public async start(): Promise<void> {
    try {
      // Initialize services
      await this.eventService.initialize();
      await this.healthService.initialize();
      await this.metricsService.initialize();

      // Start server
      const server = this.app.listen(config.INGESTION_PORT, () => {
        logger.info(`Event Ingestion Service listening on port ${config.INGESTION_PORT}`);
      });

      // Graceful shutdown
      process.on('SIGTERM', async () => {
        logger.info('SIGTERM received, shutting down gracefully...');
        server.close(async () => {
          await this.shutdown();
          process.exit(0);
        });
      });

      process.on('SIGINT', async () => {
        logger.info('SIGINT received, shutting down gracefully...');
        server.close(async () => {
          await this.shutdown();
          process.exit(0);
        });
      });

    } catch (error) {
      logger.error('Failed to start Event Ingestion Service', error);
      process.exit(1);
    }
  }

  private async shutdown(): Promise<void> {
    try {
      await Promise.all([
        this.eventService.shutdown(),
        this.healthService.shutdown(),
        this.metricsService.shutdown(),
      ]);
      logger.info('Event Ingestion Service shut down successfully');
    } catch (error) {
      logger.error('Error during shutdown', error);
    }
  }
}

// Start the server
const server = new EventIngestionServer();
server.start().catch((error) => {
  logger.error('Failed to start server', error);
  process.exit(1);
});
```

```typescript:user-whisperer/services/event-ingestion/src/services/EventIngestionService.ts
import { PubSub } from '@google-cloud/pubsub';
import { createServiceLogger } from '@userwhisperer/shared/utils/logger';
import { getConfig } from '@userwhisperer/shared/utils/config';
import { redis } from '@userwhisperer/shared/utils/redis';
import { db } from '@userwhisperer/shared/utils/database';
import { EventValidator } from '../utils/EventValidator';
import { EventProcessor } from '../utils/EventProcessor';
import { EventDeduplicator } from '../utils/EventDeduplicator';
import { EventEnricher } from '../utils/EventEnricher';
import { v4 as uuidv4 } from 'uuid';
import crypto from 'crypto';

const config = getConfig();
const logger = createServiceLogger('event-ingestion-service');

export interface IncomingEvent {
  event_type: string;
  user_id: string;
  app_id?: string;
  session_id?: string;
  properties?: Record<string, any>;
  context?: Record<string, any>;
  timestamp?: string;
}

export interface ProcessedEvent extends IncomingEvent {
  event_id: string;
  app_id: string;
  timestamp: string;
  metadata: {
    received_at: string;
    processed_at: string;
    schema_version: string;
    source: string;
    batch_id?: string;
  };
}

export class EventIngestionService {
  private pubSub: PubSub;
  private validator: EventValidator;
  private processor: EventProcessor;
  private deduplicator: EventDeduplicator;
  private enricher: EventEnricher;
  private initialized: boolean = false;

  constructor() {
    this.pubSub = new PubSub({
      projectId: config.PUBSUB_PROJECT_ID,
    });
    
    this.validator = new EventValidator();
    this.processor = new EventProcessor();
    this.deduplicator = new EventDeduplicator();
    this.enricher = new EventEnricher();
  }

  public async initialize(): Promise<void> {
    try {
      logger.info('Initializing Event Ingestion Service...');

      // Initialize sub-components
      await this.validator.initialize();
      await this.processor.initialize();
      await this.deduplicator.initialize();
      await this.enricher.initialize();

      // Create Pub/Sub topics if they don't exist
      await this.createTopicsIfNeeded();

      this.initialized = true;
      logger.info('Event Ingestion Service initialized successfully');
    } catch (error) {
      logger.error('Failed to initialize Event Ingestion Service', error);
      throw error;
    }
  }

  public async ingestEvent(event: IncomingEvent): Promise<{ event_id: string; status: string }> {
    if (!this.initialized) {
      throw new Error('Service not initialized');
    }

    try {
      // Step 1: Generate event ID
      const eventId = this.generateEventId();
      
      // Step 2: Validate event schema
      const validationResult = await this.validator.validate(event);
      if (!validationResult.valid) {
        logger.warn('Invalid event schema', { eventId, errors: validationResult.errors });
        await this.quarantineEvent(eventId, event, validationResult.errors);
        return { event_id: eventId, status: 'quarantined' };
      }

      // Step 3: Check for duplicates
      const isDuplicate = await this.deduplicator.isDuplicate(event);
      if (isDuplicate) {
        logger.debug('Duplicate event detected', { eventId });
        return { event_id: eventId, status: 'duplicate' };
      }

      // Step 4: Enrich event with context
      const enrichedEvent = await this.enricher.enrich(event);

      // Step 5: Process and prepare final event
      const processedEvent: ProcessedEvent = {
        event_id: eventId,
        app_id: enrichedEvent.app_id || 'default',
        ...enrichedEvent,
        timestamp: enrichedEvent.timestamp || new Date().toISOString(),
        metadata: {
          received_at: new Date().toISOString(),
          processed_at: new Date().toISOString(),
          schema_version: '1.0',
          source: 'api',
        },
      };

      // Step 6: Store in database
      await this.storeEvent(processedEvent);

      // Step 7: Publish to stream
      await this.publishEvent(processedEvent);

      // Step 8: Cache event hash for deduplication
      await this.deduplicator.cacheEventHash(event);

      logger.debug('Event ingested successfully', { eventId });
      return { event_id: eventId, status: 'accepted' };

    } catch (error) {
      logger.error('Failed to ingest event', { error });
      throw error;
    }
  }

  public async ingestEventBatch(events: IncomingEvent[]): Promise<Array<{ event_id: string; status: string }>> {
    if (!this.initialized) {
      throw new Error('Service not initialized');
    }

    const batchId = uuidv4();
    const results: Array<{ event_id: string; status: string }> = [];

    logger.info('Processing event batch', { batchId, count: events.length });

    // Process events in parallel with concurrency limit
    const concurrency = 10;
    const chunks = this.chunkArray(events, concurrency);

    for (const chunk of chunks) {
      const chunkPromises = chunk.map(async (event) => {
        try {
          const result = await this.ingestEvent(event);
          return result;
        } catch (error) {
          const eventId = this.generateEventId();
          logger.error('Failed to process event in batch', { batchId, eventId, error });
          return { event_id: eventId, status: 'error' };
        }
      });

      const chunkResults = await Promise.all(chunkPromises);
      results.push(...chunkResults);
    }

    logger.info('Event batch processed', { batchId, total: events.length, accepted: results.filter(r => r.status === 'accepted').length });
    return results;
  }

  public async getEvent(eventId: string): Promise<ProcessedEvent | null> {
    try {
      // Try cache first
      const cached = await redis.getJson<ProcessedEvent>(`event:${eventId}`);
      if (cached) {
        return cached;
      }

      // Query database
      const result = await db.query(
        'SELECT * FROM events WHERE event_id = $1',
        [eventId]
      );

      if (result.rows.length === 0) {
        return null;
      }

      const event = result.rows[0];
      
      // Cache for future requests
      await redis.setJson(`event:${eventId}`, event, 3600); // 1 hour TTL
      
      return event;
    } catch (error) {
      logger.error('Failed to get event', { eventId, error });
      throw error;
    }
  }

  private generateEventId(): string {
    return `evt_${crypto.randomBytes(8).toString('hex')}`;
  }

  private async storeEvent(event: ProcessedEvent): Promise<void> {
    const query = `
      INSERT INTO events (
        event_id, app_id, user_id, session_id, event_type,
        properties, context, metadata, timestamp, created_at
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
      ON CONFLICT (event_id) DO NOTHING
    `;

    await db.query(query, [
      event.event_id,
      event.app_id,
      event.user_id,
      event.session_id,
      event.event_type,
      JSON.stringify(event.properties || {}),
      JSON.stringify(event.context || {}),
      JSON.stringify(event.metadata),
      event.timestamp,
    ]);
  }

  private async publishEvent(event: ProcessedEvent): Promise<void> {
    const topic = this.pubSub.topic('enriched-events');
    const data = Buffer.from(JSON.stringify(event));
    
    await topic.publishMessage({
      data,
      attributes: {
        eventType: event.event_type,
        userId: event.user_id,
        appId: event.app_id,
      },
      orderingKey: event.user_id, // Ensure events for same user are processed in order
    });
  }

  private async quarantineEvent(eventId: string, event: IncomingEvent, errors: string[]): Promise<void> {
    const quarantinedEvent = {
      event_id: eventId,
      original_event: event,
      errors,
      quarantined_at: new Date().toISOString(),
    };

    // Store in quarantine table
    await db.query(
      'INSERT INTO quarantined_events (event_id, data, created_at) VALUES ($1, $2, NOW())',
      [eventId, JSON.stringify(quarantinedEvent)]
    );

    // Also publish to dead letter queue for manual review
    const topic = this.pubSub.topic('quarantined-events');
    await topic.publishMessage({
      data: Buffer.from(JSON.stringify(quarantinedEvent)),
    });
  }

  private async createTopicsIfNeeded(): Promise<void> {
    const topics = ['raw-events', 'enriched-events', 'quarantined-events'];
    
    for (const topicName of topics) {
      try {
        const topic = this.pubSub.topic(topicName);
        const [exists] = await topic.exists();
        
        if (!exists) {
          await topic.create();
          logger.info(`Created Pub/Sub topic: ${topicName}`);
        }
      } catch (error) {
        logger.error(`Failed to create topic: ${topicName}`, error);
      }
    }
  }

  private chunkArray<T>(array: T[], chunkSize: number): T[][] {
    const chunks: T[][] = [];
    for (let i = 0; i < array.length; i += chunkSize) {
      chunks.push(array.slice(i, i + chunkSize));
    }
    return chunks;
  }

  public async shutdown(): Promise<void> {
    try {
      await this.pubSub.close();
      logger.info('Event Ingestion Service shut down successfully');
    } catch (error) {
      logger.error('Error during Event Ingestion Service shutdown', error);
      throw error;
    }
  }
}
```

This is a comprehensive start to the User Whisperer Platform implementation. I've created:

## What I've Built So Far:

### 1. **Project Foundation**
- Complete repository structure matching the technical specification
- Root configuration files (package.json, docker-compose.yml, Makefile, .env)
- Comprehensive README with quick start guide

### 2. **Shared Infrastructure**
- **Configuration Management** - Type-safe config with Zod validation
- **Logging System** - Winston-based structured logging with service-specific loggers
- **Database Utilities** - PostgreSQL connection pooling, transactions, health checks
- **Redis Utilities** - Caching, pub/sub, locks with comprehensive operations
- **Event Schemas** - JSON Schema validation for all event types

### 3. **Protocol Buffers**
- Event service definitions for gRPC communication
- Behavioral analysis service definitions
- Strongly typed message structures

### 4. **Event Ingestion Service** (Partially Complete)
- High-performance TypeScript/Node.js service
- Event validation, deduplication, and enrichment
- Pub/Sub integration for event streaming
- Batch processing capabilities
- Error handling and quarantine system

### 5. **Development Environment**
- Docker Compose with all services (PostgreSQL, Redis, Pub/Sub emulator, Kong API Gateway)
- Monitoring stack (Prometheus, Grafana, Jaeger)
- Development automation with Makefile

## Next Steps for Complete Implementation:

1. **Complete Event Ingestion Service** - Add remaining utilities and routes
2. **Behavioral Analysis Engine** (Python) - ML-powered user behavior analysis
3. **Decision Engine** (Python) - AI decision making with ML models
4. **Content Generation Service** (Python) - LLM-powered content creation
5. **Channel Orchestration Service** (Go) - Multi-channel message delivery
6. **API Gateway Configuration** - Kong routing and authentication
7. **Database Migrations** - SQL schema for all tables
8. **ML Model Training Pipeline** - TensorFlow/XGBoost implementations
9. **Kubernetes Deployment** - Production-ready container orchestration
10. **Monitoring & Observability** - Complete metrics and alerting

The foundation is now solid and follows enterprise-grade patterns. Each service will be:
- **Independently deployable** microservices
- **Fully observable** with metrics, logs, and traces  
- **Highly available** with proper error handling and circuit breakers
- **Scalable** with horizontal scaling capabilities
- **Secure** with authentication, authorization, and data encryption

Would you like me to continue with implementing the remaining services, or would you prefer to see the next part of the technical documentation first? 