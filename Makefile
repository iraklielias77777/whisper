.PHONY: help setup dev dev-stop dev-clean test build

# Default target
help: ## Show this help message
	@echo '🚀 User Whisperer Platform - Development Commands'
	@echo ''
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Environment Setup
setup: ## Setup development environment
	@echo "🔧 Setting up User Whisperer development environment..."
	@node scripts/setup.js
	@echo "✅ Setup complete!"

# Development Environment
dev: ## Start development environment
	@echo "🚀 Starting development services..."
	@docker-compose up -d
	@echo ""
	@echo "✅ Services started successfully!"
	@echo ""
	@echo "🌐 Access Points:"
	@echo "  API Gateway:    http://localhost:8000"
	@echo "  API Server:     http://localhost:3000"
	@echo "  Grafana:        http://localhost:3006 (admin/admin)"
	@echo "  Prometheus:     http://localhost:9090"
	@echo ""
	@echo "📊 To view logs: make dev-logs"

dev-logs: ## Show development logs
	@docker-compose logs -f

dev-stop: ## Stop development environment
	@echo "🛑 Stopping development services..."
	@docker-compose down
	@echo "✅ Services stopped"

dev-clean: ## Clean development environment
	@echo "🧹 Cleaning development environment..."
	@docker-compose down -v --remove-orphans
	@docker system prune -f
	@echo "✅ Environment cleaned"

# Testing
test: ## Run all tests
	@echo "🧪 Running all tests..."
	@npm run test
	@echo "✅ All tests completed"

# Building
build: ## Build all services
	@echo "🏗️  Building all services..."
	@npm run build
	@echo "✅ Build completed"

# Code Quality
lint: ## Run linter
	@echo "🔍 Running linter..."
	@npm run lint

typecheck: ## Run TypeScript type checking
	@echo "📝 Running TypeScript type checking..."
	@npm run typecheck

# Database
migrate: ## Run database migrations
	@echo "📊 Running database migrations..."
	@cd services/event-ingestion && npm run migrate || echo "⚠️  Migrations will be available after service setup"

# Utilities
status: ## Show service status
	@echo "📊 Service Status:"
	@docker-compose ps

clean: ## Clean build artifacts
	@echo "🧹 Cleaning build artifacts..."
	@find . -name "node_modules" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "dist" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup completed" 