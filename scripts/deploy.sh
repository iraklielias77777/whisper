#!/bin/bash

# User Whisperer Platform Deployment Script
# Supports deployment to staging and production environments

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_CONFIG_DIR="$PROJECT_ROOT/infrastructure/deployment"

# Default values
ENVIRONMENT=""
DRY_RUN=false
SKIP_TESTS=false
SKIP_BUILD=false
FORCE_DEPLOY=false
VERBOSE=false

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [ENVIRONMENT] [OPTIONS]

Deploy User Whisperer Platform to specified environment.

ENVIRONMENTS:
    staging     Deploy to staging environment
    production  Deploy to production environment

OPTIONS:
    --dry-run           Show what would be deployed without making changes
    --skip-tests        Skip running tests before deployment
    --skip-build        Skip building Docker images
    --force             Force deployment without confirmation prompts
    --verbose           Show verbose output
    -h, --help          Show this help message

EXAMPLES:
    $0 staging                    # Deploy to staging
    $0 production --dry-run       # Preview production deployment
    $0 staging --skip-tests       # Deploy to staging without running tests
    $0 production --force         # Force production deployment

PREREQUISITES:
    - Docker and Docker Compose installed
    - kubectl configured for target cluster
    - Helm 3.x installed
    - Required environment variables set
    - Valid Kubernetes context

ENVIRONMENT VARIABLES:
    For staging:
        KUBE_CONTEXT_STAGING    Kubernetes context for staging
        IMAGE_REGISTRY_STAGING  Docker registry for staging images
        
    For production:
        KUBE_CONTEXT_PROD      Kubernetes context for production
        IMAGE_REGISTRY_PROD    Docker registry for production images
        
    Common:
        DOCKER_USERNAME        Docker registry username
        DOCKER_PASSWORD        Docker registry password
        SLACK_WEBHOOK_URL      Slack webhook for notifications (optional)

EOF
}

# Parse command line arguments
parse_arguments() {
    if [[ $# -eq 0 ]]; then
        print_error "Environment is required"
        show_usage
        exit 1
    fi

    ENVIRONMENT="$1"
    shift

    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --force)
                FORCE_DEPLOY=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    # Validate environment
    if [[ "$ENVIRONMENT" != "staging" && "$ENVIRONMENT" != "production" ]]; then
        print_error "Invalid environment: $ENVIRONMENT"
        print_error "Valid environments: staging, production"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."

    local missing_tools=()

    # Check required tools
    if ! command -v docker &> /dev/null; then
        missing_tools+=("docker")
    fi

    if ! command -v kubectl &> /dev/null; then
        missing_tools+=("kubectl")
    fi

    if ! command -v helm &> /dev/null; then
        missing_tools+=("helm")
    fi

    if ! command -v jq &> /dev/null; then
        missing_tools+=("jq")
    fi

    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        print_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi

    # Check environment-specific variables
    local kube_context_var="KUBE_CONTEXT_${ENVIRONMENT^^}"
    local registry_var="IMAGE_REGISTRY_${ENVIRONMENT^^}"

    if [[ -z "${!kube_context_var:-}" ]]; then
        print_error "Missing required environment variable: $kube_context_var"
        exit 1
    fi

    if [[ -z "${!registry_var:-}" ]]; then
        print_error "Missing required environment variable: $registry_var"
        exit 1
    fi

    # Check Kubernetes context
    local kube_context="${!kube_context_var}"
    if ! kubectl config get-contexts "$kube_context" &> /dev/null; then
        print_error "Kubernetes context not found: $kube_context"
        exit 1
    fi

    print_success "Prerequisites check passed"
}

# Load environment configuration
load_environment_config() {
    print_info "Loading configuration for $ENVIRONMENT environment..."

    local config_file="$DEPLOYMENT_CONFIG_DIR/$ENVIRONMENT.env"
    
    if [[ -f "$config_file" ]]; then
        # shellcheck source=/dev/null
        source "$config_file"
        print_success "Loaded configuration from $config_file"
    else
        print_warning "Configuration file not found: $config_file"
    fi

    # Set deployment-specific variables
    case "$ENVIRONMENT" in
        staging)
            KUBE_CONTEXT="${KUBE_CONTEXT_STAGING}"
            IMAGE_REGISTRY="${IMAGE_REGISTRY_STAGING}"
            NAMESPACE="user-whisperer-staging"
            HELM_RELEASE="user-whisperer-staging"
            ;;
        production)
            KUBE_CONTEXT="${KUBE_CONTEXT_PROD}"
            IMAGE_REGISTRY="${IMAGE_REGISTRY_PROD}"
            NAMESPACE="user-whisperer-prod"
            HELM_RELEASE="user-whisperer-prod"
            ;;
    esac

    print_info "Deployment configuration:"
    print_info "  Environment: $ENVIRONMENT"
    print_info "  Kubernetes Context: $KUBE_CONTEXT"
    print_info "  Image Registry: $IMAGE_REGISTRY"
    print_info "  Namespace: $NAMESPACE"
    print_info "  Helm Release: $HELM_RELEASE"
}

# Run tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        print_warning "Skipping tests as requested"
        return 0
    fi

    print_info "Running tests..."

    # Unit tests
    print_info "Running unit tests..."
    if ! npm run test:unit; then
        print_error "Unit tests failed"
        exit 1
    fi

    # Integration tests
    print_info "Running integration tests..."
    if ! npm run test:integration; then
        print_error "Integration tests failed"
        exit 1
    fi

    # Security tests
    print_info "Running security tests..."
    if ! npm audit --audit-level moderate; then
        print_error "Security audit failed"
        exit 1
    fi

    print_success "All tests passed"
}

# Build Docker images
build_images() {
    if [[ "$SKIP_BUILD" == "true" ]]; then
        print_warning "Skipping image build as requested"
        return 0
    fi

    print_info "Building Docker images..."

    local services=("api-gateway" "event-ingestion" "behavioral-analysis" "decision-engine" "content-generator" "channel-orchestrator")
    local git_commit
    git_commit=$(git rev-parse --short HEAD)
    local timestamp
    timestamp=$(date +%Y%m%d%H%M%S)
    local image_tag="${git_commit}-${timestamp}"

    for service in "${services[@]}"; do
        local image_name="$IMAGE_REGISTRY/user-whisperer-$service:$image_tag"
        local latest_name="$IMAGE_REGISTRY/user-whisperer-$service:latest"
        
        print_info "Building $service..."
        
        if [[ "$DRY_RUN" == "true" ]]; then
            print_info "[DRY RUN] Would build: $image_name"
            continue
        fi

        # Build image
        if ! docker build -t "$image_name" -t "$latest_name" "./services/$service"; then
            print_error "Failed to build $service"
            exit 1
        fi

        # Push to registry
        print_info "Pushing $image_name..."
        if ! docker push "$image_name"; then
            print_error "Failed to push $service"
            exit 1
        fi

        if ! docker push "$latest_name"; then
            print_error "Failed to push latest tag for $service"
            exit 1
        fi
    done

    # Store image tag for later use
    echo "$image_tag" > "$PROJECT_ROOT/.deployment-image-tag"
    
    print_success "All images built and pushed successfully"
}

# Deploy to Kubernetes
deploy_to_kubernetes() {
    print_info "Deploying to Kubernetes cluster..."

    # Switch to correct context
    kubectl config use-context "$KUBE_CONTEXT"

    # Get image tag
    local image_tag
    if [[ -f "$PROJECT_ROOT/.deployment-image-tag" ]]; then
        image_tag=$(cat "$PROJECT_ROOT/.deployment-image-tag")
    else
        image_tag="latest"
    fi

    print_info "Using image tag: $image_tag"

    # Create namespace if it doesn't exist
    if [[ "$DRY_RUN" == "true" ]]; then
        print_info "[DRY RUN] Would ensure namespace: $NAMESPACE"
    else
        kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    fi

    # Deploy with Helm
    local helm_chart="$PROJECT_ROOT/helm/user-whisperer"
    local values_file="$PROJECT_ROOT/helm/values-$ENVIRONMENT.yaml"

    if [[ ! -f "$values_file" ]]; then
        print_error "Values file not found: $values_file"
        exit 1
    fi

    local helm_args=(
        "upgrade" "$HELM_RELEASE" "$helm_chart"
        "--install"
        "--namespace" "$NAMESPACE"
        "--values" "$values_file"
        "--set" "global.imageTag=$image_tag"
        "--set" "global.environment=$ENVIRONMENT"
        "--timeout" "10m"
        "--wait"
    )

    if [[ "$DRY_RUN" == "true" ]]; then
        helm_args+=("--dry-run")
        print_info "[DRY RUN] Would run: helm ${helm_args[*]}"
    fi

    if [[ "$VERBOSE" == "true" ]]; then
        helm_args+=("--debug")
    fi

    print_info "Running Helm deployment..."
    if ! helm "${helm_args[@]}"; then
        print_error "Helm deployment failed"
        exit 1
    fi

    if [[ "$DRY_RUN" != "true" ]]; then
        # Wait for rollout to complete
        print_info "Waiting for deployment to complete..."
        local deployments
        deployments=$(kubectl get deployments -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}')
        
        for deployment in $deployments; do
            print_info "Waiting for $deployment..."
            if ! kubectl rollout status deployment/"$deployment" -n "$NAMESPACE" --timeout=300s; then
                print_error "Deployment $deployment failed to complete"
                exit 1
            fi
        done
    fi

    print_success "Kubernetes deployment completed"
}

# Run post-deployment tests
run_post_deployment_tests() {
    if [[ "$DRY_RUN" == "true" ]]; then
        print_info "[DRY RUN] Would run post-deployment tests"
        return 0
    fi

    print_info "Running post-deployment tests..."

    # Get service endpoints
    local api_gateway_url
    api_gateway_url=$(kubectl get service api-gateway-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "localhost")

    if [[ "$api_gateway_url" == "localhost" ]]; then
        print_warning "Could not get LoadBalancer hostname, using port-forward for tests"
        kubectl port-forward -n "$NAMESPACE" svc/api-gateway-service 8080:80 &
        local port_forward_pid=$!
        api_gateway_url="localhost:8080"
        sleep 5
    fi

    # Health check
    print_info "Testing health endpoint..."
    local health_url="http://$api_gateway_url/health"
    local max_attempts=30
    local attempt=0

    while [[ $attempt -lt $max_attempts ]]; do
        if curl -s -f "$health_url" > /dev/null; then
            print_success "Health check passed"
            break
        fi
        
        ((attempt++))
        print_info "Health check attempt $attempt/$max_attempts..."
        sleep 10
    done

    if [[ $attempt -eq $max_attempts ]]; then
        print_error "Health check failed after $max_attempts attempts"
        if [[ -n "${port_forward_pid:-}" ]]; then
            kill $port_forward_pid 2>/dev/null || true
        fi
        exit 1
    fi

    # API smoke test
    print_info "Running API smoke tests..."
    if ! curl -s -f "$health_url/ready" > /dev/null; then
        print_error "Readiness check failed"
        if [[ -n "${port_forward_pid:-}" ]]; then
            kill $port_forward_pid 2>/dev/null || true
        fi
        exit 1
    fi

    # Clean up port-forward if used
    if [[ -n "${port_forward_pid:-}" ]]; then
        kill $port_forward_pid 2>/dev/null || true
    fi

    print_success "Post-deployment tests passed"
}

# Send notification
send_notification() {
    local status="$1"
    local message="$2"

    if [[ -z "${SLACK_WEBHOOK_URL:-}" ]]; then
        return 0
    fi

    local color
    case "$status" in
        "success") color="good" ;;
        "failure") color="danger" ;;
        *) color="warning" ;;
    esac

    local payload
    payload=$(jq -n \
        --arg text "$message" \
        --arg color "$color" \
        --arg environment "$ENVIRONMENT" \
        '{
            "attachments": [
                {
                    "color": $color,
                    "title": "User Whisperer Deployment",
                    "text": $text,
                    "fields": [
                        {
                            "title": "Environment",
                            "value": $environment,
                            "short": true
                        },
                        {
                            "title": "Timestamp",
                            "value": (now | strftime("%Y-%m-%d %H:%M:%S UTC")),
                            "short": true
                        }
                    ]
                }
            ]
        }')

    curl -s -X POST -H 'Content-type: application/json' \
        --data "$payload" \
        "$SLACK_WEBHOOK_URL" > /dev/null || true
}

# Confirm deployment for production
confirm_deployment() {
    if [[ "$ENVIRONMENT" != "production" || "$FORCE_DEPLOY" == "true" ]]; then
        return 0
    fi

    print_warning "You are about to deploy to PRODUCTION environment!"
    print_warning "This will affect live users and services."
    echo
    read -p "Are you sure you want to continue? (type 'yes' to confirm): " -r
    
    if [[ $REPLY != "yes" ]]; then
        print_info "Deployment cancelled"
        exit 0
    fi
}

# Cleanup function
cleanup() {
    local exit_code=$?
    
    if [[ $exit_code -ne 0 ]]; then
        print_error "Deployment failed with exit code $exit_code"
        send_notification "failure" "Deployment to $ENVIRONMENT failed"
    fi
    
    # Clean up any temporary files
    rm -f "$PROJECT_ROOT/.deployment-image-tag"
    
    exit $exit_code
}

# Main deployment function
main() {
    # Set up cleanup trap
    trap cleanup EXIT

    print_info "Starting deployment of User Whisperer Platform"
    print_info "Environment: $ENVIRONMENT"
    print_info "Dry run: $DRY_RUN"
    
    # Parse arguments and check prerequisites
    check_prerequisites
    load_environment_config
    
    # Confirm deployment if necessary
    confirm_deployment
    
    # Run deployment steps
    run_tests
    build_images
    deploy_to_kubernetes
    run_post_deployment_tests
    
    # Success notification
    local message="Successfully deployed User Whisperer Platform to $ENVIRONMENT"
    if [[ "$DRY_RUN" == "true" ]]; then
        message="[DRY RUN] $message"
    fi
    
    print_success "$message"
    send_notification "success" "$message"
}

# Parse arguments and run main function
parse_arguments "$@"
main
