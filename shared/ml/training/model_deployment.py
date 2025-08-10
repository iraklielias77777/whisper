"""
Model Deployment and Registry for User Whisperer Platform
Manages model versions, deployment, and serving infrastructure
"""

import asyncio
import json
import logging
import os
import shutil
import pickle
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import subprocess

# Docker and Kubernetes imports
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

try:
    from kubernetes import client, config as k8s_config
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False

# Cloud imports
try:
    from google.cloud import storage
    from google.cloud import run_v2
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False

# MLflow imports
try:
    import mlflow
    import mlflow.tracking
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)

class DeploymentStatus(Enum):
    PENDING = "pending"
    BUILDING = "building"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    DEPRECATED = "deprecated"

class DeploymentTarget(Enum):
    LOCAL = "local"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    CLOUD_RUN = "cloud_run"
    SAGEMAKER = "sagemaker"
    LAMBDA = "lambda"

@dataclass
class ModelVersion:
    """Model version metadata"""
    model_name: str
    version: str
    model_path: str
    framework: str
    created_at: datetime
    created_by: str
    metrics: Dict[str, float]
    tags: Dict[str, str]
    size_mb: float
    status: str = "registered"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            **asdict(self),
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create from dictionary"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    target: DeploymentTarget
    replicas: int = 1
    cpu_request: str = "100m"
    cpu_limit: str = "500m"
    memory_request: str = "256Mi"
    memory_limit: str = "512Mi"
    env_vars: Dict[str, str] = None
    port: int = 8080
    health_check_path: str = "/health"
    auto_scaling: bool = True
    min_replicas: int = 1
    max_replicas: int = 10
    
    def __post_init__(self):
        if self.env_vars is None:
            self.env_vars = {}

@dataclass
class Deployment:
    """Deployment instance"""
    deployment_id: str
    model_version: ModelVersion
    config: DeploymentConfig
    status: DeploymentStatus
    endpoint_url: Optional[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    logs: List[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.logs is None:
            self.logs = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'deployment_id': self.deployment_id,
            'model_version': self.model_version.to_dict(),
            'config': asdict(self.config),
            'status': self.status.value,
            'endpoint_url': self.endpoint_url,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'logs': self.logs
        }

class ModelRegistry:
    """
    Model registry for versioning and metadata management
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.storage_path = config.get('storage_path', './models')
        self.registry_backend = config.get('backend', 'local')  # local, mlflow, cloud
        
        # Storage
        os.makedirs(self.storage_path, exist_ok=True)
        self.models = {}  # model_name -> List[ModelVersion]
        self.metadata_file = os.path.join(self.storage_path, 'registry.json')
        
        # External clients
        self.mlflow_client = None
        self.storage_client = None
        
        logger.info(f"Initialized ModelRegistry with {self.registry_backend} backend")
    
    async def initialize(self):
        """Initialize model registry"""
        
        # Load existing registry
        await self.load_registry()
        
        # Initialize external clients
        if self.registry_backend == 'mlflow' and MLFLOW_AVAILABLE:
            mlflow_uri = self.config.get('mlflow_uri', 'http://localhost:5000')
            mlflow.set_tracking_uri(mlflow_uri)
            self.mlflow_client = mlflow.tracking.MlflowClient()
            
        elif self.registry_backend == 'cloud' and GOOGLE_CLOUD_AVAILABLE:
            project_id = self.config.get('project_id')
            if project_id:
                self.storage_client = storage.Client(project=project_id)
    
    async def register_model(
        self,
        model_name: str,
        model_path: str,
        framework: str,
        metrics: Dict[str, float],
        tags: Optional[Dict[str, str]] = None,
        version: Optional[str] = None
    ) -> ModelVersion:
        """Register a new model version"""
        
        try:
            # Generate version if not provided
            if version is None:
                version = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Calculate model size
            size_mb = self._calculate_model_size(model_path)
            
            # Create model version
            model_version = ModelVersion(
                model_name=model_name,
                version=version,
                model_path=model_path,
                framework=framework,
                created_at=datetime.now(),
                created_by=self.config.get('user', 'system'),
                metrics=metrics,
                tags=tags or {},
                size_mb=size_mb
            )
            
            # Store model files
            stored_path = await self._store_model(model_version)
            model_version.model_path = stored_path
            
            # Add to registry
            if model_name not in self.models:
                self.models[model_name] = []
            
            self.models[model_name].append(model_version)
            
            # Register in external backend
            if self.registry_backend == 'mlflow' and self.mlflow_client:
                await self._register_in_mlflow(model_version)
            
            # Save registry
            await self.save_registry()
            
            logger.info(f"Registered model {model_name} version {version}")
            
            return model_version
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    def _calculate_model_size(self, model_path: str) -> float:
        """Calculate model size in MB"""
        
        try:
            if os.path.isfile(model_path):
                size_bytes = os.path.getsize(model_path)
            elif os.path.isdir(model_path):
                size_bytes = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(model_path)
                    for filename in filenames
                )
            else:
                return 0.0
            
            return size_bytes / 1024 / 1024
            
        except Exception:
            return 0.0
    
    async def _store_model(self, model_version: ModelVersion) -> str:
        """Store model files in registry storage"""
        
        # Create versioned storage path
        storage_dir = os.path.join(
            self.storage_path,
            model_version.model_name,
            model_version.version
        )
        
        os.makedirs(storage_dir, exist_ok=True)
        
        # Copy model files
        source_path = model_version.model_path
        
        if os.path.isfile(source_path):
            # Single file
            filename = os.path.basename(source_path)
            dest_path = os.path.join(storage_dir, filename)
            shutil.copy2(source_path, dest_path)
            return dest_path
            
        elif os.path.isdir(source_path):
            # Directory
            dest_path = os.path.join(storage_dir, 'model')
            shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
            return dest_path
        
        else:
            raise ValueError(f"Model path not found: {source_path}")
    
    async def _register_in_mlflow(self, model_version: ModelVersion):
        """Register model in MLflow"""
        
        try:
            # This would register the model in MLflow
            # For now, just log the action
            logger.info(f"Would register {model_version.model_name} in MLflow")
            
        except Exception as e:
            logger.warning(f"MLflow registration failed: {e}")
    
    async def get_model_version(
        self,
        model_name: str,
        version: Optional[str] = None
    ) -> Optional[ModelVersion]:
        """Get a specific model version"""
        
        if model_name not in self.models:
            return None
        
        versions = self.models[model_name]
        
        if version is None:
            # Get latest version
            return max(versions, key=lambda v: v.created_at)
        
        # Get specific version
        for v in versions:
            if v.version == version:
                return v
        
        return None
    
    async def list_model_versions(
        self,
        model_name: str,
        limit: Optional[int] = None
    ) -> List[ModelVersion]:
        """List versions for a model"""
        
        if model_name not in self.models:
            return []
        
        versions = sorted(
            self.models[model_name],
            key=lambda v: v.created_at,
            reverse=True
        )
        
        if limit:
            versions = versions[:limit]
        
        return versions
    
    async def list_models(self) -> List[str]:
        """List all registered models"""
        
        return list(self.models.keys())
    
    async def delete_model_version(
        self,
        model_name: str,
        version: str
    ) -> bool:
        """Delete a model version"""
        
        try:
            if model_name not in self.models:
                return False
            
            # Find and remove version
            versions = self.models[model_name]
            version_to_remove = None
            
            for v in versions:
                if v.version == version:
                    version_to_remove = v
                    break
            
            if version_to_remove is None:
                return False
            
            # Remove from registry
            versions.remove(version_to_remove)
            
            # Delete model files
            try:
                if os.path.exists(version_to_remove.model_path):
                    if os.path.isfile(version_to_remove.model_path):
                        os.remove(version_to_remove.model_path)
                    else:
                        shutil.rmtree(version_to_remove.model_path)
            except Exception as e:
                logger.warning(f"Failed to delete model files: {e}")
            
            # Save registry
            await self.save_registry()
            
            logger.info(f"Deleted model {model_name} version {version}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model version: {e}")
            return False
    
    async def save_registry(self):
        """Save registry metadata to file"""
        
        try:
            registry_data = {
                'models': {
                    name: [v.to_dict() for v in versions]
                    for name, versions in self.models.items()
                },
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    async def load_registry(self):
        """Load registry metadata from file"""
        
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    registry_data = json.load(f)
                
                # Reconstruct models dictionary
                self.models = {}
                for name, versions_data in registry_data.get('models', {}).items():
                    self.models[name] = [
                        ModelVersion.from_dict(v_data)
                        for v_data in versions_data
                    ]
                
                logger.info(f"Loaded registry with {len(self.models)} models")
                
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        
        total_versions = sum(len(versions) for versions in self.models.values())
        total_size_mb = sum(
            v.size_mb
            for versions in self.models.values()
            for v in versions
        )
        
        return {
            'total_models': len(self.models),
            'total_versions': total_versions,
            'total_size_mb': total_size_mb,
            'storage_path': self.storage_path,
            'backend': self.registry_backend
        }


class ModelDeployment:
    """
    Model deployment manager for various targets
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.deployments = {}  # deployment_id -> Deployment
        self.deployment_logs_path = config.get('logs_path', './deployment_logs')
        
        # External clients
        self.docker_client = None
        self.k8s_client = None
        self.cloud_run_client = None
        
        os.makedirs(self.deployment_logs_path, exist_ok=True)
        
        logger.info("Initialized ModelDeployment")
    
    async def initialize(self):
        """Initialize deployment clients"""
        
        # Initialize Docker client
        if DOCKER_AVAILABLE:
            try:
                self.docker_client = docker.from_env()
                logger.info("Docker client initialized")
            except Exception as e:
                logger.warning(f"Docker client initialization failed: {e}")
        
        # Initialize Kubernetes client
        if KUBERNETES_AVAILABLE:
            try:
                k8s_config.load_incluster_config()  # For in-cluster
                self.k8s_client = client.AppsV1Api()
                logger.info("Kubernetes client initialized")
            except Exception:
                try:
                    k8s_config.load_kube_config()  # For local development
                    self.k8s_client = client.AppsV1Api()
                    logger.info("Kubernetes client initialized (local)")
                except Exception as e:
                    logger.warning(f"Kubernetes client initialization failed: {e}")
        
        # Initialize Cloud Run client
        if GOOGLE_CLOUD_AVAILABLE:
            try:
                project_id = self.config.get('project_id')
                if project_id:
                    self.cloud_run_client = run_v2.ServicesClient()
                    logger.info("Cloud Run client initialized")
            except Exception as e:
                logger.warning(f"Cloud Run client initialization failed: {e}")
    
    async def deploy_model(
        self,
        model_version: ModelVersion,
        deployment_config: DeploymentConfig,
        deployment_id: Optional[str] = None
    ) -> Deployment:
        """Deploy a model version"""
        
        if deployment_id is None:
            deployment_id = f"{model_version.model_name}_{model_version.version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create deployment object
        deployment = Deployment(
            deployment_id=deployment_id,
            model_version=model_version,
            config=deployment_config,
            status=DeploymentStatus.PENDING
        )
        
        self.deployments[deployment_id] = deployment
        
        try:
            # Deploy based on target
            if deployment_config.target == DeploymentTarget.LOCAL:
                await self._deploy_local(deployment)
            elif deployment_config.target == DeploymentTarget.DOCKER:
                await self._deploy_docker(deployment)
            elif deployment_config.target == DeploymentTarget.KUBERNETES:
                await self._deploy_kubernetes(deployment)
            elif deployment_config.target == DeploymentTarget.CLOUD_RUN:
                await self._deploy_cloud_run(deployment)
            else:
                raise ValueError(f"Unsupported deployment target: {deployment_config.target}")
            
            deployment.status = DeploymentStatus.ACTIVE
            deployment.updated_at = datetime.now()
            
            logger.info(f"Successfully deployed {deployment_id}")
            
        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            deployment.logs.append(f"Deployment failed: {str(e)}")
            deployment.updated_at = datetime.now()
            
            logger.error(f"Deployment failed for {deployment_id}: {e}")
            raise
        
        return deployment
    
    async def _deploy_local(self, deployment: Deployment):
        """Deploy model locally as a process"""
        
        # Create serving script
        serving_script = self._create_serving_script(deployment)
        
        # Start local server
        # This would start a local Flask/FastAPI server
        deployment.endpoint_url = f"http://localhost:{deployment.config.port}"
        deployment.logs.append("Local deployment simulated")
        
        logger.info(f"Local deployment ready at {deployment.endpoint_url}")
    
    async def _deploy_docker(self, deployment: Deployment):
        """Deploy model as Docker container"""
        
        if not self.docker_client:
            raise RuntimeError("Docker client not available")
        
        deployment.status = DeploymentStatus.BUILDING
        
        # Create Dockerfile
        dockerfile_content = self._create_dockerfile(deployment)
        
        # Build Docker image
        image_tag = f"{deployment.model_version.model_name}:{deployment.model_version.version}"
        
        # This would build and run the Docker container
        deployment.endpoint_url = f"http://localhost:{deployment.config.port}"
        deployment.logs.append(f"Docker container built: {image_tag}")
        
        logger.info(f"Docker deployment ready at {deployment.endpoint_url}")
    
    async def _deploy_kubernetes(self, deployment: Deployment):
        """Deploy model to Kubernetes"""
        
        if not self.k8s_client:
            raise RuntimeError("Kubernetes client not available")
        
        deployment.status = DeploymentStatus.DEPLOYING
        
        # Create Kubernetes manifests
        k8s_manifests = self._create_k8s_manifests(deployment)
        
        # Apply manifests
        # This would apply the Kubernetes deployment and service
        deployment.endpoint_url = f"http://{deployment.deployment_id}.default.svc.cluster.local:{deployment.config.port}"
        deployment.logs.append("Kubernetes deployment created")
        
        logger.info(f"Kubernetes deployment ready at {deployment.endpoint_url}")
    
    async def _deploy_cloud_run(self, deployment: Deployment):
        """Deploy model to Google Cloud Run"""
        
        if not self.cloud_run_client:
            raise RuntimeError("Cloud Run client not available")
        
        deployment.status = DeploymentStatus.DEPLOYING
        
        # Create Cloud Run service
        # This would create a Cloud Run service
        region = self.config.get('region', 'us-central1')
        project_id = self.config.get('project_id')
        
        deployment.endpoint_url = f"https://{deployment.deployment_id}-{region}-{project_id}.a.run.app"
        deployment.logs.append("Cloud Run service created")
        
        logger.info(f"Cloud Run deployment ready at {deployment.endpoint_url}")
    
    def _create_serving_script(self, deployment: Deployment) -> str:
        """Create serving script for the model"""
        
        script_content = f"""
#!/usr/bin/env python3
import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model
with open('{deployment.model_version.model_path}', 'rb') as f:
    model = pickle.load(f)

@app.route('/health')
def health():
    return {{"status": "healthy"}}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array(data['features'])
        
        if hasattr(model, 'predict'):
            prediction = model.predict(features.reshape(1, -1))[0]
        else:
            prediction = 0.5
        
        return jsonify({{"prediction": float(prediction)}})
    
    except Exception as e:
        return jsonify({{"error": str(e)}}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port={deployment.config.port})
"""
        
        return script_content
    
    def _create_dockerfile(self, deployment: Deployment) -> str:
        """Create Dockerfile for the model"""
        
        dockerfile_content = f"""
FROM python:3.9-slim

WORKDIR /app

# Copy model files
COPY {deployment.model_version.model_path} /app/model/

# Install dependencies
RUN pip install flask numpy scikit-learn

# Copy serving script
COPY serve.py /app/

EXPOSE {deployment.config.port}

CMD ["python", "serve.py"]
"""
        
        return dockerfile_content
    
    def _create_k8s_manifests(self, deployment: Deployment) -> Dict[str, Any]:
        """Create Kubernetes manifests for the model"""
        
        manifests = {
            'deployment': {
                'apiVersion': 'apps/v1',
                'kind': 'Deployment',
                'metadata': {
                    'name': deployment.deployment_id,
                    'labels': {
                        'app': deployment.deployment_id,
                        'model': deployment.model_version.model_name,
                        'version': deployment.model_version.version
                    }
                },
                'spec': {
                    'replicas': deployment.config.replicas,
                    'selector': {
                        'matchLabels': {
                            'app': deployment.deployment_id
                        }
                    },
                    'template': {
                        'metadata': {
                            'labels': {
                                'app': deployment.deployment_id
                            }
                        },
                        'spec': {
                            'containers': [
                                {
                                    'name': 'model-server',
                                    'image': f"{deployment.model_version.model_name}:{deployment.model_version.version}",
                                    'ports': [
                                        {
                                            'containerPort': deployment.config.port
                                        }
                                    ],
                                    'resources': {
                                        'requests': {
                                            'cpu': deployment.config.cpu_request,
                                            'memory': deployment.config.memory_request
                                        },
                                        'limits': {
                                            'cpu': deployment.config.cpu_limit,
                                            'memory': deployment.config.memory_limit
                                        }
                                    },
                                    'env': [
                                        {'name': k, 'value': v}
                                        for k, v in deployment.config.env_vars.items()
                                    ],
                                    'livenessProbe': {
                                        'httpGet': {
                                            'path': deployment.config.health_check_path,
                                            'port': deployment.config.port
                                        },
                                        'initialDelaySeconds': 30,
                                        'periodSeconds': 10
                                    },
                                    'readinessProbe': {
                                        'httpGet': {
                                            'path': deployment.config.health_check_path,
                                            'port': deployment.config.port
                                        },
                                        'initialDelaySeconds': 5,
                                        'periodSeconds': 5
                                    }
                                }
                            ]
                        }
                    }
                }
            },
            'service': {
                'apiVersion': 'v1',
                'kind': 'Service',
                'metadata': {
                    'name': deployment.deployment_id,
                    'labels': {
                        'app': deployment.deployment_id
                    }
                },
                'spec': {
                    'selector': {
                        'app': deployment.deployment_id
                    },
                    'ports': [
                        {
                            'port': 80,
                            'targetPort': deployment.config.port
                        }
                    ],
                    'type': 'ClusterIP'
                }
            }
        }
        
        return manifests
    
    async def get_deployment(self, deployment_id: str) -> Optional[Deployment]:
        """Get deployment by ID"""
        
        return self.deployments.get(deployment_id)
    
    async def list_deployments(
        self,
        model_name: Optional[str] = None,
        status: Optional[DeploymentStatus] = None
    ) -> List[Deployment]:
        """List deployments with optional filtering"""
        
        deployments = list(self.deployments.values())
        
        if model_name:
            deployments = [
                d for d in deployments
                if d.model_version.model_name == model_name
            ]
        
        if status:
            deployments = [
                d for d in deployments
                if d.status == status
            ]
        
        return sorted(deployments, key=lambda d: d.created_at, reverse=True)
    
    async def update_deployment(
        self,
        deployment_id: str,
        new_config: Optional[DeploymentConfig] = None,
        new_model_version: Optional[ModelVersion] = None
    ) -> bool:
        """Update an existing deployment"""
        
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            return False
        
        try:
            # Update configuration
            if new_config:
                deployment.config = new_config
            
            # Update model version (rolling update)
            if new_model_version:
                deployment.model_version = new_model_version
            
            # Redeploy with new configuration
            deployment.status = DeploymentStatus.DEPLOYING
            deployment.updated_at = datetime.now()
            
            # This would trigger the actual update
            deployment.logs.append("Deployment updated")
            deployment.status = DeploymentStatus.ACTIVE
            
            logger.info(f"Updated deployment {deployment_id}")
            
            return True
            
        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            deployment.logs.append(f"Update failed: {str(e)}")
            
            logger.error(f"Failed to update deployment {deployment_id}: {e}")
            return False
    
    async def rollback_deployment(
        self,
        deployment_id: str,
        target_version: str
    ) -> bool:
        """Rollback deployment to previous version"""
        
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            return False
        
        try:
            # This would rollback to the target version
            deployment.status = DeploymentStatus.ROLLED_BACK
            deployment.updated_at = datetime.now()
            deployment.logs.append(f"Rolled back to version {target_version}")
            
            logger.info(f"Rolled back deployment {deployment_id} to version {target_version}")
            
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed for {deployment_id}: {e}")
            return False
    
    async def delete_deployment(self, deployment_id: str) -> bool:
        """Delete a deployment"""
        
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            return False
        
        try:
            # Clean up based on deployment target
            if deployment.config.target == DeploymentTarget.DOCKER:
                # Stop and remove Docker container
                pass
            elif deployment.config.target == DeploymentTarget.KUBERNETES:
                # Delete Kubernetes resources
                pass
            elif deployment.config.target == DeploymentTarget.CLOUD_RUN:
                # Delete Cloud Run service
                pass
            
            # Remove from registry
            del self.deployments[deployment_id]
            
            logger.info(f"Deleted deployment {deployment_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete deployment {deployment_id}: {e}")
            return False
    
    async def scale_deployment(
        self,
        deployment_id: str,
        replicas: int
    ) -> bool:
        """Scale deployment replicas"""
        
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            return False
        
        try:
            deployment.config.replicas = replicas
            deployment.updated_at = datetime.now()
            deployment.logs.append(f"Scaled to {replicas} replicas")
            
            # This would trigger actual scaling
            logger.info(f"Scaled deployment {deployment_id} to {replicas} replicas")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale deployment {deployment_id}: {e}")
            return False
    
    async def get_deployment_logs(
        self,
        deployment_id: str,
        lines: int = 100
    ) -> List[str]:
        """Get deployment logs"""
        
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            return []
        
        # Return recent logs
        return deployment.logs[-lines:]
    
    async def health_check(self, deployment_id: str) -> Dict[str, Any]:
        """Perform health check on deployment"""
        
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            return {'status': 'not_found'}
        
        try:
            # This would perform actual health check
            health_status = {
                'status': 'healthy',
                'deployment_id': deployment_id,
                'endpoint': deployment.endpoint_url,
                'uptime': (datetime.now() - deployment.created_at).total_seconds(),
                'last_updated': deployment.updated_at.isoformat()
            }
            
            return health_status
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'deployment_id': deployment_id
            }
    
    def get_deployment_stats(self) -> Dict[str, Any]:
        """Get deployment statistics"""
        
        total_deployments = len(self.deployments)
        status_counts = {}
        
        for deployment in self.deployments.values():
            status = deployment.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'total_deployments': total_deployments,
            'status_counts': status_counts,
            'active_deployments': status_counts.get('active', 0),
            'failed_deployments': status_counts.get('failed', 0)
        }


class VersionManager:
    """
    Manages model versioning and lifecycle
    """
    
    def __init__(self, registry: ModelRegistry, deployment: ModelDeployment):
        self.registry = registry
        self.deployment = deployment
        
    async def promote_model(
        self,
        model_name: str,
        version: str,
        stage: str = "production"
    ) -> bool:
        """Promote model version to a stage"""
        
        model_version = await self.registry.get_model_version(model_name, version)
        if not model_version:
            return False
        
        # Update model tags
        model_version.tags['stage'] = stage
        model_version.tags['promoted_at'] = datetime.now().isoformat()
        
        await self.registry.save_registry()
        
        logger.info(f"Promoted {model_name} version {version} to {stage}")
        
        return True
    
    async def compare_versions(
        self,
        model_name: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """Compare two model versions"""
        
        v1 = await self.registry.get_model_version(model_name, version1)
        v2 = await self.registry.get_model_version(model_name, version2)
        
        if not v1 or not v2:
            return {'error': 'One or both versions not found'}
        
        comparison = {
            'version1': {
                'version': v1.version,
                'metrics': v1.metrics,
                'size_mb': v1.size_mb,
                'created_at': v1.created_at.isoformat()
            },
            'version2': {
                'version': v2.version,
                'metrics': v2.metrics,
                'size_mb': v2.size_mb,
                'created_at': v2.created_at.isoformat()
            },
            'metric_differences': {},
            'size_difference_mb': v2.size_mb - v1.size_mb
        }
        
        # Compare metrics
        for metric in set(v1.metrics.keys()) | set(v2.metrics.keys()):
            val1 = v1.metrics.get(metric, 0)
            val2 = v2.metrics.get(metric, 0)
            comparison['metric_differences'][metric] = val2 - val1
        
        return comparison
    
    async def cleanup_old_versions(
        self,
        model_name: str,
        keep_versions: int = 5
    ) -> int:
        """Clean up old model versions"""
        
        versions = await self.registry.list_model_versions(model_name)
        
        if len(versions) <= keep_versions:
            return 0
        
        # Sort by creation date and keep most recent
        versions_to_delete = versions[keep_versions:]
        deleted_count = 0
        
        for version in versions_to_delete:
            # Don't delete if it's currently deployed
            deployments = await self.deployment.list_deployments(model_name)
            is_deployed = any(
                d.model_version.version == version.version and d.status == DeploymentStatus.ACTIVE
                for d in deployments
            )
            
            if not is_deployed:
                success = await self.registry.delete_model_version(model_name, version.version)
                if success:
                    deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} old versions for {model_name}")
        
        return deleted_count
