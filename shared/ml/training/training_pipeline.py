"""
Model Training Pipeline for User Whisperer Platform
End-to-end training infrastructure with MLflow integration and deployment
"""

import asyncio
import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
import pandas as pd

# ML and validation imports
try:
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, mean_squared_error, mean_absolute_error,
        classification_report, confusion_matrix
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# MLflow imports
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.tensorflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available. Model tracking will be limited.")

# Google Cloud imports
try:
    from google.cloud import bigquery
    from google.cloud import storage
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False
    logging.warning("Google Cloud not available. Data extraction will be limited.")

# Feature engineering and model imports
from ..feature_engineering.feature_pipeline import FeatureEngineeringPipeline
from ..models.churn_prediction import ChurnPredictionModel
from ..models.content_optimization import ThompsonSamplingBandit, NeuralBandit
from ..models.timing_optimization import TimingOptimizationModel

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    experiment_name: str
    model_type: str
    data_source: str
    training_period_days: int = 90
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    
    # Model-specific parameters
    model_params: Dict[str, Any] = None
    
    # Training parameters
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 0.001
    early_stopping_patience: int = 5
    
    # Data parameters
    use_time_split: bool = True
    min_samples_per_user: int = 10
    max_features: Optional[int] = None
    
    # Deployment parameters
    deployment_thresholds: Dict[str, float] = None
    auto_deploy: bool = False
    
    # MLflow parameters
    mlflow_uri: str = "http://localhost:5000"
    mlflow_experiment: str = "user_whisperer_training"
    
    # BigQuery parameters
    bigquery_project: str = ""
    bigquery_dataset: str = ""
    
    def __post_init__(self):
        if self.model_params is None:
            self.model_params = {}
        
        if self.deployment_thresholds is None:
            self.deployment_thresholds = {
                'auc': 0.75,
                'precision': 0.70,
                'recall': 0.65,
                'f1': 0.67
            }

@dataclass
class TrainingMetrics:
    """Training metrics container"""
    model_name: str
    training_accuracy: float
    validation_accuracy: float
    test_accuracy: float
    training_loss: float
    validation_loss: float
    auc_score: float
    precision: float
    recall: float
    f1_score: float
    training_time: float
    inference_time: float
    model_size_mb: float
    feature_count: int
    sample_count: int
    timestamp: datetime

@dataclass
class ValidationStrategy:
    """Validation strategy configuration"""
    method: str = "holdout"  # "holdout", "cross_validation", "time_series"
    cv_folds: int = 5
    time_split_folds: int = 5
    validation_size: float = 0.2
    test_size: float = 0.2
    stratify: bool = True
    random_state: int = 42

class ModelTrainingPipeline:
    """
    End-to-end model training pipeline with MLflow integration
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.bigquery_client = None
        self.storage_client = None
        self.feature_pipeline = None
        self.models = {}
        self.training_results = {}
        
        # Initialize clients
        self._initialize_clients()
        
        # Initialize feature pipeline
        self.feature_pipeline = FeatureEngineeringPipeline(config.model_params)
        
        logger.info(f"Initialized ModelTrainingPipeline for {config.model_type}")
    
    def _initialize_clients(self):
        """Initialize external service clients"""
        
        try:
            if GOOGLE_CLOUD_AVAILABLE and self.config.bigquery_project:
                self.bigquery_client = bigquery.Client(project=self.config.bigquery_project)
                self.storage_client = storage.Client(project=self.config.bigquery_project)
                logger.info("Initialized Google Cloud clients")
        except Exception as e:
            logger.warning(f"Failed to initialize Google Cloud clients: {e}")
        
        try:
            if MLFLOW_AVAILABLE:
                mlflow.set_tracking_uri(self.config.mlflow_uri)
                mlflow.set_experiment(self.config.mlflow_experiment)
                logger.info("Initialized MLflow tracking")
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow: {e}")
    
    async def run_training_pipeline(self) -> Dict[str, Any]:
        """Run complete training pipeline"""
        
        try:
            pipeline_start = datetime.now()
            
            # Start MLflow run
            with mlflow.start_run() if MLFLOW_AVAILABLE else self._dummy_context():
                # Log configuration
                if MLFLOW_AVAILABLE:
                    mlflow.log_params({
                        'model_type': self.config.model_type,
                        'training_period_days': self.config.training_period_days,
                        'test_size': self.config.test_size,
                        'batch_size': self.config.batch_size,
                        'learning_rate': self.config.learning_rate
                    })
                
                # Step 1: Data extraction
                logger.info("Extracting training data...")
                train_data = await self.extract_training_data()
                
                if train_data is None or len(train_data) == 0:
                    raise ValueError("No training data extracted")
                
                if MLFLOW_AVAILABLE:
                    mlflow.log_metric("raw_samples", len(train_data))
                
                # Step 2: Feature engineering
                logger.info("Engineering features...")
                X, y, feature_names = await self.prepare_features(train_data)
                
                if MLFLOW_AVAILABLE:
                    mlflow.log_metrics({
                        "feature_count": X.shape[1],
                        "training_samples": X.shape[0],
                        "positive_class_ratio": y.mean() if len(np.unique(y)) == 2 else 0
                    })
                
                # Step 3: Data splitting
                logger.info("Splitting data...")
                data_splits = self.split_data(X, y)
                
                # Step 4: Train models
                logger.info("Training models...")
                models = await self.train_all_models(data_splits, feature_names)
                
                # Step 5: Evaluate models
                logger.info("Evaluating models...")
                evaluation_results = await self.evaluate_models(models, data_splits)
                
                # Step 6: Select best model
                best_model_info = self.select_best_model(evaluation_results)
                
                # Step 7: Deploy model if criteria met
                if self.config.auto_deploy and self.should_deploy(best_model_info):
                    logger.info("Deploying model...")
                    deployment_info = await self.deploy_model(best_model_info)
                else:
                    deployment_info = None
                
                # Step 8: Log artifacts and results
                await self.log_artifacts(models, evaluation_results, feature_names)
                
                # Calculate total training time
                training_time = (datetime.now() - pipeline_start).total_seconds()
                
                # Compile results
                pipeline_results = {
                    'training_time': training_time,
                    'models_trained': len(models),
                    'best_model': best_model_info,
                    'evaluation_results': evaluation_results,
                    'deployment_info': deployment_info,
                    'feature_count': X.shape[1],
                    'sample_count': X.shape[0],
                    'timestamp': pipeline_start.isoformat()
                }
                
                logger.info(f"Training pipeline completed in {training_time:.2f} seconds")
                return pipeline_results
                
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise
    
    def _dummy_context(self):
        """Dummy context manager when MLflow is not available"""
        from contextlib import nullcontext
        return nullcontext()
    
    async def extract_training_data(self) -> Optional[pd.DataFrame]:
        """Extract training data from configured source"""
        
        if self.config.data_source == "bigquery" and self.bigquery_client:
            return await self._extract_from_bigquery()
        elif self.config.data_source == "file":
            return await self._extract_from_file()
        else:
            return await self._generate_sample_data()
    
    async def _extract_from_bigquery(self) -> pd.DataFrame:
        """Extract training data from BigQuery"""
        
        if not self.bigquery_client:
            raise ValueError("BigQuery client not initialized")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.training_period_days)
        
        query = f"""
        WITH user_events AS (
            SELECT 
                user_id,
                event_type,
                timestamp,
                properties,
                context
            FROM `{self.config.bigquery_project}.{self.config.bigquery_dataset}.events`
            WHERE created_at >= '{start_date.isoformat()}'
            AND created_at <= '{end_date.isoformat()}'
        ),
        user_profiles AS (
            SELECT 
                user_id,
                created_at,
                last_active_at,
                lifecycle_stage,
                engagement_score,
                subscription_status,
                ltv_prediction,
                churn_risk_score,
                CASE 
                    WHEN lifecycle_stage = 'churned' THEN 1 
                    ELSE 0 
                END as churned
            FROM `{self.config.bigquery_project}.{self.config.bigquery_dataset}.user_profiles`
        ),
        user_event_counts AS (
            SELECT 
                user_id,
                COUNT(*) as event_count
            FROM user_events
            GROUP BY user_id
            HAVING COUNT(*) >= {self.config.min_samples_per_user}
        )
        SELECT 
            ue.*,
            up.created_at as user_created_at,
            up.last_active_at,
            up.lifecycle_stage,
            up.engagement_score,
            up.subscription_status,
            up.ltv_prediction,
            up.churn_risk_score,
            up.churned
        FROM user_events ue
        JOIN user_profiles up ON ue.user_id = up.user_id
        JOIN user_event_counts uec ON ue.user_id = uec.user_id
        ORDER BY ue.user_id, ue.timestamp
        """
        
        try:
            df = self.bigquery_client.query(query).to_dataframe()
            logger.info(f"Extracted {len(df)} records from BigQuery")
            return df
        except Exception as e:
            logger.error(f"BigQuery extraction failed: {e}")
            return None
    
    async def _extract_from_file(self) -> pd.DataFrame:
        """Extract training data from file"""
        
        # This would load from a configured file path
        # For now, return sample data
        return await self._generate_sample_data()
    
    async def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample training data for testing"""
        
        logger.info("Generating sample training data")
        
        n_users = 1000
        n_events_per_user = 50
        
        data = []
        
        for user_id in range(n_users):
            # Generate user profile
            user_created = datetime.now() - timedelta(days=np.random.randint(1, 365))
            churned = np.random.choice([0, 1], p=[0.8, 0.2])
            engagement_score = np.random.uniform(0, 1)
            
            # Generate events for user
            for event_idx in range(np.random.randint(10, n_events_per_user)):
                event_time = user_created + timedelta(
                    hours=np.random.randint(0, 8760)  # Random hour in year
                )
                
                event_types = ['page_view', 'click', 'purchase', 'login', 'feature_use']
                event_type = np.random.choice(event_types)
                
                data.append({
                    'user_id': f'user_{user_id}',
                    'event_type': event_type,
                    'timestamp': event_time.isoformat(),
                    'properties': json.dumps({'value': np.random.uniform(0, 100)}),
                    'context': json.dumps({'device': 'desktop', 'browser': 'chrome'}),
                    'user_created_at': user_created.isoformat(),
                    'engagement_score': engagement_score,
                    'churned': churned
                })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} sample records")
        return df
    
    async def prepare_features(
        self,
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and labels from raw data"""
        
        # Group data by user
        grouped = data.groupby('user_id')
        
        features_list = []
        labels_list = []
        
        for user_id, user_data in grouped:
            try:
                # Extract user profile
                user_profile = user_data.iloc[0].to_dict()
                
                # Extract events
                events = []
                for _, event_row in user_data.iterrows():
                    event = {
                        'event_type': event_row['event_type'],
                        'timestamp': event_row['timestamp'],
                        'properties': json.loads(event_row.get('properties', '{}')),
                        'context': json.loads(event_row.get('context', '{}'))
                    }
                    events.append(event)
                
                # Extract features using feature pipeline
                features = self.feature_pipeline.extract_features(
                    user_profile,
                    events
                )
                
                # Convert to array
                feature_array = self._dict_to_array(features)
                
                # Extract label based on model type
                if self.config.model_type == 'churn_prediction':
                    label = user_profile.get('churned', 0)
                elif self.config.model_type == 'engagement_prediction':
                    label = 1 if user_profile.get('engagement_score', 0) > 0.5 else 0
                else:
                    label = 0
                
                features_list.append(feature_array)
                labels_list.append(label)
                
            except Exception as e:
                logger.warning(f"Failed to process user {user_id}: {e}")
                continue
        
        if not features_list:
            raise ValueError("No features extracted from data")
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        # Get feature names
        feature_names = self._get_feature_names()
        
        logger.info(f"Prepared features: {X.shape}, labels: {y.shape}")
        
        return X, y, feature_names
    
    def _dict_to_array(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert feature dictionary to numpy array"""
        
        # Get consistent feature ordering
        feature_names = self._get_feature_names()
        
        # Create array with proper ordering
        feature_array = np.zeros(len(feature_names))
        
        for i, name in enumerate(feature_names):
            if name in features:
                try:
                    feature_array[i] = float(features[name])
                except (ValueError, TypeError):
                    feature_array[i] = 0.0
        
        return feature_array
    
    def _get_feature_names(self) -> List[str]:
        """Get consistent feature names"""
        
        # This would return a consistent list of feature names
        # For now, return a sample list
        feature_names = []
        
        # Add temporal features
        temporal_features = [
            'temporal_days_since_signup', 'temporal_hours_since_last_activity',
            'temporal_is_weekend', 'temporal_is_business_hours', 'temporal_hour_sin', 'temporal_hour_cos'
        ]
        feature_names.extend(temporal_features)
        
        # Add behavioral features
        behavioral_features = [
            'behavioral_total_events', 'behavioral_unique_event_types',
            'behavioral_events_per_hour', 'behavioral_has_error_pattern'
        ]
        feature_names.extend(behavioral_features)
        
        # Add engagement features
        engagement_features = [
            'engagement_engagement_score', 'engagement_total_sessions',
            'engagement_avg_session_length', 'engagement_features_adopted'
        ]
        feature_names.extend(engagement_features)
        
        # Add monetization features
        monetization_features = [
            'monetization_is_paid', 'monetization_total_purchases',
            'monetization_total_revenue', 'monetization_is_in_trial'
        ]
        feature_names.extend(monetization_features)
        
        return feature_names
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Split data for training, validation, and testing"""
        
        if not SKLEARN_AVAILABLE:
            # Simple manual split
            n_samples = len(X)
            test_size = int(n_samples * self.config.test_size)
            val_size = int(n_samples * self.config.validation_size)
            
            X_test = X[-test_size:]
            y_test = y[-test_size:]
            
            X_val = X[-(test_size + val_size):-test_size]
            y_val = y[-(test_size + val_size):-test_size]
            
            X_train = X[:-(test_size + val_size)]
            y_train = y[:-(test_size + val_size)]
            
            return {
                'X_train': X_train, 'y_train': y_train,
                'X_val': X_val, 'y_val': y_val,
                'X_test': X_test, 'y_test': y_test
            }
        
        # Use time-based split for time series data
        if self.config.use_time_split:
            # Assume data is already sorted by time
            n_samples = len(X)
            
            # Split chronologically
            train_end = int(n_samples * (1 - self.config.test_size - self.config.validation_size))
            val_end = int(n_samples * (1 - self.config.test_size))
            
            X_train = X[:train_end]
            y_train = y[:train_end]
            
            X_val = X[train_end:val_end]
            y_val = y[train_end:val_end]
            
            X_test = X[val_end:]
            y_test = y[val_end:]
            
        else:
            # Random split with stratification
            stratify = y if len(np.unique(y)) > 1 else None
            
            # First split: separate test set
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=stratify
            )
            
            # Second split: separate train and validation
            val_size_adjusted = self.config.validation_size / (1 - self.config.test_size)
            stratify_temp = y_temp if len(np.unique(y_temp)) > 1 else None
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size_adjusted,
                random_state=self.config.random_state,
                stratify=stratify_temp
            )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    async def train_all_models(
        self,
        data_splits: Dict[str, np.ndarray],
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Train all configured models"""
        
        models = {}
        
        if self.config.model_type == 'churn_prediction':
            models['churn'] = await self._train_churn_model(data_splits, feature_names)
        
        elif self.config.model_type == 'content_optimization':
            models['content_bandit'] = await self._train_content_optimization_model(data_splits, feature_names)
        
        elif self.config.model_type == 'timing_optimization':
            models['timing'] = await self._train_timing_model(data_splits, feature_names)
        
        elif self.config.model_type == 'all':
            # Train all model types
            models['churn'] = await self._train_churn_model(data_splits, feature_names)
            models['content_bandit'] = await self._train_content_optimization_model(data_splits, feature_names)
            models['timing'] = await self._train_timing_model(data_splits, feature_names)
        
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        return models
    
    async def _train_churn_model(
        self,
        data_splits: Dict[str, np.ndarray],
        feature_names: List[str]
    ) -> ChurnPredictionModel:
        """Train churn prediction model"""
        
        logger.info("Training churn prediction model...")
        
        # Initialize model with config
        model_config = {
            **self.config.model_params,
            'sequence_length': 30,
            'feature_dim': len(feature_names),
            'learning_rate': self.config.learning_rate,
            'batch_size': self.config.batch_size
        }
        
        model = ChurnPredictionModel(model_config)
        
        # Prepare sequences for LSTM
        X_train_seq = self._prepare_sequences_for_lstm(data_splits['X_train'])
        X_val_seq = self._prepare_sequences_for_lstm(data_splits['X_val'])
        
        # Train model
        training_start = datetime.now()
        
        history = model.train(
            X_train_seq,
            data_splits['y_train'],
            X_val_seq,
            data_splits['y_val'],
            epochs=self.config.epochs
        )
        
        training_time = (datetime.now() - training_start).total_seconds()
        
        # Log training metrics
        if MLFLOW_AVAILABLE and history:
            for metric, values in history.items():
                if values:
                    mlflow.log_metric(f"churn_{metric}_final", values[-1])
        
        logger.info(f"Churn model trained in {training_time:.2f} seconds")
        
        return model
    
    async def _train_content_optimization_model(
        self,
        data_splits: Dict[str, np.ndarray],
        feature_names: List[str]
    ) -> ThompsonSamplingBandit:
        """Train content optimization model"""
        
        logger.info("Training content optimization model...")
        
        # Initialize Thompson Sampling bandit
        model_config = {
            'n_actions': 100,
            'feature_dim': len(feature_names),
            'alpha': 1.0,
            'beta': 1.0
        }
        
        model = ThompsonSamplingBandit(**model_config)
        
        # For bandits, we simulate training with historical data
        # In practice, this would use real interaction data
        X_train = data_splits['X_train']
        y_train = data_splits['y_train']
        
        for i in range(min(1000, len(X_train))):
            # Create dummy action and context
            from ..models.content_optimization import Action, Context
            
            action = Action(
                id=f"action_{i % 10}",
                features=X_train[i][:20],  # First 20 features as action features
                metadata={'type': 'email'}
            )
            
            context = Context(
                user_features=X_train[i][:50],
                temporal_features=X_train[i][50:60],
                historical_features=X_train[i][60:80],
                channel_features=X_train[i][80:90]
            )
            
            # Simulate reward based on label
            reward = float(y_train[i])
            
            # Update bandit
            model.update(action, reward, context)
        
        logger.info("Content optimization model training completed")
        
        return model
    
    async def _train_timing_model(
        self,
        data_splits: Dict[str, np.ndarray],
        feature_names: List[str]
    ) -> TimingOptimizationModel:
        """Train timing optimization model"""
        
        logger.info("Training timing optimization model...")
        
        model_config = {
            'seasonality_mode': 'multiplicative',
            'changepoint_prior_scale': 0.05
        }
        
        model = TimingOptimizationModel(model_config)
        
        # For timing models, we would need engagement history data
        # For now, simulate training
        logger.info("Timing optimization model initialized (requires engagement history for training)")
        
        return model
    
    def _prepare_sequences_for_lstm(
        self,
        X: np.ndarray,
        sequence_length: int = 30
    ) -> np.ndarray:
        """Prepare sequences for LSTM models"""
        
        n_samples, n_features = X.shape
        
        # Create sequences by repeating features over time
        # In practice, this would use actual temporal sequences
        sequences = np.zeros((n_samples, sequence_length, n_features))
        
        for i in range(n_samples):
            # Create sequence with slight variations
            base_features = X[i]
            for t in range(sequence_length):
                # Add time-based variation
                noise = np.random.normal(0, 0.01, n_features)
                sequences[i, t] = base_features + noise
        
        return sequences
    
    async def evaluate_models(
        self,
        models: Dict[str, Any],
        data_splits: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate all trained models"""
        
        evaluation_results = {}
        
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name} model...")
            
            try:
                # Prepare test data based on model type
                if model_name == 'churn':
                    X_test = self._prepare_sequences_for_lstm(data_splits['X_test'])
                    y_test = data_splits['y_test']
                    
                    # Make predictions
                    y_pred, y_proba = model.predict(X_test)
                    
                elif model_name in ['content_bandit', 'timing']:
                    # For bandit and timing models, evaluation is different
                    # Use simple metrics for now
                    X_test = data_splits['X_test']
                    y_test = data_splits['y_test']
                    y_pred = np.random.choice([0, 1], size=len(y_test))
                    y_proba = np.random.uniform(0, 1, size=len(y_test))
                
                else:
                    continue
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_test, y_pred, y_proba)
                
                # Calculate inference time
                inference_start = datetime.now()
                if model_name == 'churn':
                    model.predict(X_test[:10])
                inference_time = (datetime.now() - inference_start).total_seconds()
                
                evaluation_results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'predictions': y_pred.tolist(),
                    'probabilities': y_proba.tolist(),
                    'inference_time': inference_time,
                    'test_samples': len(y_test)
                }
                
                # Log metrics to MLflow
                if MLFLOW_AVAILABLE:
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(f"{model_name}_{metric_name}", value)
                
                logger.info(f"{model_name} evaluation completed")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                continue
        
        return evaluation_results
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        
        metrics = {}
        
        try:
            if SKLEARN_AVAILABLE:
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
                metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
                metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
                metrics['f1'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
                
                # AUC only for binary classification
                if len(np.unique(y_true)) == 2:
                    metrics['auc'] = roc_auc_score(y_true, y_proba)
                else:
                    metrics['auc'] = 0.0
            else:
                # Manual calculation
                metrics['accuracy'] = np.mean(y_true == y_pred)
                
                # Simple precision/recall calculation
                tp = np.sum((y_true == 1) & (y_pred == 1))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                fn = np.sum((y_true == 1) & (y_pred == 0))
                
                metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0.0
                metrics['auc'] = 0.5  # Placeholder
                
        except Exception as e:
            logger.warning(f"Metric calculation failed: {e}")
            metrics = {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0
            }
        
        return metrics
    
    def select_best_model(
        self,
        evaluation_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Select best model based on evaluation metrics"""
        
        if not evaluation_results:
            raise ValueError("No evaluation results available")
        
        # Score models based on weighted metrics
        scores = {}
        
        for model_name, results in evaluation_results.items():
            metrics = results['metrics']
            
            # Weighted score combining multiple metrics
            score = (
                metrics.get('auc', 0) * 0.3 +
                metrics.get('f1', 0) * 0.3 +
                metrics.get('precision', 0) * 0.2 +
                metrics.get('recall', 0) * 0.2
            )
            
            scores[model_name] = score
        
        # Get best model
        best_model_name = max(scores, key=scores.get)
        best_model_info = evaluation_results[best_model_name]
        best_model_info['model_name'] = best_model_name
        best_model_info['composite_score'] = scores[best_model_name]
        
        # Log best model info
        if MLFLOW_AVAILABLE:
            mlflow.log_metrics({
                "best_model_score": scores[best_model_name],
                f"best_{best_model_name}_auc": best_model_info['metrics'].get('auc', 0),
                f"best_{best_model_name}_f1": best_model_info['metrics'].get('f1', 0)
            })
            mlflow.log_param("best_model", best_model_name)
        
        logger.info(f"Best model: {best_model_name} (score: {scores[best_model_name]:.4f})")
        
        return best_model_info
    
    def should_deploy(self, best_model_info: Dict[str, Any]) -> bool:
        """Determine if model should be deployed"""
        
        metrics = best_model_info['metrics']
        thresholds = self.config.deployment_thresholds
        
        # Check minimum performance thresholds
        for metric, threshold in thresholds.items():
            if metrics.get(metric, 0) < threshold:
                logger.info(f"Model below threshold for {metric}: {metrics.get(metric, 0):.4f} < {threshold}")
                return False
        
        logger.info("Model meets deployment thresholds")
        return True
    
    async def deploy_model(self, best_model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy model to production"""
        
        model = best_model_info['model']
        model_name = best_model_info['model_name']
        
        logger.info(f"Deploying {model_name} model to production...")
        
        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f"models/{model_name}_{timestamp}"
        
        os.makedirs(model_path, exist_ok=True)
        model.save_model(model_path)
        
        # Register model in MLflow if available
        if MLFLOW_AVAILABLE:
            try:
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name=f"user_whisperer_{model_name}"
                )
                logger.info("Model registered in MLflow")
            except Exception as e:
                logger.warning(f"MLflow model registration failed: {e}")
        
        # Deploy to serving infrastructure
        deployment_info = await self._deploy_to_serving(model_path, model_name)
        
        deployment_result = {
            'model_path': model_path,
            'deployment_timestamp': timestamp,
            'model_name': model_name,
            'metrics': best_model_info['metrics'],
            'deployment_info': deployment_info
        }
        
        logger.info("Model deployment completed")
        
        return deployment_result
    
    async def _deploy_to_serving(self, model_path: str, model_name: str) -> Dict[str, Any]:
        """Deploy model to serving infrastructure"""
        
        # This would implement actual deployment to serving infrastructure
        # For example: Kubernetes, SageMaker, Cloud Run, etc.
        
        deployment_info = {
            'status': 'simulated',
            'endpoint': f"https://api.userwhisperer.com/ml/{model_name}",
            'deployment_method': 'kubernetes',
            'replicas': 3,
            'resource_limits': {
                'cpu': '1000m',
                'memory': '2Gi'
            }
        }
        
        logger.info(f"Model deployed to simulated endpoint: {deployment_info['endpoint']}")
        
        return deployment_info
    
    async def log_artifacts(
        self,
        models: Dict[str, Any],
        evaluation_results: Dict[str, Dict[str, Any]],
        feature_names: List[str]
    ):
        """Log artifacts to MLflow and storage"""
        
        try:
            # Log feature names
            feature_info = {
                'feature_names': feature_names,
                'feature_count': len(feature_names),
                'timestamp': datetime.now().isoformat()
            }
            
            os.makedirs('artifacts', exist_ok=True)
            
            with open('artifacts/feature_info.json', 'w') as f:
                json.dump(feature_info, f, indent=2)
            
            # Log evaluation results
            with open('artifacts/evaluation_results.json', 'w') as f:
                # Remove model objects for JSON serialization
                serializable_results = {}
                for model_name, results in evaluation_results.items():
                    serializable_results[model_name] = {
                        'metrics': results['metrics'],
                        'inference_time': results['inference_time'],
                        'test_samples': results['test_samples']
                    }
                json.dump(serializable_results, f, indent=2)
            
            # Log to MLflow if available
            if MLFLOW_AVAILABLE:
                mlflow.log_artifacts('artifacts')
            
            logger.info("Artifacts logged successfully")
            
        except Exception as e:
            logger.error(f"Failed to log artifacts: {e}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training pipeline"""
        
        return {
            'config': {
                'model_type': self.config.model_type,
                'training_period_days': self.config.training_period_days,
                'test_size': self.config.test_size,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate
            },
            'results': self.training_results,
            'feature_pipeline_stats': self.feature_pipeline.get_feature_stats() if self.feature_pipeline else {},
            'models_trained': len(self.models)
        }
