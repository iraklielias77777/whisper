"""
Online Learning System for User Whisperer Platform
Real-time model updates, feedback processing, and adaptive learning
"""

import asyncio
import json
import logging
import pickle
import os
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
import numpy as np

# Redis imports
try:
    import redis
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available. Online learning will be limited.")

# Database imports
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

# ML imports
try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Model imports
from ..models.churn_prediction import ChurnPredictionModel
from ..models.content_optimization import ThompsonSamplingBandit, NeuralBandit, Action, Context
from ..models.timing_optimization import TimingOptimizationModel

logger = logging.getLogger(__name__)

@dataclass
class FeedbackEvent:
    """Feedback event structure"""
    user_id: str
    model_name: str
    action_id: str
    prediction: float
    actual_outcome: str
    reward: float
    features: List[float]
    context: Dict[str, Any]
    timestamp: datetime
    is_critical: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackEvent':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

@dataclass
class ModelPerformanceMetrics:
    """Model performance metrics"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    prediction_count: int
    feedback_count: int
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            **asdict(self),
            'last_updated': self.last_updated.isoformat()
        }

class OnlineLearningSystem:
    """
    Real-time online learning system with model updates and feedback processing
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.redis_client = None
        self.async_redis_client = None
        self.db_pool = None
        
        # Model storage
        self.models = {}
        self.model_versions = {}
        self.model_performance = {}
        
        # Feedback processing
        self.feedback_buffer = deque(maxlen=config.get('buffer_size', 10000))
        self.feedback_processors = {}
        
        # Update configuration
        self.update_frequency = config.get('update_frequency', 100)
        self.evaluation_interval = config.get('evaluation_interval', 3600)  # 1 hour
        self.min_feedback_threshold = config.get('min_feedback_threshold', 10)
        
        # Performance thresholds
        self.performance_thresholds = config.get('performance_thresholds', {
            'min_accuracy': 0.7,
            'min_precision': 0.65,
            'min_recall': 0.6,
            'min_f1': 0.62
        })
        
        # System state
        self.is_running = False
        self.last_evaluation = datetime.now()
        
        logger.info("Initialized OnlineLearningSystem")
    
    async def initialize(self):
        """Initialize the online learning system"""
        
        try:
            # Initialize Redis connections
            if REDIS_AVAILABLE:
                redis_config = self.config.get('redis', {})
                self.redis_client = redis.Redis(**redis_config)
                self.async_redis_client = await aioredis.from_url(
                    f"redis://{redis_config.get('host', 'localhost')}:{redis_config.get('port', 6379)}"
                )
                logger.info("Redis connections initialized")
            
            # Initialize database connection
            if ASYNCPG_AVAILABLE:
                db_config = self.config.get('database', {})
                self.db_pool = await asyncpg.create_pool(**db_config)
                logger.info("Database connection pool initialized")
            
            # Load existing models
            await self.load_models()
            
            # Initialize feedback processors
            self._initialize_feedback_processors()
            
            logger.info("OnlineLearningSystem initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize OnlineLearningSystem: {e}")
            raise
    
    async def start(self):
        """Start the online learning system"""
        
        if not self.is_running:
            self.is_running = True
            
            # Start background tasks
            tasks = [
                asyncio.create_task(self.update_loop()),
                asyncio.create_task(self.feedback_processing_loop()),
                asyncio.create_task(self.evaluation_loop()),
                asyncio.create_task(self.cleanup_loop())
            ]
            
            logger.info("OnlineLearningSystem started")
            
            # Wait for tasks to complete (they run indefinitely)
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop(self):
        """Stop the online learning system"""
        
        self.is_running = False
        
        # Save models before stopping
        await self.save_models()
        
        # Close connections
        if self.async_redis_client:
            await self.async_redis_client.close()
        
        if self.db_pool:
            await self.db_pool.close()
        
        logger.info("OnlineLearningSystem stopped")
    
    async def update_loop(self):
        """Main update loop for processing feedback and updating models"""
        
        while self.is_running:
            try:
                # Check if enough feedback accumulated
                if len(self.feedback_buffer) >= self.update_frequency:
                    await self.update_models()
                
                # Process pending feedback
                await self.process_pending_feedback()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Update loop error: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    async def feedback_processing_loop(self):
        """Process incoming feedback from message queue"""
        
        if not self.async_redis_client:
            logger.warning("Redis not available, skipping feedback processing loop")
            return
        
        pubsub = self.async_redis_client.pubsub()
        await pubsub.subscribe('ml_feedback_stream')
        
        try:
            while self.is_running:
                message = await pubsub.get_message(timeout=1.0)
                
                if message and message['type'] == 'message':
                    try:
                        feedback_data = json.loads(message['data'])
                        feedback = FeedbackEvent.from_dict(feedback_data)
                        await self.process_feedback(feedback)
                    except Exception as e:
                        logger.error(f"Failed to process feedback message: {e}")
                
        except Exception as e:
            logger.error(f"Feedback processing loop error: {e}")
        finally:
            await pubsub.unsubscribe('ml_feedback_stream')
    
    async def evaluation_loop(self):
        """Periodic model evaluation loop"""
        
        while self.is_running:
            try:
                # Check if evaluation is due
                if (datetime.now() - self.last_evaluation).total_seconds() >= self.evaluation_interval:
                    await self.evaluate_all_models()
                    self.last_evaluation = datetime.now()
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Evaluation loop error: {e}")
                await asyncio.sleep(600)  # Wait longer on error
    
    async def cleanup_loop(self):
        """Cleanup old data and maintain system health"""
        
        while self.is_running:
            try:
                # Clean old feedback data
                await self.cleanup_old_feedback()
                
                # Clean old model versions
                await self.cleanup_old_model_versions()
                
                # Monitor system health
                await self.monitor_system_health()
                
                # Sleep for 1 hour
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(1800)  # Wait 30 minutes on error
    
    def _initialize_feedback_processors(self):
        """Initialize feedback processors for different model types"""
        
        self.feedback_processors = {
            'churn_prediction': self._process_churn_feedback,
            'content_optimization': self._process_content_feedback,
            'timing_optimization': self._process_timing_feedback
        }
    
    async def process_feedback(self, feedback: FeedbackEvent):
        """Process individual feedback event"""
        
        try:
            # Add to buffer
            self.feedback_buffer.append(feedback)
            
            # Store in Redis for persistence
            if self.async_redis_client:
                await self.async_redis_client.lpush(
                    'feedback_history',
                    json.dumps(feedback.to_dict())
                )
                # Keep only recent feedback
                await self.async_redis_client.ltrim('feedback_history', 0, 9999)
            
            # Immediate processing for critical feedback
            if feedback.is_critical:
                await self.process_critical_feedback(feedback)
            
            # Update model-specific processor
            processor = self.feedback_processors.get(feedback.model_name)
            if processor:
                await processor(feedback)
            
            logger.debug(f"Processed feedback for {feedback.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to process feedback: {e}")
    
    async def process_critical_feedback(self, feedback: FeedbackEvent):
        """Process critical feedback immediately"""
        
        logger.info(f"Processing critical feedback for {feedback.model_name}")
        
        try:
            model = self.models.get(feedback.model_name)
            if not model:
                logger.warning(f"Model {feedback.model_name} not found for critical feedback")
                return
            
            # Immediate model update based on type
            if feedback.model_name == 'content_optimization' and isinstance(model, (ThompsonSamplingBandit, NeuralBandit)):
                # Update bandit with immediate feedback
                action = Action(
                    id=feedback.action_id,
                    features=np.array(feedback.features[:20]),  # Action features
                    metadata={'critical': True}
                )
                
                context = Context(
                    user_features=np.array(feedback.features[:50]),
                    temporal_features=np.array(feedback.features[50:60]),
                    historical_features=np.array(feedback.features[60:80]),
                    channel_features=np.array(feedback.features[80:90])
                )
                
                model.update(action, feedback.reward, context)
                
            elif feedback.model_name == 'timing_optimization' and isinstance(model, TimingOptimizationModel):
                # Update timing model
                send_time = datetime.fromisoformat(feedback.context.get('send_time', datetime.now().isoformat()))
                channel = feedback.context.get('channel', 'email')
                
                model.update_with_feedback(
                    feedback.user_id,
                    channel,
                    send_time,
                    feedback.actual_outcome,
                    feedback.reward
                )
            
            # Save updated model
            await self.save_model(feedback.model_name)
            
        except Exception as e:
            logger.error(f"Critical feedback processing failed: {e}")
    
    async def _process_churn_feedback(self, feedback: FeedbackEvent):
        """Process feedback for churn prediction model"""
        
        # For churn models, feedback is typically delayed
        # Store for batch retraining
        await self.store_training_example(
            feedback.model_name,
            feedback.features,
            1.0 if feedback.actual_outcome == 'churned' else 0.0,
            feedback.timestamp
        )
    
    async def _process_content_feedback(self, feedback: FeedbackEvent):
        """Process feedback for content optimization model"""
        
        model = self.models.get('content_optimization')
        if not model:
            return
        
        # Update bandit model immediately
        try:
            action = Action(
                id=feedback.action_id,
                features=np.array(feedback.features[:20]),
                metadata={'type': feedback.context.get('content_type', 'email')}
            )
            
            context = Context(
                user_features=np.array(feedback.features[:50]),
                temporal_features=np.array(feedback.features[50:60]),
                historical_features=np.array(feedback.features[60:80]),
                channel_features=np.array(feedback.features[80:90])
            )
            
            model.update(action, feedback.reward, context)
            
        except Exception as e:
            logger.error(f"Content feedback processing failed: {e}")
    
    async def _process_timing_feedback(self, feedback: FeedbackEvent):
        """Process feedback for timing optimization model"""
        
        model = self.models.get('timing_optimization')
        if not model:
            return
        
        try:
            send_time = datetime.fromisoformat(
                feedback.context.get('send_time', datetime.now().isoformat())
            )
            channel = feedback.context.get('channel', 'email')
            
            model.update_with_feedback(
                feedback.user_id,
                channel,
                send_time,
                feedback.actual_outcome,
                feedback.reward
            )
            
        except Exception as e:
            logger.error(f"Timing feedback processing failed: {e}")
    
    async def update_models(self):
        """Update models with accumulated feedback"""
        
        if not self.feedback_buffer:
            return
        
        logger.info(f"Updating models with {len(self.feedback_buffer)} feedback events")
        
        try:
            # Group feedback by model
            feedback_by_model = defaultdict(list)
            for feedback in self.feedback_buffer:
                feedback_by_model[feedback.model_name].append(feedback)
            
            # Update each model
            for model_name, feedback_list in feedback_by_model.items():
                await self.update_single_model(model_name, feedback_list)
            
            # Clear processed feedback
            self.feedback_buffer.clear()
            
            # Save updated models
            await self.save_models()
            
            logger.info("Model updates completed")
            
        except Exception as e:
            logger.error(f"Model update failed: {e}")
    
    async def update_single_model(self, model_name: str, feedback_list: List[FeedbackEvent]):
        """Update a single model with its feedback"""
        
        model = self.models.get(model_name)
        if not model:
            logger.warning(f"Model {model_name} not found for update")
            return
        
        logger.info(f"Updating {model_name} with {len(feedback_list)} feedback events")
        
        try:
            if model_name == 'churn_prediction' and isinstance(model, ChurnPredictionModel):
                await self._update_churn_model(model, feedback_list)
            
            elif model_name == 'content_optimization' and isinstance(model, (ThompsonSamplingBandit, NeuralBandit)):
                await self._update_content_model(model, feedback_list)
            
            elif model_name == 'timing_optimization' and isinstance(model, TimingOptimizationModel):
                await self._update_timing_model(model, feedback_list)
            
            else:
                logger.warning(f"Unknown model type for {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to update {model_name}: {e}")
    
    async def _update_churn_model(self, model: ChurnPredictionModel, feedback_list: List[FeedbackEvent]):
        """Update churn prediction model"""
        
        # For neural models, we need to retrain periodically
        # For now, store feedback for batch retraining
        
        training_data = []
        for feedback in feedback_list:
            label = 1.0 if feedback.actual_outcome == 'churned' else 0.0
            training_data.append({
                'features': feedback.features,
                'label': label,
                'timestamp': feedback.timestamp
            })
        
        # Store for batch retraining
        await self.store_training_data(model.__class__.__name__, training_data)
        
        # Schedule retraining if enough data accumulated
        if len(training_data) >= 1000:  # Arbitrary threshold
            await self.schedule_model_retraining(model.__class__.__name__)
    
    async def _update_content_model(self, model: Union[ThompsonSamplingBandit, NeuralBandit], feedback_list: List[FeedbackEvent]):
        """Update content optimization model"""
        
        # Bandits can be updated incrementally
        for feedback in feedback_list:
            try:
                action = Action(
                    id=feedback.action_id,
                    features=np.array(feedback.features[:20]),
                    metadata={'timestamp': feedback.timestamp.isoformat()}
                )
                
                context = Context(
                    user_features=np.array(feedback.features[:50]),
                    temporal_features=np.array(feedback.features[50:60]),
                    historical_features=np.array(feedback.features[60:80]),
                    channel_features=np.array(feedback.features[80:90])
                )
                
                model.update(action, feedback.reward, context)
                
            except Exception as e:
                logger.warning(f"Failed to update content model with feedback: {e}")
    
    async def _update_timing_model(self, model: TimingOptimizationModel, feedback_list: List[FeedbackEvent]):
        """Update timing optimization model"""
        
        for feedback in feedback_list:
            try:
                send_time = datetime.fromisoformat(
                    feedback.context.get('send_time', feedback.timestamp.isoformat())
                )
                channel = feedback.context.get('channel', 'email')
                
                model.update_with_feedback(
                    feedback.user_id,
                    channel,
                    send_time,
                    feedback.actual_outcome,
                    feedback.reward
                )
                
            except Exception as e:
                logger.warning(f"Failed to update timing model with feedback: {e}")
    
    async def evaluate_all_models(self):
        """Evaluate performance of all models"""
        
        logger.info("Evaluating all models")
        
        for model_name, model in self.models.items():
            try:
                metrics = await self.evaluate_model(model_name, model)
                self.model_performance[model_name] = metrics
                
                # Check for performance degradation
                if self._is_performance_degraded(metrics):
                    await self.handle_model_degradation(model_name, metrics)
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
    
    async def evaluate_model(self, model_name: str, model: Any) -> ModelPerformanceMetrics:
        """Evaluate a single model's performance"""
        
        # Get recent test data
        test_data = await self.get_test_data(model_name)
        
        if not test_data:
            # Return placeholder metrics
            return ModelPerformanceMetrics(
                model_name=model_name,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                prediction_count=0,
                feedback_count=0,
                last_updated=datetime.now()
            )
        
        # Make predictions
        predictions = []
        actuals = []
        
        for example in test_data:
            try:
                if model_name == 'churn_prediction':
                    # Prepare sequence for LSTM
                    features = np.array(example['features'])
                    sequence = features.reshape(1, 1, -1)  # Simplified
                    pred, _ = model.predict(sequence)
                    predictions.append(pred[0])
                
                elif model_name in ['content_optimization', 'timing_optimization']:
                    # For bandits and timing, use simple prediction
                    predictions.append(0.5)  # Placeholder
                
                actuals.append(example['label'])
                
            except Exception as e:
                logger.warning(f"Prediction failed for {model_name}: {e}")
                continue
        
        # Calculate metrics
        if predictions and actuals and SKLEARN_AVAILABLE:
            # Convert to binary predictions
            binary_preds = [1 if p > 0.5 else 0 for p in predictions]
            
            accuracy = accuracy_score(actuals, binary_preds)
            precision = precision_score(actuals, binary_preds, average='binary', zero_division=0)
            recall = recall_score(actuals, binary_preds, average='binary', zero_division=0)
            f1 = f1_score(actuals, binary_preds, average='binary', zero_division=0)
        else:
            # Manual calculation
            correct = sum(1 for p, a in zip(predictions, actuals) if (p > 0.5) == bool(a))
            accuracy = correct / len(predictions) if predictions else 0.0
            precision = recall = f1 = accuracy  # Simplified
        
        metrics = ModelPerformanceMetrics(
            model_name=model_name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            prediction_count=len(predictions),
            feedback_count=len(self.feedback_buffer),
            last_updated=datetime.now()
        )
        
        # Store metrics
        if self.async_redis_client:
            await self.async_redis_client.set(
                f"model_metrics:{model_name}",
                json.dumps(metrics.to_dict()),
                ex=86400  # 24 hours
            )
        
        return metrics
    
    def _is_performance_degraded(self, metrics: ModelPerformanceMetrics) -> bool:
        """Check if model performance has degraded"""
        
        thresholds = self.performance_thresholds
        
        degraded = (
            metrics.accuracy < thresholds['min_accuracy'] or
            metrics.precision < thresholds['min_precision'] or
            metrics.recall < thresholds['min_recall'] or
            metrics.f1_score < thresholds['min_f1']
        )
        
        return degraded
    
    async def handle_model_degradation(self, model_name: str, metrics: ModelPerformanceMetrics):
        """Handle degraded model performance"""
        
        logger.warning(f"Model {model_name} performance degraded: {metrics.to_dict()}")
        
        try:
            # Rollback to previous version
            await self.rollback_model(model_name)
            
            # Schedule full retraining
            await self.schedule_model_retraining(model_name)
            
            # Send alert
            await self.send_performance_alert(model_name, metrics)
            
        except Exception as e:
            logger.error(f"Failed to handle degradation for {model_name}: {e}")
    
    async def rollback_model(self, model_name: str):
        """Rollback model to previous version"""
        
        logger.info(f"Rolling back {model_name} to previous version")
        
        try:
            # Get previous version
            versions = self.model_versions.get(model_name, [])
            if len(versions) < 2:
                logger.warning(f"No previous version available for {model_name}")
                return
            
            # Load previous version
            previous_version = versions[-2]
            await self.load_model_version(model_name, previous_version)
            
            logger.info(f"Rolled back {model_name} to version {previous_version}")
            
        except Exception as e:
            logger.error(f"Rollback failed for {model_name}: {e}")
    
    async def schedule_model_retraining(self, model_name: str):
        """Schedule full model retraining"""
        
        logger.info(f"Scheduling retraining for {model_name}")
        
        # This would trigger the training pipeline
        # For now, just log the request
        retraining_request = {
            'model_name': model_name,
            'reason': 'performance_degradation',
            'timestamp': datetime.now().isoformat(),
            'priority': 'high'
        }
        
        if self.async_redis_client:
            await self.async_redis_client.lpush(
                'retraining_queue',
                json.dumps(retraining_request)
            )
    
    async def send_performance_alert(self, model_name: str, metrics: ModelPerformanceMetrics):
        """Send performance degradation alert"""
        
        alert = {
            'type': 'model_performance_degradation',
            'model_name': model_name,
            'metrics': metrics.to_dict(),
            'timestamp': datetime.now().isoformat(),
            'severity': 'high'
        }
        
        logger.warning(f"Performance alert: {alert}")
        
        # This would send to monitoring system
        if self.async_redis_client:
            await self.async_redis_client.publish('alerts', json.dumps(alert))
    
    async def get_test_data(self, model_name: str, limit: int = 1000) -> List[Dict]:
        """Get test data for model evaluation"""
        
        if not self.db_pool:
            return []
        
        try:
            async with self.db_pool.acquire() as conn:
                # Get recent labeled data
                query = """
                    SELECT features, label, timestamp
                    FROM model_test_data
                    WHERE model_name = $1
                    AND timestamp > NOW() - INTERVAL '7 days'
                    ORDER BY timestamp DESC
                    LIMIT $2
                """
                
                rows = await conn.fetch(query, model_name, limit)
                
                return [
                    {
                        'features': json.loads(row['features']) if isinstance(row['features'], str) else row['features'],
                        'label': row['label'],
                        'timestamp': row['timestamp']
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Failed to get test data for {model_name}: {e}")
            return []
    
    async def store_training_data(self, model_name: str, training_data: List[Dict]):
        """Store training data for batch retraining"""
        
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                # Insert training examples
                for example in training_data:
                    await conn.execute(
                        """
                        INSERT INTO model_training_data (model_name, features, label, timestamp)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT DO NOTHING
                        """,
                        model_name,
                        json.dumps(example['features']),
                        example['label'],
                        example['timestamp']
                    )
                    
        except Exception as e:
            logger.error(f"Failed to store training data: {e}")
    
    async def store_training_example(self, model_name: str, features: List[float], label: float, timestamp: datetime):
        """Store a single training example"""
        
        await self.store_training_data(model_name, [{
            'features': features,
            'label': label,
            'timestamp': timestamp
        }])
    
    async def process_pending_feedback(self):
        """Process any pending feedback from Redis"""
        
        if not self.async_redis_client:
            return
        
        try:
            # Get pending feedback
            feedback_data = await self.async_redis_client.lrange('pending_feedback', 0, 99)
            
            for data in feedback_data:
                try:
                    feedback = FeedbackEvent.from_dict(json.loads(data))
                    await self.process_feedback(feedback)
                except Exception as e:
                    logger.error(f"Failed to process pending feedback: {e}")
            
            # Remove processed feedback
            if feedback_data:
                await self.async_redis_client.ltrim('pending_feedback', len(feedback_data), -1)
                
        except Exception as e:
            logger.error(f"Failed to process pending feedback: {e}")
    
    async def load_models(self):
        """Load existing models from storage"""
        
        if not self.redis_client:
            logger.warning("Redis not available, cannot load models")
            return
        
        try:
            # Get model keys
            model_keys = self.redis_client.keys('model:*')
            
            for key in model_keys:
                try:
                    model_name = key.decode().split(':')[1]
                    model_data = self.redis_client.get(key)
                    
                    if model_data:
                        model = pickle.loads(model_data)
                        self.models[model_name] = model
                        logger.info(f"Loaded model: {model_name}")
                        
                except Exception as e:
                    logger.error(f"Failed to load model from {key}: {e}")
            
            logger.info(f"Loaded {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
    
    async def save_models(self):
        """Save all models to storage"""
        
        for model_name in self.models:
            await self.save_model(model_name)
    
    async def save_model(self, model_name: str):
        """Save a single model to storage"""
        
        if not self.redis_client:
            return
        
        model = self.models.get(model_name)
        if not model:
            return
        
        try:
            # Serialize model
            model_data = pickle.dumps(model)
            
            # Create version
            version = datetime.now().strftime('%Y%m%d%H%M%S')
            
            # Save current version
            self.redis_client.set(
                f'model:{model_name}',
                model_data,
                ex=86400 * 7  # 7 days TTL
            )
            
            # Save versioned backup
            self.redis_client.set(
                f'model:{model_name}:v{version}',
                model_data,
                ex=86400 * 30  # 30 days TTL
            )
            
            # Update version history
            if model_name not in self.model_versions:
                self.model_versions[model_name] = []
            
            self.model_versions[model_name].append(version)
            
            # Keep only recent versions
            if len(self.model_versions[model_name]) > 10:
                self.model_versions[model_name] = self.model_versions[model_name][-10:]
            
            logger.debug(f"Saved model {model_name} version {version}")
            
        except Exception as e:
            logger.error(f"Failed to save model {model_name}: {e}")
    
    async def load_model_version(self, model_name: str, version: str):
        """Load a specific model version"""
        
        if not self.redis_client:
            return
        
        try:
            model_data = self.redis_client.get(f'model:{model_name}:v{version}')
            
            if model_data:
                model = pickle.loads(model_data)
                self.models[model_name] = model
                logger.info(f"Loaded {model_name} version {version}")
                
        except Exception as e:
            logger.error(f"Failed to load {model_name} version {version}: {e}")
    
    async def cleanup_old_feedback(self):
        """Clean up old feedback data"""
        
        if not self.async_redis_client:
            return
        
        try:
            # Keep only recent feedback
            await self.async_redis_client.ltrim('feedback_history', 0, 9999)
            
            # Clean old training data
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    await conn.execute(
                        """
                        DELETE FROM model_training_data
                        WHERE timestamp < NOW() - INTERVAL '30 days'
                        """
                    )
                    
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    async def cleanup_old_model_versions(self):
        """Clean up old model versions"""
        
        if not self.redis_client:
            return
        
        try:
            # Get all versioned model keys
            keys = self.redis_client.keys('model:*:v*')
            
            # Group by model
            model_versions = defaultdict(list)
            
            for key in keys:
                key_str = key.decode()
                parts = key_str.split(':')
                if len(parts) >= 3:
                    model_name = parts[1]
                    version = parts[2]
                    model_versions[model_name].append((version, key_str))
            
            # Keep only recent versions for each model
            for model_name, versions in model_versions.items():
                # Sort by version (timestamp)
                versions.sort(key=lambda x: x[0])
                
                # Delete old versions (keep last 5)
                if len(versions) > 5:
                    for version, key in versions[:-5]:
                        self.redis_client.delete(key)
                        logger.debug(f"Deleted old model version: {key}")
                        
        except Exception as e:
            logger.error(f"Model version cleanup failed: {e}")
    
    async def monitor_system_health(self):
        """Monitor system health and log metrics"""
        
        try:
            health_metrics = {
                'active_models': len(self.models),
                'feedback_buffer_size': len(self.feedback_buffer),
                'last_evaluation': self.last_evaluation.isoformat(),
                'is_running': self.is_running,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add model performance metrics
            for model_name, metrics in self.model_performance.items():
                health_metrics[f'{model_name}_accuracy'] = metrics.accuracy
                health_metrics[f'{model_name}_f1'] = metrics.f1_score
            
            # Store health metrics
            if self.async_redis_client:
                await self.async_redis_client.set(
                    'online_learning_health',
                    json.dumps(health_metrics),
                    ex=3600  # 1 hour
                )
            
            logger.info(f"System health: {health_metrics}")
            
        except Exception as e:
            logger.error(f"Health monitoring failed: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        
        return {
            'is_running': self.is_running,
            'active_models': list(self.models.keys()),
            'feedback_buffer_size': len(self.feedback_buffer),
            'last_evaluation': self.last_evaluation.isoformat(),
            'model_performance': {
                name: metrics.to_dict()
                for name, metrics in self.model_performance.items()
            },
            'config': {
                'update_frequency': self.update_frequency,
                'evaluation_interval': self.evaluation_interval,
                'performance_thresholds': self.performance_thresholds
            }
        }


class AdaptiveLearningRate:
    """
    Adaptive learning rate scheduler for online learning
    """
    
    def __init__(
        self,
        initial_rate: float = 0.01,
        decay_factor: float = 0.99,
        min_rate: float = 0.0001,
        max_rate: float = 0.1,
        adaptation_window: int = 100
    ):
        self.initial_rate = initial_rate
        self.current_rate = initial_rate
        self.decay_factor = decay_factor
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.adaptation_window = adaptation_window
        
        # Performance tracking
        self.iteration = 0
        self.performance_history = deque(maxlen=adaptation_window)
        self.last_adaptation = 0
        
        logger.info(f"Initialized AdaptiveLearningRate with initial rate {initial_rate}")
    
    def get_rate(self) -> float:
        """Get current learning rate"""
        return self.current_rate
    
    def step(self, performance_metric: Optional[float] = None):
        """Update learning rate based on performance"""
        
        self.iteration += 1
        
        if performance_metric is not None:
            self.performance_history.append(performance_metric)
        
        # Adapt rate every adaptation_window steps
        if (self.iteration - self.last_adaptation) >= self.adaptation_window:
            self._adapt_rate()
            self.last_adaptation = self.iteration
        else:
            # Apply decay
            self.current_rate = max(
                self.min_rate,
                self.current_rate * self.decay_factor
            )
    
    def _adapt_rate(self):
        """Adapt learning rate based on performance history"""
        
        if len(self.performance_history) < 2:
            return
        
        # Calculate performance trend
        recent_performance = list(self.performance_history)[-10:]
        trend = self._calculate_trend(recent_performance)
        
        if trend > 0.01:  # Improving
            # Increase learning rate slightly
            self.current_rate = min(
                self.max_rate,
                self.current_rate * 1.1
            )
        elif trend < -0.01:  # Degrading
            # Decrease learning rate
            self.current_rate = max(
                self.min_rate,
                self.current_rate * 0.9
            )
        # Else: stable performance, keep current rate
        
        logger.debug(f"Adapted learning rate to {self.current_rate:.6f} (trend: {trend:.4f})")
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate performance trend"""
        
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend
        x = np.arange(len(values))
        try:
            slope = np.polyfit(x, values, 1)[0]
            return slope
        except:
            return 0.0
    
    def reset(self):
        """Reset learning rate to initial value"""
        
        self.iteration = 0
        self.current_rate = self.initial_rate
        self.performance_history.clear()
        self.last_adaptation = 0
        
        logger.info("Reset learning rate to initial value")


class OnlineFeatureStore:
    """
    Real-time feature store for online learning
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.redis_client = None
        self.feature_ttl = config.get('feature_ttl', 3600)  # 1 hour
        self.cache_size = config.get('cache_size', 10000)
        
        # Local cache
        self.local_cache = {}
        self.cache_timestamps = {}
        
        logger.info("Initialized OnlineFeatureStore")
    
    async def initialize(self):
        """Initialize feature store"""
        
        if REDIS_AVAILABLE:
            redis_config = self.config.get('redis', {})
            self.redis_client = await aioredis.from_url(
                f"redis://{redis_config.get('host', 'localhost')}:{redis_config.get('port', 6379)}"
            )
    
    async def get_features(
        self,
        user_id: str,
        feature_names: List[str]
    ) -> np.ndarray:
        """Get features for a user"""
        
        features = []
        
        for feature_name in feature_names:
            value = await self.get_feature(user_id, feature_name)
            features.append(value)
        
        return np.array(features)
    
    async def get_feature(
        self,
        user_id: str,
        feature_name: str
    ) -> float:
        """Get a single feature value"""
        
        cache_key = f"{user_id}:{feature_name}"
        
        # Check local cache first
        if cache_key in self.local_cache:
            timestamp = self.cache_timestamps.get(cache_key, datetime.min)
            if (datetime.now() - timestamp).total_seconds() < self.feature_ttl:
                return self.local_cache[cache_key]
        
        # Check Redis cache
        if self.redis_client:
            redis_key = f"feature:{user_id}:{feature_name}"
            
            try:
                value = await self.redis_client.get(redis_key)
                if value is not None:
                    float_value = float(value)
                    self._update_local_cache(cache_key, float_value)
                    return float_value
            except Exception as e:
                logger.warning(f"Redis get failed for {redis_key}: {e}")
        
        # Calculate feature on-demand
        value = await self.calculate_feature(user_id, feature_name)
        
        # Cache the calculated value
        await self.set_feature(user_id, feature_name, value)
        
        return value
    
    async def set_feature(
        self,
        user_id: str,
        feature_name: str,
        value: float
    ):
        """Set a feature value"""
        
        cache_key = f"{user_id}:{feature_name}"
        
        # Update local cache
        self._update_local_cache(cache_key, value)
        
        # Update Redis cache
        if self.redis_client:
            redis_key = f"feature:{user_id}:{feature_name}"
            
            try:
                await self.redis_client.setex(
                    redis_key,
                    self.feature_ttl,
                    str(value)
                )
            except Exception as e:
                logger.warning(f"Redis set failed for {redis_key}: {e}")
    
    def _update_local_cache(self, cache_key: str, value: float):
        """Update local cache with size management"""
        
        # Remove old entries if cache is full
        if len(self.local_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = min(
                self.cache_timestamps.keys(),
                key=lambda k: self.cache_timestamps[k]
            )
            del self.local_cache[oldest_key]
            del self.cache_timestamps[oldest_key]
        
        # Add new entry
        self.local_cache[cache_key] = value
        self.cache_timestamps[cache_key] = datetime.now()
    
    async def calculate_feature(
        self,
        user_id: str,
        feature_name: str
    ) -> float:
        """Calculate feature value on-demand"""
        
        # This would implement actual feature calculation
        # For now, return a placeholder value
        
        # Simulate different feature types
        if 'engagement' in feature_name:
            return np.random.uniform(0, 1)
        elif 'activity' in feature_name:
            return np.random.uniform(0, 100)
        elif 'days_since' in feature_name:
            return np.random.uniform(0, 365)
        else:
            return 0.0
    
    async def batch_update_features(
        self,
        updates: List[Tuple[str, str, float]]  # (user_id, feature_name, value)
    ):
        """Batch update multiple features"""
        
        for user_id, feature_name, value in updates:
            await self.set_feature(user_id, feature_name, value)
    
    async def invalidate_user_features(self, user_id: str):
        """Invalidate all cached features for a user"""
        
        # Remove from local cache
        keys_to_remove = [k for k in self.local_cache.keys() if k.startswith(f"{user_id}:")]
        for key in keys_to_remove:
            del self.local_cache[key]
            del self.cache_timestamps[key]
        
        # Remove from Redis
        if self.redis_client:
            pattern = f"feature:{user_id}:*"
            
            try:
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
            except Exception as e:
                logger.warning(f"Redis invalidation failed for pattern {pattern}: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        
        return {
            'local_cache_size': len(self.local_cache),
            'max_cache_size': self.cache_size,
            'feature_ttl': self.feature_ttl,
            'redis_available': self.redis_client is not None
        }


class FeedbackProcessor:
    """
    Processes and validates feedback events
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.validation_rules = config.get('validation_rules', {})
        
    def validate_feedback(self, feedback: FeedbackEvent) -> Tuple[bool, str]:
        """Validate feedback event"""
        
        # Check required fields
        if not feedback.user_id:
            return False, "Missing user_id"
        
        if not feedback.model_name:
            return False, "Missing model_name"
        
        if not feedback.features:
            return False, "Missing features"
        
        # Check feature dimensions
        expected_dim = self.validation_rules.get('feature_dimension')
        if expected_dim and len(feedback.features) != expected_dim:
            return False, f"Feature dimension mismatch: expected {expected_dim}, got {len(feedback.features)}"
        
        # Check reward range
        min_reward = self.validation_rules.get('min_reward', -1.0)
        max_reward = self.validation_rules.get('max_reward', 1.0)
        
        if not (min_reward <= feedback.reward <= max_reward):
            return False, f"Reward out of range: {feedback.reward}"
        
        return True, "Valid"
    
    def preprocess_feedback(self, feedback: FeedbackEvent) -> FeedbackEvent:
        """Preprocess feedback event"""
        
        # Normalize features
        if feedback.features:
            features = np.array(feedback.features)
            
            # Clip outliers
            features = np.clip(features, -10, 10)
            
            # Normalize if needed
            if self.config.get('normalize_features', False):
                if np.std(features) > 0:
                    features = (features - np.mean(features)) / np.std(features)
            
            feedback.features = features.tolist()
        
        # Normalize reward
        if self.config.get('normalize_reward', False):
            feedback.reward = max(0.0, min(1.0, feedback.reward))
        
        return feedback
