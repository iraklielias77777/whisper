"""
Data Loading and Management for ML Training Pipeline
Efficient data loading, batching, and preprocessing for model training
"""

import asyncio
import logging
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Iterator, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

# Google Cloud imports
try:
    from google.cloud import bigquery
    from google.cloud import storage
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False

# Database imports
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

# Feature pipeline import
from ..feature_engineering.feature_pipeline import FeatureEngineeringPipeline

logger = logging.getLogger(__name__)

@dataclass
class DataBatch:
    """Container for a batch of training data"""
    X: np.ndarray
    y: np.ndarray
    metadata: Dict[str, Any]
    batch_id: str
    timestamp: datetime
    
    def __len__(self) -> int:
        return len(self.X)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'X_shape': self.X.shape,
            'y_shape': self.y.shape,
            'metadata': self.metadata,
            'batch_id': self.batch_id,
            'timestamp': self.timestamp.isoformat(),
            'size': len(self)
        }

@dataclass
class DataSplit:
    """Container for train/validation/test splits"""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    metadata: Dict[str, Any]
    
    def get_train_batch(self) -> DataBatch:
        """Get training data as batch"""
        return DataBatch(
            X=self.X_train,
            y=self.y_train,
            metadata={'split': 'train', **self.metadata},
            batch_id=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now()
        )
    
    def get_val_batch(self) -> DataBatch:
        """Get validation data as batch"""
        return DataBatch(
            X=self.X_val,
            y=self.y_val,
            metadata={'split': 'validation', **self.metadata},
            batch_id=f"val_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now()
        )
    
    def get_test_batch(self) -> DataBatch:
        """Get test data as batch"""
        return DataBatch(
            X=self.X_test,
            y=self.y_test,
            metadata={'split': 'test', **self.metadata},
            batch_id=f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now()
        )

class DataLoader(ABC):
    """Abstract base class for data loaders"""
    
    @abstractmethod
    async def load_data(self, **kwargs) -> pd.DataFrame:
        """Load raw data"""
        pass
    
    @abstractmethod
    async def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and labels"""
        pass

class MLDataLoader:
    """
    Comprehensive data loader for ML training with multiple data sources
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.feature_pipeline = None
        
        # Data source clients
        self.bigquery_client = None
        self.storage_client = None
        self.db_pool = None
        
        # Caching
        self.cache_enabled = config.get('cache_enabled', True)
        self.cache_ttl = config.get('cache_ttl', 3600)  # 1 hour
        self.data_cache = {}
        
        # Data processing
        self.batch_size = config.get('batch_size', 1000)
        self.max_workers = config.get('max_workers', 4)
        
        # Initialize feature pipeline
        if 'feature_config' in config:
            self.feature_pipeline = FeatureEngineeringPipeline(config['feature_config'])
        
        logger.info("Initialized MLDataLoader")
    
    async def initialize(self):
        """Initialize data loader connections"""
        
        try:
            # Initialize Google Cloud clients
            if GOOGLE_CLOUD_AVAILABLE:
                project_id = self.config.get('project_id')
                if project_id:
                    self.bigquery_client = bigquery.Client(project=project_id)
                    self.storage_client = storage.Client(project=project_id)
                    logger.info("Google Cloud clients initialized")
            
            # Initialize database connection
            if ASYNCPG_AVAILABLE and 'database' in self.config:
                db_config = self.config['database']
                self.db_pool = await asyncpg.create_pool(**db_config)
                logger.info("Database connection pool created")
            
            logger.info("MLDataLoader initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize MLDataLoader: {e}")
            raise
    
    async def load_training_data(
        self,
        model_type: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Load training data for a specific model type"""
        
        # Set default date range
        if end_date is None:
            end_date = datetime.now()
        
        if start_date is None:
            start_date = end_date - timedelta(days=90)  # 90 days default
        
        # Check cache first
        cache_key = f"{model_type}_{start_date.date()}_{end_date.date()}_{limit}"
        
        if self.cache_enabled and cache_key in self.data_cache:
            cache_entry = self.data_cache[cache_key]
            if (datetime.now() - cache_entry['timestamp']).total_seconds() < self.cache_ttl:
                logger.info(f"Using cached data for {cache_key}")
                return cache_entry['data']
        
        # Load data based on configured source
        data_source = self.config.get('data_source', 'bigquery')
        
        if data_source == 'bigquery':
            data = await self.load_from_bigquery(model_type, start_date, end_date, limit)
        elif data_source == 'database':
            data = await self.load_from_database(model_type, start_date, end_date, limit)
        elif data_source == 'file':
            data = await self.load_from_file(model_type, start_date, end_date, limit)
        else:
            data = await self.generate_sample_data(model_type, start_date, end_date, limit)
        
        # Cache the result
        if self.cache_enabled and data is not None:
            self.data_cache[cache_key] = {
                'data': data,
                'timestamp': datetime.now()
            }
        
        logger.info(f"Loaded {len(data) if data is not None else 0} records for {model_type}")
        
        return data
    
    async def load_from_bigquery(
        self,
        model_type: str,
        start_date: datetime,
        end_date: datetime,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Load data from BigQuery"""
        
        if not self.bigquery_client:
            raise ValueError("BigQuery client not initialized")
        
        # Get model-specific query
        query = self._get_bigquery_query(model_type, start_date, end_date, limit)
        
        try:
            logger.info(f"Executing BigQuery query for {model_type}")
            query_job = self.bigquery_client.query(query)
            df = query_job.to_dataframe()
            
            logger.info(f"BigQuery returned {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"BigQuery load failed: {e}")
            raise
    
    def _get_bigquery_query(
        self,
        model_type: str,
        start_date: datetime,
        end_date: datetime,
        limit: Optional[int] = None
    ) -> str:
        """Get BigQuery query for specific model type"""
        
        dataset = self.config.get('dataset', 'user_whisperer')
        
        if model_type == 'churn_prediction':
            query = f"""
            WITH user_events AS (
                SELECT 
                    user_id,
                    event_type,
                    timestamp,
                    properties,
                    context
                FROM `{dataset}.events`
                WHERE timestamp >= '{start_date.isoformat()}'
                AND timestamp <= '{end_date.isoformat()}'
            ),
            user_profiles AS (
                SELECT 
                    user_id,
                    created_at,
                    last_active_at,
                    lifecycle_stage,
                    engagement_score,
                    subscription_status,
                    CASE 
                        WHEN lifecycle_stage = 'churned' THEN 1 
                        ELSE 0 
                    END as churned
                FROM `{dataset}.user_profiles`
            ),
            user_event_counts AS (
                SELECT 
                    user_id,
                    COUNT(*) as event_count
                FROM user_events
                GROUP BY user_id
                HAVING COUNT(*) >= 10
            )
            SELECT 
                ue.*,
                up.created_at as user_created_at,
                up.last_active_at,
                up.lifecycle_stage,
                up.engagement_score,
                up.subscription_status,
                up.churned
            FROM user_events ue
            JOIN user_profiles up ON ue.user_id = up.user_id
            JOIN user_event_counts uec ON ue.user_id = uec.user_id
            ORDER BY ue.user_id, ue.timestamp
            """
            
        elif model_type == 'content_optimization':
            query = f"""
            WITH message_events AS (
                SELECT 
                    user_id,
                    message_id,
                    content_type,
                    channel,
                    sent_at,
                    opened_at,
                    clicked_at,
                    converted_at,
                    properties
                FROM `{dataset}.message_history`
                WHERE sent_at >= '{start_date.isoformat()}'
                AND sent_at <= '{end_date.isoformat()}'
            )
            SELECT 
                me.*,
                up.engagement_score,
                up.lifecycle_stage,
                CASE 
                    WHEN me.converted_at IS NOT NULL THEN 1.0
                    WHEN me.clicked_at IS NOT NULL THEN 0.7
                    WHEN me.opened_at IS NOT NULL THEN 0.3
                    ELSE 0.0
                END as reward
            FROM message_events me
            JOIN `{dataset}.user_profiles` up ON me.user_id = up.user_id
            ORDER BY me.sent_at
            """
            
        elif model_type == 'timing_optimization':
            query = f"""
            SELECT 
                user_id,
                channel,
                sent_at,
                opened_at,
                clicked_at,
                EXTRACT(HOUR FROM sent_at) as send_hour,
                EXTRACT(DAYOFWEEK FROM sent_at) as send_day,
                CASE 
                    WHEN clicked_at IS NOT NULL THEN 2.0
                    WHEN opened_at IS NOT NULL THEN 1.0
                    ELSE 0.0
                END as engagement_score
            FROM `{dataset}.message_history`
            WHERE sent_at >= '{start_date.isoformat()}'
            AND sent_at <= '{end_date.isoformat()}'
            ORDER BY sent_at
            """
            
        else:
            # Generic query
            query = f"""
            SELECT *
            FROM `{dataset}.events`
            WHERE timestamp >= '{start_date.isoformat()}'
            AND timestamp <= '{end_date.isoformat()}'
            ORDER BY timestamp
            """
        
        # Add limit if specified
        if limit:
            query += f"\nLIMIT {limit}"
        
        return query
    
    async def load_from_database(
        self,
        model_type: str,
        start_date: datetime,
        end_date: datetime,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Load data from PostgreSQL database"""
        
        if not self.db_pool:
            raise ValueError("Database connection not initialized")
        
        # Get model-specific query
        query = self._get_database_query(model_type, start_date, end_date, limit)
        
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query)
                
                # Convert to DataFrame
                if rows:
                    df = pd.DataFrame([dict(row) for row in rows])
                else:
                    df = pd.DataFrame()
                
                logger.info(f"Database returned {len(df)} records")
                return df
                
        except Exception as e:
            logger.error(f"Database load failed: {e}")
            raise
    
    def _get_database_query(
        self,
        model_type: str,
        start_date: datetime,
        end_date: datetime,
        limit: Optional[int] = None
    ) -> str:
        """Get database query for specific model type"""
        
        if model_type == 'churn_prediction':
            query = """
            SELECT 
                e.user_id,
                e.event_type,
                e.timestamp,
                e.properties,
                e.context,
                up.created_at as user_created_at,
                up.last_active_at,
                up.lifecycle_stage,
                up.engagement_score,
                CASE 
                    WHEN up.lifecycle_stage = 'churned' THEN 1 
                    ELSE 0 
                END as churned
            FROM events e
            JOIN user_profiles up ON e.user_id = up.id
            WHERE e.timestamp >= $1 AND e.timestamp <= $2
            AND e.user_id IN (
                SELECT user_id 
                FROM events 
                WHERE timestamp >= $1 AND timestamp <= $2
                GROUP BY user_id 
                HAVING COUNT(*) >= 10
            )
            ORDER BY e.user_id, e.timestamp
            """
            
        elif model_type == 'content_optimization':
            query = """
            SELECT 
                mh.user_id,
                mh.message_id,
                mh.content_type,
                mh.channel,
                mh.sent_at,
                mh.opened_at,
                mh.clicked_at,
                mh.converted_at,
                up.engagement_score,
                CASE 
                    WHEN mh.converted_at IS NOT NULL THEN 1.0
                    WHEN mh.clicked_at IS NOT NULL THEN 0.7
                    WHEN mh.opened_at IS NOT NULL THEN 0.3
                    ELSE 0.0
                END as reward
            FROM message_history mh
            JOIN user_profiles up ON mh.user_id = up.id
            WHERE mh.sent_at >= $1 AND mh.sent_at <= $2
            ORDER BY mh.sent_at
            """
            
        else:
            # Generic query
            query = """
            SELECT *
            FROM events
            WHERE timestamp >= $1 AND timestamp <= $2
            ORDER BY timestamp
            """
        
        # Add limit if specified
        if limit:
            query += f" LIMIT {limit}"
        
        return query
    
    async def load_from_file(
        self,
        model_type: str,
        start_date: datetime,
        end_date: datetime,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Load data from file"""
        
        file_path = self.config.get('file_path')
        if not file_path or not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return await self.generate_sample_data(model_type, start_date, end_date, limit)
        
        try:
            # Load based on file extension
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            # Apply date filter if timestamp column exists
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df[
                    (df['timestamp'] >= start_date) & 
                    (df['timestamp'] <= end_date)
                ]
            
            # Apply limit
            if limit and len(df) > limit:
                df = df.head(limit)
            
            logger.info(f"Loaded {len(df)} records from file")
            return df
            
        except Exception as e:
            logger.error(f"File load failed: {e}")
            return await self.generate_sample_data(model_type, start_date, end_date, limit)
    
    async def generate_sample_data(
        self,
        model_type: str,
        start_date: datetime,
        end_date: datetime,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Generate sample data for testing"""
        
        logger.info(f"Generating sample data for {model_type}")
        
        n_users = min(limit or 1000, 1000)
        n_events_per_user = 50
        
        data = []
        
        for user_id in range(n_users):
            # Generate user profile
            user_created = start_date + timedelta(
                days=np.random.randint(0, (end_date - start_date).days)
            )
            
            # Generate label based on model type
            if model_type == 'churn_prediction':
                churned = np.random.choice([0, 1], p=[0.8, 0.2])
                engagement_score = np.random.uniform(0, 1)
            elif model_type == 'content_optimization':
                churned = 0
                engagement_score = np.random.uniform(0, 1)
            else:
                churned = 0
                engagement_score = np.random.uniform(0, 1)
            
            # Generate events for user
            for event_idx in range(np.random.randint(10, n_events_per_user)):
                event_time = user_created + timedelta(
                    hours=np.random.randint(0, int((end_date - user_created).total_seconds() / 3600))
                )
                
                if event_time > end_date:
                    continue
                
                event_types = ['page_view', 'click', 'purchase', 'login', 'feature_use']
                event_type = np.random.choice(event_types)
                
                row = {
                    'user_id': f'user_{user_id}',
                    'event_type': event_type,
                    'timestamp': event_time,
                    'properties': json.dumps({'value': np.random.uniform(0, 100)}),
                    'context': json.dumps({'device': 'desktop', 'browser': 'chrome'}),
                    'user_created_at': user_created,
                    'engagement_score': engagement_score,
                    'churned': churned
                }
                
                # Add model-specific fields
                if model_type == 'content_optimization':
                    row.update({
                        'message_id': f'msg_{event_idx}',
                        'content_type': np.random.choice(['email', 'push', 'sms']),
                        'channel': np.random.choice(['email', 'mobile', 'web']),
                        'sent_at': event_time,
                        'opened_at': event_time + timedelta(minutes=np.random.randint(1, 60)) if np.random.random() > 0.3 else None,
                        'clicked_at': event_time + timedelta(minutes=np.random.randint(1, 120)) if np.random.random() > 0.7 else None,
                        'reward': np.random.uniform(0, 1)
                    })
                
                elif model_type == 'timing_optimization':
                    row.update({
                        'channel': np.random.choice(['email', 'sms', 'push']),
                        'sent_at': event_time,
                        'opened_at': event_time + timedelta(minutes=np.random.randint(1, 60)) if np.random.random() > 0.4 else None,
                        'clicked_at': event_time + timedelta(minutes=np.random.randint(1, 120)) if np.random.random() > 0.8 else None,
                        'send_hour': event_time.hour,
                        'send_day': event_time.weekday(),
                        'engagement_score': np.random.uniform(0, 2)
                    })
                
                data.append(row)
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} sample records")
        
        return df
    
    async def prepare_features(
        self,
        data: pd.DataFrame,
        model_type: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and labels from raw data"""
        
        if self.feature_pipeline is None:
            raise ValueError("Feature pipeline not initialized")
        
        logger.info(f"Preparing features for {model_type}")
        
        # Group data by user
        if 'user_id' not in data.columns:
            raise ValueError("Data must contain 'user_id' column")
        
        grouped = data.groupby('user_id')
        
        features_list = []
        labels_list = []
        
        for user_id, user_data in grouped:
            try:
                # Extract user profile (first row)
                user_profile = user_data.iloc[0].to_dict()
                
                # Extract events
                events = []
                for _, event_row in user_data.iterrows():
                    event = {
                        'event_type': event_row.get('event_type', ''),
                        'timestamp': event_row.get('timestamp', event_row.get('sent_at', '')),
                        'properties': self._parse_json_field(event_row.get('properties', '{}')),
                        'context': self._parse_json_field(event_row.get('context', '{}'))
                    }
                    events.append(event)
                
                # Extract features using pipeline
                features = self.feature_pipeline.extract_features(
                    user_profile,
                    events
                )
                
                # Convert to array
                feature_array = self._dict_to_array(features)
                
                # Extract label based on model type
                label = self._extract_label(user_profile, model_type)
                
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
    
    def _parse_json_field(self, field: Any) -> Dict:
        """Parse JSON field safely"""
        
        if isinstance(field, str):
            try:
                return json.loads(field)
            except json.JSONDecodeError:
                return {}
        elif isinstance(field, dict):
            return field
        else:
            return {}
    
    def _extract_label(self, user_profile: Dict, model_type: str) -> float:
        """Extract label based on model type"""
        
        if model_type == 'churn_prediction':
            return float(user_profile.get('churned', 0))
        
        elif model_type == 'content_optimization':
            return float(user_profile.get('reward', 0))
        
        elif model_type == 'timing_optimization':
            return float(user_profile.get('engagement_score', 0))
        
        elif model_type == 'engagement_prediction':
            engagement = user_profile.get('engagement_score', 0)
            return 1.0 if engagement > 0.5 else 0.0
        
        else:
            return 0.0
    
    def _dict_to_array(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert feature dictionary to numpy array"""
        
        feature_names = self._get_feature_names()
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
        
        # This should match the feature pipeline output
        feature_names = []
        
        # Temporal features
        temporal_features = [
            'temporal_days_since_signup', 'temporal_hours_since_last_activity',
            'temporal_is_weekend', 'temporal_is_business_hours', 
            'temporal_hour_sin', 'temporal_hour_cos', 'temporal_day_sin', 'temporal_day_cos'
        ]
        feature_names.extend(temporal_features)
        
        # Behavioral features
        behavioral_features = [
            'behavioral_total_events', 'behavioral_unique_event_types',
            'behavioral_events_per_hour', 'behavioral_has_error_pattern',
            'behavioral_has_purchase_intent', 'behavioral_avg_inter_event_time'
        ]
        feature_names.extend(behavioral_features)
        
        # Engagement features
        engagement_features = [
            'engagement_engagement_score', 'engagement_total_sessions',
            'engagement_avg_session_length', 'engagement_features_adopted',
            'engagement_feature_adoption_rate', 'engagement_avg_session_engagement'
        ]
        feature_names.extend(engagement_features)
        
        # Monetization features
        monetization_features = [
            'monetization_is_paid', 'monetization_total_purchases',
            'monetization_total_revenue', 'monetization_is_in_trial',
            'monetization_days_until_trial_end', 'monetization_avg_purchase_value'
        ]
        feature_names.extend(monetization_features)
        
        # Session features
        session_features = [
            'session_total_sessions', 'session_avg_session_length',
            'session_max_session_length', 'session_avg_session_engagement',
            'session_session_completion_rate'
        ]
        feature_names.extend(session_features)
        
        # Content features
        content_features = [
            'content_content_interaction_count', 'content_unique_content_types',
            'content_avg_content_time_seconds', 'content_avg_content_completion_rate'
        ]
        feature_names.extend(content_features)
        
        # Device features
        device_features = [
            'device_unique_devices', 'device_primary_device_mobile',
            'device_primary_device_desktop', 'device_cross_platform_usage'
        ]
        feature_names.extend(device_features)
        
        # Geographic features
        geographic_features = [
            'geographic_unique_countries', 'geographic_location_consistency',
            'geographic_international_usage', 'geographic_timezone_travel'
        ]
        feature_names.extend(geographic_features)
        
        return feature_names
    
    async def create_data_splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True
    ) -> DataSplit:
        """Create train/validation/test splits"""
        
        try:
            from sklearn.model_selection import train_test_split
            SKLEARN_AVAILABLE = True
        except ImportError:
            SKLEARN_AVAILABLE = False
        
        if SKLEARN_AVAILABLE:
            # Use sklearn for splitting
            stratify_array = y if stratify and len(np.unique(y)) > 1 else None
            
            # First split: separate test set
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_array
            )
            
            # Second split: separate train and validation
            val_size_adjusted = val_size / (1 - test_size)
            stratify_temp = y_temp if stratify and len(np.unique(y_temp)) > 1 else None
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size_adjusted,
                random_state=random_state,
                stratify=stratify_temp
            )
        else:
            # Manual splitting
            n_samples = len(X)
            test_size_abs = int(n_samples * test_size)
            val_size_abs = int(n_samples * val_size)
            
            # Shuffle indices
            np.random.seed(random_state)
            indices = np.random.permutation(n_samples)
            
            # Split indices
            test_indices = indices[:test_size_abs]
            val_indices = indices[test_size_abs:test_size_abs + val_size_abs]
            train_indices = indices[test_size_abs + val_size_abs:]
            
            # Create splits
            X_test = X[test_indices]
            y_test = y[test_indices]
            
            X_val = X[val_indices]
            y_val = y[val_indices]
            
            X_train = X[train_indices]
            y_train = y[train_indices]
        
        metadata = {
            'total_samples': len(X),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'test_size': test_size,
            'val_size': val_size,
            'random_state': random_state,
            'stratify': stratify,
            'split_timestamp': datetime.now().isoformat()
        }
        
        return DataSplit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            metadata=metadata
        )
    
    async def create_batch_iterator(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: Optional[int] = None,
        shuffle: bool = True
    ) -> Iterator[DataBatch]:
        """Create iterator for batched data loading"""
        
        if batch_size is None:
            batch_size = self.batch_size
        
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]
            
            batch = DataBatch(
                X=batch_X,
                y=batch_y,
                metadata={
                    'batch_size': len(batch_X),
                    'start_idx': start_idx,
                    'end_idx': end_idx
                },
                batch_id=f"batch_{start_idx}_{end_idx}",
                timestamp=datetime.now()
            )
            
            yield batch
    
    async def save_processed_data(
        self,
        data_split: DataSplit,
        output_path: str
    ):
        """Save processed data splits"""
        
        os.makedirs(output_path, exist_ok=True)
        
        try:
            # Save as numpy arrays
            np.save(os.path.join(output_path, 'X_train.npy'), data_split.X_train)
            np.save(os.path.join(output_path, 'y_train.npy'), data_split.y_train)
            np.save(os.path.join(output_path, 'X_val.npy'), data_split.X_val)
            np.save(os.path.join(output_path, 'y_val.npy'), data_split.y_val)
            np.save(os.path.join(output_path, 'X_test.npy'), data_split.X_test)
            np.save(os.path.join(output_path, 'y_test.npy'), data_split.y_test)
            
            # Save metadata
            metadata = {
                'feature_names': data_split.feature_names,
                'metadata': data_split.metadata,
                'shapes': {
                    'X_train': data_split.X_train.shape,
                    'X_val': data_split.X_val.shape,
                    'X_test': data_split.X_test.shape
                }
            }
            
            with open(os.path.join(output_path, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved processed data to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
            raise
    
    async def load_processed_data(self, input_path: str) -> DataSplit:
        """Load previously processed data splits"""
        
        try:
            # Load arrays
            X_train = np.load(os.path.join(input_path, 'X_train.npy'))
            y_train = np.load(os.path.join(input_path, 'y_train.npy'))
            X_val = np.load(os.path.join(input_path, 'X_val.npy'))
            y_val = np.load(os.path.join(input_path, 'y_val.npy'))
            X_test = np.load(os.path.join(input_path, 'X_test.npy'))
            y_test = np.load(os.path.join(input_path, 'y_test.npy'))
            
            # Load metadata
            with open(os.path.join(input_path, 'metadata.json'), 'r') as f:
                metadata_dict = json.load(f)
            
            feature_names = metadata_dict['feature_names']
            metadata = metadata_dict['metadata']
            
            logger.info(f"Loaded processed data from {input_path}")
            
            return DataSplit(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                feature_names=feature_names,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to load processed data: {e}")
            raise
    
    def get_data_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about the loaded data"""
        
        stats = {
            'total_records': len(data),
            'columns': list(data.columns),
            'data_types': data.dtypes.to_dict(),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
            'null_counts': data.isnull().sum().to_dict(),
            'unique_users': data['user_id'].nunique() if 'user_id' in data.columns else 0,
            'date_range': {
                'start': data['timestamp'].min().isoformat() if 'timestamp' in data.columns else None,
                'end': data['timestamp'].max().isoformat() if 'timestamp' in data.columns else None
            }
        }
        
        # Add numeric column statistics
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            stats['numeric_summary'] = data[numeric_columns].describe().to_dict()
        
        return stats
    
    async def cleanup_cache(self):
        """Clean up data cache"""
        
        current_time = datetime.now()
        
        # Remove expired cache entries
        expired_keys = [
            key for key, value in self.data_cache.items()
            if (current_time - value['timestamp']).total_seconds() > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.data_cache[key]
        
        logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")


class FeatureStore:
    """
    Feature store for caching and serving ML features
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.storage_backend = config.get('storage_backend', 'memory')  # memory, redis, database
        self.ttl = config.get('ttl', 3600)  # 1 hour
        
        # In-memory storage
        self.memory_store = {}
        self.timestamps = {}
        
        # External connections
        self.redis_client = None
        self.db_pool = None
        
        logger.info(f"Initialized FeatureStore with {self.storage_backend} backend")
    
    async def initialize(self):
        """Initialize feature store connections"""
        
        if self.storage_backend == 'redis':
            # Initialize Redis connection
            pass
        elif self.storage_backend == 'database':
            # Initialize database connection
            pass
    
    async def get_features(
        self,
        user_id: str,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Get features for a user"""
        
        features = {}
        
        for feature_name in feature_names:
            value = await self.get_feature(user_id, feature_name)
            features[feature_name] = value
        
        return features
    
    async def get_feature(
        self,
        user_id: str,
        feature_name: str
    ) -> float:
        """Get a single feature value"""
        
        cache_key = f"{user_id}:{feature_name}"
        
        if self.storage_backend == 'memory':
            if cache_key in self.memory_store:
                timestamp = self.timestamps.get(cache_key, datetime.min)
                if (datetime.now() - timestamp).total_seconds() < self.ttl:
                    return self.memory_store[cache_key]
        
        # Feature not in cache or expired
        # This would calculate the feature value
        value = await self._calculate_feature(user_id, feature_name)
        
        # Cache the value
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
        
        if self.storage_backend == 'memory':
            self.memory_store[cache_key] = value
            self.timestamps[cache_key] = datetime.now()
    
    async def _calculate_feature(
        self,
        user_id: str,
        feature_name: str
    ) -> float:
        """Calculate feature value (placeholder)"""
        
        # This would implement actual feature calculation
        return 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get feature store statistics"""
        
        return {
            'storage_backend': self.storage_backend,
            'cached_features': len(self.memory_store),
            'ttl': self.ttl,
            'memory_usage_mb': sum(
                len(str(k)) + len(str(v)) for k, v in self.memory_store.items()
            ) / 1024 / 1024
        }
