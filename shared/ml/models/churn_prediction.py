"""
Churn Prediction Model for User Whisperer Platform
LSTM-based model with attention mechanism for predicting user churn probability
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import joblib
import logging
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)

class ChurnPredictionModel:
    """
    LSTM-based churn prediction model with attention mechanism
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.sequence_length = config.get('sequence_length', 30)  # days
        self.feature_dim = config.get('feature_dim', 150)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        
        # Model hyperparameters
        self.hidden_units = config.get('hidden_units', [128, 64])
        self.dropout_rate = config.get('dropout_rate', 0.2)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.l2_regularization = config.get('l2_regularization', 0.01)
        
        # Training parameters
        self.batch_size = config.get('batch_size', 256)
        self.epochs = config.get('epochs', 50)
        self.validation_split = config.get('validation_split', 0.2)
        
        logger.info(f"Initialized ChurnPredictionModel with config: {config}")
        
    def build_model(self) -> keras.Model:
        """Build LSTM model architecture with attention mechanism"""
        
        try:
            # Input layer
            inputs = layers.Input(
                shape=(self.sequence_length, self.feature_dim),
                name='sequence_input'
            )
            
            # First LSTM layer with return sequences
            lstm1 = layers.LSTM(
                units=self.hidden_units[0],
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                kernel_regularizer=keras.regularizers.l2(self.l2_regularization),
                name='lstm_1'
            )(inputs)
            
            # Multi-head attention mechanism
            attention = layers.MultiHeadAttention(
                num_heads=4,
                key_dim=32,
                dropout=0.1,
                name='multi_head_attention'
            )(lstm1, lstm1)
            
            # Add & Norm
            attention_output = layers.Add(name='attention_add')([lstm1, attention])
            attention_output = layers.LayerNormalization(name='attention_norm')(attention_output)
            
            # Second LSTM layer
            lstm2 = layers.LSTM(
                units=self.hidden_units[1] if len(self.hidden_units) > 1 else 64,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                kernel_regularizer=keras.regularizers.l2(self.l2_regularization),
                name='lstm_2'
            )(attention_output)
            
            # Batch normalization
            normalized = layers.BatchNormalization(name='batch_norm')(lstm2)
            
            # Dense layers with dropout
            dense1 = layers.Dense(
                32,
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(self.l2_regularization),
                name='dense_1'
            )(normalized)
            dropout1 = layers.Dropout(0.3, name='dropout_1')(dense1)
            
            dense2 = layers.Dense(
                16,
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(self.l2_regularization),
                name='dense_2'
            )(dropout1)
            dropout2 = layers.Dropout(0.2, name='dropout_2')(dense2)
            
            # Output layer
            outputs = layers.Dense(
                1, 
                activation='sigmoid',
                name='churn_probability'
            )(dropout2)
            
            # Create model
            model = models.Model(inputs=inputs, outputs=outputs, name='churn_prediction_model')
            
            # Custom learning rate schedule
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.learning_rate,
                decay_steps=1000,
                decay_rate=0.9
            )
            
            # Compile model
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                loss='binary_crossentropy',
                metrics=[
                    'accuracy',
                    keras.metrics.AUC(name='auc'),
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.BinaryAccuracy(name='binary_accuracy')
                ]
            )
            
            self.model = model
            logger.info(f"Built model with architecture: {model.summary()}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to build model: {e}")
            raise
    
    def prepare_sequences(
        self,
        user_events: List[Dict],
        window_size: Optional[int] = None
    ) -> np.ndarray:
        """Prepare event sequences for prediction"""
        
        if window_size is None:
            window_size = self.sequence_length
            
        try:
            # Group events by day
            daily_features = {}
            
            for event in user_events:
                try:
                    timestamp_str = event.get('timestamp', event.get('created_at', ''))
                    if not timestamp_str:
                        continue
                        
                    date = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).date()
                    
                    if date not in daily_features:
                        daily_features[date] = self.initialize_daily_features()
                    
                    self.update_daily_features(daily_features[date], event)
                    
                except Exception as e:
                    logger.warning(f"Failed to process event: {e}")
                    continue
            
            # Create sequence
            sequence = []
            current_date = datetime.now().date()
            
            for i in range(window_size):
                date = current_date - timedelta(days=window_size - 1 - i)
                
                if date in daily_features:
                    features = self.extract_feature_vector(daily_features[date])
                else:
                    features = np.zeros(self.feature_dim)
                
                sequence.append(features)
            
            return np.array(sequence)
            
        except Exception as e:
            logger.error(f"Failed to prepare sequences: {e}")
            return np.zeros((window_size, self.feature_dim))
    
    def initialize_daily_features(self) -> Dict:
        """Initialize daily feature structure"""
        
        return {
            'event_count': 0,
            'session_count': 0,
            'unique_event_types': set(),
            'error_count': 0,
            'feature_usage': {},
            'time_spent': 0,
            'page_views': 0,
            'actions_completed': 0,
            'revenue_events': 0,
            'support_events': 0,
            'engagement_events': 0,
            'conversion_events': 0,
            'first_event_hour': None,
            'last_event_hour': None,
            'event_hours': []
        }
    
    def update_daily_features(
        self,
        features: Dict,
        event: Dict
    ):
        """Update daily features with event data"""
        
        try:
            features['event_count'] += 1
            event_type = event.get('event_type', '')
            features['unique_event_types'].add(event_type)
            
            # Error events
            if 'error' in event_type.lower():
                features['error_count'] += 1
            
            # Feature usage
            if event_type.startswith('feature_'):
                feature_name = event_type.replace('feature_', '')
                features['feature_usage'][feature_name] = \
                    features['feature_usage'].get(feature_name, 0) + 1
            
            # Session events
            if event_type == 'session_start':
                features['session_count'] += 1
            
            # Page views
            if event_type == 'page_view':
                features['page_views'] += 1
            
            # Actions
            if event_type in ['click', 'submit', 'download', 'action_completed']:
                features['actions_completed'] += 1
            
            # Revenue events
            if event_type in ['purchase', 'subscription_started', 'payment_completed']:
                features['revenue_events'] += 1
            
            # Support events
            if event_type in ['support_ticket', 'help_viewed', 'contact_support']:
                features['support_events'] += 1
            
            # Engagement events
            if event_type in ['content_viewed', 'feature_used', 'tutorial_completed']:
                features['engagement_events'] += 1
            
            # Conversion events
            if event_type in ['signup_completed', 'trial_started', 'upgrade_completed']:
                features['conversion_events'] += 1
            
            # Time-based features
            if 'timestamp' in event or 'created_at' in event:
                timestamp_str = event.get('timestamp', event.get('created_at', ''))
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    hour = timestamp.hour
                    
                    if features['first_event_hour'] is None:
                        features['first_event_hour'] = hour
                    features['last_event_hour'] = hour
                    features['event_hours'].append(hour)
                    
                except Exception:
                    pass
            
            # Duration from properties
            properties = event.get('properties', {})
            if 'duration' in properties:
                try:
                    duration = float(properties['duration'])
                    features['time_spent'] += duration
                except (ValueError, TypeError):
                    pass
                    
        except Exception as e:
            logger.warning(f"Failed to update daily features: {e}")
    
    def extract_feature_vector(self, daily_features: Dict) -> np.ndarray:
        """Extract feature vector from daily features"""
        
        try:
            vector = np.zeros(self.feature_dim)
            
            # Basic counters (0-19)
            vector[0] = daily_features['event_count']
            vector[1] = daily_features['session_count']
            vector[2] = len(daily_features['unique_event_types'])
            vector[3] = daily_features['error_count']
            vector[4] = daily_features['page_views']
            vector[5] = daily_features['actions_completed']
            vector[6] = daily_features['revenue_events']
            vector[7] = daily_features['support_events']
            vector[8] = daily_features['engagement_events']
            vector[9] = daily_features['conversion_events']
            vector[10] = daily_features['time_spent'] / 3600  # Convert to hours
            
            # Feature usage (11-60)
            for i, (feature, count) in enumerate(
                list(daily_features['feature_usage'].items())[:50]
            ):
                vector[11 + i] = count
            
            # Derived metrics (61-80)
            if daily_features['session_count'] > 0:
                vector[61] = daily_features['event_count'] / daily_features['session_count']
                vector[62] = daily_features['time_spent'] / daily_features['session_count']
            
            if daily_features['event_count'] > 0:
                vector[63] = daily_features['error_count'] / daily_features['event_count']
                vector[64] = daily_features['page_views'] / daily_features['event_count']
                vector[65] = daily_features['actions_completed'] / daily_features['event_count']
            
            # Engagement indicators (81-90)
            vector[81] = 1 if daily_features['event_count'] > 10 else 0
            vector[82] = 1 if daily_features['session_count'] > 2 else 0
            vector[83] = 1 if daily_features['time_spent'] > 1800 else 0  # >30 min
            vector[84] = 1 if daily_features['revenue_events'] > 0 else 0
            vector[85] = 1 if daily_features['error_count'] > 5 else 0
            vector[86] = 1 if len(daily_features['feature_usage']) > 3 else 0
            
            # Time-based features (91-100)
            if daily_features['event_hours']:
                hours = daily_features['event_hours']
                vector[91] = np.mean(hours)  # Average hour
                vector[92] = np.std(hours) if len(hours) > 1 else 0  # Hour spread
                vector[93] = len(set(hours))  # Unique hours
                vector[94] = max(hours) - min(hours) if len(hours) > 1 else 0  # Hour range
            
            # Activity pattern features (101-120)
            vector[101] = 1 if daily_features['first_event_hour'] and daily_features['first_event_hour'] < 9 else 0  # Early starter
            vector[102] = 1 if daily_features['last_event_hour'] and daily_features['last_event_hour'] > 20 else 0  # Late user
            
            # Intensity features (121-140)
            if daily_features['event_count'] > 0:
                vector[121] = daily_features['engagement_events'] / daily_features['event_count']
                vector[122] = daily_features['support_events'] / daily_features['event_count']
                vector[123] = daily_features['conversion_events'] / daily_features['event_count']
            
            # Reserved for additional features (141-149)
            
            return vector
            
        except Exception as e:
            logger.error(f"Failed to extract feature vector: {e}")
            return np.zeros(self.feature_dim)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> Dict:
        """Train the model"""
        
        try:
            if self.model is None:
                self.build_model()
            
            epochs = epochs or self.epochs
            batch_size = batch_size or self.batch_size
            
            logger.info(f"Starting training with {X_train.shape[0]} samples")
            
            # Scale features
            X_train_scaled = self.scale_features(X_train, fit=True)
            
            if X_val is not None:
                X_val_scaled = self.scale_features(X_val, fit=False)
                validation_data = (X_val_scaled, y_val)
            else:
                validation_data = None
            
            # Prepare callbacks
            callbacks_list = self.get_callbacks()
            
            # Calculate class weights for imbalanced data
            class_weights = self.calculate_class_weights(y_train)
            
            # Train
            history = self.model.fit(
                X_train_scaled,
                y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks_list,
                class_weight=class_weights,
                verbose=1
            )
            
            self.is_fitted = True
            logger.info("Training completed successfully")
            
            return history.history
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def get_callbacks(self) -> List[keras.callbacks.Callback]:
        """Get training callbacks"""
        
        callbacks_list = []
        
        # Early stopping
        callbacks_list.append(
            callbacks.EarlyStopping(
                monitor='val_auc' if 'val_auc' in self.model.metrics_names else 'val_loss',
                patience=5,
                restore_best_weights=True,
                mode='max' if 'val_auc' in self.model.metrics_names else 'min',
                verbose=1
            )
        )
        
        # Reduce learning rate on plateau
        callbacks_list.append(
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        )
        
        # Model checkpoint
        os.makedirs('models', exist_ok=True)
        callbacks_list.append(
            callbacks.ModelCheckpoint(
                filepath='models/churn_model_best.h5',
                monitor='val_auc' if 'val_auc' in self.model.metrics_names else 'val_loss',
                save_best_only=True,
                mode='max' if 'val_auc' in self.model.metrics_names else 'min',
                verbose=1
            )
        )
        
        # TensorBoard logging
        log_dir = f'logs/churn/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        os.makedirs(log_dir, exist_ok=True)
        callbacks_list.append(
            callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        )
        
        return callbacks_list
    
    def scale_features(
        self,
        X: np.ndarray,
        fit: bool = False
    ) -> np.ndarray:
        """Scale features for training"""
        
        try:
            # Reshape for scaling
            original_shape = X.shape
            X_reshaped = X.reshape(-1, self.feature_dim)
            
            if fit:
                X_scaled = self.scaler.fit_transform(X_reshaped)
                logger.info("Fitted feature scaler")
            else:
                X_scaled = self.scaler.transform(X_reshaped)
            
            # Reshape back
            return X_scaled.reshape(original_shape)
            
        except Exception as e:
            logger.error(f"Feature scaling failed: {e}")
            return X
    
    def calculate_class_weights(self, y: np.ndarray) -> Dict:
        """Calculate class weights for imbalanced data"""
        
        try:
            classes = np.unique(y)
            weights = compute_class_weight(
                'balanced',
                classes=classes,
                y=y
            )
            
            class_weights = dict(zip(classes.astype(int), weights))
            logger.info(f"Calculated class weights: {class_weights}")
            
            return class_weights
            
        except Exception as e:
            logger.warning(f"Failed to calculate class weights: {e}")
            return {}
    
    def predict(
        self,
        X: np.ndarray,
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions"""
        
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not trained or loaded")
        
        try:
            # Scale features
            X_scaled = self.scale_features(X, fit=False)
            
            # Predict probabilities
            probabilities = self.model.predict(X_scaled, verbose=0)
            
            # Flatten if needed
            if len(probabilities.shape) > 1:
                probabilities = probabilities.flatten()
            
            # Apply threshold
            predictions = (probabilities > threshold).astype(int)
            
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_single(
        self,
        user_events: List[Dict],
        threshold: float = 0.5
    ) -> Tuple[int, float]:
        """Make prediction for a single user"""
        
        try:
            # Prepare sequence
            sequence = self.prepare_sequences(user_events)
            X = sequence.reshape(1, *sequence.shape)
            
            # Predict
            predictions, probabilities = self.predict(X, threshold)
            
            return predictions[0], probabilities[0]
            
        except Exception as e:
            logger.error(f"Single prediction failed: {e}")
            return 0, 0.0
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model performance"""
        
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not trained or loaded")
        
        try:
            # Scale features
            X_test_scaled = self.scale_features(X_test, fit=False)
            
            # Evaluate
            results = self.model.evaluate(X_test_scaled, y_test, verbose=0)
            
            # Create results dictionary
            metrics = {}
            for i, metric in enumerate(self.model.metrics_names):
                metrics[metric] = results[i]
            
            logger.info(f"Evaluation results: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {}
    
    def save_model(self, path: str):
        """Save model and scaler"""
        
        try:
            os.makedirs(path, exist_ok=True)
            
            # Save Keras model
            if self.model:
                self.model.save(f"{path}/churn_model.h5")
                logger.info(f"Saved model to {path}/churn_model.h5")
            
            # Save scaler
            joblib.dump(self.scaler, f"{path}/churn_scaler.pkl")
            logger.info(f"Saved scaler to {path}/churn_scaler.pkl")
            
            # Save config and metadata
            metadata = {
                'config': self.config,
                'feature_names': self.feature_names,
                'is_fitted': self.is_fitted,
                'sequence_length': self.sequence_length,
                'feature_dim': self.feature_dim,
                'model_version': '1.0',
                'saved_at': datetime.now().isoformat()
            }
            
            with open(f"{path}/churn_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved metadata to {path}/churn_metadata.json")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, path: str):
        """Load model and scaler"""
        
        try:
            # Load Keras model
            model_path = f"{path}/churn_model.h5"
            if os.path.exists(model_path):
                self.model = keras.models.load_model(model_path)
                logger.info(f"Loaded model from {model_path}")
            
            # Load scaler
            scaler_path = f"{path}/churn_scaler.pkl"
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Loaded scaler from {scaler_path}")
            
            # Load metadata
            metadata_path = f"{path}/churn_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.config.update(metadata.get('config', {}))
                self.feature_names = metadata.get('feature_names', [])
                self.is_fitted = metadata.get('is_fitted', False)
                self.sequence_length = metadata.get('sequence_length', self.sequence_length)
                self.feature_dim = metadata.get('feature_dim', self.feature_dim)
                
                logger.info(f"Loaded metadata from {metadata_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance (approximation for neural networks)"""
        
        if not self.is_fitted or not self.model:
            return {}
        
        try:
            # For neural networks, we can approximate importance
            # using the magnitude of the first layer weights
            if len(self.model.layers) > 0:
                first_layer = self.model.layers[1]  # Skip input layer
                if hasattr(first_layer, 'get_weights'):
                    weights = first_layer.get_weights()[0]  # Get weight matrix
                    
                    # Calculate average absolute weight for each input feature
                    feature_importance = np.mean(np.abs(weights), axis=1)
                    
                    # Normalize
                    feature_importance = feature_importance / np.sum(feature_importance)
                    
                    # Create dictionary with feature names
                    importance_dict = {}
                    for i in range(min(len(feature_importance), self.feature_dim)):
                        feature_name = f"feature_{i}"
                        importance_dict[feature_name] = float(feature_importance[i])
                    
                    return importance_dict
            
            return {}
            
        except Exception as e:
            logger.warning(f"Failed to calculate feature importance: {e}")
            return {}
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary information"""
        
        summary = {
            'model_type': 'ChurnPredictionModel',
            'is_fitted': self.is_fitted,
            'sequence_length': self.sequence_length,
            'feature_dim': self.feature_dim,
            'config': self.config
        }
        
        if self.model:
            summary['total_parameters'] = self.model.count_params()
            summary['trainable_parameters'] = sum([
                tf.keras.utils.count_params(layer.trainable_weights)
                for layer in self.model.layers
            ])
        
        return summary
