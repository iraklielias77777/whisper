"""
Timing Optimization Model for User Whisperer Platform
Prophet-based model for predicting optimal message send times
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
import json
import os
import pickle
from dataclasses import dataclass

# Prophet import with fallback
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    try:
        from fbprophet import Prophet
        PROPHET_AVAILABLE = True
    except ImportError:
        PROPHET_AVAILABLE = False
        logging.warning("Prophet not available. Timing optimization will use heuristics only.")

logger = logging.getLogger(__name__)

@dataclass
class TimingPrediction:
    """Represents a timing prediction"""
    timestamp: datetime
    engagement_score: float
    confidence_interval: Tuple[float, float]
    factors: Dict[str, float]

class TimingOptimizationModel:
    """
    Prophet-based timing optimization model with fallback heuristics
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.models = {}  # Per-user models
        self.global_model = None
        self.user_patterns = {}  # User-specific patterns
        
        # Model parameters
        self.seasonality_mode = config.get('seasonality_mode', 'multiplicative')
        self.changepoint_prior_scale = config.get('changepoint_prior_scale', 0.05)
        self.seasonality_prior_scale = config.get('seasonality_prior_scale', 10.0)
        self.interval_width = config.get('interval_width', 0.95)
        
        # Minimum data requirements
        self.min_data_points = config.get('min_data_points', 168)  # 1 week hourly
        self.min_user_data = config.get('min_user_data', 50)
        
        # Heuristic patterns
        self.default_patterns = self._load_default_patterns()
        
        logger.info(f"Initialized TimingOptimizationModel with Prophet={'available' if PROPHET_AVAILABLE else 'unavailable'}")
    
    def _load_default_patterns(self) -> Dict[str, Any]:
        """Load default timing patterns for fallback"""
        
        return {
            'optimal_hours': {
                'email': [10, 14, 20],  # 10 AM, 2 PM, 8 PM
                'sms': [12, 18],        # Noon, 6 PM
                'push': [9, 13, 19],    # 9 AM, 1 PM, 7 PM
                'in_app': [11, 15, 21]  # 11 AM, 3 PM, 9 PM
            },
            'day_of_week_multipliers': {
                0: 0.9,   # Monday
                1: 1.1,   # Tuesday
                2: 1.2,   # Wednesday
                3: 1.1,   # Thursday
                4: 0.8,   # Friday
                5: 0.6,   # Saturday
                6: 0.7    # Sunday
            },
            'seasonal_adjustments': {
                'holiday_periods': 0.7,
                'vacation_months': [7, 8, 12],  # July, August, December
                'high_engagement_months': [1, 3, 9, 10]  # January, March, September, October
            }
        }
    
    def build_model(
        self,
        user_id: Optional[str] = None,
        **kwargs
    ) -> Optional[object]:
        """Build Prophet model for user or global"""
        
        if not PROPHET_AVAILABLE:
            logger.warning("Prophet not available, using heuristic model")
            return None
        
        try:
            model = Prophet(
                seasonality_mode=self.seasonality_mode,
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_prior_scale=self.seasonality_prior_scale,
                interval_width=self.interval_width,
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False
            )
            
            # Add custom seasonalities
            model.add_seasonality(
                name='hourly',
                period=1,
                fourier_order=5
            )
            
            model.add_seasonality(
                name='business_hours',
                period=1,
                fourier_order=3,
                condition_name='is_business_hours'
            )
            
            # Add country holidays if specified
            country = self.config.get('country')
            if country:
                try:
                    model.add_country_holidays(country_name=country)
                except Exception as e:
                    logger.warning(f"Failed to add holidays for {country}: {e}")
            
            logger.debug(f"Built Prophet model for {'user ' + user_id if user_id else 'global'}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to build Prophet model: {e}")
            return None
    
    def prepare_data(
        self,
        engagement_history: List[Dict],
        channel: str = 'email'
    ) -> pd.DataFrame:
        """Prepare data for Prophet training"""
        
        try:
            data = []
            
            for event in engagement_history:
                try:
                    # Parse timestamp
                    timestamp_str = event.get('timestamp', event.get('sent_at', ''))
                    if not timestamp_str:
                        continue
                    
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    
                    # Calculate engagement score based on event type
                    engagement = self._calculate_engagement_score(event, channel)
                    
                    if engagement > 0:
                        data.append({
                            'ds': timestamp,
                            'y': engagement
                        })
                        
                except Exception as e:
                    logger.warning(f"Failed to process engagement event: {e}")
                    continue
            
            if not data:
                return pd.DataFrame(columns=['ds', 'y'])
            
            df = pd.DataFrame(data)
            
            # Sort by timestamp
            df = df.sort_values('ds')
            
            # Aggregate by hour to reduce noise
            df['hour'] = df['ds'].dt.floor('H')
            df_hourly = df.groupby('hour').agg({
                'y': ['sum', 'count', 'mean']
            }).reset_index()
            
            # Flatten column names
            df_hourly.columns = ['ds', 'engagement_sum', 'event_count', 'engagement_avg']
            
            # Use weighted engagement (sum with count weighting)
            df_hourly['y'] = df_hourly['engagement_sum'] * np.log1p(df_hourly['event_count'])
            
            # Add regressors
            df_hourly = self._add_regressors(df_hourly)
            
            logger.debug(f"Prepared {len(df_hourly)} data points for training")
            
            return df_hourly[['ds', 'y', 'is_business_hours']]
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            return pd.DataFrame(columns=['ds', 'y'])
    
    def _calculate_engagement_score(
        self,
        event: Dict,
        channel: str
    ) -> float:
        """Calculate engagement score from event"""
        
        event_type = event.get('event_type', '')
        
        # Base scores by event type
        base_scores = {
            'message_sent': 0.1,
            'message_delivered': 0.2,
            'message_opened': 1.0,
            'message_clicked': 2.0,
            'conversion': 3.0,
            'unsubscribe': -2.0,
            'bounce': -1.0,
            'spam': -3.0
        }
        
        base_score = base_scores.get(event_type, 0.0)
        
        # Channel-specific adjustments
        channel_multipliers = {
            'email': 1.0,
            'sms': 1.2,
            'push': 0.8,
            'in_app': 1.5
        }
        
        multiplier = channel_multipliers.get(channel, 1.0)
        
        # Time-based adjustments
        timestamp_str = event.get('timestamp', event.get('sent_at', ''))
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                
                # Boost score for business hours
                if 9 <= timestamp.hour <= 17:
                    multiplier *= 1.1
                
                # Reduce score for late night/early morning
                if timestamp.hour < 7 or timestamp.hour > 22:
                    multiplier *= 0.7
                    
            except Exception:
                pass
        
        return max(0.0, base_score * multiplier)
    
    def _add_regressors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add additional regressors to the data"""
        
        try:
            # Business hours indicator
            df['is_business_hours'] = (
                (df['ds'].dt.hour >= 9) & 
                (df['ds'].dt.hour <= 17) & 
                (df['ds'].dt.weekday < 5)
            ).astype(int)
            
            # Weekend indicator
            df['is_weekend'] = (df['ds'].dt.weekday >= 5).astype(int)
            
            # Hour of day (cyclical)
            df['hour_sin'] = np.sin(2 * np.pi * df['ds'].dt.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['ds'].dt.hour / 24)
            
            # Day of week (cyclical)
            df['dow_sin'] = np.sin(2 * np.pi * df['ds'].dt.weekday / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['ds'].dt.weekday / 7)
            
            return df
            
        except Exception as e:
            logger.warning(f"Failed to add regressors: {e}")
            return df
    
    def train_user_model(
        self,
        user_id: str,
        engagement_history: List[Dict],
        channel: str = 'email'
    ) -> bool:
        """Train model for specific user"""
        
        try:
            # Prepare data
            df = self.prepare_data(engagement_history, channel)
            
            if len(df) < self.min_user_data:
                logger.info(f"Not enough data for user {user_id} model ({len(df)} < {self.min_user_data})")
                return False
            
            if not PROPHET_AVAILABLE:
                # Store user patterns for heuristic model
                self._extract_user_patterns(user_id, df, channel)
                return True
            
            # Build and train Prophet model
            model = self.build_model(user_id)
            if not model:
                return False
            
            # Add regressors
            model.add_regressor('is_business_hours')
            
            # Fit model
            model.fit(df)
            
            # Store model
            model_key = f"{user_id}_{channel}"
            self.models[model_key] = {
                'model': model,
                'trained_at': datetime.now(),
                'data_points': len(df),
                'channel': channel
            }
            
            logger.info(f"Trained model for user {user_id} on {channel} with {len(df)} data points")
            
            return True
            
        except Exception as e:
            logger.error(f"User model training failed for {user_id}: {e}")
            return False
    
    def _extract_user_patterns(
        self,
        user_id: str,
        df: pd.DataFrame,
        channel: str
    ):
        """Extract user-specific patterns for heuristic model"""
        
        try:
            if df.empty:
                return
            
            # Hour-of-day pattern
            hourly_engagement = df.groupby(df['ds'].dt.hour)['y'].mean()
            top_hours = hourly_engagement.nlargest(3).index.tolist()
            
            # Day-of-week pattern
            daily_engagement = df.groupby(df['ds'].dt.weekday)['y'].mean()
            best_days = daily_engagement.nlargest(3).index.tolist()
            
            # Store patterns
            pattern_key = f"{user_id}_{channel}"
            self.user_patterns[pattern_key] = {
                'optimal_hours': top_hours,
                'best_days': best_days,
                'hourly_scores': hourly_engagement.to_dict(),
                'daily_scores': daily_engagement.to_dict(),
                'last_updated': datetime.now()
            }
            
            logger.debug(f"Extracted patterns for user {user_id}: hours={top_hours}, days={best_days}")
            
        except Exception as e:
            logger.warning(f"Pattern extraction failed for user {user_id}: {e}")
    
    def train_global_model(
        self,
        all_engagement_history: List[Dict],
        channel: str = 'email'
    ) -> bool:
        """Train global model for users without enough individual data"""
        
        try:
            # Prepare aggregated data
            df = self.prepare_data(all_engagement_history, channel)
            
            if len(df) < self.min_data_points:
                logger.warning(f"Not enough global data ({len(df)} < {self.min_data_points})")
                return False
            
            if not PROPHET_AVAILABLE:
                # Extract global patterns
                self._extract_user_patterns('global', df, channel)
                return True
            
            # Build and train global model
            model = self.build_model()
            if not model:
                return False
            
            # Add regressors
            model.add_regressor('is_business_hours')
            
            # Fit model
            model.fit(df)
            
            # Store model
            self.global_model = {
                'model': model,
                'trained_at': datetime.now(),
                'data_points': len(df),
                'channel': channel
            }
            
            logger.info(f"Trained global model on {channel} with {len(df)} data points")
            
            return True
            
        except Exception as e:
            logger.error(f"Global model training failed: {e}")
            return False
    
    def predict_optimal_times(
        self,
        user_id: str,
        channel: str = 'email',
        next_hours: int = 48,
        top_k: int = 5
    ) -> List[TimingPrediction]:
        """Predict optimal send times for user"""
        
        try:
            model_key = f"{user_id}_{channel}"
            
            # Use user model if available, otherwise global or heuristic
            if model_key in self.models and PROPHET_AVAILABLE:
                return self._predict_with_prophet(
                    self.models[model_key]['model'],
                    next_hours,
                    top_k
                )
            elif self.global_model and PROPHET_AVAILABLE:
                return self._predict_with_prophet(
                    self.global_model['model'],
                    next_hours,
                    top_k
                )
            else:
                # Use heuristic prediction
                return self._predict_heuristic(
                    user_id,
                    channel,
                    next_hours,
                    top_k
                )
                
        except Exception as e:
            logger.error(f"Prediction failed for user {user_id}: {e}")
            return self._predict_heuristic(user_id, channel, next_hours, top_k)
    
    def _predict_with_prophet(
        self,
        model: object,
        next_hours: int,
        top_k: int
    ) -> List[TimingPrediction]:
        """Make predictions using Prophet model"""
        
        try:
            # Create future dataframe
            future = model.make_future_dataframe(
                periods=next_hours,
                freq='H'
            )
            
            # Add regressors for future periods
            future = self._add_regressors(future)
            
            # Make predictions
            forecast = model.predict(future)
            
            # Get recent predictions
            recent_forecast = forecast.tail(next_hours)
            
            # Create prediction objects
            predictions = []
            
            for _, row in recent_forecast.iterrows():
                timestamp = row['ds'].to_pydatetime()
                
                # Skip past times
                if timestamp <= datetime.now():
                    continue
                
                prediction = TimingPrediction(
                    timestamp=timestamp,
                    engagement_score=max(0.0, row['yhat']),
                    confidence_interval=(
                        max(0.0, row['yhat_lower']),
                        row['yhat_upper']
                    ),
                    factors={
                        'trend': row.get('trend', 0.0),
                        'seasonal': row.get('seasonal', 0.0),
                        'daily': row.get('daily', 0.0),
                        'weekly': row.get('weekly', 0.0)
                    }
                )
                
                predictions.append(prediction)
            
            # Sort by engagement score and return top k
            predictions.sort(key=lambda x: x.engagement_score, reverse=True)
            
            return predictions[:top_k]
            
        except Exception as e:
            logger.error(f"Prophet prediction failed: {e}")
            return []
    
    def _predict_heuristic(
        self,
        user_id: str,
        channel: str,
        next_hours: int,
        top_k: int
    ) -> List[TimingPrediction]:
        """Make predictions using heuristic patterns"""
        
        try:
            predictions = []
            now = datetime.now()
            
            # Get user patterns or use defaults
            pattern_key = f"{user_id}_{channel}"
            
            if pattern_key in self.user_patterns:
                patterns = self.user_patterns[pattern_key]
                optimal_hours = patterns['optimal_hours']
                hourly_scores = patterns.get('hourly_scores', {})
                daily_scores = patterns.get('daily_scores', {})
            else:
                # Use global patterns if available
                global_key = f"global_{channel}"
                if global_key in self.user_patterns:
                    patterns = self.user_patterns[global_key]
                    optimal_hours = patterns['optimal_hours']
                    hourly_scores = patterns.get('hourly_scores', {})
                    daily_scores = patterns.get('daily_scores', {})
                else:
                    # Use default patterns
                    optimal_hours = self.default_patterns['optimal_hours'].get(channel, [10, 14, 20])
                    hourly_scores = {}
                    daily_scores = {}
            
            # Generate predictions for next hours
            for i in range(next_hours):
                timestamp = now + timedelta(hours=i)
                
                # Skip past times
                if timestamp <= now:
                    continue
                
                # Calculate base score
                hour = timestamp.hour
                dow = timestamp.weekday()
                
                # Hour-based score
                if hourly_scores and hour in hourly_scores:
                    hour_score = hourly_scores[hour]
                elif hour in optimal_hours:
                    hour_score = 0.8
                else:
                    hour_score = 0.3
                
                # Day-of-week adjustment
                if daily_scores and dow in daily_scores:
                    dow_multiplier = daily_scores[dow] / max(daily_scores.values()) if daily_scores.values() else 1.0
                else:
                    dow_multiplier = self.default_patterns['day_of_week_multipliers'].get(dow, 1.0)
                
                # Business hours boost
                business_hours_boost = 1.2 if 9 <= hour <= 17 and dow < 5 else 1.0
                
                # Calculate final score
                engagement_score = hour_score * dow_multiplier * business_hours_boost
                
                # Confidence interval (heuristic)
                uncertainty = 0.2
                lower_bound = max(0.0, engagement_score - uncertainty)
                upper_bound = engagement_score + uncertainty
                
                prediction = TimingPrediction(
                    timestamp=timestamp,
                    engagement_score=engagement_score,
                    confidence_interval=(lower_bound, upper_bound),
                    factors={
                        'hour_score': hour_score,
                        'day_multiplier': dow_multiplier,
                        'business_hours': business_hours_boost,
                        'pattern_source': 'user' if pattern_key in self.user_patterns else 'default'
                    }
                )
                
                predictions.append(prediction)
            
            # Sort by engagement score and return top k
            predictions.sort(key=lambda x: x.engagement_score, reverse=True)
            
            return predictions[:top_k]
            
        except Exception as e:
            logger.error(f"Heuristic prediction failed: {e}")
            return []
    
    def update_with_feedback(
        self,
        user_id: str,
        channel: str,
        send_time: datetime,
        outcome: str,
        engagement_score: Optional[float] = None
    ):
        """Update model with delivery feedback"""
        
        try:
            # Calculate engagement score if not provided
            if engagement_score is None:
                engagement_score = self._calculate_engagement_score(
                    {'event_type': f'message_{outcome}', 'timestamp': send_time.isoformat()},
                    channel
                )
            
            # Store feedback for retraining
            feedback = {
                'user_id': user_id,
                'channel': channel,
                'timestamp': send_time.isoformat(),
                'outcome': outcome,
                'engagement_score': engagement_score,
                'hour': send_time.hour,
                'dow': send_time.weekday(),
                'recorded_at': datetime.now().isoformat()
            }
            
            # Update user patterns for heuristic model
            pattern_key = f"{user_id}_{channel}"
            if pattern_key in self.user_patterns:
                patterns = self.user_patterns[pattern_key]
                
                # Update hourly scores
                hour = send_time.hour
                if 'hourly_scores' not in patterns:
                    patterns['hourly_scores'] = {}
                
                # Exponential moving average
                alpha = 0.1
                current_score = patterns['hourly_scores'].get(hour, 0.5)
                patterns['hourly_scores'][hour] = (
                    alpha * engagement_score + (1 - alpha) * current_score
                )
                
                patterns['last_updated'] = datetime.now()
            
            logger.debug(f"Updated timing feedback for user {user_id}: {outcome} at {send_time}")
            
            # This would typically store in database for periodic model retraining
            # await self.store_feedback(feedback)
            
        except Exception as e:
            logger.error(f"Feedback update failed: {e}")
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model statistics and performance metrics"""
        
        stats = {
            'prophet_available': PROPHET_AVAILABLE,
            'user_models': len(self.models),
            'user_patterns': len(self.user_patterns),
            'global_model_trained': self.global_model is not None,
            'models_by_channel': {}
        }
        
        # Count models by channel
        for key in self.models:
            if '_' in key:
                channel = key.split('_')[-1]
                if channel not in stats['models_by_channel']:
                    stats['models_by_channel'][channel] = 0
                stats['models_by_channel'][channel] += 1
        
        # Add global model info
        if self.global_model:
            stats['global_model'] = {
                'trained_at': self.global_model['trained_at'].isoformat(),
                'data_points': self.global_model['data_points'],
                'channel': self.global_model['channel']
            }
        
        return stats
    
    def save_model(self, path: str):
        """Save timing optimization models"""
        
        try:
            os.makedirs(path, exist_ok=True)
            
            # Save configuration and metadata
            metadata = {
                'config': self.config,
                'prophet_available': PROPHET_AVAILABLE,
                'model_count': len(self.models),
                'pattern_count': len(self.user_patterns),
                'saved_at': datetime.now().isoformat()
            }
            
            with open(f"{path}/timing_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save user patterns
            patterns_serializable = {}
            for key, pattern in self.user_patterns.items():
                patterns_serializable[key] = {
                    **pattern,
                    'last_updated': pattern['last_updated'].isoformat()
                }
            
            with open(f"{path}/timing_patterns.json", 'w') as f:
                json.dump(patterns_serializable, f, indent=2)
            
            # Save Prophet models if available
            if PROPHET_AVAILABLE:
                models_info = {}
                
                for key, model_info in self.models.items():
                    model_path = f"{path}/timing_model_{key}.pkl"
                    
                    with open(model_path, 'wb') as f:
                        pickle.dump(model_info['model'], f)
                    
                    models_info[key] = {
                        'trained_at': model_info['trained_at'].isoformat(),
                        'data_points': model_info['data_points'],
                        'channel': model_info['channel'],
                        'model_file': f"timing_model_{key}.pkl"
                    }
                
                if self.global_model:
                    global_path = f"{path}/timing_global_model.pkl"
                    with open(global_path, 'wb') as f:
                        pickle.dump(self.global_model['model'], f)
                    
                    models_info['global'] = {
                        'trained_at': self.global_model['trained_at'].isoformat(),
                        'data_points': self.global_model['data_points'],
                        'channel': self.global_model['channel'],
                        'model_file': 'timing_global_model.pkl'
                    }
                
                with open(f"{path}/timing_models_info.json", 'w') as f:
                    json.dump(models_info, f, indent=2)
            
            logger.info(f"Saved TimingOptimizationModel to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save timing model: {e}")
            raise
    
    def load_model(self, path: str):
        """Load timing optimization models"""
        
        try:
            # Load metadata
            metadata_path = f"{path}/timing_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
                self.config.update(metadata.get('config', {}))
            
            # Load user patterns
            patterns_path = f"{path}/timing_patterns.json"
            if os.path.exists(patterns_path):
                with open(patterns_path, 'r') as f:
                    patterns_data = json.load(f)
                
                self.user_patterns = {}
                for key, pattern in patterns_data.items():
                    self.user_patterns[key] = {
                        **pattern,
                        'last_updated': datetime.fromisoformat(pattern['last_updated'])
                    }
            
            # Load Prophet models if available
            if PROPHET_AVAILABLE:
                models_info_path = f"{path}/timing_models_info.json"
                if os.path.exists(models_info_path):
                    with open(models_info_path, 'r') as f:
                        models_info = json.load(f)
                    
                    # Load user models
                    for key, info in models_info.items():
                        if key == 'global':
                            continue
                        
                        model_path = f"{path}/{info['model_file']}"
                        if os.path.exists(model_path):
                            with open(model_path, 'rb') as f:
                                model = pickle.load(f)
                            
                            self.models[key] = {
                                'model': model,
                                'trained_at': datetime.fromisoformat(info['trained_at']),
                                'data_points': info['data_points'],
                                'channel': info['channel']
                            }
                    
                    # Load global model
                    if 'global' in models_info:
                        global_info = models_info['global']
                        global_path = f"{path}/{global_info['model_file']}"
                        
                        if os.path.exists(global_path):
                            with open(global_path, 'rb') as f:
                                model = pickle.load(f)
                            
                            self.global_model = {
                                'model': model,
                                'trained_at': datetime.fromisoformat(global_info['trained_at']),
                                'data_points': global_info['data_points'],
                                'channel': global_info['channel']
                            }
            
            logger.info(f"Loaded TimingOptimizationModel from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load timing model: {e}")
            raise
