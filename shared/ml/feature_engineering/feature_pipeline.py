"""
Feature Engineering Pipeline for User Whisperer Platform
Comprehensive feature extraction and processing for ML models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
import json
import hashlib
from collections import defaultdict, Counter
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
    from sklearn.feature_extraction import FeatureHasher
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Some feature processing will be limited.")

logger = logging.getLogger(__name__)

@dataclass
class FeatureConfig:
    """Configuration for feature extraction"""
    include_temporal: bool = True
    include_behavioral: bool = True
    include_engagement: bool = True
    include_monetization: bool = True
    include_session: bool = True
    include_content: bool = True
    include_device: bool = True
    include_geographic: bool = True
    
    # Processing options
    normalize_features: bool = True
    handle_missing: str = "mean"  # "mean", "median", "zero", "drop"
    feature_selection: bool = False
    max_features: Optional[int] = None

class FeatureEngineeringPipeline:
    """
    Comprehensive feature engineering pipeline for ML models
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.feature_config = FeatureConfig(**config.get('features', {}))
        self.feature_extractors = []
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.feature_importance = {}
        
        # Initialize extractors based on configuration
        self.initialize_extractors()
        
        logger.info(f"Initialized FeatureEngineeringPipeline with {len(self.feature_extractors)} extractors")
    
    def initialize_extractors(self):
        """Initialize all feature extractors based on configuration"""
        
        extractors = []
        
        if self.feature_config.include_temporal:
            extractors.append(TemporalFeatureExtractor())
        
        if self.feature_config.include_behavioral:
            extractors.append(BehavioralFeatureExtractor())
        
        if self.feature_config.include_engagement:
            extractors.append(EngagementFeatureExtractor())
        
        if self.feature_config.include_monetization:
            extractors.append(MonetizationFeatureExtractor())
        
        if self.feature_config.include_session:
            extractors.append(SessionFeatureExtractor())
        
        if self.feature_config.include_content:
            extractors.append(ContentFeatureExtractor())
        
        if self.feature_config.include_device:
            extractors.append(DeviceFeatureExtractor())
        
        if self.feature_config.include_geographic:
            extractors.append(GeographicFeatureExtractor())
        
        self.feature_extractors = extractors
    
    def extract_features(
        self,
        user_data: Dict,
        events: List[Dict],
        context: Optional[Dict] = None
    ) -> Dict[str, Union[float, int]]:
        """Extract all features for a user"""
        
        try:
            features = {}
            
            # Extract features from each extractor
            for extractor in self.feature_extractors:
                try:
                    extractor_features = extractor.extract(
                        user_data,
                        events,
                        context
                    )
                    
                    # Add extractor prefix to avoid name conflicts
                    extractor_name = extractor.__class__.__name__.replace('FeatureExtractor', '').lower()
                    prefixed_features = {
                        f"{extractor_name}_{key}": value
                        for key, value in extractor_features.items()
                    }
                    
                    features.update(prefixed_features)
                    
                except Exception as e:
                    logger.warning(f"Feature extraction failed for {extractor.__class__.__name__}: {e}")
                    continue
            
            # Handle missing values
            features = self._handle_missing_values(features)
            
            # Apply feature selection if configured
            if self.feature_config.feature_selection and self.feature_names:
                features = self._select_features(features)
            
            logger.debug(f"Extracted {len(features)} features")
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction pipeline failed: {e}")
            return {}
    
    def extract_features_batch(
        self,
        users_data: List[Dict],
        users_events: List[List[Dict]],
        contexts: Optional[List[Dict]] = None
    ) -> pd.DataFrame:
        """Extract features for multiple users efficiently"""
        
        try:
            if contexts is None:
                contexts = [None] * len(users_data)
            
            features_list = []
            
            for i, (user_data, events, context) in enumerate(zip(users_data, users_events, contexts)):
                try:
                    features = self.extract_features(user_data, events, context)
                    features['user_index'] = i
                    features_list.append(features)
                    
                except Exception as e:
                    logger.warning(f"Failed to extract features for user {i}: {e}")
                    continue
            
            if not features_list:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(features_list)
            
            # Handle missing values at DataFrame level
            df = self._handle_missing_values_df(df)
            
            # Normalize features if configured
            if self.feature_config.normalize_features:
                df = self._normalize_features_df(df)
            
            logger.info(f"Extracted features for {len(df)} users with {len(df.columns)} features")
            
            return df
            
        except Exception as e:
            logger.error(f"Batch feature extraction failed: {e}")
            return pd.DataFrame()
    
    def _handle_missing_values(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Handle missing values in feature dictionary"""
        
        cleaned_features = {}
        
        for key, value in features.items():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                if self.feature_config.handle_missing == "zero":
                    cleaned_features[key] = 0.0
                elif self.feature_config.handle_missing == "mean":
                    # Use global mean or default
                    cleaned_features[key] = self._get_default_value(key)
                elif self.feature_config.handle_missing == "drop":
                    continue  # Skip this feature
                else:
                    cleaned_features[key] = 0.0
            else:
                cleaned_features[key] = float(value) if isinstance(value, (int, float)) else value
        
        return cleaned_features
    
    def _handle_missing_values_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in DataFrame"""
        
        if df.empty:
            return df
        
        try:
            if self.feature_config.handle_missing == "mean":
                # Fill with mean for numeric columns
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
                
                # Fill with mode for categorical columns
                categorical_columns = df.select_dtypes(exclude=[np.number]).columns
                for col in categorical_columns:
                    if col != 'user_index':
                        mode_value = df[col].mode()
                        if not mode_value.empty:
                            df[col] = df[col].fillna(mode_value[0])
                        else:
                            df[col] = df[col].fillna("unknown")
            
            elif self.feature_config.handle_missing == "median":
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
            
            elif self.feature_config.handle_missing == "zero":
                df = df.fillna(0)
            
            elif self.feature_config.handle_missing == "drop":
                df = df.dropna()
            
            return df
            
        except Exception as e:
            logger.warning(f"Missing value handling failed: {e}")
            return df.fillna(0)
    
    def _normalize_features_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features in DataFrame"""
        
        if not SKLEARN_AVAILABLE or df.empty:
            return df
        
        try:
            # Get numeric columns (excluding user_index)
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            numeric_columns = [col for col in numeric_columns if col != 'user_index']
            
            if not numeric_columns:
                return df
            
            # Fit scaler if not already fitted
            if 'main_scaler' not in self.scalers:
                self.scalers['main_scaler'] = StandardScaler()
                scaled_values = self.scalers['main_scaler'].fit_transform(df[numeric_columns])
            else:
                scaled_values = self.scalers['main_scaler'].transform(df[numeric_columns])
            
            # Update DataFrame
            df_scaled = df.copy()
            df_scaled[numeric_columns] = scaled_values
            
            return df_scaled
            
        except Exception as e:
            logger.warning(f"Feature normalization failed: {e}")
            return df
    
    def _select_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Select most important features"""
        
        if not self.feature_names:
            return features
        
        # Select based on importance or predefined list
        selected_features = {}
        
        for feature_name in self.feature_names:
            if feature_name in features:
                selected_features[feature_name] = features[feature_name]
        
        # Limit number of features if specified
        if self.feature_config.max_features and len(selected_features) > self.feature_config.max_features:
            # Sort by importance if available
            if self.feature_importance:
                sorted_features = sorted(
                    selected_features.items(),
                    key=lambda x: self.feature_importance.get(x[0], 0),
                    reverse=True
                )
                selected_features = dict(sorted_features[:self.feature_config.max_features])
            else:
                # Take first N features
                feature_items = list(selected_features.items())
                selected_features = dict(feature_items[:self.feature_config.max_features])
        
        return selected_features
    
    def _get_default_value(self, feature_name: str) -> float:
        """Get default value for a feature"""
        
        # Common defaults based on feature patterns
        defaults = {
            'count': 0.0,
            'score': 0.5,
            'rate': 0.0,
            'ratio': 0.0,
            'avg': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'total': 0.0,
            'days': 0.0,
            'hours': 0.0
        }
        
        feature_lower = feature_name.lower()
        
        for pattern, default_value in defaults.items():
            if pattern in feature_lower:
                return default_value
        
        return 0.0
    
    def fit_feature_processors(self, df: pd.DataFrame):
        """Fit feature processors (scalers, encoders) on training data"""
        
        if not SKLEARN_AVAILABLE or df.empty:
            return
        
        try:
            # Fit scalers for numeric features
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            numeric_columns = [col for col in numeric_columns if col != 'user_index']
            
            if numeric_columns:
                self.scalers['main_scaler'] = StandardScaler()
                self.scalers['main_scaler'].fit(df[numeric_columns])
            
            # Fit encoders for categorical features
            categorical_columns = df.select_dtypes(exclude=[np.number]).columns
            categorical_columns = [col for col in categorical_columns if col != 'user_index']
            
            for col in categorical_columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    
                try:
                    # Handle missing values before fitting
                    values = df[col].fillna('unknown')
                    self.encoders[col].fit(values)
                    
                except Exception as e:
                    logger.warning(f"Failed to fit encoder for {col}: {e}")
            
            logger.info("Fitted feature processors successfully")
            
        except Exception as e:
            logger.error(f"Failed to fit feature processors: {e}")
    
    def update_feature_importance(self, importance_dict: Dict[str, float]):
        """Update feature importance scores"""
        
        self.feature_importance.update(importance_dict)
        
        # Update feature selection list
        if self.feature_config.feature_selection:
            sorted_features = sorted(
                importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            max_features = self.feature_config.max_features or len(sorted_features)
            self.feature_names = [name for name, _ in sorted_features[:max_features]]
    
    def get_feature_stats(self) -> Dict[str, Any]:
        """Get feature extraction statistics"""
        
        return {
            'total_extractors': len(self.feature_extractors),
            'extractor_types': [
                extractor.__class__.__name__ 
                for extractor in self.feature_extractors
            ],
            'total_features': len(self.feature_names),
            'feature_importance_available': len(self.feature_importance) > 0,
            'scalers_fitted': len(self.scalers),
            'encoders_fitted': len(self.encoders),
            'config': {
                'normalize_features': self.feature_config.normalize_features,
                'handle_missing': self.feature_config.handle_missing,
                'feature_selection': self.feature_config.feature_selection,
                'max_features': self.feature_config.max_features
            }
        }


class BaseFeatureExtractor(ABC):
    """Base class for feature extractors"""
    
    @abstractmethod
    def extract(
        self,
        user_data: Dict,
        events: List[Dict],
        context: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Extract features from user data and events"""
        pass
    
    def safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division with default value"""
        return numerator / denominator if denominator != 0 else default
    
    def safe_log(self, value: float, default: float = 0.0) -> float:
        """Safe logarithm with default value"""
        return np.log1p(max(0, value)) if value > 0 else default


class TemporalFeatureExtractor(BaseFeatureExtractor):
    """Extract time-based features"""
    
    def extract(
        self,
        user_data: Dict,
        events: List[Dict],
        context: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Extract temporal features"""
        
        features = {}
        now = datetime.now()
        
        try:
            # User tenure
            created_at_str = user_data.get('created_at', '')
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                    tenure_days = (now - created_at).days
                    tenure_hours = (now - created_at).total_seconds() / 3600
                    
                    features['days_since_signup'] = tenure_days
                    features['hours_since_signup'] = tenure_hours
                    features['log_tenure_days'] = self.safe_log(tenure_days + 1)
                    
                    # Tenure categories
                    features['is_new_user'] = 1 if tenure_days < 7 else 0
                    features['is_recent_user'] = 1 if tenure_days < 30 else 0
                    features['is_established_user'] = 1 if tenure_days > 90 else 0
                    
                except Exception:
                    features['days_since_signup'] = 0
                    features['hours_since_signup'] = 0
            
            # Last activity
            last_active_str = user_data.get('last_active_at', '')
            if last_active_str:
                try:
                    last_active = datetime.fromisoformat(last_active_str.replace('Z', '+00:00'))
                    hours_since_last = (now - last_active).total_seconds() / 3600
                    days_since_last = (now - last_active).days
                    
                    features['hours_since_last_activity'] = hours_since_last
                    features['days_since_last_activity'] = days_since_last
                    features['log_hours_since_last'] = self.safe_log(hours_since_last + 1)
                    
                    # Activity recency categories
                    features['active_today'] = 1 if hours_since_last < 24 else 0
                    features['active_this_week'] = 1 if days_since_last < 7 else 0
                    features['dormant'] = 1 if days_since_last > 30 else 0
                    
                except Exception:
                    features['hours_since_last_activity'] = 999999
                    features['days_since_last_activity'] = 999999
            
            # Current time features
            current_time = context.get('current_time') if context else None
            if current_time:
                try:
                    current = datetime.fromisoformat(current_time.replace('Z', '+00:00'))
                except Exception:
                    current = now
            else:
                current = now
            
            # Time of day features
            features['hour_of_day'] = current.hour
            features['day_of_week'] = current.weekday()
            features['day_of_month'] = current.day
            features['week_of_year'] = current.isocalendar()[1]
            features['month_of_year'] = current.month
            features['quarter'] = (current.month - 1) // 3 + 1
            
            # Binary time features
            features['is_weekend'] = 1 if current.weekday() >= 5 else 0
            features['is_business_hours'] = 1 if 9 <= current.hour <= 17 and current.weekday() < 5 else 0
            features['is_morning'] = 1 if 6 <= current.hour <= 12 else 0
            features['is_afternoon'] = 1 if 12 <= current.hour <= 18 else 0
            features['is_evening'] = 1 if 18 <= current.hour <= 22 else 0
            features['is_night'] = 1 if current.hour >= 22 or current.hour <= 6 else 0
            
            # Cyclical encoding for time (preserves circular nature)
            features['hour_sin'] = np.sin(2 * np.pi * current.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * current.hour / 24)
            features['day_sin'] = np.sin(2 * np.pi * current.weekday() / 7)
            features['day_cos'] = np.cos(2 * np.pi * current.weekday() / 7)
            features['month_sin'] = np.sin(2 * np.pi * current.month / 12)
            features['month_cos'] = np.cos(2 * np.pi * current.month / 12)
            
            # Event timing patterns
            if events:
                event_hours = []
                event_days = []
                
                for event in events:
                    timestamp_str = event.get('timestamp', event.get('created_at', ''))
                    if timestamp_str:
                        try:
                            event_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            event_hours.append(event_time.hour)
                            event_days.append(event_time.weekday())
                        except Exception:
                            continue
                
                if event_hours:
                    features['avg_event_hour'] = np.mean(event_hours)
                    features['std_event_hour'] = np.std(event_hours) if len(event_hours) > 1 else 0
                    features['most_active_hour'] = max(set(event_hours), key=event_hours.count)
                    
                    # Hour diversity
                    unique_hours = len(set(event_hours))
                    features['hour_diversity'] = unique_hours / 24.0
                
                if event_days:
                    features['avg_event_day'] = np.mean(event_days)
                    features['most_active_day'] = max(set(event_days), key=event_days.count)
                    
                    # Day diversity
                    unique_days = len(set(event_days))
                    features['day_diversity'] = unique_days / 7.0
            
        except Exception as e:
            logger.warning(f"Temporal feature extraction failed: {e}")
        
        return features


class BehavioralFeatureExtractor(BaseFeatureExtractor):
    """Extract behavioral pattern features"""
    
    def extract(
        self,
        user_data: Dict,
        events: List[Dict],
        context: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Extract behavioral features"""
        
        features = {}
        
        try:
            if not events:
                return self._get_default_behavioral_features()
            
            # Basic event statistics
            features['total_events'] = len(events)
            features['log_total_events'] = self.safe_log(len(events))
            
            # Event type diversity
            event_types = [e.get('event_type', 'unknown') for e in events]
            unique_event_types = set(event_types)
            features['unique_event_types'] = len(unique_event_types)
            features['event_type_diversity'] = len(unique_event_types) / max(len(events), 1)
            
            # Event type distribution
            event_type_counts = Counter(event_types)
            total_events = len(events)
            
            # Common event types
            common_types = ['click', 'view', 'page_view', 'purchase', 'error', 'search', 'login', 'logout']
            for event_type in common_types:
                count = event_type_counts.get(event_type, 0)
                features[f'event_type_{event_type}_count'] = count
                features[f'event_type_{event_type}_ratio'] = count / total_events
            
            # Time-based patterns
            timestamps = []
            for event in events:
                timestamp_str = event.get('timestamp', event.get('created_at', ''))
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        timestamps.append(timestamp)
                    except Exception:
                        continue
            
            if len(timestamps) > 1:
                # Sort timestamps
                timestamps.sort()
                
                # Inter-event times
                inter_event_times = [
                    (timestamps[i+1] - timestamps[i]).total_seconds()
                    for i in range(len(timestamps)-1)
                ]
                
                features['avg_inter_event_time'] = np.mean(inter_event_times)
                features['std_inter_event_time'] = np.std(inter_event_times)
                features['min_inter_event_time'] = min(inter_event_times)
                features['max_inter_event_time'] = max(inter_event_times)
                features['median_inter_event_time'] = np.median(inter_event_times)
                
                # Log transformed times
                features['log_avg_inter_event_time'] = self.safe_log(features['avg_inter_event_time'])
                
                # Behavioral velocity and intensity
                time_span = (timestamps[-1] - timestamps[0]).total_seconds()
                if time_span > 0:
                    features['events_per_second'] = len(events) / time_span
                    features['events_per_minute'] = len(events) / (time_span / 60)
                    features['events_per_hour'] = len(events) / (time_span / 3600)
                    features['log_events_per_hour'] = self.safe_log(features['events_per_hour'])
                
                # Activity bursts (periods of high activity)
                burst_threshold = np.percentile(inter_event_times, 25)  # Bottom 25%
                burst_events = sum(1 for t in inter_event_times if t <= burst_threshold)
                features['burst_ratio'] = burst_events / len(inter_event_times)
                
            # Pattern detection
            features['has_error_pattern'] = self._detect_error_pattern(events)
            features['has_purchase_intent'] = self._detect_purchase_intent(events)
            features['has_engagement_decline'] = self._detect_engagement_decline(events)
            features['has_feature_exploration'] = self._detect_feature_exploration(events)
            
            # Event sequence patterns
            features.update(self._extract_sequence_patterns(events))
            
            # User journey stage indicators
            features.update(self._extract_journey_indicators(events))
            
        except Exception as e:
            logger.warning(f"Behavioral feature extraction failed: {e}")
            features.update(self._get_default_behavioral_features())
        
        return features
    
    def _get_default_behavioral_features(self) -> Dict[str, float]:
        """Get default features for users with no events"""
        
        return {
            'total_events': 0,
            'unique_event_types': 0,
            'events_per_hour': 0,
            'avg_inter_event_time': 0,
            'std_inter_event_time': 0,
            'has_error_pattern': 0,
            'has_purchase_intent': 0,
            'has_engagement_decline': 0,
            'has_feature_exploration': 0
        }
    
    def _detect_error_pattern(self, events: List[Dict]) -> float:
        """Detect if user has error patterns"""
        
        recent_events = events[-20:] if len(events) > 20 else events
        error_events = [
            e for e in recent_events
            if 'error' in e.get('event_type', '').lower()
        ]
        
        # High error rate in recent activity
        if len(recent_events) > 0:
            error_rate = len(error_events) / len(recent_events)
            return 1.0 if error_rate > 0.1 else error_rate * 10
        
        return 0.0
    
    def _detect_purchase_intent(self, events: List[Dict]) -> float:
        """Detect purchase intent signals"""
        
        intent_signals = [
            'pricing_viewed', 'cart_add', 'checkout_start', 'payment_method_add',
            'product_compare', 'wishlist_add', 'discount_code', 'trial_start'
        ]
        
        recent_events = events[-30:] if len(events) > 30 else events
        intent_events = [
            e for e in recent_events
            if any(signal in e.get('event_type', '') for signal in intent_signals)
        ]
        
        return min(1.0, len(intent_events) / 5.0)  # Normalize to 0-1
    
    def _detect_engagement_decline(self, events: List[Dict]) -> float:
        """Detect engagement decline pattern"""
        
        if len(events) < 10:
            return 0.0
        
        # Compare recent vs historical activity
        recent_events = events[-7:] if len(events) > 7 else events
        historical_events = events[:-7] if len(events) > 14 else events[:len(events)//2]
        
        if not historical_events:
            return 0.0
        
        recent_activity = len(recent_events) / max(7, len(recent_events))
        historical_activity = len(historical_events) / max(len(historical_events), 1)
        
        if historical_activity > 0:
            decline_ratio = 1 - (recent_activity / historical_activity)
            return max(0.0, min(1.0, decline_ratio))
        
        return 0.0
    
    def _detect_feature_exploration(self, events: List[Dict]) -> float:
        """Detect feature exploration behavior"""
        
        feature_events = [
            e for e in events
            if e.get('event_type', '').startswith('feature_') or
               'tutorial' in e.get('event_type', '') or
               'help' in e.get('event_type', '')
        ]
        
        unique_features = set(e.get('event_type', '') for e in feature_events)
        
        # Normalize based on total events and unique features
        exploration_score = len(unique_features) / max(len(events), 1)
        return min(1.0, exploration_score * 10)  # Amplify the signal
    
    def _extract_sequence_patterns(self, events: List[Dict]) -> Dict[str, float]:
        """Extract common behavioral sequences"""
        
        features = {}
        
        if len(events) < 3:
            return features
        
        # Get event type sequences
        event_types = [e.get('event_type', '') for e in events[-50:]]  # Last 50 events
        
        # Common sequences
        sequences = {
            'view_to_purchase': ['product_view', 'cart_add', 'purchase'],
            'onboarding_flow': ['signup', 'profile_setup', 'tutorial_start'],
            'help_seeking': ['error', 'help_search', 'support_contact'],
            'feature_adoption': ['feature_view', 'feature_try', 'feature_use']
        }
        
        for seq_name, seq_pattern in sequences.items():
            features[f'sequence_{seq_name}'] = self._find_sequence_in_events(event_types, seq_pattern)
        
        return features
    
    def _find_sequence_in_events(self, event_types: List[str], pattern: List[str]) -> float:
        """Find if a sequence pattern exists in events"""
        
        pattern_matches = 0
        
        for i in range(len(event_types) - len(pattern) + 1):
            window = event_types[i:i+len(pattern)]
            
            # Check if pattern matches (allowing for partial matches)
            matches = sum(1 for j, p in enumerate(pattern) if j < len(window) and p in window[j])
            
            if matches == len(pattern):
                pattern_matches += 1
            elif matches >= len(pattern) * 0.7:  # Partial match
                pattern_matches += 0.5
        
        # Normalize by possible occurrences
        max_occurrences = max(1, len(event_types) - len(pattern) + 1)
        return pattern_matches / max_occurrences
    
    def _extract_journey_indicators(self, events: List[Dict]) -> Dict[str, float]:
        """Extract user journey stage indicators"""
        
        features = {}
        
        # Journey stages based on event types
        onboarding_events = ['signup', 'welcome', 'tutorial', 'profile_setup']
        activation_events = ['first_use', 'goal_complete', 'value_received']
        growth_events = ['feature_adopt', 'invite_sent', 'content_create']
        retention_events = ['habit_form', 'regular_use', 'milestone_reach']
        
        total_events = len(events)
        
        if total_events > 0:
            for stage, stage_events in [
                ('onboarding', onboarding_events),
                ('activation', activation_events),
                ('growth', growth_events),
                ('retention', retention_events)
            ]:
                stage_count = sum(
                    1 for e in events
                    if any(se in e.get('event_type', '') for se in stage_events)
                )
                features[f'journey_{stage}_ratio'] = stage_count / total_events
                features[f'journey_{stage}_count'] = stage_count
        
        return features


class EngagementFeatureExtractor(BaseFeatureExtractor):
    """Extract engagement-related features"""
    
    def extract(
        self,
        user_data: Dict,
        events: List[Dict],
        context: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Extract engagement features"""
        
        features = {}
        
        try:
            # User profile engagement metrics
            features['engagement_score'] = user_data.get('engagement_score', 0.0)
            features['lifecycle_stage_encoded'] = self._encode_lifecycle_stage(
                user_data.get('lifecycle_stage', 'new')
            )
            
            # Calculate engagement from events
            engagement_events = self._filter_engagement_events(events)
            
            features['engagement_event_count'] = len(engagement_events)
            features['engagement_event_ratio'] = self.safe_divide(
                len(engagement_events), len(events)
            )
            
            # Engagement intensity over time
            if engagement_events:
                features.update(self._calculate_engagement_intensity(engagement_events))
            
            # Session-based engagement
            sessions = self._identify_sessions(events)
            if sessions:
                features.update(self._calculate_session_engagement(sessions))
            
            # Feature adoption and depth
            features.update(self._calculate_feature_engagement(events))
            
            # Content engagement
            features.update(self._calculate_content_engagement(events))
            
            # Communication engagement
            features.update(self._calculate_communication_engagement(events))
            
        except Exception as e:
            logger.warning(f"Engagement feature extraction failed: {e}")
        
        return features
    
    def _encode_lifecycle_stage(self, stage: str) -> float:
        """Encode lifecycle stage as numeric value"""
        
        stages = {
            'new': 0.0,
            'onboarding': 0.2,
            'activated': 0.4,
            'engaged': 0.6,
            'power_user': 0.8,
            'at_risk': -0.2,
            'dormant': -0.4,
            'churned': -0.6,
            'reactivated': 0.3
        }
        
        return stages.get(stage.lower(), 0.0)
    
    def _filter_engagement_events(self, events: List[Dict]) -> List[Dict]:
        """Filter events that indicate engagement"""
        
        engagement_types = [
            'feature_used', 'content_viewed', 'action_completed', 'goal_achieved',
            'tutorial_completed', 'milestone_reached', 'content_created',
            'social_interaction', 'collaboration', 'customization'
        ]
        
        return [
            e for e in events
            if any(et in e.get('event_type', '') for et in engagement_types)
        ]
    
    def _calculate_engagement_intensity(self, engagement_events: List[Dict]) -> Dict[str, float]:
        """Calculate engagement intensity metrics"""
        
        features = {}
        
        if not engagement_events:
            return features
        
        # Time-based intensity
        timestamps = []
        for event in engagement_events:
            timestamp_str = event.get('timestamp', event.get('created_at', ''))
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    timestamps.append(timestamp)
                except Exception:
                    continue
        
        if len(timestamps) > 1:
            timestamps.sort()
            time_span = (timestamps[-1] - timestamps[0]).total_seconds() / 3600  # hours
            
            if time_span > 0:
                features['engagement_events_per_hour'] = len(engagement_events) / time_span
                features['log_engagement_intensity'] = self.safe_log(features['engagement_events_per_hour'])
        
        # Engagement event diversity
        engagement_types = set(e.get('event_type', '') for e in engagement_events)
        features['engagement_type_diversity'] = len(engagement_types)
        features['engagement_breadth'] = len(engagement_types) / max(len(engagement_events), 1)
        
        # Recent vs historical engagement
        now = datetime.now()
        recent_threshold = now - timedelta(days=7)
        
        recent_engagement = 0
        historical_engagement = 0
        
        for event in engagement_events:
            timestamp_str = event.get('timestamp', event.get('created_at', ''))
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    if timestamp > recent_threshold:
                        recent_engagement += 1
                    else:
                        historical_engagement += 1
                except Exception:
                    continue
        
        features['recent_engagement_ratio'] = self.safe_divide(
            recent_engagement, len(engagement_events)
        )
        
        # Engagement trend
        if historical_engagement > 0:
            features['engagement_trend'] = recent_engagement / historical_engagement
        else:
            features['engagement_trend'] = 1.0 if recent_engagement > 0 else 0.0
        
        return features
    
    def _identify_sessions(
        self,
        events: List[Dict],
        timeout_minutes: int = 30
    ) -> List[List[Dict]]:
        """Identify user sessions from events"""
        
        if not events:
            return []
        
        # Sort events by timestamp
        sorted_events = []
        for event in events:
            timestamp_str = event.get('timestamp', event.get('created_at', ''))
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    sorted_events.append((timestamp, event))
                except Exception:
                    continue
        
        sorted_events.sort(key=lambda x: x[0])
        
        if not sorted_events:
            return []
        
        sessions = []
        current_session = [sorted_events[0][1]]
        last_time = sorted_events[0][0]
        
        for timestamp, event in sorted_events[1:]:
            if (timestamp - last_time).total_seconds() > timeout_minutes * 60:
                # New session
                sessions.append(current_session)
                current_session = [event]
            else:
                current_session.append(event)
            
            last_time = timestamp
        
        if current_session:
            sessions.append(current_session)
        
        return sessions
    
    def _calculate_session_engagement(self, sessions: List[List[Dict]]) -> Dict[str, float]:
        """Calculate session-based engagement metrics"""
        
        features = {}
        
        if not sessions:
            return features
        
        # Session statistics
        session_lengths = [len(session) for session in sessions]
        features['total_sessions'] = len(sessions)
        features['avg_session_length'] = np.mean(session_lengths)
        features['max_session_length'] = max(session_lengths)
        features['min_session_length'] = min(session_lengths)
        features['std_session_length'] = np.std(session_lengths) if len(session_lengths) > 1 else 0
        
        # Session engagement depth
        engagement_per_session = []
        for session in sessions:
            engagement_events = self._filter_engagement_events(session)
            engagement_ratio = len(engagement_events) / len(session) if session else 0
            engagement_per_session.append(engagement_ratio)
        
        features['avg_session_engagement'] = np.mean(engagement_per_session)
        features['max_session_engagement'] = max(engagement_per_session) if engagement_per_session else 0
        
        # Session consistency
        features['session_consistency'] = 1.0 - (np.std(session_lengths) / max(np.mean(session_lengths), 1))
        
        return features
    
    def _calculate_feature_engagement(self, events: List[Dict]) -> Dict[str, float]:
        """Calculate feature adoption and engagement"""
        
        features = {}
        
        # Feature usage events
        feature_events = [
            e for e in events
            if e.get('event_type', '').startswith('feature_')
        ]
        
        if feature_events:
            # Unique features used
            unique_features = set(
                e.get('event_type', '').replace('feature_', '')
                for e in feature_events
            )
            
            features['features_adopted'] = len(unique_features)
            features['feature_adoption_rate'] = len(unique_features) / 50.0  # Assuming 50 total features
            
            # Feature usage depth
            feature_counts = Counter(
                e.get('event_type', '') for e in feature_events
            )
            
            if feature_counts:
                features['feature_depth'] = np.mean(list(feature_counts.values()))
                features['feature_breadth'] = len(feature_counts)
                features['max_feature_usage'] = max(feature_counts.values())
                
                # Feature stickiness (features used multiple times)
                sticky_features = sum(1 for count in feature_counts.values() if count > 1)
                features['feature_stickiness'] = self.safe_divide(sticky_features, len(feature_counts))
        else:
            features['features_adopted'] = 0
            features['feature_adoption_rate'] = 0
            features['feature_depth'] = 0
            features['feature_breadth'] = 0
        
        return features
    
    def _calculate_content_engagement(self, events: List[Dict]) -> Dict[str, float]:
        """Calculate content engagement metrics"""
        
        features = {}
        
        content_events = [
            e for e in events
            if any(content_type in e.get('event_type', '') for content_type in 
                   ['content_viewed', 'article_read', 'video_watched', 'document_opened'])
        ]
        
        features['content_engagement_count'] = len(content_events)
        features['content_engagement_ratio'] = self.safe_divide(len(content_events), len(events))
        
        # Content depth (time spent)
        total_content_time = 0
        for event in content_events:
            duration = event.get('properties', {}).get('duration', 0)
            try:
                total_content_time += float(duration)
            except (ValueError, TypeError):
                continue
        
        features['total_content_time'] = total_content_time / 60.0  # Convert to minutes
        features['avg_content_time'] = self.safe_divide(
            total_content_time, len(content_events)
        ) / 60.0 if content_events else 0
        
        return features
    
    def _calculate_communication_engagement(self, events: List[Dict]) -> Dict[str, float]:
        """Calculate communication and messaging engagement"""
        
        features = {}
        
        communication_events = [
            e for e in events
            if any(comm_type in e.get('event_type', '') for comm_type in 
                   ['message_opened', 'message_clicked', 'email_opened', 'notification_clicked'])
        ]
        
        features['communication_engagement_count'] = len(communication_events)
        features['communication_engagement_ratio'] = self.safe_divide(
            len(communication_events), len(events)
        )
        
        # Response rates
        message_sent_events = [
            e for e in events
            if 'message_sent' in e.get('event_type', '') or 'email_sent' in e.get('event_type', '')
        ]
        
        if message_sent_events:
            features['message_response_rate'] = self.safe_divide(
                len(communication_events), len(message_sent_events)
            )
        else:
            features['message_response_rate'] = 0
        
        return features


class MonetizationFeatureExtractor(BaseFeatureExtractor):
    """Extract monetization-related features"""
    
    def extract(
        self,
        user_data: Dict,
        events: List[Dict],
        context: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Extract monetization features"""
        
        features = {}
        
        try:
            # User subscription and payment data
            features['is_paid'] = 1 if user_data.get('subscription_status') in ['paid', 'active'] else 0
            features['ltv_prediction'] = user_data.get('ltv_prediction', 0.0)
            features['upgrade_probability'] = user_data.get('upgrade_probability', 0.0)
            features['churn_risk_score'] = user_data.get('churn_risk_score', 0.0)
            
            # Trial status
            trial_end_str = user_data.get('trial_ends_at', '')
            if trial_end_str:
                try:
                    trial_end = datetime.fromisoformat(trial_end_str.replace('Z', '+00:00'))
                    now = datetime.now(trial_end.tzinfo) if trial_end.tzinfo else datetime.now()
                    days_until_trial_end = (trial_end - now).days
                    
                    features['days_until_trial_end'] = max(0, days_until_trial_end)
                    features['is_in_trial'] = 1 if days_until_trial_end > 0 else 0
                    features['trial_days_remaining_ratio'] = max(0, days_until_trial_end) / 30.0  # Assume 30-day trial
                except Exception:
                    features['days_until_trial_end'] = 0
                    features['is_in_trial'] = 0
                    features['trial_days_remaining_ratio'] = 0
            else:
                features['days_until_trial_end'] = 0
                features['is_in_trial'] = 0
                features['trial_days_remaining_ratio'] = 0
            
            # Subscription duration
            subscription_start_str = user_data.get('subscription_started_at', '')
            if subscription_start_str:
                try:
                    subscription_start = datetime.fromisoformat(subscription_start_str.replace('Z', '+00:00'))
                    now = datetime.now(subscription_start.tzinfo) if subscription_start.tzinfo else datetime.now()
                    subscription_days = (now - subscription_start).days
                    features['subscription_duration_days'] = max(0, subscription_days)
                    features['log_subscription_duration'] = self.safe_log(subscription_days + 1)
                except Exception:
                    features['subscription_duration_days'] = 0
            else:
                features['subscription_duration_days'] = 0
            
            # Revenue and purchase events
            monetization_events = self._filter_monetization_events(events)
            features['monetization_event_count'] = len(monetization_events)
            features['monetization_event_ratio'] = self.safe_divide(
                len(monetization_events), len(events)
            )
            
            # Purchase analysis
            purchase_events = [
                e for e in events
                if 'purchase' in e.get('event_type', '') or 'payment' in e.get('event_type', '')
            ]
            
            features.update(self._analyze_purchases(purchase_events))
            
            # Pricing page engagement
            pricing_events = [
                e for e in events
                if any(term in e.get('event_type', '') for term in ['pricing', 'plan', 'upgrade'])
            ]
            
            features['pricing_page_views'] = len(pricing_events)
            features['pricing_engagement_ratio'] = self.safe_divide(
                len(pricing_events), len(events)
            )
            
            # Payment method and billing
            features.update(self._analyze_payment_behavior(events))
            
            # Conversion funnel analysis
            features.update(self._analyze_conversion_funnel(events))
            
        except Exception as e:
            logger.warning(f"Monetization feature extraction failed: {e}")
        
        return features
    
    def _filter_monetization_events(self, events: List[Dict]) -> List[Dict]:
        """Filter events related to monetization"""
        
        monetization_types = [
            'purchase', 'subscription_started', 'payment_added', 'billing_updated',
            'pricing_viewed', 'upgrade_clicked', 'plan_changed', 'discount_applied',
            'refund_requested', 'chargeback', 'payment_failed'
        ]
        
        return [
            e for e in events
            if any(mt in e.get('event_type', '') for mt in monetization_types)
        ]
    
    def _analyze_purchases(self, purchase_events: List[Dict]) -> Dict[str, float]:
        """Analyze purchase behavior"""
        
        features = {}
        
        features['total_purchases'] = len(purchase_events)
        
        if not purchase_events:
            features['total_revenue'] = 0.0
            features['avg_purchase_value'] = 0.0
            features['max_purchase_value'] = 0.0
            features['purchase_frequency'] = 0.0
            return features
        
        # Extract purchase amounts
        amounts = []
        for event in purchase_events:
            properties = event.get('properties', {})
            amount = properties.get('amount', properties.get('value', 0))
            
            try:
                amounts.append(float(amount))
            except (ValueError, TypeError):
                continue
        
        if amounts:
            features['total_revenue'] = sum(amounts)
            features['avg_purchase_value'] = np.mean(amounts)
            features['max_purchase_value'] = max(amounts)
            features['min_purchase_value'] = min(amounts)
            features['std_purchase_value'] = np.std(amounts) if len(amounts) > 1 else 0
            features['log_total_revenue'] = self.safe_log(features['total_revenue'])
        else:
            features['total_revenue'] = 0.0
            features['avg_purchase_value'] = 0.0
            features['max_purchase_value'] = 0.0
        
        # Purchase timing analysis
        timestamps = []
        for event in purchase_events:
            timestamp_str = event.get('timestamp', event.get('created_at', ''))
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    timestamps.append(timestamp)
                except Exception:
                    continue
        
        if len(timestamps) > 1:
            timestamps.sort()
            time_span = (timestamps[-1] - timestamps[0]).total_seconds() / (24 * 3600)  # days
            features['purchase_frequency'] = len(purchase_events) / max(time_span, 1)
            
            # Time between purchases
            inter_purchase_times = [
                (timestamps[i+1] - timestamps[i]).days
                for i in range(len(timestamps)-1)
            ]
            
            features['avg_days_between_purchases'] = np.mean(inter_purchase_times)
            features['std_days_between_purchases'] = np.std(inter_purchase_times) if len(inter_purchase_times) > 1 else 0
        else:
            features['purchase_frequency'] = 0.0
        
        return features
    
    def _analyze_payment_behavior(self, events: List[Dict]) -> Dict[str, float]:
        """Analyze payment-related behavior"""
        
        features = {}
        
        # Payment failures
        payment_failure_events = [
            e for e in events
            if 'payment_failed' in e.get('event_type', '') or 'payment_declined' in e.get('event_type', '')
        ]
        
        features['payment_failure_count'] = len(payment_failure_events)
        features['payment_failure_ratio'] = self.safe_divide(
            len(payment_failure_events), len(events)
        )
        
        # Payment method changes
        payment_method_events = [
            e for e in events
            if 'payment_method' in e.get('event_type', '') or 'billing' in e.get('event_type', '')
        ]
        
        features['payment_method_changes'] = len(payment_method_events)
        
        # Refund requests
        refund_events = [
            e for e in events
            if 'refund' in e.get('event_type', '') or 'chargeback' in e.get('event_type', '')
        ]
        
        features['refund_request_count'] = len(refund_events)
        
        return features
    
    def _analyze_conversion_funnel(self, events: List[Dict]) -> Dict[str, float]:
        """Analyze conversion funnel progression"""
        
        features = {}
        
        # Define funnel stages
        funnel_stages = {
            'awareness': ['page_view', 'content_view', 'search'],
            'interest': ['pricing_view', 'feature_explore', 'demo_request'],
            'consideration': ['trial_start', 'product_compare', 'contact_sales'],
            'purchase': ['cart_add', 'checkout_start', 'purchase_complete'],
            'retention': ['login', 'feature_use', 'content_consume']
        }
        
        # Count events in each stage
        for stage, stage_events in funnel_stages.items():
            stage_count = sum(
                1 for e in events
                if any(se in e.get('event_type', '') for se in stage_events)
            )
            
            features[f'funnel_{stage}_count'] = stage_count
            features[f'funnel_{stage}_ratio'] = self.safe_divide(stage_count, len(events))
        
        # Funnel conversion rates
        awareness_count = features.get('funnel_awareness_count', 0)
        interest_count = features.get('funnel_interest_count', 0)
        consideration_count = features.get('funnel_consideration_count', 0)
        purchase_count = features.get('funnel_purchase_count', 0)
        
        features['awareness_to_interest_rate'] = self.safe_divide(interest_count, awareness_count)
        features['interest_to_consideration_rate'] = self.safe_divide(consideration_count, interest_count)
        features['consideration_to_purchase_rate'] = self.safe_divide(purchase_count, consideration_count)
        features['overall_conversion_rate'] = self.safe_divide(purchase_count, awareness_count)
        
        return features


class SessionFeatureExtractor(BaseFeatureExtractor):
    """Extract session-based features"""
    
    def extract(
        self,
        user_data: Dict,
        events: List[Dict],
        context: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Extract session features"""
        
        features = {}
        
        try:
            # Identify sessions
            sessions = self._identify_sessions(events)
            
            if not sessions:
                return self._get_default_session_features()
            
            # Session count and basic stats
            features['total_sessions'] = len(sessions)
            features['log_total_sessions'] = self.safe_log(len(sessions))
            
            # Session length analysis
            session_lengths = [len(session) for session in sessions]
            features['avg_session_length'] = np.mean(session_lengths)
            features['max_session_length'] = max(session_lengths)
            features['min_session_length'] = min(session_lengths)
            features['std_session_length'] = np.std(session_lengths) if len(session_lengths) > 1 else 0
            features['median_session_length'] = np.median(session_lengths)
            
            # Session duration analysis
            session_durations = self._calculate_session_durations(sessions)
            if session_durations:
                features['avg_session_duration_minutes'] = np.mean(session_durations)
                features['max_session_duration_minutes'] = max(session_durations)
                features['total_session_time_minutes'] = sum(session_durations)
                features['log_total_session_time'] = self.safe_log(sum(session_durations))
            
            # Session frequency
            features.update(self._calculate_session_frequency(sessions))
            
            # Session quality metrics
            features.update(self._calculate_session_quality(sessions))
            
            # Session patterns
            features.update(self._analyze_session_patterns(sessions))
            
        except Exception as e:
            logger.warning(f"Session feature extraction failed: {e}")
            features.update(self._get_default_session_features())
        
        return features
    
    def _identify_sessions(
        self,
        events: List[Dict],
        timeout_minutes: int = 30
    ) -> List[List[Dict]]:
        """Identify user sessions from events"""
        
        if not events:
            return []
        
        # Sort events by timestamp
        sorted_events = []
        for event in events:
            timestamp_str = event.get('timestamp', event.get('created_at', ''))
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    sorted_events.append((timestamp, event))
                except Exception:
                    continue
        
        sorted_events.sort(key=lambda x: x[0])
        
        if not sorted_events:
            return []
        
        sessions = []
        current_session = [sorted_events[0][1]]
        last_time = sorted_events[0][0]
        
        for timestamp, event in sorted_events[1:]:
            if (timestamp - last_time).total_seconds() > timeout_minutes * 60:
                # New session
                sessions.append(current_session)
                current_session = [event]
            else:
                current_session.append(event)
            
            last_time = timestamp
        
        if current_session:
            sessions.append(current_session)
        
        return sessions
    
    def _calculate_session_durations(self, sessions: List[List[Dict]]) -> List[float]:
        """Calculate duration of each session in minutes"""
        
        durations = []
        
        for session in sessions:
            if len(session) < 2:
                durations.append(0.0)
                continue
            
            timestamps = []
            for event in session:
                timestamp_str = event.get('timestamp', event.get('created_at', ''))
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        timestamps.append(timestamp)
                    except Exception:
                        continue
            
            if len(timestamps) >= 2:
                timestamps.sort()
                duration = (timestamps[-1] - timestamps[0]).total_seconds() / 60.0
                durations.append(duration)
            else:
                durations.append(0.0)
        
        return durations
    
    def _calculate_session_frequency(self, sessions: List[List[Dict]]) -> Dict[str, float]:
        """Calculate session frequency metrics"""
        
        features = {}
        
        if len(sessions) < 2:
            features['sessions_per_day'] = 0.0
            features['avg_days_between_sessions'] = 0.0
            return features
        
        # Get session start times
        session_times = []
        for session in sessions:
            if session:
                timestamp_str = session[0].get('timestamp', session[0].get('created_at', ''))
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        session_times.append(timestamp)
                    except Exception:
                        continue
        
        if len(session_times) >= 2:
            session_times.sort()
            
            # Calculate frequency
            time_span = (session_times[-1] - session_times[0]).total_seconds() / (24 * 3600)  # days
            features['sessions_per_day'] = len(sessions) / max(time_span, 1)
            
            # Time between sessions
            inter_session_times = [
                (session_times[i+1] - session_times[i]).total_seconds() / (24 * 3600)
                for i in range(len(session_times)-1)
            ]
            
            features['avg_days_between_sessions'] = np.mean(inter_session_times)
            features['std_days_between_sessions'] = np.std(inter_session_times) if len(inter_session_times) > 1 else 0
        
        return features
    
    def _calculate_session_quality(self, sessions: List[List[Dict]]) -> Dict[str, float]:
        """Calculate session quality metrics"""
        
        features = {}
        
        if not sessions:
            return features
        
        # Session depth (events per session)
        session_depths = [len(session) for session in sessions]
        features['avg_session_depth'] = np.mean(session_depths)
        features['session_depth_consistency'] = 1.0 - (np.std(session_depths) / max(np.mean(session_depths), 1))
        
        # Session engagement
        engagement_scores = []
        for session in sessions:
            engagement_events = [
                e for e in session
                if any(et in e.get('event_type', '') for et in 
                       ['feature_used', 'content_viewed', 'action_completed'])
            ]
            
            engagement_score = len(engagement_events) / len(session) if session else 0
            engagement_scores.append(engagement_score)
        
        features['avg_session_engagement'] = np.mean(engagement_scores)
        features['max_session_engagement'] = max(engagement_scores) if engagement_scores else 0
        
        # Session completion (sessions ending with goal achievement)
        completion_events = ['purchase', 'signup_complete', 'goal_achieved', 'task_complete']
        completed_sessions = 0
        
        for session in sessions:
            if session:
                last_event = session[-1]
                if any(ce in last_event.get('event_type', '') for ce in completion_events):
                    completed_sessions += 1
        
        features['session_completion_rate'] = self.safe_divide(completed_sessions, len(sessions))
        
        return features
    
    def _analyze_session_patterns(self, sessions: List[List[Dict]]) -> Dict[str, float]:
        """Analyze session patterns and behaviors"""
        
        features = {}
        
        if not sessions:
            return features
        
        # Session start patterns
        session_start_hours = []
        session_start_days = []
        
        for session in sessions:
            if session:
                timestamp_str = session[0].get('timestamp', session[0].get('created_at', ''))
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        session_start_hours.append(timestamp.hour)
                        session_start_days.append(timestamp.weekday())
                    except Exception:
                        continue
        
        if session_start_hours:
            features['avg_session_start_hour'] = np.mean(session_start_hours)
            features['session_hour_diversity'] = len(set(session_start_hours)) / 24.0
            
            # Peak session hours
            hour_counts = Counter(session_start_hours)
            if hour_counts:
                features['peak_session_hour'] = hour_counts.most_common(1)[0][0]
                features['peak_hour_concentration'] = hour_counts.most_common(1)[0][1] / len(session_start_hours)
        
        if session_start_days:
            features['session_day_diversity'] = len(set(session_start_days)) / 7.0
            features['weekend_session_ratio'] = sum(1 for day in session_start_days if day >= 5) / len(session_start_days)
        
        # Session type classification
        short_sessions = sum(1 for session in sessions if len(session) <= 3)
        medium_sessions = sum(1 for session in sessions if 4 <= len(session) <= 10)
        long_sessions = sum(1 for session in sessions if len(session) > 10)
        
        total_sessions = len(sessions)
        features['short_session_ratio'] = short_sessions / total_sessions
        features['medium_session_ratio'] = medium_sessions / total_sessions
        features['long_session_ratio'] = long_sessions / total_sessions
        
        return features
    
    def _get_default_session_features(self) -> Dict[str, float]:
        """Get default session features for users with no sessions"""
        
        return {
            'total_sessions': 0,
            'avg_session_length': 0,
            'avg_session_duration_minutes': 0,
            'sessions_per_day': 0,
            'avg_session_engagement': 0,
            'session_completion_rate': 0
        }


class ContentFeatureExtractor(BaseFeatureExtractor):
    """Extract content interaction features"""
    
    def extract(
        self,
        user_data: Dict,
        events: List[Dict],
        context: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Extract content features"""
        
        features = {}
        
        try:
            # Content consumption events
            content_events = self._filter_content_events(events)
            
            features['content_interaction_count'] = len(content_events)
            features['content_interaction_ratio'] = self.safe_divide(
                len(content_events), len(events)
            )
            
            if content_events:
                features.update(self._analyze_content_consumption(content_events))
                features.update(self._analyze_content_preferences(content_events))
                features.update(self._analyze_content_depth(content_events))
            
        except Exception as e:
            logger.warning(f"Content feature extraction failed: {e}")
        
        return features
    
    def _filter_content_events(self, events: List[Dict]) -> List[Dict]:
        """Filter content-related events"""
        
        content_types = [
            'content_viewed', 'article_read', 'video_watched', 'document_opened',
            'tutorial_accessed', 'help_viewed', 'blog_read', 'guide_followed'
        ]
        
        return [
            e for e in events
            if any(ct in e.get('event_type', '') for ct in content_types)
        ]
    
    def _analyze_content_consumption(self, content_events: List[Dict]) -> Dict[str, float]:
        """Analyze content consumption patterns"""
        
        features = {}
        
        # Content types consumed
        content_types = [e.get('event_type', '') for e in content_events]
        unique_content_types = set(content_types)
        
        features['unique_content_types'] = len(unique_content_types)
        features['content_type_diversity'] = len(unique_content_types) / max(len(content_events), 1)
        
        # Time spent on content
        total_time = 0
        time_events = 0
        
        for event in content_events:
            duration = event.get('properties', {}).get('duration', 0)
            try:
                duration_seconds = float(duration)
                total_time += duration_seconds
                time_events += 1
            except (ValueError, TypeError):
                continue
        
        if time_events > 0:
            features['avg_content_time_seconds'] = total_time / time_events
            features['total_content_time_minutes'] = total_time / 60.0
            features['log_total_content_time'] = self.safe_log(total_time / 60.0)
        
        return features
    
    def _analyze_content_preferences(self, content_events: List[Dict]) -> Dict[str, float]:
        """Analyze content preferences"""
        
        features = {}
        
        # Content categories
        categories = []
        for event in content_events:
            category = event.get('properties', {}).get('category', 'unknown')
            if category != 'unknown':
                categories.append(category)
        
        if categories:
            category_counts = Counter(categories)
            features['preferred_content_categories'] = len(category_counts)
            
            # Top category preference strength
            if category_counts:
                top_category_count = category_counts.most_common(1)[0][1]
                features['top_category_preference'] = top_category_count / len(categories)
        
        return features
    
    def _analyze_content_depth(self, content_events: List[Dict]) -> Dict[str, float]:
        """Analyze content engagement depth"""
        
        features = {}
        
        # Completion rates
        completed_content = 0
        completion_events = 0
        
        for event in content_events:
            properties = event.get('properties', {})
            
            if 'completion_rate' in properties:
                try:
                    completion_rate = float(properties['completion_rate'])
                    completed_content += completion_rate
                    completion_events += 1
                except (ValueError, TypeError):
                    continue
            elif 'completed' in properties:
                completed_content += 1 if properties['completed'] else 0
                completion_events += 1
        
        if completion_events > 0:
            features['avg_content_completion_rate'] = completed_content / completion_events
            features['high_completion_content_ratio'] = sum(
                1 for event in content_events
                if event.get('properties', {}).get('completion_rate', 0) > 0.8
            ) / len(content_events)
        
        return features


class DeviceFeatureExtractor(BaseFeatureExtractor):
    """Extract device and technical features"""
    
    def extract(
        self,
        user_data: Dict,
        events: List[Dict],
        context: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Extract device features"""
        
        features = {}
        
        try:
            # Device information from events
            devices = []
            browsers = []
            platforms = []
            
            for event in events:
                context_data = event.get('context', {})
                
                # Device type
                device = context_data.get('device_type', 'unknown')
                if device != 'unknown':
                    devices.append(device)
                
                # Browser
                browser = context_data.get('browser', 'unknown')
                if browser != 'unknown':
                    browsers.append(browser)
                
                # Platform
                platform = context_data.get('platform', 'unknown')
                if platform != 'unknown':
                    platforms.append(platform)
            
            # Device diversity
            if devices:
                unique_devices = set(devices)
                features['unique_devices'] = len(unique_devices)
                features['device_diversity'] = len(unique_devices) / len(devices)
                
                # Device preferences
                device_counts = Counter(devices)
                if device_counts:
                    primary_device = device_counts.most_common(1)[0][0]
                    features['primary_device_mobile'] = 1 if 'mobile' in primary_device.lower() else 0
                    features['primary_device_desktop'] = 1 if 'desktop' in primary_device.lower() else 0
                    features['primary_device_tablet'] = 1 if 'tablet' in primary_device.lower() else 0
            
            # Browser patterns
            if browsers:
                unique_browsers = set(browsers)
                features['unique_browsers'] = len(unique_browsers)
                features['browser_consistency'] = 1.0 - (len(unique_browsers) / len(browsers))
            
            # Platform patterns
            if platforms:
                unique_platforms = set(platforms)
                features['unique_platforms'] = len(unique_platforms)
                features['cross_platform_usage'] = 1 if len(unique_platforms) > 1 else 0
            
        except Exception as e:
            logger.warning(f"Device feature extraction failed: {e}")
        
        return features


class GeographicFeatureExtractor(BaseFeatureExtractor):
    """Extract geographic and location features"""
    
    def extract(
        self,
        user_data: Dict,
        events: List[Dict],
        context: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Extract geographic features"""
        
        features = {}
        
        try:
            # Location information from events
            countries = []
            cities = []
            timezones = []
            
            for event in events:
                context_data = event.get('context', {})
                geo_data = context_data.get('geo', {})
                
                # Country
                country = geo_data.get('country', 'unknown')
                if country != 'unknown':
                    countries.append(country)
                
                # City
                city = geo_data.get('city', 'unknown')
                if city != 'unknown':
                    cities.append(city)
                
                # Timezone
                timezone = context_data.get('timezone', 'unknown')
                if timezone != 'unknown':
                    timezones.append(timezone)
            
            # Location consistency
            if countries:
                unique_countries = set(countries)
                features['unique_countries'] = len(unique_countries)
                features['location_consistency'] = 1.0 - (len(unique_countries) / len(countries))
                features['international_usage'] = 1 if len(unique_countries) > 1 else 0
            
            if cities:
                unique_cities = set(cities)
                features['unique_cities'] = len(unique_cities)
                features['city_mobility'] = len(unique_cities) / len(cities)
            
            # Timezone patterns
            if timezones:
                unique_timezones = set(timezones)
                features['unique_timezones'] = len(unique_timezones)
                features['timezone_travel'] = 1 if len(unique_timezones) > 1 else 0
            
        except Exception as e:
            logger.warning(f"Geographic feature extraction failed: {e}")
        
        return features
