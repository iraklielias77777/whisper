"""
Real-Time Analytics Integration
Analyzes metrics and adapts strategy in real-time for optimal performance
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
import numpy as np

# Statistical analysis imports
try:
    from scipy import stats
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .config_schemas import AISystemConfiguration

logger = logging.getLogger(__name__)

@dataclass
class AnalyticsConfiguration:
    """Configuration for analytics engine"""
    baseline_window: int = 7  # days
    anomaly_threshold: float = 2.5  # standard deviations
    adaptation_sensitivity: float = 0.15
    min_sample_size: int = 50
    confidence_threshold: float = 0.75
    real_time_enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class MetricBaseline:
    """Baseline metrics for comparison"""
    metric_name: str
    baseline_value: float
    standard_deviation: float
    sample_count: int
    calculated_at: datetime
    confidence_interval: Tuple[float, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'calculated_at': self.calculated_at.isoformat()
        }

@dataclass
class AdaptationResult:
    """Result of analytics-driven adaptation"""
    adaptation_id: str
    customer_id: str
    timestamp: datetime
    trigger_metrics: Dict[str, float]
    baseline_comparison: Dict[str, Any]
    adaptations_applied: List[Dict[str, Any]]
    confidence_score: float
    expected_impact: Dict[str, Any]
    monitoring_period: int  # minutes
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }

class PerformanceAnalyzer:
    """Analyzes performance metrics and identifies optimization opportunities"""
    
    def __init__(self, config: AnalyticsConfiguration):
        self.config = config
        self.metric_history = defaultdict(lambda: deque(maxlen=1000))
        self.baselines = {}
        self.anomaly_detector = None
        
        if SCIPY_AVAILABLE:
            self.scaler = StandardScaler()
            self.anomaly_detector = DBSCAN(eps=0.3, min_samples=5)
    
    def update_metrics(self, customer_id: str, metrics: Dict[str, float]):
        """Update metrics for performance tracking"""
        timestamp = datetime.utcnow()
        
        for metric_name, value in metrics.items():
            metric_key = f"{customer_id}_{metric_name}"
            self.metric_history[metric_key].append({
                'timestamp': timestamp,
                'value': value,
                'customer_id': customer_id
            })
        
        # Update baselines if enough data
        self._update_baselines(customer_id, metrics)
    
    def _update_baselines(self, customer_id: str, current_metrics: Dict[str, float]):
        """Update baseline metrics for comparison"""
        for metric_name, current_value in current_metrics.items():
            metric_key = f"{customer_id}_{metric_name}"
            history = list(self.metric_history[metric_key])
            
            if len(history) >= self.config.min_sample_size:
                values = [h['value'] for h in history[-self.config.min_sample_size:]]
                
                baseline = MetricBaseline(
                    metric_name=metric_name,
                    baseline_value=np.mean(values),
                    standard_deviation=np.std(values),
                    sample_count=len(values),
                    calculated_at=datetime.utcnow(),
                    confidence_interval=stats.norm.interval(
                        0.95, 
                        loc=np.mean(values), 
                        scale=np.std(values)
                    ) if SCIPY_AVAILABLE else (np.mean(values) - np.std(values), np.mean(values) + np.std(values))
                )
                
                self.baselines[metric_key] = baseline
    
    def detect_significant_changes(self, customer_id: str, current_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Detect statistically significant changes in metrics"""
        significant_changes = []
        
        for metric_name, current_value in current_metrics.items():
            metric_key = f"{customer_id}_{metric_name}"
            baseline = self.baselines.get(metric_key)
            
            if baseline and baseline.standard_deviation > 0:
                z_score = abs((current_value - baseline.baseline_value) / baseline.standard_deviation)
                
                if z_score > self.config.anomaly_threshold:
                    direction = 'positive' if current_value > baseline.baseline_value else 'negative'
                    magnitude = abs(current_value - baseline.baseline_value) / baseline.baseline_value
                    
                    significant_changes.append({
                        'metric': metric_name,
                        'current_value': current_value,
                        'baseline_value': baseline.baseline_value,
                        'z_score': z_score,
                        'direction': direction,
                        'magnitude': magnitude,
                        'significance': 'high' if z_score > 3 else 'medium',
                        'confidence': min(0.99, z_score / self.config.anomaly_threshold)
                    })
        
        return significant_changes
    
    def predict_trend(self, customer_id: str, metric_name: str, horizon_minutes: int = 60) -> Dict[str, Any]:
        """Predict metric trend for the next period"""
        metric_key = f"{customer_id}_{metric_name}"
        history = list(self.metric_history[metric_key])
        
        if len(history) < 10:
            return {'prediction': 'insufficient_data', 'confidence': 0.0}
        
        # Simple linear trend analysis
        recent_values = [h['value'] for h in history[-20:]]  # Last 20 points
        timestamps = [(h['timestamp'] - history[-20]['timestamp']).total_seconds() for h in history[-20:]]
        
        if SCIPY_AVAILABLE:
            slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps, recent_values)
            
            # Predict value at horizon
            future_timestamp = horizon_minutes * 60  # Convert to seconds
            predicted_value = slope * future_timestamp + intercept
            
            return {
                'prediction': predicted_value,
                'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                'trend_strength': abs(r_value),
                'confidence': max(0.0, 1.0 - p_value),
                'change_rate': slope,
                'prediction_horizon_minutes': horizon_minutes
            }
        else:
            # Simple moving average trend
            short_term = np.mean(recent_values[-5:])
            long_term = np.mean(recent_values[-15:])
            trend = 'increasing' if short_term > long_term else 'decreasing'
            
            return {
                'prediction': short_term,
                'trend_direction': trend,
                'trend_strength': abs(short_term - long_term) / long_term if long_term != 0 else 0,
                'confidence': 0.6,
                'prediction_horizon_minutes': horizon_minutes
            }

class RealTimeAnalyticsEngine:
    """
    Real-time analytics engine for dynamic decision making and adaptation
    """
    
    def __init__(self, config: AISystemConfiguration):
        self.config = config
        self.analytics_config = AnalyticsConfiguration()
        
        # Initialize components
        self.performance_analyzer = PerformanceAnalyzer(self.analytics_config)
        self.metrics_buffer = deque(maxlen=1000)
        self.adaptation_history = deque(maxlen=500)
        
        # State tracking
        self.active_adaptations = {}
        self.customer_baselines = {}
        self.global_performance = {}
        
        logger.info("Real-Time Analytics Engine initialized")
    
    async def analyze_and_adapt(
        self,
        customer_id: str,
        current_metrics: Dict[str, float],
        context: Optional[Dict[str, Any]] = None
    ) -> AdaptationResult:
        """
        Analyze metrics and generate adaptations in real-time
        """
        adaptation_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        
        try:
            logger.info(f"Starting real-time analysis for customer {customer_id}", {
                'adaptation_id': adaptation_id,
                'metrics_count': len(current_metrics)
            })
            
            # Update performance metrics
            self.performance_analyzer.update_metrics(customer_id, current_metrics)
            
            # Get baseline performance for comparison
            baseline = self._get_customer_baseline(customer_id)
            
            # Calculate deviations from baseline
            deviations = self._calculate_deviations(current_metrics, baseline)
            
            # Detect significant changes
            significant_changes = self.performance_analyzer.detect_significant_changes(
                customer_id, 
                current_metrics
            )
            
            # Generate adaptations based on analysis
            adaptations = await self._generate_adaptations(
                customer_id,
                current_metrics,
                significant_changes,
                context or {}
            )
            
            # Calculate confidence and expected impact
            confidence_score = self._calculate_adaptation_confidence(
                significant_changes,
                adaptations
            )
            
            expected_impact = await self._estimate_impact(
                customer_id,
                adaptations,
                current_metrics
            )
            
            # Create adaptation result
            result = AdaptationResult(
                adaptation_id=adaptation_id,
                customer_id=customer_id,
                timestamp=timestamp,
                trigger_metrics=current_metrics,
                baseline_comparison={
                    'baseline': baseline,
                    'deviations': deviations,
                    'significant_changes': significant_changes
                },
                adaptations_applied=adaptations,
                confidence_score=confidence_score,
                expected_impact=expected_impact,
                monitoring_period=self._calculate_monitoring_period(adaptations)
            )
            
            # Store for tracking
            self.adaptation_history.append(result)
            self.active_adaptations[adaptation_id] = result
            
            # Apply adaptations if confidence is high enough
            if confidence_score >= self.analytics_config.confidence_threshold:
                await self._apply_adaptations(customer_id, adaptations)
                logger.info(f"Adaptations applied successfully", {
                    'adaptation_id': adaptation_id,
                    'customer_id': customer_id,
                    'adaptations_count': len(adaptations),
                    'confidence': confidence_score
                })
            else:
                logger.info(f"Adaptations generated but not applied due to low confidence", {
                    'adaptation_id': adaptation_id,
                    'customer_id': customer_id,
                    'confidence': confidence_score,
                    'threshold': self.analytics_config.confidence_threshold
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Real-time analysis failed for customer {customer_id}: {e}")
            return self._create_error_adaptation_result(adaptation_id, customer_id, str(e))
    
    def _get_customer_baseline(self, customer_id: str) -> Dict[str, float]:
        """Get baseline performance metrics for customer"""
        baseline = self.customer_baselines.get(customer_id, {})
        
        if not baseline:
            # Use global baseline if customer baseline not available
            baseline = self.global_performance.copy()
        
        # Default baseline values
        default_baseline = {
            'engagement_rate': 0.25,
            'conversion_rate': 0.15,
            'response_time_ms': 500,
            'satisfaction_score': 0.75,
            'churn_risk': 0.3
        }
        
        # Merge with defaults
        for key, default_value in default_baseline.items():
            if key not in baseline:
                baseline[key] = default_value
        
        return baseline
    
    def _calculate_deviations(
        self, 
        current_metrics: Dict[str, float], 
        baseline: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate deviations from baseline"""
        deviations = {}
        
        for metric_name, current_value in current_metrics.items():
            baseline_value = baseline.get(metric_name, current_value)
            
            if baseline_value == 0:
                percentage_change = 0
            else:
                percentage_change = (current_value - baseline_value) / baseline_value
            
            absolute_change = current_value - baseline_value
            
            deviations[metric_name] = {
                'current': current_value,
                'baseline': baseline_value,
                'absolute_change': absolute_change,
                'percentage_change': percentage_change,
                'direction': 'positive' if absolute_change > 0 else 'negative',
                'magnitude': abs(percentage_change)
            }
        
        return deviations
    
    async def _generate_adaptations(
        self,
        customer_id: str,
        current_metrics: Dict[str, float],
        significant_changes: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate specific adaptations based on analysis"""
        adaptations = []
        
        for change in significant_changes:
            metric = change['metric']
            direction = change['direction']
            magnitude = change['magnitude']
            
            # Generate adaptations based on specific metrics
            if metric == 'engagement_rate' and direction == 'negative':
                adaptations.extend(await self._generate_engagement_adaptations(
                    customer_id, change, context
                ))
            
            elif metric == 'conversion_rate':
                adaptations.extend(await self._generate_conversion_adaptations(
                    customer_id, change, context, direction
                ))
            
            elif metric == 'response_time_ms' and direction == 'positive':
                adaptations.extend(await self._generate_performance_adaptations(
                    customer_id, change, context
                ))
            
            elif metric == 'churn_risk' and direction == 'positive':
                adaptations.extend(await self._generate_retention_adaptations(
                    customer_id, change, context
                ))
            
            elif metric == 'satisfaction_score' and direction == 'negative':
                adaptations.extend(await self._generate_satisfaction_adaptations(
                    customer_id, change, context
                ))
        
        # Remove duplicates and rank by priority
        unique_adaptations = self._deduplicate_adaptations(adaptations)
        ranked_adaptations = self._rank_adaptations(unique_adaptations, significant_changes)
        
        return ranked_adaptations[:10]  # Limit to top 10 adaptations
    
    async def _generate_engagement_adaptations(
        self,
        customer_id: str,
        change: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate adaptations for declining engagement"""
        adaptations = []
        
        magnitude = change['magnitude']
        
        if magnitude > 0.3:  # Significant engagement drop
            adaptations.append({
                'type': 'content_refresh',
                'action': 'switch_to_alternative_content',
                'urgency': 'high',
                'params': {
                    'content_style': 'attention_grabbing',
                    'personalization_boost': 1.5,
                    'urgency_increase': True
                },
                'expected_impact': 0.2,
                'confidence': 0.8,
                'reasoning': f'Engagement dropped by {magnitude:.1%}, switching to high-impact content'
            })
            
            if magnitude > 0.5:  # Severe engagement drop
                adaptations.append({
                    'type': 'channel_switch',
                    'action': 'activate_alternative_channel',
                    'urgency': 'critical',
                    'params': {
                        'channels': ['sms', 'push'],
                        'message': 'urgent_reengagement'
                    },
                    'expected_impact': 0.3,
                    'confidence': 0.7,
                    'reasoning': f'Severe engagement drop ({magnitude:.1%}), activating alternative channels'
                })
        
        return adaptations
    
    async def _generate_conversion_adaptations(
        self,
        customer_id: str,
        change: Dict[str, Any],
        context: Dict[str, Any],
        direction: str
    ) -> List[Dict[str, Any]]:
        """Generate adaptations for conversion rate changes"""
        adaptations = []
        magnitude = change['magnitude']
        
        if direction == 'positive' and magnitude > 0.2:
            # Conversion improving - capitalize on momentum
            adaptations.append({
                'type': 'momentum_capture',
                'action': 'increase_touchpoints',
                'urgency': 'high',
                'params': {
                    'frequency_multiplier': 1.3,
                    'upsell_enabled': True,
                    'referral_request': True
                },
                'expected_impact': 0.15,
                'confidence': 0.85,
                'reasoning': f'Conversion improving by {magnitude:.1%}, capitalizing on momentum'
            })
        
        elif direction == 'negative' and magnitude > 0.2:
            # Conversion declining - intervention needed
            adaptations.append({
                'type': 'conversion_recovery',
                'action': 'optimize_funnel',
                'urgency': 'high',
                'params': {
                    'simplify_cta': True,
                    'reduce_friction': True,
                    'add_urgency': True,
                    'social_proof_boost': True
                },
                'expected_impact': 0.25,
                'confidence': 0.75,
                'reasoning': f'Conversion declining by {magnitude:.1%}, optimizing conversion funnel'
            })
        
        return adaptations
    
    async def _generate_retention_adaptations(
        self,
        customer_id: str,
        change: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate adaptations for increased churn risk"""
        adaptations = []
        magnitude = change['magnitude']
        
        if magnitude > 0.2:  # Significant churn risk increase
            adaptations.append({
                'type': 'retention_intervention',
                'action': 'activate_retention_protocol',
                'urgency': 'critical',
                'params': {
                    'offer_incentive': True,
                    'personal_outreach': True,
                    'support_priority': 'high',
                    'success_manager_assignment': True
                },
                'expected_impact': 0.4,
                'confidence': 0.9,
                'reasoning': f'Churn risk increased by {magnitude:.1%}, activating comprehensive retention protocol'
            })
            
            adaptations.append({
                'type': 'value_demonstration',
                'action': 'showcase_unused_features',
                'urgency': 'high',
                'params': {
                    'feature_tour': True,
                    'value_calculation': True,
                    'success_stories': True
                },
                'expected_impact': 0.3,
                'confidence': 0.8,
                'reasoning': 'Demonstrating additional value to reduce churn risk'
            })
        
        return adaptations
    
    def _calculate_adaptation_confidence(
        self,
        significant_changes: List[Dict[str, Any]],
        adaptations: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence in proposed adaptations"""
        if not adaptations:
            return 0.0
        
        # Base confidence from adaptations
        adaptation_confidences = [a.get('confidence', 0.5) for a in adaptations]
        base_confidence = np.mean(adaptation_confidences)
        
        # Boost confidence based on significance of changes
        significance_boost = 0.0
        for change in significant_changes:
            if change.get('significance') == 'high':
                significance_boost += 0.1
            elif change.get('significance') == 'medium':
                significance_boost += 0.05
        
        # Combine confidences
        total_confidence = min(0.95, base_confidence + significance_boost)
        
        return max(0.1, total_confidence)
    
    async def _estimate_impact(
        self,
        customer_id: str,
        adaptations: List[Dict[str, Any]],
        current_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Estimate expected impact of adaptations"""
        if not adaptations:
            return {'total_impact': 0.0, 'metric_improvements': {}}
        
        # Calculate expected improvements by metric
        metric_improvements = {}
        
        for adaptation in adaptations:
            expected_impact = adaptation.get('expected_impact', 0.1)
            adaptation_type = adaptation.get('type', 'unknown')
            
            # Map adaptation types to metrics they affect
            impact_mapping = {
                'content_refresh': ['engagement_rate'],
                'channel_switch': ['engagement_rate', 'response_rate'],
                'momentum_capture': ['conversion_rate', 'revenue'],
                'conversion_recovery': ['conversion_rate'],
                'retention_intervention': ['churn_risk', 'satisfaction_score'],
                'value_demonstration': ['engagement_rate', 'churn_risk'],
                'performance_optimization': ['response_time_ms', 'satisfaction_score']
            }
            
            affected_metrics = impact_mapping.get(adaptation_type, ['engagement_rate'])
            
            for metric in affected_metrics:
                if metric not in metric_improvements:
                    metric_improvements[metric] = 0.0
                metric_improvements[metric] += expected_impact
        
        # Calculate total impact
        total_impact = sum(metric_improvements.values()) / len(metric_improvements) if metric_improvements else 0.0
        
        return {
            'total_impact': min(1.0, total_impact),
            'metric_improvements': metric_improvements,
            'affected_metrics': list(metric_improvements.keys()),
            'confidence': self._calculate_impact_confidence(adaptations)
        }
    
    def _calculate_impact_confidence(self, adaptations: List[Dict[str, Any]]) -> float:
        """Calculate confidence in impact estimates"""
        if not adaptations:
            return 0.0
        
        confidences = [a.get('confidence', 0.5) for a in adaptations]
        return np.mean(confidences)
    
    async def _apply_adaptations(self, customer_id: str, adaptations: List[Dict[str, Any]]):
        """Apply the generated adaptations"""
        try:
            for adaptation in adaptations:
                adaptation_type = adaptation.get('type')
                action = adaptation.get('action')
                params = adaptation.get('params', {})
                
                logger.info(f"Applying adaptation: {adaptation_type}", {
                    'customer_id': customer_id,
                    'action': action,
                    'urgency': adaptation.get('urgency'),
                    'expected_impact': adaptation.get('expected_impact')
                })
                
                # This would integrate with the actual services to apply changes
                # For now, we'll just log the adaptations
                await self._execute_adaptation(customer_id, adaptation_type, action, params)
            
        except Exception as e:
            logger.error(f"Failed to apply adaptations for customer {customer_id}: {e}")
    
    async def _execute_adaptation(
        self, 
        customer_id: str, 
        adaptation_type: str, 
        action: str, 
        params: Dict[str, Any]
    ):
        """Execute a specific adaptation"""
        # This would integrate with the orchestration service to make changes
        # Implementation would depend on the specific adaptation type
        logger.info(f"Executed adaptation: {adaptation_type} - {action}", {
            'customer_id': customer_id,
            'params': params
        })
    
    # Additional helper methods...
    def _deduplicate_adaptations(self, adaptations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate adaptations"""
        seen = set()
        unique = []
        
        for adaptation in adaptations:
            key = f"{adaptation.get('type')}_{adaptation.get('action')}"
            if key not in seen:
                seen.add(key)
                unique.append(adaptation)
        
        return unique
    
    def _rank_adaptations(
        self, 
        adaptations: List[Dict[str, Any]], 
        significant_changes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rank adaptations by priority"""
        def adaptation_score(adaptation):
            urgency_scores = {'critical': 1.0, 'high': 0.8, 'medium': 0.6, 'low': 0.4}
            urgency = urgency_scores.get(adaptation.get('urgency', 'medium'), 0.6)
            confidence = adaptation.get('confidence', 0.5)
            impact = adaptation.get('expected_impact', 0.1)
            
            return urgency * 0.4 + confidence * 0.3 + impact * 0.3
        
        return sorted(adaptations, key=adaptation_score, reverse=True)
    
    def _calculate_monitoring_period(self, adaptations: List[Dict[str, Any]]) -> int:
        """Calculate how long to monitor adaptations"""
        urgency_periods = {
            'critical': 15,  # 15 minutes
            'high': 30,      # 30 minutes
            'medium': 60,    # 1 hour
            'low': 120       # 2 hours
        }
        
        if not adaptations:
            return 60
        
        max_urgency = max(
            urgency_periods.get(a.get('urgency', 'medium'), 60) 
            for a in adaptations
        )
        
        return max_urgency
    
    def _create_error_adaptation_result(
        self, 
        adaptation_id: str, 
        customer_id: str, 
        error: str
    ) -> AdaptationResult:
        """Create error result when analysis fails"""
        return AdaptationResult(
            adaptation_id=adaptation_id,
            customer_id=customer_id,
            timestamp=datetime.utcnow(),
            trigger_metrics={},
            baseline_comparison={'error': error},
            adaptations_applied=[],
            confidence_score=0.0,
            expected_impact={'total_impact': 0.0, 'error': error},
            monitoring_period=60
        )
