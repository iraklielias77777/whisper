"""
Performance Optimizer for User Whisperer Platform
Provides automated performance monitoring, optimization, and scaling decisions
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
import time
from dataclasses import dataclass
import statistics

logger = logging.getLogger(__name__)

class PerformanceMetric(Enum):
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    CACHE_HIT_RATE = "cache_hit_rate"
    DATABASE_CONNECTIONS = "database_connections"
    QUEUE_LENGTH = "queue_length"

class OptimizationAction(Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    CACHE_WARM = "cache_warm"
    CACHE_INVALIDATE = "cache_invalidate"
    INDEX_REBUILD = "index_rebuild"
    QUERY_OPTIMIZE = "query_optimize"
    CONNECTION_POOL_ADJUST = "connection_pool_adjust"
    ALERT = "alert"

@dataclass
class PerformanceThreshold:
    metric: PerformanceMetric
    warning_threshold: float
    critical_threshold: float
    action: OptimizationAction
    duration_seconds: int = 300  # 5 minutes

@dataclass
class PerformanceAlert:
    timestamp: datetime
    metric: PerformanceMetric
    current_value: float
    threshold: float
    severity: str  # 'warning' or 'critical'
    component: str
    suggested_action: OptimizationAction

class PerformanceOptimizer:
    """
    Automated performance monitoring and optimization system
    """
    
    def __init__(
        self,
        cache_manager=None,
        database_manager=None,
        storage_manager=None,
        config: Dict = None
    ):
        self.cache_manager = cache_manager
        self.database_manager = database_manager
        self.storage_manager = storage_manager
        self.config = config or {}
        
        self.metrics_history = {}
        self.thresholds = {}
        self.alerts = []
        self.optimization_jobs = {}
        self.is_monitoring = False
        
        # Performance counters
        self.request_counter = 0
        self.error_counter = 0
        self.start_time = time.time()
        
    async def initialize(self):
        """Initialize performance monitoring"""
        
        # Load default thresholds
        await self.load_default_thresholds()
        
        # Start monitoring
        await self.start_monitoring()
        
        logger.info("Performance optimizer initialized")
    
    async def load_default_thresholds(self):
        """Load default performance thresholds"""
        
        default_thresholds = [
            # Response time thresholds
            PerformanceThreshold(
                metric=PerformanceMetric.RESPONSE_TIME,
                warning_threshold=1.0,  # 1 second
                critical_threshold=3.0,  # 3 seconds
                action=OptimizationAction.CACHE_WARM,
                duration_seconds=300
            ),
            
            # Error rate thresholds
            PerformanceThreshold(
                metric=PerformanceMetric.ERROR_RATE,
                warning_threshold=0.05,  # 5%
                critical_threshold=0.10,  # 10%
                action=OptimizationAction.ALERT,
                duration_seconds=60
            ),
            
            # Cache hit rate thresholds
            PerformanceThreshold(
                metric=PerformanceMetric.CACHE_HIT_RATE,
                warning_threshold=0.8,  # 80%
                critical_threshold=0.6,  # 60%
                action=OptimizationAction.CACHE_WARM,
                duration_seconds=600
            ),
            
            # Database connection thresholds
            PerformanceThreshold(
                metric=PerformanceMetric.DATABASE_CONNECTIONS,
                warning_threshold=80,  # 80% of pool
                critical_threshold=95,  # 95% of pool
                action=OptimizationAction.CONNECTION_POOL_ADJUST,
                duration_seconds=300
            ),
            
            # Memory usage thresholds
            PerformanceThreshold(
                metric=PerformanceMetric.MEMORY_USAGE,
                warning_threshold=0.8,  # 80%
                critical_threshold=0.9,  # 90%
                action=OptimizationAction.SCALE_UP,
                duration_seconds=600
            ),
            
            # CPU usage thresholds
            PerformanceThreshold(
                metric=PerformanceMetric.CPU_USAGE,
                warning_threshold=0.7,  # 70%
                critical_threshold=0.85,  # 85%
                action=OptimizationAction.SCALE_UP,
                duration_seconds=300
            )
        ]
        
        for threshold in default_thresholds:
            self.add_threshold(threshold)
    
    def add_threshold(self, threshold: PerformanceThreshold):
        """Add performance threshold"""
        
        key = threshold.metric.value
        self.thresholds[key] = threshold
        
        # Initialize metrics history
        if key not in self.metrics_history:
            self.metrics_history[key] = []
        
        logger.info(f"Added threshold: {threshold.metric.value} - {threshold.warning_threshold}/{threshold.critical_threshold}")
    
    async def start_monitoring(self):
        """Start performance monitoring"""
        
        if self.is_monitoring:
            logger.warning("Performance monitoring already running")
            return
        
        self.is_monitoring = True
        
        # Start metric collection
        self.optimization_jobs['metrics'] = asyncio.create_task(
            self.collect_metrics_periodically()
        )
        
        # Start threshold monitoring
        self.optimization_jobs['monitoring'] = asyncio.create_task(
            self.monitor_thresholds_periodically()
        )
        
        # Start optimization execution
        self.optimization_jobs['optimization'] = asyncio.create_task(
            self.execute_optimizations_periodically()
        )
        
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        
        self.is_monitoring = False
        
        for job_name, job in self.optimization_jobs.items():
            job.cancel()
            try:
                await job
            except asyncio.CancelledError:
                pass
        
        self.optimization_jobs.clear()
        logger.info("Performance monitoring stopped")
    
    async def collect_metrics_periodically(self):
        """Periodically collect performance metrics"""
        
        while self.is_monitoring:
            try:
                await asyncio.sleep(30)  # Collect every 30 seconds
                
                metrics = await self.collect_current_metrics()
                
                for metric_name, value in metrics.items():
                    if metric_name in self.metrics_history:
                        self.metrics_history[metric_name].append({
                            'timestamp': datetime.utcnow(),
                            'value': value
                        })
                        
                        # Keep last 1000 measurements
                        if len(self.metrics_history[metric_name]) > 1000:
                            self.metrics_history[metric_name] = self.metrics_history[metric_name][-500:]
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection failed: {e}")
                await asyncio.sleep(60)
    
    async def collect_current_metrics(self) -> Dict[str, float]:
        """Collect current performance metrics"""
        
        metrics = {}
        
        try:
            # Response time and throughput
            current_time = time.time()
            uptime = current_time - self.start_time
            
            if uptime > 0:
                metrics['throughput'] = self.request_counter / uptime
                metrics['error_rate'] = self.error_counter / max(1, self.request_counter)
            
            # Cache metrics
            if self.cache_manager:
                cache_stats = await self.cache_manager.get_stats()
                overall_stats = cache_stats.get('overall', {})
                metrics['cache_hit_rate'] = overall_stats.get('overall_hit_rate', 0.0)
            
            # Database metrics
            if self.database_manager:
                db_health = await self.database_manager.health_check()
                if db_health.get('healthy', False):
                    pools = db_health.get('pools', {})
                    write_pool = pools.get('write_pool', {})
                    
                    if write_pool:
                        size = write_pool.get('size', 0)
                        max_size = write_pool.get('max_size', 1)
                        metrics['database_connections'] = (size / max_size) * 100
            
            # System metrics (placeholder - would integrate with actual monitoring)
            metrics['cpu_usage'] = await self.get_cpu_usage()
            metrics['memory_usage'] = await self.get_memory_usage()
            
            # Storage metrics
            if self.storage_manager:
                storage_stats = await self.storage_manager.get_storage_stats()
                # Extract relevant metrics from storage stats
                
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
        
        return metrics
    
    async def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except ImportError:
            # Fallback: simulate CPU usage
            return 0.3  # 30% placeholder
    
    async def get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.percent / 100.0
        except ImportError:
            # Fallback: simulate memory usage
            return 0.4  # 40% placeholder
    
    async def monitor_thresholds_periodically(self):
        """Periodically check performance thresholds"""
        
        while self.is_monitoring:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                for metric_name, threshold in self.thresholds.items():
                    await self.check_threshold(metric_name, threshold)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Threshold monitoring failed: {e}")
    
    async def check_threshold(
        self,
        metric_name: str,
        threshold: PerformanceThreshold
    ):
        """Check if metric exceeds threshold"""
        
        history = self.metrics_history.get(metric_name, [])
        if not history:
            return
        
        # Get recent measurements within threshold duration
        cutoff_time = datetime.utcnow() - timedelta(seconds=threshold.duration_seconds)
        recent_measurements = [
            measurement for measurement in history
            if measurement['timestamp'] > cutoff_time
        ]
        
        if not recent_measurements:
            return
        
        # Calculate average for threshold period
        values = [m['value'] for m in recent_measurements]
        avg_value = statistics.mean(values)
        
        # Special handling for metrics where lower is worse
        if threshold.metric in [PerformanceMetric.CACHE_HIT_RATE]:
            # For cache hit rate, alert when value is BELOW threshold
            if avg_value < threshold.critical_threshold:
                await self.create_alert(
                    threshold.metric,
                    avg_value,
                    threshold.critical_threshold,
                    'critical',
                    metric_name,
                    threshold.action
                )
            elif avg_value < threshold.warning_threshold:
                await self.create_alert(
                    threshold.metric,
                    avg_value,
                    threshold.warning_threshold,
                    'warning',
                    metric_name,
                    threshold.action
                )
        else:
            # For most metrics, alert when value is ABOVE threshold
            if avg_value > threshold.critical_threshold:
                await self.create_alert(
                    threshold.metric,
                    avg_value,
                    threshold.critical_threshold,
                    'critical',
                    metric_name,
                    threshold.action
                )
            elif avg_value > threshold.warning_threshold:
                await self.create_alert(
                    threshold.metric,
                    avg_value,
                    threshold.warning_threshold,
                    'warning',
                    metric_name,
                    threshold.action
                )
    
    async def create_alert(
        self,
        metric: PerformanceMetric,
        current_value: float,
        threshold: float,
        severity: str,
        component: str,
        suggested_action: OptimizationAction
    ):
        """Create performance alert"""
        
        # Check for duplicate alerts (avoid spam)
        recent_alerts = [
            alert for alert in self.alerts
            if (alert.metric == metric and 
                alert.component == component and
                alert.timestamp > datetime.utcnow() - timedelta(minutes=10))
        ]
        
        if recent_alerts:
            return  # Don't create duplicate alert
        
        alert = PerformanceAlert(
            timestamp=datetime.utcnow(),
            metric=metric,
            current_value=current_value,
            threshold=threshold,
            severity=severity,
            component=component,
            suggested_action=suggested_action
        )
        
        self.alerts.append(alert)
        
        # Limit alert history
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-500:]
        
        # Log alert
        logger.warning(
            f"Performance alert: {metric.value} = {current_value:.3f} "
            f"({'above' if current_value > threshold else 'below'} {threshold:.3f}) "
            f"- {severity} - suggested action: {suggested_action.value}"
        )
        
        # Trigger immediate optimization if critical
        if severity == 'critical':
            await self.execute_optimization_action(suggested_action, component, alert)
    
    async def execute_optimizations_periodically(self):
        """Periodically execute optimization actions"""
        
        while self.is_monitoring:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Execute pending optimizations
                await self.execute_pending_optimizations()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Optimization execution failed: {e}")
    
    async def execute_pending_optimizations(self):
        """Execute pending optimization actions"""
        
        # Get recent critical alerts
        recent_critical_alerts = [
            alert for alert in self.alerts
            if (alert.severity == 'critical' and
                alert.timestamp > datetime.utcnow() - timedelta(minutes=30))
        ]
        
        # Group by action type
        actions_to_execute = {}
        for alert in recent_critical_alerts:
            action = alert.suggested_action
            if action not in actions_to_execute:
                actions_to_execute[action] = []
            actions_to_execute[action].append(alert)
        
        # Execute actions
        for action, alerts in actions_to_execute.items():
            try:
                await self.execute_optimization_action(action, 'system', alerts[0])
            except Exception as e:
                logger.error(f"Failed to execute optimization {action.value}: {e}")
    
    async def execute_optimization_action(
        self,
        action: OptimizationAction,
        component: str,
        alert: PerformanceAlert
    ):
        """Execute specific optimization action"""
        
        try:
            if action == OptimizationAction.CACHE_WARM:
                await self.warm_cache()
                
            elif action == OptimizationAction.CACHE_INVALIDATE:
                await self.invalidate_cache()
                
            elif action == OptimizationAction.CONNECTION_POOL_ADJUST:
                await self.adjust_connection_pool()
                
            elif action == OptimizationAction.INDEX_REBUILD:
                await self.suggest_index_rebuild()
                
            elif action == OptimizationAction.QUERY_OPTIMIZE:
                await self.optimize_queries()
                
            elif action == OptimizationAction.SCALE_UP:
                await self.suggest_scale_up(component)
                
            elif action == OptimizationAction.SCALE_DOWN:
                await self.suggest_scale_down(component)
                
            elif action == OptimizationAction.ALERT:
                await self.send_alert_notification(alert)
            
            logger.info(f"Executed optimization action: {action.value} for {component}")
            
        except Exception as e:
            logger.error(f"Failed to execute optimization {action.value}: {e}")
    
    async def warm_cache(self):
        """Warm cache with frequently accessed data"""
        
        if not self.cache_manager:
            return
        
        try:
            # This would trigger cache warming
            # For now, just log the action
            logger.info("Triggering cache warming optimization")
            
            # If cache warmer is available, trigger it
            if hasattr(self.cache_manager, 'cache_warmer'):
                warmer = self.cache_manager.cache_warmer
                if warmer:
                    await warmer.warm_user_profiles()
                    await warmer.warm_ml_features()
                    
        except Exception as e:
            logger.error(f"Cache warming failed: {e}")
    
    async def invalidate_cache(self):
        """Invalidate stale cache entries"""
        
        if not self.cache_manager:
            return
        
        try:
            # Invalidate potentially stale data
            patterns_to_invalidate = [
                'user_profile:*',
                'ml_features:*'
            ]
            
            for pattern in patterns_to_invalidate:
                await self.cache_manager.invalidate(pattern)
                
            logger.info("Cache invalidation optimization completed")
            
        except Exception as e:
            logger.error(f"Cache invalidation failed: {e}")
    
    async def adjust_connection_pool(self):
        """Adjust database connection pool size"""
        
        try:
            # This would adjust connection pool settings
            # For now, just log the recommendation
            logger.info("Recommending database connection pool adjustment")
            
            # In a real implementation, this would:
            # 1. Analyze current pool utilization
            # 2. Adjust min/max pool sizes
            # 3. Implement gradual scaling
            
        except Exception as e:
            logger.error(f"Connection pool adjustment failed: {e}")
    
    async def suggest_index_rebuild(self):
        """Suggest database index rebuild"""
        
        try:
            # Analyze query performance and suggest index optimizations
            logger.info("Analyzing database performance for index optimization")
            
            # This would:
            # 1. Identify slow queries
            # 2. Analyze index usage
            # 3. Suggest new indexes or rebuild existing ones
            
        except Exception as e:
            logger.error(f"Index analysis failed: {e}")
    
    async def optimize_queries(self):
        """Optimize database queries"""
        
        try:
            # Analyze and optimize slow queries
            logger.info("Analyzing query performance for optimization")
            
            # This would:
            # 1. Identify slow queries from pg_stat_statements
            # 2. Suggest query optimizations
            # 3. Cache frequently used query results
            
        except Exception as e:
            logger.error(f"Query optimization failed: {e}")
    
    async def suggest_scale_up(self, component: str):
        """Suggest scaling up resources"""
        
        logger.warning(f"Recommending scale up for {component}")
        
        # This would:
        # 1. Check current resource utilization
        # 2. Calculate optimal scaling factor
        # 3. Trigger auto-scaling if enabled
        # 4. Send alert to operations team
    
    async def suggest_scale_down(self, component: str):
        """Suggest scaling down resources"""
        
        logger.info(f"Recommending scale down for {component}")
        
        # This would:
        # 1. Verify sustained low utilization
        # 2. Ensure redundancy is maintained
        # 3. Trigger gradual scale down if safe
    
    async def send_alert_notification(self, alert: PerformanceAlert):
        """Send alert notification to operations team"""
        
        try:
            # This would send notifications via:
            # 1. Email
            # 2. Slack/Teams
            # 3. PagerDuty
            # 4. Webhook
            
            logger.critical(
                f"PERFORMANCE ALERT: {alert.metric.value} "
                f"{alert.current_value:.3f} "
                f"({'exceeded' if alert.current_value > alert.threshold else 'below'}) "
                f"threshold {alert.threshold:.3f}"
            )
            
        except Exception as e:
            logger.error(f"Alert notification failed: {e}")
    
    def record_request(self, response_time: float = None, error: bool = False):
        """Record request metrics"""
        
        self.request_counter += 1
        
        if error:
            self.error_counter += 1
        
        if response_time:
            # Add to response time history
            if 'response_time' not in self.metrics_history:
                self.metrics_history['response_time'] = []
            
            self.metrics_history['response_time'].append({
                'timestamp': datetime.utcnow(),
                'value': response_time
            })
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        current_time = datetime.utcnow()
        uptime = time.time() - self.start_time
        
        # Calculate recent statistics
        recent_alerts = [
            alert for alert in self.alerts
            if alert.timestamp > current_time - timedelta(hours=24)
        ]
        
        critical_alerts = [
            alert for alert in recent_alerts
            if alert.severity == 'critical'
        ]
        
        # Metric summaries
        metric_summaries = {}
        for metric_name, history in self.metrics_history.items():
            if history:
                recent_history = [
                    h for h in history
                    if h['timestamp'] > current_time - timedelta(hours=1)
                ]
                
                if recent_history:
                    values = [h['value'] for h in recent_history]
                    metric_summaries[metric_name] = {
                        'current': values[-1] if values else 0,
                        'average': statistics.mean(values),
                        'min': min(values),
                        'max': max(values),
                        'samples': len(values)
                    }
        
        return {
            'timestamp': current_time.isoformat(),
            'uptime_seconds': uptime,
            'requests': {
                'total': self.request_counter,
                'errors': self.error_counter,
                'error_rate': self.error_counter / max(1, self.request_counter),
                'throughput': self.request_counter / max(1, uptime)
            },
            'alerts': {
                'total_24h': len(recent_alerts),
                'critical_24h': len(critical_alerts),
                'recent': [
                    {
                        'timestamp': alert.timestamp.isoformat(),
                        'metric': alert.metric.value,
                        'severity': alert.severity,
                        'value': alert.current_value,
                        'threshold': alert.threshold
                    }
                    for alert in recent_alerts[-10:]  # Last 10 alerts
                ]
            },
            'metrics': metric_summaries,
            'thresholds': {
                metric_name: {
                    'warning': threshold.warning_threshold,
                    'critical': threshold.critical_threshold,
                    'action': threshold.action.value
                }
                for metric_name, threshold in self.thresholds.items()
            },
            'monitoring_active': self.is_monitoring
        }
    
    async def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get current optimization recommendations"""
        
        recommendations = []
        
        # Analyze recent performance data
        current_metrics = await self.collect_current_metrics()
        
        for metric_name, value in current_metrics.items():
            threshold = self.thresholds.get(metric_name)
            if not threshold:
                continue
            
            # Check if optimization is needed
            if threshold.metric in [PerformanceMetric.CACHE_HIT_RATE]:
                # Lower is worse
                if value < threshold.warning_threshold:
                    recommendations.append({
                        'metric': metric_name,
                        'current_value': value,
                        'threshold': threshold.warning_threshold,
                        'severity': 'critical' if value < threshold.critical_threshold else 'warning',
                        'action': threshold.action.value,
                        'description': f"Cache hit rate is low ({value:.1%}). Consider cache warming or increasing cache size."
                    })
            else:
                # Higher is worse
                if value > threshold.warning_threshold:
                    recommendations.append({
                        'metric': metric_name,
                        'current_value': value,
                        'threshold': threshold.warning_threshold,
                        'severity': 'critical' if value > threshold.critical_threshold else 'warning',
                        'action': threshold.action.value,
                        'description': f"{metric_name.replace('_', ' ').title()} is high ({value:.3f}). Consider {threshold.action.value.replace('_', ' ')}."
                    })
        
        return recommendations


# Singleton instance
_performance_optimizer = None

def get_performance_optimizer() -> PerformanceOptimizer:
    """Get singleton performance optimizer instance"""
    global _performance_optimizer
    if _performance_optimizer is None:
        raise RuntimeError("Performance optimizer not initialized")
    return _performance_optimizer

def initialize_performance_optimizer(
    cache_manager=None,
    database_manager=None,
    storage_manager=None,
    config: Dict = None
) -> PerformanceOptimizer:
    """Initialize singleton performance optimizer"""
    global _performance_optimizer
    _performance_optimizer = PerformanceOptimizer(
        cache_manager, database_manager, storage_manager, config
    )
    return _performance_optimizer
