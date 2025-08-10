"""
Adaptive Learning Pipeline
Real-time learning with pattern detection, strategy evolution, and continuous improvement
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
import numpy as np

# ML and statistics imports
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .config_schemas import AISystemConfiguration, LearningConfiguration

logger = logging.getLogger(__name__)

@dataclass
class LearningEvent:
    """Individual learning event structure"""
    event_id: str
    customer_id: str
    interaction_id: str
    timestamp: datetime
    event_type: str  # success, failure, neutral, anomaly
    context: Dict[str, Any]
    outcome_metrics: Dict[str, float]
    confidence_score: float
    learning_value: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class PatternDiscovery:
    """Discovered pattern structure"""
    pattern_id: str
    pattern_type: str
    description: str
    confidence: float
    occurrences: int
    success_rate: float
    customer_segments: List[str]
    conditions: Dict[str, Any]
    discovered_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'discovered_at': self.discovered_at.isoformat()
        }

@dataclass
class LearningResult:
    """Result of learning pipeline processing"""
    learning_id: str
    timestamp: datetime
    events_processed: int
    patterns_discovered: List[PatternDiscovery]
    strategies_evolved: List[Dict[str, Any]]
    performance_insights: Dict[str, Any]
    system_updates: List[Dict[str, Any]]
    confidence_score: float
    significance_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat(),
            'patterns_discovered': [p.to_dict() for p in self.patterns_discovered]
        }

class AnomalyDetector:
    """Detects anomalies in customer behavior and system performance"""
    
    def __init__(self):
        self.baseline_metrics = {}
        self.anomaly_threshold = 2.5  # Standard deviations
        
    def check(self, signals: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for anomalies in signals"""
        anomalies = []
        
        for metric, value in signals.items():
            if isinstance(value, (int, float)):
                anomaly = self._check_statistical_anomaly(metric, value)
                if anomaly:
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _check_statistical_anomaly(self, metric: str, value: float) -> Optional[Dict[str, Any]]:
        """Check for statistical anomaly"""
        if metric not in self.baseline_metrics:
            self.baseline_metrics[metric] = {'values': deque(maxlen=100), 'mean': 0, 'std': 1}
            return None
        
        baseline = self.baseline_metrics[metric]
        baseline['values'].append(value)
        
        if len(baseline['values']) < 10:
            return None
        
        values = list(baseline['values'])
        mean = np.mean(values)
        std = np.std(values)
        
        baseline['mean'] = mean
        baseline['std'] = std
        
        if std > 0:
            z_score = abs((value - mean) / std)
            if z_score > self.anomaly_threshold:
                return {
                    'type': 'statistical_anomaly',
                    'metric': metric,
                    'value': value,
                    'z_score': z_score,
                    'severity': 'high' if z_score > 3 else 'medium'
                }
        
        return None

class StrategyEvolver:
    """Evolves strategies using genetic algorithms and reinforcement learning"""
    
    def __init__(self, config: AISystemConfiguration):
        self.config = config
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.elite_size = 5
        self.generation = 0
        
    async def evolve(
        self,
        performance_data: Dict[str, Any],
        patterns: List[PatternDiscovery]
    ) -> List[Dict[str, Any]]:
        """Evolve new strategies based on performance and patterns"""
        try:
            logger.info("Starting strategy evolution", {
                'generation': self.generation,
                'performance_entries': len(performance_data),
                'patterns': len(patterns)
            })
            
            # Create initial population from existing strategies
            population = self._create_population(performance_data)
            
            # Run evolution cycles
            for generation in range(10):  # 10 generations per evolution
                # Evaluate fitness
                fitness_scores = self._evaluate_fitness(population, patterns)
                
                # Select parents
                parents = self._select_parents(population, fitness_scores)
                
                # Create offspring through crossover
                offspring = self._crossover(parents)
                
                # Apply mutations
                offspring = self._mutate(offspring)
                
                # Select next generation
                population = self._select_next_generation(
                    population + offspring,
                    fitness_scores
                )
            
            # Extract best strategies
            best_strategies = self._extract_best_strategies(population)
            
            # Validate strategies
            validated = await self._validate_strategies(best_strategies)
            
            self.generation += 1
            
            logger.info("Strategy evolution completed", {
                'new_strategies': len(validated),
                'generation': self.generation
            })
            
            return validated
            
        except Exception as e:
            logger.error(f"Strategy evolution failed: {e}")
            return []
    
    def _create_population(self, performance_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create initial population of strategies"""
        population = []
        
        # Include top performing strategies
        top_strategies = sorted(
            performance_data.items(),
            key=lambda x: x[1].get('success_rate', 0),
            reverse=True
        )[:self.elite_size]
        
        for strategy_name, metrics in top_strategies:
            population.append({
                'genes': self._encode_strategy(strategy_name, metrics),
                'fitness': metrics.get('success_rate', 0),
                'origin': 'elite',
                'strategy_name': strategy_name
            })
        
        # Add variations of successful strategies
        while len(population) < self.population_size // 2:
            base = np.random.choice(population)
            variant = self._create_variant(base)
            population.append(variant)
        
        # Add random strategies for diversity
        while len(population) < self.population_size:
            population.append({
                'genes': self._random_genes(),
                'fitness': 0,
                'origin': 'random',
                'strategy_name': f'random_{len(population)}'
            })
        
        return population
    
    def _encode_strategy(self, strategy_name: str, metrics: Dict[str, Any]) -> List[float]:
        """Encode strategy into genetic representation"""
        # This is a simplified encoding - in production this would be more sophisticated
        return [
            metrics.get('success_rate', 0.5),
            metrics.get('engagement_rate', 0.5),
            metrics.get('conversion_rate', 0.5),
            metrics.get('response_time', 0.5),
            metrics.get('personalization_level', 0.5),
            np.random.random(),  # Innovation factor
            np.random.random(),  # Risk factor
            np.random.random()   # Complexity factor
        ]
    
    def _random_genes(self) -> List[float]:
        """Generate random genetic material"""
        return [np.random.random() for _ in range(8)]
    
    def _evaluate_fitness(self, population: List[Dict], patterns: List[PatternDiscovery]) -> List[float]:
        """Evaluate fitness of each strategy"""
        fitness_scores = []
        
        for individual in population:
            genes = individual['genes']
            
            # Base fitness from performance
            base_fitness = individual.get('fitness', 0)
            
            # Pattern alignment bonus
            pattern_bonus = self._calculate_pattern_alignment(genes, patterns)
            
            # Diversity bonus
            diversity_bonus = self._calculate_diversity_bonus(genes, population)
            
            # Innovation penalty/bonus based on risk tolerance
            innovation_factor = genes[5] if len(genes) > 5 else 0.5
            innovation_bonus = self._calculate_innovation_bonus(innovation_factor)
            
            total_fitness = (
                base_fitness * 0.6 +
                pattern_bonus * 0.2 +
                diversity_bonus * 0.1 +
                innovation_bonus * 0.1
            )
            
            fitness_scores.append(max(0, min(1, total_fitness)))
        
        return fitness_scores
    
    def _calculate_pattern_alignment(self, genes: List[float], patterns: List[PatternDiscovery]) -> float:
        """Calculate how well strategy aligns with discovered patterns"""
        if not patterns:
            return 0.5
        
        alignment_score = 0.0
        for pattern in patterns:
            if pattern.success_rate > 0.7:  # Only consider successful patterns
                # Simple alignment calculation - in production this would be more sophisticated
                pattern_genes = [pattern.success_rate, pattern.confidence, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                similarity = 1 - np.mean([abs(g1 - g2) for g1, g2 in zip(genes, pattern_genes)])
                alignment_score += similarity * pattern.confidence
        
        return alignment_score / len(patterns) if patterns else 0.5

class AdaptiveLearningPipeline:
    """
    Real-time learning pipeline that adapts strategies based on outcomes
    """
    
    def __init__(self, config: AISystemConfiguration):
        self.config = config
        self.learning_config = config.learning_config
        
        # Initialize components
        self.experience_buffer = deque(maxlen=10000)
        self.strategy_performance = defaultdict(lambda: {
            'success_count': 0,
            'total_count': 0,
            'success_rate': 0.0,
            'confidence_scores': [],
            'last_updated': datetime.utcnow()
        })
        self.pattern_library = {}
        self.anomaly_detector = AnomalyDetector()
        self.strategy_evolver = StrategyEvolver(config)
        
        # Learning state
        self.learning_cycles = 0
        self.last_evolution = datetime.utcnow()
        self.performance_baseline = {}
        
        logger.info("Adaptive Learning Pipeline initialized")
    
    async def process_interaction(self, interaction: Dict[str, Any]) -> LearningResult:
        """Process interaction for learning opportunities"""
        learning_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        try:
            logger.info("Processing interaction for learning", {
                'learning_id': learning_id,
                'customer_id': interaction.get('customer_id'),
                'interaction_id': interaction.get('interaction_id')
            })
            
            # Extract learning signals
            signals = self._extract_signals(interaction)
            
            # Create learning event
            learning_event = LearningEvent(
                event_id=str(uuid.uuid4()),
                customer_id=interaction.get('customer_id', ''),
                interaction_id=interaction.get('interaction_id', ''),
                timestamp=start_time,
                event_type=self._classify_event_type(signals),
                context=interaction.get('context', {}),
                outcome_metrics=signals.get('outcome_metrics', {}),
                confidence_score=signals.get('confidence', 0.5),
                learning_value=self._estimate_learning_value(signals)
            )
            
            # Update performance metrics
            self._update_performance(learning_event, interaction)
            
            # Detect patterns
            patterns = await self._detect_patterns(learning_event, signals)
            
            # Check for anomalies
            anomalies = self.anomaly_detector.check(signals)
            
            # Determine if evolution is needed
            evolution_needed = self._should_evolve(learning_event, anomalies)
            
            # Evolve strategies if needed
            evolved_strategies = []
            if evolution_needed:
                evolved_strategies = await self.strategy_evolver.evolve(
                    dict(self.strategy_performance),
                    patterns
                )
                self.last_evolution = start_time
            
            # Generate performance insights
            performance_insights = self._generate_performance_insights()
            
            # Determine system updates
            system_updates = self._determine_system_updates(
                learning_event,
                patterns,
                evolved_strategies,
                anomalies
            )
            
            # Calculate significance
            significance_score = self._calculate_significance(
                learning_event,
                patterns,
                evolved_strategies,
                anomalies
            )
            
            # Store experience for future learning
            self.experience_buffer.append(learning_event)
            
            # Create learning result
            result = LearningResult(
                learning_id=learning_id,
                timestamp=start_time,
                events_processed=1,
                patterns_discovered=patterns,
                strategies_evolved=evolved_strategies,
                performance_insights=performance_insights,
                system_updates=system_updates,
                confidence_score=learning_event.confidence_score,
                significance_score=significance_score
            )
            
            self.learning_cycles += 1
            
            logger.info("Learning processing completed", {
                'learning_id': learning_id,
                'patterns_found': len(patterns),
                'strategies_evolved': len(evolved_strategies),
                'significance': significance_score,
                'processing_time_ms': (datetime.utcnow() - start_time).total_seconds() * 1000
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Learning processing failed: {e}")
            return self._create_error_result(learning_id, str(e))
    
    def _extract_signals(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Extract learning signals from interaction"""
        try:
            signals = {
                # Timing signals
                'timing_signal': self._analyze_timing(interaction),
                
                # Content signals
                'content_signal': self._analyze_content(interaction),
                
                # Channel signals
                'channel_signal': self._analyze_channel(interaction),
                
                # Response signals
                'response_signal': self._analyze_response(interaction),
                
                # Context signals
                'context_signal': self._analyze_context(interaction),
                
                # Outcome signals
                'outcome_signal': self._analyze_outcome(interaction)
            }
            
            # Add derived signals
            signals['composite_score'] = self._calculate_composite_score(signals)
            signals['surprise_factor'] = self._calculate_surprise(interaction, signals)
            signals['learning_value'] = self._estimate_learning_value(signals)
            signals['confidence'] = interaction.get('ai_response', {}).get('confidence', 0.5)
            
            # Extract outcome metrics
            signals['outcome_metrics'] = self._extract_outcome_metrics(interaction)
            
            return signals
            
        except Exception as e:
            logger.error(f"Signal extraction failed: {e}")
            return {'error': str(e), 'confidence': 0.0}
    
    def _analyze_timing(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze timing-related signals"""
        try:
            delivery = interaction.get('delivery', {})
            response_time = delivery.get('metrics', {}).get('delivery_time_ms', 0)
            
            return {
                'response_time_ms': response_time,
                'timing_optimal': response_time < 1000,
                'timing_score': max(0, 1 - (response_time / 5000))  # Score based on 5s max
            }
        except Exception:
            return {'response_time_ms': 0, 'timing_optimal': False, 'timing_score': 0.5}
    
    def _analyze_content(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze content-related signals"""
        try:
            content = interaction.get('content', {})
            ai_response = interaction.get('ai_response', {})
            
            return {
                'content_confidence': ai_response.get('confidence', 0.5),
                'personalization_level': content.get('personalization_level', 1),
                'content_length': len(content.get('content', {}).get('body', '')),
                'has_cta': bool(content.get('content', {}).get('cta_text')),
                'tone_match': True  # Would be calculated based on customer preferences
            }
        except Exception:
            return {'content_confidence': 0.5, 'personalization_level': 1, 'content_length': 0}
    
    def _analyze_channel(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze channel-related signals"""
        try:
            delivery = interaction.get('delivery', {})
            channels = delivery.get('channels_used', {})
            
            return {
                'primary_channel': channels.get('primary', 'email'),
                'channel_success': delivery.get('status') == 'success',
                'delivery_confirmed': delivery.get('metrics', {}).get('delivered', False)
            }
        except Exception:
            return {'primary_channel': 'email', 'channel_success': False, 'delivery_confirmed': False}
    
    def _analyze_response(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze response-related signals"""
        try:
            feedback = interaction.get('feedback', {})
            
            return {
                'response_received': bool(feedback.get('response_data')),
                'response_positive': feedback.get('sentiment', 'neutral') == 'positive',
                'engagement_score': feedback.get('engagement_score', 0.5)
            }
        except Exception:
            return {'response_received': False, 'response_positive': False, 'engagement_score': 0.5}
    
    def _analyze_context(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze contextual signals"""
        try:
            context = interaction.get('context', {})
            
            return {
                'lifecycle_stage': context.get('user_profile', {}).get('lifecycle_stage', 'unknown'),
                'churn_risk': context.get('behavioral_metrics', {}).get('churn_risk', 0.5),
                'engagement_level': context.get('behavioral_metrics', {}).get('engagement_score', 0.5)
            }
        except Exception:
            return {'lifecycle_stage': 'unknown', 'churn_risk': 0.5, 'engagement_level': 0.5}
    
    def _analyze_outcome(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze outcome-related signals"""
        try:
            delivery = interaction.get('delivery', {})
            feedback = interaction.get('feedback', {})
            
            success_indicators = {
                'delivered': delivery.get('status') == 'success',
                'opened': feedback.get('opened', False),
                'clicked': feedback.get('clicked', False),
                'responded': feedback.get('responded', False),
                'converted': feedback.get('converted', False)
            }
            
            success_score = sum(success_indicators.values()) / len(success_indicators)
            
            return {
                'success_indicators': success_indicators,
                'success_score': success_score,
                'objective_achieved': success_score > 0.5
            }
        except Exception:
            return {'success_indicators': {}, 'success_score': 0.0, 'objective_achieved': False}
    
    def _calculate_composite_score(self, signals: Dict[str, Any]) -> float:
        """Calculate composite success score from all signals"""
        try:
            scores = []
            
            # Timing score
            timing = signals.get('timing_signal', {})
            scores.append(timing.get('timing_score', 0.5))
            
            # Content score
            content = signals.get('content_signal', {})
            scores.append(content.get('content_confidence', 0.5))
            
            # Response score
            response = signals.get('response_signal', {})
            scores.append(response.get('engagement_score', 0.5))
            
            # Outcome score
            outcome = signals.get('outcome_signal', {})
            scores.append(outcome.get('success_score', 0.5))
            
            return np.mean(scores) if scores else 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_surprise(self, interaction: Dict[str, Any], signals: Dict[str, Any]) -> float:
        """Calculate surprise factor for unexpected outcomes"""
        try:
            expected_success = interaction.get('ai_response', {}).get('confidence', 0.5)
            actual_success = signals.get('composite_score', 0.5)
            
            surprise = abs(expected_success - actual_success)
            return min(surprise * 2, 1.0)  # Normalize to 0-1
            
        except Exception:
            return 0.0
    
    def _estimate_learning_value(self, signals: Dict[str, Any]) -> float:
        """Estimate the learning value of this interaction"""
        try:
            # High learning value for surprising outcomes
            surprise_factor = signals.get('surprise_factor', 0.0)
            
            # High learning value for extreme successes or failures
            composite_score = signals.get('composite_score', 0.5)
            extreme_factor = abs(composite_score - 0.5) * 2
            
            # High learning value for rare contexts
            context_rarity = 0.5  # Would be calculated based on historical data
            
            learning_value = (surprise_factor * 0.4 + extreme_factor * 0.4 + context_rarity * 0.2)
            return min(learning_value, 1.0)
            
        except Exception:
            return 0.5
    
    def _classify_event_type(self, signals: Dict[str, Any]) -> str:
        """Classify the type of learning event"""
        composite_score = signals.get('composite_score', 0.5)
        surprise_factor = signals.get('surprise_factor', 0.0)
        
        if surprise_factor > 0.7:
            return 'anomaly'
        elif composite_score > 0.8:
            return 'success'
        elif composite_score < 0.3:
            return 'failure'
        else:
            return 'neutral'
    
    def _extract_outcome_metrics(self, interaction: Dict[str, Any]) -> Dict[str, float]:
        """Extract specific outcome metrics"""
        try:
            feedback = interaction.get('feedback', {})
            delivery = interaction.get('delivery', {})
            
            return {
                'delivery_success_rate': 1.0 if delivery.get('status') == 'success' else 0.0,
                'open_rate': 1.0 if feedback.get('opened') else 0.0,
                'click_rate': 1.0 if feedback.get('clicked') else 0.0,
                'response_rate': 1.0 if feedback.get('responded') else 0.0,
                'conversion_rate': 1.0 if feedback.get('converted') else 0.0,
                'engagement_score': feedback.get('engagement_score', 0.5),
                'satisfaction_score': feedback.get('satisfaction_score', 0.5)
            }
        except Exception:
            return {}
    
    async def _detect_patterns(self, event: LearningEvent, signals: Dict[str, Any]) -> List[PatternDiscovery]:
        """Detect patterns in the learning events"""
        patterns = []
        
        try:
            # Check against known patterns
            for pattern_id, pattern_def in self.pattern_library.items():
                if self._matches_pattern(event, signals, pattern_def):
                    # Update pattern statistics
                    pattern_def['occurrences'] += 1
                    
                    # Create pattern discovery object
                    pattern = PatternDiscovery(
                        pattern_id=pattern_id,
                        pattern_type=pattern_def['type'],
                        description=pattern_def['description'],
                        confidence=self._calculate_pattern_confidence(pattern_def),
                        occurrences=pattern_def['occurrences'],
                        success_rate=pattern_def.get('success_rate', 0.5),
                        customer_segments=pattern_def.get('segments', []),
                        conditions=pattern_def.get('conditions', {}),
                        discovered_at=datetime.utcnow()
                    )
                    patterns.append(pattern)
            
            # Discover new patterns using clustering if we have enough data
            if SKLEARN_AVAILABLE and len(self.experience_buffer) > 50:
                new_patterns = await self._discover_new_patterns()
                patterns.extend(new_patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            return []
    
    def _matches_pattern(self, event: LearningEvent, signals: Dict[str, Any], pattern_def: Dict) -> bool:
        """Check if event matches a known pattern"""
        try:
            conditions = pattern_def.get('conditions', {})
            
            for condition_key, condition_value in conditions.items():
                if condition_key in signals:
                    signal_value = signals[condition_key]
                    if isinstance(condition_value, dict):
                        # Range condition
                        if 'min' in condition_value and signal_value < condition_value['min']:
                            return False
                        if 'max' in condition_value and signal_value > condition_value['max']:
                            return False
                    else:
                        # Exact match condition
                        if signal_value != condition_value:
                            return False
            
            return True
            
        except Exception:
            return False
    
    def _calculate_pattern_confidence(self, pattern_def: Dict) -> float:
        """Calculate confidence in pattern"""
        occurrences = pattern_def.get('occurrences', 0)
        success_rate = pattern_def.get('success_rate', 0.5)
        
        # Confidence increases with occurrences and success rate
        confidence = min(0.95, (occurrences / 100) * success_rate)
        return max(0.1, confidence)
    
    # Additional methods would continue here...
    # (Including pattern discovery, strategy evolution, performance tracking, etc.)
