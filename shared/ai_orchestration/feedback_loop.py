"""
Intelligent Feedback Loop with Self-Correction
Processes feedback and implements self-correction mechanisms
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

from .config_schemas import AISystemConfiguration, FeedbackConfiguration

logger = logging.getLogger(__name__)

@dataclass
class FeedbackEvent:
    """Structured feedback event"""
    feedback_id: str
    customer_id: str
    interaction_id: str
    timestamp: datetime
    feedback_type: str  # explicit, implicit, behavioral, outcome
    content: Dict[str, Any]
    confidence: float
    source: str
    priority: str  # low, normal, high, critical
    learning_weight: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class CorrectionAction:
    """Self-correction action"""
    correction_id: str
    correction_type: str  # immediate, strategic, systemic
    target_component: str
    action: str
    parameters: Dict[str, Any]
    urgency: str
    expected_impact: float
    confidence: float
    rollback_enabled: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class FeedbackProcessingResult:
    """Result of feedback processing"""
    processing_id: str
    feedback_event: FeedbackEvent
    insights_extracted: List[Dict[str, Any]]
    corrections_generated: List[CorrectionAction]
    learning_updates: List[Dict[str, Any]]
    confidence_score: float
    processing_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'feedback_event': self.feedback_event.to_dict(),
            'corrections_generated': [c.to_dict() for c in self.corrections_generated]
        }

class FeedbackClassifier:
    """Classifies and prioritizes feedback"""
    
    def __init__(self):
        self.classification_rules = {
            'explicit_positive': {
                'indicators': ['satisfied', 'helpful', 'good', 'excellent', 'love'],
                'weight': 1.0,
                'priority': 'normal'
            },
            'explicit_negative': {
                'indicators': ['unsatisfied', 'unhelpful', 'bad', 'terrible', 'hate'],
                'weight': 1.5,
                'priority': 'high'
            },
            'implicit_engagement': {
                'indicators': ['opened', 'clicked', 'viewed', 'downloaded'],
                'weight': 0.7,
                'priority': 'normal'
            },
            'implicit_disengagement': {
                'indicators': ['unsubscribed', 'blocked', 'deleted', 'ignored'],
                'weight': 1.2,
                'priority': 'high'
            },
            'behavioral_positive': {
                'indicators': ['increased_usage', 'feature_adoption', 'upgrade'],
                'weight': 0.9,
                'priority': 'normal'
            },
            'behavioral_negative': {
                'indicators': ['decreased_usage', 'feature_abandonment', 'downgrade'],
                'weight': 1.3,
                'priority': 'high'
            },
            'outcome_success': {
                'indicators': ['conversion', 'retention', 'referral', 'expansion'],
                'weight': 1.5,
                'priority': 'normal'
            },
            'outcome_failure': {
                'indicators': ['churn', 'cancellation', 'complaint', 'refund'],
                'weight': 2.0,
                'priority': 'critical'
            }
        }
    
    def classify_feedback(self, feedback_content: Dict[str, Any]) -> Dict[str, Any]:
        """Classify feedback and determine priority"""
        try:
            classification_scores = {}
            
            # Convert content to searchable text
            searchable_text = self._extract_searchable_text(feedback_content)
            
            # Score against each classification
            for classification, rules in self.classification_rules.items():
                score = self._calculate_classification_score(searchable_text, rules['indicators'])
                if score > 0:
                    classification_scores[classification] = {
                        'score': score,
                        'weight': rules['weight'],
                        'priority': rules['priority']
                    }
            
            if not classification_scores:
                return {
                    'primary_classification': 'unclassified',
                    'confidence': 0.1,
                    'priority': 'low',
                    'weight': 0.5
                }
            
            # Select primary classification
            primary = max(classification_scores.items(), key=lambda x: x[1]['score'])
            
            return {
                'primary_classification': primary[0],
                'confidence': min(0.95, primary[1]['score']),
                'priority': primary[1]['priority'],
                'weight': primary[1]['weight'],
                'all_classifications': classification_scores
            }
            
        except Exception as e:
            logger.error(f"Feedback classification failed: {e}")
            return {
                'primary_classification': 'error',
                'confidence': 0.0,
                'priority': 'low',
                'weight': 0.1
            }
    
    def _extract_searchable_text(self, content: Dict[str, Any]) -> str:
        """Extract searchable text from feedback content"""
        text_parts = []
        
        for key, value in content.items():
            if isinstance(value, str):
                text_parts.append(value.lower())
            elif isinstance(value, bool):
                text_parts.append(str(value).lower())
            elif isinstance(value, (int, float)):
                text_parts.append(str(value))
        
        return ' '.join(text_parts)
    
    def _calculate_classification_score(self, text: str, indicators: List[str]) -> float:
        """Calculate classification score based on indicators"""
        matches = sum(1 for indicator in indicators if indicator.lower() in text)
        return matches / len(indicators) if indicators else 0.0

class InsightExtractor:
    """Extracts actionable insights from feedback"""
    
    def __init__(self):
        self.insight_patterns = {
            'content_issues': {
                'patterns': ['confusing', 'unclear', 'too long', 'too short', 'irrelevant'],
                'insight_type': 'content_optimization'
            },
            'timing_issues': {
                'patterns': ['too early', 'too late', 'wrong time', 'bad timing'],
                'insight_type': 'timing_optimization'
            },
            'channel_issues': {
                'patterns': ['wrong channel', 'prefer email', 'prefer sms', 'too many notifications'],
                'insight_type': 'channel_optimization'
            },
            'personalization_issues': {
                'patterns': ['not relevant', 'generic', 'doesn\'t apply', 'not for me'],
                'insight_type': 'personalization_improvement'
            },
            'frequency_issues': {
                'patterns': ['too frequent', 'too often', 'spam', 'bombarding'],
                'insight_type': 'frequency_adjustment'
            }
        }
    
    def extract_insights(
        self, 
        feedback: FeedbackEvent, 
        classification: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract actionable insights from feedback"""
        insights = []
        
        try:
            feedback_text = self._get_feedback_text(feedback.content)
            
            # Extract pattern-based insights
            for issue_type, config in self.insight_patterns.items():
                if self._matches_patterns(feedback_text, config['patterns']):
                    insights.append({
                        'insight_type': config['insight_type'],
                        'issue_category': issue_type,
                        'confidence': self._calculate_pattern_confidence(
                            feedback_text, 
                            config['patterns']
                        ),
                        'feedback_source': feedback.source,
                        'customer_id': feedback.customer_id,
                        'priority': classification.get('priority', 'normal'),
                        'actionable': True
                    })
            
            # Extract metric-based insights
            metric_insights = self._extract_metric_insights(feedback)
            insights.extend(metric_insights)
            
            # Extract context-specific insights
            context_insights = self._extract_context_insights(feedback, classification)
            insights.extend(context_insights)
            
            return insights
            
        except Exception as e:
            logger.error(f"Insight extraction failed: {e}")
            return []
    
    def _get_feedback_text(self, content: Dict[str, Any]) -> str:
        """Extract text content from feedback"""
        text_fields = ['message', 'comment', 'feedback', 'review', 'response']
        
        for field in text_fields:
            if field in content and isinstance(content[field], str):
                return content[field].lower()
        
        # Fallback to any string values
        for value in content.values():
            if isinstance(value, str) and len(value) > 10:
                return value.lower()
        
        return ''
    
    def _matches_patterns(self, text: str, patterns: List[str]) -> bool:
        """Check if text matches any of the patterns"""
        return any(pattern.lower() in text for pattern in patterns)
    
    def _calculate_pattern_confidence(self, text: str, patterns: List[str]) -> float:
        """Calculate confidence in pattern match"""
        matches = sum(1 for pattern in patterns if pattern.lower() in text)
        return min(0.9, matches / len(patterns) * 2) if patterns else 0.0
    
    def _extract_metric_insights(self, feedback: FeedbackEvent) -> List[Dict[str, Any]]:
        """Extract insights from numerical metrics in feedback"""
        insights = []
        
        for key, value in feedback.content.items():
            if isinstance(value, (int, float)) and key.endswith(('_score', '_rating', '_rate')):
                if value < 0.3:  # Low score
                    insights.append({
                        'insight_type': 'performance_issue',
                        'metric': key,
                        'value': value,
                        'severity': 'high' if value < 0.2 else 'medium',
                        'confidence': 0.8,
                        'actionable': True
                    })
                elif value > 0.8:  # High score
                    insights.append({
                        'insight_type': 'performance_success',
                        'metric': key,
                        'value': value,
                        'confidence': 0.7,
                        'actionable': False  # Success doesn't need immediate action
                    })
        
        return insights
    
    def _extract_context_insights(
        self, 
        feedback: FeedbackEvent, 
        classification: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract insights based on context"""
        insights = []
        
        # Time-based insights
        hour = feedback.timestamp.hour
        if hour < 6 or hour > 22:
            insights.append({
                'insight_type': 'timing_optimization',
                'issue': 'outside_business_hours',
                'time': hour,
                'confidence': 0.6,
                'actionable': True
            })
        
        # Feedback type insights
        if feedback.feedback_type == 'explicit_negative' and classification.get('priority') == 'critical':
            insights.append({
                'insight_type': 'escalation_required',
                'reason': 'critical_negative_feedback',
                'confidence': 0.9,
                'actionable': True,
                'urgent': True
            })
        
        return insights

class SelfCorrectionEngine:
    """Implements self-correction based on feedback analysis"""
    
    def __init__(self, config: FeedbackConfiguration):
        self.config = config
        self.correction_history = deque(maxlen=1000)
        self.active_corrections = {}
        self.rollback_stack = deque(maxlen=100)
    
    async def generate_corrections(
        self, 
        feedback: FeedbackEvent, 
        insights: List[Dict[str, Any]]
    ) -> List[CorrectionAction]:
        """Generate self-corrections based on feedback and insights"""
        corrections = []
        
        try:
            for insight in insights:
                if not insight.get('actionable', False):
                    continue
                
                insight_type = insight.get('insight_type')
                
                if insight_type == 'content_optimization':
                    corrections.extend(await self._generate_content_corrections(insight, feedback))
                
                elif insight_type == 'timing_optimization':
                    corrections.extend(await self._generate_timing_corrections(insight, feedback))
                
                elif insight_type == 'channel_optimization':
                    corrections.extend(await self._generate_channel_corrections(insight, feedback))
                
                elif insight_type == 'personalization_improvement':
                    corrections.extend(await self._generate_personalization_corrections(insight, feedback))
                
                elif insight_type == 'frequency_adjustment':
                    corrections.extend(await self._generate_frequency_corrections(insight, feedback))
                
                elif insight_type == 'performance_issue':
                    corrections.extend(await self._generate_performance_corrections(insight, feedback))
                
                elif insight_type == 'escalation_required':
                    corrections.extend(await self._generate_escalation_corrections(insight, feedback))
            
            # Rank corrections by priority and confidence
            ranked_corrections = self._rank_corrections(corrections)
            
            return ranked_corrections
            
        except Exception as e:
            logger.error(f"Correction generation failed: {e}")
            return []
    
    async def _generate_content_corrections(
        self, 
        insight: Dict[str, Any], 
        feedback: FeedbackEvent
    ) -> List[CorrectionAction]:
        """Generate content-related corrections"""
        corrections = []
        
        issue_category = insight.get('issue_category', '')
        
        if 'confusing' in issue_category or 'unclear' in issue_category:
            corrections.append(CorrectionAction(
                correction_id=str(uuid.uuid4()),
                correction_type='immediate',
                target_component='content_generator',
                action='simplify_messaging',
                parameters={
                    'customer_id': feedback.customer_id,
                    'simplification_level': 'high',
                    'clarity_focus': True
                },
                urgency='high',
                expected_impact=0.3,
                confidence=insight.get('confidence', 0.5),
                rollback_enabled=True
            ))
        
        if 'too long' in issue_category:
            corrections.append(CorrectionAction(
                correction_id=str(uuid.uuid4()),
                correction_type='immediate',
                target_component='content_generator',
                action='reduce_content_length',
                parameters={
                    'customer_id': feedback.customer_id,
                    'target_reduction': 0.3,
                    'preserve_key_points': True
                },
                urgency='medium',
                expected_impact=0.2,
                confidence=insight.get('confidence', 0.5),
                rollback_enabled=True
            ))
        
        return corrections
    
    async def _generate_timing_corrections(
        self, 
        insight: Dict[str, Any], 
        feedback: FeedbackEvent
    ) -> List[CorrectionAction]:
        """Generate timing-related corrections"""
        corrections = []
        
        if insight.get('issue') == 'outside_business_hours':
            corrections.append(CorrectionAction(
                correction_id=str(uuid.uuid4()),
                correction_type='strategic',
                target_component='timing_optimizer',
                action='restrict_to_business_hours',
                parameters={
                    'customer_id': feedback.customer_id,
                    'business_hours': {'start': 9, 'end': 17},
                    'timezone_aware': True
                },
                urgency='medium',
                expected_impact=0.25,
                confidence=insight.get('confidence', 0.6),
                rollback_enabled=True
            ))
        
        return corrections
    
    async def _generate_escalation_corrections(
        self, 
        insight: Dict[str, Any], 
        feedback: FeedbackEvent
    ) -> List[CorrectionAction]:
        """Generate escalation corrections for critical issues"""
        corrections = []
        
        if insight.get('urgent', False):
            corrections.append(CorrectionAction(
                correction_id=str(uuid.uuid4()),
                correction_type='immediate',
                target_component='escalation_manager',
                action='escalate_to_human',
                parameters={
                    'customer_id': feedback.customer_id,
                    'escalation_reason': insight.get('reason', 'critical_feedback'),
                    'priority': 'critical',
                    'notify_manager': True
                },
                urgency='critical',
                expected_impact=0.8,
                confidence=0.9,
                rollback_enabled=False
            ))
        
        return corrections
    
    def _rank_corrections(self, corrections: List[CorrectionAction]) -> List[CorrectionAction]:
        """Rank corrections by priority"""
        urgency_scores = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        
        def correction_score(correction):
            urgency = urgency_scores.get(correction.urgency, 1)
            confidence = correction.confidence
            impact = correction.expected_impact
            
            return urgency * 0.4 + confidence * 0.3 + impact * 0.3
        
        return sorted(corrections, key=correction_score, reverse=True)
    
    async def apply_corrections(self, corrections: List[CorrectionAction]) -> Dict[str, Any]:
        """Apply corrections and track results"""
        application_results = {
            'applied': [],
            'failed': [],
            'skipped': []
        }
        
        for correction in corrections:
            try:
                # Check if correction should be applied
                if not self._should_apply_correction(correction):
                    application_results['skipped'].append({
                        'correction_id': correction.correction_id,
                        'reason': 'safety_check_failed'
                    })
                    continue
                
                # Apply the correction
                result = await self._execute_correction(correction)
                
                if result['success']:
                    application_results['applied'].append(result)
                    
                    # Add to rollback stack if enabled
                    if correction.rollback_enabled:
                        self.rollback_stack.append({
                            'correction': correction,
                            'applied_at': datetime.utcnow(),
                            'rollback_data': result.get('rollback_data')
                        })
                else:
                    application_results['failed'].append(result)
                
            except Exception as e:
                logger.error(f"Failed to apply correction {correction.correction_id}: {e}")
                application_results['failed'].append({
                    'correction_id': correction.correction_id,
                    'error': str(e)
                })
        
        return application_results
    
    def _should_apply_correction(self, correction: CorrectionAction) -> bool:
        """Safety check before applying correction"""
        # Check confidence threshold
        if correction.confidence < self.config.correction_threshold:
            return False
        
        # Check if similar correction was recently applied
        recent_corrections = [
            c for c in self.correction_history 
            if c.target_component == correction.target_component and
            (datetime.utcnow() - c.applied_at).total_seconds() < 3600  # 1 hour
        ]
        
        if len(recent_corrections) > 3:  # Too many recent corrections
            return False
        
        return True
    
    async def _execute_correction(self, correction: CorrectionAction) -> Dict[str, Any]:
        """Execute a specific correction"""
        try:
            # This would integrate with the actual services to apply corrections
            # For now, we'll simulate the execution
            
            logger.info(f"Executing correction: {correction.action}", {
                'correction_id': correction.correction_id,
                'target_component': correction.target_component,
                'urgency': correction.urgency
            })
            
            # Simulate execution result
            result = {
                'success': True,
                'correction_id': correction.correction_id,
                'applied_at': datetime.utcnow().isoformat(),
                'target_component': correction.target_component,
                'action': correction.action,
                'rollback_data': {
                    'previous_state': 'simulated_state',
                    'rollback_params': correction.parameters.copy()
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Correction execution failed: {e}")
            return {
                'success': False,
                'correction_id': correction.correction_id,
                'error': str(e)
            }

class IntelligentFeedbackLoop:
    """
    Main feedback processing system with intelligent analysis and self-correction
    """
    
    def __init__(self, config: AISystemConfiguration):
        self.config = config
        self.feedback_config = FeedbackConfiguration()
        
        # Initialize components
        self.feedback_classifier = FeedbackClassifier()
        self.insight_extractor = InsightExtractor()
        self.correction_engine = SelfCorrectionEngine(self.feedback_config)
        
        # Processing queues
        self.feedback_queue = asyncio.Queue()
        self.processing_history = deque(maxlen=1000)
        
        # State tracking
        self.total_processed = 0
        self.correction_success_rate = 0.0
        
        logger.info("Intelligent Feedback Loop initialized")
    
    async def process_feedback(self, feedback_data: Dict[str, Any]) -> FeedbackProcessingResult:
        """
        Process feedback and generate corrections
        """
        processing_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        try:
            # Create feedback event
            feedback_event = FeedbackEvent(
                feedback_id=feedback_data.get('feedback_id', str(uuid.uuid4())),
                customer_id=feedback_data.get('customer_id', ''),
                interaction_id=feedback_data.get('interaction_id', ''),
                timestamp=datetime.utcnow(),
                feedback_type=feedback_data.get('feedback_type', 'implicit'),
                content=feedback_data.get('content', {}),
                confidence=feedback_data.get('confidence', 0.5),
                source=feedback_data.get('source', 'unknown'),
                priority=feedback_data.get('priority', 'normal'),
                learning_weight=feedback_data.get('learning_weight', 1.0)
            )
            
            logger.info(f"Processing feedback event {feedback_event.feedback_id}", {
                'processing_id': processing_id,
                'customer_id': feedback_event.customer_id,
                'feedback_type': feedback_event.feedback_type
            })
            
            # Classify feedback
            classification = self.feedback_classifier.classify_feedback(feedback_event.content)
            
            # Extract insights
            insights = self.insight_extractor.extract_insights(feedback_event, classification)
            
            # Generate corrections
            corrections = await self.correction_engine.generate_corrections(feedback_event, insights)
            
            # Generate learning updates
            learning_updates = self._generate_learning_updates(feedback_event, insights, classification)
            
            # Calculate confidence
            confidence_score = self._calculate_processing_confidence(
                classification, 
                insights, 
                corrections
            )
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Create result
            result = FeedbackProcessingResult(
                processing_id=processing_id,
                feedback_event=feedback_event,
                insights_extracted=insights,
                corrections_generated=corrections,
                learning_updates=learning_updates,
                confidence_score=confidence_score,
                processing_time_ms=processing_time
            )
            
            # Apply corrections if confidence is high enough
            if confidence_score >= self.feedback_config.correction_threshold and corrections:
                correction_results = await self.correction_engine.apply_corrections(corrections)
                result.learning_updates.append({
                    'type': 'correction_application',
                    'results': correction_results
                })
            
            # Store for tracking
            self.processing_history.append(result)
            self.total_processed += 1
            
            logger.info(f"Feedback processing completed", {
                'processing_id': processing_id,
                'insights_count': len(insights),
                'corrections_count': len(corrections),
                'confidence': confidence_score,
                'processing_time_ms': processing_time
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Feedback processing failed: {e}")
            return self._create_error_result(processing_id, str(e), start_time)
    
    async def process_immediate_feedback(self, delivery_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process immediate feedback from delivery results"""
        try:
            immediate_feedback = {
                'feedback_id': str(uuid.uuid4()),
                'customer_id': delivery_result.get('customer_id', ''),
                'interaction_id': delivery_result.get('interaction_id', ''),
                'feedback_type': 'immediate',
                'content': {
                    'delivery_status': delivery_result.get('status'),
                    'delivery_time_ms': delivery_result.get('metrics', {}).get('delivery_time_ms', 0),
                    'channels_used': delivery_result.get('channels_used', {}),
                    'initial_confidence': delivery_result.get('metrics', {}).get('initial_confidence', 0.5)
                },
                'confidence': 0.8,
                'source': 'delivery_system',
                'priority': 'normal'
            }
            
            # Process the immediate feedback
            result = await self.process_feedback(immediate_feedback)
            
            return {
                'processing_id': result.processing_id,
                'insights': result.insights_extracted,
                'corrections': len(result.corrections_generated),
                'confidence': result.confidence_score
            }
            
        except Exception as e:
            logger.error(f"Immediate feedback processing failed: {e}")
            return {'error': str(e), 'confidence': 0.0}
    
    def _generate_learning_updates(
        self, 
        feedback: FeedbackEvent, 
        insights: List[Dict[str, Any]], 
        classification: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate learning updates based on feedback processing"""
        learning_updates = []
        
        # Classification learning
        learning_updates.append({
            'type': 'classification_learning',
            'customer_id': feedback.customer_id,
            'classification': classification['primary_classification'],
            'confidence': classification['confidence'],
            'weight': feedback.learning_weight
        })
        
        # Insight learning
        for insight in insights:
            if insight.get('actionable'):
                learning_updates.append({
                    'type': 'insight_learning',
                    'insight_type': insight['insight_type'],
                    'customer_id': feedback.customer_id,
                    'confidence': insight.get('confidence', 0.5),
                    'weight': feedback.learning_weight
                })
        
        # Pattern learning
        if feedback.feedback_type == 'explicit' and classification.get('priority') in ['high', 'critical']:
            learning_updates.append({
                'type': 'pattern_learning',
                'pattern': 'high_priority_explicit_feedback',
                'customer_id': feedback.customer_id,
                'trigger_conditions': {
                    'feedback_type': feedback.feedback_type,
                    'priority': classification.get('priority'),
                    'timestamp_hour': feedback.timestamp.hour
                },
                'weight': feedback.learning_weight * 1.5
            })
        
        return learning_updates
    
    def _calculate_processing_confidence(
        self, 
        classification: Dict[str, Any], 
        insights: List[Dict[str, Any]], 
        corrections: List[CorrectionAction]
    ) -> float:
        """Calculate confidence in feedback processing"""
        # Base confidence from classification
        classification_confidence = classification.get('confidence', 0.5)
        
        # Confidence from insights
        insight_confidences = [i.get('confidence', 0.5) for i in insights]
        insight_confidence = np.mean(insight_confidences) if insight_confidences else 0.5
        
        # Confidence from corrections
        correction_confidences = [c.confidence for c in corrections]
        correction_confidence = np.mean(correction_confidences) if correction_confidences else 0.5
        
        # Weighted average
        total_confidence = (
            classification_confidence * 0.3 +
            insight_confidence * 0.4 +
            correction_confidence * 0.3
        )
        
        return min(0.95, max(0.1, total_confidence))
    
    def _create_error_result(
        self, 
        processing_id: str, 
        error: str, 
        start_time: datetime
    ) -> FeedbackProcessingResult:
        """Create error result when processing fails"""
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return FeedbackProcessingResult(
            processing_id=processing_id,
            feedback_event=FeedbackEvent(
                feedback_id='error',
                customer_id='',
                interaction_id='',
                timestamp=datetime.utcnow(),
                feedback_type='error',
                content={'error': error},
                confidence=0.0,
                source='error',
                priority='low',
                learning_weight=0.0
            ),
            insights_extracted=[],
            corrections_generated=[],
            learning_updates=[],
            confidence_score=0.0,
            processing_time_ms=processing_time
        )
