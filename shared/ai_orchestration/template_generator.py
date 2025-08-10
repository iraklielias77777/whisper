"""
Dynamic Template Generation System
Generates customer-specific templates that evolve with interactions
"""

import json
import uuid
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np

from .config_schemas import AISystemConfiguration

logger = logging.getLogger(__name__)

@dataclass
class TemplateVariant:
    """A/B test variant for template"""
    variant_id: str
    description: str
    risk_level: float
    elements: Dict[str, Any]
    performance_score: float = 0.0
    test_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class CustomerTemplate:
    """Customer-specific template structure"""
    template_id: str
    customer_id: str
    objective: str
    created_at: datetime
    version: str
    
    # Dynamic content blocks
    content_blocks: Dict[str, Any]
    
    # Personalization parameters
    personalization: Dict[str, Any]
    
    # Timing parameters
    timing: Dict[str, Any]
    
    # Channel preferences
    channels: Dict[str, Any]
    
    # A/B testing variants
    variants: List[TemplateVariant]
    
    # Success criteria
    success_criteria: Dict[str, Any]
    
    # Learning hooks
    learning_hooks: Dict[str, Any]
    
    # Performance tracking
    performance_metrics: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'created_at': self.created_at.isoformat(),
            'variants': [v.to_dict() for v in self.variants]
        }

class DynamicTemplateGenerator:
    """
    Generates customer-specific templates that evolve with interactions
    """
    
    def __init__(self, config: AISystemConfiguration):
        self.config = config
        self.template_cache = {}
        self.performance_tracker = {}
        self.evolution_history = {}
        self.global_patterns = {}
        
        # Initialize base templates
        self._initialize_base_templates()
        
        logger.info("Dynamic Template Generator initialized")
    
    def _initialize_base_templates(self):
        """Initialize base template structures"""
        self.base_templates = {
            'retention_critical': {
                'structure': {
                    'greeting': 'personalized_urgent',
                    'problem_acknowledgment': 'empathetic',
                    'value_reminder': 'specific_benefits',
                    'solution_offer': 'immediate_help',
                    'call_to_action': 'low_friction',
                    'closing': 'supportive'
                },
                'tone': 'urgent_helpful',
                'personalization_level': 5,
                'urgency': 'high'
            },
            'growth_opportunity': {
                'structure': {
                    'greeting': 'achievement_focused',
                    'success_recognition': 'specific_metrics',
                    'opportunity_presentation': 'value_based',
                    'roi_demonstration': 'quantified',
                    'call_to_action': 'consultation',
                    'closing': 'partnership'
                },
                'tone': 'professional_enthusiastic',
                'personalization_level': 4,
                'urgency': 'medium'
            },
            'onboarding_activation': {
                'structure': {
                    'greeting': 'welcoming_supportive',
                    'progress_acknowledgment': 'encouraging',
                    'next_step_guidance': 'clear_simple',
                    'value_preview': 'outcome_focused',
                    'call_to_action': 'guided_action',
                    'closing': 'available_support'
                },
                'tone': 'educational_friendly',
                'personalization_level': 3,
                'urgency': 'low'
            },
            'engagement_maintenance': {
                'structure': {
                    'greeting': 'familiar_friendly',
                    'value_delivery': 'relevant_content',
                    'feature_highlight': 'unexplored_potential',
                    'community_connection': 'peer_success',
                    'call_to_action': 'exploration',
                    'closing': 'ongoing_value'
                },
                'tone': 'helpful_informative',
                'personalization_level': 4,
                'urgency': 'low'
            }
        }
    
    async def generate_customer_template(
        self,
        customer_id: str,
        objective: str,
        context: Dict[str, Any]
    ) -> CustomerTemplate:
        """
        Generate a personalized template for specific customer
        """
        try:
            logger.info(f"Generating template for customer {customer_id}", {
                'objective': objective,
                'context_keys': list(context.keys())
            })
            
            # Load customer history
            history = await self._load_customer_history(customer_id)
            
            # Analyze what works for this customer
            success_patterns = self._analyze_success_patterns(history)
            failure_patterns = self._analyze_failure_patterns(history)
            
            # Get base template structure
            base_template = self._select_base_template(objective)
            
            # Generate template
            template = CustomerTemplate(
                template_id=f"template_{customer_id}_{int(datetime.utcnow().timestamp())}",
                customer_id=customer_id,
                objective=objective,
                created_at=datetime.utcnow(),
                version=self._get_template_version(customer_id),
                
                # Dynamic content blocks
                content_blocks=self._generate_content_blocks(history, context, base_template),
                
                # Personalization parameters
                personalization=self._generate_personalization_config(history, success_patterns),
                
                # Timing parameters
                timing=self._generate_timing_config(history, objective),
                
                # Channel preferences
                channels=self._generate_channel_config(history, objective),
                
                # A/B testing variants
                variants=self._generate_variants(success_patterns, failure_patterns, base_template),
                
                # Success criteria
                success_criteria=self._define_success_criteria(objective, history),
                
                # Learning hooks
                learning_hooks=self._setup_learning_hooks(objective),
                
                # Performance tracking
                performance_metrics=self._initialize_performance_tracking()
            )
            
            # Apply machine learning optimizations
            template = await self._apply_ml_optimizations(template, history)
            
            # Store template for performance tracking
            self.template_cache[template.template_id] = template
            
            logger.info(f"Template generated successfully", {
                'template_id': template.template_id,
                'customer_id': customer_id,
                'variants': len(template.variants)
            })
            
            return template
            
        except Exception as e:
            logger.error(f"Template generation failed for customer {customer_id}: {e}")
            return self._generate_fallback_template(customer_id, objective, context)
    
    def _select_base_template(self, objective: str) -> Dict[str, Any]:
        """Select appropriate base template"""
        objective_mapping = {
            'retention_critical': 'retention_critical',
            'payment_failed': 'retention_critical',
            'trial_conversion': 'growth_opportunity',
            'monetization_upsell': 'growth_opportunity',
            'onboarding_activation': 'onboarding_activation',
            'engagement_maintenance': 'engagement_maintenance',
            'reactivation': 'engagement_maintenance'
        }
        
        template_key = objective_mapping.get(objective, 'engagement_maintenance')
        return self.base_templates[template_key]
    
    def _generate_content_blocks(
        self, 
        history: Dict[str, Any], 
        context: Dict[str, Any], 
        base_template: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate dynamic content blocks"""
        try:
            content_blocks = {}
            
            # Greeting block
            content_blocks['greeting'] = self._generate_greeting(history, context)
            
            # Hook block
            content_blocks['hook'] = self._generate_hook(history, context)
            
            # Value proposition block
            content_blocks['value_proposition'] = self._generate_value_prop(history, context)
            
            # Social proof block
            content_blocks['social_proof'] = self._generate_social_proof(history, context)
            
            # Call to action block
            content_blocks['call_to_action'] = self._generate_cta(history, context)
            
            # Closing block
            content_blocks['closing'] = self._generate_closing(history, context)
            
            # Dynamic elements based on objective
            objective = context.get('objective', 'engagement_maintenance')
            if objective == 'retention_critical':
                content_blocks['urgency_element'] = self._generate_urgency_element(history)
                content_blocks['value_reminder'] = self._generate_value_reminder(history)
            elif objective in ['growth_opportunity', 'monetization_upsell']:
                content_blocks['roi_calculation'] = self._generate_roi_calculation(history)
                content_blocks['success_story'] = self._generate_success_story(history)
            elif objective == 'onboarding_activation':
                content_blocks['progress_indicator'] = self._generate_progress_indicator(history)
                content_blocks['next_steps'] = self._generate_next_steps(history)
            
            return content_blocks
            
        except Exception as e:
            logger.error(f"Content block generation failed: {e}")
            return self._get_default_content_blocks()
    
    def _generate_greeting(self, history: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate personalized greeting based on history"""
        try:
            # Analyze previous successful greetings
            successful_greetings = self._get_successful_elements(history, 'greeting')
            
            # Determine greeting style based on context
            time_since_last = context.get('time_since_last_interaction', 0)
            relationship_depth = history.get('relationship_depth', 0)
            customer_name = history.get('name', 'there')
            
            if time_since_last > 30:  # Days
                style = 'reengagement'
                variations = [
                    f"It's been a while, {customer_name}!",
                    f"Welcome back, {customer_name}!",
                    f"Great to see you again, {customer_name}!"
                ]
            elif relationship_depth > 0.7:
                style = 'familiar'
                variations = [
                    f"Hey {customer_name}!",
                    f"Hi {customer_name}, hope you're doing well!",
                    f"{customer_name}, quick update for you:"
                ]
            else:
                style = 'professional'
                variations = [
                    f"Hello {customer_name},",
                    f"Hi {customer_name},",
                    f"Dear {customer_name},"
                ]
            
            # Select based on historical performance
            selected = self._select_best_variation(variations, successful_greetings)
            
            return {
                'primary': selected,
                'alternatives': variations,
                'style': style,
                'personalization_level': self._calculate_personalization_level(history),
                'customer_name': customer_name
            }
            
        except Exception as e:
            logger.error(f"Greeting generation failed: {e}")
            return self._get_default_greeting()
    
    def _generate_hook(self, history: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate attention-grabbing hook"""
        try:
            customer_stage = history.get('lifecycle_stage', 'active')
            recent_behavior = context.get('behavioral_metrics', {})
            trigger_event = context.get('trigger_event', {})
            
            # Hook strategies based on context
            if customer_stage == 'at_risk':
                hooks = [
                    "We noticed you might be facing some challenges...",
                    "Your success is important to us, so we wanted to reach out...",
                    "We have some ideas that might help you get more value..."
                ]
            elif trigger_event.get('event_type') == 'feature_adoption':
                hooks = [
                    "You're already seeing great results, but there's more...",
                    "Based on how you're using our platform...",
                    "Other customers like you have discovered..."
                ]
            else:
                hooks = [
                    "We have something personalized for you...",
                    "Based on your recent activity...",
                    "Here's an opportunity you might find interesting..."
                ]
            
            return {
                'primary': self._select_best_variation(hooks, []),
                'alternatives': hooks,
                'context_based': True,
                'urgency_level': self._determine_urgency_level(context)
            }
            
        except Exception as e:
            logger.error(f"Hook generation failed: {e}")
            return {'primary': "We have something important to share with you."}
    
    def _generate_value_prop(self, history: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate value proposition"""
        try:
            objective = context.get('objective', 'engagement_maintenance')
            customer_profile = history.get('profile', {})
            
            value_props = {
                'retention_critical': [
                    "Get back on track with personalized support",
                    "Unlock the full potential you're missing",
                    "Let us help you achieve your goals faster"
                ],
                'growth_opportunity': [
                    "Expand your success with advanced features",
                    "Increase efficiency by 40% with premium tools",
                    "Join top performers using our enterprise features"
                ],
                'onboarding_activation': [
                    "Complete your setup in just 5 minutes",
                    "See immediate results with guided setup",
                    "Start achieving your goals today"
                ]
            }
            
            objective_props = value_props.get(objective, [
                "Discover personalized insights for your success",
                "Optimize your workflow with smart recommendations",
                "Achieve better results with less effort"
            ])
            
            # Customize based on customer profile
            customized_props = self._customize_value_props(objective_props, customer_profile)
            
            return {
                'primary': customized_props[0] if customized_props else objective_props[0],
                'alternatives': customized_props or objective_props,
                'objective_aligned': True,
                'quantified': self._add_quantification(customer_profile)
            }
            
        except Exception as e:
            logger.error(f"Value proposition generation failed: {e}")
            return {'primary': "Discover new ways to succeed with our platform."}
    
    def _generate_social_proof(self, history: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate social proof elements"""
        try:
            customer_segment = history.get('segment', 'general')
            company_size = history.get('company_size', 'small')
            industry = history.get('industry', 'technology')
            
            # Segment-specific social proof
            social_proof_elements = {
                'enterprise': [
                    "Fortune 500 companies trust us with their most critical workflows",
                    "Enterprise teams see 3x productivity improvements",
                    "Join 1000+ enterprise customers achieving exceptional results"
                ],
                'startup': [
                    "Fastest-growing startups use our platform to scale",
                    "Startup founders report 50% faster time-to-market",
                    "Join 5000+ innovative startups building with us"
                ],
                'smb': [
                    "Small businesses increase revenue by 25% on average",
                    "Over 10,000 growing businesses rely on our platform",
                    "Join successful small business owners seeing real results"
                ]
            }
            
            segment_elements = social_proof_elements.get(customer_segment, social_proof_elements['smb'])
            
            # Industry-specific elements
            industry_elements = self._get_industry_social_proof(industry)
            
            # Combine and rank
            all_elements = segment_elements + industry_elements
            ranked_elements = self._rank_social_proof(all_elements, history)
            
            return {
                'primary': ranked_elements[0] if ranked_elements else "Join thousands of successful customers",
                'alternatives': ranked_elements,
                'segment_specific': True,
                'industry_relevant': bool(industry_elements)
            }
            
        except Exception as e:
            logger.error(f"Social proof generation failed: {e}")
            return {'primary': "Join thousands of successful customers like you."}
    
    def _generate_cta(self, history: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate call-to-action"""
        try:
            objective = context.get('objective', 'engagement_maintenance')
            urgency_level = self._determine_urgency_level(context)
            customer_preferences = history.get('cta_preferences', {})
            
            # Objective-specific CTAs
            cta_options = {
                'retention_critical': [
                    "Get Help Now",
                    "Schedule Call",
                    "Contact Support",
                    "Start Recovery"
                ],
                'growth_opportunity': [
                    "Explore Upgrade",
                    "View Pricing",
                    "Start Free Trial",
                    "Schedule Demo"
                ],
                'onboarding_activation': [
                    "Continue Setup",
                    "Complete Profile",
                    "Start Tutorial",
                    "Get Started"
                ],
                'engagement_maintenance': [
                    "Learn More",
                    "Try Feature",
                    "View Tutorial",
                    "Explore Now"
                ]
            }
            
            objective_ctas = cta_options.get(objective, cta_options['engagement_maintenance'])
            
            # Adjust based on urgency
            if urgency_level == 'high':
                urgency_ctas = [f"{cta} Today" for cta in objective_ctas]
                objective_ctas = urgency_ctas + objective_ctas
            
            # Filter based on customer preferences
            if customer_preferences:
                filtered_ctas = [cta for cta in objective_ctas 
                               if any(pref in cta.lower() for pref in customer_preferences.get('preferred_actions', []))]
                if filtered_ctas:
                    objective_ctas = filtered_ctas
            
            return {
                'primary': objective_ctas[0],
                'alternatives': objective_ctas,
                'urgency_level': urgency_level,
                'objective_aligned': True,
                'friction_level': self._calculate_friction_level(objective_ctas[0])
            }
            
        except Exception as e:
            logger.error(f"CTA generation failed: {e}")
            return {'primary': "Learn More", 'friction_level': 'low'}
    
    def _generate_closing(self, history: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate closing message"""
        try:
            relationship_depth = history.get('relationship_depth', 0)
            support_history = history.get('support_interactions', 0)
            brand_voice = context.get('brand_voice', 'professional_friendly')
            
            if relationship_depth > 0.8:
                closings = [
                    "As always, we're here if you need anything.",
                    "Looking forward to your continued success!",
                    "Thanks for being an amazing customer."
                ]
            elif support_history > 0:
                closings = [
                    "Our support team is standing by if you need help.",
                    "Questions? We're here to help you succeed.",
                    "Need assistance? Just reach out anytime."
                ]
            else:
                closings = [
                    "We're here to support your success.",
                    "Feel free to reach out with any questions.",
                    "Thank you for choosing our platform."
                ]
            
            # Adjust for brand voice
            if 'casual' in brand_voice:
                closings = [closing.replace('We\'re', 'We are').replace('Thank you', 'Thanks') for closing in closings]
            
            return {
                'primary': closings[0],
                'alternatives': closings,
                'relationship_appropriate': True,
                'brand_aligned': True
            }
            
        except Exception as e:
            logger.error(f"Closing generation failed: {e}")
            return {'primary': "Thank you for your time."}
    
    def _generate_variants(
        self,
        success_patterns: Dict[str, Any],
        failure_patterns: Dict[str, Any],
        base_template: Dict[str, Any]
    ) -> List[TemplateVariant]:
        """Generate A/B test variants based on patterns"""
        try:
            variants = []
            
            # Conservative variant (proven elements)
            conservative = TemplateVariant(
                variant_id=str(uuid.uuid4()),
                description='Uses only proven successful elements',
                risk_level=0.1,
                elements=self._select_proven_elements(success_patterns)
            )
            variants.append(conservative)
            
            # Balanced variant (mix of proven and new)
            balanced = TemplateVariant(
                variant_id=str(uuid.uuid4()),
                description='Balances proven and experimental elements',
                risk_level=0.5,
                elements=self._mix_elements(success_patterns, 0.7)
            )
            variants.append(balanced)
            
            # Experimental variant (new approaches)
            experimental = TemplateVariant(
                variant_id=str(uuid.uuid4()),
                description='Tests new approaches avoiding past failures',
                risk_level=0.8,
                elements=self._generate_experimental_elements(failure_patterns)
            )
            variants.append(experimental)
            
            # ML-optimized variant
            ml_optimized = TemplateVariant(
                variant_id=str(uuid.uuid4()),
                description='Generated using ML predictions',
                risk_level=0.6,
                elements=self._generate_ml_elements(success_patterns, failure_patterns)
            )
            variants.append(ml_optimized)
            
            return variants
            
        except Exception as e:
            logger.error(f"Variant generation failed: {e}")
            return [self._get_default_variant()]
    
    # Helper methods for template generation
    async def _load_customer_history(self, customer_id: str) -> Dict[str, Any]:
        """Load customer history for template generation"""
        try:
            # This would integrate with the existing behavioral analysis service
            # For now, return mock data structure
            return {
                'customer_id': customer_id,
                'total_interactions': 0,
                'successful_interactions': [],
                'failed_interactions': [],
                'preferences': {},
                'lifecycle_stage': 'active',
                'relationship_depth': 0.5,
                'name': 'Customer',
                'company_size': 'small',
                'industry': 'technology'
            }
        except Exception as e:
            logger.error(f"Failed to load customer history: {e}")
            return {'customer_id': customer_id, 'total_interactions': 0}
    
    def _analyze_success_patterns(self, history: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze successful interaction patterns"""
        successful_interactions = history.get('successful_interactions', [])
        if not successful_interactions:
            return {}
        
        # Analyze patterns in successful interactions
        patterns = {
            'greeting_styles': self._extract_pattern_frequency(successful_interactions, 'greeting_style'),
            'content_lengths': self._extract_pattern_frequency(successful_interactions, 'content_length'),
            'tones': self._extract_pattern_frequency(successful_interactions, 'tone'),
            'cta_types': self._extract_pattern_frequency(successful_interactions, 'cta_type'),
            'timing': self._extract_pattern_frequency(successful_interactions, 'send_time')
        }
        
        return patterns
    
    def _analyze_failure_patterns(self, history: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze failed interaction patterns"""
        failed_interactions = history.get('failed_interactions', [])
        if not failed_interactions:
            return {}
        
        # Analyze patterns in failed interactions
        patterns = {
            'avoid_greeting_styles': self._extract_pattern_frequency(failed_interactions, 'greeting_style'),
            'avoid_content_lengths': self._extract_pattern_frequency(failed_interactions, 'content_length'),
            'avoid_tones': self._extract_pattern_frequency(failed_interactions, 'tone'),
            'avoid_cta_types': self._extract_pattern_frequency(failed_interactions, 'cta_type'),
            'avoid_timing': self._extract_pattern_frequency(failed_interactions, 'send_time')
        }
        
        return patterns
    
    # Additional helper methods would continue here...
    # (Including ML optimization, performance tracking, variant testing, etc.)
