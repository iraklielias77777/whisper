"""
Dynamic Prompt Generation Engine
Creates and evolves prompts based on performance and context
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
class PromptTemplate:
    """Template for dynamic prompt generation"""
    template_id: str
    name: str
    objective: str
    base_template: str
    variables: List[str]
    performance_score: float
    usage_count: int
    created_at: datetime
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }

class PromptOptimizationEngine:
    """
    Optimizes prompts based on performance feedback using ML techniques
    """
    
    def __init__(self):
        self.optimization_history = []
        self.performance_tracker = {}
        self.genetic_pool = []
        
    def optimize(self, prompt: str, performance_history: Dict) -> str:
        """Optimize prompt based on performance history"""
        try:
            # Simple optimization for now - in production this would use ML
            if not performance_history:
                return prompt
            
            # Extract performance patterns
            successful_elements = self._extract_successful_elements(performance_history)
            failed_elements = self._extract_failed_elements(performance_history)
            
            # Apply optimizations
            optimized = self._apply_optimizations(prompt, successful_elements, failed_elements)
            
            return optimized
            
        except Exception as e:
            logger.error(f"Prompt optimization failed: {e}")
            return prompt
    
    def _extract_successful_elements(self, history: Dict) -> List[str]:
        """Extract elements from successful prompts"""
        successful = []
        for interaction in history.get('successful_interactions', []):
            if interaction.get('confidence', 0) > 0.8:
                successful.extend(interaction.get('prompt_elements', []))
        return list(set(successful))
    
    def _extract_failed_elements(self, history: Dict) -> List[str]:
        """Extract elements from failed prompts"""
        failed = []
        for interaction in history.get('failed_interactions', []):
            if interaction.get('confidence', 1) < 0.4:
                failed.extend(interaction.get('prompt_elements', []))
        return list(set(failed))
    
    def _apply_optimizations(self, prompt: str, successful: List[str], failed: List[str]) -> str:
        """Apply optimization based on successful and failed elements"""
        optimized = prompt
        
        # Add successful elements if not present
        for element in successful[:3]:  # Top 3 successful elements
            if element not in prompt:
                optimized += f"\n\nIMPORTANT: {element}"
        
        # Remove or modify failed elements
        for element in failed:
            if element in optimized:
                optimized = optimized.replace(element, f"[AVOID: {element}]")
        
        return optimized

class DynamicPromptEngine:
    """
    Generates and evolves prompts based on performance and context
    """
    
    def __init__(self, config: AISystemConfiguration):
        self.config = config
        self.prompt_library = {}
        self.performance_history = {}
        self.evolution_state = {}
        self.optimization_engine = PromptOptimizationEngine()
        
        # Initialize master prompt templates
        self._initialize_master_templates()
        
        logger.info("Dynamic Prompt Engine initialized")
    
    def _initialize_master_templates(self):
        """Initialize master prompt templates"""
        self.master_templates = {
            "retention_critical": """
# AI CUSTOMER RETENTION SPECIALIST

You are an expert retention specialist with deep understanding of user psychology and behavioral analysis.

## PRIMARY OBJECTIVE
{objective}

## CUSTOMER CONTEXT
- Customer ID: {customer_id}
- Lifecycle Stage: {lifecycle_stage}
- Churn Risk: {churn_risk:.1%} (CRITICAL LEVEL)
- Engagement Score: {engagement_score:.1%}
- Days Since Last Active: {days_inactive}
- Lifetime Value: ${ltv:,.2f}

## BEHAVIORAL ANALYSIS
{behavioral_analysis}

## HISTORICAL CONTEXT
- Previous Successful Interventions: {successful_interventions}
- Failed Approaches: {failed_approaches}
- Communication Preferences: {communication_preferences}
- Optimal Contact Times: {optimal_timing}

## YOUR MISSION
Create a highly personalized retention strategy that:
1. Addresses the specific pain points causing churn risk
2. Leverages psychological triggers that resonate with this user
3. Provides immediate value to re-engage the customer
4. Creates a clear path back to regular engagement
5. Feels authentic and helpful, not desperate or pushy

## CONSTRAINTS
- Message must feel genuine and customer-focused
- Cannot offer discounts exceeding {max_discount}%
- Must respect user's communication preferences
- Should align with brand voice: {brand_voice}
- Must comply with: {compliance_requirements}

## OUTPUT REQUIREMENTS
Generate a structured response with:

### 1. PRIMARY MESSAGE
- Maximum 150 words
- Personalized greeting using customer's name
- Clear value proposition
- Empathetic tone acknowledging their situation
- Specific benefits relevant to their use case

### 2. SUBJECT LINE
- Maximum 50 characters
- Attention-grabbing but not clickbait
- Personalized when possible
- Urgency appropriate to churn risk level

### 3. CALL-TO-ACTION
- Maximum 5 words
- Action-oriented and specific
- Low-friction next step
- Aligned with retention objective

### 4. FOLLOW-UP SEQUENCE
- 3 progressive touchpoints over 14 days
- Escalating value propositions
- Different channels if primary fails
- Human handoff trigger conditions

### 5. SUCCESS PROBABILITY
- Estimated likelihood of re-engagement (0-100%)
- Confidence level in your approach (0-100%)
- Key risk factors to monitor
- Alternative approaches if primary fails

### 6. REASONING CHAIN
- Why you chose this approach
- Key behavioral insights utilized
- Psychological principles applied
- Expected customer response pathway

## PERFORMANCE OPTIMIZATION
Base your approach on these learned patterns:
- What has worked before: {historical_success_patterns}
- What to avoid: {historical_failure_patterns}
- Optimal timing insights: {timing_patterns}
- Channel effectiveness: {channel_performance}

## SELF-EVALUATION CRITERIA
Before finalizing, ensure your response:
- [ ] Addresses root cause of churn risk
- [ ] Provides immediate tangible value
- [ ] Feels personal and authentic
- [ ] Has clear, low-friction next steps
- [ ] Includes measurement framework
- [ ] Considers customer's emotional state
- [ ] Aligns with their success metrics

Generate your complete retention strategy now, ensuring maximum effectiveness for this specific customer situation.
""",

            "growth_opportunity": """
# AI GROWTH OPTIMIZATION SPECIALIST

You are an expert growth strategist focused on customer expansion and value maximization.

## PRIMARY OBJECTIVE
{objective}

## OPPORTUNITY ANALYSIS
- Customer ID: {customer_id}
- Current Plan: {current_plan}
- Usage Level: {usage_percentage}% of plan limits
- Growth Trajectory: {growth_trend}
- Account Value: ${current_mrr:,.2f}/month
- Expansion Potential: ${expansion_potential:,.2f}

## CUSTOMER PROFILE
- Company Size: {company_size}
- Industry: {industry}
- Use Case: {primary_use_case}
- Success Metrics: {success_metrics}
- Decision Maker Role: {decision_maker_role}

## BEHAVIORAL INSIGHTS
- Feature Usage Patterns: {feature_usage}
- Engagement Trends: {engagement_trends}
- Pain Points Identified: {pain_points}
- Success Moments: {success_moments}
- Team Growth Indicators: {team_growth}

## EXPANSION OPPORTUNITY
Available Upgrades:
{available_upgrades}

ROI Indicators:
- Time Savings: {time_savings_hours}h/month
- Cost Savings: ${cost_savings:,.2f}/month
- Efficiency Gains: {efficiency_gains}%
- Risk Mitigation: {risk_reduction}

## YOUR MISSION
Design an expansion strategy that:
1. Feels like natural progression, not aggressive sales
2. Addresses specific needs based on usage patterns
3. Demonstrates clear ROI and value
4. Maintains trust and relationship quality
5. Aligns with their business objectives

## STRATEGIC APPROACH
Focus on:
- Value-based positioning (not feature-based)
- Business impact quantification
- Risk mitigation benefits
- Competitive advantages
- Team/organizational benefits

## CONSTRAINTS
- No aggressive sales tactics
- Must provide genuine value proposition
- Respect budget indicators: {budget_signals}
- Align with customer's growth stage
- Consider seasonal/timing factors: {timing_considerations}

## OUTPUT REQUIREMENTS

### 1. VALUE PROPOSITION
- Clear business impact statement
- Quantified benefits where possible
- Risk mitigation elements
- Competitive differentiation
- Success story parallels

### 2. TIMING STRATEGY
- Optimal approach timing
- Key trigger events to leverage
- Seasonal considerations
- Decision-making timeline
- Budget cycle alignment

### 3. MESSAGING SEQUENCE
- Primary approach message
- Follow-up touchpoints (3-5)
- Objection handling responses
- Social proof integration
- Success story positioning

### 4. CHANNEL STRATEGY
- Primary communication channel
- Multi-touch sequence design
- Stakeholder involvement plan
- Decision maker engagement
- Champion development approach

### 5. SUCCESS METRICS
- Primary conversion indicator
- Secondary engagement metrics
- Timeline expectations
- ROI measurement framework
- Success probability estimate

### 6. RISK MITIGATION
- Potential objections and responses
- Competitive threats and positioning
- Budget/timing challenges
- Internal stakeholder alignment
- Alternative approaches

## PSYCHOLOGICAL FRAMEWORK
Leverage these principles:
- Social proof from similar companies
- Authority through expertise demonstration
- Reciprocity via value-first approach
- Commitment through collaborative planning
- Urgency through opportunity cost

## SUCCESS PATTERNS
Apply these learned insights:
- High-converting value props: {successful_value_props}
- Optimal timing indicators: {timing_success_patterns}
- Effective social proof: {social_proof_examples}
- Champion development tactics: {champion_strategies}

Generate your comprehensive expansion strategy ensuring maximum conversion probability while maintaining relationship integrity.
""",

            "onboarding_activation": """
# AI ONBOARDING SPECIALIST

You are an expert in user activation and onboarding optimization with deep knowledge of user psychology and product adoption.

## PRIMARY OBJECTIVE
{objective}

## NEW USER CONTEXT
- Customer ID: {customer_id}
- Signup Date: {signup_date}
- Days Since Signup: {days_since_signup}
- Source: {acquisition_source}
- Use Case Indicated: {indicated_use_case}
- Team Size: {team_size}
- Company Type: {company_type}

## ACTIVATION STATUS
- Profile Completion: {profile_completion}%
- Setup Steps Completed: {setup_completion}%
- First Value Moment: {first_value_achieved}
- Feature Exploration: {features_explored}/{total_features}
- Time in Product: {time_in_product} minutes

## BEHAVIORAL SIGNALS
- Login Frequency: {login_frequency}
- Session Duration: {avg_session_duration} minutes
- Feature Adoption Rate: {feature_adoption_rate}%
- Help Documentation Usage: {help_usage}
- Support Interactions: {support_tickets}

## ONBOARDING PROGRESS
Completed Steps:
{completed_steps}

Remaining Critical Steps:
{remaining_steps}

Identified Blockers:
{onboarding_blockers}

## YOUR MISSION
Create an activation strategy that:
1. Accelerates time-to-first-value
2. Reduces onboarding friction and confusion
3. Builds confidence and product understanding
4. Creates positive momentum and engagement
5. Establishes lasting product habits

## ACTIVATION FRAMEWORK
Focus on the "Aha Moment" sequence:
1. Core Value Discovery
2. Feature Mastery Building
3. Workflow Integration
4. Team Collaboration (if applicable)
5. Advanced Value Realization

## CONSTRAINTS
- Must respect user's learning pace
- Cannot overwhelm with too many features
- Should align with indicated use case
- Maintain helpful, educational tone
- Provide just-in-time guidance

## OUTPUT REQUIREMENTS

### 1. ACTIVATION MESSAGE
- Personalized progress acknowledgment
- Clear next step guidance
- Value-focused motivation
- Success story inspiration
- Immediate action item

### 2. GUIDED WORKFLOW
- Step-by-step activation sequence
- Interactive tutorials recommended
- Progressive disclosure strategy
- Milestone celebrations
- Progress tracking elements

### 3. VALUE DEMONSTRATION
- Quick wins identification
- Use case specific examples
- ROI calculation assistance
- Success metrics setup
- Benchmark comparisons

### 4. SUPPORT STRATEGY
- Proactive help triggers
- Resource recommendations
- Community engagement opportunities
- Expert consultation options
- Feedback collection points

### 5. MOMENTUM BUILDING
- Achievement recognition
- Progress visualization
- Social proof integration
- Team invitation prompts
- Advanced feature previews

### 6. SUCCESS METRICS
- Activation milestone targets
- Engagement level goals
- Time-to-value benchmarks
- Feature adoption targets
- Retention probability

## PERSONALIZATION ELEMENTS
Customize based on:
- Role/Job Title: {user_role}
- Industry Vertical: {industry}
- Company Size: {company_size}
- Technical Proficiency: {tech_savviness}
- Previous Tool Experience: {previous_tools}

## PROVEN PATTERNS
Apply these successful strategies:
- High-converting onboarding flows: {successful_flows}
- Effective activation triggers: {activation_triggers}
- Value demonstration tactics: {value_demo_patterns}
- Momentum building techniques: {momentum_strategies}

## PSYCHOLOGICAL PRINCIPLES
Utilize:
- Goal gradient effect for progress motivation
- Social proof through peer success stories
- Reciprocity via helpful guidance
- Achievement recognition for dopamine
- Curiosity gap for feature exploration

Generate your complete activation strategy optimized for this user's specific context and needs.
""",

            "engagement_maintenance": """
# AI ENGAGEMENT SPECIALIST

You are an expert in customer engagement and relationship management with deep understanding of user lifecycle optimization.

## PRIMARY OBJECTIVE
{objective}

## CUSTOMER CONTEXT
- Customer ID: {customer_id}
- Account Age: {account_age} months
- Lifecycle Stage: {lifecycle_stage}
- Current Engagement Level: {engagement_score:.1%}
- Engagement Trend: {engagement_trend}
- Last Activity: {last_activity}

## ENGAGEMENT METRICS
- Login Frequency: {login_frequency}
- Feature Usage Breadth: {feature_breadth}%
- Session Quality Score: {session_quality}
- Content Interaction Rate: {content_interaction}%
- Community Participation: {community_activity}

## BEHAVIORAL PATTERNS
Recent Activity:
{recent_activity_summary}

Usage Patterns:
{usage_patterns}

Preference Indicators:
{preference_indicators}

## ENGAGEMENT OPPORTUNITY
Identified Opportunities:
{engagement_opportunities}

Trending Features:
{trending_features}

Unused High-Value Features:
{unused_features}

## YOUR MISSION
Design an engagement strategy that:
1. Maintains consistent product interaction
2. Deepens feature adoption and mastery
3. Creates value discovery moments
4. Builds product habit strength
5. Prevents engagement decay

## ENGAGEMENT FRAMEWORK
Utilize the Hook Model:
1. Trigger: Relevant engagement prompt
2. Action: Low-friction valuable action
3. Reward: Immediate value delivery
4. Investment: Commitment building activity

## CONSTRAINTS
- Must respect user's time and attention
- Should align with their workflow patterns
- Cannot feel manipulative or pushy
- Maintain value-first approach
- Consider engagement fatigue risks

## OUTPUT REQUIREMENTS

### 1. ENGAGEMENT MESSAGE
- Contextual value proposition
- Personalized feature highlight
- Success story relevance
- Clear action invitation
- Benefit articulation

### 2. VALUE DELIVERY
- Immediate utility provision
- Skill building opportunity
- Efficiency improvement
- Outcome enhancement
- Knowledge expansion

### 3. HABIT REINFORCEMENT
- Routine integration suggestions
- Progress tracking elements
- Achievement recognition
- Streak building mechanics
- Social accountability options

### 4. DISCOVERY FACILITATION
- Feature exploration prompts
- Use case expansion ideas
- Advanced technique introduction
- Best practice sharing
- Peer learning opportunities

### 5. MOMENTUM SUSTAINING
- Progress celebration
- Goal setting assistance
- Challenge introduction
- Community connection
- Expert interaction

### 6. MEASUREMENT FRAMEWORK
- Engagement lift targets
- Feature adoption goals
- Session quality improvements
- Retention strengthening
- Value realization metrics

## PERSONALIZATION DEPTH
Customize for:
- User Persona: {user_persona}
- Skill Level: {skill_level}
- Motivation Type: {motivation_drivers}
- Learning Style: {learning_preference}
- Time Availability: {time_constraints}

## PROVEN STRATEGIES
Leverage successful patterns:
- High-engagement content types: {engaging_content}
- Effective feature introduction methods: {feature_intro_success}
- Habit formation techniques: {habit_patterns}
- Value demonstration approaches: {value_demo_methods}

## PSYCHOLOGICAL DRIVERS
Apply:
- Mastery motivation for skill building
- Autonomy support for self-direction
- Purpose connection for meaning
- Progress visualization for satisfaction
- Social connection for belonging

Generate your engagement strategy ensuring sustained value delivery and habit strengthening.
"""
        }
    
    async def generate_master_prompt(
        self,
        customer_profile: Dict[str, Any],
        context: Dict[str, Any],
        objective: str,
        constraints: List[str] = None
    ) -> str:
        """
        Generate a dynamic, context-aware master prompt
        """
        try:
            # Determine operational mode
            mode = self._determine_mode(customer_profile, context)
            
            # Calculate confidence requirement
            confidence_req = self._calculate_confidence_requirement(objective)
            
            # Select appropriate base template
            base_template = self._select_base_template(objective)
            
            # Extract dynamic variables
            variables = self._extract_variables(customer_profile, context, objective)
            
            # Build prompt with dynamic content
            prompt = self._build_dynamic_prompt(
                base_template,
                variables,
                mode,
                confidence_req,
                constraints or []
            )
            
            # Add learned elements
            prompt = self._add_learned_elements(prompt, customer_profile)
            
            # Add performance adjustments
            prompt = self._add_performance_adjustments(prompt, customer_profile)
            
            # Optimize based on history
            prompt = self.optimization_engine.optimize(
                prompt,
                self.performance_history.get(customer_profile.get('customer_id'), {})
            )
            
            logger.info(f"Generated master prompt for objective: {objective}", {
                'customer_id': customer_profile.get('customer_id'),
                'mode': mode,
                'confidence_required': confidence_req,
                'prompt_length': len(prompt)
            })
            
            return prompt
            
        except Exception as e:
            logger.error(f"Failed to generate master prompt: {e}")
            return self._get_fallback_prompt(objective, customer_profile)
    
    def _determine_mode(self, profile: Dict, context: Dict) -> str:
        """Determine operational mode based on context and profile"""
        confidence = profile.get('model_confidence', 0.5)
        interactions = profile.get('total_interactions', 0)
        
        if interactions < 10:
            return "EXPLORATION"
        elif confidence > 0.85:
            return "AUTONOMOUS"
        elif confidence > 0.65:
            return "GUIDED"
        else:
            return "CONSERVATIVE"
    
    def _calculate_confidence_requirement(self, objective: str) -> float:
        """Calculate required confidence based on objective criticality"""
        confidence_mapping = {
            'retention_critical': 0.90,
            'payment_failed': 0.85,
            'trial_conversion': 0.80,
            'growth_opportunity': 0.75,
            'onboarding_activation': 0.70,
            'engagement_maintenance': 0.65,
            'reactivation': 0.75,
            'support_resolution': 0.80
        }
        return confidence_mapping.get(objective, 0.70)
    
    def _select_base_template(self, objective: str) -> str:
        """Select the most appropriate base template"""
        template_mapping = {
            'retention_critical': 'retention_critical',
            'payment_failed': 'retention_critical',
            'trial_conversion': 'growth_opportunity', 
            'monetization_upsell': 'growth_opportunity',
            'onboarding_activation': 'onboarding_activation',
            'engagement_maintenance': 'engagement_maintenance',
            'reactivation': 'engagement_maintenance'
        }
        
        template_key = template_mapping.get(objective, 'engagement_maintenance')
        return self.master_templates.get(template_key, self.master_templates['engagement_maintenance'])
    
    def _extract_variables(self, profile: Dict, context: Dict, objective: str) -> Dict[str, Any]:
        """Extract variables for prompt template"""
        return {
            # Customer identification
            'customer_id': profile.get('customer_id', 'unknown'),
            'objective': objective,
            
            # Profile data
            'lifecycle_stage': profile.get('lifecycle_stage', 'active'),
            'engagement_score': profile.get('engagement_score', 0.5),
            'churn_risk': profile.get('churn_risk', 0.5),
            'ltv': profile.get('lifetime_value', 500.0),
            'days_inactive': profile.get('days_since_last_active', 0),
            
            # Behavioral analysis
            'behavioral_analysis': self._format_behavioral_analysis(profile),
            'successful_interventions': self._format_success_history(profile),
            'failed_approaches': self._format_failure_history(profile),
            'communication_preferences': self._format_preferences(profile),
            'optimal_timing': self._format_timing_preferences(profile),
            
            # Context data
            'brand_voice': context.get('brand_voice', 'professional_friendly'),
            'max_discount': context.get('max_discount_percent', 20),
            'compliance_requirements': ', '.join(context.get('compliance', ['GDPR'])),
            
            # Historical patterns
            'historical_success_patterns': self._get_success_patterns(profile),
            'historical_failure_patterns': self._get_failure_patterns(profile),
            'timing_patterns': self._get_timing_patterns(profile),
            'channel_performance': self._get_channel_performance(profile),
            
            # Current context
            'timestamp': datetime.utcnow().isoformat(),
            'session_context': context.get('session_data', {}),
            'trigger_event': context.get('trigger_event', {})
        }
    
    def _build_dynamic_prompt(
        self,
        template: str,
        variables: Dict[str, Any],
        mode: str,
        confidence_req: float,
        constraints: List[str]
    ) -> str:
        """Build the dynamic prompt with all context"""
        try:
            # Format the base template with variables
            formatted_prompt = template.format(**variables)
            
            # Add mode-specific instructions
            mode_instructions = {
                'EXPLORATION': """
## EXPLORATION MODE ACTIVE
You are in exploration mode. This means:
- Try innovative approaches that haven't been tested
- Be willing to experiment with new strategies
- Focus on learning and pattern discovery
- Document insights for future optimization
- Higher risk tolerance is acceptable
""",
                'AUTONOMOUS': """
## AUTONOMOUS MODE ACTIVE  
You are operating autonomously with high confidence:
- Use proven strategies and approaches
- Apply learned patterns from successful interactions
- Make decisions without requiring approval
- Focus on execution excellence
- Minimal human oversight needed
""",
                'GUIDED': """
## GUIDED MODE ACTIVE
You are operating with moderate confidence:
- Use a mix of proven and experimental approaches
- Provide detailed reasoning for decisions
- Flag any uncertainties for review
- Apply learned patterns where applicable
- Medium risk tolerance
""",
                'CONSERVATIVE': """
## CONSERVATIVE MODE ACTIVE
You are operating conservatively due to limited data:
- Use only proven, safe approaches
- Avoid experimental strategies
- Provide extensive reasoning and alternatives
- Flag for human review if uncertain
- Minimize risk of negative outcomes
"""
            }
            
            formatted_prompt += mode_instructions.get(mode, '')
            
            # Add confidence requirements
            formatted_prompt += f"""

## CONFIDENCE REQUIREMENTS
- Minimum confidence threshold: {confidence_req:.0%}
- If confidence falls below threshold, flag for human review
- Provide confidence score (0-100%) for each recommendation
- Explain factors contributing to confidence level
"""
            
            # Add constraints
            if constraints:
                formatted_prompt += f"""

## OPERATIONAL CONSTRAINTS
{chr(10).join(f"- {constraint}" for constraint in constraints)}
"""
            
            # Add current timestamp for context
            formatted_prompt += f"""

## CURRENT CONTEXT
- Timestamp: {datetime.utcnow().isoformat()}
- Processing Mode: {mode}
- Confidence Required: {confidence_req:.0%}
- Session: {self.session_id if hasattr(self, 'session_id') else 'unknown'}
"""
            
            return formatted_prompt
            
        except KeyError as e:
            logger.error(f"Missing variable in template: {e}")
            return self._get_fallback_prompt(variables.get('objective', 'unknown'), variables)
        except Exception as e:
            logger.error(f"Failed to build dynamic prompt: {e}")
            return self._get_fallback_prompt(variables.get('objective', 'unknown'), variables)
    
    def _format_behavioral_analysis(self, profile: Dict) -> str:
        """Format behavioral analysis data"""
        analysis = profile.get('behavioral_analysis', {})
        if not analysis:
            return "No detailed behavioral analysis available."
        
        return f"""
- Engagement Pattern: {analysis.get('engagement_pattern', 'Unknown')}
- Activity Level: {analysis.get('activity_level', 'Unknown')}
- Feature Preferences: {', '.join(analysis.get('preferred_features', []))}
- Time Patterns: {analysis.get('active_time_patterns', 'Unknown')}
- Session Behavior: {analysis.get('session_characteristics', 'Unknown')}
"""
    
    def _format_success_history(self, profile: Dict) -> str:
        """Format successful intervention history"""
        successes = profile.get('success_history', [])
        if not successes:
            return "No previous successful interventions recorded."
        
        formatted = []
        for success in successes[-3:]:  # Last 3 successes
            formatted.append(f"• {success.get('approach', 'Unknown')}: {success.get('outcome', 'Positive response')}")
        
        return '\n'.join(formatted)
    
    def _format_failure_history(self, profile: Dict) -> str:
        """Format failed intervention history"""
        failures = profile.get('failure_history', [])
        if not failures:
            return "No previous failed approaches recorded."
        
        formatted = []
        for failure in failures[-3:]:  # Last 3 failures
            formatted.append(f"• {failure.get('approach', 'Unknown')}: {failure.get('reason', 'Poor response')}")
        
        return '\n'.join(formatted)
    
    def _add_learned_elements(self, prompt: str, profile: Dict) -> str:
        """Add learned elements from customer history"""
        learned_elements = profile.get('learned_preferences', {})
        
        if learned_elements:
            prompt += f"""

## LEARNED CUSTOMER PREFERENCES
Based on {profile.get('total_interactions', 0)} previous interactions:

- Preferred Communication Style: {learned_elements.get('communication_style', 'Professional')}
- Optimal Message Length: {learned_elements.get('message_length', 'Medium')}
- Response Time Patterns: {learned_elements.get('response_timing', 'Business hours')}
- Content Preferences: {learned_elements.get('content_type', 'Educational')}
- Channel Effectiveness: {learned_elements.get('best_channels', 'Email')}

APPLY THESE PREFERENCES TO MAXIMIZE EFFECTIVENESS.
"""
        
        return prompt
    
    def _add_performance_adjustments(self, prompt: str, profile: Dict) -> str:
        """Add performance-based adjustments"""
        recent_performance = profile.get('recent_performance', {})
        
        if recent_performance:
            prompt += f"""

## RECENT PERFORMANCE INSIGHTS
- Last 30 days success rate: {recent_performance.get('success_rate', 0.5):.1%}
- Engagement trend: {recent_performance.get('engagement_trend', 'Stable')}
- Response rate trend: {recent_performance.get('response_trend', 'Stable')}

ADJUST STRATEGY BASED ON THESE PERFORMANCE INDICATORS.
"""
        
        return prompt
    
    def _get_fallback_prompt(self, objective: str, profile: Dict) -> str:
        """Generate fallback prompt when main generation fails"""
        return f"""
# FALLBACK AI ASSISTANT

## OBJECTIVE
{objective}

## CUSTOMER
- ID: {profile.get('customer_id', 'unknown')}
- Stage: {profile.get('lifecycle_stage', 'unknown')}

## INSTRUCTIONS
Generate a helpful, personalized response for this customer based on the objective.
Be professional, helpful, and focused on providing value.

## OUTPUT
Provide a clear recommendation with reasoning.
"""
    
    # Additional helper methods for data extraction and formatting
    def _format_preferences(self, profile: Dict) -> str:
        """Format communication preferences"""
        prefs = profile.get('communication_preferences', {})
        return f"Channel: {prefs.get('channel', 'Email')}, Frequency: {prefs.get('frequency', 'Weekly')}, Tone: {prefs.get('tone', 'Professional')}"
    
    def _format_timing_preferences(self, profile: Dict) -> str:
        """Format optimal timing preferences"""
        timing = profile.get('optimal_timing', {})
        return f"Best hours: {timing.get('hours', '9-17')}, Time zone: {timing.get('timezone', 'UTC')}, Days: {timing.get('days', 'Weekdays')}"
    
    def _get_success_patterns(self, profile: Dict) -> str:
        """Get successful pattern descriptions"""
        patterns = profile.get('success_patterns', [])
        return ', '.join(patterns) if patterns else "None identified yet"
    
    def _get_failure_patterns(self, profile: Dict) -> str:
        """Get failure pattern descriptions"""
        patterns = profile.get('failure_patterns', [])
        return ', '.join(patterns) if patterns else "None identified yet"
    
    def _get_timing_patterns(self, profile: Dict) -> str:
        """Get timing success patterns"""
        patterns = profile.get('timing_patterns', {})
        return f"Best send times: {patterns.get('optimal_hours', 'Unknown')}"
    
    def _get_channel_performance(self, profile: Dict) -> str:
        """Get channel performance data"""
        performance = profile.get('channel_performance', {})
        return ', '.join([f"{k}: {v:.1%}" for k, v in performance.items()]) if performance else "No data"
