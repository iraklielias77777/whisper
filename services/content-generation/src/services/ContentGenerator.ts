import {
  ContentRequest,
  GeneratedContent,
  ContentVariation,
  PersonalizationContext,
  ContentGenConfig,
  Template,
  ContentValidationResult,
  ContentMetadata,
  PersonalizationData
} from '../types';
import { logger, eventsCache } from '@userwhisperer/shared';
import { TemplateEngine } from './TemplateEngine';
import { PersonalizationEngine } from './PersonalizationEngine';
import { ContentValidator } from './ContentValidator';
import { LLMProvider } from './LLMProvider';
import { v4 as uuidv4 } from 'uuid';

export class ContentGenerator {
  private config: ContentGenConfig;
  private templateEngine: TemplateEngine;
  private personalizationEngine: PersonalizationEngine;
  private contentValidator: ContentValidator;
  private llmProvider: LLMProvider;

  constructor(config: ContentGenConfig) {
    this.config = config;
    this.templateEngine = new TemplateEngine(config.template_engine);
    this.personalizationEngine = new PersonalizationEngine();
    this.contentValidator = new ContentValidator();
    this.llmProvider = new LLMProvider(config.llm_providers);
  }

  /**
   * Main content generation pipeline
   */
  public async generateContent(request: ContentRequest): Promise<GeneratedContent> {
    const startTime = Date.now();
    const contentId = uuidv4();

    try {
      logger.info('Starting content generation', {
        contentId,
        userId: request.user_id,
        interventionType: request.intervention_type,
        channel: request.channel,
        personalizationLevel: request.personalization_level
      });

      // Step 1: Check cache for similar content
      const cachedContent = await this.checkContentCache(request);
      if (cachedContent && request.personalization_level === 'low') {
        logger.info('Content served from cache', { contentId, userId: request.user_id });
        return cachedContent;
      }

      // Step 2: Select base template
      const template = await this.templateEngine.selectTemplate(
        request.intervention_type,
        request.channel,
        request.strategy
      );

      // Step 3: Gather personalization data
      const personalizationContext = await this.personalizationEngine.gatherPersonalizationData(
        request.user_id,
        request.user_context,
        request.strategy
      );

      // Step 4: Generate content variations
      const variations = await this.generateVariations(
        template,
        personalizationContext,
        request
      );

      // Step 5: Score and select best variation
      const bestVariation = await this.selectBestVariation(variations, request);

      // Step 6: Apply final personalization
      const finalContent = await this.applyFinalPersonalization(
        bestVariation.content,
        personalizationContext,
        request
      );

      // Step 7: Validate content
      const validation = await this.contentValidator.validate(finalContent, request);
      
      let processedContent = finalContent;
      if (!validation.is_valid) {
        logger.warn('Content validation failed, applying fixes', {
          contentId,
          issues: validation.issues.map(i => `${i.type}: ${i.message}`)
        });
        
        processedContent = await this.applyValidationFixes(finalContent, validation);
      }

      // Step 8: Enhance metadata
      processedContent.metadata = await this.enhanceMetadata(
        processedContent,
        template,
        personalizationContext,
        Date.now() - startTime
      );

      // Step 9: Cache generated content
      await this.cacheContent(request, processedContent);

      const processingTime = Date.now() - startTime;

      logger.info('Content generation completed', {
        contentId,
        userId: request.user_id,
        generationMethod: processedContent.metadata.generation_method,
        qualityScore: processedContent.metadata.quality_score,
        processingTime: `${processingTime}ms`
      });

      return processedContent;

    } catch (error) {
      const processingTime = Date.now() - startTime;
      
      logger.error('Content generation failed', {
        contentId,
        userId: request.user_id,
        error: error instanceof Error ? error.message : String(error),
        processingTime: `${processingTime}ms`
      });

      // Return fallback content
      return await this.generateFallbackContent(request, contentId);
    }
  }

  /**
   * Generate multiple content variations using different approaches
   */
  private async generateVariations(
    template: Template,
    personalizationContext: PersonalizationContext,
    request: ContentRequest
  ): Promise<ContentVariation[]> {
    const variations: ContentVariation[] = [];

    try {
      // Generate LLM-based variations if enabled and available
      if (this.config.llm_providers.openai || this.config.llm_providers.anthropic) {
        const llmVariations = await this.generateLLMVariations(
          template,
          personalizationContext,
          request
        );
        variations.push(...llmVariations);
      }

      // Always generate template-based variation as fallback
      const templateVariation = await this.generateTemplateVariation(
        template,
        personalizationContext,
        request
      );
      variations.push(templateVariation);

      // Limit variations to configured maximum
      return variations.slice(0, this.config.max_variations);

    } catch (error) {
      logger.error('Variation generation failed', {
        userId: request.user_id,
        templateId: template.id,
        error: error instanceof Error ? error.message : String(error)
      });

      // Return template variation as minimum viable content
      const templateVariation = await this.generateTemplateVariation(
        template,
        personalizationContext,
        request
      );
      return [templateVariation];
    }
  }

  /**
   * Generate variations using LLM providers
   */
  private async generateLLMVariations(
    template: Template,
    personalizationContext: PersonalizationContext,
    request: ContentRequest
  ): Promise<ContentVariation[]> {
    const variations: ContentVariation[] = [];

    try {
      // Build generation prompt
      const prompt = this.buildGenerationPrompt(template, personalizationContext, request);

      // Generate with primary LLM
      let llmResults = await this.llmProvider.generateContent(prompt, {
        model: 'primary',
        variations: Math.min(3, this.config.max_variations),
        temperature: this.getLLMTemperature(request.personalization_level)
      });

      // Convert LLM results to variations
      for (const [index, result] of llmResults.entries()) {
        const variation: ContentVariation = {
          id: uuidv4(),
          content: this.parseLLMResult(result, request.user_id),
          score: 0, // Will be scored later
          generation_params: {
            method: 'llm',
            model: result.model_used,
            prompt_length: prompt.length,
            variation_index: index
          }
        };
        variations.push(variation);
      }

    } catch (error) {
      logger.error('LLM generation failed', {
        userId: request.user_id,
        templateId: template.id,
        error: error instanceof Error ? error.message : String(error)
      });
      
      // LLM failures are expected, don't throw
    }

    return variations;
  }

  /**
   * Generate template-based variation
   */
  private async generateTemplateVariation(
    template: Template,
    personalizationContext: PersonalizationContext,
    request: ContentRequest
  ): Promise<ContentVariation> {
    // Prepare template variables
    const variables = this.prepareTemplateVariables(personalizationContext, request);

    // Render template
    const renderedContent = await this.templateEngine.renderTemplate(template, variables);

    const content: GeneratedContent = {
      content_id: uuidv4(),
      ...renderedContent,
      metadata: {
        template_id: template.id,
        generation_method: 'template',
        personalization_score: this.calculatePersonalizationScore(variables),
        readability_score: 0.7, // Will be calculated later
        sentiment_score: 0.6,   // Will be calculated later
        word_count: this.countWords(renderedContent.body),
        character_count: renderedContent.body.length,
        language: 'en',
        quality_score: 0.7 // Will be calculated later
      },
      personalization_data: {
        elements_used: Object.keys(variables),
        user_specific_data: variables,
        dynamic_content: {},
        personalization_confidence: 0.8
      },
      generated_at: new Date()
    };

    return {
      id: uuidv4(),
      content,
      score: 0, // Will be scored later
      generation_params: {
        method: 'template',
        template_id: template.id,
        variables_count: Object.keys(variables).length
      }
    };
  }

  /**
   * Build LLM generation prompt
   */
  private buildGenerationPrompt(
    template: Template,
    personalizationContext: PersonalizationContext,
    request: ContentRequest
  ): string {
    const userContext = request.user_context;
    const strategy = request.strategy;

    return `
Generate a personalized ${request.channel} message for the following context:

USER PROFILE:
- Name: ${userContext.name || 'User'}
- Days as customer: ${userContext.days_since_signup}
- Engagement level: ${this.getEngagementLevel(userContext.engagement_score)}
- Lifecycle stage: ${userContext.lifecycle_stage}
- Subscription: ${userContext.subscription_plan || 'free'}
- Churn risk: ${Math.round((userContext.churn_risk || 0) * 100)}%
- Upgrade probability: ${Math.round((userContext.upgrade_probability || 0) * 100)}%

RECENT ACTIVITY:
${userContext.recent_activity.slice(0, 3).map(activity => 
  `- ${activity.event_type} (${activity.timestamp})`
).join('\n')}

BEHAVIORAL SIGNALS:
- Most used feature: ${userContext.behavioral_signals.most_used_feature || 'Not available'}
- Usage trend: ${userContext.behavioral_signals.usage_trend}
- Session frequency: ${userContext.behavioral_signals.session_frequency} per week
- Hitting limits: ${userContext.behavioral_signals.hitting_limits ? 'Yes' : 'No'}

MESSAGE OBJECTIVE:
- Type: ${request.intervention_type}
- Channel: ${request.channel}
- Tone: ${strategy.tone}
- Approach: ${strategy.approach}
- Key messages: ${strategy.key_messages.join(', ')}
- CTA type: ${strategy.cta_type}
- Urgency: ${request.urgency || 'medium'}

PERSONALIZATION DATA:
${this.formatPersonalizationData(personalizationContext)}

CONTENT REQUIREMENTS:
- Personalization level: ${request.personalization_level}
- Language: ${userContext.locale || 'en-US'}
- Channel: ${request.channel}

${request.channel === 'email' ? `
Generate a JSON response with:
{
  "subject": "Compelling email subject line",
  "preview_text": "Email preview text (50-90 chars)",
  "body": "Main email content with personalization",
  "cta_text": "Call to action button text",
  "cta_link": "{{cta_link}}" // Use placeholder
}
` : request.channel === 'push' ? `
Generate a JSON response with:
{
  "title": "Push notification title (50 chars max)",
  "body": "Push notification message (120 chars max)",
  "cta_text": "Action button text",
  "cta_link": "{{cta_link}}" // Use placeholder
}
` : `
Generate a JSON response with:
{
  "body": "Main message content with personalization",
  "cta_text": "Call to action text",
  "cta_link": "{{cta_link}}" // Use placeholder
}
`}

Guidelines:
1. Make it personally relevant to this specific user
2. Use their name and specific data points naturally
3. Address their current lifecycle stage and behavior
4. Create urgency without being pushy
5. Include clear value proposition
6. Use social proof when appropriate
7. Keep tone consistent with their preferences
8. Ensure CTA is compelling and specific
`;
  }

  /**
   * Select the best variation from generated options
   */
  private async selectBestVariation(
    variations: ContentVariation[],
    request: ContentRequest
  ): Promise<ContentVariation> {
    // Score each variation
    for (const variation of variations) {
      variation.score = await this.scoreVariation(variation, request);
    }

    // Sort by score (highest first)
    variations.sort((a, b) => b.score - a.score);

    // Log scoring results for analysis
    logger.debug('Variation scoring completed', {
      userId: request.user_id,
      variationScores: variations.map(v => ({
        id: v.id,
        method: v.generation_params.method,
        score: Math.round(v.score * 1000) / 1000
      }))
    });

    return variations[0];
  }

  /**
   * Score a content variation based on multiple factors
   */
  private async scoreVariation(
    variation: ContentVariation,
    request: ContentRequest
  ): Promise<number> {
    let score = 0.0;

    const content = variation.content;

    // 1. Readability score (20%)
    const readabilityScore = this.calculateReadabilityScore(content.body);
    score += readabilityScore * 0.2;

    // 2. Personalization score (30%)
    const personalizationScore = content.metadata.personalization_score || 0.5;
    score += personalizationScore * 0.3;

    // 3. CTA strength (20%)
    const ctaScore = this.evaluateCTAStrength(content.cta_text);
    score += ctaScore * 0.2;

    // 4. Length appropriateness (15%)
    const lengthScore = this.evaluateLength(content, request.channel);
    score += lengthScore * 0.15;

    // 5. Sentiment alignment (15%)
    const sentimentScore = this.evaluateSentiment(content.body, request.strategy.tone);
    score += sentimentScore * 0.15;

    return Math.min(1.0, Math.max(0.0, score));
  }

  /**
   * Apply final personalization touches
   */
  private async applyFinalPersonalization(
    content: GeneratedContent,
    personalizationContext: PersonalizationContext,
    request: ContentRequest
  ): Promise<GeneratedContent> {
    const personalizedContent = { ...content };

    // Add dynamic personalization elements
    if (personalizationContext.value_metrics) {
      personalizedContent.body = this.injectValueMetrics(
        personalizedContent.body,
        personalizationContext.value_metrics
      );
    }

    if (personalizationContext.social_proof_data && request.strategy.social_proof) {
      personalizedContent.body = this.injectSocialProof(
        personalizedContent.body,
        personalizationContext.social_proof_data
      );
    }

    if (personalizationContext.urgency_factors) {
      personalizedContent.body = this.injectUrgencyElements(
        personalizedContent.body,
        personalizationContext.urgency_factors
      );
    }

    return personalizedContent;
  }

  /**
   * Check content cache for similar requests
   */
  private async checkContentCache(request: ContentRequest): Promise<GeneratedContent | null> {
    try {
      const cacheKey = this.generateCacheKey(request);
      const cached = await eventsCache.get(cacheKey);

      if (cached) {
        const cachedContent = JSON.parse(cached) as GeneratedContent;
        
        // Update access time
        await eventsCache.expire(cacheKey, this.config.cache_ttl);
        
        return cachedContent;
      }

      return null;

    } catch (error) {
      logger.error('Cache check failed', {
        userId: request.user_id,
        error: error instanceof Error ? error.message : String(error)
      });
      return null;
    }
  }

  /**
   * Cache generated content
   */
  private async cacheContent(
    request: ContentRequest,
    content: GeneratedContent
  ): Promise<void> {
    try {
      const cacheKey = this.generateCacheKey(request);
      await eventsCache.setex(
        cacheKey,
        this.config.cache_ttl,
        JSON.stringify(content)
      );

      logger.debug('Content cached successfully', {
        userId: request.user_id,
        contentId: content.content_id,
        cacheKey
      });

    } catch (error) {
      logger.error('Content caching failed', {
        userId: request.user_id,
        contentId: content.content_id,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  /**
   * Generate fallback content when main generation fails
   */
  private async generateFallbackContent(
    request: ContentRequest,
    contentId: string
  ): Promise<GeneratedContent> {
    try {
      // Use simple template or static content
      const fallbackTemplate = await this.templateEngine.getFallbackTemplate(
        request.intervention_type,
        request.channel
      );

      const basicVariables = {
        user_name: request.user_context.name || 'there',
        product_name: 'our platform',
        company_name: 'User Whisperer'
      };

      const renderedContent = await this.templateEngine.renderTemplate(
        fallbackTemplate,
        basicVariables
      );

      return {
        content_id: contentId,
        ...renderedContent,
        metadata: {
          generation_method: 'fallback',
          personalization_score: 0.3,
          readability_score: 0.7,
          sentiment_score: 0.6,
          word_count: this.countWords(renderedContent.body),
          character_count: renderedContent.body.length,
          language: 'en',
          quality_score: 0.5
        },
        personalization_data: {
          elements_used: Object.keys(basicVariables),
          user_specific_data: basicVariables,
          dynamic_content: {},
          personalization_confidence: 0.3
        },
        generated_at: new Date()
      };

    } catch (error) {
      logger.error('Fallback content generation failed', {
        userId: request.user_id,
        contentId,
        error: error instanceof Error ? error.message : String(error)
      });

      // Ultimate fallback - static content
      return {
        content_id: contentId,
        body: `Hi ${request.user_context.name || 'there'}! We have an important update for you.`,
        cta_text: 'Learn More',
        cta_link: '{{cta_link}}',
        metadata: {
          generation_method: 'fallback',
          personalization_score: 0.1,
          readability_score: 0.8,
          sentiment_score: 0.6,
          word_count: 10,
          character_count: 50,
          language: 'en',
          quality_score: 0.3
        },
        personalization_data: {
          elements_used: ['user_name'],
          user_specific_data: { user_name: request.user_context.name || 'there' },
          dynamic_content: {},
          personalization_confidence: 0.1
        },
        generated_at: new Date()
      };
    }
  }

  // Helper methods

  private generateCacheKey(request: ContentRequest): string {
    const keyData = {
      intervention: request.intervention_type,
      channel: request.channel,
      lifecycle: request.user_context.lifecycle_stage,
      personalization: request.personalization_level,
      strategy_approach: request.strategy.approach
    };
    
    return `content:${Buffer.from(JSON.stringify(keyData)).toString('base64')}`;
  }

  private getLLMTemperature(personalizationLevel: string): number {
    switch (personalizationLevel) {
      case 'high': return 0.9;
      case 'medium': return 0.7;
      case 'low': return 0.5;
      default: return 0.7;
    }
  }

  private parseLLMResult(result: any, userId: string): GeneratedContent {
    try {
      const parsed = typeof result.content === 'string' 
        ? JSON.parse(result.content) 
        : result.content;

      return {
        content_id: uuidv4(),
        subject: parsed.subject,
        preview_text: parsed.preview_text,
        title: parsed.title,
        body: parsed.body || '',
        cta_text: parsed.cta_text || 'Learn More',
        cta_link: parsed.cta_link || '{{cta_link}}',
        metadata: {
          generation_method: 'llm',
          model_used: result.model_used,
          personalization_score: 0.8,
          readability_score: 0,
          sentiment_score: 0,
          word_count: this.countWords(parsed.body || ''),
          character_count: (parsed.body || '').length,
          language: 'en',
          quality_score: 0
        },
        personalization_data: {
          elements_used: [],
          user_specific_data: {},
          dynamic_content: {},
          personalization_confidence: 0.8
        },
        generated_at: new Date()
      };

    } catch (error) {
      logger.error('LLM result parsing failed', {
        userId,
        error: error instanceof Error ? error.message : String(error)
      });

      // Return basic content structure
      return {
        content_id: uuidv4(),
        body: result.content || 'Generated content',
        cta_text: 'Learn More',
        cta_link: '{{cta_link}}',
        metadata: {
          generation_method: 'llm',
          personalization_score: 0.5,
          readability_score: 0.6,
          sentiment_score: 0.6,
          word_count: 10,
          character_count: 50,
          language: 'en',
          quality_score: 0.5
        },
        personalization_data: {
          elements_used: [],
          user_specific_data: {},
          dynamic_content: {},
          personalization_confidence: 0.5
        },
        generated_at: new Date()
      };
    }
  }

  private prepareTemplateVariables(
    personalizationContext: PersonalizationContext,
    request: ContentRequest
  ): Record<string, any> {
    return {
      ...personalizationContext.user_data,
      ...personalizationContext.behavioral_insights,
      ...personalizationContext.contextual_data,
      user_name: request.user_context.name || 'there',
      company_name: 'User Whisperer',
      current_date: new Date().toLocaleDateString(),
      urgency_level: request.urgency || 'medium'
    };
  }

  private calculatePersonalizationScore(variables: Record<string, any>): number {
    const personalElements = Object.keys(variables).filter(key => 
      key.includes('user_') || 
      key.includes('personalized_') ||
      key.includes('custom_')
    );
    
    return Math.min(1.0, personalElements.length / 10);
  }

  private countWords(text: string): number {
    return text.trim().split(/\s+/).length;
  }

  private getEngagementLevel(score: number): string {
    if (score >= 0.8) return 'high';
    if (score >= 0.5) return 'medium';
    return 'low';
  }

  private formatPersonalizationData(context: PersonalizationContext): string {
    const items = [];
    
    if (context.value_metrics) {
      items.push(`Value metrics: ${JSON.stringify(context.value_metrics)}`);
    }
    
    if (context.social_proof_data) {
      items.push(`Social proof: ${context.social_proof_data.similar_users_count || 0} similar users`);
    }
    
    return items.join('\n');
  }

  private calculateReadabilityScore(text: string): number {
    // Simple readability score based on sentence and word length
    const sentences = text.split(/[.!?]+/).length;
    const words = this.countWords(text);
    const avgWordsPerSentence = words / sentences;
    
    // Ideal: 15-20 words per sentence
    if (avgWordsPerSentence >= 15 && avgWordsPerSentence <= 20) return 1.0;
    if (avgWordsPerSentence >= 10 && avgWordsPerSentence <= 25) return 0.8;
    return 0.6;
  }

  private evaluateCTAStrength(cta: string): number {
    const strongWords = ['get', 'start', 'join', 'discover', 'unlock', 'upgrade', 'claim'];
    const hasStrongWord = strongWords.some(word => 
      cta.toLowerCase().includes(word)
    );
    
    if (hasStrongWord && cta.length >= 5 && cta.length <= 25) return 1.0;
    if (cta.length >= 3 && cta.length <= 30) return 0.7;
    return 0.5;
  }

  private evaluateLength(content: GeneratedContent, channel: string): number {
    const wordCount = content.metadata.word_count;
    
    const idealRanges: Record<string, [number, number]> = {
      'email': [150, 300],
      'sms': [15, 50],
      'push': [10, 30],
      'in_app': [20, 100]
    };
    
    const [min, max] = idealRanges[channel] || [50, 200];
    
    if (wordCount >= min && wordCount <= max) return 1.0;
    if (wordCount >= min * 0.7 && wordCount <= max * 1.3) return 0.8;
    return 0.6;
  }

  private evaluateSentiment(text: string, desiredTone: string): number {
    // Simple sentiment evaluation
    const positiveWords = ['great', 'excellent', 'amazing', 'fantastic', 'wonderful'];
    const negativeWords = ['problem', 'issue', 'difficult', 'trouble', 'sorry'];
    
    const hasPositive = positiveWords.some(word => text.toLowerCase().includes(word));
    const hasNegative = negativeWords.some(word => text.toLowerCase().includes(word));
    
    if (desiredTone.includes('positive') && hasPositive && !hasNegative) return 1.0;
    if (desiredTone.includes('supportive') && !hasNegative) return 0.9;
    return 0.7;
  }

  private injectValueMetrics(body: string, metrics: any): string {
    // Inject value metrics into content
    if (metrics.time_saved) {
      body = body.replace(/\{\{time_saved\}\}/g, `${metrics.time_saved} hours`);
    }
    
    return body;
  }

  private injectSocialProof(body: string, socialProof: any): string {
    // Inject social proof elements
    if (socialProof.similar_users_count) {
      body = body.replace(/\{\{similar_users\}\}/g, `${socialProof.similar_users_count} users like you`);
    }
    
    return body;
  }

  private injectUrgencyElements(body: string, urgencyFactors: any): string {
    // Inject urgency indicators
    if (urgencyFactors.deadline) {
      const deadline = new Date(urgencyFactors.deadline).toLocaleDateString();
      body = body.replace(/\{\{deadline\}\}/g, deadline);
    }
    
    return body;
  }

  private async enhanceMetadata(
    content: GeneratedContent,
    template: Template,
    personalizationContext: PersonalizationContext,
    processingTime: number
  ): Promise<ContentMetadata> {
    const metadata = content.metadata;

    // Calculate missing scores
    metadata.readability_score = this.calculateReadabilityScore(content.body);
    metadata.sentiment_score = this.evaluateSentiment(content.body, 'positive');
    metadata.quality_score = (
      metadata.readability_score * 0.3 +
      metadata.personalization_score * 0.4 +
      metadata.sentiment_score * 0.3
    );

    return {
      ...metadata,
      template_id: template.id,
      estimated_reading_time: Math.ceil(metadata.word_count / 200), // 200 words per minute
      variation_id: uuidv4()
    };
  }

  private async applyValidationFixes(
    content: GeneratedContent,
    validation: ContentValidationResult
  ): Promise<GeneratedContent> {
    const fixedContent = { ...content };

    // Apply basic fixes for common issues
    for (const issue of validation.issues) {
      if (issue.type === 'length' && issue.severity === 'high') {
        // Truncate if too long
        if (fixedContent.body.length > 500) {
          fixedContent.body = fixedContent.body.substring(0, 480) + '...';
        }
      }
    }

    return fixedContent;
  }

  /**
   * Health check for content generator
   */
  public async healthCheck(): Promise<{ healthy: boolean; details: any }> {
    const details: any = {};
    let healthy = true;

    try {
      // Check component health
      details.template_engine = await this.templateEngine.healthCheck();
      details.personalization_engine = await this.personalizationEngine.healthCheck();
      details.content_validator = await this.contentValidator.healthCheck();
      details.llm_provider = await this.llmProvider.healthCheck();

      // Check if any component is unhealthy
      const componentHealth = [
        details.template_engine.healthy,
        details.personalization_engine.healthy,
        details.content_validator.healthy,
        details.llm_provider.healthy
      ];

      if (componentHealth.some(h => !h)) {
        healthy = false;
        details.error = 'One or more components unhealthy';
      }

      details.config = {
        max_variations: this.config.max_variations,
        quality_threshold: this.config.quality_threshold,
        cache_ttl: this.config.cache_ttl
      };

    } catch (error) {
      healthy = false;
      details.error = error instanceof Error ? error.message : String(error);
    }

    return { healthy, details };
  }
} 