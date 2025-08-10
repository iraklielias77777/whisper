import { 
  ContentValidationResult, 
  ContentIssue, 
  ComplianceStatus, 
  GeneratedContent, 
  ContentRequest,
  QualityMetrics 
} from '../types';
import { logger } from '@userwhisperer/shared';

export class ContentValidator {
  private complianceRules: Record<string, any> = {};
  private qualityThresholds: Record<string, number> = {};
  private bannedWords: Set<string> = new Set();
  private spamTriggers: Set<string> = new Set();

  constructor() {
    this.initializeRules();
    this.initializeQualityThresholds();
    this.initializeBannedWords();
    this.initializeSpamTriggers();
  }

  public async validate(
    content: GeneratedContent,
    request: ContentRequest
  ): Promise<ContentValidationResult> {
    return this.validateContent(content, request);
  }

  public async validateContent(
    content: GeneratedContent,
    request: ContentRequest
  ): Promise<ContentValidationResult> {
    logger.info(`Validating content for user ${request.user_id}`);

    try {
      const issues: ContentIssue[] = [];
      const qualityMetrics = await this.calculateQualityMetrics(content, request);
      
      // Run all validation checks in parallel
      const [
        grammarIssues,
        complianceIssues,
        qualityIssues,
        lengthIssues,
        toneIssues,
        spamIssues,
        personalizationIssues
      ] = await Promise.all([
        this.checkGrammar(content),
        this.checkCompliance(content, request),
        this.checkQuality(content, request, qualityMetrics),
        this.checkLength(content, request),
        this.checkTone(content, request),
        this.checkSpamTriggers(content),
        this.checkPersonalization(content, request)
      ]);

      issues.push(
        ...grammarIssues,
        ...complianceIssues,
        ...qualityIssues,
        ...lengthIssues,
        ...toneIssues,
        ...spamIssues,
        ...personalizationIssues
      );

      // Calculate overall quality score
      const qualityScore = this.calculateOverallQualityScore(qualityMetrics, issues);
      
      // Determine if content passes validation
      const passes = this.determineValidationResult(issues, qualityScore);

      // Generate suggestions for improvement
      const suggestions = this.generateSuggestions(issues, qualityMetrics);

      const result: ContentValidationResult = {
        is_valid: passes,
        passes,
        quality_score: qualityScore,
        issues,
        suggestions,
        quality_metrics: qualityMetrics,
        compliance_status: this.getComplianceStatus(complianceIssues),
        validated_at: new Date()
      };

      logger.info(`Content validation completed for user ${request.user_id}`, {
        passes,
        quality_score: qualityScore,
        issue_count: issues.length
      });

      return result;

    } catch (error) {
      logger.error(`Content validation failed for user ${request.user_id}:`, error);
      
      return {
        is_valid: false,
        passes: false,
        quality_score: 0,
        issues: [{
          type: 'validation_error',
          severity: 'critical',
          message: `Validation failed: ${error instanceof Error ? error.message : String(error)}`,
          field: 'system',
          suggestion: 'Please try generating content again'
        }],
        suggestions: [],
        quality_metrics: this.getDefaultQualityMetrics(),
        compliance_status: { 
          compliant: false, 
          gdpr_compliant: false,
          can_spam_compliant: false,
          accessibility_score: 0,
          content_warnings: ['validation_error'],
          checks_passed: [], 
          checks_failed: ['validation_error'] 
        },
        validated_at: new Date()
      };
    }
  }

  private async calculateQualityMetrics(
    content: GeneratedContent,
    request: ContentRequest
  ): Promise<QualityMetrics> {
    const textToAnalyze = [
      content.subject,
      content.body,
      content.cta_text
    ].filter(Boolean).join(' ');

    return {
      readability_score: this.calculateReadabilityScore(textToAnalyze),
      sentiment_score: this.calculateSentimentScore(textToAnalyze),
      personalization_score: this.calculatePersonalizationScore(content, request),
      relevance_score: this.calculateRelevanceScore(content, request),
      spam_score: this.calculateSpamScore(textToAnalyze),
      cta_strength: this.calculateCtaStrength(content.cta_text),
      engagement_potential: this.calculateEngagementPotential(content, request),
      length_appropriateness: this.calculateLengthAppropriatenesss(content, request),
      sentiment_alignment: this.calculateSentimentAlignment(content, request),
      urgency_effectiveness: this.calculateUrgencyEffectiveness(content, request),
      trust_indicators: this.calculateTrustIndicators(content),
      clarity_score: this.calculateClarityScore(textToAnalyze),
      brand_alignment: this.calculateBrandAlignment(content, request)
    };
  }

  private calculateSentimentScore(text: string): number {
    // Simple sentiment analysis based on positive/negative keywords
    const positiveWords = ['great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'good', 'love', 'best'];
    const negativeWords = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointing'];
    
    const words = text.toLowerCase().split(/\s+/);
    const positiveCount = words.filter(word => positiveWords.includes(word)).length;
    const negativeCount = words.filter(word => negativeWords.includes(word)).length;
    
    if (positiveCount + negativeCount === 0) return 0.5; // Neutral
    return positiveCount / (positiveCount + negativeCount);
  }

  private calculateRelevanceScore(content: GeneratedContent, request: ContentRequest): number {
    // Calculate how relevant the content is to the user context
    const userContext = request.user_context;
    let relevanceScore = 0.5; // Base score
    
    // Check if content mentions user's industry or company
    const contentText = `${content.subject} ${content.body}`.toLowerCase();
    if (userContext.industry && contentText.includes(userContext.industry.toLowerCase())) {
      relevanceScore += 0.2;
    }
    if (userContext.company_name && contentText.includes(userContext.company_name.toLowerCase())) {
      relevanceScore += 0.2;
    }
    
    return Math.min(relevanceScore, 1.0);
  }

  private calculateSpamScore(text: string): number {
    // Calculate spam likelihood based on spam triggers
    const spamIndicators = Array.from(this.spamTriggers);
    const textLower = text.toLowerCase();
    
    let spamCount = 0;
    spamIndicators.forEach(indicator => {
      if (textLower.includes(indicator)) {
        spamCount++;
      }
    });
    
    // Return score between 0 (no spam) and 1 (high spam)
    return Math.min(spamCount / 5, 1.0);
  }

  private calculateReadabilityScore(text: string): number {
    // Simplified Flesch Reading Ease calculation
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0).length;
    const words = text.split(/\s+/).filter(w => w.length > 0).length;
    const syllables = this.countSyllables(text);

    if (sentences === 0 || words === 0) return 0;

    const avgSentenceLength = words / sentences;
    const avgSyllablesPerWord = syllables / words;

    const fleschScore = 206.835 - (1.015 * avgSentenceLength) - (84.6 * avgSyllablesPerWord);
    
    // Convert to 0-1 scale
    return Math.max(0, Math.min(1, fleschScore / 100));
  }

  private countSyllables(text: string): number {
    // Simple syllable counting algorithm
    const words = text.toLowerCase().split(/\s+/);
    let totalSyllables = 0;

    for (const word of words) {
      const cleanWord = word.replace(/[^a-z]/g, '');
      if (cleanWord.length === 0) continue;

      let syllables = cleanWord.split(/[aeiouy]+/).length - 1;
      if (cleanWord.endsWith('e')) syllables -= 1;
      if (syllables === 0) syllables = 1;
      
      totalSyllables += syllables;
    }

    return totalSyllables;
  }

  private calculatePersonalizationScore(content: GeneratedContent, request: ContentRequest): number {
    let score = 0;
    let checks = 0;

    // Check for user name usage
    if (request.user_context?.user_details?.first_name) {
      checks++;
      if (content.body.includes(request.user_context.user_details.first_name)) {
        score += 0.3;
      }
    }

    // Check for company/context references
    if (request.user_context?.user_details?.company) {
      checks++;
      if (content.body.includes(request.user_context.user_details.company)) {
        score += 0.2;
      }
    }

    // Check for usage-specific personalization
    const favoriteFeatures = request.user_context?.usage_insights?.favorite_features;
    if (favoriteFeatures && favoriteFeatures.length > 0) {
      checks++;
      const hasFeatureReference = favoriteFeatures.some(feature => 
        content.body.toLowerCase().includes(feature.toLowerCase())
      );
      if (hasFeatureReference) {
        score += 0.3;
      }
    }

    // Check for behavioral insights
    if (request.user_context?.usage_insights?.activity_stats) {
      checks++;
      // If content references user activity or achievements
      const activityKeywords = ['activity', 'usage', 'progress', 'achievement'];
      const hasActivityReference = activityKeywords.some(keyword =>
        content.body.toLowerCase().includes(keyword)
      );
      if (hasActivityReference) {
        score += 0.2;
      }
    }

    return checks > 0 ? score : 0.1; // Minimum baseline score
  }

  private calculateCtaStrength(ctaText: string): number {
    if (!ctaText) return 0;

    let score = 0;
    const cta = ctaText.toLowerCase();

    // Strong action verbs
    const strongVerbs = ['get', 'start', 'discover', 'unlock', 'achieve', 'boost', 'transform', 'improve'];
    if (strongVerbs.some(verb => cta.includes(verb))) score += 0.3;

    // Urgency indicators
    const urgencyWords = ['now', 'today', 'limited', 'exclusive', 'instant'];
    if (urgencyWords.some(word => cta.includes(word))) score += 0.2;

    // Value proposition
    const valueWords = ['free', 'save', 'benefit', 'advantage', 'results'];
    if (valueWords.some(word => cta.includes(word))) score += 0.2;

    // Clarity and brevity (optimal length 2-5 words)
    const wordCount = ctaText.split(/\s+/).length;
    if (wordCount >= 2 && wordCount <= 5) score += 0.3;

    return Math.min(score, 1);
  }

  private calculateEngagementPotential(content: GeneratedContent, request: ContentRequest): number {
    let score = 0;

    // Subject line quality (for email)
    if (content.subject) {
      const subjectLength = content.subject.length;
      if (subjectLength >= 30 && subjectLength <= 50) score += 0.2;
      
      // Personalization in subject
      if (request.user_context?.user_details?.first_name && 
          content.subject.includes(request.user_context.user_details.first_name)) {
        score += 0.2;
      }

      // Curiosity/benefit in subject
      const engagementWords = ['discover', 'secret', 'exclusive', 'new', 'improve', 'save'];
      if (content.subject && engagementWords.some(word => content.subject!.toLowerCase().includes(word))) {
        score += 0.2;
      }
    }

    // Body content engagement
    const bodyText = content.body.toLowerCase();
    
    // Questions to engage reader
    if (bodyText.includes('?')) score += 0.1;
    
    // Social proof indicators
    const socialProofWords = ['customers', 'users', 'companies', 'success', 'proven'];
    if (socialProofWords.some(word => bodyText.includes(word))) score += 0.1;

    // Personal relevance
    const personalWords = ['you', 'your', 'yourself'];
    const personalCount = personalWords.reduce((count, word) => 
      count + (bodyText.match(new RegExp(word, 'g')) || []).length, 0);
    if (personalCount >= 3) score += 0.2;

    return Math.min(score, 1);
  }

  private calculateLengthAppropriatenesss(content: GeneratedContent, request: ContentRequest): number {
    const channel = request.channel;
    const preferredLength = request.user_context?.communication_preferences?.preferred_length || 'medium';
    
    let optimalRange: [number, number];
    
    // Define optimal ranges by channel and preference
    if (channel === 'sms') {
      optimalRange = [50, 160];
    } else if (channel === 'push') {
      optimalRange = [20, 60];
    } else if (channel === 'email') {
      switch (preferredLength) {
        case 'short':
          optimalRange = [100, 200];
          break;
        case 'long':
          optimalRange = [400, 800];
          break;
        default:
          optimalRange = [200, 400];
      }
    } else {
      optimalRange = [100, 300];
    }

    const bodyLength = content.body.length;
    const [min, max] = optimalRange;

    if (bodyLength >= min && bodyLength <= max) {
      return 1.0;
    } else if (bodyLength < min) {
      return Math.max(0.5, bodyLength / min);
    } else {
      return Math.max(0.5, max / bodyLength);
    }
  }

  private calculateSentimentAlignment(content: GeneratedContent, request: ContentRequest): number {
    const desiredTone = request.user_context?.communication_preferences?.preferred_tone || 'professional';
    const contentText = content.body.toLowerCase();

    const toneKeywords = {
      professional: ['professional', 'business', 'efficient', 'reliable', 'quality'],
      friendly: ['friendly', 'warm', 'welcome', 'happy', 'enjoy'],
      urgent: ['urgent', 'immediate', 'now', 'quickly', 'deadline'],
      casual: ['hey', 'awesome', 'cool', 'great', 'amazing']
    };

    const targetKeywords = toneKeywords[desiredTone] || toneKeywords.professional;
    const matchCount = targetKeywords.filter(keyword => 
      contentText.includes(keyword)
    ).length;

    return Math.min(matchCount / targetKeywords.length, 1);
  }

  private calculateUrgencyEffectiveness(content: GeneratedContent, request: ContentRequest): number {
    const urgencyLevel = request.strategy?.urgency || 'medium';
    const contentText = content.body.toLowerCase();

    const urgencyIndicators = {
      low: ['consider', 'when ready', 'at your convenience'],
      medium: ['soon', 'this week', 'opportunity'],
      high: ['today', 'limited time', 'act now'],
      critical: ['immediate', 'urgent', 'expires']
    };

    const expectedIndicators = urgencyIndicators[urgencyLevel] || [];
    const matchCount = expectedIndicators.filter(indicator =>
      contentText.includes(indicator)
    ).length;

    return expectedIndicators.length > 0 ? matchCount / expectedIndicators.length : 0.5;
  }

  private calculateTrustIndicators(content: GeneratedContent): number {
    const text = content.body.toLowerCase();
    let score = 0;

    // Trust signals
    const trustWords = ['secure', 'verified', 'guaranteed', 'trusted', 'certified', 'proven'];
    if (trustWords.some(word => text.includes(word))) score += 0.3;

    // Transparency indicators
    const transparencyWords = ['honest', 'transparent', 'clear', 'straightforward'];
    if (transparencyWords.some(word => text.includes(word))) score += 0.2;

    // Social proof
    const socialProofWords = ['rated', 'reviewed', 'testimonial', 'customers'];
    if (socialProofWords.some(word => text.includes(word))) score += 0.3;

    // Professional language (absence of spam triggers)
    const hasSpamTriggers = Array.from(this.spamTriggers).some(trigger => 
      text.includes(trigger)
    );
    if (!hasSpamTriggers) score += 0.2;

    return Math.min(score, 1);
  }

  private calculateClarityScore(text: string): number {
    let score = 1.0;

    // Penalize overly complex sentences
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    const avgWordsPerSentence = text.split(/\s+/).length / sentences.length;
    
    if (avgWordsPerSentence > 20) score -= 0.2;
    if (avgWordsPerSentence > 30) score -= 0.3;

    // Penalize jargon and complex words
    const complexWords = text.split(/\s+/).filter(word => 
      word.length > 10 || this.isJargon(word)
    );
    const complexWordRatio = complexWords.length / text.split(/\s+/).length;
    
    if (complexWordRatio > 0.1) score -= 0.2;
    if (complexWordRatio > 0.2) score -= 0.3;

    return Math.max(score, 0);
  }

  private calculateBrandAlignment(content: GeneratedContent, request: ContentRequest): number {
    // This would integrate with brand guidelines
    // For now, return a baseline score
    return 0.8;
  }

  private isJargon(word: string): boolean {
    const jargonWords = new Set([
      'synergy', 'leverage', 'optimization', 'operationalize', 'paradigm',
      'scalability', 'deliverables', 'actionable', 'bandwidth', 'holistic'
    ]);
    
    return jargonWords.has(word.toLowerCase());
  }

  private async checkGrammar(content: GeneratedContent): Promise<ContentIssue[]> {
    const issues: ContentIssue[] = [];
    
    // Basic grammar checks
    const texts = [content.subject, content.body, content.cta_text].filter(Boolean);
    
    for (let i = 0; i < texts.length; i++) {
      const text = texts[i];
      const fieldName = i === 0 ? 'subject' : i === 1 ? 'body' : 'cta_text';
      
      // Check for common grammar issues
      if (text && text.includes('  ')) {
        issues.push({
          type: 'grammar',
          severity: 'low',
          message: 'Multiple consecutive spaces found',
          field: fieldName,
          suggestion: 'Remove extra spaces'
        });
      }

      // Check capitalization
      if (fieldName === 'subject' && text && text.length > 0 && text[0] !== text[0].toUpperCase()) {
        issues.push({
          type: 'grammar',
          severity: 'medium',
          message: 'Subject should start with capital letter',
          field: fieldName,
          suggestion: 'Capitalize the first letter'
        });
      }

      // Check for incomplete sentences in body
      if (fieldName === 'body' && text && !text.trim().match(/[.!?]$/)) {
        issues.push({
          type: 'grammar',
          severity: 'medium',
          message: 'Content should end with proper punctuation',
          field: fieldName,
          suggestion: 'Add appropriate ending punctuation'
        });
      }
    }

    return issues;
  }

  private async checkCompliance(content: GeneratedContent, request: ContentRequest): Promise<ContentIssue[]> {
    const issues: ContentIssue[] = [];
    const allText = [content.subject, content.body, content.cta_text].filter(Boolean).join(' ').toLowerCase();

    // Check for required disclaimers
    if (request.intervention_type === 'monetization' && request.strategy?.special_offer) {
      if (!allText.includes('terms') && !allText.includes('conditions')) {
        issues.push({
          type: 'compliance',
          severity: 'high',
          message: 'Special offers should include terms and conditions reference',
          field: 'body',
          suggestion: 'Add "Terms and conditions apply" or similar'
        });
      }
    }

    // Check for CAN-SPAM compliance (email)
    if (request.channel === 'email') {
      if (!allText.includes('unsubscribe')) {
        issues.push({
          type: 'compliance',
          severity: 'critical',
          message: 'Email must include unsubscribe option',
          field: 'body',
          suggestion: 'Add unsubscribe link or text'
        });
      }
    }

    // Check for banned words
    for (const bannedWord of this.bannedWords) {
      if (allText.includes(bannedWord)) {
        issues.push({
          type: 'compliance',
          severity: 'high',
          message: `Contains banned word: "${bannedWord}"`,
          field: 'content',
          suggestion: `Remove or replace "${bannedWord}"`
        });
      }
    }

    return issues;
  }

  private async checkQuality(
    content: GeneratedContent, 
    request: ContentRequest, 
    metrics: QualityMetrics
  ): Promise<ContentIssue[]> {
    const issues: ContentIssue[] = [];

    // Check readability
    if (metrics.readability_score < this.qualityThresholds.readability) {
      issues.push({
        type: 'quality',
        severity: 'medium',
        message: 'Content may be difficult to read',
        field: 'body',
        suggestion: 'Use shorter sentences and simpler words'
      });
    }

    // Check personalization
    if (metrics.personalization_score < this.qualityThresholds.personalization) {
      issues.push({
        type: 'quality',
        severity: 'low',
        message: 'Content could be more personalized',
        field: 'body',
        suggestion: 'Include more user-specific information'
      });
    }

    // Check CTA strength
    if (metrics.cta_strength < this.qualityThresholds.cta_strength) {
      issues.push({
        type: 'quality',
        severity: 'medium',
        message: 'Call-to-action could be stronger',
        field: 'cta_text',
        suggestion: 'Use more compelling action words'
      });
    }

    return issues;
  }

  private async checkLength(content: GeneratedContent, request: ContentRequest): Promise<ContentIssue[]> {
    const issues: ContentIssue[] = [];
    const channel = request.channel;

    // Check subject line length (email)
    if (content.subject && channel === 'email') {
      if (content.subject.length > 60) {
        issues.push({
          type: 'length',
          severity: 'medium',
          message: 'Subject line may be too long for mobile devices',
          field: 'subject',
          suggestion: 'Keep subject line under 60 characters'
        });
      } else if (content.subject.length < 20) {
        issues.push({
          type: 'length',
          severity: 'low',
          message: 'Subject line may be too short',
          field: 'subject',
          suggestion: 'Consider making subject line more descriptive'
        });
      }
    }

    // Check body length by channel
    const bodyLength = content.body.length;
    
    if (channel === 'sms' && bodyLength > 160) {
      issues.push({
        type: 'length',
        severity: 'high',
        message: 'SMS message exceeds single message length',
        field: 'body',
        suggestion: 'Shorten message to under 160 characters'
      });
    } else if (channel === 'push' && bodyLength > 100) {
      issues.push({
        type: 'length',
        severity: 'medium',
        message: 'Push notification may be truncated',
        field: 'body',
        suggestion: 'Keep push notifications under 100 characters'
      });
    }

    return issues;
  }

  private async checkTone(content: GeneratedContent, request: ContentRequest): Promise<ContentIssue[]> {
    const issues: ContentIssue[] = [];
    const preferredTone = request.user_context?.communication_preferences?.preferred_tone || 'professional';
    const contentText = content.body.toLowerCase();

    // Check for tone mismatches
    if (preferredTone === 'professional') {
      const casualWords = ['hey', 'awesome', 'cool', 'dude', 'folks'];
      const foundCasual = casualWords.filter(word => contentText.includes(word));
      
      if (foundCasual.length > 0) {
        issues.push({
          type: 'tone',
          severity: 'medium',
          message: `Casual language may not match professional tone preference`,
          field: 'body',
          suggestion: `Consider replacing: ${foundCasual.join(', ')}`
        });
      }
    } else if (preferredTone === 'casual') {
      const formalWords = ['furthermore', 'henceforth', 'subsequently', 'therefore'];
      const foundFormal = formalWords.filter(word => contentText.includes(word));
      
      if (foundFormal.length > 0) {
        issues.push({
          type: 'tone',
          severity: 'medium',
          message: `Formal language may not match casual tone preference`,
          field: 'body',
          suggestion: `Consider using simpler alternatives`
        });
      }
    }

    return issues;
  }

  private async checkSpamTriggers(content: GeneratedContent): Promise<ContentIssue[]> {
    const issues: ContentIssue[] = [];
    const allText = [content.subject, content.body, content.cta_text].filter(Boolean).join(' ').toLowerCase();

    const foundTriggers = Array.from(this.spamTriggers).filter(trigger => 
      allText.includes(trigger)
    );

    for (const trigger of foundTriggers) {
      issues.push({
        type: 'spam',
        severity: 'high',
        message: `Contains potential spam trigger: "${trigger}"`,
        field: 'content',
        suggestion: `Consider replacing or removing "${trigger}"`
      });
    }

    return issues;
  }

  private async checkPersonalization(content: GeneratedContent, request: ContentRequest): Promise<ContentIssue[]> {
    const issues: ContentIssue[] = [];

    // Check if user name is available but not used
    const firstName = request.user_context?.user_details?.first_name;
    if (firstName && !content.body.includes(firstName) && request.personalization_level !== 'low') {
      issues.push({
        type: 'personalization',
        severity: 'low',
        message: 'User name available but not used',
        field: 'body',
        suggestion: `Consider including "${firstName}" for better personalization`
      });
    }

    // Check for generic placeholders
    const placeholders = ['[NAME]', '[COMPANY]', '{user}', '{name}'];
    const foundPlaceholders = placeholders.filter(placeholder =>
      content.body.includes(placeholder) || content.subject?.includes(placeholder)
    );

    for (const placeholder of foundPlaceholders) {
      issues.push({
        type: 'personalization',
        severity: 'critical',
        message: `Unfilled placeholder found: ${placeholder}`,
        field: 'content',
        suggestion: `Replace ${placeholder} with actual user data`
      });
    }

    return issues;
  }

  private calculateOverallQualityScore(metrics: QualityMetrics, issues: ContentIssue[]): number {
    // Start with average of quality metrics
    const metricsAverage = (
      metrics.readability_score +
      metrics.personalization_score +
      metrics.cta_strength +
      metrics.engagement_potential +
      metrics.length_appropriateness +
      metrics.sentiment_alignment +
      metrics.clarity_score +
      metrics.brand_alignment
    ) / 8;

    // Apply penalties for issues
    let penalty = 0;
    for (const issue of issues) {
      switch (issue.severity) {
        case 'critical':
          penalty += 0.3;
          break;
        case 'high':
          penalty += 0.2;
          break;
        case 'medium':
          penalty += 0.1;
          break;
        case 'low':
          penalty += 0.05;
          break;
      }
    }

    return Math.max(0, Math.min(1, metricsAverage - penalty));
  }

  private determineValidationResult(issues: ContentIssue[], qualityScore: number): boolean {
    // Fail if there are critical issues
    const criticalIssues = issues.filter(issue => issue.severity === 'critical');
    if (criticalIssues.length > 0) return false;

    // Fail if quality score is too low
    if (qualityScore < 0.5) return false;

    // Fail if too many high severity issues
    const highIssues = issues.filter(issue => issue.severity === 'high');
    if (highIssues.length > 3) return false;

    return true;
  }

  private generateSuggestions(issues: ContentIssue[], metrics: QualityMetrics): string[] {
    const suggestions: string[] = [];

    // Aggregate suggestions from issues
    const uniqueSuggestions = new Set(issues.map(issue => issue.suggestion).filter(Boolean));
    suggestions.push(...Array.from(uniqueSuggestions).filter((s): s is string => typeof s === 'string'));

    // Add metric-based suggestions
    if (metrics.readability_score < 0.6) {
      suggestions.push('Simplify language and use shorter sentences');
    }

    if (metrics.personalization_score < 0.5) {
      suggestions.push('Add more user-specific details and references');
    }

    if (metrics.cta_strength < 0.6) {
      suggestions.push('Use stronger action verbs in call-to-action');
    }

    if (metrics.engagement_potential < 0.6) {
      suggestions.push('Add questions or engaging elements to increase interaction');
    }

    return suggestions.filter(Boolean).slice(0, 5); // Limit to top 5 suggestions
  }

  private getComplianceStatus(complianceIssues: ContentIssue[]): ComplianceStatus {
    const criticalCompliance = complianceIssues.filter(issue => 
      issue.type === 'compliance' && issue.severity === 'critical'
    );

    const highCompliance = complianceIssues.filter(issue => 
      issue.type === 'compliance' && issue.severity === 'high'
    );

    const checksPassed = [];
    const checksFailed = [];

    if (criticalCompliance.length === 0) {
      checksPassed.push('critical_compliance');
    } else {
      checksFailed.push('critical_compliance');
    }

    if (highCompliance.length === 0) {
      checksPassed.push('high_compliance');
    } else {
      checksFailed.push('high_compliance');
    }

    // Add specific compliance checks
    const allComplianceText = complianceIssues.map(issue => issue.message).join(' ').toLowerCase();
    
    if (!allComplianceText.includes('unsubscribe')) {
      checksPassed.push('unsubscribe_link');
    } else {
      checksFailed.push('unsubscribe_link');
    }

    if (!allComplianceText.includes('banned word')) {
      checksPassed.push('banned_words');
    } else {
      checksFailed.push('banned_words');
    }

    return {
      compliant: checksFailed.length === 0,
      gdpr_compliant: !checksFailed.includes('gdpr_violations'),
      can_spam_compliant: !checksFailed.includes('spam_violations'),
      accessibility_score: 0.8,
      content_warnings: checksFailed,
      checks_passed: checksPassed,
      checks_failed: checksFailed
    };
  }

  private getDefaultQualityMetrics(): QualityMetrics {
    return {
      readability_score: 0,
      sentiment_score: 0,
      personalization_score: 0,
      relevance_score: 0,
      spam_score: 0,
      cta_strength: 0,
      engagement_potential: 0,
      length_appropriateness: 0,
      sentiment_alignment: 0,
      urgency_effectiveness: 0,
      trust_indicators: 0,
      clarity_score: 0,
      brand_alignment: 0
    };
  }

  private initializeRules(): void {
    this.complianceRules = {
      email: {
        requires_unsubscribe: true,
        max_subject_length: 100,
        requires_sender_identification: true
      },
      sms: {
        max_length: 160,
        requires_opt_out: true,
        business_hours_only: true
      },
      push: {
        max_title_length: 50,
        max_body_length: 100,
        requires_consent: true
      }
    };
  }

  private initializeQualityThresholds(): void {
    this.qualityThresholds = {
      readability: 0.6,
      personalization: 0.4,
      cta_strength: 0.5,
      engagement: 0.5,
      overall_quality: 0.6
    };
  }

  private initializeBannedWords(): void {
    this.bannedWords = new Set([
      'guaranteed', 'risk-free', 'no risk', 'promise', 'swear',
      'urgent', 'act now', 'limited time', 'expires today',
      'free money', 'get rich', 'make money fast'
    ]);
  }

  private initializeSpamTriggers(): void {
    this.spamTriggers = new Set([
      'buy now', 'click here', 'congratulations', 'winner',
      'cash bonus', 'earn extra', 'extra income', 'fast cash',
      'financial freedom', 'get paid', 'guaranteed', 'home based',
      'increase sales', 'make money', 'money back', 'no cost',
      'no fees', 'no hidden', 'no investment', 'no purchase',
      'order now', 'promise', 'risk free', 'satisfaction guaranteed',
      'save big', 'special promotion', 'this is not spam', 'urgent'
    ]);
  }

  // Duplicate methods removed - using the ones defined earlier

  public async healthCheck(): Promise<{ healthy: boolean; details: any }> {
    try {
      return {
        healthy: true,
        details: {
          compliance_rules_loaded: Object.keys(this.complianceRules).length > 0,
          quality_thresholds_set: Object.keys(this.qualityThresholds).length > 0,
          banned_words_count: this.bannedWords.size,
          spam_triggers_count: this.spamTriggers.size
        }
      };
    } catch (error) {
      return {
        healthy: false,
        details: { error: error instanceof Error ? error.message : String(error) }
      };
    }
  }
} 