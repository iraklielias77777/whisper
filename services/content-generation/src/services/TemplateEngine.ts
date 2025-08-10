import { Template, TemplateStructure, ContentStrategy } from '../types';
import { logger } from '@userwhisperer/shared';
import * as nunjucks from 'nunjucks';
import * as handlebars from 'handlebars';

export class TemplateEngine {
  private engine: 'nunjucks' | 'handlebars';
  private nunjucksEnv!: nunjucks.Environment; // Will be initialized in constructor
  private templates: Map<string, Template> = new Map();

  constructor(engine: 'nunjucks' | 'handlebars' = 'nunjucks') {
    this.engine = engine;
    this.initializeEngine();
    this.loadDefaultTemplates();
  }

  private initializeEngine(): void {
    if (this.engine === 'nunjucks') {
      this.nunjucksEnv = new nunjucks.Environment();
      this.nunjucksEnv.addFilter('capitalize', (str: string) => 
        str.charAt(0).toUpperCase() + str.slice(1)
      );
    } else if (this.engine === 'handlebars') {
      handlebars.registerHelper('capitalize', (str: string) => 
        str.charAt(0).toUpperCase() + str.slice(1)
      );
    }
  }

  public async selectTemplate(
    interventionType: string,
    channel: string,
    strategy: ContentStrategy
  ): Promise<Template> {
    const templateKey = `${interventionType}_${channel}`;
    const template = this.templates.get(templateKey) || this.templates.get('default_email');
    
    if (!template) {
      throw new Error(`No template found for ${interventionType} on ${channel}`);
    }

    return template;
  }

  public async renderTemplate(template: Template, variables: Record<string, any>): Promise<{
    subject?: string;
    preview_text?: string;
    title?: string;
    body: string;
    cta_text: string;
    cta_link: string;
  }> {
    try {
      const rendered = {
        subject: template.structure.subject ? this.renderString(template.structure.subject, variables) : undefined,
        preview_text: template.structure.preview_text ? this.renderString(template.structure.preview_text, variables) : undefined,
        title: template.structure.title ? this.renderString(template.structure.title, variables) : undefined,
        body: this.renderString(template.structure.body, variables),
        cta_text: this.renderString(template.structure.cta, variables),
        cta_link: '{{cta_link}}'
      };

      return rendered;
    } catch (error) {
      logger.error('Template rendering failed', {
        templateId: template.id,
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  private renderString(template: string, variables: Record<string, any>): string {
    try {
      if (this.engine === 'nunjucks') {
        return this.nunjucksEnv.renderString(template, variables);
      } else {
        const compiledTemplate = handlebars.compile(template);
        return compiledTemplate(variables);
      }
    } catch (error) {
      logger.error('String rendering failed', {
        template: template.substring(0, 100),
        error: error instanceof Error ? error.message : String(error)
      });
      return template; // Return original if rendering fails
    }
  }

  public async getFallbackTemplate(interventionType: string, channel: string): Promise<Template> {
    return {
      id: 'fallback',
      name: 'Fallback Template',
      description: 'Generic fallback template',
      intervention_type: interventionType,
      channel: channel,
      structure: {
        subject: 'Important Update',
        body: 'Hi {{user_name}}! We have an important update for you.',
        cta: 'Learn More'
      },
      variables: [],
      success_rate: 0.5,
      usage_count: 0,
      created_at: new Date(),
      updated_at: new Date(),
      tags: ['fallback'],
      version: '1.0'
    };
  }

  private loadDefaultTemplates(): void {
    // Retention Email Template
    this.templates.set('retention_email', {
      id: 'retention_email_v1',
      name: 'Retention Email - Value Reminder',
      description: 'Email template for re-engaging users',
      intervention_type: 'retention',
      channel: 'email',
      structure: {
        subject: "{{user_name}}, you're missing out on {{value_metric}}",
        preview_text: "See what you could achieve this week",
        body: `Hi {{user_name}},

We noticed you haven't been around lately, and we wanted to share what you're missing:

• Users like you save {{time_saved_hours}} hours per week
• {{feature_highlight}} is waiting for you
• We've made improvements you'll love

{{social_proof_message}}

Ready to jump back in?`,
        cta: "Continue where you left off"
      },
      variables: [
        { name: 'user_name', type: 'string', required: true, description: 'User\'s name' },
        { name: 'value_metric', type: 'string', required: false, description: 'Key value metric' },
        { name: 'time_saved_hours', type: 'number', required: false, description: 'Hours saved' },
        { name: 'feature_highlight', type: 'string', required: false, description: 'Feature to highlight' },
        { name: 'social_proof_message', type: 'string', required: false, description: 'Social proof element' }
      ],
      success_rate: 0.32,
      usage_count: 1250,
      created_at: new Date(),
      updated_at: new Date(),
      tags: ['retention', 'email'],
      version: '1.0'
    });

    // Monetization Email Template
    this.templates.set('monetization_email', {
      id: 'monetization_email_v1',
      name: 'Monetization Email - Upgrade Prompt',
      description: 'Email template for upgrade campaigns',
      intervention_type: 'monetization',
      channel: 'email',
      structure: {
        subject: "Unlock {{blocked_feature}} and more",
        preview_text: "You've hit your limit - here's how to keep going",
        body: `Hi {{user_name}},

Wow! You've {{achievement}} - that's incredible!

You've reached the limit of your {{current_plan}} plan. To keep your momentum going, upgrade to {{recommended_plan}} and get:

{{upgrade_benefits}}

Plus, as a valued user, we're offering you {{special_offer}}.

{{urgency_message}}`,
        cta: "Upgrade now and save {{discount}}"
      },
      variables: [
        { name: 'user_name', type: 'string', required: true, description: 'User\'s name' },
        { name: 'blocked_feature', type: 'string', required: true, description: 'Feature user can\'t access' },
        { name: 'achievement', type: 'string', required: true, description: 'What user achieved' },
        { name: 'current_plan', type: 'string', required: true, description: 'Current subscription plan' },
        { name: 'recommended_plan', type: 'string', required: true, description: 'Recommended plan' },
        { name: 'upgrade_benefits', type: 'string', required: true, description: 'Benefits of upgrading' },
        { name: 'special_offer', type: 'string', required: false, description: 'Special offer details' },
        { name: 'urgency_message', type: 'string', required: false, description: 'Urgency indicator' },
        { name: 'discount', type: 'string', required: false, description: 'Discount amount' }
      ],
      success_rate: 0.28,
      usage_count: 890,
      created_at: new Date(),
      updated_at: new Date(),
      tags: ['monetization', 'email'],
      version: '1.0'
    });

    // Push Notification Templates
    this.templates.set('retention_push', {
      id: 'retention_push_v1',
      name: 'Retention Push - Gentle Nudge',
      description: 'Push notification for gentle re-engagement',
      intervention_type: 'retention',
      channel: 'push',
      structure: {
        title: "{{user_name}}, you have {{count}} new {{items}}",
        body: "{{teaser_text}}. Tap to see more.",
        cta: "View Now"
      },
      variables: [
        { name: 'user_name', type: 'string', required: true, description: 'User\'s name' },
        { name: 'count', type: 'number', required: true, description: 'Number of items' },
        { name: 'items', type: 'string', required: true, description: 'Type of items' },
        { name: 'teaser_text', type: 'string', required: true, description: 'Teaser content' }
      ],
      success_rate: 0.38,
      usage_count: 2150,
      created_at: new Date(),
      updated_at: new Date(),
      tags: ['retention', 'push'],
      version: '1.0'
    });

    // Default fallback template
    this.templates.set('default_email', {
      id: 'default_email_v1',
      name: 'Default Email Template',
      description: 'Generic email template',
      intervention_type: 'general',
      channel: 'email',
      structure: {
        subject: "Important update for {{user_name}}",
        body: `Hi {{user_name}},

We have an important update to share with you.

{{message_content}}

Best regards,
The Team`,
        cta: "Learn More"
      },
      variables: [
        { name: 'user_name', type: 'string', required: true, description: 'User\'s name' },
        { name: 'message_content', type: 'string', required: true, description: 'Main message content' }
      ],
      success_rate: 0.25,
      usage_count: 500,
      created_at: new Date(),
      updated_at: new Date(),
      tags: ['default', 'email'],
      version: '1.0'
    });
  }

  public async healthCheck(): Promise<{ healthy: boolean; details: any }> {
    const details: any = {};
    let healthy = true;

    try {
      details.engine = this.engine;
      details.templates_loaded = this.templates.size;
      
      // Test template rendering
      const testTemplate = 'Hello {{name}}!';
      const testVars = { name: 'Test' };
      const rendered = this.renderString(testTemplate, testVars);
      
      details.rendering_test = rendered === 'Hello Test!' ? 'passed' : 'failed';
      
      if (details.rendering_test === 'failed') {
        healthy = false;
      }

    } catch (error) {
      healthy = false;
      details.error = error instanceof Error ? error.message : String(error);
    }

    return { healthy, details };
  }
} 