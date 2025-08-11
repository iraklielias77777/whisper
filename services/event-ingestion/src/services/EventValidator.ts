import Ajv from 'ajv';
import addFormats from 'ajv-formats';
import { ValidationError } from './EventProcessor';

export interface ValidationResult {
  isValid: boolean;
  errors: ValidationError[];
}

export class EventValidator {
  private ajv: Ajv;
  private validators: Map<string, any>;
  
  constructor() {
    this.ajv = new Ajv({ allErrors: true });
    addFormats(this.ajv);
    this.validators = new Map();
    this.loadSchemas();
  }
  
  private loadSchemas(): void {
    // Base event schema
    const baseEventSchema = {
      type: 'object',
      required: ['event_type', 'user_id', 'timestamp'],
      properties: {
        event_type: {
          type: 'string',
          pattern: '^[a-z_]+$',
          minLength: 1,
          maxLength: 100
        },
        user_id: {
          type: 'string',
          minLength: 1,
          maxLength: 255
        },
        timestamp: {
          type: 'string',
          format: 'date-time'
        },
        event_id: {
          type: 'string',
          pattern: '^evt_[a-zA-Z0-9]{16}$'
        },
        app_id: {
          type: 'string',
          pattern: '^app_[a-zA-Z0-9]{16}$'
        },
        properties: {
          type: 'object',
          additionalProperties: true
        },
        context: {
          type: 'object',
          properties: {
            ip: { 
              type: 'string', 
              anyOf: [
                { format: 'ipv4' },
                { format: 'ipv6' }
              ]
            },
            user_agent: { type: 'string', maxLength: 1000 },
            session_id: { type: 'string', maxLength: 255 },
            device_type: { 
              type: 'string',
              enum: ['mobile', 'tablet', 'desktop', 'unknown']
            },
            referrer: { type: 'string', maxLength: 1000 },
            utm_source: { type: 'string', maxLength: 255 },
            utm_medium: { type: 'string', maxLength: 255 },
            utm_campaign: { type: 'string', maxLength: 255 },
            utm_content: { type: 'string', maxLength: 255 },
            utm_term: { type: 'string', maxLength: 255 }
          },
          additionalProperties: false
        }
      },
      additionalProperties: false
    };
    
    this.validators.set('base', this.ajv.compile(baseEventSchema));
    
    // Purchase event schema
    this.validators.set('purchase', this.ajv.compile({
      ...baseEventSchema,
      properties: {
        ...baseEventSchema.properties,
        properties: {
          type: 'object',
          required: ['amount', 'currency', 'product_id'],
          properties: {
            amount: { 
              type: 'number', 
              minimum: 0,
              maximum: 1000000
            },
            currency: { 
              type: 'string', 
              pattern: '^[A-Z]{3}$' 
            },
            product_id: { 
              type: 'string',
              minLength: 1,
              maxLength: 255
            },
            quantity: { 
              type: 'integer', 
              minimum: 1,
              maximum: 10000
            },
            discount_code: { type: 'string', maxLength: 100 },
            order_id: { type: 'string', maxLength: 255 },
            payment_method: {
              type: 'string',
              enum: ['credit_card', 'debit_card', 'paypal', 'stripe', 'bank_transfer', 'other']
            }
          },
          additionalProperties: false
        }
      }
    }));
    
    // User signup event schema
    this.validators.set('user_signup', this.ajv.compile({
      ...baseEventSchema,
      properties: {
        ...baseEventSchema.properties,
        properties: {
          type: 'object',
          required: ['email'],
          properties: {
            email: { 
              type: 'string', 
              format: 'email',
              maxLength: 255
            },
            name: { type: 'string', maxLength: 255 },
            plan: { 
              type: 'string',
              enum: ['free', 'basic', 'premium', 'enterprise']
            },
            referral_code: { type: 'string', maxLength: 100 },
            signup_method: {
              type: 'string',
              enum: ['email', 'google', 'facebook', 'github', 'linkedin', 'other']
            }
          },
          additionalProperties: false
        }
      }
    }));
    
    // Feature usage event schema
    this.validators.set('feature_used', this.ajv.compile({
      ...baseEventSchema,
      properties: {
        ...baseEventSchema.properties,
        properties: {
          type: 'object',
          required: ['feature_name'],
          properties: {
            feature_name: { 
              type: 'string',
              minLength: 1,
              maxLength: 255
            },
            feature_category: { type: 'string', maxLength: 100 },
            usage_duration: { type: 'number', minimum: 0 },
            success: { type: 'boolean' },
            error_message: { type: 'string', maxLength: 1000 },
            metadata: {
              type: 'object',
              additionalProperties: true
            }
          },
          additionalProperties: false
        }
      }
    }));
    
    // Error event schema
    this.validators.set('error_encountered', this.ajv.compile({
      ...baseEventSchema,
      properties: {
        ...baseEventSchema.properties,
        properties: {
          type: 'object',
          required: ['error_type', 'error_message'],
          properties: {
            error_type: {
              type: 'string',
              enum: ['validation', 'network', 'server', 'client', 'unknown']
            },
            error_message: { 
              type: 'string',
              minLength: 1,
              maxLength: 1000
            },
            error_code: { type: 'string', maxLength: 100 },
            stack_trace: { type: 'string', maxLength: 5000 },
            url: { type: 'string', maxLength: 1000 },
            line_number: { type: 'integer', minimum: 0 },
            column_number: { type: 'integer', minimum: 0 },
            severity: {
              type: 'string',
              enum: ['low', 'medium', 'high', 'critical']
            }
          },
          additionalProperties: false
        }
      }
    }));
  }
  
  async validate(event: any): Promise<ValidationResult> {
    const errors: ValidationError[] = [];
    
    // First validate against base schema
    const baseValidator = this.validators.get('base');
    if (!baseValidator) {
      return {
        isValid: false,
        errors: [{ field: 'schema', message: 'Base validator not found' }]
      };
    }
    
    if (!baseValidator(event)) {
      errors.push(...this.formatErrors(baseValidator.errors || []));
    }
    
    // Then validate against specific event type schema if exists
    if (event.event_type && typeof event.event_type === 'string') {
      const specificValidator = this.validators.get(event.event_type);
      if (specificValidator && !specificValidator(event)) {
        errors.push(...this.formatErrors(specificValidator.errors || []));
      }
    }
    
    // Additional business logic validation
    const businessErrors = await this.validateBusinessRules(event);
    errors.push(...businessErrors);
    
    return { 
      isValid: errors.length === 0, 
      errors 
    };
  }
  
  private async validateBusinessRules(event: any): Promise<ValidationError[]> {
    const errors: ValidationError[] = [];
    
    // Timestamp validation
    if (event.timestamp) {
      const eventTime = new Date(event.timestamp);
      const now = new Date();
      
      // Timestamp cannot be in the future (allow 5 minutes skew)
      if (eventTime.getTime() > now.getTime() + 300000) {
        errors.push({
          field: 'timestamp',
          message: 'Timestamp cannot be more than 5 minutes in the future'
        });
      }
      
      // Timestamp cannot be older than 7 days
      const sevenDaysAgo = new Date();
      sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
      if (eventTime < sevenDaysAgo) {
        errors.push({
          field: 'timestamp',
          message: 'Timestamp cannot be older than 7 days'
        });
      }
      
      // Check for valid date
      if (isNaN(eventTime.getTime())) {
        errors.push({
          field: 'timestamp',
          message: 'Invalid timestamp format'
        });
      }
    }
    
    // User ID validation
    if (event.user_id) {
      // Check for suspicious patterns
      if (event.user_id.includes('test') || event.user_id.includes('admin')) {
        errors.push({
          field: 'user_id',
          message: 'User ID contains suspicious patterns'
        });
      }
      
      // Check for SQL injection patterns
      const sqlInjectionPattern = /('|(\\')|(--|;|\\\||<|>|%|script|union|select|insert|update|delete|drop|create|alter)/i;
      if (sqlInjectionPattern.test(event.user_id)) {
        errors.push({
          field: 'user_id',
          message: 'User ID contains potentially harmful content'
        });
      }
    }
    
    // Event type validation
    if (event.event_type) {
      // List of allowed event types
      const allowedEventTypes = [
        'user_signup', 'user_login', 'user_logout', 'user_activated',
        'purchase', 'subscription_started', 'subscription_cancelled', 'subscription_upgraded',
        'feature_used', 'feature_enabled', 'feature_disabled',
        'page_viewed', 'button_clicked', 'form_submitted',
        'error_encountered', 'support_ticket_created',
        'trial_started', 'trial_ended', 'trial_converted',
        'payment_failed', 'payment_succeeded',
        'api_call_made', 'export_created', 'import_completed',
        'notification_sent', 'notification_clicked', 'notification_dismissed',
        'session_started', 'session_ended'
      ];
      
      if (!allowedEventTypes.includes(event.event_type)) {
        errors.push({
          field: 'event_type',
          message: `Unknown event type: ${event.event_type}`
        });
      }
    }
    
    // Properties validation
    if (event.properties) {
      // Check properties size
      const propertiesString = JSON.stringify(event.properties);
      if (propertiesString.length > 50000) { // 50KB limit
        errors.push({
          field: 'properties',
          message: 'Properties object is too large (max 50KB)'
        });
      }
      
      // Check for deeply nested objects
      if (this.getObjectDepth(event.properties) > 10) {
        errors.push({
          field: 'properties',
          message: 'Properties object is too deeply nested (max 10 levels)'
        });
      }
    }
    
    // Purchase-specific validations
    if (event.event_type === 'purchase' && event.properties) {
      const { amount, currency, quantity = 1 } = event.properties;
      
      // Validate amount vs currency
      if (currency && amount) {
        const minAmounts: Record<string, number> = {
          'USD': 0.01, 'EUR': 0.01, 'GBP': 0.01,
          'JPY': 1, 'KRW': 1, 'INR': 1
        };
        
        const minAmount = minAmounts[currency] || 0.01;
        if (amount < minAmount) {
          errors.push({
            field: 'properties.amount',
            message: `Amount too small for currency ${currency} (min: ${minAmount})`
          });
        }
      }
      
      // Validate reasonable quantity
      if (quantity > 1000) {
        errors.push({
          field: 'properties.quantity',
          message: 'Quantity seems unreasonably high'
        });
      }
    }
    
    return errors;
  }
  
  private formatErrors(ajvErrors: any[]): ValidationError[] {
    return ajvErrors.map(error => ({
      field: error.instancePath ? error.instancePath.substring(1) : error.schemaPath,
      message: error.message || 'Validation error'
    }));
  }
  
  private getObjectDepth(obj: any, depth = 0): number {
    if (depth > 20) return depth; // Prevent infinite recursion
    
    if (typeof obj !== 'object' || obj === null) {
      return depth;
    }
    
    let maxDepth = depth;
    for (const key in obj) {
      if (obj.hasOwnProperty(key)) {
        const currentDepth = this.getObjectDepth(obj[key], depth + 1);
        maxDepth = Math.max(maxDepth, currentDepth);
      }
    }
    
    return maxDepth;
  }
  
  // Add custom validation for specific event types
  addCustomValidator(eventType: string, schema: any): void {
    try {
      const validator = this.ajv.compile(schema);
      this.validators.set(eventType, validator);
    } catch (error) {
      throw new Error(`Failed to compile schema for event type ${eventType}: ${error}`);
    }
  }
  
  // Get validation statistics
  getValidationStats(): { supportedEventTypes: string[]; totalValidators: number } {
    return {
      supportedEventTypes: Array.from(this.validators.keys()),
      totalValidators: this.validators.size
    };
  }
} 