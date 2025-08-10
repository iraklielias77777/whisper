import { logger } from '@userwhisperer/shared';

export interface EmailMessage {
  to: string;
  from?: string;
  subject: string;
  html: string;
  text?: string;
  attachments?: EmailAttachment[];
  headers?: Record<string, string>;
  tracking?: EmailTrackingOptions;
}

export interface EmailAttachment {
  filename: string;
  content: Buffer | string;
  contentType?: string;
  disposition?: 'attachment' | 'inline';
  cid?: string;
}

export interface EmailTrackingOptions {
  open_tracking?: boolean;
  click_tracking?: boolean;
  unsubscribe_tracking?: boolean;
  custom_args?: Record<string, string>;
}

export interface EmailDeliveryResult {
  message_id: string;
  status: 'sent' | 'queued' | 'failed';
  recipient: string;
  provider: string;
  sent_at: Date;
  error?: string;
  tracking_id?: string;
}

export class EmailService {
  private apiKey: string;
  private provider: 'sendgrid' | 'ses' | 'mailgun' | 'mock';
  private baseUrl: string;
  private defaultFrom: string;

  constructor(config: {
    api_key: string;
    provider?: 'sendgrid' | 'ses' | 'mailgun' | 'mock';
    base_url?: string;
    default_from?: string;
  }) {
    this.apiKey = config.api_key || 'mock-api-key';
    this.provider = config.provider || 'mock';
    this.baseUrl = config.base_url || '';
    this.defaultFrom = config.default_from || 'noreply@userwhisperer.ai';
    
    logger.info(`EmailService initialized with provider: ${this.provider}`);
  }

  public async initialize(): Promise<void> {
    logger.info('Initializing EmailService');
    // In production, this would initialize the email provider SDKs
    try {
      switch (this.provider) {
        case 'sendgrid':
          // Initialize SendGrid client
          break;
        case 'ses':
          // Initialize AWS SES client
          break;
        case 'mailgun':
          // Initialize Mailgun client
          break;
        case 'mock':
        default:
          // No initialization needed for mock
          break;
      }
      logger.info('EmailService initialized successfully');
    } catch (error) {
      logger.error('Failed to initialize EmailService:', error);
      throw error;
    }
  }

  public async send(request: any): Promise<any> {
    const { content, user_id } = request;
    
    // Build email message from delivery request
    const emailMessage: EmailMessage = {
      to: await this.getUserEmail(user_id),
      subject: content.subject || 'Message from User Whisperer',
      html: content.html_body || `<p>${content.text_body || content.body || ''}</p>`,
      text: content.text_body || content.body || '',
      from: content.from_email || this.defaultFrom,
      tracking: {
        open_tracking: true,
        click_tracking: true,
        custom_args: {
          message_id: request.message_id,
          user_id: request.user_id
        }
      }
    };

    const result = await this.sendEmail(emailMessage);
    
    return {
      success: result.status === 'sent',
      channel: 'email',
      provider_message_id: result.message_id,
      delivered_at: result.sent_at,
      error: result.error,
      should_retry: result.status === 'failed' && !result.error?.includes('invalid')
    };
  }

  private async getUserEmail(userId: string): Promise<string> {
    // In production, this would query the database for user email
    // For now, return a mock email
    return `user-${userId}@example.com`;
  }

  public async sendEmail(message: EmailMessage): Promise<EmailDeliveryResult> {
    logger.info(`Sending email to ${message.to} via ${this.provider}`, {
      subject: message.subject,
      provider: this.provider
    });

    try {
      switch (this.provider) {
        case 'sendgrid':
          return await this.sendViaSendGrid(message);
        case 'ses':
          return await this.sendViaAWSSES(message);
        case 'mailgun':
          return await this.sendViaMailgun(message);
        case 'mock':
        default:
          return await this.sendViaMock(message);
      }
    } catch (error) {
      logger.error('Email sending failed', {
        recipient: message.to,
        provider: this.provider,
        error: error instanceof Error ? error.message : String(error)
      });

      return {
        message_id: `failed_${Date.now()}`,
        status: 'failed',
        recipient: message.to,
        provider: this.provider,
        sent_at: new Date(),
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  public async sendBulkEmails(messages: EmailMessage[]): Promise<EmailDeliveryResult[]> {
    logger.info(`Sending bulk emails (${messages.length} messages) via ${this.provider}`);

    const results = await Promise.allSettled(
      messages.map(message => this.sendEmail(message))
    );

    return results.map((result, index) => {
      if (result.status === 'fulfilled') {
        return result.value;
      } else {
        return {
          message_id: `bulk_failed_${Date.now()}_${index}`,
          status: 'failed' as const,
          recipient: messages[index].to,
          provider: this.provider,
          sent_at: new Date(),
          error: result.reason instanceof Error ? result.reason.message : 'Bulk send failed'
        };
      }
    });
  }

  private async sendViaSendGrid(message: EmailMessage): Promise<EmailDeliveryResult> {
    // Mock SendGrid implementation - in production, use @sendgrid/mail
    logger.info('SendGrid email sending (mock implementation)', {
      to: message.to,
      subject: message.subject
    });

    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 100));

    const messageId = `sg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    return {
      message_id: messageId,
      status: 'sent',
      recipient: message.to,
      provider: 'sendgrid',
      sent_at: new Date(),
      tracking_id: `track_${messageId}`
    };
  }

  private async sendViaAWSSES(message: EmailMessage): Promise<EmailDeliveryResult> {
    // Mock AWS SES implementation - in production, use AWS SDK
    logger.info('AWS SES email sending (mock implementation)', {
      to: message.to,
      subject: message.subject
    });

    await new Promise(resolve => setTimeout(resolve, 150));

    const messageId = `ses_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    return {
      message_id: messageId,
      status: 'sent',
      recipient: message.to,
      provider: 'ses',
      sent_at: new Date()
    };
  }

  private async sendViaMailgun(message: EmailMessage): Promise<EmailDeliveryResult> {
    // Mock Mailgun implementation - in production, use mailgun-js
    logger.info('Mailgun email sending (mock implementation)', {
      to: message.to,
      subject: message.subject
    });

    await new Promise(resolve => setTimeout(resolve, 120));

    const messageId = `mg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    return {
      message_id: messageId,
      status: 'sent',
      recipient: message.to,
      provider: 'mailgun',
      sent_at: new Date()
    };
  }

  private async sendViaMock(message: EmailMessage): Promise<EmailDeliveryResult> {
    logger.info('Mock email sending', {
      to: message.to,
      subject: message.subject,
      from: message.from || this.defaultFrom
    });

    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, 50));

    const messageId = `mock_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    // Simulate occasional failures (5% failure rate)
    if (Math.random() < 0.05) {
      throw new Error('Mock email delivery failure');
    }

    return {
      message_id: messageId,
      status: 'sent',
      recipient: message.to,
      provider: 'mock',
      sent_at: new Date(),
      tracking_id: `mock_track_${messageId}`
    };
  }

  public async getDeliveryStatus(messageId: string): Promise<{
    status: 'sent' | 'delivered' | 'opened' | 'clicked' | 'bounced' | 'failed';
    events: Array<{
      event: string;
      timestamp: Date;
      details?: Record<string, any>;
    }>;
  }> {
    logger.info(`Getting delivery status for message: ${messageId}`);

    // Mock implementation - in production, query provider APIs
    return {
      status: 'delivered',
      events: [
        {
          event: 'sent',
          timestamp: new Date(Date.now() - 60000)
        },
        {
          event: 'delivered',
          timestamp: new Date(Date.now() - 30000)
        }
      ]
    };
  }

  public async validateEmail(email: string): Promise<{
    valid: boolean;
    reason?: string;
    suggestion?: string;
  }> {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    
    if (!emailRegex.test(email)) {
      return {
        valid: false,
        reason: 'Invalid email format'
      };
    }

    // Additional validation could include DNS checks, etc.
    return {
      valid: true
    };
  }

  public async healthCheck(): Promise<{ healthy: boolean; details: any }> {
    try {
      // Test basic functionality
      const testResult = await this.validateEmail('test@example.com');
      
      return {
        healthy: true,
        details: {
          provider: this.provider,
          api_key_configured: this.apiKey !== 'mock-api-key',
          default_from: this.defaultFrom,
          test_validation: testResult.valid
        }
      };
    } catch (error) {
      return {
        healthy: false,
        details: {
          error: error instanceof Error ? error.message : 'Unknown error'
        }
      };
    }
  }
}
