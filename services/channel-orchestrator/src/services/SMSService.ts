import { logger } from '@userwhisperer/shared';

export interface SMSMessage {
  to: string;
  from?: string;
  body: string;
  media_url?: string[];
  status_callback?: string;
  max_price?: number;
  validity_period?: number;
}

export interface SMSDeliveryResult {
  message_id: string;
  status: 'sent' | 'queued' | 'failed' | 'delivered' | 'undelivered';
  recipient: string;
  provider: string;
  sent_at: Date;
  segments: number;
  price?: number;
  currency?: string;
  error?: string;
}

export class SMSService {
  private accountSid: string;
  private authToken: string;
  private provider: 'twilio' | 'aws-sns' | 'vonage' | 'mock';
  private defaultFrom: string;

  constructor(config: {
    account_sid: string;
    auth_token: string;
    provider?: 'twilio' | 'aws-sns' | 'vonage' | 'mock';
    default_from?: string;
  }) {
    this.accountSid = config.account_sid || 'mock-account-sid';
    this.authToken = config.auth_token || 'mock-auth-token';
    this.provider = config.provider || 'mock';
    this.defaultFrom = config.default_from || '+1234567890';
    
    logger.info(`SMSService initialized with provider: ${this.provider}`);
  }

  public async initialize(): Promise<void> {
    logger.info('Initializing SMSService');
    try {
      switch (this.provider) {
        case 'twilio':
          // Initialize Twilio client
          break;
        case 'aws-sns':
          // Initialize AWS SNS client
          break;
        case 'vonage':
          // Initialize Vonage client
          break;
        case 'mock':
        default:
          // No initialization needed for mock
          break;
      }
      logger.info('SMSService initialized successfully');
    } catch (error) {
      logger.error('Failed to initialize SMSService:', error);
      throw error;
    }
  }

  public async send(request: any): Promise<any> {
    const { content, user_id } = request;
    
    // Build SMS message from delivery request
    const smsMessage: SMSMessage = {
      to: await this.getUserPhoneNumber(user_id),
      body: content.body || content.text || 'Message from User Whisperer',
      from: content.from_number || this.defaultFrom
    };

    const result = await this.sendSMS(smsMessage);
    
    return {
      success: result.status === 'sent',
      channel: 'sms',
      provider_message_id: result.message_id,
      delivered_at: result.sent_at,
      error: result.error,
      should_retry: result.status === 'failed' && !result.error?.includes('invalid')
    };
  }

  private async getUserPhoneNumber(userId: string): Promise<string> {
    // In production, this would query the database for user phone number
    // For now, return a mock phone number
    return `+1555${userId.slice(-7).padStart(7, '0')}`;
  }

  public async sendSMS(message: SMSMessage): Promise<SMSDeliveryResult> {
    logger.info(`Sending SMS to ${message.to} via ${this.provider}`, {
      body_length: message.body.length,
      provider: this.provider
    });

    try {
      // Validate phone number format
      const phoneValidation = this.validatePhoneNumber(message.to);
      if (!phoneValidation.valid) {
        throw new Error(`Invalid phone number: ${phoneValidation.reason}`);
      }

      // Calculate SMS segments (160 characters per segment for basic SMS)
      const segments = Math.ceil(message.body.length / 160);

      switch (this.provider) {
        case 'twilio':
          return await this.sendViaTwilio(message, segments);
        case 'aws-sns':
          return await this.sendViaAWSSNS(message, segments);
        case 'vonage':
          return await this.sendViaVonage(message, segments);
        case 'mock':
        default:
          return await this.sendViaMock(message, segments);
      }
    } catch (error) {
      logger.error('SMS sending failed', {
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
        segments: Math.ceil(message.body.length / 160),
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  public async sendBulkSMS(messages: SMSMessage[]): Promise<SMSDeliveryResult[]> {
    logger.info(`Sending bulk SMS (${messages.length} messages) via ${this.provider}`);

    const results = await Promise.allSettled(
      messages.map(message => this.sendSMS(message))
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
          segments: Math.ceil(messages[index].body.length / 160),
          error: result.reason instanceof Error ? result.reason.message : 'Bulk send failed'
        };
      }
    });
  }

  private async sendViaTwilio(message: SMSMessage, segments: number): Promise<SMSDeliveryResult> {
    // Mock Twilio implementation - in production, use twilio SDK
    logger.info('Twilio SMS sending (mock implementation)', {
      to: message.to,
      segments
    });

    await new Promise(resolve => setTimeout(resolve, 200));

    const messageId = `tw_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    return {
      message_id: messageId,
      status: 'sent',
      recipient: message.to,
      provider: 'twilio',
      sent_at: new Date(),
      segments,
      price: segments * 0.0075, // $0.0075 per segment
      currency: 'USD'
    };
  }

  private async sendViaAWSSNS(message: SMSMessage, segments: number): Promise<SMSDeliveryResult> {
    // Mock AWS SNS implementation - in production, use AWS SDK
    logger.info('AWS SNS SMS sending (mock implementation)', {
      to: message.to,
      segments
    });

    await new Promise(resolve => setTimeout(resolve, 180));

    const messageId = `sns_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    return {
      message_id: messageId,
      status: 'sent',
      recipient: message.to,
      provider: 'aws-sns',
      sent_at: new Date(),
      segments,
      price: segments * 0.0055, // Lower cost via AWS
      currency: 'USD'
    };
  }

  private async sendViaVonage(message: SMSMessage, segments: number): Promise<SMSDeliveryResult> {
    // Mock Vonage implementation - in production, use @vonage/server-sdk
    logger.info('Vonage SMS sending (mock implementation)', {
      to: message.to,
      segments
    });

    await new Promise(resolve => setTimeout(resolve, 150));

    const messageId = `vg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    return {
      message_id: messageId,
      status: 'sent',
      recipient: message.to,
      provider: 'vonage',
      sent_at: new Date(),
      segments,
      price: segments * 0.0065,
      currency: 'USD'
    };
  }

  private async sendViaMock(message: SMSMessage, segments: number): Promise<SMSDeliveryResult> {
    logger.info('Mock SMS sending', {
      to: message.to,
      body_preview: message.body.substring(0, 50) + (message.body.length > 50 ? '...' : ''),
      segments
    });

    await new Promise(resolve => setTimeout(resolve, 100));

    const messageId = `mock_sms_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    // Simulate occasional failures (3% failure rate)
    if (Math.random() < 0.03) {
      throw new Error('Mock SMS delivery failure');
    }

    return {
      message_id: messageId,
      status: 'sent',
      recipient: message.to,
      provider: 'mock',
      sent_at: new Date(),
      segments,
      price: segments * 0.01, // Mock pricing
      currency: 'USD'
    };
  }

  public validatePhoneNumber(phoneNumber: string): {
    valid: boolean;
    reason?: string;
    formatted?: string;
  } {
    // Basic phone number validation
    const cleanNumber = phoneNumber.replace(/[\s\-\(\)]/g, '');
    
    // Must start with + and have 10-15 digits
    const phoneRegex = /^\+[1-9]\d{9,14}$/;
    
    if (!phoneRegex.test(cleanNumber)) {
      return {
        valid: false,
        reason: 'Phone number must be in international format (+1234567890)'
      };
    }

    return {
      valid: true,
      formatted: cleanNumber
    };
  }

  public async getDeliveryStatus(messageId: string): Promise<{
    status: 'sent' | 'delivered' | 'failed' | 'undelivered';
    events: Array<{
      event: string;
      timestamp: Date;
      details?: Record<string, any>;
    }>;
  }> {
    logger.info(`Getting SMS delivery status for message: ${messageId}`);

    // Mock implementation
    return {
      status: 'delivered',
      events: [
        {
          event: 'sent',
          timestamp: new Date(Date.now() - 30000)
        },
        {
          event: 'delivered',
          timestamp: new Date(Date.now() - 10000)
        }
      ]
    };
  }

  public getEstimatedCost(message: string, provider?: string): {
    segments: number;
    estimated_cost: number;
    currency: string;
  } {
    const segments = Math.ceil(message.length / 160);
    const rates = {
      twilio: 0.0075,
      'aws-sns': 0.0055,
      vonage: 0.0065,
      mock: 0.01
    };

    const rate = rates[provider as keyof typeof rates] || rates[this.provider as keyof typeof rates] || rates.mock;

    return {
      segments,
      estimated_cost: segments * rate,
      currency: 'USD'
    };
  }

  public async healthCheck(): Promise<{ healthy: boolean; details: any }> {
    try {
      const testValidation = this.validatePhoneNumber('+1234567890');
      const testCost = this.getEstimatedCost('Test message');
      
      return {
        healthy: true,
        details: {
          provider: this.provider,
          credentials_configured: this.accountSid !== 'mock-account-sid',
          default_from: this.defaultFrom,
          test_validation: testValidation.valid,
          test_cost_calculation: testCost
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
