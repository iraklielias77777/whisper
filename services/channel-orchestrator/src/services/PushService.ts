import { logger } from '@userwhisperer/shared';

export interface PushMessage {
  to: string | string[]; // Device token(s) or topic
  title: string;
  body: string;
  data?: Record<string, any>;
  image?: string;
  icon?: string;
  badge?: number;
  sound?: string;
  click_action?: string;
  ttl?: number; // Time to live in seconds
  priority?: 'normal' | 'high';
  collapse_key?: string;
  delivery_receipt_requested?: boolean;
}

export interface PushDeliveryResult {
  message_id: string;
  status: 'sent' | 'failed' | 'partial';
  provider: string;
  sent_at: Date;
  success_count: number;
  failure_count: number;
  results?: Array<{
    token: string;
    status: 'success' | 'failed';
    error?: string;
  }>;
  error?: string;
}

export interface DeviceRegistration {
  user_id: string;
  device_token: string;
  platform: 'ios' | 'android' | 'web';
  app_version?: string;
  os_version?: string;
  device_model?: string;
  timezone?: string;
  language?: string;
  registered_at: Date;
  last_seen?: Date;
  active: boolean;
}

export class PushService {
  private firebaseServiceAccount: any;
  private apnsKey: string;
  private provider: 'firebase' | 'apns' | 'web-push' | 'mock';
  private deviceRegistry: Map<string, DeviceRegistration[]> = new Map();

  constructor(config: {
    firebase_service_account?: any;
    apns_key?: string;
    provider?: 'firebase' | 'apns' | 'web-push' | 'mock';
    web_push_vapid_keys?: {
      public_key: string;
      private_key: string;
      subject: string;
    };
  }) {
    this.firebaseServiceAccount = config.firebase_service_account;
    this.apnsKey = config.apns_key || 'mock-apns-key';
    this.provider = config.provider || 'mock';
    
    logger.info(`PushService initialized with provider: ${this.provider}`);
  }

  public async initialize(): Promise<void> {
    logger.info('Initializing PushService');
    try {
      switch (this.provider) {
        case 'firebase':
          // Initialize Firebase Admin SDK
          break;
        case 'apns':
          // Initialize APNS client
          break;
        case 'web-push':
          // Initialize Web Push client
          break;
        case 'mock':
        default:
          // No initialization needed for mock
          break;
      }
      logger.info('PushService initialized successfully');
    } catch (error) {
      logger.error('Failed to initialize PushService:', error);
      throw error;
    }
  }

  public async send(request: any): Promise<any> {
    const { content, user_id } = request;
    
    // Build push message from delivery request
    const pushMessage: PushMessage = {
      to: await this.getUserDeviceTokens(user_id),
      title: content.title || content.subject || 'User Whisperer',
      body: content.body || content.text || 'You have a new message',
      data: content.data || {},
      priority: 'high',
      ttl: 3600 // 1 hour
    };

    const result = await this.sendPushNotification(pushMessage);
    
    return {
      success: result.status === 'sent' || result.status === 'partial',
      channel: 'push',
      provider_message_id: result.message_id,
      delivered_at: result.sent_at,
      error: result.error,
      should_retry: result.status === 'failed' && !result.error?.includes('No active devices')
    };
  }

  private async getUserDeviceTokens(userId: string): Promise<string[]> {
    // In production, this would query the database for user device tokens
    // For now, return mock device tokens
    const userDevices = this.deviceRegistry.get(userId);
    if (userDevices && userDevices.length > 0) {
      return userDevices.filter(device => device.active).map(device => device.device_token);
    }
    
    // Mock device tokens for testing
    return [`mock_token_${userId}_ios`, `mock_token_${userId}_android`];
  }

  public async sendPushNotification(message: PushMessage): Promise<PushDeliveryResult> {
    logger.info(`Sending push notification via ${this.provider}`, {
      title: message.title,
      recipients: Array.isArray(message.to) ? message.to.length : 1,
      provider: this.provider
    });

    try {
      switch (this.provider) {
        case 'firebase':
          return await this.sendViaFirebase(message);
        case 'apns':
          return await this.sendViaAPNS(message);
        case 'web-push':
          return await this.sendViaWebPush(message);
        case 'mock':
        default:
          return await this.sendViaMock(message);
      }
    } catch (error) {
      logger.error('Push notification sending failed', {
        provider: this.provider,
        error: error instanceof Error ? error.message : String(error)
      });

      return {
        message_id: `failed_${Date.now()}`,
        status: 'failed',
        provider: this.provider,
        sent_at: new Date(),
        success_count: 0,
        failure_count: Array.isArray(message.to) ? message.to.length : 1,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  public async sendToUser(userId: string, message: Omit<PushMessage, 'to'>): Promise<PushDeliveryResult> {
    const userDevices = this.deviceRegistry.get(userId) || [];
    const activeDevices = userDevices.filter(device => device.active);

    if (activeDevices.length === 0) {
      logger.warn(`No active devices found for user ${userId}`);
      return {
        message_id: `no_devices_${Date.now()}`,
        status: 'failed',
        provider: this.provider,
        sent_at: new Date(),
        success_count: 0,
        failure_count: 0,
        error: 'No active devices found for user'
      };
    }

    const tokens = activeDevices.map(device => device.device_token);
    return await this.sendPushNotification({
      ...message,
      to: tokens
    });
  }

  public async sendToTopic(topic: string, message: Omit<PushMessage, 'to'>): Promise<PushDeliveryResult> {
    return await this.sendPushNotification({
      ...message,
      to: topic
    });
  }

  private async sendViaFirebase(message: PushMessage): Promise<PushDeliveryResult> {
    // Mock Firebase implementation - in production, use firebase-admin SDK
    logger.info('Firebase push notification sending (mock implementation)', {
      title: message.title,
      recipients: Array.isArray(message.to) ? message.to.length : 1
    });

    await new Promise(resolve => setTimeout(resolve, 150));

    const messageId = `fcm_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const recipients = Array.isArray(message.to) ? message.to : [message.to];

    // Mock some failures (2% failure rate per token)
    const results = recipients.map(token => ({
      token,
      status: Math.random() < 0.02 ? 'failed' as const : 'success' as const,
      error: Math.random() < 0.02 ? 'Invalid token' : undefined
    }));

    const successCount = results.filter(r => r.status === 'success').length;
    const failureCount = results.filter(r => r.status === 'failed').length;

    return {
      message_id: messageId,
      status: failureCount === 0 ? 'sent' : (successCount > 0 ? 'partial' : 'failed'),
      provider: 'firebase',
      sent_at: new Date(),
      success_count: successCount,
      failure_count: failureCount,
      results
    };
  }

  private async sendViaAPNS(message: PushMessage): Promise<PushDeliveryResult> {
    // Mock APNS implementation - in production, use node-apn
    logger.info('APNS push notification sending (mock implementation)', {
      title: message.title,
      recipients: Array.isArray(message.to) ? message.to.length : 1
    });

    await new Promise(resolve => setTimeout(resolve, 200));

    const messageId = `apns_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const recipients = Array.isArray(message.to) ? message.to : [message.to];

    const results = recipients.map(token => ({
      token,
      status: Math.random() < 0.03 ? 'failed' as const : 'success' as const,
      error: Math.random() < 0.03 ? 'BadDeviceToken' : undefined
    }));

    const successCount = results.filter(r => r.status === 'success').length;
    const failureCount = results.filter(r => r.status === 'failed').length;

    return {
      message_id: messageId,
      status: failureCount === 0 ? 'sent' : (successCount > 0 ? 'partial' : 'failed'),
      provider: 'apns',
      sent_at: new Date(),
      success_count: successCount,
      failure_count: failureCount,
      results
    };
  }

  private async sendViaWebPush(message: PushMessage): Promise<PushDeliveryResult> {
    // Mock Web Push implementation - in production, use web-push library
    logger.info('Web Push notification sending (mock implementation)', {
      title: message.title,
      recipients: Array.isArray(message.to) ? message.to.length : 1
    });

    await new Promise(resolve => setTimeout(resolve, 120));

    const messageId = `wp_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const recipients = Array.isArray(message.to) ? message.to : [message.to];

    const results = recipients.map(token => ({
      token,
      status: Math.random() < 0.05 ? 'failed' as const : 'success' as const,
      error: Math.random() < 0.05 ? 'Endpoint not found' : undefined
    }));

    const successCount = results.filter(r => r.status === 'success').length;
    const failureCount = results.filter(r => r.status === 'failed').length;

    return {
      message_id: messageId,
      status: failureCount === 0 ? 'sent' : (successCount > 0 ? 'partial' : 'failed'),
      provider: 'web-push',
      sent_at: new Date(),
      success_count: successCount,
      failure_count: failureCount,
      results
    };
  }

  private async sendViaMock(message: PushMessage): Promise<PushDeliveryResult> {
    logger.info('Mock push notification sending', {
      title: message.title,
      body: message.body,
      recipients: Array.isArray(message.to) ? message.to.length : 1
    });

    await new Promise(resolve => setTimeout(resolve, 50));

    const messageId = `mock_push_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const recipients = Array.isArray(message.to) ? message.to : [message.to];

    // Mock some failures (1% failure rate)
    const results = recipients.map(token => ({
      token,
      status: Math.random() < 0.01 ? 'failed' as const : 'success' as const,
      error: Math.random() < 0.01 ? 'Mock delivery failure' : undefined
    }));

    const successCount = results.filter(r => r.status === 'success').length;
    const failureCount = results.filter(r => r.status === 'failed').length;

    return {
      message_id: messageId,
      status: failureCount === 0 ? 'sent' : (successCount > 0 ? 'partial' : 'failed'),
      provider: 'mock',
      sent_at: new Date(),
      success_count: successCount,
      failure_count: failureCount,
      results
    };
  }

  public async registerDevice(registration: DeviceRegistration): Promise<void> {
    logger.info(`Registering device for user ${registration.user_id}`, {
      platform: registration.platform,
      device_token: registration.device_token.substring(0, 10) + '...'
    });

    const userDevices = this.deviceRegistry.get(registration.user_id) || [];
    
    // Remove existing registration for this token
    const filteredDevices = userDevices.filter(
      device => device.device_token !== registration.device_token
    );
    
    // Add new registration
    filteredDevices.push(registration);
    
    this.deviceRegistry.set(registration.user_id, filteredDevices);
  }

  public async unregisterDevice(userId: string, deviceToken: string): Promise<void> {
    logger.info(`Unregistering device for user ${userId}`);

    const userDevices = this.deviceRegistry.get(userId) || [];
    const filteredDevices = userDevices.filter(
      device => device.device_token !== deviceToken
    );
    
    this.deviceRegistry.set(userId, filteredDevices);
  }

  public async getUserDevices(userId: string): Promise<DeviceRegistration[]> {
    return this.deviceRegistry.get(userId) || [];
  }

  public async subscribeToTopic(deviceToken: string, topic: string): Promise<void> {
    logger.info(`Subscribing device to topic: ${topic}`, {
      device_token: deviceToken.substring(0, 10) + '...'
    });

    // Mock implementation - in production, use provider-specific topic subscription
    await new Promise(resolve => setTimeout(resolve, 100));
  }

  public async unsubscribeFromTopic(deviceToken: string, topic: string): Promise<void> {
    logger.info(`Unsubscribing device from topic: ${topic}`, {
      device_token: deviceToken.substring(0, 10) + '...'
    });

    await new Promise(resolve => setTimeout(resolve, 100));
  }

  public async getDeliveryStatus(messageId: string): Promise<{
    status: 'sent' | 'delivered' | 'failed';
    events: Array<{
      event: string;
      timestamp: Date;
      details?: Record<string, any>;
    }>;
  }> {
    logger.info(`Getting push notification delivery status for message: ${messageId}`);

    // Mock implementation
    return {
      status: 'delivered',
      events: [
        {
          event: 'sent',
          timestamp: new Date(Date.now() - 15000)
        },
        {
          event: 'delivered',
          timestamp: new Date(Date.now() - 5000)
        }
      ]
    };
  }

  public async healthCheck(): Promise<{ healthy: boolean; details: any }> {
    try {
      const totalDevices = Array.from(this.deviceRegistry.values())
        .reduce((total, devices) => total + devices.length, 0);

      return {
        healthy: true,
        details: {
          provider: this.provider,
          service_account_configured: !!this.firebaseServiceAccount,
          apns_key_configured: this.apnsKey !== 'mock-apns-key',
          registered_devices: totalDevices,
          active_users: this.deviceRegistry.size
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
