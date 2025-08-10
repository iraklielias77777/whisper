import { PubSub, Message, Subscription, Topic } from '@google-cloud/pubsub';
import { createServiceLogger } from './logger';

export interface MessageHandler<T = any> {
  (data: T, message: Message): Promise<void>;
}

export interface QueueConfig {
  projectId?: string;
  emulatorHost?: string;
  topicPrefix?: string;
  subscriptionPrefix?: string;
  deadLetterPolicy?: {
    maxDeliveryAttempts: number;
    deadLetterTopic?: string;
  };
}

export interface PublishOptions {
  attributes?: Record<string, string>;
  orderingKey?: string;
}

export interface SubscriptionOptions {
  ackDeadlineSeconds?: number;
  maxExtensionSeconds?: number;
  maxMessages?: number;
  allowExcessMessages?: boolean;
  enableMessageOrdering?: boolean;
}

export class MessageQueue {
  private pubsub: PubSub;
  private config: QueueConfig;
  private topics: Map<string, Topic> = new Map();
  private subscriptions: Map<string, Subscription> = new Map();
  private handlers: Map<string, MessageHandler> = new Map();
  private logger = createServiceLogger('message-queue');

  constructor(config: QueueConfig = {}) {
    this.config = {
      projectId: config.projectId || process.env.GOOGLE_CLOUD_PROJECT || 'user-whisperer-dev',
      emulatorHost: config.emulatorHost || process.env.PUBSUB_EMULATOR_HOST,
      topicPrefix: config.topicPrefix || 'uw',
      subscriptionPrefix: config.subscriptionPrefix || 'uw',
      ...config
    };

    // Configure Pub/Sub client
    const clientConfig: any = {
      projectId: this.config.projectId
    };

    // Use emulator if configured
    if (this.config.emulatorHost) {
      process.env.PUBSUB_EMULATOR_HOST = this.config.emulatorHost;
      this.logger.info('Using Pub/Sub emulator', { host: this.config.emulatorHost });
    }

    this.pubsub = new PubSub(clientConfig);
  }

  async initialize(): Promise<void> {
    this.logger.info('Initializing Message Queue...', { 
      projectId: this.config.projectId,
      emulator: !!this.config.emulatorHost 
    });

    try {
      // Test connection
      await this.pubsub.getTopics();
      this.logger.info('Message Queue initialized successfully');
    } catch (error) {
      this.logger.error('Failed to initialize Message Queue', { error });
      throw error;
    }
  }

  async createTopic(topicName: string): Promise<Topic> {
    const fullTopicName = `${this.config.topicPrefix}-${topicName}`;
    
    if (this.topics.has(fullTopicName)) {
      return this.topics.get(fullTopicName)!;
    }

    try {
      const topic = this.pubsub.topic(fullTopicName);
      const [exists] = await topic.exists();

      if (!exists) {
        await topic.create();
        this.logger.info('Created topic', { topic: fullTopicName });
      }

      this.topics.set(fullTopicName, topic);
      return topic;
    } catch (error) {
      this.logger.error('Failed to create topic', { topic: fullTopicName, error });
      throw error;
    }
  }

  async publish<T>(topicName: string, data: T, options: PublishOptions = {}): Promise<string> {
    try {
      const topic = await this.createTopic(topicName);
      const message: any = {
        data: Buffer.from(JSON.stringify(data)),
        attributes: {
          timestamp: new Date().toISOString(),
          source: 'user-whisperer',
          ...options.attributes
        }
      };

      if (options.orderingKey) {
        message.orderingKey = options.orderingKey;
      }

      const messageId = await topic.publishMessage(message);
      
      this.logger.debug('Message published', { 
        topic: topicName, 
        messageId,
        dataSize: message.data.length 
      });

      return messageId;
    } catch (error) {
      this.logger.error('Failed to publish message', { topic: topicName, error });
      throw error;
    }
  }

  async subscribe<T>(
    topicName: string, 
    subscriptionName: string,
    handler: MessageHandler<T>,
    options: SubscriptionOptions = {}
  ): Promise<void> {
    const fullTopicName = `${this.config.topicPrefix}-${topicName}`;
    const fullSubscriptionName = `${this.config.subscriptionPrefix}-${subscriptionName}`;

    try {
      const topic = await this.createTopic(topicName);
      
      let subscription = this.pubsub.subscription(fullSubscriptionName);
      const [subscriptionExists] = await subscription.exists();

      if (!subscriptionExists) {
        const subscriptionConfig: any = {
          ackDeadlineSeconds: options.ackDeadlineSeconds || 60,
          messageRetentionDuration: { seconds: 604800 }, // 7 days
          enableMessageOrdering: options.enableMessageOrdering || false
        };

        if (this.config.deadLetterPolicy) {
          subscriptionConfig.deadLetterPolicy = this.config.deadLetterPolicy;
        }

        await topic.createSubscription(fullSubscriptionName, subscriptionConfig);
        this.logger.info('Created subscription', { 
          topic: fullTopicName, 
          subscription: fullSubscriptionName 
        });
      }

      // Note: Subscription options are configured during creation above

      // Set up message handler
      subscription.on('message', async (message: Message) => {
        await this.handleMessage(message, handler, subscriptionName);
      });

      subscription.on('error', (error: any) => {
        this.logger.error('Subscription error', { 
          subscription: fullSubscriptionName, 
          error 
        });
      });

      this.subscriptions.set(fullSubscriptionName, subscription);
      this.handlers.set(fullSubscriptionName, handler);

      this.logger.info('Subscription active', { 
        topic: fullTopicName, 
        subscription: fullSubscriptionName 
      });

    } catch (error) {
      this.logger.error('Failed to create subscription', { 
        topic: topicName, 
        subscription: subscriptionName, 
        error 
      });
      throw error;
    }
  }

  private async handleMessage<T>(
    message: Message, 
    handler: MessageHandler<T>,
    subscriptionName: string
  ): Promise<void> {
    const startTime = Date.now();
    let data: T;

    try {
      // Parse message data
      data = JSON.parse(message.data.toString());
      
      this.logger.debug('Processing message', {
        messageId: message.id,
        subscription: subscriptionName,
        attributes: message.attributes
      });

      // Call handler
      await handler(data, message);

      // Acknowledge message
      message.ack();

      const processingTime = Date.now() - startTime;
      this.logger.debug('Message processed successfully', {
        messageId: message.id,
        processingTime: `${processingTime}ms`
      });

    } catch (error) {
      const processingTime = Date.now() - startTime;
      
      this.logger.error('Message processing failed', {
        messageId: message.id,
        subscription: subscriptionName,
        processingTime: `${processingTime}ms`,
        error: error instanceof Error ? error.message : String(error),
        deliveryAttempt: message.deliveryAttempt
      });

      // Nack message to retry or send to dead letter queue
      message.nack();
    }
  }

  async publishBatch<T>(topicName: string, messages: T[], options: PublishOptions = {}): Promise<string[]> {
    try {
      const topic = await this.createTopic(topicName);
      const publishPromises = messages.map(data => 
        this.publish(topicName, data, options)
      );

      const messageIds = await Promise.all(publishPromises);
      
      this.logger.info('Batch published', { 
        topic: topicName, 
        count: messages.length,
        messageIds: messageIds.length 
      });

      return messageIds;
    } catch (error) {
      this.logger.error('Failed to publish batch', { topic: topicName, error });
      throw error;
    }
  }

  async getSubscriptionInfo(subscriptionName: string): Promise<any> {
    const fullSubscriptionName = `${this.config.subscriptionPrefix}-${subscriptionName}`;
    
    try {
      const subscription = this.pubsub.subscription(fullSubscriptionName);
      const [metadata] = await subscription.getMetadata();
      
      return {
        name: metadata.name,
        topic: metadata.topic,
        ackDeadlineSeconds: metadata.ackDeadlineSeconds,
        messageRetentionDuration: metadata.messageRetentionDuration,
        deadLetterPolicy: metadata.deadLetterPolicy
      };
    } catch (error) {
      this.logger.error('Failed to get subscription info', { subscription: subscriptionName, error });
      throw error;
    }
  }

  async getTopicInfo(topicName: string): Promise<any> {
    const fullTopicName = `${this.config.topicPrefix}-${topicName}`;
    
    try {
      const topic = this.pubsub.topic(fullTopicName);
      const [metadata] = await topic.getMetadata();
      
      return {
        name: metadata.name,
        messageStoragePolicy: metadata.messageStoragePolicy,
        schemaSettings: metadata.schemaSettings
      };
    } catch (error) {
      this.logger.error('Failed to get topic info', { topic: topicName, error });
      throw error;
    }
  }

  async close(): Promise<void> {
    this.logger.info('Closing Message Queue connections...');

    // Close all subscriptions
    const closePromises = Array.from(this.subscriptions.values()).map(subscription => {
      return subscription.close();
    });

    await Promise.all(closePromises);

    // Clear internal state
    this.topics.clear();
    this.subscriptions.clear();
    this.handlers.clear();

    this.logger.info('Message Queue connections closed');
  }

  // Health check
  async healthCheck(): Promise<boolean> {
    try {
      await this.pubsub.getTopics({ pageSize: 1 });
      return true;
    } catch (error) {
      this.logger.error('Message Queue health check failed', { error });
      return false;
    }
  }

  // Get queue statistics
  getStats(): {
    activeTopics: number;
    activeSubscriptions: number;
    totalHandlers: number;
  } {
    return {
      activeTopics: this.topics.size,
      activeSubscriptions: this.subscriptions.size,
      totalHandlers: this.handlers.size
    };
  }
}

// Predefined topic names for the platform
export const TOPICS = {
  USER_EVENTS: 'user-events',
  BEHAVIORAL_ANALYSIS: 'behavioral-analysis',
  INTERVENTION_DECISIONS: 'intervention-decisions',
  CONTENT_GENERATION: 'content-generation',
  MESSAGE_DELIVERY: 'message-delivery',
  SYSTEM_EVENTS: 'system-events'
} as const;

// Predefined subscription names
export const SUBSCRIPTIONS = {
  EVENT_PROCESSOR: 'event-processor',
  BEHAVIORAL_ANALYZER: 'behavioral-analyzer',
  DECISION_MAKER: 'decision-maker',
  CONTENT_GENERATOR: 'content-generator',
  MESSAGE_SENDER: 'message-sender',
  SYSTEM_MONITOR: 'system-monitor'
} as const;
