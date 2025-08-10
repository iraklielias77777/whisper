import { eventsCache } from '@userwhisperer/shared';
import crypto from 'crypto';
import { RawEvent } from './EventProcessor';

export class Deduplicator {
  private ttl: number = 86400; // 24 hours in seconds
  
  constructor(ttl?: number) {
    if (ttl) {
      this.ttl = ttl;
    }
  }
  
  async isDuplicate(event: RawEvent): Promise<boolean> {
    const hash = this.calculateHash(event);
    const key = `event:duplicate:${hash}`;
    
    // Check if key already exists
    const exists = await eventsCache.checkEventDuplicate(key);
    return exists;
  }
  
  async markProcessed(eventId: string, ttlSeconds?: number): Promise<void> {
    const ttl = ttlSeconds || this.ttl;
    await eventsCache.markEventProcessed(eventId, ttl);
  }
  
  private calculateHash(event: RawEvent): string {
    // Create deterministic hash from event properties
    const hashInput = JSON.stringify({
      user_id: event.user_id,
      event_type: event.event_type,
      timestamp: event.timestamp,
      // Include key properties that make event unique
      properties: this.extractKeyProperties(event.properties),
      context: this.extractKeyContext(event.context)
    });
    
    return crypto
      .createHash('sha256')
      .update(hashInput)
      .digest('hex');
  }
  
  private extractKeyProperties(properties: any): any {
    if (!properties) return {};
    
    // Extract only properties that determine uniqueness
    const keyProps: any = {};
    
    const importantKeys = [
      'product_id', 'order_id', 'transaction_id',
      'page_url', 'feature_name', 'error_code',
      'amount', 'currency', 'subscription_id',
      'session_id', 'api_endpoint', 'file_id',
      'message_id', 'notification_id'
    ];
    
    for (const key of importantKeys) {
      if (properties[key] !== undefined) {
        keyProps[key] = properties[key];
      }
    }
    
    return keyProps;
  }
  
  private extractKeyContext(context: any): any {
    if (!context) return {};
    
    // Extract key context that affects uniqueness
    const keyContext: any = {};
    
    const importantContextKeys = [
      'session_id', 'ip', 'device_type'
    ];
    
    for (const key of importantContextKeys) {
      if (context[key] !== undefined) {
        keyContext[key] = context[key];
      }
    }
    
    return keyContext;
  }
  
  // Get deduplication statistics
  async getStats(): Promise<{
    totalChecks: number;
    duplicatesFound: number;
    duplicateRate: number;
  }> {
    try {
      const totalChecks = await eventsCache.getCounter('deduplication.checks.total') || 0;
      const duplicatesFound = await eventsCache.getCounter('deduplication.duplicates.found') || 0;
      
      return {
        totalChecks,
        duplicatesFound,
        duplicateRate: totalChecks > 0 ? duplicatesFound / totalChecks : 0
      };
    } catch (error) {
      return {
        totalChecks: 0,
        duplicatesFound: 0,
        duplicateRate: 0
      };
    }
  }
  
  // Update statistics
  async updateStats(isDuplicate: boolean): Promise<void> {
    try {
      await eventsCache.incrementCounter('deduplication.checks.total');
      if (isDuplicate) {
        await eventsCache.incrementCounter('deduplication.duplicates.found');
      }
    } catch (error) {
      // Ignore metrics errors
    }
  }
  
  // Clear all deduplication data (for testing/maintenance)
  async clearAll(): Promise<void> {
    // This would need to be implemented with a pattern-based deletion
    // For now, just a placeholder
    console.warn('clearAll() not implemented - would require Redis SCAN and DEL operations');
  }
} 