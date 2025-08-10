import { database } from '@userwhisperer/shared';
import * as geoip from 'geoip-lite';
import { UAParser } from 'ua-parser-js';
import { RawEvent, EnrichedEvent } from './EventProcessor';

export interface UserContext {
  is_new_user: boolean;
  user_id: string;
  external_user_id?: string;
  email?: string;
  name?: string;
  lifecycle_stage?: string;
  engagement_score?: number;
  churn_risk_score?: number;
  ltv_prediction?: number;
  created_at?: Date;
  subscription_status?: string;
  subscription_plan?: string;
  last_active_at?: Date;
  days_since_signup?: number;
  days_since_active?: number;
}

export class EventEnricher {
  private userCache: Map<string, UserContext>;
  private cacheExpiry: Map<string, number>;
  private cacheTtl: number = 300000; // 5 minutes
  
  constructor() {
    this.userCache = new Map();
    this.cacheExpiry = new Map();
    
    // Clean up expired cache entries every minute
    setInterval(() => this.cleanupCache(), 60000);
  }
  
  async enrich(event: RawEvent): Promise<EnrichedEvent> {
    const enrichedEvent: EnrichedEvent = {
      ...event,
      event_id: event.event_id!,
      app_id: event.app_id!,
      enrichment: {
        timestamp: new Date().toISOString()
      },
      category: 'other' // Will be set by categorizeEvent
    };
    
    // Enrich with user context
    try {
      const userContext = await this.getUserContext(event.user_id);
      enrichedEvent.user_context = userContext;
    } catch (error) {
      console.warn('Failed to enrich with user context:', error);
    }
    
    // Enrich with geo data from IP
    if (event.context?.ip) {
      try {
        const geo = geoip.lookup(event.context.ip);
        if (geo) {
          enrichedEvent.geo = {
            country: geo.country,
            region: geo.region,
            city: geo.city,
            timezone: geo.timezone,
            coordinates: geo.ll
          };
        }
      } catch (error) {
        console.warn('Failed to enrich with geo data:', error);
      }
    }
    
    // Parse user agent
    if (event.context?.user_agent) {
      try {
        const parser = new UAParser(event.context.user_agent);
        const result = parser.getResult();
        
        enrichedEvent.device = {
          browser: result.browser.name || 'unknown',
          browser_version: result.browser.version || 'unknown',
          os: result.os.name || 'unknown',
          os_version: result.os.version || 'unknown',
          device_type: this.normalizeDeviceType(result.device.type) || event.context.device_type || 'desktop',
          device_vendor: result.device.vendor,
          device_model: result.device.model
        };
      } catch (error) {
        console.warn('Failed to parse user agent:', error);
        enrichedEvent.device = {
          browser: 'unknown',
          browser_version: 'unknown',
          os: 'unknown',
          os_version: 'unknown',
          device_type: event.context?.device_type || 'unknown'
        };
      }
    }
    
    // Calculate session metrics
    if (event.context?.session_id) {
      try {
        const sessionMetrics = await this.getSessionMetrics(event.context.session_id);
        enrichedEvent.session_metrics = sessionMetrics;
      } catch (error) {
        console.warn('Failed to get session metrics:', error);
      }
    }
    
    // Add event category for routing
    enrichedEvent.category = this.categorizeEvent(event);
    
    // Add processing metadata
    if (!enrichedEvent.metadata) {
      enrichedEvent.metadata = {};
    }
    enrichedEvent.metadata.processed_at = new Date().toISOString();
    
    // Add enrichment data with required timestamp
    if (!enrichedEvent.enrichment) {
      enrichedEvent.enrichment = {
        timestamp: new Date().toISOString()
      };
    }
    enrichedEvent.enrichment.timezone_offset = this.getTimezoneOffset(enrichedEvent.geo?.timezone);
    enrichedEvent.enrichment.local_timestamp = this.convertToLocalTime(event.timestamp, enrichedEvent.geo?.timezone);
    
    return enrichedEvent;
  }
  
  private async getUserContext(userId: string): Promise<UserContext> {
    // Check cache first
    const cached = this.userCache.get(userId);
    const expiry = this.cacheExpiry.get(userId);
    
    if (cached && expiry && Date.now() < expiry) {
      return cached;
    }
    
    try {
      const query = `
        SELECT 
          external_user_id,
          email,
          name,
          lifecycle_stage,
          engagement_score,
          churn_risk_score,
          ltv_prediction,
          created_at,
          subscription_status,
          subscription_plan,
          last_active_at
        FROM user_profiles
        WHERE external_user_id = $1
      `;
      
      const result = await database.query(query, [userId]);
      
      let userContext: UserContext;
      
      if (result.rows.length === 0) {
        // New user
        userContext = {
          is_new_user: true,
          user_id: userId
        };
      } else {
        const user = result.rows[0];
        userContext = {
          is_new_user: false,
          user_id: userId,
          ...user,
          days_since_signup: user.created_at ? this.daysSince(user.created_at) : null,
          days_since_active: user.last_active_at ? this.daysSince(user.last_active_at) : null
        };
      }
      
      // Cache for 5 minutes
      this.userCache.set(userId, userContext);
      this.cacheExpiry.set(userId, Date.now() + this.cacheTtl);
      
      return userContext;
      
    } catch (error) {
      console.error('Error fetching user context:', error);
      // Return minimal context on error
      return {
        is_new_user: false, // Assume existing user on error
        user_id: userId
      };
    }
  }
  
  private async getSessionMetrics(sessionId: string): Promise<any> {
    try {
      // Get session events from last 24 hours
      const query = `
        SELECT 
          COUNT(*) as event_count,
          MIN(timestamp) as session_start,
          MAX(timestamp) as session_end,
          COUNT(DISTINCT event_type) as unique_event_types,
          array_agg(DISTINCT event_type) as event_types
        FROM events.raw_events 
        WHERE session_id = $1 
          AND timestamp > NOW() - INTERVAL '24 hours'
      `;
      
      const result = await database.query(query, [sessionId]);
      
      if (result.rows.length > 0) {
        const row = result.rows[0];
        const sessionStart = new Date(row.session_start);
        const sessionEnd = new Date(row.session_end);
        const duration = sessionEnd.getTime() - sessionStart.getTime();
        
        return {
          event_count: parseInt(row.event_count),
          session_duration: Math.max(duration, 0), // milliseconds
          unique_event_types: parseInt(row.unique_event_types),
          event_types: row.event_types || [],
          session_start: row.session_start,
          session_end: row.session_end,
          is_active_session: duration < 1800000 // 30 minutes
        };
      }
      
      return {
        event_count: 0,
        session_duration: 0,
        unique_event_types: 0,
        event_types: [],
        is_active_session: true
      };
      
    } catch (error) {
      console.error('Error getting session metrics:', error);
      return null;
    }
  }
  
  private categorizeEvent(event: RawEvent): string {
    const eventType = event.event_type;
    
    // Monetization events
    if (['purchase', 'subscription_started', 'subscription_upgraded',
         'payment_method_added', 'trial_started', 'trial_converted',
         'payment_succeeded', 'subscription_cancelled'].includes(eventType)) {
      return 'monetization';
    }
    
    // Engagement events
    if (['feature_used', 'page_viewed', 'button_clicked',
         'content_viewed', 'search_performed', 'form_submitted',
         'item_shared', 'item_favorited', 'api_call_made'].includes(eventType)) {
      return 'engagement';
    }
    
    // Lifecycle events
    if (['user_signup', 'user_activated', 'user_login', 'user_logout',
         'subscription_cancelled', 'account_deleted', 'user_churned'].includes(eventType)) {
      return 'lifecycle';
    }
    
    // Support events
    if (['error_encountered', 'support_ticket_created', 
         'feedback_submitted', 'help_viewed', 'notification_clicked'].includes(eventType)) {
      return 'support';
    }
    
    // System events
    if (['session_started', 'session_ended', 'export_created',
         'import_completed', 'notification_sent'].includes(eventType)) {
      return 'system';
    }
    
    return 'other';
  }
  
  private normalizeDeviceType(deviceType?: string): string {
    if (!deviceType) return 'desktop';
    
    const normalized = deviceType.toLowerCase();
    
    if (normalized.includes('mobile') || normalized.includes('phone')) {
      return 'mobile';
    }
    
    if (normalized.includes('tablet') || normalized.includes('pad')) {
      return 'tablet';
    }
    
    return 'desktop';
  }
  
  private daysSince(date: Date): number {
    const diff = Date.now() - new Date(date).getTime();
    return Math.floor(diff / (1000 * 60 * 60 * 24));
  }
  
  private getTimezoneOffset(timezone?: string): number {
    if (!timezone) return 0;
    
    try {
      const now = new Date();
      const utc = new Date(now.getTime() + (now.getTimezoneOffset() * 60000));
      const targetTime = new Date(utc.toLocaleString('en-US', { timeZone: timezone }));
      return (targetTime.getTime() - utc.getTime()) / (1000 * 60); // minutes
    } catch (error) {
      return 0;
    }
  }
  
  private convertToLocalTime(utcTimestamp: string, timezone?: string): string {
    if (!timezone) return utcTimestamp;
    
    try {
      const date = new Date(utcTimestamp);
      return date.toLocaleString('en-US', { 
        timeZone: timezone,
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false
      });
    } catch (error) {
      return utcTimestamp;
    }
  }
  
  private cleanupCache(): void {
    const now = Date.now();
    
    for (const [userId, expiry] of this.cacheExpiry.entries()) {
      if (now >= expiry) {
        this.userCache.delete(userId);
        this.cacheExpiry.delete(userId);
      }
    }
  }
  
  // Get cache statistics
  getCacheStats(): { 
    size: number; 
    hitRate: number; 
    memoryUsage: number;
  } {
    const size = this.userCache.size;
    
    // Estimate memory usage (rough calculation)
    let memoryUsage = 0;
    for (const [key, value] of this.userCache.entries()) {
      memoryUsage += key.length * 2; // UTF-16 characters
      memoryUsage += JSON.stringify(value).length * 2;
    }
    
    return {
      size,
      hitRate: 0, // Would need to track hits/misses to calculate
      memoryUsage: Math.round(memoryUsage / 1024) // KB
    };
  }
  
  // Clear cache manually
  clearCache(): void {
    this.userCache.clear();
    this.cacheExpiry.clear();
  }
} 