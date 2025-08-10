/**
 * User Whisperer JavaScript SDK
 * Production-ready client-side analytics and engagement platform integration
 */

class UserWhisperer {
  constructor(config = {}) {
    // Configuration validation
    if (!config.apiKey) {
      throw new Error('UserWhisperer: apiKey is required');
    }
    if (!config.appId) {
      throw new Error('UserWhisperer: appId is required');
    }

    // Configuration
    this.apiKey = config.apiKey;
    this.appId = config.appId;
    this.endpoint = config.endpoint || 'https://api.userwhisperer.ai';
    this.debug = config.debug || false;
    this.batchSize = config.batchSize || 100;
    this.flushInterval = config.flushInterval || 5000;
    this.retryLimit = config.retryLimit || 3;
    this.timeout = config.timeout || 10000;
    this.enableCompression = config.enableCompression !== false;
    this.disableLocalStorage = config.disableLocalStorage || false;
    
    // State
    this.eventQueue = [];
    this.userId = this.getPersistedUserId();
    this.sessionId = this.generateSessionId();
    this.flushTimer = null;
    this.retryQueue = [];
    this.isOnline = typeof navigator !== 'undefined' ? navigator.onLine : true;
    this.pageLoadTime = Date.now();
    
    // Feature flags
    this.features = {
      sessionTracking: true,
      pageTracking: true,
      errorTracking: true,
      performanceTracking: config.performanceTracking !== false,
      autoTrack: config.autoTrack !== false
    };
    
    // Initialize
    this.initialize();
  }
  
  initialize() {
    this.log('Initializing UserWhisperer SDK...');
    
    // Set up automatic flushing
    this.startFlushTimer();
    
    // Handle browser events
    if (typeof window !== 'undefined') {
      // Page unload - ensure data is sent
      window.addEventListener('beforeunload', () => {
        this.flush(true);
      });
      
      // Page visibility changes
      document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'hidden') {
          this.flush();
        } else if (document.visibilityState === 'visible') {
          this.track('$page_focus');
        }
      });
      
      // Online/offline events
      window.addEventListener('online', () => {
        this.isOnline = true;
        this.log('Connection restored, processing retry queue');
        this.processRetryQueue();
      });
      
      window.addEventListener('offline', () => {
        this.isOnline = false;
        this.log('Connection lost, queuing events for retry');
      });
      
      // Performance tracking
      if (this.features.performanceTracking) {
        this.setupPerformanceTracking();
      }
      
      // Auto-tracking
      if (this.features.autoTrack) {
        this.setupAutoTracking();
      }
      
      // Error tracking
      if (this.features.errorTracking) {
        this.setupErrorTracking();
      }
    }
    
    // Load persisted events
    this.loadPersistedEvents();
    
    // Track initialization
    this.track('$sdk_initialized', {
      sdk_version: '1.0.0',
      config: {
        batchSize: this.batchSize,
        flushInterval: this.flushInterval,
        endpoint: this.endpoint
      }
    });
    
    this.log('UserWhisperer initialized successfully');
  }
  
  // ========== Core Tracking Methods ==========
  
  track(eventType, properties = {}, options = {}) {
    if (!this.validateEventType(eventType)) {
      this.error(`Invalid event type: ${eventType}`);
      return null;
    }
    
    const event = {
      id: this.generateEventId(),
      app_id: this.appId,
      user_id: this.userId,
      session_id: this.sessionId,
      event_type: eventType,
      properties: this.sanitizeProperties(properties),
      context: this.getContext(),
      timestamp: new Date().toISOString(),
      ...options
    };
    
    this.addToQueue(event);
    
    // Auto-flush if batch size reached
    if (this.eventQueue.length >= this.batchSize) {
      this.flush();
    }
    
    this.log('Event tracked:', eventType, properties);
    return event.id;
  }
  
  identify(userId, traits = {}) {
    if (!userId) {
      this.error('identify: userId is required');
      return;
    }
    
    const previousUserId = this.userId;
    this.userId = userId;
    this.persistUserId(userId);
    
    // Send identify event
    this.track('$identify', {
      ...traits,
      $user_id: userId,
      $previous_id: previousUserId
    });
    
    this.log(`User identified: ${userId}`);
    return userId;
  }
  
  alias(newUserId) {
    if (!newUserId) {
      this.error('alias: newUserId is required');
      return;
    }
    
    const previousUserId = this.userId;
    
    this.track('$alias', {
      previous_id: previousUserId,
      new_id: newUserId
    });
    
    this.userId = newUserId;
    this.persistUserId(newUserId);
    
    this.log(`User aliased: ${previousUserId} -> ${newUserId}`);
  }
  
  group(groupId, traits = {}) {
    if (!groupId) {
      this.error('group: groupId is required');
      return;
    }
    
    this.track('$group', {
      group_id: groupId,
      ...traits
    });
    
    this.log(`User grouped: ${groupId}`);
  }
  
  page(name, properties = {}) {
    const pageData = {
      page_name: name,
      ...properties,
      ...this.getPageProperties()
    };
    
    this.track('$page_view', pageData);
    
    this.log(`Page viewed: ${name}`);
  }
  
  screen(name, properties = {}) {
    this.track('$screen_view', {
      screen_name: name,
      ...properties
    });
    
    this.log(`Screen viewed: ${name}`);
  }
  
  // ========== Queue Management ==========
  
  addToQueue(event) {
    // Add event to queue
    this.eventQueue.push(event);
    
    // Persist to localStorage for reliability
    this.persistQueue();
    
    // Emit event for listeners
    this.emit('event_queued', event);
  }
  
  async flush(synchronous = false) {
    if (this.eventQueue.length === 0) {
      return Promise.resolve();
    }
    
    // Get events to send
    const events = [...this.eventQueue];
    this.eventQueue = [];
    
    // Clear persisted queue
    this.clearPersistedQueue();
    
    this.log(`Flushing ${events.length} events...`);
    
    if (synchronous) {
      // Use sendBeacon for reliability on page unload
      return this.sendBeacon(events);
    } else {
      // Normal async send
      return this.sendEvents(events);
    }
  }
  
  async sendEvents(events, retryCount = 0) {
    if (!this.isOnline) {
      this.log('Offline, adding events to retry queue');
      this.addToRetryQueue(events);
      return;
    }
    
    const startTime = Date.now();
    
    try {
      const response = await this.request('/v1/events/batch', {
        method: 'POST',
        body: JSON.stringify({ 
          events,
          sdk_version: '1.0.0',
          client_timestamp: new Date().toISOString()
        }),
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.apiKey}`,
          'X-App-ID': this.appId,
          'X-SDK-Version': '1.0.0'
        }
      });
      
      const responseTime = Date.now() - startTime;
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }
      
      const result = await response.json();
      
      // Handle partial failures
      if (result.failed && result.failed.length > 0) {
        this.handleFailedEvents(result.failed);
        this.log(`Partially successful: ${events.length - result.failed.length}/${events.length} events sent`);
      } else {
        this.log(`Successfully sent ${events.length} events in ${responseTime}ms`);
      }
      
      // Track send performance
      this.track('$events_sent', {
        count: events.length,
        response_time: responseTime,
        retry_count: retryCount
      });
      
      this.emit('events_sent', { events, result });
      
    } catch (error) {
      this.error('Failed to send events:', error);
      
      if (retryCount < this.retryLimit) {
        // Exponential backoff with jitter
        const baseDelay = Math.min(1000 * Math.pow(2, retryCount), 30000);
        const jitter = Math.random() * 1000;
        const delay = baseDelay + jitter;
        
        this.log(`Retrying in ${delay}ms (attempt ${retryCount + 1}/${this.retryLimit})`);
        
        setTimeout(() => {
          this.sendEvents(events, retryCount + 1);
        }, delay);
      } else {
        this.log('Max retries reached, adding to retry queue');
        this.addToRetryQueue(events);
      }
      
      this.emit('events_failed', { events, error, retryCount });
    }
  }
  
  sendBeacon(events) {
    if (typeof navigator !== 'undefined' && typeof navigator.sendBeacon === 'function') {
      const url = `${this.endpoint}/v1/events/batch`;
      const data = JSON.stringify({
        events,
        api_key: this.apiKey,
        app_id: this.appId,
        sdk_version: '1.0.0'
      });
      
      const success = navigator.sendBeacon(url, data);
      
      if (success) {
        this.log(`Sent ${events.length} events via beacon`);
      } else {
        this.log('Beacon send failed, falling back to sync XHR');
        this.sendSynchronous(events);
      }
    } else {
      // Fallback to synchronous XHR
      this.sendSynchronous(events);
    }
  }
  
  sendSynchronous(events) {
    try {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', `${this.endpoint}/v1/events/batch`, false);
      xhr.setRequestHeader('Content-Type', 'application/json');
      xhr.setRequestHeader('Authorization', `Bearer ${this.apiKey}`);
      xhr.setRequestHeader('X-App-ID', this.appId);
      
      const data = JSON.stringify({
        events,
        sdk_version: '1.0.0'
      });
      
      xhr.send(data);
      
      if (xhr.status >= 200 && xhr.status < 300) {
        this.log(`Synchronously sent ${events.length} events`);
      } else {
        this.error(`Synchronous send failed: ${xhr.status}`);
      }
    } catch (error) {
      this.error('Synchronous send error:', error);
    }
  }
  
  handleFailedEvents(failedEvents) {
    // Log failed events for debugging
    this.error('Failed events:', failedEvents);
    
    // Could implement specific retry logic for failed events
    // For now, just log them
  }
  
  // ========== Context Collection ==========
  
  getContext() {
    const context = {
      page: this.getPageContext(),
      device: this.getDeviceContext(),
      session: this.getSessionContext(),
      library: {
        name: 'userwhisperer-js',
        version: '1.0.0'
      }
    };
    
    // Add UTM parameters if present
    const utm = this.getUTMParameters();
    if (Object.keys(utm).length > 0) {
      context.campaign = utm;
    }
    
    // Add performance data if available
    if (this.features.performanceTracking) {
      context.performance = this.getPerformanceContext();
    }
    
    return context;
  }
  
  getPageContext() {
    if (typeof window === 'undefined') {
      return {};
    }
    
    return {
      url: window.location.href,
      path: window.location.pathname,
      referrer: document.referrer,
      search: window.location.search,
      hash: window.location.hash,
      title: document.title,
      host: window.location.host,
      protocol: window.location.protocol
    };
  }
  
  getDeviceContext() {
    if (typeof window === 'undefined') {
      return {};
    }
    
    const context = {
      user_agent: navigator.userAgent,
      language: navigator.language,
      languages: navigator.languages,
      platform: navigator.platform,
      cookie_enabled: navigator.cookieEnabled,
      java_enabled: navigator.javaEnabled ? navigator.javaEnabled() : false,
      online: navigator.onLine
    };
    
    // Screen information
    if (window.screen) {
      context.screen_width = window.screen.width;
      context.screen_height = window.screen.height;
      context.screen_color_depth = window.screen.colorDepth;
      context.screen_pixel_depth = window.screen.pixelDepth;
    }
    
    // Viewport information
    context.viewport_width = window.innerWidth;
    context.viewport_height = window.innerHeight;
    
    // Timezone
    try {
      context.timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
      context.timezone_offset = new Date().getTimezoneOffset();
    } catch (e) {
      // Timezone detection failed
    }
    
    // Connection information (if available)
    if (navigator.connection) {
      context.connection = {
        effective_type: navigator.connection.effectiveType,
        downlink: navigator.connection.downlink,
        rtt: navigator.connection.rtt,
        save_data: navigator.connection.saveData
      };
    }
    
    return context;
  }
  
  getSessionContext() {
    return {
      session_id: this.sessionId,
      session_start: this.getStoredSessionId()?.started_at,
      page_load_time: this.pageLoadTime,
      session_duration: Date.now() - (this.getStoredSessionId()?.started_at || Date.now())
    };
  }
  
  getPerformanceContext() {
    if (typeof window === 'undefined' || !window.performance) {
      return {};
    }
    
    const perf = window.performance;
    const timing = perf.timing;
    
    if (!timing) {
      return {};
    }
    
    return {
      dns_time: timing.domainLookupEnd - timing.domainLookupStart,
      connect_time: timing.connectEnd - timing.connectStart,
      request_time: timing.responseStart - timing.requestStart,
      response_time: timing.responseEnd - timing.responseStart,
      dom_load_time: timing.domContentLoadedEventEnd - timing.navigationStart,
      page_load_time: timing.loadEventEnd - timing.navigationStart,
      redirect_time: timing.redirectEnd - timing.redirectStart
    };
  }
  
  getPageProperties() {
    if (typeof window === 'undefined') {
      return {};
    }
    
    return {
      url: window.location.href,
      path: window.location.pathname,
      referrer: document.referrer,
      search: window.location.search,
      title: document.title
    };
  }
  
  getUTMParameters() {
    if (typeof window === 'undefined') {
      return {};
    }
    
    const params = new URLSearchParams(window.location.search);
    const utm = {};
    
    ['utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content'].forEach(key => {
      const value = params.get(key);
      if (value) {
        utm[key] = value;
      }
    });
    
    return utm;
  }
  
  // ========== Auto-Tracking Setup ==========
  
  setupAutoTracking() {
    if (typeof window === 'undefined') return;
    
    // Track page loads automatically
    if (this.features.pageTracking) {
      // Initial page view
      this.page(document.title);
      
      // Track hash changes
      window.addEventListener('hashchange', () => {
        this.page(document.title);
      });
      
      // Track pushstate/popstate for SPA navigation
      const originalPushState = history.pushState;
      const originalReplaceState = history.replaceState;
      
      history.pushState = (...args) => {
        originalPushState.apply(history, args);
        setTimeout(() => this.page(document.title), 0);
      };
      
      history.replaceState = (...args) => {
        originalReplaceState.apply(history, args);
        setTimeout(() => this.page(document.title), 0);
      };
      
      window.addEventListener('popstate', () => {
        setTimeout(() => this.page(document.title), 0);
      });
    }
    
    // Track clicks on elements with data-track attributes
    document.addEventListener('click', (event) => {
      const element = event.target.closest('[data-track]');
      if (element) {
        const eventType = element.getAttribute('data-track');
        const properties = {};
        
        // Collect data-track-* attributes
        Array.from(element.attributes).forEach(attr => {
          if (attr.name.startsWith('data-track-')) {
            const key = attr.name.replace('data-track-', '');
            properties[key] = attr.value;
          }
        });
        
        // Add element information
        properties.element_tag = element.tagName.toLowerCase();
        properties.element_text = element.textContent?.trim();
        properties.element_id = element.id;
        properties.element_class = element.className;
        
        this.track(eventType || 'element_clicked', properties);
      }
    });
    
    // Track form submissions
    document.addEventListener('submit', (event) => {
      const form = event.target;
      if (form.tagName === 'FORM') {
        this.track('form_submitted', {
          form_id: form.id,
          form_name: form.name,
          form_action: form.action,
          form_method: form.method
        });
      }
    });
  }
  
  setupPerformanceTracking() {
    if (typeof window === 'undefined' || !window.performance) return;
    
    // Track page load performance
    window.addEventListener('load', () => {
      setTimeout(() => {
        const perfData = this.getPerformanceContext();
        this.track('$page_performance', perfData);
      }, 0);
    });
    
    // Track long tasks (if available)
    if ('PerformanceObserver' in window) {
      try {
        const longTaskObserver = new PerformanceObserver((list) => {
          list.getEntries().forEach((entry) => {
            this.track('$long_task', {
              duration: entry.duration,
              start_time: entry.startTime,
              name: entry.name
            });
          });
        });
        
        longTaskObserver.observe({ entryTypes: ['longtask'] });
      } catch (e) {
        // Long task observation not supported
      }
    }
  }
  
  setupErrorTracking() {
    if (typeof window === 'undefined') return;
    
    // Track JavaScript errors
    window.addEventListener('error', (event) => {
      this.track('$js_error', {
        message: event.message,
        filename: event.filename,
        line_number: event.lineno,
        column_number: event.colno,
        stack: event.error?.stack,
        user_agent: navigator.userAgent
      });
    });
    
    // Track unhandled promise rejections
    window.addEventListener('unhandledrejection', (event) => {
      this.track('$unhandled_rejection', {
        reason: event.reason?.toString(),
        stack: event.reason?.stack,
        user_agent: navigator.userAgent
      });
    });
  }
  
  // ========== Utility Methods ==========
  
  generateEventId() {
    return `evt_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
  
  generateSessionId() {
    const stored = this.getStoredSessionId();
    
    if (stored && this.isSessionValid(stored)) {
      return stored.id;
    }
    
    const newSession = {
      id: `ses_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      started_at: Date.now()
    };
    
    this.storeSessionId(newSession);
    return newSession.id;
  }
  
  isSessionValid(session) {
    const SESSION_TIMEOUT = 30 * 60 * 1000; // 30 minutes
    return Date.now() - session.started_at < SESSION_TIMEOUT;
  }
  
  getStoredSessionId() {
    if (this.disableLocalStorage || typeof sessionStorage === 'undefined') return null;
    
    try {
      const stored = sessionStorage.getItem('uw_session');
      return stored ? JSON.parse(stored) : null;
    } catch (e) {
      return null;
    }
  }
  
  storeSessionId(session) {
    if (this.disableLocalStorage || typeof sessionStorage === 'undefined') return;
    
    try {
      sessionStorage.setItem('uw_session', JSON.stringify(session));
    } catch (e) {
      this.error('Failed to store session:', e);
    }
  }
  
  getPersistedUserId() {
    if (this.disableLocalStorage || typeof localStorage === 'undefined') {
      return this.generateAnonymousId();
    }
    
    try {
      const userId = localStorage.getItem('uw_user_id');
      return userId || this.generateAnonymousId();
    } catch (e) {
      return this.generateAnonymousId();
    }
  }
  
  persistUserId(userId) {
    if (this.disableLocalStorage || typeof localStorage === 'undefined') return;
    
    try {
      localStorage.setItem('uw_user_id', userId);
    } catch (e) {
      this.error('Failed to persist user ID:', e);
    }
  }
  
  generateAnonymousId() {
    return `anon_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
  
  validateEventType(eventType) {
    // Event type validation rules
    const pattern = /^[a-zA-Z_$][a-zA-Z0-9_]*$/;
    return typeof eventType === 'string' && 
           pattern.test(eventType) && 
           eventType.length <= 100 &&
           eventType.length > 0;
  }
  
  sanitizeProperties(properties) {
    // Remove undefined and function values, limit depth
    const sanitized = {};
    
    for (const [key, value] of Object.entries(properties)) {
      if (value !== undefined && typeof value !== 'function') {
        // Prevent circular references and limit depth
        try {
          JSON.stringify(value);
          sanitized[key] = value;
        } catch (e) {
          // Skip values that can't be serialized
          this.log(`Skipping non-serializable property: ${key}`);
        }
      }
    }
    
    return sanitized;
  }
  
  // ========== Timer Management ==========
  
  startFlushTimer() {
    this.stopFlushTimer();
    
    this.flushTimer = setInterval(() => {
      this.flush();
    }, this.flushInterval);
  }
  
  stopFlushTimer() {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
      this.flushTimer = null;
    }
  }
  
  // ========== Retry Queue Management ==========
  
  addToRetryQueue(events) {
    this.retryQueue.push(...events);
    
    // Limit retry queue size to prevent memory issues
    const MAX_RETRY_QUEUE = 1000;
    if (this.retryQueue.length > MAX_RETRY_QUEUE) {
      this.retryQueue = this.retryQueue.slice(-MAX_RETRY_QUEUE);
      this.log(`Retry queue truncated to ${MAX_RETRY_QUEUE} events`);
    }
    
    // Persist retry queue
    this.persistRetryQueue();
  }
  
  async processRetryQueue() {
    if (this.retryQueue.length === 0) {
      return;
    }
    
    this.log(`Processing retry queue with ${this.retryQueue.length} events`);
    
    const events = [...this.retryQueue];
    this.retryQueue = [];
    this.clearPersistedRetryQueue();
    
    await this.sendEvents(events);
  }
  
  persistRetryQueue() {
    if (this.disableLocalStorage || typeof localStorage === 'undefined') return;
    
    try {
      const key = `uw_retry_${this.appId}`;
      localStorage.setItem(key, JSON.stringify(this.retryQueue));
    } catch (e) {
      this.error('Failed to persist retry queue:', e);
    }
  }
  
  clearPersistedRetryQueue() {
    if (this.disableLocalStorage || typeof localStorage === 'undefined') return;
    
    try {
      const key = `uw_retry_${this.appId}`;
      localStorage.removeItem(key);
    } catch (e) {
      this.error('Failed to clear retry queue:', e);
    }
  }
  
  loadPersistedEvents() {
    if (this.disableLocalStorage || typeof localStorage === 'undefined') return;
    
    try {
      // Load queued events
      const queueKey = `uw_queue_${this.appId}`;
      const queueData = localStorage.getItem(queueKey);
      
      if (queueData) {
        const events = JSON.parse(queueData);
        this.eventQueue.push(...events);
        localStorage.removeItem(queueKey);
        this.log(`Loaded ${events.length} persisted events`);
      }
      
      // Load retry queue
      const retryKey = `uw_retry_${this.appId}`;
      const retryData = localStorage.getItem(retryKey);
      
      if (retryData) {
        const events = JSON.parse(retryData);
        this.retryQueue.push(...events);
        localStorage.removeItem(retryKey);
        
        // Process retry queue if online
        if (this.isOnline) {
          setTimeout(() => this.processRetryQueue(), 1000);
        }
        
        this.log(`Loaded ${events.length} retry events`);
      }
    } catch (e) {
      this.error('Failed to load persisted events:', e);
    }
  }
  
  persistQueue() {
    if (this.disableLocalStorage || typeof localStorage === 'undefined') return;
    
    try {
      const key = `uw_queue_${this.appId}`;
      localStorage.setItem(key, JSON.stringify(this.eventQueue));
    } catch (e) {
      this.error('Failed to persist queue:', e);
    }
  }
  
  clearPersistedQueue() {
    if (this.disableLocalStorage || typeof localStorage === 'undefined') return;
    
    try {
      const key = `uw_queue_${this.appId}`;
      localStorage.removeItem(key);
    } catch (e) {
      this.error('Failed to clear queue:', e);
    }
  }
  
  // ========== Network Requests ==========
  
  async request(path, options = {}) {
    const url = `${this.endpoint}${path}`;
    
    const fetchOptions = {
      ...options,
      signal: AbortSignal.timeout ? AbortSignal.timeout(this.timeout) : undefined
    };
    
    // Add timeout support for older browsers
    if (!AbortSignal.timeout) {
      const controller = new AbortController();
      fetchOptions.signal = controller.signal;
      
      setTimeout(() => {
        controller.abort();
      }, this.timeout);
    }
    
    return fetch(url, fetchOptions);
  }
  
  // ========== Event Emitter ==========
  
  emit(eventName, data) {
    if (this.listeners && this.listeners[eventName]) {
      this.listeners[eventName].forEach(callback => {
        try {
          callback(data);
        } catch (e) {
          this.error('Event listener error:', e);
        }
      });
    }
  }
  
  on(eventName, callback) {
    if (!this.listeners) {
      this.listeners = {};
    }
    
    if (!this.listeners[eventName]) {
      this.listeners[eventName] = [];
    }
    
    this.listeners[eventName].push(callback);
  }
  
  off(eventName, callback) {
    if (!this.listeners || !this.listeners[eventName]) {
      return;
    }
    
    this.listeners[eventName] = this.listeners[eventName].filter(cb => cb !== callback);
  }
  
  // ========== Logging ==========
  
  log(...args) {
    if (this.debug && typeof console !== 'undefined') {
      console.log('[UserWhisperer]', ...args);
    }
  }
  
  error(...args) {
    if (typeof console !== 'undefined') {
      console.error('[UserWhisperer]', ...args);
    }
  }
  
  // ========== Public API ==========
  
  setUserId(userId) {
    return this.identify(userId);
  }
  
  getUserId() {
    return this.userId;
  }
  
  getSessionId() {
    return this.sessionId;
  }
  
  reset() {
    // Clear user data
    this.userId = this.generateAnonymousId();
    this.persistUserId(this.userId);
    
    // Generate new session
    this.sessionId = this.generateSessionId();
    
    // Clear queues
    this.eventQueue = [];
    this.retryQueue = [];
    this.clearPersistedQueue();
    this.clearPersistedRetryQueue();
    
    this.log('SDK reset - new anonymous user created');
  }
  
  // ========== Cleanup ==========
  
  destroy() {
    // Flush remaining events
    this.flush(true);
    
    // Stop timer
    this.stopFlushTimer();
    
    // Remove event listeners
    if (typeof window !== 'undefined') {
      window.removeEventListener('beforeunload', this.flush);
      window.removeEventListener('online', this.processRetryQueue);
      window.removeEventListener('offline', () => {});
    }
    
    // Clear state
    this.eventQueue = [];
    this.retryQueue = [];
    this.listeners = {};
    
    this.log('SDK destroyed');
  }
}

// Export for different module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = UserWhisperer;
} else if (typeof define === 'function' && define.amd) {
  define([], function() {
    return UserWhisperer;
  });
} else if (typeof window !== 'undefined') {
  window.UserWhisperer = UserWhisperer;
}
