/**
 * TypeScript definitions for User Whisperer JavaScript SDK
 */

export interface UserWhispererConfig {
  /** API key for authentication */
  apiKey: string;
  /** Application ID */
  appId: string;
  /** API endpoint URL */
  endpoint?: string;
  /** Enable debug logging */
  debug?: boolean;
  /** Number of events to batch before sending */
  batchSize?: number;
  /** Interval in milliseconds to flush events */
  flushInterval?: number;
  /** Number of retry attempts for failed requests */
  retryLimit?: number;
  /** Request timeout in milliseconds */
  timeout?: number;
  /** Enable gzip compression */
  enableCompression?: boolean;
  /** Disable localStorage usage */
  disableLocalStorage?: boolean;
  /** Enable performance tracking */
  performanceTracking?: boolean;
  /** Enable automatic event tracking */
  autoTrack?: boolean;
}

export interface EventProperties {
  [key: string]: any;
}

export interface UserTraits {
  email?: string;
  name?: string;
  firstName?: string;
  lastName?: string;
  phone?: string;
  avatar?: string;
  createdAt?: string | Date;
  [key: string]: any;
}

export interface GroupTraits {
  name?: string;
  description?: string;
  website?: string;
  industry?: string;
  employees?: number;
  plan?: string;
  [key: string]: any;
}

export interface EventOptions {
  timestamp?: string | Date;
  context?: Partial<EventContext>;
  integrations?: {
    [integration: string]: boolean | { [key: string]: any };
  };
}

export interface EventContext {
  page?: PageContext;
  device?: DeviceContext;
  session?: SessionContext;
  campaign?: CampaignContext;
  library?: LibraryContext;
  performance?: PerformanceContext;
  location?: LocationContext;
}

export interface PageContext {
  url?: string;
  path?: string;
  referrer?: string;
  search?: string;
  hash?: string;
  title?: string;
  host?: string;
  protocol?: string;
}

export interface DeviceContext {
  user_agent?: string;
  screen_width?: number;
  screen_height?: number;
  viewport_width?: number;
  viewport_height?: number;
  timezone?: string;
  timezone_offset?: number;
  language?: string;
  languages?: string[];
  platform?: string;
  cookie_enabled?: boolean;
  java_enabled?: boolean;
  online?: boolean;
  connection?: ConnectionContext;
}

export interface ConnectionContext {
  effective_type?: string;
  downlink?: number;
  rtt?: number;
  save_data?: boolean;
}

export interface SessionContext {
  session_id?: string;
  session_start?: number;
  page_load_time?: number;
  session_duration?: number;
}

export interface CampaignContext {
  utm_source?: string;
  utm_medium?: string;
  utm_campaign?: string;
  utm_term?: string;
  utm_content?: string;
}

export interface LibraryContext {
  name?: string;
  version?: string;
}

export interface PerformanceContext {
  dns_time?: number;
  connect_time?: number;
  request_time?: number;
  response_time?: number;
  dom_load_time?: number;
  page_load_time?: number;
  redirect_time?: number;
}

export interface LocationContext {
  city?: string;
  region?: string;
  country?: string;
  continent?: string;
  latitude?: number;
  longitude?: number;
}

export interface Event {
  id: string;
  app_id: string;
  user_id: string;
  session_id: string;
  event_type: string;
  properties: EventProperties;
  context: EventContext;
  timestamp: string;
}

export interface EventResponse {
  success: boolean;
  event_id?: string;
  failed?: Array<{
    event_id: string;
    error: string;
  }>;
}

export interface BatchResponse {
  success: boolean;
  processed: number;
  failed?: Array<{
    event_id: string;
    error: string;
  }>;
}

export type EventCallback = (data: any) => void;

export declare class UserWhisperer {
  constructor(config: UserWhispererConfig);
  
  // Core tracking methods
  track(eventType: string, properties?: EventProperties, options?: EventOptions): string | null;
  identify(userId: string, traits?: UserTraits): string;
  alias(newUserId: string): void;
  group(groupId: string, traits?: GroupTraits): void;
  page(name?: string, properties?: EventProperties): void;
  screen(name: string, properties?: EventProperties): void;
  
  // Queue management
  flush(synchronous?: boolean): Promise<void>;
  
  // User management
  setUserId(userId: string): string;
  getUserId(): string;
  getSessionId(): string;
  reset(): void;
  
  // Event listeners
  on(eventName: string, callback: EventCallback): void;
  off(eventName: string, callback: EventCallback): void;
  
  // Lifecycle
  destroy(): void;
  
  // Configuration
  readonly config: UserWhispererConfig;
  readonly apiKey: string;
  readonly appId: string;
  readonly endpoint: string;
}

export default UserWhisperer;
