import { Pool, PoolClient, QueryResult, QueryResultRow } from 'pg';
import Config from './config';
import logger from './logger';

export class DatabaseConnection {
  private pool: Pool;
  private static instance: DatabaseConnection;

  private constructor() {
    // Use DATABASE_URL if available, otherwise individual config
    const connectionConfig = Config.POSTGRES_URL ? {
      connectionString: Config.POSTGRES_URL,
      max: Config.POSTGRES_MAX_CONNECTIONS,
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 10000,
      ssl: Config.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false,
    } : {
      host: Config.POSTGRES_HOST,
      port: Config.POSTGRES_PORT,
      database: Config.POSTGRES_DB,
      user: Config.POSTGRES_USER,
      password: Config.POSTGRES_PASSWORD,
      max: Config.POSTGRES_MAX_CONNECTIONS,
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 10000,
      ssl: Config.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false,
    };

    this.pool = new Pool(connectionConfig);

    this.pool.on('error', (err: any) => {
      logger.error('Unexpected error on idle client', { error: err });
    });

    this.pool.on('connect', () => {
      logger.debug('New database connection established');
    });

    this.pool.on('remove', () => {
      logger.debug('Database connection removed from pool');
    });
  }

  public static getInstance(): DatabaseConnection {
    if (!DatabaseConnection.instance) {
      DatabaseConnection.instance = new DatabaseConnection();
    }
    return DatabaseConnection.instance;
  }

  public async query<T extends QueryResultRow = any>(text: string, params?: any[]): Promise<QueryResult<T>> {
    const start = Date.now();
    try {
      const result = await this.pool.query(text, params);
      const duration = Date.now() - start;
      
      logger.debug('Executed query', {
        query: text,
        duration: `${duration}ms`,
        rows: result.rowCount,
      });
      
      return result;
    } catch (error) {
      const duration = Date.now() - start;
      logger.error('Database query error', {
        query: text,
        params,
        duration: `${duration}ms`,
        error,
      });
      throw error;
    }
  }

  public async getClient(): Promise<PoolClient> {
    return this.pool.connect();
  }

  public async transaction<T>(callback: (client: PoolClient) => Promise<T>): Promise<T> {
    const client = await this.getClient();
    
    try {
      await client.query('BEGIN');
      const result = await callback(client);
      await client.query('COMMIT');
      return result;
    } catch (error) {
      await client.query('ROLLBACK');
      throw error;
    } finally {
      client.release();
    }
  }

  public async healthCheck(): Promise<boolean> {
    try {
      const result = await this.query('SELECT 1 as health_check');
      return result.rows.length === 1 && result.rows[0].health_check === 1;
    } catch (error) {
      logger.error('Database health check failed', { error });
      return false;
    }
  }

  public async close(): Promise<void> {
    await this.pool.end();
    logger.info('Database pool closed');
  }

  public get poolInfo() {
    return {
      totalCount: this.pool.totalCount,
      idleCount: this.pool.idleCount,
      waitingCount: this.pool.waitingCount,
    };
  }
}

// Event-specific database operations
export class EventsDatabase {
  private db: DatabaseConnection;

  constructor() {
    this.db = DatabaseConnection.getInstance();
  }

  async insertEvent(event: any): Promise<string> {
    const query = `
      INSERT INTO events.raw_events (
        event_id, app_id, user_id, session_id, event_type,
        properties, context, timestamp, received_at
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
      RETURNING event_id
    `;
    
    const values = [
      event.event_id,
      event.app_id,
      event.user_id,
      event.session_id,
      event.event_type,
      JSON.stringify(event.properties || {}),
      JSON.stringify(event.context || {}),
      event.timestamp,
      new Date()
    ];

    const result = await this.db.query(query, values);
    return result.rows[0].event_id;
  }

  async insertEventBatch(events: any[]): Promise<string[]> {
    return this.db.transaction(async (client) => {
      const eventIds: string[] = [];
      
      for (const event of events) {
        const query = `
          INSERT INTO events.raw_events (
            event_id, app_id, user_id, session_id, event_type,
            properties, context, timestamp, received_at
          ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
          RETURNING event_id
        `;
        
        const values = [
          event.event_id,
          event.app_id,
          event.user_id,
          event.session_id,
          event.event_type,
          JSON.stringify(event.properties || {}),
          JSON.stringify(event.context || {}),
          event.timestamp,
          new Date()
        ];

        const result = await client.query(query, values);
        eventIds.push(result.rows[0].event_id);
      }
      
      return eventIds;
    });
  }

  async getEvent(eventId: string): Promise<any> {
    const query = `
      SELECT * FROM events.raw_events 
      WHERE event_id = $1
    `;
    
    const result = await this.db.query(query, [eventId]);
    return result.rows[0];
  }

  async checkEventExists(eventId: string): Promise<boolean> {
    const query = `
      SELECT 1 FROM events.raw_events 
      WHERE event_id = $1
    `;
    
    const result = await this.db.query(query, [eventId]);
    return result.rows.length > 0;
  }
}

// Export singleton instance
export const database = DatabaseConnection.getInstance();
export const eventsDb = new EventsDatabase(); 