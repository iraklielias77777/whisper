"""
Database Manager for User Whisperer Platform
Provides connection pooling, retry logic, and optimized query patterns
"""

import asyncpg
import asyncio
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager
import json
import hashlib
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    host: str
    port: int
    user: str
    password: str
    database: str
    read_replicas: Optional[List[str]] = None
    min_pool_size: int = 10
    max_pool_size: int = 50
    max_queries: int = 50000
    max_inactive_connection_lifetime: int = 300
    command_timeout: int = 60

class DatabaseManager:
    """
    Manages all database operations with connection pooling and retry logic
    """
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pool = None
        self.read_pool = None  # Separate pool for read replicas
        self._initialized = False
        
    async def initialize(self):
        """Initialize database connection pools"""
        
        if self._initialized:
            return
            
        try:
            # Main write pool
            self.pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                user=self.config.user,
                password=self.config.password,
                database=self.config.database,
                min_size=self.config.min_pool_size,
                max_size=self.config.max_pool_size,
                max_queries=self.config.max_queries,
                max_inactive_connection_lifetime=self.config.max_inactive_connection_lifetime,
                command_timeout=self.config.command_timeout
            )
            
            # Read replica pool
            if self.config.read_replicas:
                # Use first read replica (in production, implement load balancing)
                read_host = self.config.read_replicas[0]
                self.read_pool = await asyncpg.create_pool(
                    host=read_host,
                    port=self.config.port,
                    user=self.config.user,
                    password=self.config.password,
                    database=self.config.database,
                    min_size=self.config.min_pool_size * 2,  # More connections for reads
                    max_size=self.config.max_pool_size * 2,
                    max_queries=self.config.max_queries * 2,
                    max_inactive_connection_lifetime=self.config.max_inactive_connection_lifetime,
                    command_timeout=30  # Shorter timeout for reads
                )
                logger.info(f"Initialized read replica pool for {read_host}")
            else:
                self.read_pool = self.pool
                
            self._initialized = True
            logger.info("Database manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database manager: {e}")
            raise
    
    async def close(self):
        """Close all database connections"""
        
        if self.pool:
            await self.pool.close()
            
        if self.read_pool and self.read_pool != self.pool:
            await self.read_pool.close()
            
        self._initialized = False
        logger.info("Database manager closed")
    
    @asynccontextmanager
    async def acquire_connection(self, read_only: bool = False):
        """Acquire a database connection from the appropriate pool"""
        
        if not self._initialized:
            await self.initialize()
            
        pool = self.read_pool if read_only else self.pool
        
        async with pool.acquire() as connection:
            yield connection
    
    async def execute_with_retry(
        self,
        query: str,
        params: List = None,
        max_retries: int = 3
    ) -> Any:
        """Execute query with retry logic"""
        
        for attempt in range(max_retries):
            try:
                async with self.acquire_connection() as conn:
                    if params:
                        result = await conn.execute(query, *params)
                    else:
                        result = await conn.execute(query)
                    return result
                    
            except asyncpg.PostgresConnectionError as e:
                if attempt == max_retries - 1:
                    logger.error(f"Connection failed after {max_retries} attempts: {e}")
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                logger.warning(f"Connection attempt {attempt + 1} failed, retrying...")
                
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                raise
    
    async def fetch_with_retry(
        self,
        query: str,
        params: List = None,
        max_retries: int = 3,
        read_only: bool = True
    ) -> List[Dict]:
        """Fetch data with retry logic"""
        
        for attempt in range(max_retries):
            try:
                async with self.acquire_connection(read_only=read_only) as conn:
                    if params:
                        rows = await conn.fetch(query, *params)
                    else:
                        rows = await conn.fetch(query)
                    
                    return [dict(row) for row in rows]
                    
            except asyncpg.PostgresConnectionError as e:
                if attempt == max_retries - 1:
                    logger.error(f"Connection failed after {max_retries} attempts: {e}")
                    raise
                await asyncio.sleep(2 ** attempt)
                logger.warning(f"Fetch attempt {attempt + 1} failed, retrying...")
                
            except Exception as e:
                logger.error(f"Query fetch failed: {e}")
                raise
    
    async def fetchrow_with_retry(
        self,
        query: str,
        params: List = None,
        max_retries: int = 3,
        read_only: bool = True
    ) -> Optional[Dict]:
        """Fetch single row with retry logic"""
        
        for attempt in range(max_retries):
            try:
                async with self.acquire_connection(read_only=read_only) as conn:
                    if params:
                        row = await conn.fetchrow(query, *params)
                    else:
                        row = await conn.fetchrow(query)
                    
                    return dict(row) if row else None
                    
            except asyncpg.PostgresConnectionError as e:
                if attempt == max_retries - 1:
                    logger.error(f"Connection failed after {max_retries} attempts: {e}")
                    raise
                await asyncio.sleep(2 ** attempt)
                
            except Exception as e:
                logger.error(f"Query fetchrow failed: {e}")
                raise
    
    async def batch_insert(
        self,
        table: str,
        records: List[Dict],
        on_conflict: Optional[str] = None,
        schema: str = 'core'
    ) -> int:
        """Perform batch insert with COPY for performance"""
        
        if not records:
            return 0
        
        try:
            # Get column names from first record
            columns = list(records[0].keys())
            full_table_name = f"{schema}.{table}"
            
            async with self.acquire_connection() as conn:
                # Use COPY for bulk insert
                result = await conn.copy_records_to_table(
                    table_name=full_table_name,
                    records=records,
                    columns=columns
                )
                
                logger.info(f"Batch inserted {len(records)} records into {full_table_name}")
                return len(records)
                
        except Exception as e:
            logger.error(f"Batch insert failed for {table}: {e}")
            raise
    
    async def upsert_user_profile(self, user_data: Dict) -> str:
        """Upsert user profile with conflict resolution"""
        
        query = """
            INSERT INTO core.user_profiles (
                app_id, external_user_id, email, email_hash,
                phone, phone_hash, name, subscription_status,
                subscription_plan, lifecycle_stage, timezone,
                channel_preferences, preferred_language
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            ON CONFLICT (app_id, external_user_id)
            DO UPDATE SET
                email = EXCLUDED.email,
                email_hash = EXCLUDED.email_hash,
                phone = EXCLUDED.phone,
                phone_hash = EXCLUDED.phone_hash,
                name = EXCLUDED.name,
                subscription_status = EXCLUDED.subscription_status,
                subscription_plan = EXCLUDED.subscription_plan,
                lifecycle_stage = EXCLUDED.lifecycle_stage,
                timezone = EXCLUDED.timezone,
                channel_preferences = EXCLUDED.channel_preferences,
                preferred_language = EXCLUDED.preferred_language,
                updated_at = NOW(),
                last_active_at = NOW()
            RETURNING id
        """
        
        try:
            async with self.acquire_connection() as conn:
                result = await conn.fetchval(
                    query,
                    user_data['app_id'],
                    user_data['external_user_id'],
                    user_data.get('email'),
                    self.hash_value(user_data.get('email')),
                    user_data.get('phone'),
                    self.hash_value(user_data.get('phone')),
                    user_data.get('name'),
                    user_data.get('subscription_status', 'free'),
                    user_data.get('subscription_plan'),
                    user_data.get('lifecycle_stage', 'new'),
                    user_data.get('timezone', 'UTC'),
                    json.dumps(user_data.get('channel_preferences', {
                        "email": True, "sms": False, "push": True
                    })),
                    user_data.get('preferred_language', 'en')
                )
                
                return str(result)
                
        except Exception as e:
            logger.error(f"User profile upsert failed: {e}")
            raise
    
    async def get_user_profile(
        self, 
        app_id: str, 
        external_user_id: str
    ) -> Optional[Dict]:
        """Get user profile by app_id and external_user_id"""
        
        query = """
            SELECT * FROM core.user_profiles 
            WHERE app_id = $1 AND external_user_id = $2 AND deleted_at IS NULL
        """
        
        return await self.fetchrow_with_retry(
            query, 
            [app_id, external_user_id],
            read_only=True
        )
    
    async def update_user_behavioral_scores(
        self,
        user_id: str,
        scores: Dict[str, float]
    ) -> bool:
        """Update user behavioral scores"""
        
        query = """
            UPDATE core.user_profiles 
            SET 
                engagement_score = $2,
                churn_risk_score = $3,
                ltv_prediction = $4,
                upgrade_probability = $5,
                updated_at = NOW()
            WHERE id = $1
        """
        
        try:
            await self.execute_with_retry(
                query,
                [
                    user_id,
                    scores.get('engagement_score'),
                    scores.get('churn_risk_score'),
                    scores.get('ltv_prediction'),
                    scores.get('upgrade_probability')
                ]
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to update behavioral scores for user {user_id}: {e}")
            return False
    
    async def insert_event(self, event_data: Dict) -> str:
        """Insert event with automatic partitioning"""
        
        query = """
            INSERT INTO core.events (
                app_id, user_id, event_type, event_category,
                properties, context, user_context, geo_data,
                device_data, session_metrics, created_at, processed_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            RETURNING id
        """
        
        try:
            event_id = await self.execute_with_retry(
                query,
                [
                    event_data['app_id'],
                    event_data['user_id'],
                    event_data['event_type'],
                    event_data.get('event_category'),
                    json.dumps(event_data.get('properties', {})),
                    json.dumps(event_data.get('context', {})),
                    json.dumps(event_data.get('user_context', {})),
                    json.dumps(event_data.get('geo_data', {})),
                    json.dumps(event_data.get('device_data', {})),
                    json.dumps(event_data.get('session_metrics', {})),
                    event_data.get('created_at', datetime.utcnow()),
                    datetime.utcnow()
                ]
            )
            
            return str(event_id)
            
        except Exception as e:
            logger.error(f"Failed to insert event: {e}")
            raise
    
    async def get_user_events(
        self,
        user_id: str,
        event_types: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get user events with optional filtering"""
        
        conditions = ["user_id = $1"]
        params = [user_id]
        param_counter = 2
        
        if event_types:
            placeholders = ','.join([f'${i}' for i in range(param_counter, param_counter + len(event_types))])
            conditions.append(f"event_type IN ({placeholders})")
            params.extend(event_types)
            param_counter += len(event_types)
        
        if start_date:
            conditions.append(f"created_at >= ${param_counter}")
            params.append(start_date)
            param_counter += 1
        
        if end_date:
            conditions.append(f"created_at <= ${param_counter}")
            params.append(end_date)
            param_counter += 1
        
        query = f"""
            SELECT * FROM core.events 
            WHERE {' AND '.join(conditions)}
            ORDER BY created_at DESC 
            LIMIT ${param_counter}
        """
        params.append(limit)
        
        return await self.fetch_with_retry(query, params, read_only=True)
    
    async def insert_message_history(self, message_data: Dict) -> str:
        """Insert message delivery record"""
        
        query = """
            INSERT INTO core.message_history (
                app_id, user_id, message_id, channel, message_type,
                subject, content, scheduled_at, status, provider_id
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING id
        """
        
        try:
            result = await self.execute_with_retry(
                query,
                [
                    message_data['app_id'],
                    message_data['user_id'],
                    message_data['message_id'],
                    message_data['channel'],
                    message_data['message_type'],
                    message_data.get('subject'),
                    message_data.get('content'),
                    message_data.get('scheduled_at', datetime.utcnow()),
                    message_data.get('status', 'scheduled'),
                    message_data.get('provider_id')
                ]
            )
            
            return str(result)
            
        except Exception as e:
            logger.error(f"Failed to insert message history: {e}")
            raise
    
    async def update_message_status(
        self,
        message_id: str,
        status: str,
        delivered_at: Optional[datetime] = None,
        opened_at: Optional[datetime] = None,
        clicked_at: Optional[datetime] = None
    ) -> bool:
        """Update message delivery status"""
        
        query = """
            UPDATE core.message_history 
            SET 
                status = $2,
                delivered_at = COALESCE($3, delivered_at),
                opened_at = COALESCE($4, opened_at),
                clicked_at = COALESCE($5, clicked_at),
                updated_at = NOW()
            WHERE message_id = $1
        """
        
        try:
            await self.execute_with_retry(
                query,
                [message_id, status, delivered_at, opened_at, clicked_at]
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to update message status: {e}")
            return False
    
    async def aggregate_daily_metrics(
        self,
        user_id: str,
        date: datetime,
        metrics: Dict
    ) -> bool:
        """Insert or update daily user metrics"""
        
        query = """
            INSERT INTO analytics.user_metrics_daily (
                user_id, date, app_id, event_count, session_count,
                total_session_duration_seconds, unique_event_types,
                features_used, pages_viewed, actions_completed,
                errors_encountered, messages_sent, messages_opened,
                messages_clicked, daily_engagement_score
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
            ON CONFLICT (user_id, date)
            DO UPDATE SET
                event_count = EXCLUDED.event_count,
                session_count = EXCLUDED.session_count,
                total_session_duration_seconds = EXCLUDED.total_session_duration_seconds,
                unique_event_types = EXCLUDED.unique_event_types,
                features_used = EXCLUDED.features_used,
                pages_viewed = EXCLUDED.pages_viewed,
                actions_completed = EXCLUDED.actions_completed,
                errors_encountered = EXCLUDED.errors_encountered,
                messages_sent = EXCLUDED.messages_sent,
                messages_opened = EXCLUDED.messages_opened,
                messages_clicked = EXCLUDED.messages_clicked,
                daily_engagement_score = EXCLUDED.daily_engagement_score,
                updated_at = NOW()
        """
        
        try:
            await self.execute_with_retry(
                query,
                [
                    user_id,
                    date.date(),
                    metrics['app_id'],
                    metrics.get('event_count', 0),
                    metrics.get('session_count', 0),
                    metrics.get('total_session_duration_seconds', 0),
                    metrics.get('unique_event_types', 0),
                    metrics.get('features_used', []),
                    metrics.get('pages_viewed', 0),
                    metrics.get('actions_completed', 0),
                    metrics.get('errors_encountered', 0),
                    metrics.get('messages_sent', 0),
                    metrics.get('messages_opened', 0),
                    metrics.get('messages_clicked', 0),
                    metrics.get('daily_engagement_score')
                ]
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to aggregate daily metrics: {e}")
            return False
    
    @staticmethod
    def hash_value(value: Optional[str]) -> Optional[str]:
        """Hash sensitive values for indexing"""
        if not value:
            return None
        
        return hashlib.sha256(value.encode()).hexdigest()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check"""
        
        try:
            async with self.acquire_connection(read_only=True) as conn:
                # Test basic connectivity
                result = await conn.fetchval("SELECT 1")
                
                # Check pool status
                pool_info = {
                    'write_pool': {
                        'size': self.pool.get_size(),
                        'min_size': self.pool.get_min_size(),
                        'max_size': self.pool.get_max_size(),
                        'idle_size': self.pool.get_idle_size()
                    }
                }
                
                if self.read_pool != self.pool:
                    pool_info['read_pool'] = {
                        'size': self.read_pool.get_size(),
                        'min_size': self.read_pool.get_min_size(),
                        'max_size': self.read_pool.get_max_size(),
                        'idle_size': self.read_pool.get_idle_size()
                    }
                
                return {
                    'healthy': True,
                    'connected': result == 1,
                    'pools': pool_info,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                'healthy': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }


class QueryBuilder:
    """
    Dynamic query builder for complex queries
    """
    
    def __init__(self):
        self.select_clause = []
        self.from_clause = None
        self.join_clauses = []
        self.where_clauses = []
        self.group_by_clause = []
        self.having_clauses = []
        self.order_by_clause = []
        self.limit_clause = None
        self.params = []
    
    def select(self, *columns):
        """Add SELECT columns"""
        self.select_clause.extend(columns)
        return self
    
    def from_table(self, table, schema='core'):
        """Set FROM table"""
        self.from_clause = f"{schema}.{table}"
        return self
    
    def join(self, table, on_condition, join_type='JOIN', schema='core'):
        """Add JOIN clause"""
        self.join_clauses.append(f"{join_type} {schema}.{table} ON {on_condition}")
        return self
    
    def left_join(self, table, on_condition, schema='core'):
        """Add LEFT JOIN clause"""
        return self.join(table, on_condition, 'LEFT JOIN', schema)
    
    def where(self, condition, param=None):
        """Add WHERE condition"""
        if param is not None:
            self.params.append(param)
            param_index = len(self.params)
            condition = condition.replace('?', f'${param_index}')
        
        self.where_clauses.append(condition)
        return self
    
    def where_in(self, column, values):
        """Add WHERE IN condition"""
        if not values:
            return self
            
        placeholders = []
        for value in values:
            self.params.append(value)
            placeholders.append(f'${len(self.params)}')
        
        self.where_clauses.append(f"{column} IN ({','.join(placeholders)})")
        return self
    
    def group_by(self, *columns):
        """Add GROUP BY columns"""
        self.group_by_clause.extend(columns)
        return self
    
    def having(self, condition, param=None):
        """Add HAVING condition"""
        if param is not None:
            self.params.append(param)
            param_index = len(self.params)
            condition = condition.replace('?', f'${param_index}')
        
        self.having_clauses.append(condition)
        return self
    
    def order_by(self, column, direction='ASC'):
        """Add ORDER BY clause"""
        self.order_by_clause.append(f"{column} {direction}")
        return self
    
    def limit(self, count, offset=0):
        """Add LIMIT and OFFSET"""
        limit_parts = [f"LIMIT {count}"]
        if offset > 0:
            limit_parts.append(f"OFFSET {offset}")
        self.limit_clause = " ".join(limit_parts)
        return self
    
    def build(self):
        """Build the final query and parameters"""
        query_parts = []
        
        # SELECT
        if self.select_clause:
            query_parts.append(f"SELECT {', '.join(self.select_clause)}")
        else:
            query_parts.append("SELECT *")
        
        # FROM
        if self.from_clause:
            query_parts.append(f"FROM {self.from_clause}")
        
        # JOIN
        for join in self.join_clauses:
            query_parts.append(join)
        
        # WHERE
        if self.where_clauses:
            query_parts.append(f"WHERE {' AND '.join(self.where_clauses)}")
        
        # GROUP BY
        if self.group_by_clause:
            query_parts.append(f"GROUP BY {', '.join(self.group_by_clause)}")
        
        # HAVING
        if self.having_clauses:
            query_parts.append(f"HAVING {' AND '.join(self.having_clauses)}")
        
        # ORDER BY
        if self.order_by_clause:
            query_parts.append(f"ORDER BY {', '.join(self.order_by_clause)}")
        
        # LIMIT
        if self.limit_clause:
            query_parts.append(self.limit_clause)
        
        return ' '.join(query_parts), self.params


# Singleton instance
_db_manager = None

def get_database_manager() -> DatabaseManager:
    """Get singleton database manager instance"""
    global _db_manager
    if _db_manager is None:
        raise RuntimeError("Database manager not initialized. Call initialize_database_manager() first.")
    return _db_manager

def initialize_database_manager(config: DatabaseConfig):
    """Initialize singleton database manager"""
    global _db_manager
    _db_manager = DatabaseManager(config)
    return _db_manager 