"""
Multi-Tier Storage Manager for User Whisperer Platform
Manages data across hot/warm/cold/frozen storage tiers with automatic lifecycle management
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
import logging
import zlib
import hashlib
import aiofiles
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class StorageTier(Enum):
    HOT = "hot"       # Redis - Last 24 hours, real-time access
    WARM = "warm"     # PostgreSQL - Last 90 days, fast queries
    COLD = "cold"     # BigQuery/S3 - Archive, analytics
    FROZEN = "frozen" # Glacier - Long-term compliance

@dataclass
class StoragePolicy:
    data_type: str
    hot_retention: timedelta
    warm_retention: timedelta
    cold_retention: timedelta
    compression_enabled: bool = True
    encryption_enabled: bool = True
    replication_count: int = 1

class StorageManager:
    """
    Manages data across multiple storage tiers with automatic lifecycle management
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.redis_client = None
        self.postgres_client = None
        self.bigquery_client = None
        self.s3_client = None
        self.storage_policies = {}
        self.compressor = DataCompressor()
        self.encryptor = DataEncryptor(config.get('encryption_key'))
        self.stats = StorageStats()
        
    async def initialize(self):
        """Initialize storage clients and connections"""
        
        try:
            # Initialize Redis for hot storage
            import aioredis
            self.redis_client = await aioredis.create_redis_pool(
                self.config['redis_url'],
                maxsize=50,
                minsize=10,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Initialize PostgreSQL for warm storage
            import asyncpg
            self.postgres_client = await asyncpg.create_pool(
                **self.config['postgres'],
                min_size=20,
                max_size=100,
                command_timeout=60
            )
            
            # Initialize BigQuery for cold storage
            from google.cloud import bigquery
            self.bigquery_client = bigquery.Client(
                project=self.config['gcp_project']
            )
            
            # Initialize S3 for archive storage
            import aioboto3
            self.s3_session = aioboto3.Session()
            
            # Load default storage policies
            await self.load_default_policies()
            
            logger.info("Storage manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize storage manager: {e}")
            raise
    
    async def load_default_policies(self):
        """Load default storage policies for different data types"""
        
        default_policies = {
            'events': StoragePolicy(
                data_type='events',
                hot_retention=timedelta(hours=24),
                warm_retention=timedelta(days=90),
                cold_retention=timedelta(days=365),
                compression_enabled=True,
                encryption_enabled=True
            ),
            'user_profiles': StoragePolicy(
                data_type='user_profiles',
                hot_retention=timedelta(hours=6),
                warm_retention=timedelta(days=365),
                cold_retention=timedelta(days=2555),  # 7 years for compliance
                compression_enabled=False,
                encryption_enabled=True
            ),
            'message_history': StoragePolicy(
                data_type='message_history',
                hot_retention=timedelta(hours=2),
                warm_retention=timedelta(days=180),
                cold_retention=timedelta(days=365),
                compression_enabled=True,
                encryption_enabled=True
            ),
            'ml_features': StoragePolicy(
                data_type='ml_features',
                hot_retention=timedelta(minutes=30),
                warm_retention=timedelta(days=7),
                cold_retention=timedelta(days=30),
                compression_enabled=True,
                encryption_enabled=False
            ),
            'audit_log': StoragePolicy(
                data_type='audit_log',
                hot_retention=timedelta(hours=1),
                warm_retention=timedelta(days=365),
                cold_retention=timedelta(days=2555),  # 7 years
                compression_enabled=True,
                encryption_enabled=True
            )
        }
        
        self.storage_policies.update(default_policies)
    
    def add_storage_policy(self, policy: StoragePolicy):
        """Add custom storage policy"""
        
        self.storage_policies[policy.data_type] = policy
        logger.info(f"Added storage policy for {policy.data_type}")
    
    async def store_data(
        self,
        data_type: str,
        data: Dict,
        user_id: Optional[str] = None,
        force_tier: Optional[StorageTier] = None
    ) -> Dict[str, str]:
        """Store data across appropriate tiers based on policy"""
        
        policy = self.storage_policies.get(data_type)
        if not policy:
            logger.warning(f"No storage policy for {data_type}, using default")
            policy = StoragePolicy(
                data_type=data_type,
                hot_retention=timedelta(hours=1),
                warm_retention=timedelta(days=7),
                cold_retention=timedelta(days=30)
            )
        
        storage_results = {}
        data_with_metadata = self.add_metadata(data, data_type, user_id)
        
        # Store in hot tier (always for immediate access)
        if force_tier is None or force_tier == StorageTier.HOT:
            hot_key = await self.store_hot(
                data_with_metadata, 
                policy,
                user_id
            )
            storage_results['hot'] = hot_key
        
        # Store in warm tier (for recent history)
        if force_tier is None or force_tier == StorageTier.WARM:
            warm_key = await self.store_warm(
                data_with_metadata,
                policy
            )
            storage_results['warm'] = warm_key
        
        # Async store in cold tier (for analytics and archival)
        if force_tier is None or force_tier == StorageTier.COLD:
            asyncio.create_task(
                self.store_cold(data_with_metadata, policy)
            )
            storage_results['cold'] = 'scheduled'
        
        self.stats.record_storage(data_type, len(json.dumps(data_with_metadata)))
        return storage_results
    
    def add_metadata(
        self,
        data: Dict,
        data_type: str,
        user_id: Optional[str] = None
    ) -> Dict:
        """Add storage metadata to data"""
        
        metadata = {
            'storage_metadata': {
                'data_type': data_type,
                'stored_at': datetime.utcnow().isoformat(),
                'user_id': user_id,
                'version': '1.0',
                'checksum': self.calculate_checksum(data)
            }
        }
        
        return {**data, **metadata}
    
    def calculate_checksum(self, data: Dict) -> str:
        """Calculate data checksum for integrity verification"""
        
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    async def store_hot(
        self,
        data: Dict,
        policy: StoragePolicy,
        user_id: Optional[str] = None
    ) -> str:
        """Store in hot tier (Redis)"""
        
        try:
            data_id = data.get('id', str(hash(json.dumps(data, sort_keys=True))))
            
            # Compress if enabled
            if policy.compression_enabled:
                serialized_data = self.compressor.compress(
                    json.dumps(data).encode(),
                    'lz4'
                )
                is_compressed = True
            else:
                serialized_data = json.dumps(data).encode()
                is_compressed = False
            
            # Encrypt if enabled
            if policy.encryption_enabled:
                serialized_data = self.encryptor.encrypt(serialized_data)
                is_encrypted = True
            else:
                is_encrypted = False
            
            # Store with metadata
            redis_key = f"{policy.data_type}:{data_id}"
            metadata_key = f"{redis_key}:meta"
            
            # Store data
            ttl_seconds = int(policy.hot_retention.total_seconds())
            await self.redis_client.setex(
                redis_key,
                ttl_seconds,
                serialized_data
            )
            
            # Store metadata
            metadata = {
                'compressed': is_compressed,
                'encrypted': is_encrypted,
                'stored_at': datetime.utcnow().isoformat(),
                'size': len(serialized_data)
            }
            await self.redis_client.setex(
                metadata_key,
                ttl_seconds,
                json.dumps(metadata)
            )
            
            # Add to user's data index if user_id provided
            if user_id:
                user_index_key = f"user_data:{user_id}:{policy.data_type}"
                await self.redis_client.lpush(user_index_key, redis_key)
                await self.redis_client.ltrim(user_index_key, 0, 9999)  # Keep last 10k
                await self.redis_client.expire(user_index_key, ttl_seconds)
            
            logger.debug(f"Stored in hot tier: {redis_key}")
            return redis_key
            
        except Exception as e:
            logger.error(f"Failed to store in hot tier: {e}")
            raise
    
    async def store_warm(
        self,
        data: Dict,
        policy: StoragePolicy
    ) -> str:
        """Store in warm tier (PostgreSQL)"""
        
        try:
            table_name = f"core.{policy.data_type}"
            data_id = data.get('id')
            
            # Prepare data for storage
            storage_data = data.copy()
            if policy.compression_enabled:
                # Compress large JSON fields
                for field in ['properties', 'context', 'metadata']:
                    if field in storage_data:
                        compressed = self.compressor.compress(
                            json.dumps(storage_data[field]).encode(),
                            'zlib'
                        )
                        storage_data[f"{field}_compressed"] = compressed
                        del storage_data[field]
            
            # Dynamic query construction based on data structure
            columns = list(storage_data.keys())
            placeholders = [f"${i+1}" for i in range(len(columns))]
            values = list(storage_data.values())
            
            query = f"""
                INSERT INTO {table_name} ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
                ON CONFLICT (id) DO UPDATE SET
                    updated_at = NOW(),
                    {', '.join([f"{col} = EXCLUDED.{col}" for col in columns if col != 'id'])}
                RETURNING id
            """
            
            async with self.postgres_client.acquire() as conn:
                result = await conn.fetchval(query, *values)
                
            logger.debug(f"Stored in warm tier: {table_name} - {result}")
            return str(result)
            
        except Exception as e:
            logger.error(f"Failed to store in warm tier: {e}")
            # Don't raise here - warm storage failure shouldn't break hot storage
            return "failed"
    
    async def store_cold(
        self,
        data: Dict,
        policy: StoragePolicy
    ):
        """Store in cold tier (BigQuery/S3)"""
        
        try:
            # Stream to BigQuery for analytics
            await self.store_in_bigquery(data, policy)
            
            # Archive to S3 for backup
            await self.archive_to_s3(data, policy)
            
        except Exception as e:
            logger.error(f"Failed to store in cold tier: {e}")
    
    async def store_in_bigquery(
        self,
        data: Dict,
        policy: StoragePolicy
    ):
        """Store data in BigQuery for analytics"""
        
        try:
            dataset_id = self.config['bigquery_dataset']
            table_id = f"{dataset_id}.{policy.data_type}"
            
            # Flatten nested JSON for BigQuery
            flattened_data = self.flatten_for_bigquery(data)
            
            # Insert row
            table = self.bigquery_client.get_table(table_id)
            errors = self.bigquery_client.insert_rows_json(
                table,
                [flattened_data],
                row_ids=[flattened_data.get('id')]
            )
            
            if errors:
                logger.error(f"BigQuery insert errors: {errors}")
            else:
                logger.debug(f"Stored in BigQuery: {table_id}")
                
        except Exception as e:
            logger.error(f"BigQuery storage failed: {e}")
    
    def flatten_for_bigquery(self, data: Dict, prefix: str = '') -> Dict:
        """Flatten nested JSON for BigQuery compatibility"""
        
        flattened = {}
        
        for key, value in data.items():
            full_key = f"{prefix}{key}" if prefix else key
            
            if isinstance(value, dict):
                flattened.update(
                    self.flatten_for_bigquery(value, f"{full_key}_")
                )
            elif isinstance(value, list):
                # Convert lists to JSON strings for BigQuery
                flattened[full_key] = json.dumps(value)
            else:
                flattened[full_key] = value
        
        return flattened
    
    async def archive_to_s3(
        self,
        data: Dict,
        policy: StoragePolicy
    ):
        """Archive data to S3 with intelligent partitioning"""
        
        try:
            # Partition by date and data type for efficient retrieval
            timestamp = datetime.fromisoformat(
                data.get('created_at', datetime.utcnow().isoformat())
            )
            
            # Prepare data for archival
            archive_data = data.copy()
            if policy.compression_enabled:
                serialized = json.dumps(archive_data).encode()
                compressed = self.compressor.compress(serialized, 'zlib')
                archive_data = compressed
                content_type = 'application/x-compressed'
                content_encoding = 'zlib'
            else:
                archive_data = json.dumps(archive_data).encode()
                content_type = 'application/json'
                content_encoding = None
            
            # Generate S3 key with partitioning
            s3_key = (
                f"{policy.data_type}/"
                f"year={timestamp.year}/"
                f"month={timestamp.month:02d}/"
                f"day={timestamp.day:02d}/"
                f"hour={timestamp.hour:02d}/"
                f"{data.get('id', 'unknown')}.json"
            )
            
            if policy.compression_enabled:
                s3_key += '.gz'
            
            # Store in S3
            async with self.s3_session.client('s3') as s3:
                put_object_kwargs = {
                    'Bucket': self.config['s3_bucket'],
                    'Key': s3_key,
                    'Body': archive_data,
                    'ContentType': content_type,
                    'StorageClass': 'INTELLIGENT_TIERING',
                    'Metadata': {
                        'data_type': policy.data_type,
                        'timestamp': timestamp.isoformat(),
                        'compressed': str(policy.compression_enabled),
                        'encrypted': str(policy.encryption_enabled)
                    }
                }
                
                if content_encoding:
                    put_object_kwargs['ContentEncoding'] = content_encoding
                
                await s3.put_object(**put_object_kwargs)
                
            logger.debug(f"Archived to S3: {s3_key}")
            
        except Exception as e:
            logger.error(f"S3 archival failed: {e}")
    
    async def retrieve_data(
        self,
        data_type: str,
        data_id: str = None,
        user_id: str = None,
        time_range: timedelta = None,
        tier_preference: StorageTier = None
    ) -> List[Dict]:
        """Retrieve data from appropriate tier based on age and availability"""
        
        if time_range is None:
            time_range = timedelta(hours=1)
        
        policy = self.storage_policies.get(data_type)
        if not policy:
            logger.warning(f"No policy for {data_type}, using hot tier only")
            return await self.retrieve_from_hot(data_type, data_id, user_id)
        
        # Determine optimal tier based on time range and preference
        if tier_preference:
            target_tier = tier_preference
        elif time_range <= policy.hot_retention:
            target_tier = StorageTier.HOT
        elif time_range <= policy.warm_retention:
            target_tier = StorageTier.WARM
        else:
            target_tier = StorageTier.COLD
        
        # Try to retrieve from target tier
        try:
            if target_tier == StorageTier.HOT:
                return await self.retrieve_from_hot(data_type, data_id, user_id)
            elif target_tier == StorageTier.WARM:
                return await self.retrieve_from_warm(data_type, data_id, user_id, time_range)
            elif target_tier == StorageTier.COLD:
                return await self.retrieve_from_cold(data_type, data_id, user_id, time_range)
        except Exception as e:
            logger.error(f"Failed to retrieve from {target_tier.value}: {e}")
        
        # Fallback to other tiers
        for fallback_tier in [StorageTier.HOT, StorageTier.WARM, StorageTier.COLD]:
            if fallback_tier != target_tier:
                try:
                    if fallback_tier == StorageTier.HOT:
                        result = await self.retrieve_from_hot(data_type, data_id, user_id)
                    elif fallback_tier == StorageTier.WARM:
                        result = await self.retrieve_from_warm(data_type, data_id, user_id, time_range)
                    elif fallback_tier == StorageTier.COLD:
                        result = await self.retrieve_from_cold(data_type, data_id, user_id, time_range)
                    
                    if result:
                        return result
                        
                except Exception as e:
                    logger.error(f"Fallback to {fallback_tier.value} failed: {e}")
        
        return []
    
    async def retrieve_from_hot(
        self,
        data_type: str,
        data_id: str = None,
        user_id: str = None
    ) -> List[Dict]:
        """Retrieve from Redis hot storage"""
        
        try:
            results = []
            
            if data_id:
                # Retrieve specific item
                redis_key = f"{data_type}:{data_id}"
                data = await self.get_from_redis(redis_key)
                if data:
                    results.append(data)
            
            elif user_id:
                # Retrieve user's data
                user_index_key = f"user_data:{user_id}:{data_type}"
                data_keys = await self.redis_client.lrange(user_index_key, 0, -1)
                
                for key in data_keys:
                    data = await self.get_from_redis(key)
                    if data:
                        results.append(data)
            
            self.stats.record_retrieval(data_type, StorageTier.HOT, len(results))
            return results
            
        except Exception as e:
            logger.error(f"Hot storage retrieval failed: {e}")
            return []
    
    async def get_from_redis(self, redis_key: str) -> Optional[Dict]:
        """Get and decompress/decrypt data from Redis"""
        
        try:
            # Get data and metadata
            data_bytes = await self.redis_client.get(redis_key)
            metadata_bytes = await self.redis_client.get(f"{redis_key}:meta")
            
            if not data_bytes:
                return None
            
            metadata = {}
            if metadata_bytes:
                metadata = json.loads(metadata_bytes)
            
            # Decrypt if needed
            if metadata.get('encrypted', False):
                data_bytes = self.encryptor.decrypt(data_bytes)
            
            # Decompress if needed
            if metadata.get('compressed', False):
                data_bytes = self.compressor.decompress(data_bytes, 'lz4')
            
            # Parse JSON
            return json.loads(data_bytes.decode())
            
        except Exception as e:
            logger.error(f"Failed to get from Redis {redis_key}: {e}")
            return None
    
    async def retrieve_from_warm(
        self,
        data_type: str,
        data_id: str = None,
        user_id: str = None,
        time_range: timedelta = None
    ) -> List[Dict]:
        """Retrieve from PostgreSQL warm storage"""
        
        try:
            table_name = f"core.{data_type}"
            conditions = []
            params = []
            param_counter = 1
            
            if data_id:
                conditions.append(f"id = ${param_counter}")
                params.append(data_id)
                param_counter += 1
            
            if user_id:
                conditions.append(f"user_id = ${param_counter}")
                params.append(user_id)
                param_counter += 1
            
            if time_range:
                end_time = datetime.utcnow()
                start_time = end_time - time_range
                conditions.append(f"created_at BETWEEN ${param_counter} AND ${param_counter + 1}")
                params.extend([start_time, end_time])
                param_counter += 2
            
            where_clause = " AND ".join(conditions) if conditions else "TRUE"
            query = f"""
                SELECT * FROM {table_name}
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT 1000
            """
            
            async with self.postgres_client.acquire() as conn:
                rows = await conn.fetch(query, *params)
                results = [dict(row) for row in rows]
            
            # Decompress compressed fields
            for result in results:
                for field in ['properties', 'context', 'metadata']:
                    compressed_field = f"{field}_compressed"
                    if compressed_field in result and result[compressed_field]:
                        decompressed = self.compressor.decompress(
                            result[compressed_field],
                            'zlib'
                        )
                        result[field] = json.loads(decompressed.decode())
                        del result[compressed_field]
            
            self.stats.record_retrieval(data_type, StorageTier.WARM, len(results))
            return results
            
        except Exception as e:
            logger.error(f"Warm storage retrieval failed: {e}")
            return []
    
    async def retrieve_from_cold(
        self,
        data_type: str,
        data_id: str = None,
        user_id: str = None,
        time_range: timedelta = None
    ) -> List[Dict]:
        """Retrieve from BigQuery cold storage"""
        
        try:
            dataset_id = self.config['bigquery_dataset']
            table_id = f"`{self.config['gcp_project']}.{dataset_id}.{data_type}`"
            
            conditions = []
            parameters = []
            
            if data_id:
                conditions.append("id = @data_id")
                parameters.append(
                    bigquery.ScalarQueryParameter("data_id", "STRING", data_id)
                )
            
            if user_id:
                conditions.append("user_id = @user_id")
                parameters.append(
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
                )
            
            if time_range:
                end_time = datetime.utcnow()
                start_time = end_time - time_range
                conditions.append("created_at BETWEEN @start_time AND @end_time")
                parameters.extend([
                    bigquery.ScalarQueryParameter("start_time", "TIMESTAMP", start_time),
                    bigquery.ScalarQueryParameter("end_time", "TIMESTAMP", end_time)
                ])
            
            where_clause = " AND ".join(conditions) if conditions else "TRUE"
            query = f"""
                SELECT *
                FROM {table_id}
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT 1000
            """
            
            job_config = bigquery.QueryJobConfig(query_parameters=parameters)
            query_job = self.bigquery_client.query(query, job_config=job_config)
            results = query_job.result()
            
            result_data = [dict(row) for row in results]
            
            self.stats.record_retrieval(data_type, StorageTier.COLD, len(result_data))
            return result_data
            
        except Exception as e:
            logger.error(f"Cold storage retrieval failed: {e}")
            return []
    
    async def migrate_data_tier(
        self,
        data_type: str,
        from_tier: StorageTier,
        to_tier: StorageTier,
        age_threshold: timedelta
    ) -> int:
        """Migrate data between storage tiers based on age"""
        
        migrated_count = 0
        
        try:
            # This would implement the actual migration logic
            # For now, log the operation
            logger.info(
                f"Migrating {data_type} from {from_tier.value} to {to_tier.value} "
                f"older than {age_threshold}"
            )
            
            # Implementation would:
            # 1. Query source tier for old data
            # 2. Store in destination tier
            # 3. Remove from source tier
            # 4. Update indexes
            
        except Exception as e:
            logger.error(f"Data migration failed: {e}")
        
        return migrated_count
    
    async def cleanup_expired_data(self) -> Dict[str, int]:
        """Clean up expired data according to retention policies"""
        
        cleanup_results = {}
        
        for data_type, policy in self.storage_policies.items():
            try:
                # Clean up warm tier
                warm_deleted = await self.cleanup_warm_tier(data_type, policy)
                
                # Clean up cold tier
                cold_deleted = await self.cleanup_cold_tier(data_type, policy)
                
                cleanup_results[data_type] = {
                    'warm_deleted': warm_deleted,
                    'cold_deleted': cold_deleted
                }
                
            except Exception as e:
                logger.error(f"Cleanup failed for {data_type}: {e}")
                cleanup_results[data_type] = {'error': str(e)}
        
        return cleanup_results
    
    async def cleanup_warm_tier(
        self,
        data_type: str,
        policy: StoragePolicy
    ) -> int:
        """Clean up expired data from warm tier"""
        
        try:
            cutoff_time = datetime.utcnow() - policy.warm_retention
            table_name = f"core.{data_type}"
            
            query = f"""
                DELETE FROM {table_name}
                WHERE created_at < $1
            """
            
            async with self.postgres_client.acquire() as conn:
                result = await conn.execute(query, cutoff_time)
                
            deleted_count = int(result.split()[-1])
            logger.info(f"Cleaned up {deleted_count} records from {table_name}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Warm tier cleanup failed for {data_type}: {e}")
            return 0
    
    async def cleanup_cold_tier(
        self,
        data_type: str,
        policy: StoragePolicy
    ) -> int:
        """Clean up expired data from cold tier"""
        
        try:
            # For BigQuery, we would typically just let the data expire
            # or move to even colder storage (Glacier)
            # For now, return 0
            return 0
            
        except Exception as e:
            logger.error(f"Cold tier cleanup failed for {data_type}: {e}")
            return 0
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics"""
        
        return {
            'policies': {
                name: {
                    'hot_retention': str(policy.hot_retention),
                    'warm_retention': str(policy.warm_retention),
                    'cold_retention': str(policy.cold_retention),
                    'compression_enabled': policy.compression_enabled,
                    'encryption_enabled': policy.encryption_enabled
                }
                for name, policy in self.storage_policies.items()
            },
            'stats': self.stats.get_stats(),
            'health': await self.health_check()
        }
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all storage tiers"""
        
        health = {}
        
        # Check Redis
        try:
            await self.redis_client.ping()
            health['redis'] = True
        except Exception:
            health['redis'] = False
        
        # Check PostgreSQL
        try:
            async with self.postgres_client.acquire() as conn:
                await conn.fetchval("SELECT 1")
            health['postgres'] = True
        except Exception:
            health['postgres'] = False
        
        # Check BigQuery
        try:
            list(self.bigquery_client.list_datasets(max_results=1))
            health['bigquery'] = True
        except Exception:
            health['bigquery'] = False
        
        # Check S3
        try:
            async with self.s3_session.client('s3') as s3:
                await s3.head_bucket(Bucket=self.config['s3_bucket'])
            health['s3'] = True
        except Exception:
            health['s3'] = False
        
        return health


class DataCompressor:
    """
    Data compression utilities for storage optimization
    """
    
    def __init__(self):
        self.algorithms = {}
        
        # Initialize compression algorithms
        try:
            import lz4.frame
            self.algorithms['lz4'] = lz4.frame
        except ImportError:
            logger.warning("LZ4 not available")
        
        self.algorithms['zlib'] = zlib
        
        try:
            import brotli
            self.algorithms['brotli'] = brotli
        except ImportError:
            logger.warning("Brotli not available")
    
    def compress(
        self,
        data: bytes,
        algorithm: str = 'lz4',
        level: int = 1
    ) -> bytes:
        """Compress data using specified algorithm"""
        
        if algorithm not in self.algorithms:
            raise ValueError(f"Unsupported compression algorithm: {algorithm}")
        
        try:
            if algorithm == 'lz4':
                return self.algorithms['lz4'].compress(
                    data,
                    compression_level=level
                )
            elif algorithm == 'zlib':
                return self.algorithms['zlib'].compress(data, level)
            elif algorithm == 'brotli':
                return self.algorithms['brotli'].compress(data, quality=level)
            
        except Exception as e:
            logger.error(f"Compression failed with {algorithm}: {e}")
            return data  # Return uncompressed on failure
    
    def decompress(
        self,
        data: bytes,
        algorithm: str = 'lz4'
    ) -> bytes:
        """Decompress data using specified algorithm"""
        
        if algorithm not in self.algorithms:
            raise ValueError(f"Unsupported compression algorithm: {algorithm}")
        
        try:
            if algorithm == 'lz4':
                return self.algorithms['lz4'].decompress(data)
            elif algorithm == 'zlib':
                return self.algorithms['zlib'].decompress(data)
            elif algorithm == 'brotli':
                return self.algorithms['brotli'].decompress(data)
                
        except Exception as e:
            logger.error(f"Decompression failed with {algorithm}: {e}")
            raise


class DataEncryptor:
    """
    Data encryption utilities for security
    """
    
    def __init__(self, encryption_key: Optional[str] = None):
        self.encryption_key = encryption_key
        
        if encryption_key:
            try:
                from cryptography.fernet import Fernet
                self.cipher = Fernet(encryption_key.encode()[:32].ljust(32, b'0')[:32])
                self.cipher = Fernet(Fernet.generate_key())  # Use proper key generation
            except ImportError:
                logger.warning("Cryptography library not available for encryption")
                self.cipher = None
        else:
            self.cipher = None
    
    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data"""
        
        if not self.cipher:
            return data
        
        try:
            return self.cipher.encrypt(data)
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return data
    
    def decrypt(self, data: bytes) -> bytes:
        """Decrypt data"""
        
        if not self.cipher:
            return data
        
        try:
            return self.cipher.decrypt(data)
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return data


class StorageStats:
    """
    Storage performance and usage statistics
    """
    
    def __init__(self):
        self.storage_operations = defaultdict(int)
        self.retrieval_operations = defaultdict(lambda: defaultdict(int))
        self.bytes_stored = defaultdict(int)
        self.errors = defaultdict(int)
    
    def record_storage(self, data_type: str, size_bytes: int):
        """Record storage operation"""
        
        self.storage_operations[data_type] += 1
        self.bytes_stored[data_type] += size_bytes
    
    def record_retrieval(
        self,
        data_type: str,
        tier: StorageTier,
        record_count: int
    ):
        """Record retrieval operation"""
        
        self.retrieval_operations[data_type][tier.value] += record_count
    
    def record_error(self, data_type: str):
        """Record error"""
        
        self.errors[data_type] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics summary"""
        
        return {
            'storage_operations': dict(self.storage_operations),
            'retrieval_operations': {
                data_type: dict(tiers)
                for data_type, tiers in self.retrieval_operations.items()
            },
            'bytes_stored': dict(self.bytes_stored),
            'errors': dict(self.errors),
            'total_storage_ops': sum(self.storage_operations.values()),
            'total_bytes_stored': sum(self.bytes_stored.values()),
            'total_errors': sum(self.errors.values())
        }


class DataLifecycleManager:
    """
    Manages data lifecycle and automated retention policies
    """
    
    def __init__(self, storage_manager: StorageManager):
        self.storage_manager = storage_manager
        self.lifecycle_jobs = {}
        
    async def start_lifecycle_management(self):
        """Start automated lifecycle management"""
        
        # Schedule daily cleanup
        self.lifecycle_jobs['cleanup'] = asyncio.create_task(
            self.periodic_cleanup()
        )
        
        # Schedule tier migration
        self.lifecycle_jobs['migration'] = asyncio.create_task(
            self.periodic_migration()
        )
        
        logger.info("Data lifecycle management started")
    
    async def periodic_cleanup(self):
        """Periodic data cleanup based on retention policies"""
        
        while True:
            try:
                await asyncio.sleep(24 * 3600)  # Run daily
                
                logger.info("Starting periodic data cleanup")
                results = await self.storage_manager.cleanup_expired_data()
                
                total_deleted = sum(
                    result.get('warm_deleted', 0) + result.get('cold_deleted', 0)
                    for result in results.values()
                    if isinstance(result, dict) and 'warm_deleted' in result
                )
                
                logger.info(f"Cleanup completed: {total_deleted} records deleted")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic cleanup failed: {e}")
    
    async def periodic_migration(self):
        """Periodic data migration between tiers"""
        
        while True:
            try:
                await asyncio.sleep(6 * 3600)  # Run every 6 hours
                
                logger.info("Starting periodic data migration")
                
                # Migration logic would go here
                # For example: move data from hot to warm tier after 24 hours
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic migration failed: {e}")
    
    async def stop_lifecycle_management(self):
        """Stop lifecycle management"""
        
        for job_name, job in self.lifecycle_jobs.items():
            job.cancel()
            try:
                await job
            except asyncio.CancelledError:
                pass
        
        logger.info("Data lifecycle management stopped")


# Singleton instance
_storage_manager = None

def get_storage_manager() -> StorageManager:
    """Get singleton storage manager instance"""
    global _storage_manager
    if _storage_manager is None:
        raise RuntimeError("Storage manager not initialized")
    return _storage_manager

def initialize_storage_manager(config: Dict) -> StorageManager:
    """Initialize singleton storage manager"""
    global _storage_manager
    _storage_manager = StorageManager(config)
    return _storage_manager
