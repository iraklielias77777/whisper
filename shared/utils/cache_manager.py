"""
Multi-Level Cache Manager for User Whisperer Platform
Implements L1 (Memory), L2 (Redis), L3 (CDN) caching with intelligent promotion and warming
"""

from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timedelta
import asyncio
import json
import hashlib
import logging
import time
from enum import Enum
from collections import OrderedDict
from dataclasses import dataclass
import weakref

logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    L1_MEMORY = "memory"     # In-process memory cache
    L2_REDIS = "redis"       # Distributed Redis cache
    L3_CDN = "cdn"          # Edge CDN cache

@dataclass
class CacheEntry:
    value: Any
    created_at: float
    accessed_at: float
    access_count: int
    expires_at: Optional[float] = None
    size: int = 0

class CacheManager:
    """
    Multi-level caching system with intelligent promotion and eviction
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.l1_cache = MemoryCache(
            max_size=config.get('l1_max_size', 1000),
            ttl=config.get('l1_ttl', 300)  # 5 minutes
        )
        self.l2_cache = None  # Redis, initialized async
        self.l3_cache = CDNCache(config.get('cdn', {}))
        self.stats = CacheStats()
        self.cache_warming_enabled = config.get('cache_warming', True)
        self.promotion_threshold = config.get('promotion_threshold', 2)
        
    async def initialize(self):
        """Initialize cache connections"""
        
        try:
            # Initialize Redis L2 cache
            import aioredis
            self.l2_cache = RedisCache(
                await aioredis.create_redis_pool(
                    self.config['redis_url'],
                    maxsize=50,
                    minsize=10,
                    retry_on_timeout=True
                ),
                self.config.get('l2_ttl', 3600)  # 1 hour
            )
            
            # Initialize CDN cache
            await self.l3_cache.initialize()
            
            logger.info("Cache manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize cache manager: {e}")
            raise
    
    async def get(
        self,
        key: str,
        fetch_fn: Optional[Callable] = None,
        ttl: Optional[int] = None,
        promote: bool = True
    ) -> Optional[Any]:
        """
        Get value from cache with intelligent tier promotion
        """
        
        start_time = time.time()
        
        try:
            # Try L1 (memory) first
            value = await self.l1_cache.get(key)
            if value is not None:
                self.stats.record_hit(CacheLevel.L1_MEMORY, time.time() - start_time)
                return value
            
            # Try L2 (Redis)
            value = await self.l2_cache.get(key)
            if value is not None:
                # Promote to L1 if accessed frequently
                if promote and await self.should_promote_to_l1(key):
                    await self.l1_cache.set(key, value, ttl)
                
                self.stats.record_hit(CacheLevel.L2_REDIS, time.time() - start_time)
                return value
            
            # Try L3 (CDN)
            value = await self.l3_cache.get(key)
            if value is not None:
                # Promote to L1 and L2
                if promote:
                    await self.l1_cache.set(key, value, ttl)
                    await self.l2_cache.set(key, value, ttl)
                
                self.stats.record_hit(CacheLevel.L3_CDN, time.time() - start_time)
                return value
            
            # Cache miss - fetch if function provided
            self.stats.record_miss(time.time() - start_time)
            
            if fetch_fn:
                fetch_start = time.time()
                
                if asyncio.iscoroutinefunction(fetch_fn):
                    value = await fetch_fn()
                else:
                    value = await asyncio.get_event_loop().run_in_executor(
                        None, fetch_fn
                    )
                
                fetch_time = time.time() - fetch_start
                self.stats.record_fetch(fetch_time)
                
                if value is not None:
                    await self.set(key, value, ttl)
                
                return value
            
            return None
            
        except Exception as e:
            logger.error(f"Cache get failed for key {key}: {e}")
            self.stats.record_error()
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        level: Optional[CacheLevel] = None
    ):
        """Set value in appropriate cache levels"""
        
        try:
            # Determine cache levels to update
            if level:
                levels = [level]
            else:
                levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]
                
                # Only cache in CDN for specific patterns
                if self.should_cache_in_cdn(key, value):
                    levels.append(CacheLevel.L3_CDN)
            
            # Set in specified levels
            tasks = []
            
            if CacheLevel.L1_MEMORY in levels:
                tasks.append(self.l1_cache.set(key, value, ttl))
            
            if CacheLevel.L2_REDIS in levels:
                tasks.append(self.l2_cache.set(key, value, ttl))
            
            if CacheLevel.L3_CDN in levels:
                tasks.append(self.l3_cache.set(key, value, ttl))
            
            # Execute in parallel
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            self.stats.record_set()
            
        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {e}")
            self.stats.record_error()
    
    async def should_promote_to_l1(self, key: str) -> bool:
        """Determine if key should be promoted to L1 cache"""
        
        # Check access frequency in L2
        access_count = await self.l2_cache.get_access_count(key)
        return access_count >= self.promotion_threshold
    
    def should_cache_in_cdn(self, key: str, value: Any) -> bool:
        """Determine if content should be cached in CDN"""
        
        # Cache static content, user profiles, frequently accessed data
        cdn_patterns = ['user_profile:', 'template:', 'static:', 'public:']
        
        return any(key.startswith(pattern) for pattern in cdn_patterns)
    
    async def delete(self, key: str):
        """Delete key from all cache levels"""
        
        try:
            tasks = [
                self.l1_cache.delete(key),
                self.l2_cache.delete(key),
                self.l3_cache.delete(key)
            ]
            
            await asyncio.gather(*tasks, return_exceptions=True)
            self.stats.record_delete()
            
        except Exception as e:
            logger.error(f"Cache delete failed for key {key}: {e}")
            self.stats.record_error()
    
    async def invalidate(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        
        try:
            tasks = [
                self.l1_cache.invalidate_pattern(pattern),
                self.l2_cache.invalidate_pattern(pattern),
                self.l3_cache.invalidate_pattern(pattern)
            ]
            
            await asyncio.gather(*tasks, return_exceptions=True)
            self.stats.record_invalidation()
            
        except Exception as e:
            logger.error(f"Cache invalidation failed for pattern {pattern}: {e}")
            self.stats.record_error()
    
    async def get_multi(
        self,
        keys: List[str],
        fetch_fn: Optional[Callable[[List[str]], Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Get multiple values efficiently"""
        
        results = {}
        missing_keys = []
        
        # Try to get all keys from L1 first
        for key in keys:
            value = await self.l1_cache.get(key)
            if value is not None:
                results[key] = value
            else:
                missing_keys.append(key)
        
        if not missing_keys:
            return results
        
        # Try L2 for missing keys
        l2_results = await self.l2_cache.get_multi(missing_keys)
        results.update(l2_results)
        
        # Update missing keys list
        missing_keys = [key for key in missing_keys if key not in l2_results]
        
        if missing_keys and fetch_fn:
            # Fetch missing data
            if asyncio.iscoroutinefunction(fetch_fn):
                fetched = await fetch_fn(missing_keys)
            else:
                fetched = await asyncio.get_event_loop().run_in_executor(
                    None, fetch_fn, missing_keys
                )
            
            # Cache fetched data
            if fetched:
                set_tasks = [
                    self.set(key, value)
                    for key, value in fetched.items()
                ]
                await asyncio.gather(*set_tasks, return_exceptions=True)
                
                results.update(fetched)
        
        return results
    
    async def set_multi(
        self,
        items: Dict[str, Any],
        ttl: Optional[int] = None
    ):
        """Set multiple values efficiently"""
        
        try:
            tasks = [
                self.set(key, value, ttl)
                for key, value in items.items()
            ]
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Multi-set failed: {e}")
            self.stats.record_error()
    
    def generate_key(self, *args, **kwargs) -> str:
        """Generate deterministic cache key from arguments"""
        
        key_data = {
            'args': args,
            'kwargs': {k: v for k, v in sorted(kwargs.items())}
        }
        
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        
        return {
            'overall': self.stats.get_stats(),
            'l1': await self.l1_cache.get_stats(),
            'l2': await self.l2_cache.get_stats(),
            'l3': await self.l3_cache.get_stats()
        }
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all cache levels"""
        
        health = {}
        
        try:
            # Test L1
            test_key = f"health_check_{time.time()}"
            await self.l1_cache.set(test_key, "test", 5)
            value = await self.l1_cache.get(test_key)
            health['l1'] = value == "test"
            await self.l1_cache.delete(test_key)
        except Exception:
            health['l1'] = False
        
        try:
            # Test L2
            health['l2'] = await self.l2_cache.health_check()
        except Exception:
            health['l2'] = False
        
        try:
            # Test L3
            health['l3'] = await self.l3_cache.health_check()
        except Exception:
            health['l3'] = False
        
        return health


class MemoryCache:
    """
    In-memory LRU cache with TTL support
    """
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.default_ttl = ttl
        self.lock = asyncio.Lock()
        self.total_size = 0
        self.max_memory = 100 * 1024 * 1024  # 100MB default
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache"""
        
        async with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                current_time = time.time()
                
                # Check expiration
                if entry.expires_at and entry.expires_at < current_time:
                    # Expired
                    self.total_size -= entry.size
                    del self.cache[key]
                    return None
                
                # Update access info
                entry.accessed_at = current_time
                entry.access_count += 1
                
                # Move to end (LRU)
                self.cache.move_to_end(key)
                return entry.value
        
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """Set value in memory cache"""
        
        ttl = ttl or self.default_ttl
        current_time = time.time()
        expires_at = current_time + ttl if ttl > 0 else None
        
        # Estimate size
        size = self.estimate_size(value)
        
        async with self.lock:
            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache[key]
                self.total_size -= old_entry.size
                del self.cache[key]
            
            # Create new entry
            entry = CacheEntry(
                value=value,
                created_at=current_time,
                accessed_at=current_time,
                access_count=1,
                expires_at=expires_at,
                size=size
            )
            
            self.cache[key] = entry
            self.total_size += size
            
            # Move to end (LRU)
            self.cache.move_to_end(key)
            
            # Evict if necessary
            await self.evict_if_needed()
    
    async def evict_if_needed(self):
        """Evict entries if cache is over limits"""
        
        # Evict by size limit
        while len(self.cache) > self.max_size:
            oldest_key, oldest_entry = self.cache.popitem(last=False)
            self.total_size -= oldest_entry.size
        
        # Evict by memory limit
        while self.total_size > self.max_memory and self.cache:
            oldest_key, oldest_entry = self.cache.popitem(last=False)
            self.total_size -= oldest_entry.size
        
        # Evict expired entries
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.expires_at and entry.expires_at < current_time
        ]
        
        for key in expired_keys:
            entry = self.cache[key]
            self.total_size -= entry.size
            del self.cache[key]
    
    def estimate_size(self, value: Any) -> int:
        """Estimate memory size of value"""
        
        try:
            import sys
            return sys.getsizeof(value)
        except Exception:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (dict, list)):
                return len(json.dumps(value))
            else:
                return 1024  # Default 1KB
    
    async def delete(self, key: str):
        """Delete key from memory cache"""
        
        async with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                self.total_size -= entry.size
                del self.cache[key]
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate entries matching pattern"""
        
        import re
        regex = re.compile(pattern)
        
        async with self.lock:
            keys_to_delete = [
                key for key in self.cache
                if regex.match(key)
            ]
            
            for key in keys_to_delete:
                entry = self.cache[key]
                self.total_size -= entry.size
                del self.cache[key]
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get memory cache statistics"""
        
        async with self.lock:
            total_entries = len(self.cache)
            expired_count = 0
            current_time = time.time()
            
            for entry in self.cache.values():
                if entry.expires_at and entry.expires_at < current_time:
                    expired_count += 1
        
        return {
            'total_entries': total_entries,
            'total_size_bytes': self.total_size,
            'max_size': self.max_size,
            'max_memory_bytes': self.max_memory,
            'expired_entries': expired_count,
            'memory_utilization': self.total_size / self.max_memory
        }


class RedisCache:
    """
    Redis-based distributed cache
    """
    
    def __init__(self, redis_client, default_ttl: int = 3600):
        self.redis = redis_client
        self.default_ttl = default_ttl
        self.key_prefix = "cache:"
        self.stats_prefix = "cache_stats:"
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        
        try:
            redis_key = f"{self.key_prefix}{key}"
            data = await self.redis.get(redis_key)
            
            if data:
                # Increment access count
                await self.increment_access_count(key)
                return json.loads(data)
            
            return None
            
        except Exception as e:
            logger.error(f"Redis get failed for key {key}: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """Set value in Redis cache"""
        
        try:
            redis_key = f"{self.key_prefix}{key}"
            serialized = json.dumps(value, default=str)
            ttl = ttl or self.default_ttl
            
            if ttl > 0:
                await self.redis.setex(redis_key, ttl, serialized)
            else:
                await self.redis.set(redis_key, serialized)
            
            # Initialize access count
            await self.set_access_count(key, 0, ttl)
            
        except Exception as e:
            logger.error(f"Redis set failed for key {key}: {e}")
    
    async def get_multi(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from Redis"""
        
        try:
            redis_keys = [f"{self.key_prefix}{key}" for key in keys]
            values = await self.redis.mget(*redis_keys)
            
            results = {}
            for i, value in enumerate(values):
                if value:
                    original_key = keys[i]
                    results[original_key] = json.loads(value)
                    # Increment access count
                    await self.increment_access_count(original_key)
            
            return results
            
        except Exception as e:
            logger.error(f"Redis multi-get failed: {e}")
            return {}
    
    async def delete(self, key: str):
        """Delete key from Redis cache"""
        
        try:
            redis_key = f"{self.key_prefix}{key}"
            stats_key = f"{self.stats_prefix}{key}"
            
            await self.redis.delete(redis_key, stats_key)
            
        except Exception as e:
            logger.error(f"Redis delete failed for key {key}: {e}")
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate entries matching pattern"""
        
        try:
            cursor = 0
            deleted_count = 0
            
            while True:
                cursor, keys = await self.redis.scan(
                    cursor,
                    match=f"{self.key_prefix}{pattern}",
                    count=100
                )
                
                if keys:
                    await self.redis.delete(*keys)
                    deleted_count += len(keys)
                
                if cursor == 0:
                    break
            
            logger.info(f"Invalidated {deleted_count} keys matching pattern {pattern}")
            
        except Exception as e:
            logger.error(f"Redis pattern invalidation failed: {e}")
    
    async def get_access_count(self, key: str) -> int:
        """Get access count for key"""
        
        try:
            stats_key = f"{self.stats_prefix}{key}"
            count = await self.redis.get(stats_key)
            return int(count) if count else 0
        except Exception:
            return 0
    
    async def set_access_count(
        self,
        key: str,
        count: int,
        ttl: Optional[int] = None
    ):
        """Set access count for key"""
        
        try:
            stats_key = f"{self.stats_prefix}{key}"
            ttl = ttl or self.default_ttl
            
            if ttl > 0:
                await self.redis.setex(stats_key, ttl, count)
            else:
                await self.redis.set(stats_key, count)
                
        except Exception:
            pass  # Non-critical operation
    
    async def increment_access_count(self, key: str):
        """Increment access count for key"""
        
        try:
            stats_key = f"{self.stats_prefix}{key}"
            await self.redis.incr(stats_key)
        except Exception:
            pass  # Non-critical operation
    
    async def health_check(self) -> bool:
        """Check Redis health"""
        
        try:
            await self.redis.ping()
            return True
        except Exception:
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics"""
        
        try:
            info = await self.redis.info()
            
            return {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'hit_rate': (
                    info.get('keyspace_hits', 0) / 
                    max(1, info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0))
                )
            }
            
        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}")
            return {}


class CDNCache:
    """
    CDN edge cache integration (placeholder for actual CDN implementation)
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.base_url = config.get('base_url')
        self.api_key = config.get('api_key')
        self.enabled = config.get('enabled', False)
        
    async def initialize(self):
        """Initialize CDN cache"""
        
        if not self.enabled:
            logger.info("CDN cache disabled")
            return
        
        # Initialize CDN client
        logger.info("CDN cache initialized")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get from CDN cache"""
        
        if not self.enabled:
            return None
        
        # This would integrate with actual CDN API
        # For now, return None (cache miss)
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """Set in CDN cache"""
        
        if not self.enabled:
            return
        
        # This would push content to CDN edge locations
        logger.debug(f"CDN cache set: {key}")
    
    async def delete(self, key: str):
        """Delete from CDN cache"""
        
        if not self.enabled:
            return
        
        # This would purge content from CDN
        logger.debug(f"CDN cache delete: {key}")
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate CDN cache entries matching pattern"""
        
        if not self.enabled:
            return
        
        # This would call CDN purge API
        logger.debug(f"CDN cache invalidate: {pattern}")
    
    async def health_check(self) -> bool:
        """Check CDN health"""
        
        if not self.enabled:
            return True
        
        # This would check CDN API health
        return True
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get CDN cache statistics"""
        
        if not self.enabled:
            return {'enabled': False}
        
        return {
            'enabled': True,
            'base_url': self.base_url,
            'regions': ['us-east-1', 'eu-west-1', 'ap-southeast-1']  # Example
        }


class CacheStats:
    """
    Cache performance statistics
    """
    
    def __init__(self):
        self.hits = {level: 0 for level in CacheLevel}
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.invalidations = 0
        self.errors = 0
        self.fetch_calls = 0
        self.total_requests = 0
        self.response_times = {level: [] for level in CacheLevel}
        self.fetch_times = []
        self.start_time = time.time()
        
    def record_hit(self, level: CacheLevel, response_time: float):
        """Record cache hit"""
        
        self.hits[level] += 1
        self.total_requests += 1
        self.response_times[level].append(response_time)
        
        # Keep only recent response times (last 1000)
        if len(self.response_times[level]) > 1000:
            self.response_times[level] = self.response_times[level][-500:]
    
    def record_miss(self, response_time: float):
        """Record cache miss"""
        
        self.misses += 1
        self.total_requests += 1
    
    def record_set(self):
        """Record cache set operation"""
        
        self.sets += 1
    
    def record_delete(self):
        """Record cache delete operation"""
        
        self.deletes += 1
    
    def record_invalidation(self):
        """Record cache invalidation operation"""
        
        self.invalidations += 1
    
    def record_error(self):
        """Record cache error"""
        
        self.errors += 1
    
    def record_fetch(self, fetch_time: float):
        """Record data fetch operation"""
        
        self.fetch_calls += 1
        self.fetch_times.append(fetch_time)
        
        # Keep only recent fetch times
        if len(self.fetch_times) > 1000:
            self.fetch_times = self.fetch_times[-500:]
    
    def get_hit_rate(self) -> float:
        """Calculate overall hit rate"""
        
        if self.total_requests == 0:
            return 0.0
        
        total_hits = sum(self.hits.values())
        return total_hits / self.total_requests
    
    def get_level_hit_rates(self) -> Dict[str, float]:
        """Get hit rates per cache level"""
        
        if self.total_requests == 0:
            return {level.value: 0.0 for level in CacheLevel}
        
        return {
            level.value: count / self.total_requests
            for level, count in self.hits.items()
        }
    
    def get_avg_response_times(self) -> Dict[str, float]:
        """Get average response times per cache level"""
        
        return {
            level.value: (
                sum(times) / len(times) if times else 0.0
            )
            for level, times in self.response_times.items()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        
        uptime = time.time() - self.start_time
        
        return {
            'uptime_seconds': uptime,
            'total_requests': self.total_requests,
            'overall_hit_rate': self.get_hit_rate(),
            'level_hit_rates': self.get_level_hit_rates(),
            'level_hits': {level.value: count for level, count in self.hits.items()},
            'misses': self.misses,
            'sets': self.sets,
            'deletes': self.deletes,
            'invalidations': self.invalidations,
            'errors': self.errors,
            'fetch_calls': self.fetch_calls,
            'avg_response_times_ms': {
                level: time_ms * 1000
                for level, time_ms in self.get_avg_response_times().items()
            },
            'avg_fetch_time_ms': (
                sum(self.fetch_times) / len(self.fetch_times) * 1000
                if self.fetch_times else 0.0
            ),
            'requests_per_second': self.total_requests / max(1, uptime)
        }


class CacheWarmer:
    """
    Pre-warms cache with frequently accessed data
    """
    
    def __init__(
        self,
        cache_manager: CacheManager,
        db_manager: Any = None
    ):
        self.cache_manager = cache_manager
        self.db_manager = db_manager
        self.warming_tasks = {}
        
    async def start_warming(self):
        """Start cache warming tasks"""
        
        warming_strategies = [
            ('user_profiles', self.warm_user_profiles),
            ('ml_features', self.warm_ml_features),
            ('templates', self.warm_templates),
            ('popular_content', self.warm_popular_content)
        ]
        
        for strategy_name, strategy_func in warming_strategies:
            self.warming_tasks[strategy_name] = asyncio.create_task(
                self.periodic_warming(strategy_name, strategy_func)
            )
        
        logger.info("Cache warming started")
    
    async def periodic_warming(
        self,
        strategy_name: str,
        strategy_func: Callable
    ):
        """Run warming strategy periodically"""
        
        while True:
            try:
                await strategy_func()
                await asyncio.sleep(300)  # Warm every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache warming failed for {strategy_name}: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute
    
    async def warm_user_profiles(self):
        """Warm cache with active user profiles"""
        
        if not self.db_manager:
            return
        
        try:
            # Get most active users
            query = """
                SELECT id, app_id, external_user_id, name, email,
                       lifecycle_stage, engagement_score, churn_risk_score
                FROM core.user_profiles
                WHERE last_active_at > NOW() - INTERVAL '24 hours'
                AND engagement_score > 0.5
                ORDER BY last_active_at DESC
                LIMIT 1000
            """
            
            profiles = await self.db_manager.fetch_with_retry(query, read_only=True)
            
            # Cache profiles
            warm_tasks = []
            for profile in profiles:
                key = f"user_profile:{profile['id']}"
                warm_tasks.append(
                    self.cache_manager.set(key, dict(profile), ttl=3600)
                )
            
            if warm_tasks:
                await asyncio.gather(*warm_tasks, return_exceptions=True)
                
            logger.info(f"Warmed {len(profiles)} user profiles")
            
        except Exception as e:
            logger.error(f"User profile warming failed: {e}")
    
    async def warm_ml_features(self):
        """Warm cache with recent ML features"""
        
        if not self.db_manager:
            return
        
        try:
            query = """
                SELECT user_id, feature_set, features, expires_at
                FROM ml.features
                WHERE computed_at > NOW() - INTERVAL '1 hour'
                AND (expires_at IS NULL OR expires_at > NOW())
                ORDER BY computed_at DESC
                LIMIT 5000
            """
            
            features = await self.db_manager.fetch_with_retry(query, read_only=True)
            
            warm_tasks = []
            for feature_set in features:
                key = f"ml_features:{feature_set['user_id']}:{feature_set['feature_set']}"
                
                # Calculate TTL
                if feature_set['expires_at']:
                    ttl = max(60, int((feature_set['expires_at'] - datetime.utcnow()).total_seconds()))
                else:
                    ttl = 1800  # 30 minutes default
                
                warm_tasks.append(
                    self.cache_manager.set(
                        key,
                        feature_set['features'],
                        ttl=ttl
                    )
                )
            
            if warm_tasks:
                await asyncio.gather(*warm_tasks, return_exceptions=True)
                
            logger.info(f"Warmed {len(features)} ML feature sets")
            
        except Exception as e:
            logger.error(f"ML features warming failed: {e}")
    
    async def warm_templates(self):
        """Warm cache with message templates"""
        
        if not self.db_manager:
            return
        
        try:
            # This would get from your templates table
            templates = []  # Placeholder
            
            warm_tasks = []
            for template in templates:
                key = f"template:{template.get('id')}"
                warm_tasks.append(
                    self.cache_manager.set(key, template, ttl=7200)  # 2 hours
                )
            
            if warm_tasks:
                await asyncio.gather(*warm_tasks, return_exceptions=True)
                
            logger.info(f"Warmed {len(templates)} templates")
            
        except Exception as e:
            logger.error(f"Template warming failed: {e}")
    
    async def warm_popular_content(self):
        """Warm cache with popular content"""
        
        try:
            # This would warm frequently accessed content
            # Based on access patterns, popular queries, etc.
            pass
            
        except Exception as e:
            logger.error(f"Popular content warming failed: {e}")
    
    async def stop_warming(self):
        """Stop cache warming"""
        
        for task_name, task in self.warming_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        logger.info("Cache warming stopped")


def cache_decorator(
    ttl: int = 300,
    key_prefix: str = None,
    cache_manager: CacheManager = None
):
    """
    Decorator for caching function results
    """
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get cache manager
            if cache_manager is None:
                # Try to get from global context or self
                if args and hasattr(args[0], 'cache_manager'):
                    cm = args[0].cache_manager
                else:
                    # Skip caching if no cache manager available
                    return await func(*args, **kwargs)
            else:
                cm = cache_manager
            
            # Generate cache key
            prefix = key_prefix or func.__name__
            cache_key = f"{prefix}:{cm.generate_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached = await cm.get(cache_key)
            if cached is not None:
                return cached
            
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Cache result
            if result is not None:
                await cm.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    
    return decorator


# Singleton instance
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get singleton cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        raise RuntimeError("Cache manager not initialized")
    return _cache_manager

def initialize_cache_manager(config: Dict) -> CacheManager:
    """Initialize singleton cache manager"""
    global _cache_manager
    _cache_manager = CacheManager(config)
    return _cache_manager
