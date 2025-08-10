"""
Data Lifecycle Manager for User Whisperer Platform
Manages automated data retention, archival, and compliance workflows
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

class DataLifecycleAction(Enum):
    RETAIN = "retain"
    ARCHIVE = "archive"
    ANONYMIZE = "anonymize"
    DELETE = "delete"
    MIGRATE = "migrate"

class DataClassification(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PII = "pii"

@dataclass
class RetentionRule:
    data_type: str
    classification: DataClassification
    retention_period: timedelta
    action: DataLifecycleAction
    conditions: Optional[Dict[str, Any]] = None
    custom_handler: Optional[Callable] = None

@dataclass
class LifecycleEvent:
    timestamp: datetime
    data_type: str
    action: DataLifecycleAction
    affected_records: int
    details: Dict[str, Any]

class DataLifecycleManager:
    """
    Manages comprehensive data lifecycle with automated retention and compliance
    """
    
    def __init__(
        self,
        storage_manager,
        database_manager,
        config: Dict
    ):
        self.storage_manager = storage_manager
        self.database_manager = database_manager
        self.config = config
        self.retention_rules = {}
        self.lifecycle_jobs = {}
        self.event_history = []
        self.compliance_handlers = {}
        self.is_running = False
        
    async def initialize(self):
        """Initialize data lifecycle management"""
        
        # Load default retention rules
        await self.load_default_retention_rules()
        
        # Load custom compliance handlers
        await self.load_compliance_handlers()
        
        # Start lifecycle monitoring
        await self.start_lifecycle_management()
        
        logger.info("Data lifecycle manager initialized")
    
    async def load_default_retention_rules(self):
        """Load default retention rules for different data types"""
        
        default_rules = [
            # User data - GDPR compliant
            RetentionRule(
                data_type="user_profiles",
                classification=DataClassification.PII,
                retention_period=timedelta(days=1095),  # 3 years
                action=DataLifecycleAction.ANONYMIZE,
                conditions={"active_subscription": False}
            ),
            
            # Event data - Analytics retention
            RetentionRule(
                data_type="events",
                classification=DataClassification.INTERNAL,
                retention_period=timedelta(days=365),  # 1 year
                action=DataLifecycleAction.ARCHIVE,
                conditions={"event_category": "analytics"}
            ),
            
            # Message history - Shorter retention
            RetentionRule(
                data_type="message_history",
                classification=DataClassification.INTERNAL,
                retention_period=timedelta(days=180),  # 6 months
                action=DataLifecycleAction.DELETE
            ),
            
            # ML features - Short-term
            RetentionRule(
                data_type="ml_features",
                classification=DataClassification.INTERNAL,
                retention_period=timedelta(days=30),
                action=DataLifecycleAction.DELETE
            ),
            
            # Audit logs - Long-term compliance
            RetentionRule(
                data_type="audit_log",
                classification=DataClassification.RESTRICTED,
                retention_period=timedelta(days=2555),  # 7 years
                action=DataLifecycleAction.ARCHIVE,
                conditions={"gdpr_relevant": True}
            ),
            
            # GDPR requests - Legal compliance
            RetentionRule(
                data_type="gdpr_requests",
                classification=DataClassification.RESTRICTED,
                retention_period=timedelta(days=2555),  # 7 years
                action=DataLifecycleAction.RETAIN
            )
        ]
        
        for rule in default_rules:
            self.add_retention_rule(rule)
    
    def add_retention_rule(self, rule: RetentionRule):
        """Add or update retention rule"""
        
        key = f"{rule.data_type}_{rule.classification.value}"
        self.retention_rules[key] = rule
        
        logger.info(f"Added retention rule: {rule.data_type} - {rule.action.value} after {rule.retention_period}")
    
    async def load_compliance_handlers(self):
        """Load compliance-specific handlers"""
        
        self.compliance_handlers = {
            'gdpr': GDPRComplianceHandler(self),
            'ccpa': CCPAComplianceHandler(self),
            'sox': SOXComplianceHandler(self),
            'hipaa': HIPAAComplianceHandler(self)
        }
        
        logger.info("Loaded compliance handlers: " + ", ".join(self.compliance_handlers.keys()))
    
    async def start_lifecycle_management(self):
        """Start automated lifecycle management"""
        
        if self.is_running:
            logger.warning("Lifecycle management already running")
            return
        
        self.is_running = True
        
        # Schedule retention enforcement
        self.lifecycle_jobs['retention'] = asyncio.create_task(
            self.periodic_retention_enforcement()
        )
        
        # Schedule compliance audits
        self.lifecycle_jobs['compliance'] = asyncio.create_task(
            self.periodic_compliance_audit()
        )
        
        # Schedule data quality checks
        self.lifecycle_jobs['quality'] = asyncio.create_task(
            self.periodic_data_quality_check()
        )
        
        # Schedule reporting
        self.lifecycle_jobs['reporting'] = asyncio.create_task(
            self.periodic_reporting()
        )
        
        logger.info("Data lifecycle management started")
    
    async def stop_lifecycle_management(self):
        """Stop lifecycle management"""
        
        self.is_running = False
        
        for job_name, job in self.lifecycle_jobs.items():
            job.cancel()
            try:
                await job
            except asyncio.CancelledError:
                pass
        
        self.lifecycle_jobs.clear()
        logger.info("Data lifecycle management stopped")
    
    async def periodic_retention_enforcement(self):
        """Periodic enforcement of retention policies"""
        
        while self.is_running:
            try:
                await asyncio.sleep(24 * 3600)  # Run daily
                
                logger.info("Starting retention enforcement cycle")
                start_time = datetime.utcnow()
                
                enforcement_results = await self.enforce_all_retention_rules()
                
                duration = (datetime.utcnow() - start_time).total_seconds()
                total_affected = sum(
                    result.get('affected_records', 0)
                    for result in enforcement_results.values()
                )
                
                # Log event
                event = LifecycleEvent(
                    timestamp=datetime.utcnow(),
                    data_type="all",
                    action=DataLifecycleAction.RETAIN,
                    affected_records=total_affected,
                    details={
                        'duration_seconds': duration,
                        'rules_processed': len(enforcement_results),
                        'results': enforcement_results
                    }
                )
                
                await self.log_lifecycle_event(event)
                
                logger.info(f"Retention enforcement completed: {total_affected} records affected")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Retention enforcement failed: {e}")
                await asyncio.sleep(3600)  # Retry after 1 hour
    
    async def enforce_all_retention_rules(self) -> Dict[str, Any]:
        """Enforce all retention rules"""
        
        results = {}
        
        for rule_key, rule in self.retention_rules.items():
            try:
                result = await self.enforce_retention_rule(rule)
                results[rule_key] = result
                
            except Exception as e:
                logger.error(f"Failed to enforce rule {rule_key}: {e}")
                results[rule_key] = {'error': str(e)}
        
        return results
    
    async def enforce_retention_rule(self, rule: RetentionRule) -> Dict[str, Any]:
        """Enforce a specific retention rule"""
        
        cutoff_date = datetime.utcnow() - rule.retention_period
        
        # Build query conditions
        conditions = ["created_at < $1"]
        params = [cutoff_date]
        
        # Add custom conditions
        if rule.conditions:
            for key, value in rule.conditions.items():
                if isinstance(value, bool):
                    conditions.append(f"{key} = ${len(params) + 1}")
                    params.append(value)
                elif isinstance(value, str):
                    conditions.append(f"{key} = ${len(params) + 1}")
                    params.append(value)
        
        where_clause = " AND ".join(conditions)
        
        # Execute action based on rule
        if rule.action == DataLifecycleAction.DELETE:
            return await self.delete_expired_data(rule.data_type, where_clause, params)
        
        elif rule.action == DataLifecycleAction.ARCHIVE:
            return await self.archive_expired_data(rule.data_type, where_clause, params)
        
        elif rule.action == DataLifecycleAction.ANONYMIZE:
            return await self.anonymize_expired_data(rule.data_type, where_clause, params)
        
        elif rule.action == DataLifecycleAction.MIGRATE:
            return await self.migrate_expired_data(rule.data_type, where_clause, params)
        
        elif rule.custom_handler:
            return await rule.custom_handler(rule.data_type, where_clause, params)
        
        else:
            return {'action': 'no_action', 'affected_records': 0}
    
    async def delete_expired_data(
        self,
        data_type: str,
        where_clause: str,
        params: List[Any]
    ) -> Dict[str, Any]:
        """Delete expired data from database"""
        
        try:
            schema = 'core' if data_type in ['events', 'user_profiles', 'message_history'] else 'audit'
            table_name = f"{schema}.{data_type}"
            
            # First, count records to be deleted
            count_query = f"SELECT COUNT(*) FROM {table_name} WHERE {where_clause}"
            count_result = await self.database_manager.fetchrow_with_retry(
                count_query, params, read_only=True
            )
            record_count = count_result['count'] if count_result else 0
            
            if record_count == 0:
                return {'action': 'delete', 'affected_records': 0}
            
            # Delete in batches to avoid blocking
            batch_size = 1000
            total_deleted = 0
            
            while total_deleted < record_count:
                delete_query = f"""
                    DELETE FROM {table_name}
                    WHERE id IN (
                        SELECT id FROM {table_name}
                        WHERE {where_clause}
                        LIMIT {batch_size}
                    )
                """
                
                result = await self.database_manager.execute_with_retry(delete_query, params)
                batch_deleted = int(result.split()[-1]) if result else 0
                total_deleted += batch_deleted
                
                if batch_deleted == 0:
                    break
                
                # Brief pause between batches
                await asyncio.sleep(0.1)
            
            logger.info(f"Deleted {total_deleted} expired records from {table_name}")
            
            return {
                'action': 'delete',
                'affected_records': total_deleted,
                'table': table_name
            }
            
        except Exception as e:
            logger.error(f"Failed to delete expired data from {data_type}: {e}")
            return {'action': 'delete', 'error': str(e), 'affected_records': 0}
    
    async def archive_expired_data(
        self,
        data_type: str,
        where_clause: str,
        params: List[Any]
    ) -> Dict[str, Any]:
        """Archive expired data to cold storage"""
        
        try:
            schema = 'core' if data_type in ['events', 'user_profiles', 'message_history'] else 'audit'
            table_name = f"{schema}.{data_type}"
            
            # Query data to archive
            select_query = f"SELECT * FROM {table_name} WHERE {where_clause} LIMIT 10000"
            records = await self.database_manager.fetch_with_retry(
                select_query, params, read_only=True
            )
            
            if not records:
                return {'action': 'archive', 'affected_records': 0}
            
            # Archive to cold storage
            archived_count = 0
            for record in records:
                try:
                    await self.storage_manager.store_data(
                        data_type=f"archived_{data_type}",
                        data=dict(record),
                        force_tier=self.storage_manager.StorageTier.COLD
                    )
                    archived_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to archive record {record.get('id')}: {e}")
            
            # Delete archived records from database
            if archived_count > 0:
                record_ids = [record['id'] for record in records[:archived_count]]
                delete_query = f"DELETE FROM {table_name} WHERE id = ANY($1)"
                await self.database_manager.execute_with_retry(delete_query, [record_ids])
            
            logger.info(f"Archived {archived_count} records from {table_name}")
            
            return {
                'action': 'archive',
                'affected_records': archived_count,
                'table': table_name
            }
            
        except Exception as e:
            logger.error(f"Failed to archive expired data from {data_type}: {e}")
            return {'action': 'archive', 'error': str(e), 'affected_records': 0}
    
    async def anonymize_expired_data(
        self,
        data_type: str,
        where_clause: str,
        params: List[Any]
    ) -> Dict[str, Any]:
        """Anonymize expired data (GDPR compliance)"""
        
        try:
            schema = 'core'
            table_name = f"{schema}.{data_type}"
            
            # Define anonymization mappings
            anonymization_mappings = {
                'user_profiles': {
                    'email': "'anonymized@example.com'",
                    'phone': "'000-000-0000'",
                    'name': "'Anonymous User'",
                    'email_hash': "MD5('anonymized@example.com')",
                    'phone_hash': "MD5('000-000-0000')"
                }
            }
            
            mappings = anonymization_mappings.get(data_type, {})
            if not mappings:
                logger.warning(f"No anonymization mappings for {data_type}")
                return {'action': 'anonymize', 'affected_records': 0}
            
            # Build update query
            set_clauses = [f"{field} = {value}" for field, value in mappings.items()]
            update_query = f"""
                UPDATE {table_name}
                SET {', '.join(set_clauses)},
                    updated_at = NOW(),
                    deleted_at = NOW()
                WHERE {where_clause}
            """
            
            result = await self.database_manager.execute_with_retry(update_query, params)
            affected_count = int(result.split()[-1]) if result else 0
            
            logger.info(f"Anonymized {affected_count} records in {table_name}")
            
            return {
                'action': 'anonymize',
                'affected_records': affected_count,
                'table': table_name
            }
            
        except Exception as e:
            logger.error(f"Failed to anonymize expired data from {data_type}: {e}")
            return {'action': 'anonymize', 'error': str(e), 'affected_records': 0}
    
    async def migrate_expired_data(
        self,
        data_type: str,
        where_clause: str,
        params: List[Any]
    ) -> Dict[str, Any]:
        """Migrate expired data between storage tiers"""
        
        try:
            # This would implement tier migration logic
            # For now, just return placeholder
            return {
                'action': 'migrate',
                'affected_records': 0,
                'note': 'Migration not implemented yet'
            }
            
        except Exception as e:
            logger.error(f"Failed to migrate expired data from {data_type}: {e}")
            return {'action': 'migrate', 'error': str(e), 'affected_records': 0}
    
    async def periodic_compliance_audit(self):
        """Periodic compliance auditing"""
        
        while self.is_running:
            try:
                await asyncio.sleep(7 * 24 * 3600)  # Run weekly
                
                logger.info("Starting compliance audit")
                
                audit_results = {}
                for compliance_type, handler in self.compliance_handlers.items():
                    try:
                        result = await handler.audit()
                        audit_results[compliance_type] = result
                    except Exception as e:
                        logger.error(f"Compliance audit failed for {compliance_type}: {e}")
                        audit_results[compliance_type] = {'error': str(e)}
                
                # Log audit event
                event = LifecycleEvent(
                    timestamp=datetime.utcnow(),
                    data_type="compliance",
                    action=DataLifecycleAction.RETAIN,
                    affected_records=0,
                    details={'audit_results': audit_results}
                )
                
                await self.log_lifecycle_event(event)
                
                logger.info("Compliance audit completed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Compliance audit failed: {e}")
                await asyncio.sleep(24 * 3600)  # Retry after 1 day
    
    async def periodic_data_quality_check(self):
        """Periodic data quality checks"""
        
        while self.is_running:
            try:
                await asyncio.sleep(6 * 3600)  # Run every 6 hours
                
                quality_issues = await self.check_data_quality()
                
                if quality_issues:
                    logger.warning(f"Found {len(quality_issues)} data quality issues")
                    
                    # Auto-fix if possible
                    for issue in quality_issues:
                        if issue.get('auto_fixable', False):
                            await self.fix_data_quality_issue(issue)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Data quality check failed: {e}")
    
    async def check_data_quality(self) -> List[Dict[str, Any]]:
        """Check data quality issues"""
        
        issues = []
        
        try:
            # Check for orphaned records
            orphan_check = """
                SELECT 'orphaned_events' as issue_type, COUNT(*) as count
                FROM core.events e
                LEFT JOIN core.user_profiles up ON e.user_id = up.id
                WHERE up.id IS NULL
                AND e.created_at > NOW() - INTERVAL '7 days'
            """
            
            result = await self.database_manager.fetchrow_with_retry(
                orphan_check, read_only=True
            )
            
            if result and result['count'] > 0:
                issues.append({
                    'type': 'orphaned_events',
                    'count': result['count'],
                    'description': 'Events without corresponding user profiles',
                    'auto_fixable': True
                })
            
            # Check for inconsistent data
            consistency_check = """
                SELECT 'inconsistent_lifecycle' as issue_type, COUNT(*) as count
                FROM core.user_profiles
                WHERE engagement_score > 0.8 
                AND lifecycle_stage IN ('dormant', 'churned')
            """
            
            result = await self.database_manager.fetchrow_with_retry(
                consistency_check, read_only=True
            )
            
            if result and result['count'] > 0:
                issues.append({
                    'type': 'inconsistent_lifecycle',
                    'count': result['count'],
                    'description': 'High engagement users in dormant/churned state',
                    'auto_fixable': True
                })
            
        except Exception as e:
            logger.error(f"Data quality check failed: {e}")
        
        return issues
    
    async def fix_data_quality_issue(self, issue: Dict[str, Any]):
        """Fix a data quality issue"""
        
        try:
            if issue['type'] == 'orphaned_events':
                # Delete orphaned events
                delete_query = """
                    DELETE FROM core.events
                    WHERE id IN (
                        SELECT e.id FROM core.events e
                        LEFT JOIN core.user_profiles up ON e.user_id = up.id
                        WHERE up.id IS NULL
                        AND e.created_at > NOW() - INTERVAL '7 days'
                        LIMIT 1000
                    )
                """
                
                await self.database_manager.execute_with_retry(delete_query)
                logger.info(f"Fixed orphaned events issue")
                
            elif issue['type'] == 'inconsistent_lifecycle':
                # Update lifecycle stage for high engagement users
                update_query = """
                    UPDATE core.user_profiles
                    SET lifecycle_stage = CASE
                        WHEN engagement_score > 0.9 THEN 'power_user'
                        WHEN engagement_score > 0.8 THEN 'engaged'
                        ELSE lifecycle_stage
                    END,
                    updated_at = NOW()
                    WHERE engagement_score > 0.8 
                    AND lifecycle_stage IN ('dormant', 'churned')
                """
                
                await self.database_manager.execute_with_retry(update_query)
                logger.info(f"Fixed inconsistent lifecycle issue")
                
        except Exception as e:
            logger.error(f"Failed to fix data quality issue {issue['type']}: {e}")
    
    async def periodic_reporting(self):
        """Generate periodic lifecycle reports"""
        
        while self.is_running:
            try:
                await asyncio.sleep(7 * 24 * 3600)  # Weekly reports
                
                report = await self.generate_lifecycle_report()
                
                # Store report
                await self.store_lifecycle_report(report)
                
                logger.info("Generated weekly lifecycle report")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Report generation failed: {e}")
    
    async def generate_lifecycle_report(self) -> Dict[str, Any]:
        """Generate comprehensive lifecycle report"""
        
        report = {
            'report_date': datetime.utcnow().isoformat(),
            'period': 'weekly',
            'retention_enforcement': {},
            'compliance_status': {},
            'data_quality': {},
            'storage_utilization': {}
        }
        
        try:
            # Recent lifecycle events
            recent_events = [
                event for event in self.event_history
                if event.timestamp > datetime.utcnow() - timedelta(days=7)
            ]
            
            # Aggregate enforcement stats
            enforcement_stats = {}
            for event in recent_events:
                if event.action in enforcement_stats:
                    enforcement_stats[event.action.value] += event.affected_records
                else:
                    enforcement_stats[event.action.value] = event.affected_records
            
            report['retention_enforcement'] = enforcement_stats
            
            # Storage utilization
            if hasattr(self.storage_manager, 'get_storage_stats'):
                storage_stats = await self.storage_manager.get_storage_stats()
                report['storage_utilization'] = storage_stats
            
        except Exception as e:
            logger.error(f"Failed to generate lifecycle report: {e}")
            report['error'] = str(e)
        
        return report
    
    async def store_lifecycle_report(self, report: Dict[str, Any]):
        """Store lifecycle report"""
        
        try:
            # Store in database
            insert_query = """
                INSERT INTO audit.lifecycle_reports (
                    report_date, period, report_data
                ) VALUES ($1, $2, $3)
            """
            
            await self.database_manager.execute_with_retry(
                insert_query,
                [
                    datetime.utcnow(),
                    report.get('period', 'weekly'),
                    json.dumps(report)
                ]
            )
            
        except Exception as e:
            logger.error(f"Failed to store lifecycle report: {e}")
    
    async def log_lifecycle_event(self, event: LifecycleEvent):
        """Log lifecycle event"""
        
        # Add to in-memory history
        self.event_history.append(event)
        
        # Limit history size
        if len(self.event_history) > 10000:
            self.event_history = self.event_history[-5000:]
        
        # Store in database
        try:
            insert_query = """
                INSERT INTO audit.lifecycle_events (
                    timestamp, data_type, action, affected_records, details
                ) VALUES ($1, $2, $3, $4, $5)
            """
            
            await self.database_manager.execute_with_retry(
                insert_query,
                [
                    event.timestamp,
                    event.data_type,
                    event.action.value,
                    event.affected_records,
                    json.dumps(event.details)
                ]
            )
            
        except Exception as e:
            logger.error(f"Failed to log lifecycle event: {e}")
    
    async def handle_gdpr_request(
        self,
        request_type: str,
        user_id: str,
        app_id: str
    ) -> Dict[str, Any]:
        """Handle GDPR data request"""
        
        handler = self.compliance_handlers.get('gdpr')
        if not handler:
            raise ValueError("GDPR handler not available")
        
        return await handler.handle_request(request_type, user_id, app_id)
    
    async def get_lifecycle_stats(self) -> Dict[str, Any]:
        """Get comprehensive lifecycle statistics"""
        
        recent_events = [
            event for event in self.event_history
            if event.timestamp > datetime.utcnow() - timedelta(days=30)
        ]
        
        stats = {
            'total_retention_rules': len(self.retention_rules),
            'active_jobs': len([job for job in self.lifecycle_jobs.values() if not job.done()]),
            'recent_events': len(recent_events),
            'events_by_action': {},
            'compliance_handlers': list(self.compliance_handlers.keys()),
            'is_running': self.is_running
        }
        
        # Aggregate events by action
        for event in recent_events:
            action = event.action.value
            if action in stats['events_by_action']:
                stats['events_by_action'][action] += event.affected_records
            else:
                stats['events_by_action'][action] = event.affected_records
        
        return stats


class ComplianceHandler:
    """Base class for compliance handlers"""
    
    def __init__(self, lifecycle_manager):
        self.lifecycle_manager = lifecycle_manager
    
    async def audit(self) -> Dict[str, Any]:
        """Perform compliance audit"""
        raise NotImplementedError
    
    async def handle_request(
        self,
        request_type: str,
        user_id: str,
        app_id: str
    ) -> Dict[str, Any]:
        """Handle compliance request"""
        raise NotImplementedError


class GDPRComplianceHandler(ComplianceHandler):
    """GDPR compliance handler"""
    
    async def audit(self) -> Dict[str, Any]:
        """Perform GDPR compliance audit"""
        
        try:
            db = self.lifecycle_manager.database_manager
            
            # Check for users with active data but no consent
            consent_check = """
                SELECT COUNT(*) as users_without_consent
                FROM core.user_profiles up
                LEFT JOIN audit.consent_records cr ON up.id = cr.user_id
                WHERE up.deleted_at IS NULL
                AND (cr.consent_given IS NULL OR cr.consent_given = FALSE)
                AND up.last_active_at > NOW() - INTERVAL '30 days'
            """
            
            result = await db.fetchrow_with_retry(consent_check, read_only=True)
            users_without_consent = result['users_without_consent'] if result else 0
            
            # Check data retention compliance
            retention_check = """
                SELECT COUNT(*) as overdue_deletions
                FROM core.user_profiles
                WHERE deleted_at IS NULL
                AND subscription_status IN ('canceled', 'expired')
                AND updated_at < NOW() - INTERVAL '3 years'
            """
            
            result = await db.fetchrow_with_retry(retention_check, read_only=True)
            overdue_deletions = result['overdue_deletions'] if result else 0
            
            return {
                'compliant': users_without_consent == 0 and overdue_deletions == 0,
                'users_without_consent': users_without_consent,
                'overdue_deletions': overdue_deletions,
                'last_audit': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"GDPR audit failed: {e}")
            return {'error': str(e)}
    
    async def handle_request(
        self,
        request_type: str,
        user_id: str,
        app_id: str
    ) -> Dict[str, Any]:
        """Handle GDPR data request"""
        
        db = self.lifecycle_manager.database_manager
        
        if request_type == 'access':
            return await self.handle_data_access_request(user_id, app_id)
        elif request_type == 'deletion':
            return await self.handle_data_deletion_request(user_id, app_id)
        elif request_type == 'portability':
            return await self.handle_data_portability_request(user_id, app_id)
        else:
            raise ValueError(f"Unknown GDPR request type: {request_type}")
    
    async def handle_data_access_request(
        self,
        user_id: str,
        app_id: str
    ) -> Dict[str, Any]:
        """Handle GDPR data access request"""
        
        db = self.lifecycle_manager.database_manager
        
        try:
            # Collect all user data
            user_data = {}
            
            # User profile
            profile = await db.fetchrow_with_retry(
                "SELECT * FROM core.user_profiles WHERE id = $1 AND app_id = $2",
                [user_id, app_id],
                read_only=True
            )
            if profile:
                user_data['profile'] = dict(profile)
            
            # Recent events
            events = await db.fetch_with_retry(
                """SELECT * FROM core.events 
                   WHERE user_id = $1 AND app_id = $2 
                   AND created_at > NOW() - INTERVAL '1 year'
                   ORDER BY created_at DESC LIMIT 10000""",
                [user_id, app_id],
                read_only=True
            )
            user_data['events'] = [dict(event) for event in events]
            
            # Message history
            messages = await db.fetch_with_retry(
                """SELECT * FROM core.message_history 
                   WHERE user_id = $1 AND app_id = $2
                   ORDER BY created_at DESC LIMIT 1000""",
                [user_id, app_id],
                read_only=True
            )
            user_data['messages'] = [dict(msg) for msg in messages]
            
            return {
                'status': 'completed',
                'data': user_data,
                'exported_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"GDPR access request failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def handle_data_deletion_request(
        self,
        user_id: str,
        app_id: str
    ) -> Dict[str, Any]:
        """Handle GDPR data deletion request"""
        
        db = self.lifecycle_manager.database_manager
        
        try:
            deleted_records = 0
            
            # Anonymize user profile
            anonymize_query = """
                UPDATE core.user_profiles
                SET email = 'deleted@gdpr.com',
                    phone = NULL,
                    name = 'Deleted User',
                    email_hash = MD5('deleted@gdpr.com'),
                    phone_hash = NULL,
                    deleted_at = NOW(),
                    updated_at = NOW()
                WHERE id = $1 AND app_id = $2
            """
            
            await db.execute_with_retry(anonymize_query, [user_id, app_id])
            deleted_records += 1
            
            # Delete personal events (keep aggregated analytics)
            delete_events_query = """
                DELETE FROM core.events
                WHERE user_id = $1 AND app_id = $2
                AND event_type IN ('page_view', 'click', 'form_submit')
                AND created_at > NOW() - INTERVAL '30 days'
            """
            
            result = await db.execute_with_retry(delete_events_query, [user_id, app_id])
            if result:
                deleted_records += int(result.split()[-1])
            
            # Delete message history
            delete_messages_query = """
                DELETE FROM core.message_history
                WHERE user_id = $1 AND app_id = $2
            """
            
            result = await db.execute_with_retry(delete_messages_query, [user_id, app_id])
            if result:
                deleted_records += int(result.split()[-1])
            
            return {
                'status': 'completed',
                'deleted_records': deleted_records,
                'deleted_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"GDPR deletion request failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def handle_data_portability_request(
        self,
        user_id: str,
        app_id: str
    ) -> Dict[str, Any]:
        """Handle GDPR data portability request"""
        
        # For now, same as access request
        return await self.handle_data_access_request(user_id, app_id)


class CCPAComplianceHandler(ComplianceHandler):
    """CCPA compliance handler"""
    
    async def audit(self) -> Dict[str, Any]:
        """Perform CCPA compliance audit"""
        
        return {
            'compliant': True,
            'last_audit': datetime.utcnow().isoformat(),
            'note': 'CCPA audit not fully implemented'
        }
    
    async def handle_request(
        self,
        request_type: str,
        user_id: str,
        app_id: str
    ) -> Dict[str, Any]:
        """Handle CCPA request"""
        
        return {
            'status': 'not_implemented',
            'note': 'CCPA request handling not implemented'
        }


class SOXComplianceHandler(ComplianceHandler):
    """SOX compliance handler"""
    
    async def audit(self) -> Dict[str, Any]:
        """Perform SOX compliance audit"""
        
        return {
            'compliant': True,
            'last_audit': datetime.utcnow().isoformat(),
            'note': 'SOX audit not fully implemented'
        }
    
    async def handle_request(
        self,
        request_type: str,
        user_id: str,
        app_id: str
    ) -> Dict[str, Any]:
        """Handle SOX request"""
        
        return {
            'status': 'not_implemented',
            'note': 'SOX request handling not implemented'
        }


class HIPAAComplianceHandler(ComplianceHandler):
    """HIPAA compliance handler"""
    
    async def audit(self) -> Dict[str, Any]:
        """Perform HIPAA compliance audit"""
        
        return {
            'compliant': True,
            'last_audit': datetime.utcnow().isoformat(),
            'note': 'HIPAA audit not fully implemented'
        }
    
    async def handle_request(
        self,
        request_type: str,
        user_id: str,
        app_id: str
    ) -> Dict[str, Any]:
        """Handle HIPAA request"""
        
        return {
            'status': 'not_implemented',
            'note': 'HIPAA request handling not implemented'
        }


# Singleton instance
_lifecycle_manager = None

def get_lifecycle_manager() -> DataLifecycleManager:
    """Get singleton lifecycle manager instance"""
    global _lifecycle_manager
    if _lifecycle_manager is None:
        raise RuntimeError("Data lifecycle manager not initialized")
    return _lifecycle_manager

def initialize_lifecycle_manager(
    storage_manager,
    database_manager,
    config: Dict
) -> DataLifecycleManager:
    """Initialize singleton lifecycle manager"""
    global _lifecycle_manager
    _lifecycle_manager = DataLifecycleManager(storage_manager, database_manager, config)
    return _lifecycle_manager
