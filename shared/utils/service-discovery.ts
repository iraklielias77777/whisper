import { redis } from '../index';
import { createServiceLogger } from './logger';

export interface ServiceInstance {
  id: string;
  name: string;
  host: string;
  port: number;
  status: 'healthy' | 'unhealthy' | 'unknown';
  lastHealthCheck: Date;
  metadata: Record<string, any>;
  endpoints: {
    health: string;
    metrics?: string;
    api?: string;
  };
}

export interface ServiceRegistration {
  name: string;
  host: string;
  port: number;
  metadata?: Record<string, any>;
  healthCheckInterval?: number;
}

export class ServiceDiscovery {
  private readonly serviceRegistry = 'service:registry';
  private readonly heartbeatInterval = 30000; // 30 seconds
  private readonly healthCheckTimeout = 5000; // 5 seconds
  private isRunning = false;
  private instanceId: string;
  private registeredService?: ServiceRegistration;
  private logger = createServiceLogger('service-discovery');

  constructor() {
    this.instanceId = `${process.env.HOSTNAME || 'localhost'}_${process.pid}_${Date.now()}`;
  }

  async registerService(registration: ServiceRegistration): Promise<string> {
    const instance: ServiceInstance = {
      id: this.instanceId,
      name: registration.name,
      host: registration.host,
      port: registration.port,
      status: 'unknown',
      lastHealthCheck: new Date(),
      metadata: registration.metadata || {},
      endpoints: {
        health: `http://${registration.host}:${registration.port}/health`,
        metrics: `http://${registration.host}:${registration.port}/metrics`,
        api: `http://${registration.host}:${registration.port}`
      }
    };

    try {
      await redis.hset(
        this.serviceRegistry,
        this.instanceId,
        JSON.stringify(instance)
      );

      this.registeredService = registration;

      this.logger.info('Service registered successfully', {
        instanceId: this.instanceId,
        serviceName: registration.name,
        host: registration.host,
        port: registration.port
      });

      // Start heartbeat
      this.startHeartbeat();

      return this.instanceId;
    } catch (error) {
      this.logger.error('Failed to register service', {
        error: error instanceof Error ? error.message : String(error),
        instanceId: this.instanceId,
        serviceName: registration.name
      });
      throw error;
    }
  }

  async unregisterService(): Promise<void> {
    try {
      await redis.hdel(this.serviceRegistry, this.instanceId);
      this.stopHeartbeat();

      this.logger.info('Service unregistered successfully', {
        instanceId: this.instanceId
      });
    } catch (error) {
      this.logger.error('Failed to unregister service', {
        error: error instanceof Error ? error.message : String(error),
        instanceId: this.instanceId
      });
      throw error;
    }
  }

  async discoverServices(serviceName?: string): Promise<ServiceInstance[]> {
    try {
      const services = await redis.hgetall(this.serviceRegistry);
      const instances: ServiceInstance[] = [];

      for (const [id, data] of Object.entries(services)) {
        try {
          const instance: ServiceInstance = JSON.parse(data);
          if (!serviceName || instance.name === serviceName) {
            instances.push(instance);
          }
        } catch (parseError) {
          this.logger.warn('Failed to parse service instance data', {
            instanceId: id,
            error: parseError instanceof Error ? parseError.message : String(parseError)
          });
        }
      }

      return instances;
    } catch (error) {
      this.logger.error('Failed to discover services', {
        error: error instanceof Error ? error.message : String(error),
        serviceName
      });
      return [];
    }
  }

  async getHealthyServices(serviceName: string): Promise<ServiceInstance[]> {
    const services = await this.discoverServices(serviceName);
    return services.filter(service => service.status === 'healthy');
  }

  async getServiceEndpoint(serviceName: string, preferredHost?: string): Promise<string | null> {
    const healthyServices = await this.getHealthyServices(serviceName);

    if (healthyServices.length === 0) {
      this.logger.warn('No healthy services found', { serviceName });
      return null;
    }

    // Prefer specific host if provided
    if (preferredHost) {
      const preferredService = healthyServices.find(s => s.host === preferredHost);
      if (preferredService) {
        return preferredService.endpoints.api!;
      }
    }

    // Round-robin selection (simple load balancing)
    const randomIndex = Math.floor(Math.random() * healthyServices.length);
    const selectedService = healthyServices[randomIndex];

    return selectedService.endpoints.api!;
  }

  private startHeartbeat(): void {
    if (this.isRunning) return;

    this.isRunning = true;
    this.performHeartbeat();
  }

  private stopHeartbeat(): void {
    this.isRunning = false;
  }

  private async performHeartbeat(): Promise<void> {
    if (!this.isRunning || !this.registeredService) return;

    try {
      // Update last heartbeat timestamp
      const instance = await this.getOwnInstance();
      if (instance) {
        instance.lastHealthCheck = new Date();
        
        // Perform health check
        const isHealthy = await this.performHealthCheck(instance);
        instance.status = isHealthy ? 'healthy' : 'unhealthy';

        await redis.hset(
          this.serviceRegistry,
          this.instanceId,
          JSON.stringify(instance)
        );

        this.logger.debug('Heartbeat sent', {
          instanceId: this.instanceId,
          status: instance.status
        });
      }
    } catch (error) {
      this.logger.error('Heartbeat failed', {
        error: error instanceof Error ? error.message : String(error),
        instanceId: this.instanceId
      });
    }

    // Schedule next heartbeat
    if (this.isRunning) {
      setTimeout(() => this.performHeartbeat(), this.heartbeatInterval);
    }
  }

  private async getOwnInstance(): Promise<ServiceInstance | null> {
    try {
      const data = await redis.hget(this.serviceRegistry, this.instanceId);
      return data ? JSON.parse(data) : null;
    } catch (error) {
      this.logger.error('Failed to get own instance', {
        error: error instanceof Error ? error.message : String(error),
        instanceId: this.instanceId
      });
      return null;
    }
  }

  private async performHealthCheck(instance: ServiceInstance): Promise<boolean> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.healthCheckTimeout);
      
      const response = await fetch(instance.endpoints.health, {
        method: 'GET',
        signal: controller.signal
      });

      clearTimeout(timeoutId);
      return response.ok;
    } catch (error) {
      this.logger.debug('Health check failed', {
        instanceId: instance.id,
        healthEndpoint: instance.endpoints.health,
        error: error instanceof Error ? error.message : String(error)
      });
      return false;
    }
  }

  async cleanupStaleServices(maxAge: number = 300000): Promise<number> {
    try {
      const services = await redis.hgetall(this.serviceRegistry);
      const now = Date.now();
      let cleaned = 0;

      for (const [id, data] of Object.entries(services)) {
        try {
          const instance: ServiceInstance = JSON.parse(data);
          const lastHeartbeat = new Date(instance.lastHealthCheck).getTime();

          if (now - lastHeartbeat > maxAge) {
            await redis.hdel(this.serviceRegistry, id);
            cleaned++;
            
            this.logger.info('Cleaned up stale service instance', {
              instanceId: id,
              serviceName: instance.name,
              lastHeartbeat: instance.lastHealthCheck
            });
          }
        } catch (parseError) {
          // Clean up corrupted entries
          await redis.hdel(this.serviceRegistry, id);
          cleaned++;
          
          this.logger.warn('Cleaned up corrupted service entry', {
            instanceId: id,
            error: parseError instanceof Error ? parseError.message : String(parseError)
          });
        }
      }

      if (cleaned > 0) {
        this.logger.info('Service cleanup completed', { cleanedServices: cleaned });
      }

      return cleaned;
    } catch (error) {
      this.logger.error('Failed to cleanup stale services', {
        error: error instanceof Error ? error.message : String(error)
      });
      return 0;
    }
  }

  async getServiceStats(): Promise<{
    totalServices: number;
    healthyServices: number;
    unhealthyServices: number;
    servicesByName: Record<string, number>;
  }> {
    const services = await this.discoverServices();
    
    const stats = {
      totalServices: services.length,
      healthyServices: services.filter(s => s.status === 'healthy').length,
      unhealthyServices: services.filter(s => s.status === 'unhealthy').length,
      servicesByName: {} as Record<string, number>
    };

    services.forEach(service => {
      stats.servicesByName[service.name] = (stats.servicesByName[service.name] || 0) + 1;
    });

    return stats;
  }

  // Graceful shutdown
  async shutdown(): Promise<void> {
    this.stopHeartbeat();
    if (this.registeredService) {
      await this.unregisterService();
    }
  }
}
