import { logger } from '@userwhisperer/shared';

export interface BackpressureConfig {
  maxQueueSize: number;
  batchSize: number;
  highWaterMark?: number;
  lowWaterMark?: number;
  processingDelayMs?: number;
}

export interface BackpressureStatus {
  queueSize: number;
  maxQueueSize: number;
  isPaused: boolean;
  isProcessing: boolean;
  throughput: number; // items per second
  averageProcessingTime: number; // ms
}

export class BackpressureManager {
  private queue: any[] = [];
  private processing: boolean = false;
  private maxQueueSize: number;
  private batchSize: number;
  private highWaterMark: number;
  private lowWaterMark: number;
  private processingDelayMs: number;
  private isPaused: boolean = false;
  
  // Metrics
  private processedCount: number = 0;
  private totalProcessingTime: number = 0;
  private lastThroughputCalculation: number = Date.now();
  private lastProcessedCount: number = 0;
  
  constructor(config: BackpressureConfig) {
    this.maxQueueSize = config.maxQueueSize;
    this.batchSize = config.batchSize;
    this.highWaterMark = config.highWaterMark || Math.floor(config.maxQueueSize * 0.8);
    this.lowWaterMark = config.lowWaterMark || Math.floor(config.maxQueueSize * 0.2);
    this.processingDelayMs = config.processingDelayMs || 10;
  }
  
  async addToQueue(item: any): Promise<void> {
    if (this.queue.length >= this.maxQueueSize) {
      throw new Error('Queue is full - rejecting item');
    }
    
    this.queue.push(item);
    
    // Check high water mark
    if (this.queue.length >= this.highWaterMark && !this.isPaused) {
      this.pauseIngestion();
    }
    
    if (!this.processing) {
      this.startProcessing();
    }
  }
  
  async addBatchToQueue(items: any[]): Promise<{ accepted: number; rejected: number }> {
    const result = { accepted: 0, rejected: 0 };
    
    for (const item of items) {
      if (this.queue.length >= this.maxQueueSize) {
        result.rejected++;
      } else {
        this.queue.push(item);
        result.accepted++;
      }
    }
    
    // Check high water mark
    if (this.queue.length >= this.highWaterMark && !this.isPaused) {
      this.pauseIngestion();
    }
    
    if (!this.processing && result.accepted > 0) {
      this.startProcessing();
    }
    
    return result;
  }
  
  private async startProcessing(): Promise<void> {
    this.processing = true;
    
    logger.debug('BackpressureManager: Starting processing', {
      queueSize: this.queue.length,
      batchSize: this.batchSize
    });
    
    while (this.queue.length > 0) {
      const startTime = Date.now();
      
      // Extract batch
      const batch = this.queue.splice(0, Math.min(this.batchSize, this.queue.length));
      
      try {
        // Process batch (this would be overridden or injected)
        await this.processBatch(batch);
        
        // Update metrics
        const processingTime = Date.now() - startTime;
        this.processedCount += batch.length;
        this.totalProcessingTime += processingTime;
        
        logger.debug('BackpressureManager: Processed batch', {
          batchSize: batch.length,
          processingTime: `${processingTime}ms`,
          remainingInQueue: this.queue.length
        });
        
      } catch (error) {
        logger.error('BackpressureManager: Batch processing failed', {
          error: error instanceof Error ? error.message : String(error),
          batchSize: batch.length,
          remainingInQueue: this.queue.length
        });
        
        // Re-queue failed items at the front
        this.queue.unshift(...batch);
        
        // Add delay before retrying
        await this.delay(1000);
      }
      
      // Check low water mark
      if (this.queue.length <= this.lowWaterMark && this.isPaused) {
        this.resumeIngestion();
      }
      
      // Small delay to prevent CPU spinning
      if (this.queue.length > 0) {
        await this.delay(this.processingDelayMs);
      }
    }
    
    this.processing = false;
    logger.debug('BackpressureManager: Processing completed', {
      totalProcessed: this.processedCount
    });
  }
  
  private async processBatch(batch: any[]): Promise<void> {
    // This is a placeholder - in real implementation, this would be injected
    // or the BackpressureManager would be extended with actual processing logic
    logger.debug('BackpressureManager: Processing batch (placeholder)', {
      batchSize: batch.length
    });
    
    // Simulate processing time
    await this.delay(Math.random() * 100);
  }
  
  private pauseIngestion(): void {
    this.isPaused = true;
    logger.warn('BackpressureManager: Pausing ingestion', {
      queueSize: this.queue.length,
      highWaterMark: this.highWaterMark
    });
    
    // Emit event or call callback to signal upstream to slow down
    this.notifyUpstream('pause');
  }
  
  private resumeIngestion(): void {
    this.isPaused = false;
    logger.info('BackpressureManager: Resuming ingestion', {
      queueSize: this.queue.length,
      lowWaterMark: this.lowWaterMark
    });
    
    // Emit event or call callback to signal upstream to resume
    this.notifyUpstream('resume');
  }
  
  private notifyUpstream(action: 'pause' | 'resume'): void {
    // This would typically emit an event or call a callback
    // For now, just log the action
    logger.info(`BackpressureManager: Notifying upstream to ${action}`);
  }
  
  private async delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
  
  public getStatus(): BackpressureStatus {
    const now = Date.now();
    const timeSinceLastCalculation = now - this.lastThroughputCalculation;
    
    // Calculate throughput (items per second)
    let throughput = 0;
    if (timeSinceLastCalculation > 0) {
      const itemsProcessed = this.processedCount - this.lastProcessedCount;
      throughput = (itemsProcessed / timeSinceLastCalculation) * 1000;
    }
    
    // Update last calculation values
    this.lastThroughputCalculation = now;
    this.lastProcessedCount = this.processedCount;
    
    // Calculate average processing time
    const averageProcessingTime = this.processedCount > 0 
      ? this.totalProcessingTime / this.processedCount 
      : 0;
    
    return {
      queueSize: this.queue.length,
      maxQueueSize: this.maxQueueSize,
      isPaused: this.isPaused,
      isProcessing: this.processing,
      throughput: Math.round(throughput * 100) / 100, // Round to 2 decimal places
      averageProcessingTime: Math.round(averageProcessingTime * 100) / 100
    };
  }
  
  public getMetrics(): {
    totalProcessed: number;
    averageProcessingTime: number;
    queueUtilization: number;
    isPaused: boolean;
  } {
    return {
      totalProcessed: this.processedCount,
      averageProcessingTime: this.processedCount > 0 
        ? this.totalProcessingTime / this.processedCount 
        : 0,
      queueUtilization: this.queue.length / this.maxQueueSize,
      isPaused: this.isPaused
    };
  }
  
  public clear(): void {
    this.queue = [];
    this.processing = false;
    this.isPaused = false;
    
    logger.info('BackpressureManager: Queue cleared');
  }
  
  public forceResume(): void {
    this.isPaused = false;
    
    if (this.queue.length > 0 && !this.processing) {
      this.startProcessing();
    }
    
    logger.info('BackpressureManager: Force resumed processing');
  }
  
  // Health check
  public isHealthy(): boolean {
    return !this.isPaused && this.queue.length < this.highWaterMark;
  }
} 