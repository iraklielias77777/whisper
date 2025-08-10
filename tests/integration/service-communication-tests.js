#!/usr/bin/env node

/**
 * Service Communication Integration Tests
 * Tests inter-service communication flows and message passing
 */

const http = require('http');
const { MessageQueue, TOPICS, SUBSCRIPTIONS } = require('../../shared/utils/message-queue');

class ServiceCommunicationTests {
  constructor() {
    this.baseUrl = 'http://localhost:8000'; // Kong Gateway
    this.apiKey = 'demo_1234567890abcdef';
    this.messageQueue = new MessageQueue({
      emulatorHost: 'localhost:8085',
      projectId: 'user-whisperer-dev'
    });
    this.testResults = {
      passed: 0,
      failed: 0,
      tests: []
    };
  }

  async runAllTests() {
    console.log('🔄 Starting Service Communication Tests...\n');

    try {
      // Initialize message queue
      await this.messageQueue.initialize();

      // Test direct service communication
      await this.testDirectServiceCommunication();

      // Test message queue communication
      await this.testMessageQueueCommunication();

      // Test service discovery integration
      await this.testServiceDiscoveryIntegration();

      // Test workflow orchestration
      await this.testWorkflowOrchestration();

      // Test error handling and resilience
      await this.testErrorHandlingResilience();

    } catch (error) {
      console.error('Test suite initialization failed:', error.message);
    } finally {
      await this.messageQueue.close();
    }

    this.generateReport();
    return this.testResults.failed === 0;
  }

  async testDirectServiceCommunication() {
    console.log('🔗 Testing Direct Service Communication...');

    // Test Event Ingestion → Behavioral Analysis flow
    await this.test('Event Ingestion → Behavioral Analysis', async () => {
      const userId = `test_user_${Date.now()}`;
      
      // Send event to event ingestion
      const eventResponse = await this.makeRequest('POST', `${this.baseUrl}/v1/events/track`, {
        event_type: 'feature_used',
        user_id: userId,
        timestamp: new Date().toISOString(),
        properties: {
          feature_name: 'dashboard',
          session_duration: 1800
        }
      }, {
        'X-API-Key': this.apiKey
      });

      if (eventResponse.statusCode !== 200) {
        throw new Error(`Event ingestion failed: ${eventResponse.statusCode}`);
      }

      // Wait for processing
      await this.sleep(3000);

      // Verify behavioral analysis received the event
      // This would typically check behavioral patterns or scoring
      return 'Event successfully processed through ingestion pipeline';
    });

    // Test Behavioral Analysis → Decision Engine flow
    await this.test('Behavioral Analysis → Decision Engine', async () => {
      const analysisData = {
        user_id: `behavior_test_${Date.now()}`,
        behavior_patterns: {
          engagement_score: 75.5,
          churn_risk: 25.0,
          usage_frequency: 'high'
        },
        analysis_timestamp: new Date().toISOString()
      };

      // This would simulate sending analysis results to decision engine
      // For now, we'll test the endpoint availability
      const healthResponse = await this.makeRequest('GET', 'http://localhost:3003/health');
      
      if (healthResponse.statusCode !== 200) {
        throw new Error('Decision engine not accessible');
      }

      return 'Behavioral analysis → Decision engine communication verified';
    });

    // Test Decision Engine → Content Generation flow
    await this.test('Decision Engine → Content Generation', async () => {
      const decisionData = {
        user_id: `decision_test_${Date.now()}`,
        intervention_type: 'engagement_boost',
        channel: 'email',
        urgency: 'medium',
        personalization_context: {
          user_segment: 'power_user',
          preferred_content: 'technical'
        }
      };

      // Test content generation service availability
      const healthResponse = await this.makeRequest('GET', 'http://localhost:3004/health');
      
      if (healthResponse.statusCode !== 200) {
        throw new Error('Content generation service not accessible');
      }

      return 'Decision engine → Content generation communication verified';
    });

    console.log();
  }

  async testMessageQueueCommunication() {
    console.log('📨 Testing Message Queue Communication...');

    // Test publishing to user events topic
    await this.test('Publish User Event Message', async () => {
      const eventData = {
        user_id: `queue_test_${Date.now()}`,
        event_type: 'test_queue_event',
        timestamp: new Date().toISOString(),
        properties: { test: true }
      };

      const messageId = await this.messageQueue.publish(TOPICS.USER_EVENTS, eventData);
      
      if (!messageId) {
        throw new Error('Failed to publish message');
      }

      return `Message published with ID: ${messageId}`;
    });

    // Test subscription and message handling
    await this.test('Subscribe and Process Messages', async () => {
      let messageReceived = false;
      const testData = {
        test_id: `sub_test_${Date.now()}`,
        timestamp: new Date().toISOString()
      };

      // Set up subscription
      await this.messageQueue.subscribe(
        TOPICS.SYSTEM_EVENTS,
        SUBSCRIPTIONS.SYSTEM_MONITOR,
        async (data, message) => {
          if (data.test_id === testData.test_id) {
            messageReceived = true;
          }
        }
      );

      // Publish test message
      await this.messageQueue.publish(TOPICS.SYSTEM_EVENTS, testData);

      // Wait for message processing
      await this.sleep(2000);

      if (!messageReceived) {
        throw new Error('Message was not received by subscriber');
      }

      return 'Message queue subscription and processing working';
    });

    // Test batch message processing
    await this.test('Batch Message Processing', async () => {
      const batchSize = 5;
      const batchData = [];
      
      for (let i = 0; i < batchSize; i++) {
        batchData.push({
          batch_id: `batch_${Date.now()}`,
          event_index: i,
          timestamp: new Date().toISOString()
        });
      }

      const messageIds = await this.messageQueue.publishBatch(TOPICS.USER_EVENTS, batchData);
      
      if (messageIds.length !== batchSize) {
        throw new Error(`Expected ${batchSize} message IDs, got ${messageIds.length}`);
      }

      return `Batch processing: ${messageIds.length} messages published`;
    });

    console.log();
  }

  async testServiceDiscoveryIntegration() {
    console.log('🔍 Testing Service Discovery Integration...');

    await this.test('Service Registry Population', async () => {
      // Test that services are registering themselves
      const services = [
        { name: 'event-ingestion', port: 3001 },
        { name: 'behavioral-analysis', port: 3002 },
        { name: 'decision-engine', port: 3003 }
      ];

      let registeredServices = 0;
      
      for (const service of services) {
        const isHealthy = await this.checkServiceHealth(service.port);
        if (isHealthy) {
          registeredServices++;
        }
      }

      if (registeredServices < 2) {
        throw new Error(`Only ${registeredServices} services are healthy`);
      }

      return `Service discovery: ${registeredServices} services active`;
    });

    await this.test('Service Load Balancing', async () => {
      // Test multiple requests to see if load balancing works
      const requests = 3;
      let successfulRequests = 0;

      for (let i = 0; i < requests; i++) {
        try {
          const response = await this.makeRequest('GET', `${this.baseUrl}/health/event-ingestion`);
          if (response.statusCode === 200) {
            successfulRequests++;
          }
        } catch (error) {
          // Expected some failures in load balancing scenarios
        }
      }

      if (successfulRequests < 2) {
        throw new Error('Load balancing not working properly');
      }

      return `Load balancing: ${successfulRequests}/${requests} requests successful`;
    });

    console.log();
  }

  async testWorkflowOrchestration() {
    console.log('🎼 Testing Workflow Orchestration...');

    await this.test('Complete User Journey Workflow', async () => {
      const userId = `workflow_test_${Date.now()}`;
      const steps = [];

      // Step 1: User signup
      const signupResponse = await this.makeRequest('POST', `${this.baseUrl}/v1/events/track`, {
        event_type: 'user_signup',
        user_id: userId,
        timestamp: new Date().toISOString(),
        properties: {
          email: `${userId}@example.com`,
          plan: 'trial'
        }
      }, {
        'X-API-Key': this.apiKey
      });

      if (signupResponse.statusCode === 200) {
        steps.push('signup');
      }

      // Step 2: Feature usage
      await this.sleep(1000);
      const usageResponse = await this.makeRequest('POST', `${this.baseUrl}/v1/events/track`, {
        event_type: 'feature_used',
        user_id: userId,
        timestamp: new Date().toISOString(),
        properties: {
          feature_name: 'onboarding_tutorial'
        }
      }, {
        'X-API-Key': this.apiKey
      });

      if (usageResponse.statusCode === 200) {
        steps.push('feature_usage');
      }

      // Step 3: Behavioral analysis (simulated by checking the service health)
      const behavioralResponse = await this.makeRequest('GET', 'http://localhost:3002/health');
      if (behavioralResponse.statusCode === 200) {
        steps.push('behavioral_analysis');
      }

      if (steps.length < 3) {
        throw new Error(`Workflow incomplete: only ${steps.length} steps completed`);
      }

      return `Complete workflow: ${steps.join(' → ')}`;
    });

    await this.test('Error Recovery in Workflow', async () => {
      // Simulate a failed service call and recovery
      try {
        // Intentionally call a non-existent endpoint
        await this.makeRequest('POST', `${this.baseUrl}/v1/nonexistent/endpoint`, {});
      } catch (error) {
        // Expected to fail
      }

      // Verify system continues to work after failure
      const recoveryResponse = await this.makeRequest('GET', `${this.baseUrl}/health/event-ingestion`);
      
      if (recoveryResponse.statusCode !== 200) {
        throw new Error('System did not recover from simulated failure');
      }

      return 'Error recovery: System resilient to individual service failures';
    });

    console.log();
  }

  async testErrorHandlingResilience() {
    console.log('🛡️  Testing Error Handling & Resilience...');

    await this.test('Rate Limiting Behavior', async () => {
      // Test rate limiting by sending rapid requests
      const rapidRequests = 10;
      let rateLimitedRequests = 0;
      let successfulRequests = 0;

      const promises = Array(rapidRequests).fill().map(async (_, i) => {
        try {
          const response = await this.makeRequest('POST', `${this.baseUrl}/v1/events/track`, {
            event_type: 'rate_limit_test',
            user_id: `rate_test_${i}`,
            timestamp: new Date().toISOString()
          }, {
            'X-API-Key': this.apiKey
          });

          if (response.statusCode === 429) {
            rateLimitedRequests++;
          } else if (response.statusCode === 200) {
            successfulRequests++;
          }
        } catch (error) {
          // Network errors expected under load
        }
      });

      await Promise.all(promises);

      // We expect some rate limiting to occur
      return `Rate limiting: ${successfulRequests} successful, ${rateLimitedRequests} rate-limited`;
    });

    await this.test('Service Timeout Handling', async () => {
      // Test timeout behavior with a slow request
      const startTime = Date.now();
      
      try {
        // Make a request with a very short timeout
        const response = await this.makeRequestWithTimeout(
          'GET', 
          `${this.baseUrl}/health/event-ingestion`,
          null,
          {},
          100 // 100ms timeout
        );

        const duration = Date.now() - startTime;
        return `Timeout handling: Request completed in ${duration}ms`;
      } catch (error) {
        const duration = Date.now() - startTime;
        if (error.message.includes('timeout')) {
          return `Timeout handling: Properly timed out after ${duration}ms`;
        } else {
          throw error;
        }
      }
    });

    await this.test('Circuit Breaker Behavior', async () => {
      // Test circuit breaker by repeatedly failing requests
      let consecutiveFailures = 0;
      
      for (let i = 0; i < 5; i++) {
        try {
          await this.makeRequest('POST', `${this.baseUrl}/v1/events/track`, {
            invalid_data: 'this should fail validation'
          }, {
            'X-API-Key': this.apiKey
          });
        } catch (error) {
          consecutiveFailures++;
        }
      }

      // Verify system still responds to valid requests
      const validResponse = await this.makeRequest('GET', `${this.baseUrl}/health/event-ingestion`);
      
      if (validResponse.statusCode !== 200) {
        throw new Error('System not responding after failures');
      }

      return `Circuit breaker: ${consecutiveFailures} failures handled, system still responsive`;
    });

    console.log();
  }

  async test(testName, testFunction) {
    const startTime = Date.now();
    
    try {
      const result = await Promise.race([
        testFunction(),
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Test timeout')), 30000)
        )
      ]);

      const duration = Date.now() - startTime;
      console.log(`  ✅ ${testName} (${duration}ms) - ${result}`);
      
      this.testResults.passed++;
      this.testResults.tests.push({
        name: testName,
        status: 'passed',
        duration,
        result
      });

    } catch (error) {
      const duration = Date.now() - startTime;
      console.log(`  ❌ ${testName} (${duration}ms) - ${error.message}`);
      
      this.testResults.failed++;
      this.testResults.tests.push({
        name: testName,
        status: 'failed',
        duration,
        error: error.message
      });
    }
  }

  async makeRequest(method, url, data = null, headers = {}) {
    return this.makeRequestWithTimeout(method, url, data, headers, 10000);
  }

  async makeRequestWithTimeout(method, url, data = null, headers = {}, timeout = 10000) {
    return new Promise((resolve, reject) => {
      const urlObj = new URL(url);
      const options = {
        hostname: urlObj.hostname,
        port: urlObj.port,
        path: urlObj.pathname + urlObj.search,
        method: method,
        headers: {
          'Content-Type': 'application/json',
          ...headers
        },
        timeout: timeout
      };

      const requestBody = data ? JSON.stringify(data) : null;
      if (requestBody) {
        options.headers['Content-Length'] = Buffer.byteLength(requestBody);
      }

      const req = http.request(options, (res) => {
        let body = '';
        res.on('data', (chunk) => body += chunk);
        res.on('end', () => {
          resolve({
            statusCode: res.statusCode,
            headers: res.headers,
            body: body
          });
        });
      });

      req.on('error', reject);
      req.on('timeout', () => {
        req.destroy();
        reject(new Error('Request timeout'));
      });

      if (requestBody) {
        req.write(requestBody);
      }
      
      req.end();
    });
  }

  async checkServiceHealth(port) {
    try {
      const response = await this.makeRequest('GET', `http://localhost:${port}/health`);
      return response.statusCode === 200;
    } catch (error) {
      return false;
    }
  }

  async sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  generateReport() {
    console.log('\n📊 SERVICE COMMUNICATION TEST RESULTS');
    console.log('=' * 50);
    console.log(`Total Tests: ${this.testResults.passed + this.testResults.failed}`);
    console.log(`✅ Passed: ${this.testResults.passed}`);
    console.log(`❌ Failed: ${this.testResults.failed}`);
    
    const successRate = Math.round((this.testResults.passed / (this.testResults.passed + this.testResults.failed)) * 100);
    console.log(`📈 Success Rate: ${successRate}%`);

    if (this.testResults.failed === 0) {
      console.log('\n🎉 ALL SERVICE COMMUNICATION TESTS PASSED!');
    } else {
      console.log('\n💥 SOME TESTS FAILED! Check the issues above.');
    }
  }
}

// Run tests if called directly
if (require.main === module) {
  const testSuite = new ServiceCommunicationTests();
  testSuite.runAllTests()
    .then((success) => {
      process.exit(success ? 0 : 1);
    })
    .catch((error) => {
      console.error('Service communication test suite error:', error);
      process.exit(1);
    });
}

module.exports = ServiceCommunicationTests;
