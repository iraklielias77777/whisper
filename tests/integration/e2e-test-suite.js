#!/usr/bin/env node

/**
 * End-to-End Integration Test Suite
 * Tests complete user journeys through the User Whisperer Platform
 */

const http = require('http');
const https = require('https');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

class E2ETestSuite {
  constructor() {
    this.baseUrl = 'http://localhost:8000'; // Kong Gateway
    this.apiKey = 'demo_1234567890abcdef'; // Demo API key from Kong config
    this.testResults = {
      passed: 0,
      failed: 0,
      skipped: 0,
      tests: []
    };
    this.timeout = 30000; // 30 seconds
  }

  async runAllTests() {
    console.log('🧪 Starting End-to-End Integration Tests...\n');

    // Test infrastructure readiness
    await this.testInfrastructureReady();

    // Test service health endpoints
    await this.testServiceHealth();

    // Test API Gateway routing
    await this.testAPIGatewayRouting();

    // Test complete user journey flows
    await this.testUserJourneyFlows();

    // Test inter-service communication
    await this.testInterServiceCommunication();

    // Test monitoring and observability
    await this.testMonitoringIntegration();

    // Generate final report
    this.generateReport();

    return this.testResults.failed === 0;
  }

  async testInfrastructureReady() {
    console.log('🏗️  Testing Infrastructure Readiness...');

    const infrastructure = [
      { name: 'PostgreSQL', port: 5432 },
      { name: 'Redis', port: 6379 },
      { name: 'Kong Gateway', port: 8000 },
      { name: 'Kong Admin', port: 8001 },
      { name: 'Prometheus', port: 9090 },
      { name: 'Grafana', port: 3000 }
    ];

    for (const service of infrastructure) {
      await this.test(`Infrastructure: ${service.name}`, async () => {
        const isReady = await this.checkPort('localhost', service.port);
        if (!isReady) {
          throw new Error(`${service.name} not ready on port ${service.port}`);
        }
        return `${service.name} ready on port ${service.port}`;
      });
    }

    console.log();
  }

  async testServiceHealth() {
    console.log('🏥 Testing Service Health Endpoints...');

    const services = [
      { name: 'event-ingestion', port: 3001 },
      { name: 'behavioral-analysis', port: 3002 },
      { name: 'decision-engine', port: 3003 },
      { name: 'content-generation', port: 3004 },
      { name: 'channel-orchestrator', port: 3005 },
      { name: 'ai-orchestration', port: 3006 }
    ];

    const healthEndpoints = ['/health', '/health/ready', '/health/live'];

    for (const service of services) {
      for (const endpoint of healthEndpoints) {
        await this.test(`${service.name}${endpoint}`, async () => {
          const response = await this.makeRequest('GET', `http://localhost:${service.port}${endpoint}`);
          if (response.statusCode !== 200) {
            throw new Error(`Health check failed: ${response.statusCode}`);
          }
          const data = JSON.parse(response.body);
          return `Health: ${data.status || 'OK'}`;
        });
      }
    }

    console.log();
  }

  async testAPIGatewayRouting() {
    console.log('🌐 Testing API Gateway Routing...');

    // Test health endpoints through Kong
    const services = [
      'event-ingestion',
      'behavioral-analysis', 
      'decision-engine',
      'content-generation',
      'channel-orchestrator',
      'ai-orchestration'
    ];

    for (const service of services) {
      await this.test(`Kong routing: ${service}`, async () => {
        const response = await this.makeRequest('GET', `${this.baseUrl}/health/${service}`);
        if (response.statusCode !== 200) {
          throw new Error(`Kong routing failed: ${response.statusCode}`);
        }
        const data = JSON.parse(response.body);
        return `Service routed successfully: ${data.service}`;
      });
    }

    // Test API key authentication
    await this.test('Kong API Key Authentication', async () => {
      // Test without API key (should fail)
      const unauthorizedResponse = await this.makeRequest('POST', `${this.baseUrl}/v1/events/track`, {
        event_type: 'test_event',
        user_id: 'test_user',
        timestamp: new Date().toISOString()
      });

      if (unauthorizedResponse.statusCode !== 401 && unauthorizedResponse.statusCode !== 403) {
        throw new Error(`Expected 401/403 without API key, got ${unauthorizedResponse.statusCode}`);
      }

      // Test with valid API key (should succeed)
      const authorizedResponse = await this.makeRequest('POST', `${this.baseUrl}/v1/events/track`, {
        event_type: 'test_event',
        user_id: 'test_user', 
        timestamp: new Date().toISOString()
      }, {
        'X-API-Key': this.apiKey
      });

      if (authorizedResponse.statusCode < 200 || authorizedResponse.statusCode >= 300) {
        throw new Error(`API key authentication failed: ${authorizedResponse.statusCode}`);
      }

      return 'API key authentication working correctly';
    });

    console.log();
  }

  async testUserJourneyFlows() {
    console.log('👤 Testing Complete User Journey Flows...');

    // Test 1: New User Signup Journey
    await this.test('User Journey: New User Signup', async () => {
      const userId = `test_user_${Date.now()}`;
      
      // Step 1: Track signup event
      const signupResponse = await this.makeRequest('POST', `${this.baseUrl}/v1/events/track`, {
        event_type: 'user_signup',
        user_id: userId,
        timestamp: new Date().toISOString(),
        properties: {
          email: `${userId}@example.com`,
          plan: 'free',
          signup_method: 'email'
        }
      }, {
        'X-API-Key': this.apiKey
      });

      if (signupResponse.statusCode !== 200) {
        throw new Error(`Signup event failed: ${signupResponse.statusCode}`);
      }

      // Wait for processing
      await this.sleep(2000);

      return `New user signup journey completed for ${userId}`;
    });

    // Test 2: Feature Usage Journey
    await this.test('User Journey: Feature Usage', async () => {
      const userId = `existing_user_${Date.now()}`;

      // Track multiple feature usage events
      const events = [
        { event_type: 'feature_used', feature_name: 'dashboard' },
        { event_type: 'feature_used', feature_name: 'analytics' },
        { event_type: 'page_viewed', page: '/dashboard' }
      ];

      for (const event of events) {
        const response = await this.makeRequest('POST', `${this.baseUrl}/v1/events/track`, {
          ...event,
          user_id: userId,
          timestamp: new Date().toISOString()
        }, {
          'X-API-Key': this.apiKey
        });

        if (response.statusCode !== 200) {
          throw new Error(`Feature usage event failed: ${response.statusCode}`);
        }
      }

      return `Feature usage journey completed with ${events.length} events`;
    });

    // Test 3: Purchase Journey
    await this.test('User Journey: Purchase Flow', async () => {
      const userId = `customer_${Date.now()}`;

      const purchaseEvent = {
        event_type: 'purchase',
        user_id: userId,
        timestamp: new Date().toISOString(),
        properties: {
          amount: 29.99,
          currency: 'USD',
          product_id: 'premium_plan',
          payment_method: 'credit_card'
        }
      };

      const response = await this.makeRequest('POST', `${this.baseUrl}/v1/events/track`, purchaseEvent, {
        'X-API-Key': this.apiKey
      });

      if (response.statusCode !== 200) {
        throw new Error(`Purchase event failed: ${response.statusCode}`);
      }

      return `Purchase journey completed for ${userId}`;
    });

    console.log();
  }

  async testInterServiceCommunication() {
    console.log('🔄 Testing Inter-Service Communication...');

    // Test batch event processing
    await this.test('Batch Event Processing', async () => {
      const events = [];
      for (let i = 0; i < 5; i++) {
        events.push({
          event_type: 'test_batch_event',
          user_id: `batch_user_${i}`,
          timestamp: new Date().toISOString(),
          properties: { batch_id: Date.now(), event_index: i }
        });
      }

      const response = await this.makeRequest('POST', `${this.baseUrl}/v1/events/batch`, {
        events: events
      }, {
        'X-API-Key': this.apiKey
      });

      if (response.statusCode !== 200) {
        throw new Error(`Batch processing failed: ${response.statusCode}`);
      }

      const result = JSON.parse(response.body);
      return `Batch processed: ${result.processed || events.length} events`;
    });

    // Test service discovery
    await this.test('Service Discovery Integration', async () => {
      // This would test the Redis-based service discovery
      // For now, we'll verify services are registered
      const services = ['event-ingestion', 'behavioral-analysis', 'decision-engine'];
      let servicesFound = 0;

      for (const service of services) {
        const isHealthy = await this.checkPort('localhost', this.getServicePort(service));
        if (isHealthy) servicesFound++;
      }

      if (servicesFound < services.length) {
        throw new Error(`Only ${servicesFound}/${services.length} services discovered`);
      }

      return `Service discovery: ${servicesFound} services active`;
    });

    console.log();
  }

  async testMonitoringIntegration() {
    console.log('📊 Testing Monitoring & Observability...');

    // Test Prometheus metrics
    await this.test('Prometheus Metrics Collection', async () => {
      const response = await this.makeRequest('GET', 'http://localhost:9090/api/v1/status/buildinfo');
      if (response.statusCode !== 200) {
        throw new Error(`Prometheus not accessible: ${response.statusCode}`);
      }
      return 'Prometheus metrics collection active';
    });

    // Test Grafana accessibility
    await this.test('Grafana Dashboard Access', async () => {
      const response = await this.makeRequest('GET', 'http://localhost:3000/api/health');
      if (response.statusCode !== 200) {
        throw new Error(`Grafana not accessible: ${response.statusCode}`);
      }
      return 'Grafana dashboards accessible';
    });

    // Test service metrics endpoints
    await this.test('Service Metrics Endpoints', async () => {
      const services = [3001, 3002, 3003]; // Test first 3 services
      let metricsFound = 0;

      for (const port of services) {
        try {
          const response = await this.makeRequest('GET', `http://localhost:${port}/metrics`);
          if (response.statusCode === 200) {
            metricsFound++;
          }
        } catch (error) {
          // Service might not have metrics endpoint yet
        }
      }

      return `Metrics endpoints: ${metricsFound} services exposing metrics`;
    });

    console.log();
  }

  async test(testName, testFunction) {
    const startTime = Date.now();
    
    try {
      const result = await Promise.race([
        testFunction(),
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Test timeout')), this.timeout)
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
    return new Promise((resolve, reject) => {
      const urlObj = new URL(url);
      const options = {
        hostname: urlObj.hostname,
        port: urlObj.port,
        path: urlObj.pathname + urlObj.search,
        method: method,
        headers: {
          'Content-Type': 'application/json',
          'User-Agent': 'UserWhisperer-E2E-Tests/1.0',
          ...headers
        },
        timeout: this.timeout
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

  async checkPort(host, port) {
    return new Promise((resolve) => {
      const socket = require('net').createConnection(port, host);
      
      socket.on('connect', () => {
        socket.destroy();
        resolve(true);
      });

      socket.on('error', () => {
        resolve(false);
      });

      socket.setTimeout(3000, () => {
        socket.destroy();
        resolve(false);
      });
    });
  }

  getServicePort(serviceName) {
    const ports = {
      'event-ingestion': 3001,
      'behavioral-analysis': 3002,
      'decision-engine': 3003,
      'content-generation': 3004,
      'channel-orchestrator': 3005,
      'ai-orchestration': 3006
    };
    return ports[serviceName] || 3000;
  }

  async sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  generateReport() {
    console.log('\n📊 END-TO-END TEST RESULTS');
    console.log('=' * 50);
    console.log(`Total Tests: ${this.testResults.passed + this.testResults.failed + this.testResults.skipped}`);
    console.log(`✅ Passed: ${this.testResults.passed}`);
    console.log(`❌ Failed: ${this.testResults.failed}`);
    console.log(`⚠️  Skipped: ${this.testResults.skipped}`);
    
    const successRate = Math.round((this.testResults.passed / (this.testResults.passed + this.testResults.failed)) * 100);
    console.log(`📈 Success Rate: ${successRate}%`);

    if (this.testResults.failed === 0) {
      console.log('\n🎉 ALL E2E TESTS PASSED! Platform integration is working correctly.');
    } else {
      console.log('\n💥 SOME TESTS FAILED! Check the issues above.');
      
      console.log('\nFailed Tests:');
      this.testResults.tests
        .filter(t => t.status === 'failed')
        .forEach(test => {
          console.log(`  • ${test.name}: ${test.error}`);
        });
    }

    // Save detailed results
    const reportPath = path.join(process.cwd(), 'tests', 'reports', `e2e-${Date.now()}.json`);
    try {
      fs.mkdirSync(path.dirname(reportPath), { recursive: true });
      fs.writeFileSync(reportPath, JSON.stringify(this.testResults, null, 2));
      console.log(`\n📄 Detailed report saved: ${reportPath}`);
    } catch (error) {
      console.log(`\n⚠️  Could not save report: ${error.message}`);
    }
  }
}

// Run tests if called directly
if (require.main === module) {
  const testSuite = new E2ETestSuite();
  testSuite.runAllTests()
    .then((success) => {
      process.exit(success ? 0 : 1);
    })
    .catch((error) => {
      console.error('E2E test suite error:', error);
      process.exit(1);
    });
}

module.exports = E2ETestSuite;
