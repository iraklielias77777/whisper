#!/usr/bin/env node

/**
 * Service Validation Script
 * Validates all microservices are properly implemented and connected
 */

const http = require('http');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const SERVICES = {
  'event-ingestion': { port: 3001, endpoints: ['/health', '/health/ready', '/health/live'] },
  'behavioral-analysis': { port: 3002, endpoints: ['/health', '/health/ready', '/health/live'] },
  'decision-engine': { port: 3003, endpoints: ['/health', '/health/ready', '/health/live'] },
  'content-generation': { port: 3004, endpoints: ['/health', '/health/ready', '/health/live'] },
  'channel-orchestrator': { port: 3005, endpoints: ['/health', '/health/ready', '/health/live'] },
  'ai-orchestration': { port: 3006, endpoints: ['/health', '/health/ready', '/health/live'] }
};

const INFRASTRUCTURE = {
  'postgres': { port: 5432, type: 'tcp' },
  'redis': { port: 6379, type: 'tcp' },
  'kong': { port: 8000, endpoints: ['/health/event-ingestion'] },
  'prometheus': { port: 9090, endpoints: ['/api/v1/status/buildinfo'] },
  'grafana': { port: 3000, endpoints: ['/api/health'] }
};

class ServiceValidator {
  constructor() {
    this.results = {
      services: {},
      infrastructure: {},
      overall: {
        passed: 0,
        failed: 0,
        warnings: 0
      }
    };
  }

  async validateAll() {
    console.log('🔍 Starting User Whisperer Platform Validation...\n');

    // Check file structure
    await this.validateFileStructure();

    // Check package dependencies
    await this.validateDependencies();

    // Check if services are running
    await this.validateServicesHealth();

    // Check infrastructure
    await this.validateInfrastructure();

    // Generate report
    this.generateReport();

    return this.results.overall.failed === 0;
  }

  async validateFileStructure() {
    console.log('📁 Validating file structure...');

    const requiredFiles = [
      'docker-compose.yml',
      'infrastructure/kong/kong.yml',
      'infrastructure/sql/schema.sql',
      'shared/package.json',
      'services/event-ingestion/Dockerfile',
      'services/behavioral-analysis/Dockerfile',
      'services/decision-engine/Dockerfile',
      'services/content-generation/Dockerfile',
      'services/channel-orchestrator/Dockerfile',
      'services/ai-orchestration/Dockerfile'
    ];

    for (const file of requiredFiles) {
      const exists = fs.existsSync(path.join(process.cwd(), file));
      if (exists) {
        console.log(`  ✅ ${file}`);
        this.results.overall.passed++;
      } else {
        console.log(`  ❌ ${file} - MISSING`);
        this.results.overall.failed++;
      }
    }

    console.log();
  }

  async validateDependencies() {
    console.log('📦 Validating package dependencies...');

    const services = Object.keys(SERVICES);
    for (const service of services) {
      const packagePath = path.join(process.cwd(), 'services', service, 'package.json');
      
      if (fs.existsSync(packagePath)) {
        try {
          const packageJson = JSON.parse(fs.readFileSync(packagePath, 'utf8'));
          
          // Check if shared dependency is properly configured
          const sharedDep = packageJson.dependencies['@userwhisperer/shared'];
          if (sharedDep === 'file:../../shared') {
            console.log(`  ✅ ${service} - Shared dependency correct`);
            this.results.overall.passed++;
          } else {
            console.log(`  ⚠️  ${service} - Shared dependency issue: ${sharedDep}`);
            this.results.overall.warnings++;
          }

          // Check Node.js version consistency
          const nodeVersion = packageJson.engines?.node;
          if (nodeVersion === '>=20.0.0') {
            console.log(`  ✅ ${service} - Node.js version correct`);
            this.results.overall.passed++;
          } else {
            console.log(`  ⚠️  ${service} - Node.js version: ${nodeVersion}`);
            this.results.overall.warnings++;
          }

        } catch (error) {
          console.log(`  ❌ ${service} - Package.json parse error: ${error.message}`);
          this.results.overall.failed++;
        }
      } else {
        console.log(`  ❌ ${service} - Package.json missing`);
        this.results.overall.failed++;
      }
    }

    console.log();
  }

  async validateServicesHealth() {
    console.log('🏥 Validating service health endpoints...');

    for (const [serviceName, config] of Object.entries(SERVICES)) {
      console.log(`  Testing ${serviceName}...`);
      this.results.services[serviceName] = {
        running: false,
        endpoints: {},
        overall: 'failed'
      };

      // Check if service is running
      const isRunning = await this.checkPort('localhost', config.port);
      this.results.services[serviceName].running = isRunning;

      if (!isRunning) {
        console.log(`    ❌ Service not running on port ${config.port}`);
        this.results.overall.failed++;
        continue;
      }

      console.log(`    ✅ Service running on port ${config.port}`);
      this.results.overall.passed++;

      // Test health endpoints
      let allEndpointsHealthy = true;
      for (const endpoint of config.endpoints) {
        const isHealthy = await this.testHttpEndpoint('localhost', config.port, endpoint);
        this.results.services[serviceName].endpoints[endpoint] = isHealthy;

        if (isHealthy) {
          console.log(`    ✅ ${endpoint} - OK`);
          this.results.overall.passed++;
        } else {
          console.log(`    ❌ ${endpoint} - FAILED`);
          this.results.overall.failed++;
          allEndpointsHealthy = false;
        }
      }

      this.results.services[serviceName].overall = allEndpointsHealthy ? 'passed' : 'failed';
    }

    console.log();
  }

  async validateInfrastructure() {
    console.log('🏗️  Validating infrastructure components...');

    for (const [component, config] of Object.entries(INFRASTRUCTURE)) {
      console.log(`  Testing ${component}...`);
      this.results.infrastructure[component] = {
        running: false,
        endpoints: {},
        overall: 'failed'
      };

      if (config.type === 'tcp') {
        const isRunning = await this.checkPort('localhost', config.port);
        this.results.infrastructure[component].running = isRunning;

        if (isRunning) {
          console.log(`    ✅ ${component} running on port ${config.port}`);
          this.results.overall.passed++;
          this.results.infrastructure[component].overall = 'passed';
        } else {
          console.log(`    ❌ ${component} not running on port ${config.port}`);
          this.results.overall.failed++;
        }
      } else {
        // HTTP endpoints
        const isRunning = await this.checkPort('localhost', config.port);
        this.results.infrastructure[component].running = isRunning;

        if (!isRunning) {
          console.log(`    ❌ ${component} not running on port ${config.port}`);
          this.results.overall.failed++;
          continue;
        }

        console.log(`    ✅ ${component} running on port ${config.port}`);
        this.results.overall.passed++;

        // Test endpoints
        let allEndpointsHealthy = true;
        for (const endpoint of config.endpoints || []) {
          const isHealthy = await this.testHttpEndpoint('localhost', config.port, endpoint);
          this.results.infrastructure[component].endpoints[endpoint] = isHealthy;

          if (isHealthy) {
            console.log(`    ✅ ${endpoint} - OK`);
            this.results.overall.passed++;
          } else {
            console.log(`    ❌ ${endpoint} - FAILED`);
            this.results.overall.failed++;
            allEndpointsHealthy = false;
          }
        }

        this.results.infrastructure[component].overall = allEndpointsHealthy ? 'passed' : 'failed';
      }
    }

    console.log();
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

  async testHttpEndpoint(host, port, path) {
    return new Promise((resolve) => {
      const options = {
        hostname: host,
        port: port,
        path: path,
        method: 'GET',
        timeout: 5000
      };

      const req = http.request(options, (res) => {
        resolve(res.statusCode >= 200 && res.statusCode < 400);
      });

      req.on('error', () => {
        resolve(false);
      });

      req.on('timeout', () => {
        req.destroy();
        resolve(false);
      });

      req.end();
    });
  }

  generateReport() {
    console.log('📊 VALIDATION REPORT');
    console.log('='.repeat(50));
    console.log(`Total Tests: ${this.results.overall.passed + this.results.overall.failed + this.results.overall.warnings}`);
    console.log(`✅ Passed: ${this.results.overall.passed}`);
    console.log(`❌ Failed: ${this.results.overall.failed}`);
    console.log(`⚠️  Warnings: ${this.results.overall.warnings}`);
    console.log();

    if (this.results.overall.failed === 0) {
      console.log('🎉 ALL VALIDATIONS PASSED! The User Whisperer Platform is ready.');
    } else {
      console.log('💥 VALIDATION FAILED! Please fix the issues above.');
      console.log();
      console.log('Quick fixes:');
      console.log('- Ensure all services are running: docker-compose up -d');
      console.log('- Check service logs: docker-compose logs [service-name]');
      console.log('- Verify Docker containers: docker-compose ps');
    }

    console.log();
    console.log('Service URLs:');
    console.log('- Kong API Gateway: http://localhost:8000');
    console.log('- Kong Admin: http://localhost:8001');
    console.log('- Prometheus: http://localhost:9090');
    console.log('- Grafana: http://localhost:3000 (admin/admin)');
    console.log('- Event Ingestion: http://localhost:3001');
    console.log();
  }
}

// Run validation if called directly
if (require.main === module) {
  const validator = new ServiceValidator();
  validator.validateAll()
    .then((success) => {
      process.exit(success ? 0 : 1);
    })
    .catch((error) => {
      console.error('Validation error:', error);
      process.exit(1);
    });
}

module.exports = ServiceValidator;
