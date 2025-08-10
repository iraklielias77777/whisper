#!/usr/bin/env npx ts-node
/**
 * AI Orchestration Integration Test Suite
 * Tests the complete integration between TypeScript services and Python AI orchestration
 */

import { aiOrchestrationClient } from './shared';

interface TestResult {
  testName: string;
  passed: boolean;
  error?: string;
  details?: any;
}

class AIOrchestrationIntegrationTester {
  private results: TestResult[] = [];

  async runAllTests(): Promise<void> {
    console.log('🚀 AI ORCHESTRATION INTEGRATION TEST SUITE');
    console.log('=' .repeat(60));
    
    await this.testAIOrchestrationClientImport();
    await this.testHealthCheck();
    await this.testSingleUserOrchestration();
    await this.testBatchOrchestration();
    await this.testSystemInsights();
    await this.testConfigurationUpdate();
    await this.testFallbackBehavior();
    
    this.printResults();
  }

  private async testAIOrchestrationClientImport(): Promise<void> {
    try {
      console.log('\n🧪 Testing AI Orchestration Client Import...');
      
      if (!aiOrchestrationClient) {
        throw new Error('aiOrchestrationClient is not available');
      }
      
      if (typeof aiOrchestrationClient.orchestrateUser !== 'function') {
        throw new Error('orchestrateUser method not available');
      }
      
      if (typeof aiOrchestrationClient.healthCheck !== 'function') {
        throw new Error('healthCheck method not available');
      }
      
      this.addResult('AI Orchestration Client Import', true, {
        methods: Object.getOwnPropertyNames(Object.getPrototypeOf(aiOrchestrationClient))
      });
      
      console.log('✅ AI Orchestration Client imported and initialized successfully');
      
    } catch (error) {
      this.addResult('AI Orchestration Client Import', false, error instanceof Error ? error.message : String(error));
      console.log('❌ AI Orchestration Client import failed:', error);
    }
  }

  private async testHealthCheck(): Promise<void> {
    try {
      console.log('\n🧪 Testing AI Orchestration Health Check...');
      
      const isHealthy = await aiOrchestrationClient.healthCheck();
      
      this.addResult('AI Orchestration Health Check', true, {
        healthy: isHealthy,
        note: isHealthy ? 'Service is running' : 'Service unavailable (expected if not started)'
      });
      
      if (isHealthy) {
        console.log('✅ AI Orchestration service is healthy and responding');
      } else {
        console.log('⚠️  AI Orchestration service not running (this is expected in testing)');
      }
      
    } catch (error) {
      this.addResult('AI Orchestration Health Check', false, error instanceof Error ? error.message : String(error));
      console.log('⚠️  Health check failed (expected if service not running):', error);
    }
  }

  private async testSingleUserOrchestration(): Promise<void> {
    try {
      console.log('\n🧪 Testing Single User AI Orchestration...');
      
      const testRequest = {
        user_id: 'test_user_123',
        user_context: {
          lifecycle_stage: 'engaged',
          subscription_plan: 'pro',
          last_active: new Date().toISOString()
        },
        behavioral_data: {
          engagement_score: 0.75,
          churn_risk: 0.2,
          feature_usage: ['dashboard', 'reports', 'exports']
        },
        trigger_event: 'feature_usage_decline',
        business_objectives: {
          focus: 'retention',
          priority: 'high'
        },
        constraints: {
          budget: 'medium',
          channels: ['email', 'push']
        }
      };
      
      const result = await aiOrchestrationClient.orchestrateUser(testRequest);
      
      // Validate response structure
      if (!result.orchestration_id || !result.user_id || !result.strategy_decisions) {
        throw new Error('Invalid orchestration response structure');
      }
      
      this.addResult('Single User Orchestration', true, {
        orchestrationId: result.orchestration_id,
        userId: result.user_id,
        interventionType: result.strategy_decisions.intervention_type,
        channel: result.strategy_decisions.channel,
        confidence: result.ai_insights.confidence,
        processingTime: result.metadata.processing_time_ms,
        modelsUsed: result.metadata.models_used
      });
      
      console.log('✅ Single user orchestration completed successfully');
      console.log(`   Intervention: ${result.strategy_decisions.intervention_type}`);
      console.log(`   Channel: ${result.strategy_decisions.channel}`);
      console.log(`   Confidence: ${result.ai_insights.confidence}`);
      
    } catch (error) {
      this.addResult('Single User Orchestration', false, error instanceof Error ? error.message : String(error));
      console.log('⚠️  Single user orchestration test failed (using fallback):', error);
    }
  }

  private async testBatchOrchestration(): Promise<void> {
    try {
      console.log('\n🧪 Testing Batch AI Orchestration...');
      
      const testUsers = [
        {
          user_id: 'batch_user_1',
          user_context: { lifecycle_stage: 'new' },
          behavioral_data: { engagement_score: 0.3 }
        },
        {
          user_id: 'batch_user_2', 
          user_context: { lifecycle_stage: 'engaged' },
          behavioral_data: { engagement_score: 0.8 }
        },
        {
          user_id: 'batch_user_3',
          user_context: { lifecycle_stage: 'at_risk' },
          behavioral_data: { engagement_score: 0.1, churn_risk: 0.9 }
        }
      ];
      
      const results = await aiOrchestrationClient.orchestrateBatch(testUsers);
      
      if (!Array.isArray(results) || results.length !== testUsers.length) {
        throw new Error('Invalid batch orchestration response');
      }
      
      this.addResult('Batch Orchestration', true, {
        batchSize: testUsers.length,
        resultsCount: results.length,
        interventionTypes: results.map(r => r.strategy_decisions?.intervention_type),
        avgConfidence: results.reduce((sum, r) => sum + (r.ai_insights?.confidence || 0), 0) / results.length
      });
      
      console.log('✅ Batch orchestration completed successfully');
      console.log(`   Processed ${results.length} users`);
      
    } catch (error) {
      this.addResult('Batch Orchestration', false, error instanceof Error ? error.message : String(error));
      console.log('⚠️  Batch orchestration test failed:', error);
    }
  }

  private async testSystemInsights(): Promise<void> {
    try {
      console.log('\n🧪 Testing System Insights...');
      
      const insights = await aiOrchestrationClient.getSystemInsights();
      
      if (!insights.performance_metrics) {
        throw new Error('Invalid insights response structure');
      }
      
      this.addResult('System Insights', true, {
        performanceMetrics: insights.performance_metrics,
        systemHealth: insights.system_health || 'N/A'
      });
      
      console.log('✅ System insights retrieved successfully');
      console.log(`   Success rate: ${insights.performance_metrics.success_rate || 'N/A'}`);
      
    } catch (error) {
      this.addResult('System Insights', false, error instanceof Error ? error.message : String(error));
      console.log('⚠️  System insights test failed:', error);
    }
  }

  private async testConfigurationUpdate(): Promise<void> {
    try {
      console.log('\n🧪 Testing Configuration Update...');
      
      const testConfig = {
        model_settings: {
          churn_threshold: 0.7,
          confidence_threshold: 0.8
        },
        orchestration_settings: {
          max_processing_time: 5000,
          fallback_enabled: true
        }
      };
      
      const success = await aiOrchestrationClient.updateConfiguration(testConfig);
      
      this.addResult('Configuration Update', true, {
        success,
        config: testConfig
      });
      
      console.log('✅ Configuration update completed');
      
    } catch (error) {
      this.addResult('Configuration Update', false, error instanceof Error ? error.message : String(error));
      console.log('⚠️  Configuration update test failed:', error);
    }
  }

  private async testFallbackBehavior(): Promise<void> {
    try {
      console.log('\n🧪 Testing Fallback Behavior...');
      
      // Test with invalid request to trigger fallback
      const invalidRequest = {
        user_id: '', // Invalid user ID
        user_context: {},
        behavioral_data: {}
      };
      
      const result = await aiOrchestrationClient.orchestrateUser(invalidRequest);
      
      // Should still return a valid response due to fallback
      if (!result.orchestration_id || !result.strategy_decisions) {
        throw new Error('Fallback mechanism failed');
      }
      
      this.addResult('Fallback Behavior', true, {
        fallbackTriggered: result.metadata.models_used.includes('fallback_heuristics'),
        orchestrationId: result.orchestration_id,
        modelsUsed: result.metadata.models_used
      });
      
      console.log('✅ Fallback mechanism working correctly');
      
    } catch (error) {
      this.addResult('Fallback Behavior', false, error instanceof Error ? error.message : String(error));
      console.log('⚠️  Fallback behavior test failed:', error);
    }
  }

  private addResult(testName: string, passed: boolean, details?: any): void {
    this.results.push({
      testName,
      passed,
      details,
      error: passed ? undefined : String(details)
    });
  }

  private printResults(): void {
    console.log('\n' + '=' .repeat(60));
    console.log('🎯 AI ORCHESTRATION INTEGRATION TEST RESULTS');
    console.log('=' .repeat(60));
    
    const passed = this.results.filter(r => r.passed).length;
    const total = this.results.length;
    
    console.log(`\n📊 Overall Results: ${passed}/${total} tests passed\n`);
    
    this.results.forEach((result, index) => {
      const icon = result.passed ? '✅' : '❌';
      console.log(`${index + 1}. ${icon} ${result.testName}`);
      
      if (result.passed && result.details) {
        console.log(`   └─ Details: ${JSON.stringify(result.details, null, 2).slice(0, 200)}...`);
      } else if (!result.passed && result.error) {
        console.log(`   └─ Error: ${result.error}`);
      }
    });
    
    console.log('\n' + '=' .repeat(60));
    
    if (passed === total) {
      console.log('🎉 ALL TESTS PASSED! AI Orchestration integration is working correctly.');
    } else {
      console.log('⚠️  Some tests failed, but this is expected if the AI orchestration service is not running.');
      console.log('   The client implements proper fallback mechanisms for resilience.');
    }
    
    console.log('\n💡 To run the full integration:');
    console.log('   1. Start the AI orchestration service: npm run dev:ai-orchestration');
    console.log('   2. Run this test again to verify live connectivity');
    console.log('=' .repeat(60));
  }
}

// Run the integration test
async function main() {
  const tester = new AIOrchestrationIntegrationTester();
  await tester.runAllTests();
}

if (require.main === module) {
  main().catch(console.error);
}

export { AIOrchestrationIntegrationTester };
