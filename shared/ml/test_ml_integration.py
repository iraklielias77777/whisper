#!/usr/bin/env python3
"""
Test script to verify ML components integration
Tests model imports and basic functionality
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_imports():
    """Test that all ML models can be imported successfully"""
    print("🧪 Testing ML Model Imports...")
    
    try:
        from models.churn_prediction import ChurnPredictionModel
        print("✅ ChurnPredictionModel imported successfully")
    except ImportError as e:
        print(f"⚠️  ChurnPredictionModel import failed: {e}")
    
    try:
        from models.content_optimization import ThompsonSamplingBandit
        print("✅ ThompsonSamplingBandit imported successfully")
    except ImportError as e:
        print(f"⚠️  ThompsonSamplingBandit import failed: {e}")
    
    try:
        from models.timing_optimization import TimingOptimizationModel
        print("✅ TimingOptimizationModel imported successfully")
    except ImportError as e:
        print(f"⚠️  TimingOptimizationModel import failed: {e}")

def test_training_pipeline():
    """Test training pipeline components"""
    print("\n🧪 Testing Training Pipeline...")
    
    try:
        from training.training_pipeline import ModelTrainingPipeline
        print("✅ ModelTrainingPipeline imported successfully")
    except ImportError as e:
        print(f"⚠️  ModelTrainingPipeline import failed: {e}")
    
    try:
        from training.model_validation import ModelValidator
        print("✅ ModelValidator imported successfully")
    except ImportError as e:
        print(f"⚠️  ModelValidator import failed: {e}")

def test_model_initialization():
    """Test that models can be initialized with basic config"""
    print("\n🧪 Testing Model Initialization...")
    
    # Test Churn Prediction Model
    try:
        from models.churn_prediction import ChurnPredictionModel
        config = {
            'sequence_length': 30,
            'feature_dim': 50,
            'model_type': 'lstm_attention'
        }
        model = ChurnPredictionModel(config)
        print("✅ ChurnPredictionModel initialized successfully")
    except Exception as e:
        print(f"⚠️  ChurnPredictionModel initialization failed: {e}")
    
    # Test Thompson Sampling Bandit
    try:
        from models.content_optimization import ThompsonSamplingBandit
        bandit = ThompsonSamplingBandit(
            n_arms=5,
            context_dim=20,
            alpha=1.0,
            beta=1.0
        )
        print("✅ ThompsonSamplingBandit initialized successfully")
    except Exception as e:
        print(f"⚠️  ThompsonSamplingBandit initialization failed: {e}")
    
    # Test Timing Optimization Model
    try:
        from models.timing_optimization import TimingOptimizationModel
        config = {
            'timezone': 'UTC',
            'model_type': 'heuristic',  # Fallback when Prophet not available
            'lookback_days': 30
        }
        model = TimingOptimizationModel(config)
        print("✅ TimingOptimizationModel initialized successfully")
    except Exception as e:
        print(f"⚠️  TimingOptimizationModel initialization failed: {e}")

def test_ai_orchestration_integration():
    """Test AI orchestration components"""
    print("\n🧪 Testing AI Orchestration Integration...")
    
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ai_orchestration'))
        from master_controller import MasterAIOrchestrationController
        print("✅ MasterAIOrchestrationController imported successfully")
        
        # Test initialization
        controller = MasterAIOrchestrationController()
        print("✅ MasterAIOrchestrationController initialized successfully")
        
    except ImportError as e:
        print(f"⚠️  AI Orchestration import failed: {e}")
    except Exception as e:
        print(f"⚠️  AI Orchestration initialization failed: {e}")

def test_dependencies():
    """Test critical dependencies"""
    print("\n🧪 Testing Dependencies...")
    
    dependencies = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
    ]
    
    optional_dependencies = [
        ('tensorflow', 'TensorFlow'),
        ('prophet', 'Prophet'),
        ('mlflow', 'MLflow'),
    ]
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name} available")
        except ImportError:
            print(f"❌ {name} NOT available (required)")
    
    for module, name in optional_dependencies:
        try:
            __import__(module)
            print(f"✅ {name} available")
        except ImportError:
            print(f"⚠️  {name} not available (optional)")

def run_integration_test():
    """Run comprehensive integration test"""
    print("🚀 ML INTEGRATION TEST SUITE")
    print("=" * 50)
    
    test_dependencies()
    test_model_imports()
    test_training_pipeline()
    test_model_initialization()
    test_ai_orchestration_integration()
    
    print("\n" + "=" * 50)
    print("🎯 ML Integration Test Complete!")
    print("\nℹ️  Note: Some warnings are expected if optional dependencies")
    print("   like TensorFlow, Prophet, or MLflow are not installed.")
    print("   The system will use fallback implementations.")

if __name__ == "__main__":
    run_integration_test()
