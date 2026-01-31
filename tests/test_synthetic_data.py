"""
Test Suite for Synthetic Data Pipelines
=======================================

Author: Bhavin Kotak
Work Email: bhavin.kotak@ibm.com
Personal Email: kotak.bhavin@gmail.com

Comprehensive tests for the synthetic data generation pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from synthetic_data_generator import (
    SyntheticDataConfig,
    TransactionGenerator,
    SyntheticDataQualityValidator
)


class TestSyntheticDataConfig:
    """Tests for configuration class"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = SyntheticDataConfig()
        assert config.n_samples == 10000
        assert config.fraud_rate == 0.01
        assert config.random_seed == 42
        
    def test_custom_config(self):
        """Test custom configuration"""
        config = SyntheticDataConfig(
            n_samples=5000,
            fraud_rate=0.05,
            random_seed=123
        )
        assert config.n_samples == 5000
        assert config.fraud_rate == 0.05
        assert config.random_seed == 123
        
    def test_merchant_categories(self):
        """Test merchant categories sum to 1.0"""
        config = SyntheticDataConfig()
        total_prob = sum(config.merchant_categories.values())
        assert abs(total_prob - 1.0) < 0.001


class TestTransactionGenerator:
    """Tests for transaction generation"""
    
    @pytest.fixture
    def generator(self):
        """Create generator fixture"""
        config = SyntheticDataConfig(
            n_samples=1000,
            fraud_rate=0.02,
            random_seed=42
        )
        return TransactionGenerator(config)
    
    def test_generate_single_transaction(self, generator):
        """Test single transaction generation"""
        transaction = generator.generate_transaction(is_fraud=False)
        
        assert 'transaction_id' in transaction
        assert 'timestamp' in transaction
        assert 'merchant_category' in transaction
        assert 'amount' in transaction
        assert 'customer_age' in transaction
        assert 'is_fraud' in transaction
        
    def test_transaction_id_uniqueness(self, generator):
        """Test that transaction IDs are unique"""
        transactions = [generator.generate_transaction() for _ in range(100)]
        ids = [t['transaction_id'] for t in transactions]
        assert len(ids) == len(set(ids))
        
    def test_customer_age_range(self, generator):
        """Test customer age is within valid range"""
        for _ in range(100):
            transaction = generator.generate_transaction()
            assert 18 <= transaction['customer_age'] <= 85
            
    def test_amount_positive(self, generator):
        """Test transaction amounts are positive"""
        for _ in range(100):
            transaction = generator.generate_transaction()
            assert transaction['amount'] > 0
            
    def test_fraud_has_reason(self, generator):
        """Test fraudulent transactions have reasons"""
        transaction = generator.generate_transaction(is_fraud=True)
        assert transaction['is_fraud'] is True
        assert transaction['fraud_reason'] is not None
        
    def test_normal_no_fraud_reason(self, generator):
        """Test normal transactions have no fraud reason"""
        transaction = generator.generate_transaction(is_fraud=False)
        assert transaction['is_fraud'] is False
        assert transaction['fraud_reason'] is None
        
    def test_dataset_generation(self, generator):
        """Test full dataset generation"""
        df = generator.generate_dataset(n_samples=500)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 500
        assert 'is_fraud' in df.columns
        
    def test_fraud_rate_approximately_correct(self, generator):
        """Test fraud rate is approximately as configured"""
        df = generator.generate_dataset(n_samples=1000)
        actual_fraud_rate = df['is_fraud'].mean()
        
        # Allow 50% tolerance due to small sample
        assert 0.01 <= actual_fraud_rate <= 0.04
        
    def test_merchant_category_valid(self, generator):
        """Test merchant categories are from configured list"""
        config = generator.config
        valid_categories = set(config.merchant_categories.keys())
        
        df = generator.generate_dataset(n_samples=100)
        generated_categories = set(df['merchant_category'].unique())
        
        assert generated_categories.issubset(valid_categories)
        
    def test_reproducibility(self):
        """Test same seed produces same statistical results"""
        config1 = SyntheticDataConfig(n_samples=100, random_seed=42)
        config2 = SyntheticDataConfig(n_samples=100, random_seed=42)
        
        gen1 = TransactionGenerator(config1)
        gen2 = TransactionGenerator(config2)
        
        df1 = gen1.generate_dataset()
        df2 = gen2.generate_dataset()
        
        # Check same fraud distribution
        assert df1['is_fraud'].sum() == df2['is_fraud'].sum()
        # Check same age distribution (deterministic with seed)
        assert df1['customer_age'].sum() == df2['customer_age'].sum()


class TestSyntheticDataQualityValidator:
    """Tests for quality validation"""
    
    @pytest.fixture
    def datasets(self):
        """Create real and synthetic datasets"""
        config_real = SyntheticDataConfig(n_samples=500, random_seed=42)
        config_synth = SyntheticDataConfig(n_samples=500, random_seed=123)
        
        gen_real = TransactionGenerator(config_real)
        gen_synth = TransactionGenerator(config_synth)
        
        real_data = gen_real.generate_dataset()
        synthetic_data = gen_synth.generate_dataset()
        
        return real_data, synthetic_data
    
    @pytest.fixture
    def validator(self, datasets):
        """Create validator fixture"""
        real_data, synthetic_data = datasets
        return SyntheticDataQualityValidator(real_data, synthetic_data)
    
    def test_statistical_fidelity(self, validator):
        """Test statistical fidelity evaluation"""
        results = validator.evaluate_statistical_fidelity()
        
        assert 'numeric_metrics' in results
        assert 'categorical_metrics' in results
        assert 'overall_score' in results
        assert 0 <= results['overall_score'] <= 1
        
    def test_wasserstein_distance_computed(self, validator):
        """Test Wasserstein distance is computed for numeric columns"""
        results = validator.evaluate_statistical_fidelity()
        
        for col in ['amount', 'customer_age', 'account_tenure_years']:
            assert col in results['numeric_metrics']
            assert 'wasserstein_distance' in results['numeric_metrics'][col]
            assert results['numeric_metrics'][col]['wasserstein_distance'] >= 0
            
    def test_diversity_evaluation(self, validator):
        """Test diversity evaluation"""
        results = validator.evaluate_diversity()
        
        assert 'range_coverage' in results
        assert 'unique_counts' in results
        assert 'edge_case_coverage' in results
        
    def test_utility_evaluation(self, validator):
        """Test utility evaluation"""
        results = validator.evaluate_utility(target_col='is_fraud')
        
        assert 'real_only' in results
        assert 'synthetic_only' in results
        assert 'combined' in results
        assert 'utility_score' in results
        
        # Check metrics exist
        for key in ['real_only', 'synthetic_only', 'combined']:
            assert 'accuracy' in results[key]
            assert 'auc_roc' in results[key]
            
    def test_privacy_evaluation(self, validator):
        """Test privacy evaluation"""
        results = validator.evaluate_privacy(n_attacks=50)
        
        assert 'membership_inference' in results
        assert 'privacy_score' in results
        assert 0 <= results['privacy_score'] <= 1
        
    def test_full_report(self, validator):
        """Test full report generation"""
        report = validator.generate_full_report()
        
        assert 'statistical_fidelity' in report
        assert 'diversity' in report
        assert 'utility' in report
        assert 'privacy' in report
        assert 'timestamp' in report
        

class TestEdgeCases:
    """Tests for edge cases and error handling"""
    
    def test_very_small_dataset(self):
        """Test with very small dataset"""
        config = SyntheticDataConfig(n_samples=10, fraud_rate=0.1, random_seed=42)
        generator = TransactionGenerator(config)
        df = generator.generate_dataset()
        
        assert len(df) == 10
        
    def test_zero_fraud_rate(self):
        """Test with zero fraud rate"""
        config = SyntheticDataConfig(n_samples=100, fraud_rate=0.0, random_seed=42)
        generator = TransactionGenerator(config)
        df = generator.generate_dataset()
        
        assert df['is_fraud'].sum() == 0
        
    def test_high_fraud_rate(self):
        """Test with high fraud rate"""
        config = SyntheticDataConfig(n_samples=100, fraud_rate=0.5, random_seed=42)
        generator = TransactionGenerator(config)
        df = generator.generate_dataset()
        
        actual_rate = df['is_fraud'].mean()
        assert 0.4 <= actual_rate <= 0.6
        
    def test_generation_stats_updated(self):
        """Test that generation statistics are tracked"""
        config = SyntheticDataConfig(n_samples=100, fraud_rate=0.1, random_seed=42)
        generator = TransactionGenerator(config)
        
        initial_count = generator.generation_stats['total_generated']
        generator.generate_dataset()
        
        assert generator.generation_stats['total_generated'] > initial_count


class TestDataIntegrity:
    """Tests for data integrity"""
    
    def test_no_null_required_fields(self):
        """Test no null values in required fields"""
        config = SyntheticDataConfig(n_samples=500, random_seed=42)
        generator = TransactionGenerator(config)
        df = generator.generate_dataset()
        
        required_fields = [
            'transaction_id', 'timestamp', 'merchant_category',
            'amount', 'customer_age', 'is_fraud'
        ]
        
        for field in required_fields:
            assert df[field].isnull().sum() == 0
            
    def test_timestamp_format(self):
        """Test timestamp is valid ISO format"""
        config = SyntheticDataConfig(n_samples=100, random_seed=42)
        generator = TransactionGenerator(config)
        df = generator.generate_dataset()
        
        for ts in df['timestamp'].head(10):
            # Should not raise exception
            datetime.fromisoformat(ts)
            
    def test_amount_precision(self):
        """Test amount has at most 2 decimal places"""
        config = SyntheticDataConfig(n_samples=100, random_seed=42)
        generator = TransactionGenerator(config)
        df = generator.generate_dataset()
        
        for amount in df['amount']:
            # Check 2 decimal precision
            assert round(amount, 2) == amount


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
