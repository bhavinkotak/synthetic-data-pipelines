"""
Synthetic Data Pipelines - Main Module
======================================

Author: Bhavin Kotak
Work Email: bhavin.kotak@ibm.com
Personal Email: kotak.bhavin@gmail.com

This module implements the synthetic data generation pipelines described in the paper:
"From Data Scarcity to Abundance: Scaling Model Training with LLM-Orchestrated 
Synthetic Data Pipelines"

Features:
- Statistical synthetic data generation
- Quality validation framework
- Privacy preservation metrics
- Downstream utility testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
from scipy import stats
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation"""
    n_samples: int = 10000
    fraud_rate: float = 0.01  # 1% fraud rate
    random_seed: int = 42
    
    # Feature distributions
    age_mean: float = 45.0
    age_std: float = 15.0
    tenure_mean: float = 5.0
    tenure_std: float = 3.0
    amount_mean: float = 150.0
    amount_std: float = 200.0
    
    # Merchant categories with probabilities
    merchant_categories: Dict[str, float] = field(default_factory=lambda: {
        '5411': 0.30,  # Grocery stores
        '5812': 0.25,  # Restaurants
        '5942': 0.10,  # Book stores
        '5311': 0.15,  # Department stores
        '5944': 0.05,  # Jewelry stores
        '5999': 0.15,  # Misc retail
    })


class TransactionGenerator:
    """
    Generate synthetic financial transaction data with realistic patterns.
    
    This generator creates transactions that exhibit:
    - Realistic statistical distributions
    - Correlations between features (age-spending, tenure-fraud risk)
    - Rare fraud events with recognizable patterns
    """
    
    def __init__(self, config: SyntheticDataConfig):
        self.config = config
        self._rng = np.random.RandomState(config.random_seed)
        self.generation_stats = {
            'total_generated': 0,
            'fraud_count': 0,
            'validation_errors': 0
        }
        
    def generate_transaction(self, is_fraud: bool = False) -> Dict[str, Any]:
        """Generate a single transaction record"""
        
        # Base customer profile
        customer_age = int(np.clip(
            self._rng.normal(self.config.age_mean, self.config.age_std),
            18, 85
        ))
        
        account_tenure = int(np.clip(
            self._rng.normal(self.config.tenure_mean, self.config.tenure_std),
            0, min(customer_age - 18, 40)
        ))
        
        # Select merchant category
        categories = list(self.config.merchant_categories.keys())
        probs = list(self.config.merchant_categories.values())
        merchant_category = self._rng.choice(categories, p=probs)
        
        # Generate amount based on category and customer profile
        base_amount = self._generate_amount(merchant_category, customer_age, account_tenure)
        
        # Geographic distance (most within 25 miles)
        if is_fraud:
            geographic_distance = self._rng.exponential(500)  # Fraud: far away
        else:
            geographic_distance = np.abs(self._rng.normal(10, 15))
        
        # Timestamp
        timestamp = datetime.now() - timedelta(
            days=self._rng.randint(0, 365),
            hours=self._rng.randint(0, 24),
            minutes=self._rng.randint(0, 60)
        )
        
        # Fraud-specific modifications
        if is_fraud:
            base_amount *= self._rng.uniform(3, 10)  # Unusually large
            fraud_reason = self._rng.choice([
                'unusual_amount', 'distant_location', 
                'rapid_succession', 'high_risk_merchant'
            ])
        else:
            fraud_reason = None
        
        transaction = {
            'transaction_id': str(uuid.uuid4()),
            'timestamp': timestamp.isoformat(),
            'merchant_category': merchant_category,
            'merchant_name': self._generate_merchant_name(merchant_category),
            'amount': round(base_amount, 2),
            'customer_age': customer_age,
            'account_tenure_years': account_tenure,
            'geographic_distance': round(geographic_distance, 2),
            'is_fraud': is_fraud,
            'fraud_reason': fraud_reason
        }
        
        self.generation_stats['total_generated'] += 1
        if is_fraud:
            self.generation_stats['fraud_count'] += 1
            
        return transaction
    
    def _generate_amount(self, category: str, age: int, tenure: int) -> float:
        """Generate realistic amount based on category and customer profile"""
        
        category_ranges = {
            '5411': (20, 300),     # Grocery
            '5812': (15, 150),     # Restaurant
            '5942': (10, 100),     # Books
            '5311': (30, 500),     # Department store
            '5944': (100, 3000),   # Jewelry
            '5999': (20, 400),     # Misc
        }
        
        min_amt, max_amt = category_ranges.get(category, (20, 200))
        
        # Age and tenure influence spending power
        spending_factor = 1.0 + (age - 30) * 0.01 + tenure * 0.02
        base = self._rng.lognormal(
            np.log((min_amt + max_amt) / 3),
            0.5
        )
        
        return np.clip(base * spending_factor, min_amt, max_amt * 2)
    
    def _generate_merchant_name(self, category: str) -> str:
        """Generate realistic merchant name based on category"""
        
        merchant_names = {
            '5411': ['Whole Foods', 'Trader Joe\'s', 'Safeway', 'Kroger', 'Publix'],
            '5812': ['Olive Garden', 'Chipotle', 'Panera Bread', 'Subway', 'Local Bistro'],
            '5942': ['Barnes & Noble', 'Books-A-Million', 'Local Bookshop'],
            '5311': ['Macy\'s', 'Nordstrom', 'JCPenney', 'Kohl\'s', 'Target'],
            '5944': ['Kay Jewelers', 'Zales', 'Tiffany & Co', 'Local Jeweler'],
            '5999': ['Amazon', 'Walmart', 'Best Buy', 'Home Depot', 'Costco'],
        }
        
        names = merchant_names.get(category, ['Generic Store'])
        return self._rng.choice(names)
    
    def generate_dataset(self, n_samples: Optional[int] = None) -> pd.DataFrame:
        """Generate complete dataset with specified fraud rate"""
        
        n = n_samples or self.config.n_samples
        n_fraud = int(n * self.config.fraud_rate)
        n_normal = n - n_fraud
        
        transactions = []
        
        # Generate normal transactions
        for _ in range(n_normal):
            transactions.append(self.generate_transaction(is_fraud=False))
        
        # Generate fraud transactions
        for _ in range(n_fraud):
            transactions.append(self.generate_transaction(is_fraud=True))
        
        df = pd.DataFrame(transactions)
        
        # Shuffle
        df = df.sample(frac=1, random_state=self.config.random_seed).reset_index(drop=True)
        
        return df


class SyntheticDataQualityValidator:
    """
    Multi-dimensional quality validation for synthetic data.
    
    Evaluates:
    1. Statistical Fidelity - Distribution matching via Wasserstein distance
    2. Diversity - Coverage of feature space and edge cases
    3. Utility - Downstream model performance
    4. Privacy - Resistance to membership inference attacks
    """
    
    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.numeric_cols = real_data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = real_data.select_dtypes(include=['object', 'bool']).columns.tolist()
        
    def evaluate_statistical_fidelity(self) -> Dict[str, Any]:
        """
        Compute statistical similarity between real and synthetic distributions.
        
        Uses:
        - Wasserstein distance for numeric features
        - Jensen-Shannon divergence for categorical features
        - Kolmogorov-Smirnov test for distribution comparison
        """
        
        results = {
            'numeric_metrics': {},
            'categorical_metrics': {},
            'overall_score': 0.0
        }
        
        # Numeric features
        wasserstein_scores = []
        for col in self.numeric_cols:
            if col in ['transaction_id']:
                continue
                
            real_vals = self.real_data[col].dropna().values
            synth_vals = self.synthetic_data[col].dropna().values
            
            # Normalize for fair comparison
            scaler = StandardScaler()
            real_norm = scaler.fit_transform(real_vals.reshape(-1, 1)).flatten()
            synth_norm = scaler.transform(synth_vals.reshape(-1, 1)).flatten()
            
            # Wasserstein distance
            w_dist = wasserstein_distance(real_norm, synth_norm)
            
            # KS test
            ks_stat, ks_pval = ks_2samp(real_norm, synth_norm)
            
            results['numeric_metrics'][col] = {
                'wasserstein_distance': round(w_dist, 4),
                'ks_statistic': round(ks_stat, 4),
                'ks_pvalue': round(ks_pval, 4),
                'real_mean': round(real_vals.mean(), 4),
                'synth_mean': round(synth_vals.mean(), 4),
                'real_std': round(real_vals.std(), 4),
                'synth_std': round(synth_vals.std(), 4)
            }
            
            # Score: 1 - normalized wasserstein (capped at 1)
            wasserstein_scores.append(max(0, 1 - w_dist))
        
        # Categorical features
        categorical_scores = []
        for col in self.categorical_cols:
            if col in ['transaction_id', 'timestamp', 'fraud_reason', 'merchant_name']:
                continue
                
            real_dist = self.real_data[col].value_counts(normalize=True)
            synth_dist = self.synthetic_data[col].value_counts(normalize=True)
            
            # Align distributions
            all_categories = set(real_dist.index) | set(synth_dist.index)
            real_aligned = [real_dist.get(c, 0) for c in all_categories]
            synth_aligned = [synth_dist.get(c, 0) for c in all_categories]
            
            # Total variation distance
            tv_dist = 0.5 * sum(abs(r - s) for r, s in zip(real_aligned, synth_aligned))
            
            results['categorical_metrics'][col] = {
                'total_variation_distance': round(tv_dist, 4),
                'real_distribution': dict(real_dist),
                'synth_distribution': dict(synth_dist)
            }
            
            categorical_scores.append(1 - tv_dist)
        
        # Overall score
        all_scores = wasserstein_scores + categorical_scores
        results['overall_score'] = round(np.mean(all_scores) if all_scores else 0.0, 4)
        
        return results
    
    def evaluate_diversity(self) -> Dict[str, Any]:
        """
        Assess coverage and diversity of synthetic data.
        
        Metrics:
        - Feature range coverage
        - Edge case representation
        - Unique value counts
        """
        
        results = {
            'range_coverage': {},
            'unique_counts': {},
            'edge_case_coverage': {}
        }
        
        for col in self.numeric_cols:
            if col in ['transaction_id']:
                continue
                
            real_min, real_max = self.real_data[col].min(), self.real_data[col].max()
            synth_min, synth_max = self.synthetic_data[col].min(), self.synthetic_data[col].max()
            
            # Range coverage
            coverage = 1.0
            if real_max > real_min:
                synth_range = synth_max - synth_min
                real_range = real_max - real_min
                coverage = min(synth_range / real_range, 1.0)
            
            results['range_coverage'][col] = round(coverage, 4)
            
            # Edge cases (values in top/bottom 5%)
            real_low = np.percentile(self.real_data[col], 5)
            real_high = np.percentile(self.real_data[col], 95)
            
            synth_low_count = (self.synthetic_data[col] <= real_low).sum()
            synth_high_count = (self.synthetic_data[col] >= real_high).sum()
            
            expected = len(self.synthetic_data) * 0.05
            results['edge_case_coverage'][col] = {
                'low_tail': round(synth_low_count / expected if expected > 0 else 0, 4),
                'high_tail': round(synth_high_count / expected if expected > 0 else 0, 4)
            }
        
        # Unique values for categorical
        for col in self.categorical_cols:
            if col in ['transaction_id', 'timestamp', 'merchant_name']:
                continue
            results['unique_counts'][col] = {
                'real': self.real_data[col].nunique(),
                'synthetic': self.synthetic_data[col].nunique()
            }
        
        return results
    
    def evaluate_utility(self, target_col: str = 'is_fraud') -> Dict[str, Any]:
        """
        Evaluate downstream model performance.
        
        Trains models on:
        1. Real data only
        2. Synthetic data only  
        3. Combined real + synthetic
        
        Compares performance on held-out real test set.
        """
        
        results = {
            'real_only': {},
            'synthetic_only': {},
            'combined': {},
            'utility_score': 0.0
        }
        
        # Prepare features
        feature_cols = [c for c in self.numeric_cols 
                       if c not in ['transaction_id', target_col]]
        
        # Prepare real data
        X_real = self.real_data[feature_cols].values
        y_real = self.real_data[target_col].astype(int).values
        
        X_synth = self.synthetic_data[feature_cols].values
        y_synth = self.synthetic_data[target_col].astype(int).values
        
        # Split real data
        X_train_real, X_test, y_train_real, y_test = train_test_split(
            X_real, y_real, test_size=0.3, random_state=42, stratify=y_real
        )
        
        # Model 1: Train on real only
        model_real = RandomForestClassifier(n_estimators=100, random_state=42)
        model_real.fit(X_train_real, y_train_real)
        y_pred_real = model_real.predict(X_test)
        y_prob_real = model_real.predict_proba(X_test)[:, 1]
        
        results['real_only'] = {
            'accuracy': round(accuracy_score(y_test, y_pred_real), 4),
            'precision': round(precision_score(y_test, y_pred_real, zero_division=0), 4),
            'recall': round(recall_score(y_test, y_pred_real, zero_division=0), 4),
            'f1': round(f1_score(y_test, y_pred_real, zero_division=0), 4),
            'auc_roc': round(roc_auc_score(y_test, y_prob_real), 4)
        }
        
        # Model 2: Train on synthetic only
        model_synth = RandomForestClassifier(n_estimators=100, random_state=42)
        model_synth.fit(X_synth, y_synth)
        y_pred_synth = model_synth.predict(X_test)
        y_prob_synth = model_synth.predict_proba(X_test)[:, 1]
        
        results['synthetic_only'] = {
            'accuracy': round(accuracy_score(y_test, y_pred_synth), 4),
            'precision': round(precision_score(y_test, y_pred_synth, zero_division=0), 4),
            'recall': round(recall_score(y_test, y_pred_synth, zero_division=0), 4),
            'f1': round(f1_score(y_test, y_pred_synth, zero_division=0), 4),
            'auc_roc': round(roc_auc_score(y_test, y_prob_synth), 4)
        }
        
        # Model 3: Train on combined
        X_combined = np.vstack([X_train_real, X_synth])
        y_combined = np.concatenate([y_train_real, y_synth])
        
        model_combined = RandomForestClassifier(n_estimators=100, random_state=42)
        model_combined.fit(X_combined, y_combined)
        y_pred_combined = model_combined.predict(X_test)
        y_prob_combined = model_combined.predict_proba(X_test)[:, 1]
        
        results['combined'] = {
            'accuracy': round(accuracy_score(y_test, y_pred_combined), 4),
            'precision': round(precision_score(y_test, y_pred_combined, zero_division=0), 4),
            'recall': round(recall_score(y_test, y_pred_combined, zero_division=0), 4),
            'f1': round(f1_score(y_test, y_pred_combined, zero_division=0), 4),
            'auc_roc': round(roc_auc_score(y_test, y_prob_combined), 4)
        }
        
        # Utility score: synthetic performance / real performance
        real_auc = results['real_only']['auc_roc']
        synth_auc = results['synthetic_only']['auc_roc']
        results['utility_score'] = round(synth_auc / real_auc if real_auc > 0 else 0, 4)
        
        return results
    
    def evaluate_privacy(self, n_attacks: int = 100) -> Dict[str, Any]:
        """
        Evaluate privacy preservation via membership inference attack simulation.
        
        A successful synthetic data generator should produce data where
        an attacker cannot determine if a specific record was in the training set.
        """
        
        results = {
            'membership_inference': {},
            'privacy_score': 0.0
        }
        
        # Simple membership inference: check if synthetic records are too similar to real
        distances = []
        
        feature_cols = [c for c in self.numeric_cols 
                       if c not in ['transaction_id']]
        
        real_features = self.real_data[feature_cols].values
        synth_features = self.synthetic_data[feature_cols].values
        
        # Normalize
        scaler = StandardScaler()
        real_norm = scaler.fit_transform(real_features)
        synth_norm = scaler.transform(synth_features)
        
        # For each synthetic record, find minimum distance to real records
        for i in range(min(n_attacks, len(synth_norm))):
            synth_record = synth_norm[i]
            min_dist = np.min(np.linalg.norm(real_norm - synth_record, axis=1))
            distances.append(min_dist)
        
        # Privacy score: higher average distance = better privacy
        avg_distance = np.mean(distances)
        
        # Normalize to 0-1 scale (assuming distances typically range 0-10)
        privacy_score = min(avg_distance / 5.0, 1.0)
        
        results['membership_inference'] = {
            'avg_min_distance': round(avg_distance, 4),
            'min_distance': round(min(distances), 4),
            'max_distance': round(max(distances), 4),
            'std_distance': round(np.std(distances), 4)
        }
        results['privacy_score'] = round(privacy_score, 4)
        
        return results
    
    def generate_full_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        
        return {
            'statistical_fidelity': self.evaluate_statistical_fidelity(),
            'diversity': self.evaluate_diversity(),
            'utility': self.evaluate_utility(),
            'privacy': self.evaluate_privacy(),
            'timestamp': datetime.now().isoformat(),
            'real_data_shape': self.real_data.shape,
            'synthetic_data_shape': self.synthetic_data.shape
        }


def main():
    """Main execution demonstrating synthetic data pipeline"""
    
    print("=" * 70)
    print("Synthetic Data Pipeline - Production Demo")
    print("Author: Bhavin Kotak (bhavin.kotak@ibm.com)")
    print("=" * 70)
    
    # Configuration
    config = SyntheticDataConfig(
        n_samples=5000,
        fraud_rate=0.02,
        random_seed=42
    )
    
    # Generate "real" baseline data
    print("\n[1/4] Generating baseline 'real' dataset...")
    real_generator = TransactionGenerator(config)
    real_data = real_generator.generate_dataset(n_samples=2000)
    print(f"   Generated {len(real_data)} real transactions")
    print(f"   Fraud rate: {real_data['is_fraud'].mean():.2%}")
    
    # Generate synthetic data
    print("\n[2/4] Generating synthetic dataset...")
    synth_config = SyntheticDataConfig(
        n_samples=5000,
        fraud_rate=0.02,
        random_seed=123  # Different seed
    )
    synth_generator = TransactionGenerator(synth_config)
    synthetic_data = synth_generator.generate_dataset()
    print(f"   Generated {len(synthetic_data)} synthetic transactions")
    print(f"   Fraud rate: {synthetic_data['is_fraud'].mean():.2%}")
    
    # Validate quality
    print("\n[3/4] Running quality validation...")
    validator = SyntheticDataQualityValidator(real_data, synthetic_data)
    report = validator.generate_full_report()
    
    # Print results
    print("\n" + "=" * 70)
    print("QUALITY VALIDATION REPORT")
    print("=" * 70)
    
    print("\n📊 Statistical Fidelity:")
    print(f"   Overall Score: {report['statistical_fidelity']['overall_score']:.2%}")
    for col, metrics in report['statistical_fidelity']['numeric_metrics'].items():
        print(f"   - {col}: Wasserstein={metrics['wasserstein_distance']:.4f}, "
              f"KS p-value={metrics['ks_pvalue']:.4f}")
    
    print("\n🎯 Diversity:")
    for col, coverage in report['diversity']['range_coverage'].items():
        print(f"   - {col}: Range coverage={coverage:.2%}")
    
    print("\n🤖 Utility (Model Performance):")
    print(f"   Real-trained AUC-ROC: {report['utility']['real_only']['auc_roc']:.4f}")
    print(f"   Synthetic-trained AUC-ROC: {report['utility']['synthetic_only']['auc_roc']:.4f}")
    print(f"   Combined-trained AUC-ROC: {report['utility']['combined']['auc_roc']:.4f}")
    print(f"   Utility Score: {report['utility']['utility_score']:.2%}")
    
    print("\n🔒 Privacy:")
    print(f"   Privacy Score: {report['privacy']['privacy_score']:.2%}")
    print(f"   Avg Min Distance: {report['privacy']['membership_inference']['avg_min_distance']:.4f}")
    
    # Save outputs
    print("\n[4/4] Saving outputs...")
    real_data.to_csv('data/real_transactions.csv', index=False)
    synthetic_data.to_csv('data/synthetic_transactions.csv', index=False)
    
    import json
    with open('data/quality_report.json', 'w') as f:
        # Convert numpy types for JSON serialization
        json.dump(report, f, indent=2, default=str)
    
    print("   Saved: data/real_transactions.csv")
    print("   Saved: data/synthetic_transactions.csv")
    print("   Saved: data/quality_report.json")
    
    print("\n✅ Pipeline completed successfully!")
    print("=" * 70)
    
    return report


if __name__ == "__main__":
    main()
