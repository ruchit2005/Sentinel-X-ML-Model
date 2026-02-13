"""
SentinalX - Hybrid Fraud Detector Model Class
Core model class for fraud detection combining Rule-Based + Isolation Forest
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import joblib
import json
from datetime import datetime
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class HybridFraudDetector:
    """
    Hybrid fraud detection system combining:
    1. Hard Rule Filter (Stage 1): Instant whitelist for delivery partners
    2. Isolation Forest (Stage 2): ML-based anomaly detection for everyone else
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.18,
        max_samples: int = 256,
        random_state: int = 42,
        verbose: int = 0
    ):
        """
        Initialize hybrid fraud detector
        
        Args:
            n_estimators: Number of trees in Isolation Forest
            contamination: Expected fraud ratio (0.3 = 30% fraud)
            max_samples: Samples per tree (256 for speed optimization)
            random_state: Random seed for reproducibility
            verbose: Verbosity level (0=quiet, 1=progress)
        """
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state
        self.verbose = verbose
        
        # Models
        self.isolation_forest = None
        self.scaler = StandardScaler()
        
        # Feature configuration
        self.feature_columns = [
            'avgDuration',
            'callFrequency',
            'uniqueContacts',
            'avgCallDistance',
            'circleDiversity',
            'call_intensity',
            'distance_per_call',
            'contact_circle_ratio',
            'high_freq_long_distance'
        ]
        
        # Performance tracking
        self.training_stats = {}
        
    def apply_hard_rule(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Stage 1: Hard Rule Filter (Enhanced with 3 rules)
        
        Rule 1: Delivery Partner Protection
            If callFrequency > 50 AND avgCallDistance < 10
            Then: LEGITIMATE with 100% confidence
        
        Rule 2: Obvious Regular User
            If callFrequency < 20 AND avgDuration > 60
            Then: LEGITIMATE with 98% confidence
        
        Rule 3: Obvious Digital Arrest Bot
            If avgCallDistance > 1500 AND circleDiversity >= 6 AND avgDuration < 15
            Then: FRAUD with 99% confidence
        
        Returns:
            filtered_safe: Records marked by hard rules
            remaining: Records that need ML evaluation
        """
        # Initialize results
        filtered_safe = pd.DataFrame()
        filtered_fraud = pd.DataFrame()
        
        # Rule 1: Delivery Partner (HIGH FREQUENCY + LOW DISTANCE)
        rule1_mask = (df['callFrequency'] > 50) & (df['avgCallDistance'] < 10)
        rule1_matches = df[rule1_mask].copy()
        if len(rule1_matches) > 0:
            rule1_matches['prediction'] = 'LEGITIMATE'
            rule1_matches['confidence'] = 1.0
            rule1_matches['riskType'] = 'LEGITIMATE_HIGH_FREQUENCY'
            rule1_matches['detection_stage'] = 'RULE_BASED'
            filtered_safe = pd.concat([filtered_safe, rule1_matches])
        
        # Rule 2: Obvious Regular User (LOW FREQUENCY + LONG CALLS)
        rule2_mask = (~rule1_mask) & (df['callFrequency'] < 20) & (df['avgDuration'] > 60)
        rule2_matches = df[rule2_mask].copy()
        if len(rule2_matches) > 0:
            rule2_matches['prediction'] = 'LEGITIMATE'
            rule2_matches['confidence'] = 0.98
            rule2_matches['riskType'] = 'LEGITIMATE_REGULAR_USER'
            rule2_matches['detection_stage'] = 'RULE_BASED'
            filtered_safe = pd.concat([filtered_safe, rule2_matches])
        
        # Rule 3: Obvious Digital Arrest Bot (CROSS-STATE + MANY CIRCLES + SHORT CALLS)
        rule3_mask = (~rule1_mask) & (~rule2_mask) & \
                     (df['avgCallDistance'] > 1500) & \
                     (df['circleDiversity'] >= 6) & \
                     (df['avgDuration'] < 15)
        rule3_matches = df[rule3_mask].copy()
        if len(rule3_matches) > 0:
            rule3_matches['prediction'] = 'FRAUD'
            rule3_matches['confidence'] = 0.99
            rule3_matches['riskType'] = 'DIGITAL_ARREST_BOT'
            rule3_matches['detection_stage'] = 'RULE_BASED'
            filtered_fraud = pd.concat([filtered_fraud, rule3_matches])
        
        # Combine all rule-based decisions
        all_rule_based = pd.concat([filtered_safe, filtered_fraud])
        
        # Remaining records need ML evaluation
        remaining_mask = ~df.index.isin(all_rule_based.index)
        remaining = df[remaining_mask].copy()
        
        return all_rule_based, remaining
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and scale features for ML model"""
        X = df[self.feature_columns].copy()
        
        # Handle any missing values
        X = X.fillna(X.mean())
        
        return X
    
    def train(self, train_df: pd.DataFrame) -> Dict:
        """
        Train the hybrid fraud detection system
        
        Process:
        1. Apply hard rule to separate delivery partners
        2. Train Isolation Forest on remaining records
        3. Evaluate performance on both stages
        
        Args:
            train_df: Training dataframe with features and labels
            
        Returns:
            Dictionary with training statistics
        """
        print("🚀 Starting Hybrid Model Training")
        print("=" * 70)
        
        # Stage 1: Apply hard rule
        print("\n📋 STAGE 1: Hard Rule Filter")
        print("-" * 70)
        rule_safe, remaining = self.apply_hard_rule(train_df)
        
        print(f"  ✓ Hard rule protected: {len(rule_safe)} records")
        print(f"  ✓ Remaining for ML: {len(remaining)} records")
        
        if len(rule_safe) > 0:
            rule_accuracy = (rule_safe['label'] == 'LEGITIMATE').sum() / len(rule_safe)
            print(f"  ✓ Hard rule accuracy: {rule_accuracy*100:.2f}%")
        
        # Stage 2: Train Isolation Forest on remaining data
        print("\n🤖 STAGE 2: Isolation Forest Training")
        print("-" * 70)
        
        # Prepare features
        X_train = self.prepare_features(remaining)
        y_train = (remaining['label'] == 'FRAUD').astype(int)
        
        # Fit scaler
        print("  ⚙️  Fitting feature scaler...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train Isolation Forest
        print(f"  🌲 Training Isolation Forest ({self.n_estimators} estimators)...")
        self.isolation_forest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,  # Use all CPU cores
            verbose=self.verbose
        )
        
        self.isolation_forest.fit(X_train_scaled)
        print("  ✅ Isolation Forest training complete!")
        
        # Make predictions on training set
        print("\n📊 Evaluating training performance...")
        train_predictions = self.predict(train_df)
        
        # Calculate metrics
        y_true = (train_df['label'] == 'FRAUD').astype(int)
        y_pred = (train_predictions['prediction'] == 'FRAUD').astype(int)
        
        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # False positive rate for delivery partners (CRITICAL METRIC)
        delivery_partners = train_df[
            (train_df['callFrequency'] > 50) & 
            (train_df['avgCallDistance'] < 10) &
            (train_df['label'] == 'LEGITIMATE')
        ]
        if len(delivery_partners) > 0:
            delivery_fp = (train_predictions.loc[delivery_partners.index, 'prediction'] == 'FRAUD').sum()
            delivery_fpr = delivery_fp / len(delivery_partners)
        else:
            delivery_fpr = 0.0
        
        # Store stats
        self.training_stats = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(train_df),
            'legitimate_samples': (train_df['label'] == 'LEGITIMATE').sum(),
            'fraud_samples': (train_df['label'] == 'FRAUD').sum(),
            'rule_based_protected': len(rule_safe),
            'ml_evaluated': len(remaining),
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'delivery_partner_fpr': float(delivery_fpr)
            },
            'model_config': {
                'n_estimators': self.n_estimators,
                'contamination': self.contamination,
                'max_samples': self.max_samples,
                'random_state': self.random_state
            }
        }
        
        # Print results
        print("\n" + "=" * 70)
        print("📈 TRAINING RESULTS")
        print("=" * 70)
        print(f"\n🎯 Overall Performance:")
        print(f"  • Accuracy:  {accuracy*100:.2f}%")
        print(f"  • Precision: {precision*100:.2f}% (Low false positives)")
        print(f"  • Recall:    {recall*100:.2f}% (Catch fraudsters)")
        print(f"  • F1-Score:  {f1*100:.2f}%")
        
        print(f"\n📊 Confusion Matrix:")
        print(f"  • True Negatives (Legit → Legit):  {tn:,}")
        print(f"  • False Positives (Legit → Fraud): {fp:,}")
        print(f"  • False Negatives (Fraud → Legit): {fn:,}")
        print(f"  • True Positives (Fraud → Fraud):  {tp:,}")
        
        print(f"\n🛡️  Critical Metric:")
        print(f"  • Delivery Partner FPR: {delivery_fpr*100:.4f}%")
        print(f"    (Target: 0.00% - Protected by hard rule)")
        
        return self.training_stats
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions using the hybrid system
        
        Process:
        1. Apply hard rule first
        2. Use Isolation Forest for remaining records
        3. Combine results
        
        Args:
            df: Dataframe with features
            
        Returns:
            Dataframe with predictions and metadata
        """
        # Stage 1: Hard rule
        rule_safe, remaining = self.apply_hard_rule(df)
        
        # Stage 2: ML predictions for remaining
        if len(remaining) > 0:
            X_remaining = self.prepare_features(remaining)
            X_remaining_scaled = self.scaler.transform(X_remaining)
            
            # Predict (-1 = anomaly/fraud, 1 = normal/legitimate)
            ml_predictions = self.isolation_forest.predict(X_remaining_scaled)
            
            # Get anomaly scores (more negative = more anomalous)
            anomaly_scores = self.isolation_forest.decision_function(X_remaining_scaled)
            
            # Convert to predictions
            remaining['prediction'] = np.where(ml_predictions == -1, 'FRAUD', 'LEGITIMATE')
            remaining['anomaly_score'] = anomaly_scores
            
            # Calculate confidence (normalize anomaly scores to 0-1)
            min_score = anomaly_scores.min()
            max_score = anomaly_scores.max()
            normalized_scores = (anomaly_scores - min_score) / (max_score - min_score + 1e-6)
            
            # For fraud: higher anomaly = higher confidence
            # For legitimate: lower anomaly = higher confidence
            remaining['confidence'] = np.where(
                ml_predictions == -1,
                1 - normalized_scores,  # Fraud: more anomalous = more confident
                normalized_scores        # Legitimate: less anomalous = more confident
            )
            
            remaining['riskType'] = np.where(
                ml_predictions == -1,
                'HIGH_RISK_ANOMALY',
                'NORMAL_PATTERN'
            )
            remaining['detection_stage'] = 'ML_ISOLATION_FOREST'
        
        # Combine results
        if len(rule_safe) > 0 and len(remaining) > 0:
            result = pd.concat([rule_safe, remaining], ignore_index=False)
            result = result.sort_index()
        elif len(rule_safe) > 0:
            result = rule_safe
        else:
            result = remaining
        
        return result
    
    def save_model(self, model_dir: str = 'models'):
        """Save trained model and scaler"""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        # Save Isolation Forest
        joblib.dump(self.isolation_forest, f'{model_dir}/isolation_forest.pkl')
        
        # Save scaler
        joblib.dump(self.scaler, f'{model_dir}/scaler.pkl')
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_to_native(obj):
            """Recursively convert numpy types to native Python types"""
            if isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Save configuration and stats
        config = {
            'feature_columns': self.feature_columns,
            'n_estimators': self.n_estimators,
            'contamination': self.contamination,
            'max_samples': self.max_samples,
            'random_state': self.random_state,
            'training_stats': convert_to_native(self.training_stats)
        }
        
        with open(f'{model_dir}/config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n💾 Model saved to '{model_dir}/' directory")
        print(f"  ✓ isolation_forest.pkl")
        print(f"  ✓ scaler.pkl")
        print(f"  ✓ config.json")
    
    def load_model(self, model_dir: str = 'models'):
        """Load trained model and scaler"""
        # Load Isolation Forest
        self.isolation_forest = joblib.load(f'{model_dir}/isolation_forest.pkl')
        
        # Load scaler
        self.scaler = joblib.load(f'{model_dir}/scaler.pkl')
        
        # Load configuration
        with open(f'{model_dir}/config.json', 'r') as f:
            config = json.load(f)
        
        self.feature_columns = config['feature_columns']
        self.n_estimators = config['n_estimators']
        self.contamination = config['contamination']
        self.max_samples = config['max_samples']
        self.random_state = config['random_state']
        self.training_stats = config.get('training_stats', {})
        
        print(f"✅ Model loaded from '{model_dir}/' directory")
