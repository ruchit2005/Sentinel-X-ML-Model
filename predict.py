"""
SentinalX - Hybrid Fraud Detection Inference System
Real-time fraud detection using Rule-Based + Isolation Forest hybrid approach
Optimized for <50ms inference time
"""

import numpy as np
import pandas as pd
import joblib
import json
from typing import Dict, List
import time


class FraudDetectionAPI:
    """
    Production-ready fraud detection API
    Provides <50ms inference time with hybrid detection
    """
    
    def __init__(self, model_dir: str = 'models'):
        """Load trained model and configurations"""
        self.model_dir = model_dir
        
        # Load models
        print("🔄 Loading fraud detection models...")
        self.isolation_forest = joblib.load(f'{model_dir}/isolation_forest.pkl')
        self.scaler = joblib.load(f'{model_dir}/scaler.pkl')
        
        # Load configuration
        with open(f'{model_dir}/config.json', 'r') as f:
            config = json.load(f)
        
        self.feature_columns = config['feature_columns']
        print("✅ Models loaded successfully!")
        
    def apply_hard_rule(self, call_frequency: float, avg_call_distance: float) -> bool:
        """
        Stage 1: Hard Rule Check
        
        Rule: callFrequency > 50 AND avgCallDistance < 10
        Result: Instant LEGITIMATE classification (Delivery Partner)
        
        Returns:
            True if hard rule is satisfied (safe user)
            False if needs ML evaluation
        """
        return (call_frequency > 50) and (avg_call_distance < 10)
    
    def calculate_features(self, user_data: Dict) -> Dict:
        """
        Calculate engineered features from raw user data
        
        Args:
            user_data: Dictionary with basic call pattern features
            
        Returns:
            Dictionary with all features including engineered ones
        """
        # Extract base features
        call_freq = user_data['callFrequency']
        avg_duration = user_data['avgDuration']
        unique_contacts = user_data['uniqueContacts']
        avg_distance = user_data['avgCallDistance']
        circle_div = user_data['circleDiversity']
        
        # Calculate engineered features
        features = {
            'avgDuration': avg_duration,
            'callFrequency': call_freq,
            'uniqueContacts': unique_contacts,
            'avgCallDistance': avg_distance,
            'circleDiversity': circle_div,
            'call_intensity': call_freq / (avg_duration + 1e-6),
            'distance_per_call': avg_distance / (call_freq + 1e-6),
            'contact_circle_ratio': unique_contacts / (circle_div + 1),
            'high_freq_long_distance': int((call_freq > 40) and (avg_distance > 1000))
        }
        
        return features
    
    def predict_single(self, user_data: Dict) -> Dict:
        """
        Predict fraud/legitimate for a single user
        
        Args:
            user_data: Dictionary with user call pattern data
                Required keys: callFrequency, avgDuration, uniqueContacts,
                              avgCallDistance, circleDiversity
        
        Returns:
            Dictionary with prediction results:
            - isAnomaly: bool
            - confidence: float (0-1)
            - riskType: str
            - detectionStage: str
            - inferenceTime: float (ms)
            - features: dict
        """
        start_time = time.time()
        
        # Calculate all features
        features = self.calculate_features(user_data)
        
        # Stage 1: Hard Rule Check
        if self.apply_hard_rule(features['callFrequency'], features['avgCallDistance']):
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return {
                'isAnomaly': False,
                'prediction': 'LEGITIMATE',
                'confidence': 1.0,
                'riskType': 'LEGITIMATE_HIGH_FREQUENCY',
                'detectionStage': 'RULE_BASED',
                'reason': 'Delivery partner pattern detected (high frequency + low distance)',
                'inferenceTime': round(inference_time, 2),
                'features': features
            }
        
        # Stage 2: ML Prediction
        # Prepare features for model
        X = pd.DataFrame([features])[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.isolation_forest.predict(X_scaled)[0]
        anomaly_score = self.isolation_forest.decision_function(X_scaled)[0]
        
        # Convert to meaningful output
        is_anomaly = (prediction == -1)
        
        # Calculate confidence (based on anomaly score)
        # More negative score = more anomalous = higher fraud confidence
        confidence = 1 / (1 + np.exp(anomaly_score))  # Sigmoid transformation
        
        # Determine risk type based on pattern
        if is_anomaly:
            if features['high_freq_long_distance']:
                risk_type = 'HIGH_RISK_CROSS_STATE_OPERATION'
            elif features['call_intensity'] > 5:
                risk_type = 'HIGH_RISK_BOT_LIKE_BEHAVIOR'
            elif features['contact_circle_ratio'] > 30:
                risk_type = 'HIGH_RISK_MASS_CALLING'
            else:
                risk_type = 'MEDIUM_RISK_ANOMALY'
        else:
            risk_type = 'NORMAL_PATTERN'
        
        inference_time = (time.time() - start_time) * 1000
        
        return {
            'isAnomaly': bool(is_anomaly),
            'prediction': 'FRAUD' if is_anomaly else 'LEGITIMATE',
            'confidence': round(float(confidence), 4),
            'anomalyScore': round(float(anomaly_score), 4),
            'riskType': risk_type,
            'detectionStage': 'ML_ISOLATION_FOREST',
            'reason': self._generate_reason(features, is_anomaly),
            'inferenceTime': round(inference_time, 2),
            'features': features
        }
    
    def predict_batch(self, users_data: List[Dict]) -> List[Dict]:
        """
        Predict fraud/legitimate for multiple users
        Optimized for batch processing
        
        Args:
            users_data: List of user data dictionaries
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for user_data in users_data:
            result = self.predict_single(user_data)
            results.append(result)
        
        return results
    
    def _generate_reason(self, features: Dict, is_anomaly: bool) -> str:
        """Generate human-readable explanation for prediction"""
        if not is_anomaly:
            return "Normal calling pattern detected"
        
        reasons = []
        
        if features['high_freq_long_distance']:
            reasons.append("High frequency calls across long distances")
        
        if features['call_intensity'] > 5:
            reasons.append("Abnormally high call intensity (short duration, high frequency)")
        
        if features['contact_circle_ratio'] > 30:
            reasons.append("Large number of contacts relative to location diversity")
        
        if features['avgCallDistance'] > 1000:
            reasons.append("Cross-state calling pattern")
        
        if features['avgDuration'] < 15:
            reasons.append("Very short average call duration")
        
        if not reasons:
            reasons.append("Anomalous pattern detected by ML model")
        
        return "; ".join(reasons)


def demo_predictions():
    """Demonstrate the fraud detection system with example cases"""
    
    print("\n" + "=" * 70)
    print("  SentinalX - Fraud Detection Demo")
    print("=" * 70)
    print()
    
    # Initialize API
    api = FraudDetectionAPI('models')
    
    # Test cases
    test_cases = [
        {
            'name': '👤 Delivery Partner',
            'data': {
                'callFrequency': 65,
                'avgDuration': 8.5,
                'uniqueContacts': 85,
                'avgCallDistance': 6.2,
                'circleDiversity': 1
            },
            'expected': 'LEGITIMATE'
        },
        {
            'name': '👤 Regular User',
            'data': {
                'callFrequency': 18,
                'avgDuration': 120,
                'uniqueContacts': 25,
                'avgCallDistance': 85,
                'circleDiversity': 2
            },
            'expected': 'LEGITIMATE'
        },
        {
            'name': '⚠️  Digital Arrest Bot',
            'data': {
                'callFrequency': 89,
                'avgDuration': 5.2,
                'uniqueContacts': 156,
                'avgCallDistance': 2100,
                'circleDiversity': 6
            },
            'expected': 'FRAUD'
        },
        {
            'name': '⚠️  Traditional Scammer',
            'data': {
                'callFrequency': 45,
                'avgDuration': 35,
                'uniqueContacts': 68,
                'avgCallDistance': 1450,
                'circleDiversity': 4
            },
            'expected': 'FRAUD'
        },
        {
            'name': '👤 Business User',
            'data': {
                'callFrequency': 32,
                'avgDuration': 65,
                'uniqueContacts': 48,
                'avgCallDistance': 320,
                'circleDiversity': 2
            },
            'expected': 'LEGITIMATE'
        }
    ]
    
    print("🔍 Running predictions on test cases...")
    print("=" * 70)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 70)
        
        # Make prediction
        result = api.predict_single(test_case['data'])
        
        # Display input
        print("📊 Input Features:")
        for key, value in test_case['data'].items():
            print(f"  • {key}: {value}")
        
        # Display result
        print(f"\n🎯 Prediction:")
        print(f"  • Result: {result['prediction']}")
        print(f"  • Confidence: {result['confidence']*100:.2f}%")
        print(f"  • Risk Type: {result['riskType']}")
        print(f"  • Detection Stage: {result['detectionStage']}")
        print(f"  • Reason: {result['reason']}")
        print(f"  • Inference Time: {result['inferenceTime']:.2f}ms ⚡")
        
        # Check if correct
        is_correct = result['prediction'] == test_case['expected']
        status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
        print(f"\n  {status} (Expected: {test_case['expected']})")
    
    print("\n" + "=" * 70)
    print("✨ Demo complete!")
    print("=" * 70)
    print()


def interactive_mode():
    """Interactive mode for manual testing"""
    
    print("\n" + "=" * 70)
    print("  SentinalX - Interactive Fraud Detection")
    print("=" * 70)
    print()
    
    # Initialize API
    api = FraudDetectionAPI('models')
    
    print("\n📝 Enter user call pattern data (or 'quit' to exit):")
    print()
    
    while True:
        try:
            print("-" * 70)
            
            # Get user input
            call_freq = input("Call Frequency (e.g., 45): ")
            if call_freq.lower() == 'quit':
                break
            
            avg_duration = input("Average Duration in seconds (e.g., 35.5): ")
            unique_contacts = input("Unique Contacts (e.g., 68): ")
            avg_distance = input("Average Call Distance in km (e.g., 1450): ")
            circle_div = input("Circle Diversity (e.g., 4): ")
            
            # Create user data
            user_data = {
                'callFrequency': float(call_freq),
                'avgDuration': float(avg_duration),
                'uniqueContacts': int(unique_contacts),
                'avgCallDistance': float(avg_distance),
                'circleDiversity': int(circle_div)
            }
            
            # Make prediction
            result = api.predict_single(user_data)
            
            # Display result
            print(f"\n🎯 Prediction Result:")
            print(f"  • Prediction: {result['prediction']}")
            print(f"  • Confidence: {result['confidence']*100:.2f}%")
            print(f"  • Risk Type: {result['riskType']}")
            print(f"  • Detection Stage: {result['detectionStage']}")
            print(f"  • Reason: {result['reason']}")
            print(f"  • Inference Time: {result['inferenceTime']:.2f}ms\n")
            
        except ValueError as e:
            print(f"❌ Invalid input: {e}")
        except KeyboardInterrupt:
            break
    
    print("\n👋 Goodbye!")


def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_mode()
    else:
        demo_predictions()


if __name__ == "__main__":
    main()
