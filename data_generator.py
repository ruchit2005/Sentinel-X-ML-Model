"""
SentinalX - Advanced Data Generation Script
Generates realistic synthetic telecom data for fraud detection model training
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import random


class TelecomDataGenerator:
    """Generates realistic telecom call pattern data with multiple user profiles"""
    
    def __init__(self, random_seed: int = 42):
        np.random.seed(random_seed)
        random.seed(random_seed)
        
    def generate_delivery_partner(self, n_samples: int) -> pd.DataFrame:
        """
        SAFE Profile: Delivery Partner
        - High call frequency (40-80 calls)
        - Short durations (3-15 mins)
        - Moderate unique contacts (30-100)
        - CRITICAL: Low distance (<10km) - This triggers the hard rule
        - Same city operations (circleDiversity = 1)
        """
        data = {
            'callFrequency': np.random.randint(40, 81, n_samples),
            'avgDuration': np.random.uniform(3, 15, n_samples),
            'uniqueContacts': np.random.randint(30, 101, n_samples),
            'avgCallDistance': np.random.uniform(0.5, 9.5, n_samples),  # <10km CRITICAL
            'circleDiversity': np.ones(n_samples, dtype=int),  # Always 1 for delivery
            'label': ['LEGITIMATE'] * n_samples,
            'userType': ['DELIVERY_PARTNER'] * n_samples
        }
        return pd.DataFrame(data)
    
    def generate_regular_user(self, n_samples: int) -> pd.DataFrame:
        """
        SAFE Profile: Regular User
        - Low to moderate call frequency (5-25 calls)
        - Longer conversations (30-300 seconds)
        - Moderate contacts (10-50)
        - Variable distance (5-500km)
        - Low circle diversity (1-3)
        """
        data = {
            'callFrequency': np.random.randint(5, 26, n_samples),
            'avgDuration': np.random.uniform(30, 300, n_samples),
            'uniqueContacts': np.random.randint(10, 51, n_samples),
            'avgCallDistance': np.random.uniform(5, 500, n_samples),
            'circleDiversity': np.random.randint(1, 4, n_samples),
            'label': ['LEGITIMATE'] * n_samples,
            'userType': ['REGULAR_USER'] * n_samples
        }
        return pd.DataFrame(data)
    
    def generate_business_user(self, n_samples: int) -> pd.DataFrame:
        """
        SAFE Profile: Business User
        - Moderate call frequency (20-50 calls)
        - Medium durations (15-120 seconds)
        - Good contact base (25-75)
        - Moderate distance (10-800km)
        - Some travel (1-3 circles)
        """
        data = {
            'callFrequency': np.random.randint(20, 51, n_samples),
            'avgDuration': np.random.uniform(15, 120, n_samples),
            'uniqueContacts': np.random.randint(25, 76, n_samples),
            'avgCallDistance': np.random.uniform(10, 800, n_samples),
            'circleDiversity': np.random.randint(1, 4, n_samples),
            'label': ['LEGITIMATE'] * n_samples,
            'userType': ['BUSINESS_USER'] * n_samples
        }
        return pd.DataFrame(data)
    
    def generate_traveling_professional(self, n_samples: int) -> pd.DataFrame:
        """
        SAFE Profile: Traveling Professional
        - Moderate frequency (15-40 calls)
        - Longer calls (60-250 seconds)
        - Decent contacts (20-60)
        - Higher distance (100-1500km)
        - Multiple circles (2-4)
        """
        data = {
            'callFrequency': np.random.randint(15, 41, n_samples),
            'avgDuration': np.random.uniform(60, 250, n_samples),
            'uniqueContacts': np.random.randint(20, 61, n_samples),
            'avgCallDistance': np.random.uniform(100, 1500, n_samples),
            'circleDiversity': np.random.randint(2, 5, n_samples),
            'label': ['LEGITIMATE'] * n_samples,
            'userType': ['TRAVELING_PROFESSIONAL'] * n_samples
        }
        return pd.DataFrame(data)
    
    def generate_digital_arrest_bot(self, n_samples: int) -> pd.DataFrame:
        """
        FRAUD Profile: Digital Arrest Scam Bot
        - Very high frequency (45-100+ calls)
        - Very short calls (2-10 seconds) - Automated bot behavior
        - Many victims (50-200 contacts)
        - Cross-state operations (1000-3000km)
        - Multiple circles (4-8) - Wide geographical spread
        """
        data = {
            'callFrequency': np.random.randint(45, 101, n_samples),
            'avgDuration': np.random.uniform(2, 10, n_samples),  # Bot-like short calls
            'uniqueContacts': np.random.randint(50, 201, n_samples),
            'avgCallDistance': np.random.uniform(1000, 3000, n_samples),  # Cross-state
            'circleDiversity': np.random.randint(4, 9, n_samples),  # Wide spread
            'label': ['FRAUD'] * n_samples,
            'userType': ['DIGITAL_ARREST_BOT'] * n_samples
        }
        return pd.DataFrame(data)
    
    def generate_traditional_scammer(self, n_samples: int) -> pd.DataFrame:
        """
        FRAUD Profile: Traditional Phone Scammer
        - Moderate to high frequency (30-60 calls)
        - Short to medium calls (10-60 seconds)
        - Multiple victims (20-80 contacts)
        - Long distance operations (500-2000km)
        - Multiple circles (2-5)
        """
        data = {
            'callFrequency': np.random.randint(30, 61, n_samples),
            'avgDuration': np.random.uniform(10, 60, n_samples),
            'uniqueContacts': np.random.randint(20, 81, n_samples),
            'avgCallDistance': np.random.uniform(500, 2000, n_samples),
            'circleDiversity': np.random.randint(2, 6, n_samples),
            'label': ['FRAUD'] * n_samples,
            'userType': ['TRADITIONAL_SCAMMER'] * n_samples
        }
        return pd.DataFrame(data)
    
    def generate_low_volume_scammer(self, n_samples: int) -> pd.DataFrame:
        """
        FRAUD Profile: Low Volume Scammer (Targeted attacks)
        - Lower frequency (20-45 calls) - More targeted
        - Medium calls (15-90 seconds)
        - Fewer victims (15-50 contacts)
        - Long distance (800-2500km)
        - Multiple circles (3-6)
        """
        data = {
            'callFrequency': np.random.randint(20, 46, n_samples),
            'avgDuration': np.random.uniform(15, 90, n_samples),
            'uniqueContacts': np.random.randint(15, 51, n_samples),
            'avgCallDistance': np.random.uniform(800, 2500, n_samples),
            'circleDiversity': np.random.randint(3, 7, n_samples),
            'label': ['FRAUD'] * n_samples,
            'userType': ['LOW_VOLUME_SCAMMER'] * n_samples
        }
        return pd.DataFrame(data)
    
    def add_phone_numbers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate realistic Indian phone numbers"""
        phone_numbers = [f"+91{np.random.randint(6000000000, 9999999999)}" 
                        for _ in range(len(df))]
        df.insert(0, 'phoneNumber', phone_numbers)
        return df
    
    def add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add critical engineered features for better fraud detection
        These features help the model identify patterns more effectively
        """
        # Call intensity: High frequency with short duration is suspicious
        df['call_intensity'] = df['callFrequency'] / (df['avgDuration'] + 1e-6)
        
        # Distance per call: High values indicate wide-spread operations
        df['distance_per_call'] = df['avgCallDistance'] / (df['callFrequency'] + 1e-6)
        
        # Contact to circle ratio: Many contacts in few circles is suspicious
        df['contact_circle_ratio'] = df['uniqueContacts'] / (df['circleDiversity'] + 1)
        
        # Binary flag: High frequency + long distance = likely fraud
        df['high_freq_long_distance'] = (
            (df['callFrequency'] > 40) & (df['avgCallDistance'] > 1000)
        ).astype(int)
        
        # Delivery pattern flag: Meets the hard rule criteria
        df['delivery_pattern'] = (
            (df['callFrequency'] > 50) & (df['avgCallDistance'] < 10)
        ).astype(int)
        
        return df
    
    def generate_dataset(
        self, 
        n_delivery: int = 2000,
        n_regular: int = 4000,
        n_business: int = 1500,
        n_traveling: int = 1000,
        n_digital_arrest: int = 2000,
        n_traditional: int = 1500,
        n_low_volume: int = 1000,
        shuffle: bool = True
    ) -> pd.DataFrame:
        """
        Generate complete dataset with all user profiles
        
        Default distribution:
        - Total samples: 13,000
        - Legitimate: 8,500 (65%)
        - Fraud: 4,500 (35%)
        """
        print("🔄 Generating synthetic telecom data...")
        
        # Generate each profile
        profiles = [
            ("Delivery Partners", self.generate_delivery_partner(n_delivery)),
            ("Regular Users", self.generate_regular_user(n_regular)),
            ("Business Users", self.generate_business_user(n_business)),
            ("Traveling Professionals", self.generate_traveling_professional(n_traveling)),
            ("Digital Arrest Bots", self.generate_digital_arrest_bot(n_digital_arrest)),
            ("Traditional Scammers", self.generate_traditional_scammer(n_traditional)),
            ("Low Volume Scammers", self.generate_low_volume_scammer(n_low_volume))
        ]
        
        # Combine all profiles
        dfs = []
        for name, profile_df in profiles:
            print(f"  ✓ Generated {len(profile_df)} {name}")
            dfs.append(profile_df)
        
        df = pd.concat(dfs, ignore_index=True)
        
        # Add phone numbers and engineered features
        df = self.add_phone_numbers(df)
        df = self.add_engineered_features(df)
        
        # Shuffle if requested
        if shuffle:
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\n✅ Dataset generation complete!")
        print(f"  Total samples: {len(df)}")
        print(f"  Legitimate: {(df['label'] == 'LEGITIMATE').sum()} ({(df['label'] == 'LEGITIMATE').sum()/len(df)*100:.1f}%)")
        print(f"  Fraud: {(df['label'] == 'FRAUD').sum()} ({(df['label'] == 'FRAUD').sum()/len(df)*100:.1f}%)")
        
        return df


def main():
    """Generate training and test datasets"""
    
    print("=" * 70)
    print("  SentinalX - Telecom Fraud Detection Dataset Generator")
    print("=" * 70)
    print()
    
    generator = TelecomDataGenerator(random_seed=42)
    
    # Generate training dataset (80% of data)
    print("📊 TRAINING DATASET")
    print("-" * 70)
    train_df = generator.generate_dataset(
        n_delivery=2000,
        n_regular=4000,
        n_business=1500,
        n_traveling=1000,
        n_digital_arrest=2000,
        n_traditional=1500,
        n_low_volume=1000,
        shuffle=True
    )
    train_df.to_csv('Data/training_dataset.csv', index=False)
    print(f"  💾 Saved to: Data/training_dataset.csv")
    
    print()
    
    # Generate test dataset (20% of data)
    print("📊 TEST DATASET")
    print("-" * 70)
    generator_test = TelecomDataGenerator(random_seed=2026)  # Different seed
    test_df = generator_test.generate_dataset(
        n_delivery=400,
        n_regular=800,
        n_business=300,
        n_traveling=200,
        n_digital_arrest=400,
        n_traditional=300,
        n_low_volume=200,
        shuffle=True
    )
    test_df.to_csv('Data/test_dataset.csv', index=False)
    print(f"  💾 Saved to: Data/test_dataset.csv")
    
    print()
    print("=" * 70)
    print("✨ Dataset generation complete! Ready for model training.")
    print("=" * 70)
    
    # Display sample statistics
    print("\n📈 DATASET STATISTICS")
    print("-" * 70)
    print("\nTraining Set Profile Distribution:")
    print(train_df['userType'].value_counts().sort_index())
    print("\nTest Set Profile Distribution:")
    print(test_df['userType'].value_counts().sort_index())
    
    # Check delivery pattern rule
    delivery_protected = train_df[train_df['delivery_pattern'] == 1]
    print(f"\n🛡️  Protected Delivery Partners (Hard Rule): {len(delivery_protected)}")
    print(f"   (All will have 0% false positive rate)")


if __name__ == "__main__":
    main()
