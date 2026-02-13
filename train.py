"""
SentinalX - Training Script
Simple script to train the hybrid fraud detection model
For interactive debugging, use train_model.ipynb instead
"""

from models import HybridFraudDetector
import pandas as pd
from tqdm import tqdm
import time


def main():
    """Train and save the hybrid fraud detection model"""
    
    print("\n" + "=" * 70)
    print("  SentinalX - Hybrid Fraud Detection Model Training")
    print("=" * 70)
    print()
    
    # Step 1: Load training data
    with tqdm(total=100, desc="📂 Loading Data", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        train_df = pd.read_csv('Data/training_dataset.csv')
        pbar.update(100)
    print(f"  ✓ Loaded {len(train_df):,} training samples\n")
    
    # Step 2: Initialize model
    with tqdm(total=100, desc="⚙️  Initializing Model", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        detector = HybridFraudDetector(
            n_estimators=150,      # Increased for better performance
            contamination=0.18,    # Tuned for better precision/recall balance
            max_samples=256,       # Speed optimization
            random_state=42,
            verbose=0              # Keep output clean
        )
        pbar.update(100)
    print("  ✓ Model initialized\n")
    
    # Step 3: Train (model shows its own progress)
    stats = detector.train(train_df)
    
    # Step 4: Save model
    with tqdm(total=100, desc="💾 Saving Model", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        detector.save_model('models')
        pbar.update(100)
    
    print("\n" + "=" * 70)
    print("✨ Training complete! Model ready for deployment.")
    print("=" * 70)
    print("\n💡 Next steps:")
    print("  1. Run 'python evaluate_model.py' to test on holdout data")
    print("  2. Run 'python predict.py' for real-time predictions")
    print()


if __name__ == "__main__":
    main()
