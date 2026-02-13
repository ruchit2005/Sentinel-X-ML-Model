"""
SentinalX - Quick Start Script
Runs the complete pipeline: Data Generation → Training → Evaluation → Demo
"""

import subprocess
import sys
import os


def print_banner(text):
    """Print a formatted banner"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)
    print()


def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print_banner(description)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True,
            check=True
        )
        print(f"\n✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error in {description}")
        print(f"Exit code: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Script not found: {script_name}")
        return False


def check_dependencies():
    """Check if required packages are installed"""
    print_banner("Checking Dependencies")
    
    required_packages = [
        'numpy',
        'pandas',
        'sklearn',
        'joblib',
        'matplotlib',
        'seaborn'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            missing.append(package)
            print(f"  ✗ {package} (missing)")
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\n📦 Installing missing dependencies...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                check=True
            )
            print("\n✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("\n❌ Failed to install dependencies")
            print("Please run: pip install -r requirements.txt")
            return False
    else:
        print("\n✅ All dependencies installed!")
    
    return True


def main():
    """Run the complete SentinalX pipeline"""
    
    print("\n" + "=" * 70)
    print("  🚀 SentinalX - Complete Pipeline Execution")
    print("=" * 70)
    print("\n  This will run:")
    print("    1. Data Generation (synthetic dataset creation)")
    print("    2. Model Training (hybrid Isolation Forest)")
    print("    3. Model Evaluation (comprehensive metrics)")
    print("    4. Prediction Demo (example predictions)")
    print("\n  Estimated time: 2-5 minutes")
    print("=" * 70)
    
    # Confirm execution
    confirm = input("\n  Proceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("\n👋 Cancelled by user")
        return
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Step 1: Generate Data
    if not run_script('data_generator.py', 'Step 1: Data Generation'):
        print("\n⚠️  Pipeline stopped due to error")
        return
    
    # Step 2: Train Model
    if not run_script('train.py', 'Step 2: Model Training'):
        print("\n⚠️  Pipeline stopped due to error")
        return
    
    # Step 3: Evaluate Model
    if not run_script('evaluate_model.py', 'Step 3: Model Evaluation'):
        print("\n⚠️  Pipeline stopped due to error")
        return
    
    # Step 4: Demo Predictions
    if not run_script('predict.py', 'Step 4: Prediction Demo'):
        print("\n⚠️  Demo failed, but model is trained")
    
    # Final Summary
    print_banner("🎉 Pipeline Execution Complete!")
    
    print("📁 Generated Files:")
    files_to_check = [
        ('Data/training_dataset.csv', 'Training dataset'),
        ('Data/test_dataset.csv', 'Test dataset'),
        ('models/isolation_forest.pkl', 'Trained model'),
        ('models/scaler.pkl', 'Feature scaler'),
        ('models/config.json', 'Model configuration'),
        ('evaluation_report.json', 'Evaluation results')
    ]
    
    for file_path, description in files_to_check:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            size_str = f"{size/1024:.1f}KB" if size < 1024*1024 else f"{size/(1024*1024):.1f}MB"
            print(f"  ✓ {description:25s} ({size_str})")
        else:
            print(f"  ✗ {description:25s} (not found)")
    
    print("\n💡 Next Steps:")
    print("  • Review 'evaluation_report.json' for detailed metrics")
    print("  • Run 'python predict.py --interactive' for manual testing")
    print("  • Check README.md for deployment guidelines")
    print("\n" + "=" * 70)
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Pipeline cancelled by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
