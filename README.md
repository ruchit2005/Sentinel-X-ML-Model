# SentinalX - Hybrid Fraud Detection System

## 🎯 Overview

SentinalX is an advanced telecom fraud detection system that uses a **hybrid approach** combining rule-based filtering with machine learning to achieve optimal performance with **zero false positives** for legitimate high-frequency users (delivery partners).

### Key Features

✅ **Hybrid Detection Architecture**
- Stage 1: Hard rule filter for instant whitelisting
- Stage 2: Isolation Forest ML for sophisticated anomaly detection

✅ **Performance Targets**
- **Accuracy**: 96-98%
- **Precision**: 95%+ (low false positives)
- **Recall**: 98%+ (catch all fraudsters)
- **Delivery Partner FPR**: 0% (guaranteed by hard rule)
- **Inference Time**: <50ms per prediction

✅ **Production Ready**
- Fast inference (<50ms)
- Scalable architecture
- Comprehensive evaluation metrics
- Easy deployment

---

## 🏗️ Architecture

### Hybrid Detection Pipeline

```
Input Data
    ↓
┌─────────────────────────────────────┐
│   STAGE 1: Hard Rule Filter        │
│                                     │
│   IF callFrequency > 50 AND         │
│      avgCallDistance < 10:          │
│       → LEGITIMATE (100% confident) │
└─────────────────────────────────────┘
    ↓ (if not matched)
┌─────────────────────────────────────┐
│   STAGE 2: Isolation Forest ML     │
│                                     │
│   • Feature engineering             │
│   • Anomaly score calculation       │
│   • Risk type classification        │
└─────────────────────────────────────┘
    ↓
Final Prediction + Confidence
```

### User Profiles Detected

#### Legitimate Users
- **Delivery Partners**: High frequency, low distance (protected by hard rule)
- **Regular Users**: Low frequency, longer calls
- **Business Users**: Moderate activity, professional patterns
- **Traveling Professionals**: Multiple locations, legitimate reasons

#### Fraud Patterns
- **Digital Arrest Bots**: Very high frequency, ultra-short calls, cross-state
- **Traditional Scammers**: Moderate volume, long distance operations
- **Low Volume Scammers**: Targeted attacks, fewer victims

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

```bash
# Clone or navigate to the project directory
cd SentinalX

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1. Generate Training Data

```bash
python data_generator.py
```

This creates:
- `Data/training_dataset.csv` (13,000 samples, 65% legitimate / 35% fraud)
- `Data/test_dataset.csv` (2,600 samples, same distribution)

### 2. Train the Model

```bash
python train_model.py
```

Output:
- `models/isolation_forest.pkl` (trained ML model)
- `models/scaler.pkl` (feature scaler)
- `models/config.json` (configuration & training stats)

### 3. Evaluate Performance

```bash
python evaluate_model.py
```

Generates comprehensive metrics:
- Overall accuracy, precision, recall
- Per-user-type performance
- Delivery partner FPR (should be 0%)
- Confusion matrix visualization
- `evaluation_report.json`

### 4. Make Predictions

**Demo Mode** (pre-configured test cases):
```bash
python predict.py
```

**Interactive Mode** (manual input):
```bash
python predict.py --interactive
```

---

## 📊 Expected Performance

Based on the hybrid architecture, you should achieve:

| Metric | Target | Why It Matters |
|--------|--------|----------------|
| Accuracy | 96-98% | Overall correctness |
| Precision (Fraud) | 95%+ | Low false positives = happy users |
| Recall (Fraud) | 98%+ | Catch all scammers |
| Delivery Partner FPR | **0%** | Hard rule guarantees protection |
| Inference Time | <50ms | Real-time processing |

---

## 🔧 Model Configuration

### Isolation Forest Parameters

```python
model = IsolationForest(
    n_estimators=100,     # Number of trees (speed vs accuracy)
    contamination=0.3,    # Expected fraud rate (30%)
    max_samples=256,      # Samples per tree (speed optimization)
    random_state=42,      # Reproducibility
    n_jobs=-1            # Use all CPU cores
)
```

### Feature Engineering

The model uses 9 key features:

**Base Features:**
1. `avgDuration` - Average call duration (seconds)
2. `callFrequency` - Number of calls in period
3. `uniqueContacts` - Number of unique contacts
4. `avgCallDistance` - Average distance between caller/receiver (km)
5. `circleDiversity` - Number of geographic circles

**Engineered Features:**
6. `call_intensity` = callFrequency / avgDuration
7. `distance_per_call` = avgCallDistance / callFrequency
8. `contact_circle_ratio` = uniqueContacts / (circleDiversity + 1)
9. `high_freq_long_distance` = (callFrequency > 40) & (avgCallDistance > 1000)

---

## 📁 Project Structure

```
SentinalX/
│
├── Data/
│   ├── training_dataset.csv    # Training data (13K samples)
│   └── test_dataset.csv         # Test data (2.6K samples)
│
├── models/                      # Saved models (created after training)
│   ├── isolation_forest.pkl
│   ├── scaler.pkl
│   └── config.json
│
├── data_generator.py            # Synthetic data generation
├── train_model.py               # Model training pipeline
├── evaluate_model.py            # Comprehensive evaluation
├── predict.py                   # Inference & predictions
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🎮 Usage Examples

### Example 1: Check a Delivery Partner

```python
from predict import FraudDetectionAPI

api = FraudDetectionAPI('models')

result = api.predict_single({
    'callFrequency': 65,
    'avgDuration': 8.5,
    'uniqueContacts': 85,
    'avgCallDistance': 6.2,
    'circleDiversity': 1
})

print(result['prediction'])      # → LEGITIMATE
print(result['confidence'])       # → 1.0
print(result['detectionStage'])  # → RULE_BASED
```

### Example 2: Detect Digital Arrest Bot

```python
result = api.predict_single({
    'callFrequency': 89,
    'avgDuration': 5.2,
    'uniqueContacts': 156,
    'avgCallDistance': 2100,
    'circleDiversity': 6
})

print(result['prediction'])      # → FRAUD
print(result['riskType'])        # → HIGH_RISK_CROSS_STATE_OPERATION
print(result['detectionStage'])  # → ML_ISOLATION_FOREST
```

### Example 3: Batch Processing

```python
users = [
    {'callFrequency': 65, 'avgDuration': 8.5, ...},
    {'callFrequency': 18, 'avgDuration': 120, ...},
    {'callFrequency': 89, 'avgDuration': 5.2, ...}
]

results = api.predict_batch(users)

for i, result in enumerate(results):
    print(f"User {i+1}: {result['prediction']}")
```

---

## 🔍 How It Works

### The Hard Rule (Critical!)

```python
if callFrequency > 50 and avgCallDistance < 10:
    return {
        "isAnomaly": False,
        "confidence": 1.0,
        "riskType": "LEGITIMATE_HIGH_FREQUENCY",
        "detectionStage": "RULE_BASED"
    }
```

**Why this works:**
- Delivery partners make **many calls** (>50) in a **small area** (<10km)
- This pattern is nearly impossible for fraud operations
- Guarantees **0% false positives** for this critical user group

### Isolation Forest Detection

For everyone else, the model:
1. Scales features using StandardScaler
2. Calculates anomaly score (more negative = more suspicious)
3. Classifies as fraud if score exceeds threshold
4. Provides confidence and risk type

---

## 📈 Performance Monitoring

### Key Metrics to Track

1. **Overall Accuracy**: Should stay >96%
2. **Precision**: Minimize false positives (target >95%)
3. **Recall**: Catch all fraud (target >98%)
4. **Delivery Partner FPR**: Must be 0%
5. **Inference Time**: Must be <50ms

### Evaluation Report

After running `evaluate_model.py`, check `evaluation_report.json` for:
- Detailed metrics by user type
- Confusion matrix breakdown
- Detection stage statistics
- Inference time analysis

---

## 🚀 Deployment Considerations

### Production Checklist

- [ ] Model trained on sufficient data (13K+ samples)
- [ ] Test accuracy meets targets (>96%)
- [ ] Delivery partner FPR is 0%
- [ ] Inference time <50ms
- [ ] Error handling implemented
- [ ] Monitoring/logging in place
- [ ] Model versioning system
- [ ] Rollback plan ready

### Optimization Tips

1. **Speed**: Use ONNX runtime for faster inference
2. **Scale**: Deploy with FastAPI or Flask
3. **Monitoring**: Track prediction distribution over time
4. **Retraining**: Schedule periodic retraining with new data

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: Model accuracy below 96%
- **Solution**: Regenerate data with more samples
- Check feature engineering calculations

**Issue**: Delivery partner FPR > 0%
- **Solution**: Verify hard rule implementation
- Check if callFrequency/avgCallDistance features are correct

**Issue**: Inference time > 50ms
- **Solution**: Reduce `max_samples` parameter
- Consider ONNX conversion for production

**Issue**: Import errors
- **Solution**: Run `pip install -r requirements.txt`
- Ensure Python 3.8+ is installed

---

## 📚 Technical Details

### Why Isolation Forest?

1. **Fast Training**: O(n log n) complexity
2. **Anomaly Detection**: No need for labeled fraud examples during training
3. **Handles High Dimensions**: Works well with engineered features
4. **Fast Inference**: <50ms per prediction
5. **No Hyperparameter Tuning**: Works well with default settings

### Why Hybrid Approach?

1. **Guarantees**: Hard rule provides absolute protection for delivery partners
2. **Flexibility**: ML handles novel fraud patterns
3. **Speed**: Rule-based stage is instant
4. **Interpretability**: Clear reasoning for both stages

---

## 📝 License

This project is part of the SentinalX fraud detection initiative.

---

## 🤝 Contributing

To improve the model:
1. Add new user profiles to `data_generator.py`
2. Tune Isolation Forest parameters in `train_model.py`
3. Add new engineered features
4. Implement additional detection stages

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review evaluation metrics
3. Verify data quality

---

## 🎉 Success Metrics

Once deployed, track:
- **Fraud catch rate**: % of actual fraud detected
- **User satisfaction**: Complaints about false positives
- **Processing speed**: Average inference time
- **Model drift**: Performance degradation over time

**Target**: 98%+ fraud detection with <1% false positive rate

---

**Built with 💙 for safer telecommunications**
