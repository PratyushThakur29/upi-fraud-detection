# 🚀 Quick Start Guide

Get started with UPI Fraud Detection in 5 minutes!

## ⚡ Fast Setup

### 1. Install Dependencies (2 minutes)
```bash
pip install -r requirements.txt
```

### 2. Train Models (5-10 minutes)
```bash
python train.py
```

Expected output:
```
🚀 UPI FRAUD DETECTION - TRAINING PIPELINE
📂 Loading UPI fraud dataset...
🔧 Preprocessing data...
🌳 Training XGBoost Model...
🧠 Training Deep Learning Model...
🔗 Creating Ensemble Model...

✅ TRAINING COMPLETE!
📊 FINAL ENSEMBLE RESULTS:
ROC-AUC:   0.9612
Precision: 0.8523
Recall:    0.8876
F1-Score:  0.8696
```

### 3. Make Predictions (instant)
```bash
python predict.py
```

## 📖 Full Documentation

- **README.md** - Complete project documentation
- **GITHUB_SETUP.md** - How to upload to GitHub
- **CONTRIBUTING.md** - How to contribute

## 🎯 Common Use Cases

### Predict Single Transaction
```python
from predict import FraudPredictor

predictor = FraudPredictor()

transaction = {
    'trans_hour': 23,
    'trans_day': 15,
    'trans_month': 12,
    'trans_year': 2024,
    'category': 5411,
    'age': 35,
    'trans_amount': 50000,
    'state': 22,  # Numeric state code
    'zip': 400001
}

result = predictor.predict_transaction(transaction)
print(result)
```

### Predict Batch
```python
import pandas as pd
from predict import FraudPredictor

predictor = FraudPredictor()
data = pd.read_csv('new_transactions.csv')
results = predictor.predict_batch(data)
results.to_csv('predictions.csv')
```

## 📂 Project Structure
```
upi_fraud_detection/
├── train.py          ← Train models
├── predict.py        ← Make predictions
├── src/              ← Source code
├── data/             ← Dataset
├── models/           ← Saved models (after training)
└── results/          ← Training results
```

## ❓ Need Help?

1. Check **README.md** for detailed docs
2. Check **GITHUB_SETUP.md** for GitHub upload
3. Run with `-h` flag: `python train.py -h`

## 🎉 You're Ready!

Start with:
```bash
python train.py
```

Happy fraud detecting! 🔒
