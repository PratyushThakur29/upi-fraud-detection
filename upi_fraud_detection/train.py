"""
Complete Training Pipeline for UPI Fraud Detection
Trains XGBoost, Deep Learning, and Ensemble models
Saves models and generates performance reports
"""

import sys
from pathlib import Path
import logging
import json
import pickle
import numpy as np
from datetime import datetime

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Import modules
from data_loader import DataLoader
from xgboost_model import XGBoostFraudDetector 
from deep_learning_model import DeepLearningDetector
from ensemble_model import EnsembleDetector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_metrics(metrics, filepath):
    """Save metrics to text file"""
    with open(filepath, 'w') as f:
        f.write("="*60 + "\n")
        f.write("UPI FRAUD DETECTION - MODEL PERFORMANCE METRICS\n")
        f.write("="*60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for model_name, model_metrics in metrics.items():
            f.write(f"\n{model_name.upper()} MODEL:\n")
            f.write("-"*40 + "\n")
            for metric, value in model_metrics.items():
                f.write(f"{metric:15s}: {value:.4f}\n")
        
        f.write("\n" + "="*60 + "\n")

def main():
    """Main training pipeline"""
    
    print("\n" + "="*60)
    print("🚀 UPI FRAUD DETECTION - TRAINING PIPELINE")
    print("="*60 + "\n")
    
    # Create directories
    models_dir = PROJECT_ROOT / 'models'
    results_dir = PROJECT_ROOT / 'results'
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    # 1. Load Data
    logger.info("📂 Loading UPI fraud dataset...")
    data_file = PROJECT_ROOT / 'data' / 'upi_fraud_dataset.csv'
    
    if not data_file.exists():
        logger.error(f"❌ Dataset not found: {data_file}")
        logger.error("Please ensure upi_fraud_dataset.csv is in the data/ directory")
        return
    
    loader = DataLoader(str(data_file))
    loader.load_data()
    
    # 2. Preprocess Data
    logger.info("🔧 Preprocessing data (applying SMOTE for class balancing)...")
    X_train, X_test, y_train, y_test = loader.preprocess(
        test_size=0.2, 
        apply_smote=True
    )
    
    logger.info(f"✅ Data ready - Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    logger.info(f"   Fraud rate in training: {y_train.mean():.2%}")
    logger.info(f"   Fraud rate in test: {y_test.mean():.2%}")
    
    # Store all metrics
    all_metrics = {}
    
    # 3. Train XGBoost Model
    print("\n" + "-"*60)
    logger.info("🌳 Training XGBoost Model...")
    print("-"*60)
    
    xgb_model = XGBoostFraudDetector()
    xgb_model.train(X_train, y_train, X_test, y_test)
    
    # Evaluate XGBoost
    xgb_metrics = xgb_model.evaluate(X_test, y_test)
    all_metrics['XGBoost'] = xgb_metrics
    
    logger.info("📊 XGBoost Metrics:")
    for metric, value in xgb_metrics.items():
        logger.info(f"   {metric}: {value:.4f}")
    
    # Save XGBoost model
    xgb_path = models_dir / 'xgboost_model.json'
    xgb_model.save(str(xgb_path))
    logger.info(f"💾 XGBoost model saved: {xgb_path}")
    
    # 4. Train Deep Learning Model
    print("\n" + "-"*60)
    logger.info("🧠 Training Deep Learning Model...")
    print("-"*60)
    
    input_dim = X_train.shape[1]
    dl_model = DeepLearningDetector(input_dim=input_dim)
    dl_model.train(X_train, y_train, X_test, y_test, epochs=50, batch_size=32)
    
    # Evaluate Deep Learning
    dl_metrics = dl_model.evaluate(X_test, y_test)
    all_metrics['Deep Learning'] = {
        'auc_roc': dl_metrics['auc'],
        'precision': dl_metrics['precision'],
        'recall': dl_metrics['recall'],
        'f1': 2 * (dl_metrics['precision'] * dl_metrics['recall']) / 
              (dl_metrics['precision'] + dl_metrics['recall'])
    }
    
    logger.info("📊 Deep Learning Metrics:")
    for metric, value in all_metrics['Deep Learning'].items():
        logger.info(f"   {metric}: {value:.4f}")
    
    # Save Deep Learning model
    dl_path = models_dir / 'deep_learning_model.h5'
    dl_model.save_model(str(dl_path))
    logger.info(f"💾 Deep Learning model saved: {dl_path}")
    
    # 5. Create Ensemble Model
    print("\n" + "-"*60)
    logger.info("🔗 Creating Ensemble Model (Weighted Voting)...")
    print("-"*60)
    
    ensemble_model = EnsembleDetector(
        base_models=[xgb_model, dl_model],
        method='weighted_voting',
        weights=[0.6, 0.4]  # 60% XGBoost, 40% Deep Learning
    )
    
    # Evaluate Ensemble
    ensemble_metrics = ensemble_model.evaluate(X_test, y_test)
    all_metrics['Ensemble'] = {
        'auc_roc': ensemble_metrics['roc_auc'],
        'f1': ensemble_metrics['f1'],
        'precision': ensemble_metrics['precision'],
        'recall': ensemble_metrics['recall']
    }
    
    logger.info("📊 Ensemble Metrics:")
    for metric, value in all_metrics['Ensemble'].items():
        logger.info(f"   {metric}: {value:.4f}")
    
    # Save Ensemble model
    ensemble_path = models_dir / 'ensemble_model.pkl'
    with open(ensemble_path, 'wb') as f:
        pickle.dump(ensemble_model, f)
    logger.info(f"💾 Ensemble model saved: {ensemble_path}")
    
    # 6. Save Results
    print("\n" + "-"*60)
    logger.info("📝 Saving results...")
    print("-"*60)
    
    metrics_file = results_dir / 'training_metrics.txt'
    save_metrics(all_metrics, metrics_file)
    logger.info(f"✅ Metrics saved: {metrics_file}")
    
    # Save scaler for inference
    scaler_path = models_dir / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(loader.scaler, f)
    logger.info(f"💾 Scaler saved: {scaler_path}")
    
    # 7. Final Summary
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print("\n📊 FINAL ENSEMBLE RESULTS:")
    print("-"*60)
    print(f"ROC-AUC:   {all_metrics['Ensemble']['auc_roc']:.4f}")
    print(f"Precision: {all_metrics['Ensemble']['precision']:.4f}")
    print(f"Recall:    {all_metrics['Ensemble']['recall']:.4f}")
    print(f"F1-Score:  {all_metrics['Ensemble']['f1']:.4f}")
    print("-"*60)
    
    print("\n📁 Models saved in: models/")
    print("   - xgboost_model.json")
    print("   - deep_learning_model.h5")
    print("   - ensemble_model.pkl")
    print("   - scaler.pkl")
    
    print("\n📈 Results saved in: results/")
    print("   - training_metrics.txt")
    
    print("\n🎉 You can now use predict.py for inference!")
    print("="*60 + "\n")

if __name__ == '__main__': 
    main()
