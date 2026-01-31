"""
Inference Script for UPI Fraud Detection
Load trained models and make predictions on new transactions
"""

import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import logging

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from xgboost_model import XGBoostFraudDetector
from deep_learning_model import DeepLearningDetector
from ensemble_model import EnsembleDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudPredictor:
    """
    Production-ready fraud predictor
    Loads trained models and makes predictions on new UPI transactions
    """
    
    def __init__(self, model_type='ensemble'):
        """
        Initialize predictor with trained models
        
        Args:
            model_type (str): 'ensemble', 'xgboost', or 'deep_learning'
        """
        self.model_type = model_type
        self.models_dir = PROJECT_ROOT / 'models'
        
        # Load scaler
        scaler_path = self.models_dir / 'scaler.pkl'
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler not found at {scaler_path}. "
                "Please run train.py first to train models."
            )
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        logger.info(f"✅ Scaler loaded from {scaler_path}")
        
        # Load model based on type
        if model_type == 'ensemble':
            self.model = self._load_ensemble()
        elif model_type == 'xgboost':
            self.model = self._load_xgboost()
        elif model_type == 'deep_learning':
            self.model = self._load_deep_learning()
        else:
            raise ValueError(f"Invalid model_type: {model_type}")
        
        logger.info(f"✅ {model_type.upper()} model loaded successfully")
    
    def _load_xgboost(self):
        """Load XGBoost model"""
        model_path = self.models_dir / 'xgboost_model.json'
        if not model_path.exists():
            raise FileNotFoundError(f"XGBoost model not found at {model_path}")
        
        model = XGBoostFraudDetector()
        model.load(str(model_path))
        return model
    
    def _load_deep_learning(self):
        """Load Deep Learning model"""
        model_path = self.models_dir / 'deep_learning_model.h5'
        if not model_path.exists():
            raise FileNotFoundError(f"Deep Learning model not found at {model_path}")
        
        # Determine input dimension from saved model
        model = DeepLearningDetector(input_dim=9)  # Default, will be overwritten
        model.load_model(str(model_path))
        return model
    
    def _load_ensemble(self):
        """Load Ensemble model"""
        model_path = self.models_dir / 'ensemble_model.pkl'
        if not model_path.exists():
            raise FileNotFoundError(f"Ensemble model not found at {model_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def _preprocess(self, data, return_dataframe=False):
        """
        Preprocess input data
        
        Args:
            data (dict or pd.DataFrame): Transaction data
            return_dataframe (bool): If True, return DataFrame with feature names (for XGBoost)
        
        Returns:
            pd.DataFrame or np.ndarray: Preprocessed features
        """
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Expected feature order (must match training)
        feature_cols = ['trans_hour', 'trans_day', 'trans_month', 'trans_year', 
                       'category', 'age', 'trans_amount', 'state', 'zip']
        
        # Handle missing columns
        for col in feature_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required feature: {col}")
        
        # Select and order features
        X = data[feature_cols].copy()
        
        # Ensure all values are numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                # Try to convert string to numeric
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except:
                    raise ValueError(f"Column '{col}' contains non-numeric values that cannot be converted")
        
        # Handle missing values
        for col in X.columns:
            if X[col].isnull().any():
                X[col].fillna(X[col].median() if X[col].dtype in ['float64', 'int64'] else 0, inplace=True)
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=feature_cols,
            index=X.index
        )
        
        # Return DataFrame (with feature names) or numpy array
        if return_dataframe:
            return X_scaled
        return X_scaled.values
    
    def predict_transaction(self, transaction_data):
        """
        Predict fraud for a single transaction
        
        Args:
            transaction_data (dict): Transaction features
            
        Returns:
            dict: Prediction results with probability and risk level
        
        Example:
            >>> predictor = FraudPredictor()
            >>> transaction = {
            ...     'trans_hour': 23,
            ...     'trans_day': 15,
            ...     'trans_month': 12,
            ...     'trans_year': 2024,
            ...     'category': 5411,
            ...     'age': 35,
            ...     'trans_amount': 50000,
            ...     'state': 22,  # Numeric state code
            ...     'zip': 400001
            ... }
            >>> result = predictor.predict_transaction(transaction)
            >>> print(result)
        """
        # Preprocess - return DataFrame to preserve feature names for XGBoost
        X = self._preprocess(transaction_data, return_dataframe=True)
        
        # Get predictions
        fraud_prob = self.model.predict_proba(X)[0]
        is_fraud = self.model.predict(X)[0]
        
        # Determine risk level
        if fraud_prob > 0.7:
            risk_level = "HIGH"
        elif fraud_prob > 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        result = {
            "is_fraud": bool(is_fraud),
            "fraud_probability": float(fraud_prob),
            "risk_level": risk_level,
            "model_used": self.model_type
        }
        
        return result
    
    def predict_batch(self, data):
        """
        Predict fraud for multiple transactions
        
        Args:
            data (pd.DataFrame): DataFrame with transaction features
            
        Returns:
            pd.DataFrame: Original data with predictions added
        """
        logger.info(f"Processing {len(data)} transactions...")
        
        # Preprocess - return DataFrame to preserve feature names for XGBoost
        X = self._preprocess(data, return_dataframe=True)
        
        # Get predictions
        probabilities = self.model.predict_proba(X)
        predictions = self.model.predict(X)
        
        # Add to dataframe
        results = data.copy()
        results['fraud_probability'] = probabilities
        results['is_fraud'] = predictions
        results['risk_level'] = pd.cut(
            probabilities, 
            bins=[0, 0.4, 0.7, 1.0],
            labels=['LOW', 'MEDIUM', 'HIGH']
        )
        
        logger.info(f"✅ Predictions complete")
        logger.info(f"   Flagged as fraud: {predictions.sum()} ({predictions.mean():.2%})")
        
        return results


def main():
    """Demo usage of the predictor"""
    
    print("\n" + "="*60)
    print("🔒 UPI FRAUD DETECTION - INFERENCE DEMO")
    print("="*60 + "\n")
    
    # Initialize predictor
    print("📥 Loading trained ensemble model...")
    predictor = FraudPredictor(model_type='ensemble')
    
    # Example transaction (high risk)
    print("\n" + "-"*60)
    print("🔍 Testing Transaction #1 (Suspicious)")
    print("-"*60)
    
    suspicious_txn = {
        'trans_hour': 3,           # Late night transaction
        'trans_day': 15,
        'trans_month': 12,
        'trans_year': 2024,
        'category': 5411,
        'age': 25,
        'trans_amount': 75000,     # Large amount
        'state': 22,               # State code (numeric)
        'zip': 400001
    }
    
    result1 = predictor.predict_transaction(suspicious_txn)
    
    print(f"\nTransaction Details:")
    print(f"  Amount: ₹{suspicious_txn['trans_amount']:,}")
    print(f"  Time: {suspicious_txn['trans_hour']}:00")
    print(f"  Category: {suspicious_txn['category']}")
    print(f"  State Code: {suspicious_txn['state']}")
    
    print(f"\n🎯 Prediction Results:")
    print(f"  Fraud: {'YES ⚠️' if result1['is_fraud'] else 'NO ✅'}")
    print(f"  Probability: {result1['fraud_probability']:.2%}")
    print(f"  Risk Level: {result1['risk_level']}")
    
    # Example transaction (legitimate)
    print("\n" + "-"*60)
    print("🔍 Testing Transaction #2 (Legitimate)")
    print("-"*60)
    
    legitimate_txn = {
        'trans_hour': 14,          # Afternoon
        'trans_day': 10,
        'trans_month': 6,
        'trans_year': 2024,
        'category': 5812,
        'age': 45,
        'trans_amount': 500,       # Small amount
        'state': 29,               # State code (numeric)
        'zip': 560001
    }
    
    result2 = predictor.predict_transaction(legitimate_txn)
    
    print(f"\nTransaction Details:")
    print(f"  Amount: ₹{legitimate_txn['trans_amount']:,}")
    print(f"  Time: {legitimate_txn['trans_hour']}:00")
    print(f"  Category: {legitimate_txn['category']}")
    print(f"  State Code: {legitimate_txn['state']}")
    
    print(f"\n🎯 Prediction Results:")
    print(f"  Fraud: {'YES ⚠️' if result2['is_fraud'] else 'NO ✅'}")
    print(f"  Probability: {result2['fraud_probability']:.2%}")
    print(f"  Risk Level: {result2['risk_level']}")
    
    print("\n" + "="*60)
    print("✅ Demo complete!")
    print("="*60 + "\n")
    
    print("💡 Usage Examples:")
    print("\n1. Single transaction prediction:")
    print("   from predict import FraudPredictor")
    print("   predictor = FraudPredictor()")
    print("   result = predictor.predict_transaction(transaction_dict)")
    print("\n2. Batch predictions:")
    print("   import pandas as pd")
    print("   data = pd.read_csv('transactions.csv')")
    print("   results = predictor.predict_batch(data)")
    print("\n")


if __name__ == "__main__":
    main()