import numpy as np
from typing import List, Dict
import logging
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnsembleDetector:
    """
    Ensemble model combining multiple fraud detectors.
    """

    # ... your __init__ and _get_model_proba stay the same ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get ensemble probability by combining base model probabilities.
        Supports weighted voting.
        """
        all_probas = []

        for i, model in enumerate(self.base_models):
            proba = self._get_model_proba(model, X)
            all_probas.append(proba * self.weights[i])

        # Combine probabilities (weighted sum)
        ensemble_proba = np.sum(all_probas, axis=0) / np.sum(self.weights)
        return ensemble_proba

    
    def __init__(self, base_models: List, method: str = 'weighted_voting', weights: List[float] = None):
        self.base_models = base_models
        self.method = method
        self.weights = weights or [1.0 / len(base_models)] * len(base_models)

    def _get_model_proba(self, model, X):
    # Case 1: Normal classifiers (XGB, LGBM, DL)
     if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if isinstance(probs, np.ndarray) and probs.ndim == 2:
            return probs[:, 1]
        return np.asarray(probs).astype(float)

    # Case 2: PyOD anomaly models
     if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.asarray(scores).astype(float)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    # Case 3: Fallback (labels)
     preds = model.predict(X)
     return np.asarray(preds).astype(float)

    
    def _get_model_proba(self, model, X):
        # Case 1: Normal classifiers (XGB, LGBM, DL)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            if isinstance(probs, np.ndarray) and probs.ndim == 2:
                return probs[:, 1]
            return np.asarray(probs).astype(float)
    
        # Case 2: PyOD anomaly models
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            scores = np.asarray(scores).astype(float)
            return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    
        # Case 3: Fallback (labels)
        preds = model.predict(X)
        return np.asarray(preds).astype(float)
    

    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba > threshold).astype(int)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        proba = self.predict_proba(X_test)
        pred = self.predict(X_test)
        
        metrics = {
            'roc_auc': roc_auc_score(y_test, proba),
            'f1': f1_score(y_test, pred),
            'precision': precision_score(y_test, pred),
            'recall': recall_score(y_test, pred),
        }
        
        logger.info(f"Ensemble Metrics: {metrics}")
        return metrics