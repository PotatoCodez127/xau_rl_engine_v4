import numpy as np
from hmmlearn import hmm
import joblib
import warnings

class ContextGatekeeper:
    """
    The Context Gatekeeper uses a Hidden Markov Model (HMM) for Regime Detection[cite: 3].
    It categorizes the macro-state of the market and physically blocks the SAC Manager
    from trading during "terrible" structure to prevent chop exploitation.
    """
    
    def __init__(self, n_components: int = 3, random_state: int = 42):
        # Default Regimes: e.g., 0 = Ranging/Chop, 1 = Trending, 2 = High Volatility
        self.n_components = n_components
        self.model = hmm.GaussianHMM(
            n_components=n_components, 
            covariance_type="full", 
            n_iter=100, 
            random_state=random_state
        )
        self.is_fitted = False

    def fit(self, features: np.ndarray):
        """
        Trains the HMM unsupervised on historical macro features 
        (e.g., 4H ATR, Fractional Diff variance).
        Expects shape: (n_samples, n_features).
        """
        # Suppress hmmlearn convergence warnings for clean console logs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(features)
        
        self.is_fitted = True

    def predict_regime(self, current_features: np.ndarray) -> int:
        """
        Predicts the current market regime based on the latest feature vector.
        """
        if not self.is_fitted:
            raise ValueError("HMM Gatekeeper must be fitted before predicting.")
            
        # hmmlearn expects 2D array
        if current_features.ndim == 1:
            current_features = current_features.reshape(1, -1)
            
        return self.model.predict(current_features)[-1]

    def authorize_execution(self, regime: int, oracle_probs: np.ndarray, conviction_threshold: float = 0.65) -> bool:
        """
        The physical macro confluence filter[cite: 3]. 
        Blocks trades during untradable noise or if the Oracle lacks conviction.
        
        oracle_probs expected format: [P(Bearish), P(Ranging), P(Bullish)]
        """
        max_conviction = np.max(oracle_probs)
        
        # 1. Block trades if the Oracle is unsure
        if max_conviction < conviction_threshold:
            return False
            
        # 2. Block trades in terrible market structure. 
        # (Assuming mapping: Regime 0 = Low Volatility Chop)
        if regime == 0:
            return False
            
        # If structure is valid and Oracle conviction is high, authorize the SAC Manager
        return True
        
    def save_model(self, filepath: str):
        if self.is_fitted:
            joblib.dump(self.model, filepath)
        
    def load_model(self, filepath: str):
        self.model = joblib.load(filepath)
        self.is_fitted = True

if __name__ == "__main__":
    # Smoke Test
    print("Initializing Context Gatekeeper HMM...")
    gatekeeper = ContextGatekeeper(n_components=3)
    
    # Generate dummy historical volatility and momentum features
    # Shape: (1000 samples, 2 features: e.g., ATR and Momentum)
    dummy_history = np.random.randn(1000, 2) 
    
    print("Fitting HMM to historical market regimes...")
    gatekeeper.fit(dummy_history)
    
    # Simulate live tick
    live_tick_features = np.array([0.5, 1.2]) 
    oracle_probabilities = np.array([0.1, 0.1, 0.8]) # Strong Bullish conviction
    
    current_regime = gatekeeper.predict_regime(live_tick_features)
    is_authorized = gatekeeper.authorize_execution(current_regime, oracle_probabilities)
    
    print(f"Detected Regime: {current_regime}")
    print(f"Trade Authorized by Gatekeeper: {is_authorized}")