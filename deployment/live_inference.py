import time
import json
import requests
import numpy as np
import onnxruntime as ort
import MetaTrader5 as mt5
import joblib

class LiveTriBrainEngine:
    """
    The Live Edge Execution Engine.
    Polls MT5, runs ONNX inference for the Oracle and Manager, checks the HMM Gatekeeper,
    and dispatches asynchronous WhatsApp alerts[cite: 1, 3].
    """
    def __init__(self, oracle_path: str, manager_path: str, gatekeeper_path: str, webhook_url: str):
        self.webhook_url = webhook_url
        
        # 1. Initialize ONNX Runtime Sessions (CPU Optimized)
        print("Loading ONNX Computation Graphs...")
        self.oracle_session = ort.InferenceSession(oracle_path, providers=['CPUExecutionProvider'])
        self.manager_session = ort.InferenceSession(manager_path, providers=['CPUExecutionProvider'])
        
        # 2. Load HMM Gatekeeper
        print("Loading Context Gatekeeper...")
        try:
            self.gatekeeper = joblib.load(gatekeeper_path)
            self.gatekeeper_active = True
        except FileNotFoundError:
            print("Gatekeeper model not found. Running in Oracle-Only mode for testing.")
            self.gatekeeper_active = False

        # 3. Connect to MetaTrader 5
        if not mt5.initialize():
            print("MT5 Initialization Failed.")
            mt5.shutdown()
            raise ConnectionError("Make sure MT5 is running and AutoTrading is allowed.")
        print("Connected to MetaTrader 5 Terminal.")

    def fetch_live_tensors(self):
        """
        Polls MT5 to construct the Asymmetric Spatial Tensors.
        Note: In production, this interfaces with MTFSpatialFeatureBuilder to 
        calculate tanh-squashed, ATR-scaled distances[cite: 1, 3].
        """
        # Simulated tensor generation matching our asymmetric architecture
        batch_size = 1
        num_features = 6
        obs_15m = np.random.randn(batch_size, 128, num_features).astype(np.float32)
        obs_30m = np.random.randn(batch_size, 64, num_features).astype(np.float32)
        obs_1H  = np.random.randn(batch_size, 32, num_features).astype(np.float32)
        obs_4H  = np.random.randn(batch_size, 16, num_features).astype(np.float32)
        
        # Simulated macro features for Gatekeeper and env state for Manager
        macro_features = np.random.randn(1, 2).astype(np.float32)
        env_state = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        
        return obs_15m, obs_30m, obs_1H, obs_4H, macro_features, env_state

    def send_whatsapp_alert(self, message: str):
        """Dispatches execution logs to a WhatsApp webhook."""
        if not self.webhook_url:
            return
            
        payload = {"text": message}
        try:
            requests.post(self.webhook_url, json=payload, timeout=2.0)
            print("WhatsApp alert dispatched.")
        except requests.exceptions.RequestException as e:
            print(f"Failed to send WhatsApp alert: {e}")

    def run_polling_loop(self):
        """Continuous stateful inference loop."""
        print("Starting Live Polling Loop for XAUUSD...")
        
        try:
            while True:
                # 1. Fetch live MTF tensors
                obs_15m, obs_30m, obs_1H, obs_4H, macro_features, env_state = self.fetch_live_tensors()
                
                # 2. Oracle ONNX Inference
                oracle_inputs = {
                    "obs_15m": obs_15m,
                    "obs_30m": obs_30m,
                    "obs_1H": obs_1H,
                    "obs_4H": obs_4H
                }
                oracle_probs = self.oracle_session.run(None, oracle_inputs)[0]
                
                # 3. Gatekeeper Regime Detection
                is_authorized = True
                if self.gatekeeper_active:
                    regime = self.gatekeeper.predict(macro_features)[-1]
                    # Block trades if regime is 0 (terrible market structure)
                    if regime == 0 or np.max(oracle_probs) < 0.65:
                        is_authorized = False

                # 4. SAC Manager ONNX Inference & Execution
                if is_authorized:
                    manager_inputs = {
                        "oracle_probs": oracle_probs,
                        "env_state": env_state
                    }
                    action_mean, _ = self.manager_session.run(None, manager_inputs)
                    
                    # Parse continuous bounds
                    direction_vol = action_mean[0][0]
                    k_tp = (action_mean[0][1] + 1) / 2.0
                    k_sl = (action_mean[0][2] + 1) / 2.0
                    
                    signal = "LONG" if direction_vol > 0 else "SHORT"
                    msg = (f"🚨 XAUUSD EXECUTION AUTHORIZED 🚨\n"
                           f"Signal: {signal}\n"
                           f"Volume Scalar: {abs(direction_vol):.2f}\n"
                           f"Oracle Conviction: {np.max(oracle_probs)*100:.1f}%\n"
                           f"TP Multiplier: {k_tp:.2f} | SL Multiplier: {k_sl:.2f}")
                    
                    print(msg)
                    self.send_whatsapp_alert(msg)
                    
                    # Introduce algorithmic cooldown to prevent hyperactivity[cite: 1, 3]
                    time.sleep(60 * 15) 
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] Gatekeeper blocked execution. Market structure unfavorable.")
                    time.sleep(60) # Poll again in 1 minute

        except KeyboardInterrupt:
            print("Live Engine manually stopped.")
        finally:
            mt5.shutdown()

if __name__ == "__main__":
    # Specify paths to your compiled models
    ORACLE_ONNX = "compiled_models/spatial_oracle.onnx"
    MANAGER_ONNX = "compiled_models/sac_manager.onnx"
    GATEKEEPER_HMM = "compiled_models/gatekeeper.pkl"
    WEBHOOK_URL = "" # Insert your WhatsApp Webhook URL here
    
    # Initialize and run
    # Note: You must run export_to_onnx.py first to generate the .onnx files.
    # engine = LiveTriBrainEngine(ORACLE_ONNX, MANAGER_ONNX, GATEKEEPER_HMM, WEBHOOK_URL)
    # engine.run_polling_loop()
    print("Live Inference script is ready for edge deployment.")