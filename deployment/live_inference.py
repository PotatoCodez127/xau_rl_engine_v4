import os
import joblib
import numpy as np
import onnxruntime as ort

class TriBrainLiveInference:
    """
    Ultra-low latency inference engine running ONNX-compiled models.
    Executes the Spatial Oracle, Context Gatekeeper, and SAC Manager in RAM.
    """
    def __init__(self, models_dir=None):
        if models_dir is None:
            models_dir = os.path.dirname(os.path.abspath(__file__))
            
        oracle_path = os.path.join(models_dir, "oracle.onnx")
        actor_path = os.path.join(models_dir, "actor.onnx")
        gatekeeper_path = os.path.join(models_dir, "gatekeeper.pkl")

        print("⚡ Loading ONNX sessions into CPU RAM...")
        # Initialize ONNX Runtime Sessions (Optimized for single-thread CPU execution)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        self.oracle_session = ort.InferenceSession(oracle_path, opts, providers=['CPUExecutionProvider'])
        self.actor_session = ort.InferenceSession(actor_path, opts, providers=['CPUExecutionProvider'])
        
        # Load Context Gatekeeper (HMM)
        if os.path.exists(gatekeeper_path):
            self.gatekeeper = joblib.load(gatekeeper_path)
            print("✅ Gatekeeper HMM restored.")
        else:
            self.gatekeeper = None
            print("⚠️ Gatekeeper HMM not found! Running un-gated.")

    def predict_action(self, mtf_dict, state_vector):
        """
        Processes a single live step.
        
        Parameters:
            mtf_dict (dict): Keys ['15m', '30m', '1H', '4H'] containing numpy arrays.
            state_vector (list/array): [Position, Margin, Unrealized PnL, Cooldown Timer]
            
        Returns:
            dict: Parsed trade signal {direction_vol, k_tp, k_sl, regime_approved}
        """
        # 1. Gatekeeper Regime Check (HMM)
        regime_approved = True
        if self.gatekeeper is not None:
            # Extract recent 15m volatility/macro features (Index 0, 1)
            macro_features = mtf_dict["15m"][-1, :2].reshape(1, -1)
            regime = self.gatekeeper.predict(macro_features)[0]
            
            # Assuming Regime 2 is High Volatility / Crisis State where trading is blocked
            if regime == 2: 
                regime_approved = False

        # 2. Format Tensors for ONNX (Add Batch Dimension: Shape (1, Sequence, Features))
        inputs_oracle = {
            "15m": np.expand_dims(mtf_dict["15m"], axis=0).astype(np.float32),
            "30m": np.expand_dims(mtf_dict["30m"], axis=0).astype(np.float32),
            "1H": np.expand_dims(mtf_dict["1H"], axis=0).astype(np.float32),
            "4H": np.expand_dims(mtf_dict["4H"], axis=0).astype(np.float32),
            "state": np.expand_dims(state_vector, axis=0).astype(np.float32)
        }

        # 3. Run Spatial Oracle Inference
        oracle_outputs = self.oracle_session.run(None, inputs_oracle)
        oracle_probs = oracle_outputs[0]

        # 4. Run SAC Actor Inference
        inputs_actor = {
            "oracle_probs": oracle_probs,
            "state": inputs_oracle["state"]
        }
        action_outputs = self.actor_session.run(None, inputs_actor)
        raw_action = action_outputs[0][0] # Shape (3,) -> [Direction/Vol, TP_Mult, SL_Mult]

        # 5. Parse Continuously Bounded SAC Output [-1.0, 1.0]
        direction_vol = raw_action[0]
        k_tp = (raw_action[1] + 1.0) / 2.0  # Scale to [0, 1]
        k_sl = (raw_action[2] + 1.0) / 2.0  # Scale to [0, 1]

        # Enforce Gatekeeper veto
        if not regime_approved:
            direction_vol = 0.0 # Force neutral flat position

        return {
            "direction_vol": float(direction_vol), # < -0.1 Short, > 0.1 Long, Else Hold
            "k_tp": float(k_tp),
            "k_sl": float(k_sl),
            "regime_approved": regime_approved
        }

if __name__ == "__main__":
    # Smoke Test: Run dummy data through the live inference engine
    num_features = 11
    dummy_mtf = {
        "15m": np.random.randn(128, num_features),
        "30m": np.random.randn(64, num_features),
        "1H": np.random.randn(32, num_features),
        "4H": np.random.randn(16, num_features),
    }
    dummy_state = np.array([0.0, 1.0, 0.0, 0.0]) # Flat position

    engine = TriBrainLiveInference()
    signal = engine.predict_action(dummy_mtf, dummy_state)
    print("\n🎯 LIVE SIGNAL OUTPUT:")
    print(signal)