import torch
import torch.nn as nn
import os

# Import the PyTorch architectures
from models.oracle_transformer import SpatialOracle
from models.manager_sac import SACActor

class ONNXOracleWrapper(nn.Module):
    """
    ONNX strictly prefers tuple/tensor inputs over Python dictionaries. 
    This wrapper flattens the Dict observation space for the ONNX compiler.
    """
    def __init__(self, oracle: SpatialOracle):
        super(ONNXOracleWrapper, self).__init__()
        self.oracle = oracle

    def forward(self, obs_15m, obs_30m, obs_1H, obs_4H):
        obs_dict = {
            "15m": obs_15m,
            "30m": obs_30m,
            "1H": obs_1H,
            "4H": obs_4H
        }
        return self.oracle(obs_dict)

def export_brains_to_onnx(output_dir: str = "deployment/compiled_models"):
    """
    Freezes the PyTorch computational graphs and exports them to ONNX format
    for high-speed, CPU-optimized live inference[cite: 1, 3].
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Initialize untrained models (In production, load your best CPCV weights here)
    num_features = 6
    oracle_pytorch = SpatialOracle(num_features=num_features)
    actor_pytorch = SACActor(oracle_dim=3, state_dim=4, action_dim=3)
    
    oracle_pytorch.eval()
    actor_pytorch.eval()

    # 2. Wrap the Oracle for ONNX
    onnx_oracle = ONNXOracleWrapper(oracle_pytorch)

    # 3. Generate Dummy Tensors matching the Asymmetric MTF Dimensions
    batch_size = 1
    dummy_15m = torch.randn(batch_size, 128, num_features)
    dummy_30m = torch.randn(batch_size, 64, num_features)
    dummy_1H  = torch.randn(batch_size, 32, num_features)
    dummy_4H  = torch.randn(batch_size, 16, num_features)
    
    dummy_oracle_probs = torch.randn(batch_size, 3)
    dummy_env_state = torch.randn(batch_size, 4)

    # 4. Export the Spatial Oracle
    oracle_path = os.path.join(output_dir, "spatial_oracle.onnx")
    print(f"Exporting Spatial Oracle to {oracle_path}...")
    torch.onnx.export(
        onnx_oracle, 
        (dummy_15m, dummy_30m, dummy_1H, dummy_4H), 
        oracle_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["obs_15m", "obs_30m", "obs_1H", "obs_4H"],
        output_names=["directional_probabilities"],
        dynamic_axes={
            "obs_15m": {0: "batch_size"},
            "obs_30m": {0: "batch_size"},
            "obs_1H":  {0: "batch_size"},
            "obs_4H":  {0: "batch_size"},
            "directional_probabilities": {0: "batch_size"}
        }
    )

    # 5. Export the SAC Risk Manager (Actor Only)
    actor_path = os.path.join(output_dir, "sac_manager.onnx")
    print(f"Exporting SAC Manager to {actor_path}...")
    torch.onnx.export(
        actor_pytorch, 
        (dummy_oracle_probs, dummy_env_state), 
        actor_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["oracle_probs", "env_state"],
        output_names=["action_mean", "action_log_std"],
        dynamic_axes={
            "oracle_probs": {0: "batch_size"},
            "env_state": {0: "batch_size"},
            "action_mean": {0: "batch_size"},
            "action_log_std": {0: "batch_size"}
        }
    )
    
    print("ONNX Export Complete. Brains are ready for Edge Deployment.")

if __name__ == "__main__":
    export_brains_to_onnx()