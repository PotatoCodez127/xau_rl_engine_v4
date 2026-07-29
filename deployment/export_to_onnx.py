import os
import sys
import torch
import joblib
from models.oracle_transformer import SpatialOracle
from models.manager_sac import SACActor

class OracleONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, m15, m30, h1, h4, state):
        obs_dict = {"15m": m15, "30m": m30, "1H": h1, "4H": h4, "state": state}
        return self.model(obs_dict)

class ActorONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, oracle_probs, state):
        # Extract only the deterministic mean and apply the tanh bounds physically to the graph
        mean, _ = self.model(oracle_probs, state)
        return torch.tanh(mean)

def export_brains_to_onnx(data_dir="data", checkpoint_dir="checkpoints", output_dir="deployment"):
    """Compiles trained PyTorch weights into ONNX graphs."""
    os.makedirs(output_dir, exist_ok=True)
    master_tensor_path = os.path.join(data_dir, "master_training_tensor.pkl")
    checkpoint_path = os.path.join(checkpoint_dir, "tribrain_checkpoint.pth")

    if not os.path.exists(master_tensor_path):
        raise FileNotFoundError(f"Missing dataset at {master_tensor_path}.")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint at {checkpoint_path}.")

    mtf_dict = joblib.load(master_tensor_path)
    num_features = mtf_dict["15m"].shape[1]
    
    device = torch.device("cpu")
    oracle = SpatialOracle(num_features=num_features).to(device)
    actor = SACActor().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    oracle.load_state_dict(checkpoint["oracle_state"])
    actor.load_state_dict(checkpoint["actor_state"])

    oracle.eval()
    actor.eval()

    oracle_wrapped = OracleONNXWrapper(oracle)
    actor_wrapped = ActorONNXWrapper(actor)

    dummy_15m = torch.randn(1, 128, num_features)
    dummy_30m = torch.randn(1, 64, num_features)
    dummy_1h = torch.randn(1, 32, num_features)
    dummy_4h = torch.randn(1, 16, num_features)
    dummy_state = torch.randn(1, 4)

    torch.onnx.export(
        oracle_wrapped,
        (dummy_15m, dummy_30m, dummy_1h, dummy_4h, dummy_state),
        os.path.join(output_dir, "oracle.onnx"),
        input_names=["15m", "30m", "1H", "4H", "state"],
        output_names=["oracle_probs"],
        opset_version=14
    )

    with torch.no_grad():
        dummy_probs = oracle_wrapped(dummy_15m, dummy_30m, dummy_1h, dummy_4h, dummy_state)

    torch.onnx.export(
        actor_wrapped,
        (dummy_probs, dummy_state),
        os.path.join(output_dir, "actor.onnx"),
        input_names=["oracle_probs", "state"],
        output_names=["action_output"],
        opset_version=14
    )
    print("ONNX Export Complete. Brains are ready for Edge Deployment.")

if __name__ == "__main__":
    export_brains_to_onnx()