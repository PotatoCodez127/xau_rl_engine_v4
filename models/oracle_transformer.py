import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Injects chronological "barcodes" into the sequence so the Transformer
    understands the temporal order of the price action[cite: 1].
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape: (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

class TimeframeExpert(nn.Module):
    """
    A specialized Transformer sub-network for a specific timeframe.
    Leverages Self-Attention to decipher wick-to-body zone interactions[cite: 1, 3].
    """
    def __init__(self, input_dim: int, d_model: int = 32, n_heads: int = 4, num_layers: int = 2):
        super(TimeframeExpert, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        # Extract the final chronological step's hidden state
        return x[:, -1, :]

class SpatialOracle(nn.Module):
    """
    The Master Brain. Routes the asymmetric MTF tensors into Timeframe Experts 
    and outputs the directional probability matrix via Softmax[cite: 1, 3].
    """
    def __init__(self, num_features: int = 6, expert_dim: int = 32):
        super(SpatialOracle, self).__init__()
        
        # Mixture of Experts: One Transformer per timeframe
        self.expert_15m = TimeframeExpert(input_dim=num_features, d_model=expert_dim)
        self.expert_30m = TimeframeExpert(input_dim=num_features, d_model=expert_dim)
        self.expert_1H  = TimeframeExpert(input_dim=num_features, d_model=expert_dim)
        self.expert_4H  = TimeframeExpert(input_dim=num_features, d_model=expert_dim)
        
        # Confluence Layer: Fuses the macro and micro experts
        fused_dim = expert_dim * 4
        self.fc_layers = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3) # 3 Outputs: P(Bearish), P(Ranging), P(Bullish)
        )
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, obs_dict: dict) -> torch.Tensor:
        """
        Expects a dictionary of tensors matching the Gymnasium Dict space.
        """
        out_15m = self.expert_15m(obs_dict["15m"])
        out_30m = self.expert_30m(obs_dict["30m"])
        out_1H  = self.expert_1H(obs_dict["1H"])
        out_4H  = self.expert_4H(obs_dict["4H"])
        
        # Concatenate expert contexts
        fused_context = torch.cat([out_15m, out_30m, out_1H, out_4H], dim=1)
        
        logits = self.fc_layers(fused_context)
        probabilities = self.softmax(logits)
        
        return probabilities

if __name__ == "__main__":
    # Structural Matrix Test
    print("Initializing Spatial Oracle...")
    oracle = SpatialOracle(num_features=6)
    
    # Simulate a single batch environment observation
    dummy_obs = {
        "15m": torch.randn(1, 128, 6),
        "30m": torch.randn(1, 64, 6),
        "1H":  torch.randn(1, 32, 6),
        "4H":  torch.randn(1, 16, 6)
    }
    
    probs = oracle(dummy_obs)
    print(f"Output Matrix Shape: {probs.shape}")
    print(f"Directional Probabilities [Bearish, Ranging, Bullish]: {probs.detach().numpy()}")