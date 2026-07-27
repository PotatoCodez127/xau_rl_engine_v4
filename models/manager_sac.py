import torch
import torch.nn as nn
from torch.distributions import Normal

class SACActor(nn.Module):
    """
    The Risk Manager (Actor).
    Observes Oracle probabilities and current execution state to dictate sizing and risk[cite: 1].
    Outputs bounded actions via Tanh to enforce Asymmetric Action Spaces[cite: 1, 3].
    """
    def __init__(self, oracle_dim: int = 3, state_dim: int = 4, action_dim: int = 3, hidden_dim: int = 256):
        super(SACActor, self).__init__()
        
        input_dim = oracle_dim + state_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # SAC requires mean and log_std for continuous action distributions
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, oracle_probs: torch.Tensor, env_state: torch.Tensor):
        x = torch.cat([oracle_probs, env_state], dim=-1)
        x = self.net(x)
        
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        # Clamp log_std to prevent numerical instability
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        return mean, log_std
        
    def sample_action(self, oracle_probs: torch.Tensor, env_state: torch.Tensor):
        """Samples an action and calculates its log probability for Entropy Maximization."""
        mean, log_std = self.forward(oracle_probs, env_state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)   # Enforce bounds [-1, 1]
        
        action = y_t
        # Enforcing bounding requires adjusting the log probability
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob

class SACCritic(nn.Module):
    """
    The Risk Evaluator (Critic).
    Evaluates the Q-value of the state-action pair to guide the Actor.
    """
    def __init__(self, oracle_dim: int = 3, state_dim: int = 4, action_dim: int = 3, hidden_dim: int = 256):
        super(SACCritic, self).__init__()
        
        input_dim = oracle_dim + state_dim + action_dim
        
        # Twin Q-Networks to mitigate overestimation bias
        self.q1_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, oracle_probs: torch.Tensor, env_state: torch.Tensor, action: torch.Tensor):
        x = torch.cat([oracle_probs, env_state, action], dim=-1)
        q1 = self.q1_net(x)
        q2 = self.q2_net(x)
        return q1, q2

if __name__ == "__main__":
    # Smoke Test
    print("Initializing SAC Risk Manager...")
    actor = SACActor()
    critic = SACCritic()
    
    # Dummy Inputs
    dummy_oracle = torch.tensor([[0.1, 0.1, 0.8]]) # Bullish Conviction
    dummy_state = torch.tensor([[0.0, 1.0, 0.0, 0.0]]) # Flat position, full margin
    
    action, log_prob = actor.sample_action(dummy_oracle, dummy_state)
    q1, q2 = critic(dummy_oracle, dummy_state, action)
    
    print(f"Sampled Action (Bounded): {action.detach().numpy()}")
    print(f"Action Log Prob: {log_prob.detach().numpy()}")
    print(f"Estimated Q-Values: Q1={q1.item():.4f}, Q2={q2.item():.4f}")