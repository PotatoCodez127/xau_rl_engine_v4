import torch
import torch.nn as nn
from torch.distributions import Normal

class SACActor(nn.Module):
    def __init__(self, oracle_dim: int = 3, state_dim: int = 4, action_dim: int = 3, hidden_dim: int = 256):
        super(SACActor, self).__init__()
        input_dim = oracle_dim + state_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, oracle_probs: torch.Tensor, env_state: torch.Tensor):
        x = torch.cat([oracle_probs, env_state], dim=-1)
        x = self.net(x)
        
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std
        
    def sample_action(self, oracle_probs: torch.Tensor, env_state: torch.Tensor):
        mean, log_std = self.forward(oracle_probs, env_state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        x_t = normal.rsample()  
        y_t = torch.tanh(x_t)   
        
        action = y_t
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob

class SACCritic(nn.Module):
    def __init__(self, oracle_dim: int = 3, state_dim: int = 4, action_dim: int = 3, hidden_dim: int = 256):
        super(SACCritic, self).__init__()
        input_dim = oracle_dim + state_dim + action_dim
        
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

class EntropyTuner(nn.Module):
    """
    Automatically tunes the SAC entropy coefficient to prevent execution paralysis.
    """
    def __init__(self, action_dim: int = 3, lr: float = 1e-4):
        super(EntropyTuner, self).__init__()
        self.target_entropy = -float(action_dim)
        self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    def update(self, log_pi: torch.Tensor):
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.optimizer.zero_grad()
        alpha_loss.backward()
        self.optimizer.step()
        return alpha_loss.item(), self.log_alpha.exp().item()

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