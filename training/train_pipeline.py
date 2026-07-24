import torch
import torch.optim as optim
import numpy as np

# Import our custom environment and brains
from envs.xau_mtf_env import XAUMTFEnv
from models.oracle_transformer import SpatialOracle
from models.gatekeeper_hmm import ContextGatekeeper
from models.manager_sac import SACActor, SACCritic

class TriBrainTrainer:
    """
    Synchronizes the Tri-Brain architecture during training[cite: 3].
    Orchestrates the flow of data from the Environment -> Oracle -> Gatekeeper -> Manager.
    """
    def __init__(self, env: XAUMTFEnv):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on device: {self.device}")

        # Initialize Brains
        self.oracle = SpatialOracle(num_features=6).to(self.device)
        self.gatekeeper = ContextGatekeeper(n_components=3)
        self.actor = SACActor().to(self.device)
        self.critic = SACCritic().to(self.device)

        # Optimizers (Oracle and SAC train jointly on the reward signal)
        self.oracle_optim = optim.Adam(self.oracle.parameters(), lr=1e-4)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=3e-4)

    def _tensorize_obs(self, obs: dict):
        """Converts NumPy observation dictionaries to PyTorch tensors."""
        tensor_obs = {}
        for k, v in obs.items():
            # Add batch dimension
            tensor_obs[k] = torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(self.device)
        return tensor_obs

    def train_episode(self, max_steps: int = 1000):
        """Runs a single training episode through the Tri-Brain system."""
        obs, _ = self.env.reset()
        episode_reward = 0.0

        for step in range(max_steps):
            tensor_obs = self._tensorize_obs(obs)

            # ---------------------------------------------------------
            # BRAIN 1: THE ORACLE (Prediction)
            # ---------------------------------------------------------
            oracle_probs = self.oracle(tensor_obs) # Shape: [1, 3] -> P(Bearish), P(Ranging), P(Bullish)

            # ---------------------------------------------------------
            # BRAIN 2: THE GATEKEEPER (Regime Filter)
            # ---------------------------------------------------------
            # Extract macro volatility features (e.g., from the 4H observation channel) to determine regime
            macro_features = obs["4H"][-1, 0:2].reshape(1, -1) 
            
            # NOTE: In a real run, Gatekeeper is pre-fitted on historical data. 
            # We mock authorization here if it hasn't been fitted yet.
            if self.gatekeeper.is_fitted:
                regime = self.gatekeeper.predict_regime(macro_features)
                is_authorized = self.gatekeeper.authorize_execution(regime, oracle_probs.detach().cpu().numpy()[0])
            else:
                is_authorized = True

            # ---------------------------------------------------------
            # BRAIN 3: THE MANAGER (Execution)
            # ---------------------------------------------------------
            if is_authorized:
                # Manager receives Oracle's conviction + current environment state (margin, cooldown, etc.)
                action, log_prob = self.actor.sample_action(oracle_probs, tensor_obs["state"])
                np_action = action.detach().cpu().numpy()[0]
            else:
                # Gatekeeper blocked the trade (terrible structure detected)
                np_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

            # Step the environment
            next_obs, reward, terminated, truncated, info = self.env.step(np_action)
            episode_reward += reward

            # NOTE: Standard SAC Replay Buffer storage and Loss backward passes go here.
            # Loss = Focal Loss for Oracle + Q-Value Bellman updates for Critic + Policy gradient for Actor[cite: 1, 3].
            
            obs = next_obs
            if terminated or truncated:
                break

        return episode_reward

if __name__ == "__main__":
    # Integration Smoke Test
    env = XAUMTFEnv(num_features=6)
    trainer = TriBrainTrainer(env)
    
    print("Starting Tri-Brain Integration Test...")
    reward = trainer.train_episode(max_steps=50)
    print(f"Integration Successful. Episodic Reward: {reward:.4f}")