import numpy as np
import torch
import os
import pickle

class MTFReplayBuffer:
    """
    A memory-efficient, pre-allocated Replay Buffer for the Tri-Brain SAC Manager.
    Handles asymmetric dictionary observation spaces and serializes directly to disk
    to survive cloud hardware timeouts[cite: 1].
    """
    def __init__(self, capacity: int, num_features: int = 6, state_dim: int = 4, action_dim: int = 3):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pre-allocate arrays to prevent RAM fragmentation and OOM crashes
        self.obs_15m = np.zeros((capacity, 128, num_features), dtype=np.float32)
        self.obs_30m = np.zeros((capacity, 64, num_features), dtype=np.float32)
        self.obs_1H = np.zeros((capacity, 32, num_features), dtype=np.float32)
        self.obs_4H = np.zeros((capacity, 16, num_features), dtype=np.float32)
        self.obs_state = np.zeros((capacity, state_dim), dtype=np.float32)

        self.next_obs_15m = np.zeros((capacity, 128, num_features), dtype=np.float32)
        self.next_obs_30m = np.zeros((capacity, 64, num_features), dtype=np.float32)
        self.next_obs_1H = np.zeros((capacity, 32, num_features), dtype=np.float32)
        self.next_obs_4H = np.zeros((capacity, 16, num_features), dtype=np.float32)
        self.next_obs_state = np.zeros((capacity, state_dim), dtype=np.float32)

        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs: dict, action: np.ndarray, reward: float, next_obs: dict, done: bool):
        """Stores a single transition into the pre-allocated buffer."""
        self.obs_15m[self.ptr] = obs["15m"]
        self.obs_30m[self.ptr] = obs["30m"]
        self.obs_1H[self.ptr] = obs["1H"]
        self.obs_4H[self.ptr] = obs["4H"]
        self.obs_state[self.ptr] = obs["state"]

        self.next_obs_15m[self.ptr] = next_obs["15m"]
        self.next_obs_30m[self.ptr] = next_obs["30m"]
        self.next_obs_1H[self.ptr] = next_obs["1H"]
        self.next_obs_4H[self.ptr] = next_obs["4H"]
        self.next_obs_state[self.ptr] = next_obs["state"]

        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """Samples a random batch of experiences and returns PyTorch tensors."""
        ind = np.random.randint(0, self.size, size=batch_size)

        batch_obs = {
            "15m": torch.FloatTensor(self.obs_15m[ind]).to(self.device),
            "30m": torch.FloatTensor(self.obs_30m[ind]).to(self.device),
            "1H": torch.FloatTensor(self.obs_1H[ind]).to(self.device),
            "4H": torch.FloatTensor(self.obs_4H[ind]).to(self.device),
            "state": torch.FloatTensor(self.obs_state[ind]).to(self.device)
        }

        batch_next_obs = {
            "15m": torch.FloatTensor(self.next_obs_15m[ind]).to(self.device),
            "30m": torch.FloatTensor(self.next_obs_30m[ind]).to(self.device),
            "1H": torch.FloatTensor(self.next_obs_1H[ind]).to(self.device),
            "4H": torch.FloatTensor(self.next_obs_4H[ind]).to(self.device),
            "state": torch.FloatTensor(self.next_obs_state[ind]).to(self.device)
        }

        actions = torch.FloatTensor(self.actions[ind]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[ind]).to(self.device)
        dones = torch.FloatTensor(self.dones[ind]).to(self.device)

        return batch_obs, actions, rewards, batch_next_obs, dones

    def save(self, filepath: str):
        """Serializes the buffer to disk."""
        print(f"Serializing Replay Buffer to {filepath}...")
        with open(filepath, 'wb') as f:
            pickle.dump({
                'ptr': self.ptr,
                'size': self.size,
                'obs_15m': self.obs_15m,
                'obs_30m': self.obs_30m,
                'obs_1H': self.obs_1H,
                'obs_4H': self.obs_4H,
                'obs_state': self.obs_state,
                'next_obs_15m': self.next_obs_15m,
                'next_obs_30m': self.next_obs_30m,
                'next_obs_1H': self.next_obs_1H,
                'next_obs_4H': self.next_obs_4H,
                'next_obs_state': self.next_obs_state,
                'actions': self.actions,
                'rewards': self.rewards,
                'dones': self.dones
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Replay Buffer successfully saved.")

    def load(self, filepath: str):
        """Loads the buffer from disk."""
        if os.path.exists(filepath):
            print(f"Restoring Replay Buffer from {filepath}...")
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.ptr = data['ptr']
                self.size = data['size']
                self.obs_15m = data['obs_15m']
                self.obs_30m = data['obs_30m']
                self.obs_1H = data['obs_1H']
                self.obs_4H = data['obs_4H']
                self.obs_state = data['obs_state']
                self.next_obs_15m = data['next_obs_15m']
                self.next_obs_30m = data['next_obs_30m']
                self.next_obs_1H = data['next_obs_1H']
                self.next_obs_4H = data['next_obs_4H']
                self.next_obs_state = data['next_obs_state']
                self.actions = data['actions']
                self.rewards = data['rewards']
                self.dones = data['dones']
            print("Replay Buffer successfully restored.")
        else:
            print("No existing Replay Buffer found. Initializing empty buffer.")