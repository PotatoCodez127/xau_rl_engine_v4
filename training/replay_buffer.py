import torch
import numpy as np
import os
import pickle

class MTFReplayBuffer:
    def __init__(self, capacity: int, num_features: int = 11, state_dim: int = 4, action_dim: int = 3, device: str = "cuda"):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.device = torch.device(device)
        
        self.obs_15m = torch.zeros((capacity, 128, num_features), dtype=torch.float32, device=self.device)
        self.obs_30m = torch.zeros((capacity, 64, num_features), dtype=torch.float32, device=self.device)
        self.obs_1H = torch.zeros((capacity, 32, num_features), dtype=torch.float32, device=self.device)
        self.obs_4H = torch.zeros((capacity, 16, num_features), dtype=torch.float32, device=self.device)
        self.obs_state = torch.zeros((capacity, state_dim), dtype=torch.float32, device=self.device)
        
        self.next_obs_15m = torch.zeros((capacity, 128, num_features), dtype=torch.float32, device=self.device)
        self.next_obs_30m = torch.zeros((capacity, 64, num_features), dtype=torch.float32, device=self.device)
        self.next_obs_1H = torch.zeros((capacity, 32, num_features), dtype=torch.float32, device=self.device)
        self.next_obs_4H = torch.zeros((capacity, 16, num_features), dtype=torch.float32, device=self.device)
        self.next_obs_state = torch.zeros((capacity, state_dim), dtype=torch.float32, device=self.device)
        
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)

    def add(self, obs: dict, action: np.ndarray, reward: float, next_obs: dict, done: bool):
        self.obs_15m[self.ptr] = torch.as_tensor(obs["15m"], dtype=torch.float32, device=self.device)
        self.obs_30m[self.ptr] = torch.as_tensor(obs["30m"], dtype=torch.float32, device=self.device)
        self.obs_1H[self.ptr] = torch.as_tensor(obs["1H"], dtype=torch.float32, device=self.device)
        self.obs_4H[self.ptr] = torch.as_tensor(obs["4H"], dtype=torch.float32, device=self.device)
        self.obs_state[self.ptr] = torch.as_tensor(obs["state"], dtype=torch.float32, device=self.device)
        
        self.next_obs_15m[self.ptr] = torch.as_tensor(next_obs["15m"], dtype=torch.float32, device=self.device)
        self.next_obs_30m[self.ptr] = torch.as_tensor(next_obs["30m"], dtype=torch.float32, device=self.device)
        self.next_obs_1H[self.ptr] = torch.as_tensor(next_obs["1H"], dtype=torch.float32, device=self.device)
        self.next_obs_4H[self.ptr] = torch.as_tensor(next_obs["4H"], dtype=torch.float32, device=self.device)
        self.next_obs_state[self.ptr] = torch.as_tensor(next_obs["state"], dtype=torch.float32, device=self.device)
        
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[self.ptr] = torch.tensor([reward], dtype=torch.float32, device=self.device)
        self.dones[self.ptr] = torch.tensor([float(done)], dtype=torch.float32, device=self.device)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        ind = torch.randint(0, self.size, size=(batch_size,), device=self.device)
        
        b_obs = {
            "15m": self.obs_15m[ind], "30m": self.obs_30m[ind],
            "1H": self.obs_1H[ind], "4H": self.obs_4H[ind], "state": self.obs_state[ind]
        }
        b_next_obs = {
            "15m": self.next_obs_15m[ind], "30m": self.next_obs_30m[ind],
            "1H": self.next_obs_1H[ind], "4H": self.next_obs_4H[ind], "state": self.next_obs_state[ind]
        }
        return b_obs, self.actions[ind], self.rewards[ind], b_next_obs, self.dones[ind]

    def save(self, filepath: str):
        state_dict = {
            'ptr': self.ptr, 'size': self.size,
            'obs_15m': self.obs_15m.cpu(), 'obs_30m': self.obs_30m.cpu(),
            'obs_1H': self.obs_1H.cpu(), 'obs_4H': self.obs_4H.cpu(), 'obs_state': self.obs_state.cpu(),
            'next_obs_15m': self.next_obs_15m.cpu(), 'next_obs_30m': self.next_obs_30m.cpu(),
            'next_obs_1H': self.next_obs_1H.cpu(), 'next_obs_4H': self.next_obs_4H.cpu(), 'next_obs_state': self.next_obs_state.cpu(),
            'actions': self.actions.cpu(), 'rewards': self.rewards.cpu(), 'dones': self.dones.cpu()
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filepath: str):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.ptr = data['ptr']
                self.size = data['size']
                self.obs_15m[:self.size] = data['obs_15m'][:self.size].to(self.device)
                self.obs_30m[:self.size] = data['obs_30m'][:self.size].to(self.device)
                self.obs_1H[:self.size] = data['obs_1H'][:self.size].to(self.device)
                self.obs_4H[:self.size] = data['obs_4H'][:self.size].to(self.device)
                self.obs_state[:self.size] = data['obs_state'][:self.size].to(self.device)
                self.next_obs_15m[:self.size] = data['next_obs_15m'][:self.size].to(self.device)
                self.next_obs_30m[:self.size] = data['next_obs_30m'][:self.size].to(self.device)
                self.next_obs_1H[:self.size] = data['next_obs_1H'][:self.size].to(self.device)
                self.next_obs_4H[:self.size] = data['next_obs_4H'][:self.size].to(self.device)
                self.next_obs_state[:self.size] = data['next_obs_state'][:self.size].to(self.device)
                self.actions[:self.size] = data['actions'][:self.size].to(self.device)
                self.rewards[:self.size] = data['rewards'][:self.size].to(self.device)
                self.dones[:self.size] = data['dones'][:self.size].to(self.device)
