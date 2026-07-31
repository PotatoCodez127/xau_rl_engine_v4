# ==============================================================================
# TRI-BRAIN XAUUSD ENGINE: MASTER COLAB TRAINING PIPELINE (V7.3 - HIGH SPEED GPU)
# ==============================================================================
import os
import sys
import zipfile
from google.colab import drive
import pandas as pd

# 1. Mount Google Drive
print("📂 [1/5] Mounting Google Drive...")
drive.mount('/content/drive', force_remount=True)

DRIVE_PROJECT_DIR = '/content/drive/MyDrive/xau_rl_engine_v4'
ZIP_PATH_IN_DRIVE = os.path.join(DRIVE_PROJECT_DIR, 'xau_rl_engine_v4.zip')
FALLBACK_ZIP = '/content/drive/MyDrive/xau_rl_engine_v4.zip'
LOCAL_WORK_DIR = '/content/xau_rl_engine_v4'

os.makedirs(os.path.join(DRIVE_PROJECT_DIR, 'checkpoints'), exist_ok=True)
os.makedirs(os.path.join(DRIVE_PROJECT_DIR, 'compiled_models'), exist_ok=True)
os.makedirs(os.path.join(DRIVE_PROJECT_DIR, 'data'), exist_ok=True)

# 2. Extract Codebase Zip
print("📦 [2/5] Unpacking codebase from Google Drive...")
active_zip = ZIP_PATH_IN_DRIVE if os.path.exists(ZIP_PATH_IN_DRIVE) else (FALLBACK_ZIP if os.path.exists(FALLBACK_ZIP) else None)

if active_zip:
    with zipfile.ZipFile(active_zip, 'r') as zip_ref:
        zip_ref.extractall(LOCAL_WORK_DIR)
    print(f"✅ Extracted from {active_zip} to {LOCAL_WORK_DIR}")
else:
    raise FileNotFoundError(f"❌ Could not find 'xau_rl_engine_v4.zip' in Google Drive!")

# 3. Resolve Project Root Directory
project_root = LOCAL_WORK_DIR
for root, dirs, _ in os.walk(LOCAL_WORK_DIR):
    if 'training' in dirs and 'models' in dirs:
        project_root = root
        break

print(f"🎯 Project Root Located at: {project_root}")
os.chdir(project_root)
sys.path.insert(0, project_root)
os.environ['PYTHONPATH'] = f"{project_root}:{os.environ.get('PYTHONPATH', '')}"

# 4. Patch envs/xau_mtf_env.py
env_file_path = os.path.join(project_root, "envs", "xau_mtf_env.py")
env_code = """import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

class XAUMTFEnv(gym.Env):
    def __init__(self, mtf_dict, start_step=128, max_steps=None):
        super(XAUMTFEnv, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mtf_dict = mtf_dict
        self.data_15m = self.mtf_dict["15m"].values if hasattr(self.mtf_dict["15m"], "values") else self.mtf_dict["15m"]

        self.mtf_tensors = {
            k: torch.tensor(v.values if hasattr(v, "values") else v, dtype=torch.float32, device=self.device)
            for k, v in self.mtf_dict.items()
        }

        self.total_steps = len(self.data_15m)
        self.start_step = start_step if start_step >= 128 else 128
        self.max_steps = max_steps if max_steps else self.total_steps
        num_cols = self.data_15m.shape[1]

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "15m": spaces.Box(low=-10, high=10, shape=(128, num_cols), dtype=np.float32),
            "30m": spaces.Box(low=-10, high=10, shape=(64, num_cols), dtype=np.float32),
            "1H": spaces.Box(low=-10, high=10, shape=(32, num_cols), dtype=np.float32),
            "4H": spaces.Box(low=-10, high=10, shape=(16, num_cols), dtype=np.float32),
            "state": spaces.Box(low=-100, high=100, shape=(4,), dtype=np.float32)
        })

        self.PIP_SCALAR = 0.10
        self.SPREAD_PIPS = 2.0
        self.CONVICTION_THRESHOLD = 0.55  # 🚀 Tightened
        self.POST_TRADE_COOLDOWN = 12     # 🚀 12 bars = 3 Hours
        self.MAX_HOLD_BARS = 32           # 🚀 32 bars = 8 Hours max duration
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.start_step
        self.position = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.cooldown = 0
        self.bars_in_trade = 0 # Track time in trade
        return self._get_obs(), {}

    def _slice_tf(self, tf, length):
        data = self.mtf_tensors[tf]
        if tf == "15m": idx = self.current_step
        elif tf == "30m": idx = self.current_step // 2
        elif tf == "1H": idx = self.current_step // 4
        elif tf == "4H": idx = self.current_step // 16
        else: idx = self.current_step

        idx = min(idx, data.shape[0])

        if idx < length:
            pad = torch.zeros((length - idx, data.shape[1]), dtype=torch.float32, device=self.device)
            return torch.vstack([pad, data[:idx]])
        return data[idx - length : idx]

    def _get_obs(self):
        state_vec = torch.tensor([self.position, 1.0, self.unrealized_pnl, self.cooldown], dtype=torch.float32, device=self.device)
        return {
            "15m": self._slice_tf("15m", 128), "30m": self._slice_tf("30m", 64),
            "1H": self._slice_tf("1H", 32), "4H": self._slice_tf("4H", 16),
            "state": state_vec
        }

    def step(self, action):
        current_price = self.data_15m[self.current_step, 3]
        raw_direction = action[0]
        k_tp = (action[1] + 1.0) / 2.0
        k_sl = (action[2] + 1.0) / 2.0

        prev_pos = self.position

        if prev_pos != 0.0:
            raw_pnl = prev_pos * (current_price - self.entry_price)
            self.unrealized_pnl = raw_pnl - (self.SPREAD_PIPS * self.PIP_SCALAR)
        else:
            self.unrealized_pnl = 0.0

        target_sl = -1.0 * ((20.0 + (k_sl * 30.0)) * self.PIP_SCALAR)
        target_tp = (40.0 + (k_tp * 60.0)) * self.PIP_SCALAR

        sl_hit = (prev_pos != 0.0) and (self.unrealized_pnl <= target_sl)
        tp_hit = (prev_pos != 0.0) and (self.unrealized_pnl >= target_tp)
        time_stop_hit = (prev_pos != 0.0) and (self.bars_in_trade >= self.MAX_HOLD_BARS)

        target_pos = prev_pos
        if self.cooldown > 0:
            self.cooldown -= 1
        else:
            if prev_pos == 0.0:
                if raw_direction > self.CONVICTION_THRESHOLD: target_pos = 1.0
                elif raw_direction < -self.CONVICTION_THRESHOLD: target_pos = -1.0
            elif prev_pos > 0.0:
                if raw_direction < -self.CONVICTION_THRESHOLD: target_pos = -1.0
            elif prev_pos < 0.0:
                if raw_direction > self.CONVICTION_THRESHOLD: target_pos = 1.0

        trade_closed, reason = False, ""
        if sl_hit:
            trade_closed, reason = True, "Stop Loss"
            target_pos, self.cooldown = 0.0, self.POST_TRADE_COOLDOWN
        elif tp_hit:
            trade_closed, reason = True, "Take Profit"
            target_pos, self.cooldown = 0.0, self.POST_TRADE_COOLDOWN
        elif time_stop_hit:
            trade_closed, reason = True, "Time Stop"
            target_pos, self.cooldown = 0.0, self.POST_TRADE_COOLDOWN
        elif prev_pos != 0.0 and target_pos != prev_pos:
            trade_closed, reason = True, "Network Flip"
            self.cooldown = self.POST_TRADE_COOLDOWN

        reward = 0.0
        if trade_closed:
            realized_pips = self.unrealized_pnl / self.PIP_SCALAR
            if reason == "Take Profit":
                reward = (realized_pips / 10.0) + 2.0
            elif reason in ["Stop Loss", "Time Stop", "Network Flip"]:
                reward = (realized_pips / 10.0)
            self.unrealized_pnl = 0.0
        else:
            if prev_pos == 0.0:
                reward = 0.0  # 🚀 Neutral flat reward
            else:
                # 🚀 Compounding time penalty
                reward = -0.001 * (self.bars_in_trade ** 1.5)

        if target_pos != 0.0 and target_pos != prev_pos:
            self.entry_price = current_price
            self.bars_in_trade = 1
        elif target_pos != 0.0 and target_pos == prev_pos:
            self.bars_in_trade += 1
        else:
            self.bars_in_trade = 0

        self.position = target_pos
        self.current_step += 1

        terminated = self.current_step >= self.max_steps - 1
        return self._get_obs(), reward, terminated, False, {}
"""
with open(env_file_path, "w") as f:
    f.write(env_code)
print("🛠️ Environment file successfully patched with Time-In-Trade penalties and strict intraday hyperparameters!")

# 4.5 Patch models/manager_sac.py
sac_file_path = os.path.join(project_root, "models", "manager_sac.py")
tuner_code = """import math

class EntropyTuner(nn.Module):
    \"\"\"
    Automatically tunes the SAC entropy coefficient with a hard floor to prevent execution paralysis.
    \"\"\"
    def __init__(self, action_dim: int = 3, lr: float = 1e-4, min_alpha: float = 0.015): # 🚀 Tightened alpha
        super(EntropyTuner, self).__init__()
        self.target_entropy = -float(action_dim)
        self.min_log_alpha = math.log(min_alpha)

        # Initialize at or above the minimum floor
        self.log_alpha = nn.Parameter(torch.tensor([max(0.0, self.min_log_alpha)], requires_grad=True))
        self.optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    def update(self, log_pi: torch.Tensor):
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.optimizer.zero_grad()
        alpha_loss.backward()
        self.optimizer.step()

        # Hard clamp log_alpha so exploration never flatlines
        with torch.no_grad():
            self.log_alpha.clamp_(min=self.min_log_alpha)

        return alpha_loss.item(), self.log_alpha.exp().item()
"""
with open(sac_file_path, "a") as f:
    f.write(tuner_code)
print("🛠️ SAC Manager successfully patched with clamped EntropyTuner (Alpha Floor 0.015)!")

# 4.7 Patch training/replay_buffer.py to be 100% GPU-Native
buffer_file_path = os.path.join(project_root, "training", "replay_buffer.py")
buffer_code = """import torch
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
"""
with open(buffer_file_path, "w") as f:
    f.write(buffer_code)
print("🛠️ Replay Buffer successfully patched for GPU-Native memory!")

# 5. Install Dependencies
print("⚡ [3/5] Installing engine dependencies...")
!pip install -q gymnasium "hmmlearn>=0.3.0" onnx onnxruntime pyarrow fastparquet joblib

# 6. Generate and Execute Resilient Training Runner
runner_code = """
import os
import sys
import copy
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.getcwd())

from envs.xau_mtf_env import XAUMTFEnv
from models.oracle_transformer import SpatialOracle
from models.gatekeeper_hmm import ContextGatekeeper
from models.manager_sac import SACActor, SACCritic, EntropyTuner
from training.cpcv_validation import PurgedCombinatorialCV
from training.replay_buffer import MTFReplayBuffer

DRIVE_DIR = "/content/drive/MyDrive/xau_rl_engine_v4"
DATA_DIR = os.path.join(DRIVE_DIR, "data")
CHECKPOINT_DIR = os.path.join(DRIVE_DIR, "checkpoints")

def run_training():
    master_tensor_path = os.path.join(DATA_DIR, "master_training_tensor.pkl")
    mtf_dict = joblib.load(master_tensor_path)
    actual_num_features = mtf_dict["15m"].shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on device: {device} | Detected Features: {actual_num_features}")

    oracle = SpatialOracle(num_features=actual_num_features).to(device)
    gatekeeper = ContextGatekeeper(n_components=3)
    actor = SACActor().to(device)
    critic = SACCritic().to(device)
    critic_target = copy.deepcopy(critic).to(device)
    entropy_tuner = EntropyTuner(action_dim=3).to(device)

    oracle_opt = torch.optim.Adam(oracle.parameters(), lr=1e-4)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)
    scaler = GradScaler('cuda')

    chkpt_path = os.path.join(CHECKPOINT_DIR, "tribrain_checkpoint.pth")
    gatekeeper_path = os.path.join(CHECKPOINT_DIR, "gatekeeper.pkl")
    buffer_path = os.path.join(CHECKPOINT_DIR, "replay_buffer.pkl")

    start_fold, start_epoch = 0, 0

    if os.path.exists(chkpt_path):
        checkpoint = torch.load(chkpt_path, map_location=device, weights_only=False)
        try:
            oracle.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in checkpoint["oracle_state"].items()})
            actor.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in checkpoint["actor_state"].items()})
            critic.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in checkpoint["critic_state"].items()})
            critic_target.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in checkpoint["critic_state"].items()})
            oracle_opt.load_state_dict(checkpoint["oracle_opt_state"])
            actor_opt.load_state_dict(checkpoint["actor_opt_state"])
            critic_opt.load_state_dict(checkpoint["critic_opt_state"])
            if "tuner_state" in checkpoint:
                entropy_tuner.load_state_dict(checkpoint["tuner_state"])
                entropy_tuner.optimizer.load_state_dict(checkpoint["tuner_opt_state"])
            if "scaler_state" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state"])
            start_fold, start_epoch = checkpoint.get("fold", 0), checkpoint.get("epoch", 0)
            print(f"🔄 Checkpoint restored! Resuming from Fold {start_fold + 1}, Epoch {start_epoch + 1}")
        except Exception as e:
            print(f"⚠️ Checkpoint mismatch: {e}")

    print("🔥 Compiling Neural Networks (torch.compile)...")
    oracle = torch.compile(oracle)
    actor = torch.compile(actor)
    critic = torch.compile(critic)

    replay_buffer = MTFReplayBuffer(capacity=10000, num_features=actual_num_features, device=device)
    if os.path.exists(buffer_path):
        try: replay_buffer.load(buffer_path)
        except: pass

    cpcv = PurgedCombinatorialCV(n_folds=6, n_test_folds=2)
    paths = list(cpcv.split(mtf_dict["15m"]))

    epochs_per_fold = 20
    batch_size = 1024
    UPDATE_FREQ = 100       # Tuned to prevent excessive backprop overhead
    UPDATE_PASSES = 2       # Tuned down to prevent single-trajectory overfitting
    gamma = 0.9438
    tau = 0.0028

    print(f"🔥 Starting Backpropagation Loop across {len(paths)} CPCV paths...")

    for fold_idx in range(start_fold, len(paths)):
        train_idx, test_idx = paths[fold_idx]

        jumps = np.where(np.diff(train_idx) > 1)[0] + 1
        train_blocks = np.split(train_idx, jumps)
        valid_blocks = [b for b in train_blocks if len(b) > 200]

        if not gatekeeper.is_fitted:
            print(f"⚙️ Fitting HMM Gatekeeper on {len(train_idx)} samples for Fold {fold_idx+1}...")
            macro_train = mtf_dict["15m"][train_idx, :2] if isinstance(mtf_dict["15m"], np.ndarray) else mtf_dict["15m"].iloc[train_idx].values[:, :2]
            gatekeeper.fit(macro_train)
            gatekeeper.save_model(gatekeeper_path)

        for epoch in range(start_epoch, epochs_per_fold):
            block_idx = valid_blocks[np.random.randint(len(valid_blocks))]

            # Cap the block to 20,000 steps to prevent single-epoch stagnation
            block_idx = block_idx[:20000]

            start_step = block_idx[0]
            max_steps = block_idx[-1]

            print(f"\\n==================== CPCV PATH {fold_idx + 1}/{len(paths)} | EPOCH {epoch + 1}/{epochs_per_fold} ====================")
            print(f"▶️ Exploring Continuous Block: {len(block_idx)} candles")

            env = XAUMTFEnv(mtf_dict=mtf_dict, start_step=start_step, max_steps=max_steps)
            obs, _ = env.reset()
            ep_reward = 0.0
            actor_losses, critic_losses, alpha_vals = [], [], []

            for step in range(len(block_idx)):
                # Data is already in GPU. Only adding the batch dimension is required.
                tensor_obs = {k: v.unsqueeze(0) for k, v in obs.items()}

                # Using inference_mode for faster continuous forward passes
                with torch.inference_mode():
                    oracle_probs = oracle(tensor_obs)
                    action, _ = actor.sample_action(oracle_probs, tensor_obs["state"])

                np_action = action.cpu().numpy()[0]
                next_obs, reward, terminated, truncated, _ = env.step(np_action)
                done = terminated or truncated

                replay_buffer.add(obs, np_action, reward, next_obs, done)
                ep_reward += reward
                obs = next_obs

                current_alpha = entropy_tuner.log_alpha.exp().item()

                if replay_buffer.size > batch_size and step % UPDATE_FREQ == 0:
                    for _ in range(UPDATE_PASSES):
                        b_obs, b_actions, b_rewards, b_next_obs, b_dones = replay_buffer.sample(batch_size)

                        with torch.no_grad():
                            next_oracle_probs = oracle(b_next_obs)
                            next_actions, next_log_pi = actor.sample_action(next_oracle_probs, b_next_obs["state"])
                            target_q1, target_q2 = critic_target(next_oracle_probs, b_next_obs["state"], next_actions)
                            target_q = torch.min(target_q1, target_q2) - current_alpha * next_log_pi
                            target_value = b_rewards + gamma * (1 - b_dones) * target_q

                        curr_oracle_probs = oracle(b_obs)
                        detached_oracle_probs = curr_oracle_probs.detach()
                        new_actions, log_pi = actor.sample_action(detached_oracle_probs, b_obs["state"])

                        alpha_loss, current_alpha = entropy_tuner.update(log_pi)

                        with autocast(device_type="cuda", dtype=torch.float16):
                            curr_q1, curr_q2 = critic(curr_oracle_probs, b_obs["state"], b_actions)
                            critic_loss = F.mse_loss(curr_q1, target_value) + F.mse_loss(curr_q2, target_value)

                        oracle_opt.zero_grad(set_to_none=True)
                        critic_opt.zero_grad(set_to_none=True)
                        scaler.scale(critic_loss).backward()
                        scaler.step(critic_opt)

                        with autocast(device_type="cuda", dtype=torch.float16):
                            q1_new, q2_new = critic(detached_oracle_probs, b_obs["state"], new_actions)
                            q_new = torch.min(q1_new, q2_new)
                            actor_loss = (current_alpha * log_pi - q_new).mean()

                        actor_opt.zero_grad(set_to_none=True)
                        scaler.scale(actor_loss).backward()
                        scaler.step(actor_opt)

                        scaler.step(oracle_opt)
                        scaler.update()

                        for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

                        actor_losses.append(actor_loss.item())
                        critic_losses.append(critic_loss.item())
                        alpha_vals.append(current_alpha)

                if step > 0 and step % 1000 == 0:
                    avg_c_loss = np.mean(critic_losses[-50:]) if critic_losses else 0.0
                    print(f"  -> Step {step}/{len(block_idx)} | Alpha: {current_alpha:.4f} | PnL Reward: {ep_reward:.2f} | C-Loss: {avg_c_loss:.4f}")

                    if step % 10000 == 0:
                        print(f"  💾 [Auto-Save] Securing checkpoint and replay buffer to Google Drive...")
                        checkpoint = {
                            "fold": fold_idx, "epoch": epoch,
                            "oracle_state": oracle.state_dict(),
                            "actor_state": actor.state_dict(),
                            "critic_state": critic.state_dict(),
                            "tuner_state": entropy_tuner.state_dict(),
                            "oracle_opt_state": oracle_opt.state_dict(),
                            "actor_opt_state": actor_opt.state_dict(),
                            "critic_opt_state": critic_opt.state_dict(),
                            "tuner_opt_state": entropy_tuner.optimizer.state_dict(),
                            "scaler_state": scaler.state_dict()
                        }
                        torch.save(checkpoint, chkpt_path)
                        replay_buffer.save(buffer_path)

                if done: break

            avg_a_loss = np.mean(actor_losses) if actor_losses else 0.0
            avg_c_loss = np.mean(critic_losses) if critic_losses else 0.0
            avg_alpha = np.mean(alpha_vals) if alpha_vals else current_alpha

            print(f"✅ END FOLD {fold_idx+1} EP {epoch+1} | Total Reward: {ep_reward:.4f} | C-Loss: {avg_c_loss:.4f} | Alpha: {avg_alpha:.4f}")

            checkpoint = {
                "fold": fold_idx, "epoch": epoch + 1,
                "oracle_state": oracle.state_dict(),
                "actor_state": actor.state_dict(),
                "critic_state": critic.state_dict(),
                "tuner_state": entropy_tuner.state_dict(),
                "oracle_opt_state": oracle_opt.state_dict(),
                "actor_opt_state": actor_opt.state_dict(),
                "critic_opt_state": critic_opt.state_dict(),
                "tuner_opt_state": entropy_tuner.optimizer.state_dict(),
                "scaler_state": scaler.state_dict()
            }
            torch.save(checkpoint, chkpt_path)
            replay_buffer.save(buffer_path)

        start_epoch = 0

if __name__ == "__main__":
    run_training()
"""

with open("run_colab_training.py", "w") as f:
    f.write(runner_code)

print("🚀 [4/5] Launching Tri-Brain PyTorch Joint Optimization Loop...")
!python run_colab_training.py
print("🎉 [5/5] Training Cycle Complete or Checkpoint Secured.")