import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

class XAUMTFEnv(gym.Env):
    def __init__(self, mtf_dict, start_step=128, max_steps=None):
        super(XAUMTFEnv, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mtf_dict = mtf_dict
        self.data_15m = self.mtf_dict["15m"].values if hasattr(self.mtf_dict["15m"], "values") else self.mtf_dict["15m"]
        
        # Convert dictionary to GPU tensors once on init
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
        self.CONVICTION_THRESHOLD = 0.55
        self.POST_TRADE_COOLDOWN = 12
        self.MAX_HOLD_BARS = 32
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.start_step
        self.position = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.cooldown = 0
        self.bars_in_trade = 0
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

        # Check trade closure triggers
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

        # Reward Structure
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
                reward = 0.0  # 🚀 Neutral flat reward (No decay penalty)
            else:
                # Compounding time penalty only while holding open risk
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