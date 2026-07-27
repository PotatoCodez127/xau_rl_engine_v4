import gym
from gym import spaces
import numpy as np

class XAUMTFEnv(gym.Env):
    def __init__(self, mtf_dict, max_steps=None):
        super(XAUMTFEnv, self).__init__()
        self.mtf_dict = mtf_dict
        self.data_15m = self.mtf_dict["15m"].values if hasattr(self.mtf_dict["15m"], "values") else self.mtf_dict["15m"]
        self.total_steps = len(self.data_15m)
        self.max_steps = max_steps if max_steps else self.total_steps
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "15m": spaces.Box(low=-10, high=10, shape=(128, 11), dtype=np.float32),
            "30m": spaces.Box(low=-10, high=10, shape=(64, 11), dtype=np.float32),
            "1H": spaces.Box(low=-10, high=10, shape=(32, 11), dtype=np.float32),
            "4H": spaces.Box(low=-10, high=10, shape=(16, 11), dtype=np.float32),
            "state": spaces.Box(low=-100, high=100, shape=(4,), dtype=np.float32)
        })

        self.PIP_SCALAR = 0.10
        self.SPREAD_PIPS = 2.0
        self.CONVICTION_THRESHOLD = 0.60
        self.reset()

    def reset(self):
        self.current_step = 128
        self.position = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.cooldown = 0
        return self._get_obs()

    def _slice_tf(self, tf, length):
        data = self.mtf_dict[tf]
        if hasattr(data, "values"): data = data.values
        idx = min(self.current_step, len(data))
        if idx < length:
            pad = np.zeros((length - idx, 11), dtype=np.float32)
            return np.vstack([pad, data[:idx]]).astype(np.float32)
        return data[idx - length : idx].astype(np.float32)

    def _get_obs(self):
        state_vec = np.array([self.position, 1.0, self.unrealized_pnl, self.cooldown], dtype=np.float32)
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
        
        # 1. Snapshot State
        prev_pos = self.position
        
        # 2. Process Unrealized PnL strictly using OLD entry price
        if prev_pos != 0.0:
            raw_pnl = prev_pos * (current_price - self.entry_price)
            self.unrealized_pnl = raw_pnl - (self.SPREAD_PIPS * self.PIP_SCALAR)
        else:
            self.unrealized_pnl = 0.0

        # 3. Dynamic Targets
        sl_pips = 20.0 + (k_sl * 30.0)
        target_sl = -1.0 * (sl_pips * self.PIP_SCALAR)
        tp_pips = 40.0 + (k_tp * 60.0)
        target_tp = tp_pips * self.PIP_SCALAR

        sl_hit = (prev_pos != 0.0) and (self.unrealized_pnl <= target_sl)
        tp_hit = (prev_pos != 0.0) and (self.unrealized_pnl >= target_tp)

        # 4. Intended Actions (The Hard Lock: No Flat Exits)
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

        # 5. Evaluate Closure
        trade_closed, reason = False, ""
        if sl_hit:
            trade_closed, reason = True, "Stop Loss"
            target_pos, self.cooldown = 0.0, 5
        elif tp_hit:
            trade_closed, reason = True, "Take Profit"
            target_pos = 0.0
        elif prev_pos != 0.0 and target_pos != prev_pos:
            trade_closed, reason = True, "Network Flip"

        # 6. Distribute Rewards BEFORE modifying Entry Price
        reward = 0.0
        if trade_closed:
            realized_pips = self.unrealized_pnl / self.PIP_SCALAR
            if reason == "Take Profit":
                reward = (realized_pips / 10.0) + 2.0 # TP Completion Bonus
            elif reason in ["Stop Loss", "Network Flip"]:
                reward = (realized_pips / 10.0)
            self.unrealized_pnl = 0.0
        else:
            if prev_pos == 0.0: reward = -0.01

        # 7. Finalize Setup for Next Step
        if target_pos != 0.0 and target_pos != prev_pos:
            self.entry_price = current_price
            
        self.position = target_pos
        self.current_step += 1
        done = self.current_step >= self.max_steps - 1
        
        return self._get_obs(), reward, done, {}