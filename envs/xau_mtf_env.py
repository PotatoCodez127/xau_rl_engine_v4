import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class XAUMTFEnv(gym.Env):
    """
    Custom Gymnasium Environment for the Tri-Brain XAUUSD Engine.
    Uses Asymmetric Spatial Tensors for the Oracle and Continuous Action Spaces for the SAC Manager.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, dataframes: dict = None, num_features: int = 6):
        super(XAUMTFEnv, self).__init__()
        
        self.dataframes = dataframes
        self.num_features = num_features 
        self.current_step = 128 # Starting index to allow the largest lookback window
        self.max_steps = 1000

        # 1. OBSERVATION SPACE: Asymmetric MTF Spatial Tensors
        self.observation_space = spaces.Dict({
            "15m": spaces.Box(low=-1.0, high=1.0, shape=(128, self.num_features), dtype=np.float32),
            "30m": spaces.Box(low=-1.0, high=1.0, shape=(64, self.num_features), dtype=np.float32),
            "1H": spaces.Box(low=-1.0, high=1.0, shape=(32, self.num_features), dtype=np.float32),
            "4H": spaces.Box(low=-1.0, high=1.0, shape=(16, self.num_features), dtype=np.float32),
            
            # Internal state: [Current Position (-1 to 1), Margin %, Unrealized PnL, Cooldown Timer]
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        })

        # 2. ACTION SPACE: Continuous SAC Output
        # [Direction/Volume, Take-Profit Multiplier, Stop-Loss Multiplier]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Internal environment tracking
        self.position = 0.0
        self.margin = 1.0
        self.unrealized_pnl = 0.0
        self.cooldown_timer = 0.0
        self.peak_equity = 1.0

    def _get_obs(self):
        """Constructs the current asymmetric tensor observation by slicing historical data."""
        if self.dataframes is None:
            # Fallback for initialization before data is loaded
            return {
                "15m": np.zeros((128, self.num_features), dtype=np.float32),
                "30m": np.zeros((64, self.num_features), dtype=np.float32),
                "1H": np.zeros((32, self.num_features), dtype=np.float32),
                "4H": np.zeros((16, self.num_features), dtype=np.float32),
                "state": np.array([self.position, self.margin, self.unrealized_pnl, self.cooldown_timer], dtype=np.float32)
            }

        # Helper to safely slice MTF arrays backwards from the current step
        def slice_tf(tf, length):
            data = self.dataframes[tf]
            # Convert pandas DataFrame to numpy if necessary
            if hasattr(data, "iloc"):
                data = data.values
                
            idx = min(self.current_step, len(data))
            
            # Pad with zeros if we are too early in the dataset
            if idx < length:
                pad = np.zeros((length - idx, self.num_features), dtype=np.float32)
                slice_data = data[:idx]
                return np.vstack([pad, slice_data]).astype(np.float32)
            
            return data[idx - length : idx].astype(np.float32)

        return {
            "15m": slice_tf("15m", 128),
            "30m": slice_tf("30m", 64),
            "1H": slice_tf("1H", 32),
            "4H": slice_tf("4H", 16),
            "state": np.array([self.position, self.margin, self.unrealized_pnl, self.cooldown_timer], dtype=np.float32)
        }

    def reset(self, seed=None, options=None):
        """Resets the environment to a clean state."""
        super().reset(seed=seed)
        self.current_step = 128
        self.position = 0.0
        self.margin = 1.0
        self.unrealized_pnl = 0.0
        self.cooldown_timer = 0.0
        self.peak_equity = 1.0
        
        info = {"msg": "Environment Reset"}
        return self._get_obs(), info

    def step(self, action):
        """Executes a single step in the environment."""
        # 1. Parse SAC Continuous Action Space
        direction_vol = action[0] 
        k_tp = (action[1] + 1) / 2.0 
        k_sl = (action[2] + 1) / 2.0 
        
        terminated = False
        truncated = False
        
        # Ensure we have data loaded and aren't out of bounds
        if self.dataframes is None:
            return self._get_obs(), 0.0, False, True, {}
            
        data_15m = self.dataframes["15m"].values if hasattr(self.dataframes["15m"], "iloc") else self.dataframes["15m"]
        
        if self.current_step >= len(data_15m) - 1:
            truncated = True
            return self._get_obs(), 0.0, terminated, truncated, {"msg": "End of data"}

        # Assuming Close price is the 4th feature (Index 3) in standard OHLCV
        prev_price = data_15m[self.current_step - 1, 3]
        
        # Advance simulation
        self.current_step += 1
        current_price = data_15m[self.current_step - 1, 3]
        
        # Cooldown mechanics
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1

        # 2. Position Management & Friction
        prev_position = self.position
        
        # Implement a deadzone: Actions between -0.1 and 0.1 are interpreted as "Close all / Hold flat"
        if abs(direction_vol) < 0.1:
            self.position = 0.0
        else:
            self.position = direction_vol
            
        # Calculate Transaction Cost (Spread/Commission) based on position change size
        position_delta = abs(self.position - prev_position)
        transaction_cost = position_delta * 0.00015 # Simulated roughly 1.5 pips penalty for switching bias
        
        # 3. Dense Reward Engineering (PnL)
        # Calculate percent change of price
        price_change = (current_price - prev_price) / prev_price
        
        # Gross step PnL = Our Position Size * Price Change
        step_pnl = self.position * price_change
        
        # Update overall running trade math
        if self.position == 0.0:
            self.unrealized_pnl = 0.0 # Reset when flat
        else:
            self.unrealized_pnl += step_pnl - transaction_cost

        # Base Reward: The immediate directional PnL scaled for neural network gradients
        reward = (step_pnl * 100.0) - (transaction_cost * 10.0)
        
        # Inactivity Penalty: Minor bleed if holding 0 position to force strategy discovery
        if self.position == 0.0:
            reward -= 0.001
            
        # 4. Target Evaluation (Simulated Take-Profit / Stop-Loss)
        sl_threshold = -0.01 * (1.0 + k_sl) # Dynamic SL between -1% and -2% drawdown
        tp_threshold = 0.02 * (1.0 + k_tp)  # Dynamic TP between +2% and +4% gain
        
        if self.unrealized_pnl <= sl_threshold:
            reward -= 1.0 # Heavy episodic penalty for hitting SL
            self.position = 0.0
            self.unrealized_pnl = 0.0
            self.cooldown_timer = 5 # Force network to pause after loss
        elif self.unrealized_pnl >= tp_threshold:
            reward += 1.0 # Bonus for hitting TP
            self.position = 0.0
            self.unrealized_pnl = 0.0

        # Terminate if Max Steps reached
        if (self.current_step - 128) > self.max_steps:
            truncated = True

        info = {
            "current_step": self.current_step, 
            "position": self.position,
            "unrealized_pnl": self.unrealized_pnl,
            "step_reward": reward
        }
        
        return self._get_obs(), float(reward), terminated, truncated, info