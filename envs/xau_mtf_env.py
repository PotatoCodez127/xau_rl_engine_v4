import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class XAUMultiTimeframeEnv(gym.Env):
    """
    Tri-Brain XAUUSD MTF Gym Environment (Version 4 - Hard Lock Edition)
    
    Physics & Market Calibrations:
    - Pip Math: $0.10 price movement = 1.0 pip.
    - Spread Friction: 2.0 pips ($0.20) per new entry / flip.
    - Target Scaling: TP = 40 to 100 pips, SL = 20 to 50 pips.
    - Hard Lock: Once entered, positions CANNOT exit to FLAT manually.
      Exits occur ONLY via TP, SL, or Directional Flip.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, feature_cols: list):
        super(XAUMultiTimeframeEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.num_samples = len(self.df)
        
        # Action Space: [direction_vol (-1 to 1), k_tp (-1 to 1), k_sl (-1 to 1)]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        
        # Observation Space matching feature matrix
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(feature_cols),), dtype=np.float32
        )
        
        # State tracking
        self.current_step = 0
        self.position = 0.0       # 0.0 = Flat, 1.0 = Long, -1.0 = Short
        self.entry_price = 0.0
        self.tp_price = 0.0
        self.sl_price = 0.0
        self.trade_duration = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0.0
        self.entry_price = 0.0
        self.tp_price = 0.0
        self.sl_price = 0.0
        self.trade_duration = 0
        
        obs = self.df.iloc[self.current_step][self.feature_cols].values.astype(np.float32)
        return obs, {}

    def _calculate_targets(self, current_price: float, position: float, raw_ktp: float, raw_ksl: float):
        """Scales continuous network outputs to pip-based price targets."""
        # Map raw [-1, 1] to pip ranges: TP [40, 100], SL [20, 50]
        tp_pips = 40.0 + ((raw_ktp + 1.0) / 2.0) * 60.0
        sl_pips = 20.0 + ((raw_ksl + 1.0) / 2.0) * 30.0
        
        # 1 Pip = $0.10 delta on XAUUSD
        tp_delta = tp_pips * 0.10
        sl_delta = sl_pips * 0.10
        
        if position == 1.0: # LONG
            tp_price = current_price + tp_delta
            sl_price = current_price - sl_delta
        else: # SHORT
            tp_price = current_price - tp_delta
            sl_price = current_price + sl_delta
            
        return tp_price, sl_price

    def step(self, action: np.ndarray):
        direction_vol, raw_ktp, raw_ksl = action[0], action[1], action[2]
        
        current_row = self.df.iloc[self.current_step]
        high_price = current_row['high']
        low_price = current_row['low']
        close_price = current_row['close']
        
        reward = 0.0
        terminated = False
        truncated = False
        
        # ------------------------------------------------------------------
        # 1. CHECK TP / SL ON ACTIVE POSITION
        # ------------------------------------------------------------------
        if self.position == 1.0: # LONG
            if high_price >= self.tp_price:
                realized_pips = (self.tp_price - self.entry_price) / 0.10
                reward = (realized_pips / 10.0) + 2.0 # Reward + TP Bonus
                self.position = 0.0
                self.entry_price = 0.0
            elif low_price <= self.sl_price:
                realized_pips = (self.sl_price - self.entry_price) / 0.10
                reward = realized_pips / 10.0
                self.position = 0.0
                self.entry_price = 0.0

        elif self.position == -1.0: # SHORT
            if low_price <= self.tp_price:
                realized_pips = (self.entry_price - self.tp_price) / 0.10
                reward = (realized_pips / 10.0) + 2.0 # Reward + TP Bonus
                self.position = 0.0
                self.entry_price = 0.0
            elif high_price >= self.sl_price:
                realized_pips = (self.entry_price - self.sl_price) / 0.10
                reward = realized_pips / 10.0
                self.position = 0.0
                self.entry_price = 0.0

        # ------------------------------------------------------------------
        # 2. HARD LOCK DIRECTIONAL LOGIC & FLIPS
        # ------------------------------------------------------------------
        if self.position == 0.0:
            # Out of market: Look for new entries with > 0.60 conviction
            if direction_vol > 0.60:
                self.position = 1.0
                self.entry_price = close_price
                self.tp_price, self.sl_price = self._calculate_targets(close_price, 1.0, raw_ktp, raw_ksl)
                reward -= 0.20 # 2.0 Pip Spread Penalty (Scaled)
                self.trade_duration = 0
            elif direction_vol < -0.60:
                self.position = -1.0
                self.entry_price = close_price
                self.tp_price, self.sl_price = self._calculate_targets(close_price, -1.0, raw_ktp, raw_ksl)
                reward -= 0.20 # 2.0 Pip Spread Penalty (Scaled)
                self.trade_duration = 0

        elif self.position == 1.0:
            # HARD LOCK: Neutral signals keep LONG position open
            if direction_vol < -0.60:
                # Directional Flip to SHORT: Close LONG first, then open SHORT
                exit_pips = (close_price - self.entry_price) / 0.10 - 2.0 # Deduct 2-pip spread
                reward += exit_pips / 10.0
                
                # Open new SHORT
                self.position = -1.0
                self.entry_price = close_price
                self.tp_price, self.sl_price = self._calculate_targets(close_price, -1.0, raw_ktp, raw_ksl)
                reward -= 0.20 # Spread Penalty for new entry
                self.trade_duration = 0
            else:
                self.trade_duration += 1

        elif self.position == -1.0:
            # HARD LOCK: Neutral signals keep SHORT position open
            if direction_vol > 0.60:
                # Directional Flip to LONG: Close SHORT first, then open LONG
                exit_pips = (self.entry_price - close_price) / 0.10 - 2.0 # Deduct 2-pip spread
                reward += exit_pips / 10.0
                
                # Open new LONG
                self.position = 1.0
                self.entry_price = close_price
                self.tp_price, self.sl_price = self._calculate_targets(close_price, 1.0, raw_ktp, raw_ksl)
                reward -= 0.20 # Spread Penalty for new entry
                self.trade_duration = 0
            else:
                self.trade_duration += 1

        # ------------------------------------------------------------------
        # 3. ADVANCE TIMESTEP & STEP OUTPUT
        # ------------------------------------------------------------------
        self.current_step += 1
        if self.current_step >= self.num_samples - 1:
            terminated = True
            
        next_obs = self.df.iloc[self.current_step][self.feature_cols].values.astype(np.float32)
        info = {
            'position': self.position,
            'entry_price': self.entry_price,
            'tp_price': self.tp_price,
            'sl_price': self.sl_price
        }
        
        return next_obs, reward, terminated, truncated, info