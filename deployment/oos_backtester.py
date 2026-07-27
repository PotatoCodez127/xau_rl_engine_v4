import numpy as np
import pandas as pd


class HardLockOOSBacktester:
    """
    Out-of-Sample Backtester (Hard Lock Enforced)
    Evaluates trained ONNX models against real-world XAUUSD physics.
    """
    def __init__(self, df: pd.DataFrame, spread_pips: float = 2.0, pip_value_usd: float = 10.0):
        self.df = df.reset_index(drop=True)
        self.spread_pips = spread_pips
        self.pip_value_usd = pip_value_usd # $10 per pip per Standard Lot
        
    def _calculate_targets(self, current_price: float, position: float, raw_ktp: float, raw_ksl: float):
        tp_pips = 40.0 + ((raw_ktp + 1.0) / 2.0) * 60.0
        sl_pips = 20.0 + ((raw_ksl + 1.0) / 2.0) * 30.0
        
        tp_delta = tp_pips * 0.10
        sl_delta = sl_pips * 0.10
        
        if position == 1.0:
            return current_price + tp_delta, current_price - sl_delta
        else:
            return current_price - tp_delta, current_price + sl_delta

    def run_backtest(self, model_predictions: np.ndarray) -> dict:
        """
        model_predictions: Array of shape (N, 3) containing [direction_vol, k_tp, k_sl]
        """
        position = 0.0
        entry_price = 0.0
        tp_price = 0.0
        sl_price = 0.0
        entry_idx = 0
        
        trades = []
        equity_curve = [0.0]
        cumulative_pnl_usd = 0.0

        for i in range(len(self.df) - 1):
            row = self.df.iloc[i]
            high_price, low_price, close_price = row['high'], row['low'], row['close']
            time_stamp = row.get('time', i)
            
            action = model_predictions[i]
            direction_vol, raw_ktp, raw_ksl = action[0], action[1], action[2]
            
            # 1. EVALUATE EXITS (TP / SL) FIRST
            if position == 1.0: # LONG
                if high_price >= tp_price:
                    pips = (tp_price - entry_price) / 0.10 - self.spread_pips
                    pnl_usd = pips * self.pip_value_usd
                    cumulative_pnl_usd += pnl_usd
                    trades.append({
                        'entry_idx': entry_idx, 'exit_idx': i, 'type': 'LONG',
                        'entry_price': entry_price, 'exit_price': tp_price,
                        'reason': 'TP', 'pips': pips, 'pnl_usd': pnl_usd
                    })
                    position = 0.0
                elif low_price <= sl_price:
                    pips = (sl_price - entry_price) / 0.10 - self.spread_pips
                    pnl_usd = pips * self.pip_value_usd
                    cumulative_pnl_usd += pnl_usd
                    trades.append({
                        'entry_idx': entry_idx, 'exit_idx': i, 'type': 'LONG',
                        'entry_price': entry_price, 'exit_price': sl_price,
                        'reason': 'SL', 'pips': pips, 'pnl_usd': pnl_usd
                    })
                    position = 0.0

            elif position == -1.0: # SHORT
                if low_price <= tp_price:
                    pips = (entry_price - tp_price) / 0.10 - self.spread_pips
                    pnl_usd = pips * self.pip_value_usd
                    cumulative_pnl_usd += pnl_usd
                    trades.append({
                        'entry_idx': entry_idx, 'exit_idx': i, 'type': 'SHORT',
                        'entry_price': entry_price, 'exit_price': tp_price,
                        'reason': 'TP', 'pips': pips, 'pnl_usd': pnl_usd
                    })
                    position = 0.0
                elif high_price >= sl_price:
                    pips = (entry_price - sl_price) / 0.10 - self.spread_pips
                    pnl_usd = pips * self.pip_value_usd
                    cumulative_pnl_usd += pnl_usd
                    trades.append({
                        'entry_idx': entry_idx, 'exit_idx': i, 'type': 'SHORT',
                        'entry_price': entry_price, 'exit_price': sl_price,
                        'reason': 'SL', 'pips': pips, 'pnl_usd': pnl_usd
                    })
                    position = 0.0

            # 2. EVALUATE HARD LOCK DIRECTION & FLIPS
            if position == 0.0:
                if direction_vol > 0.60:
                    position = 1.0
                    entry_price = close_price
                    entry_idx = i
                    tp_price, sl_price = self._calculate_targets(close_price, 1.0, raw_ktp, raw_ksl)
                elif direction_vol < -0.60:
                    position = -1.0
                    entry_price = close_price
                    entry_idx = i
                    tp_price, sl_price = self._calculate_targets(close_price, -1.0, raw_ktp, raw_ksl)

            elif position == 1.0 and direction_vol < -0.60: # FLIP LONG TO SHORT
                pips = (close_price - entry_price) / 0.10 - self.spread_pips
                pnl_usd = pips * self.pip_value_usd
                cumulative_pnl_usd += pnl_usd
                trades.append({
                    'entry_idx': entry_idx, 'exit_idx': i, 'type': 'LONG',
                    'entry_price': entry_price, 'exit_price': close_price,
                    'reason': 'FLIP', 'pips': pips, 'pnl_usd': pnl_usd
                })
                # Enter New SHORT
                position = -1.0
                entry_price = close_price
                entry_idx = i
                tp_price, sl_price = self._calculate_targets(close_price, -1.0, raw_ktp, raw_ksl)

            elif position == -1.0 and direction_vol > 0.60: # FLIP SHORT TO LONG
                pips = (entry_price - close_price) / 0.10 - self.spread_pips
                pnl_usd = pips * self.pip_value_usd
                cumulative_pnl_usd += pnl_usd
                trades.append({
                    'entry_idx': entry_idx, 'exit_idx': i, 'type': 'SHORT',
                    'entry_price': entry_price, 'exit_price': close_price,
                    'reason': 'FLIP', 'pips': pips, 'pnl_usd': pnl_usd
                })
                # Enter New LONG
                position = 1.0
                entry_price = close_price
                entry_idx = i
                tp_price, sl_price = self._calculate_targets(close_price, 1.0, raw_ktp, raw_ksl)

            equity_curve.append(cumulative_pnl_usd)

        # Metrics calculation
        trades_df = pd.DataFrame(trades)
        total_trades = len(trades_df)
        
        if total_trades > 0:
            win_rate = (trades_df['pnl_usd'] > 0).mean() * 100.0
            gross_profit = trades_df[trades_df['pnl_usd'] > 0]['pnl_usd'].sum()
            gross_loss = abs(trades_df[trades_df['pnl_usd'] < 0]['pnl_usd'].sum())
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.nan
            total_pnl_usd = trades_df['pnl_usd'].sum()
        else:
            win_rate, profit_factor, total_pnl_usd = 0.0, 0.0, 0.0

        return {
            'total_trades': total_trades,
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'total_pnl_usd': total_pnl_usd,
            'trades_df': trades_df,
            'equity_curve': equity_curve
        }