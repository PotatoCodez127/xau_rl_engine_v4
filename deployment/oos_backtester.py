import os
import sys
import joblib
import numpy as np
import pandas as pd
import onnxruntime as ort
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training.cpcv_validation import PurgedCombinatorialCV

def run_backtest():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(project_root, 'data', 'oos_holdout_tensor.pkl')
    oracle_path = os.path.join(project_root, 'deployment', 'oracle.onnx')
    actor_path = os.path.join(project_root, 'deployment', 'actor.onnx')

    print("📦 Loading Master Dataset & CPCV Splits...")
    mtf_dict = joblib.load(data_path)
    cpcv = PurgedCombinatorialCV(n_folds=6, n_test_folds=2)
    paths = list(cpcv.split(mtf_dict["15m"]))
    _, test_idx = paths[-1] 

    print("🔥 Initializing ONNX Inference Engines...")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    oracle_session = ort.InferenceSession(oracle_path, opts, providers=['CPUExecutionProvider'])
    actor_session = ort.InferenceSession(actor_path, opts, providers=['CPUExecutionProvider'])

    equity = 10000.0 
    equity_curve = [equity]
    trade_log = []
    
    position, entry_price, unrealized_pnl, cooldown = 0.0, 0.0, 0.0, 0
    # Temporarily drop the conviction threshold so the untrained network is forced to trade
    CONVICTION_THRESHOLD, PIP_SCALAR, SPREAD_PIPS = 0.00, 0.10, 2.0

    print(f"🚀 Starting Walk-Forward Simulation ({len(test_idx)} steps)...")

    # FIX 1: Timeframe alignment logic mirrored from V7.3 training environment
    def slice_tf(tf, length, current_step):
        data = mtf_dict[tf]
        if hasattr(data, "values"): data = data.values
        
        if tf == "15m": idx = current_step
        elif tf == "30m": idx = current_step // 2
        elif tf == "1H": idx = current_step // 4
        elif tf == "4H": idx = current_step // 16
        else: idx = current_step
        
        idx = min(idx, len(data))
        
        if idx < length:
            # FIX 2: Dynamic feature column width to prevent vstack crashes
            pad = np.zeros((length - idx, data.shape[1]), dtype=np.float32)
            return np.vstack([pad, data[:idx]]).astype(np.float32)
        return data[idx - length : idx].astype(np.float32)

    for i in range(128, len(test_idx)):
        idx = test_idx[i]
        
        # Detect CPCV fold gaps and force-close phantom trades
        if i > 128 and idx != test_idx[i-1] + 1:
            if position != 0.0:
                realized = unrealized_pnl * 100.0
                equity += realized
                trade_log.append({
                    "Step": test_idx[i-1], "Direction": "LONG" if position > 0 else "SHORT",
                    "PnL": round(realized, 2), "Pips": round(unrealized_pnl / PIP_SCALAR, 2),
                    "Reason": "Fold Gap Close", "Equity": round(equity, 2)
                })
                unrealized_pnl = 0.0
                position = 0.0
                cooldown = 0
                
        data_15m = mtf_dict["15m"].values if hasattr(mtf_dict["15m"], "iloc") else mtf_dict["15m"]
        current_price = data_15m[idx, 3] 

        m15, m30 = slice_tf("15m", 128, idx), slice_tf("30m", 64, idx)
        h1, h4 = slice_tf("1H", 32, idx), slice_tf("4H", 16, idx)

        state_vec = np.array([position, 1.0, unrealized_pnl, cooldown], dtype=np.float32)

        inputs_oracle = {
            "15m": np.expand_dims(m15, axis=0).astype(np.float32),
            "30m": np.expand_dims(m30, axis=0).astype(np.float32),
            "1H": np.expand_dims(h1, axis=0).astype(np.float32),
            "4H": np.expand_dims(h4, axis=0).astype(np.float32),
            "state": np.expand_dims(state_vec, axis=0).astype(np.float32)
        }
        
        oracle_probs = oracle_session.run(None, inputs_oracle)[0]
        action_outputs = actor_session.run(None, {"oracle_probs": oracle_probs, "state": inputs_oracle["state"]})
        
        # FIX 3: Re-applied np.tanh() because the ONNX wrapper exports the raw mean, not the squashed action
        raw_action = action_outputs[0][0]
        direction_vol = np.tanh(raw_action[0])
        k_tp = (np.tanh(raw_action[1]) + 1.0) / 2.0
        k_sl = (np.tanh(raw_action[2]) + 1.0) / 2.0

        # ==========================================
        # ENGINE PHYSICS
        # ==========================================
        prev_pos = position
        
        if prev_pos != 0.0:
            raw_pnl = prev_pos * (current_price - entry_price)
            unrealized_pnl = raw_pnl - (SPREAD_PIPS * PIP_SCALAR)
        else:
            unrealized_pnl = 0.0

        sl_pips, tp_pips = 20.0 + (k_sl * 30.0), 40.0 + (k_tp * 60.0)
        target_sl, target_tp = -1.0 * (sl_pips * PIP_SCALAR), tp_pips * PIP_SCALAR

        sl_hit = (prev_pos != 0.0) and (unrealized_pnl <= target_sl)
        tp_hit = (prev_pos != 0.0) and (unrealized_pnl >= target_tp)

        # ==========================================
        # THE HARD LOCK EXECUTION LOGIC
        # ==========================================
        target_pos = prev_pos
        if cooldown > 0:
            cooldown -= 1
        else:
            if prev_pos == 0.0:
                if direction_vol > CONVICTION_THRESHOLD: target_pos = 1.0
                elif direction_vol < -CONVICTION_THRESHOLD: target_pos = -1.0
            elif prev_pos > 0.0:
                if direction_vol < -CONVICTION_THRESHOLD: target_pos = -1.0
            elif prev_pos < 0.0:
                if direction_vol > CONVICTION_THRESHOLD: target_pos = 1.0

        # ==========================================
        # TRADE CLOSURE EVALUATION
        # ==========================================
        trade_closed, reason = False, ""
        if sl_hit:
            trade_closed, reason = True, "Stop Loss"
            target_pos, cooldown = 0.0, 5
        elif tp_hit:
            trade_closed, reason = True, "Take Profit"
            target_pos = 0.0
        elif prev_pos != 0.0 and target_pos != prev_pos:
            trade_closed, reason = True, "Network Flip"

        if trade_closed:
            realized = unrealized_pnl * 100.0
            equity += realized
            pips_captured = unrealized_pnl / PIP_SCALAR
            
            trade_log.append({
                "Step": idx, "Direction": "LONG" if prev_pos > 0 else "SHORT",
                "PnL": round(realized, 2), "Pips": round(pips_captured, 2),
                "Reason": reason, "Equity": round(equity, 2)
            })
            unrealized_pnl = 0.0

        # ==========================================
        # UPDATE ENTRY PRICE
        # ==========================================
        if target_pos != 0.0 and target_pos != prev_pos:
            entry_price = current_price
            
        position = target_pos
        equity_curve.append(equity)

    # Force Close any open trades at simulation end
    if position != 0.0:
        realized = unrealized_pnl * 100.0
        equity += realized
        trade_log.append({
            "Step": "END", "Direction": "LONG" if position > 0 else "SHORT",
            "PnL": round(realized, 2), "Pips": round(unrealized_pnl / PIP_SCALAR, 2),
            "Reason": "End of Backtest", "Equity": round(equity, 2)
        })

    log_df = pd.DataFrame(trade_log)
    log_df.to_csv("traders_log.csv", index=False)
    
    wins = len(log_df[log_df["PnL"] > 0]) if not log_df.empty else 0
    total_trades = len(log_df)
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    
    gross_profit = log_df[log_df["PnL"] > 0]["PnL"].sum() if not log_df.empty else 0
    gross_loss = abs(log_df[log_df["PnL"] <= 0]["PnL"].sum()) if not log_df.empty else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    print("\n" + "="*40)
    print("  OUT-OF-SAMPLE TEAR SHEET")
    print("="*40)
    print(f"Total Trades:   {total_trades}")
    print(f"Win Rate:       {win_rate:.2f}%")
    print(f"Profit Factor:  {profit_factor:.2f}")
    print(f"Final Equity:   ${equity:.2f}")
    print("="*40)

    plt.plot(equity_curve)
    plt.title("OOS Equity Curve - Tri-Brain System")
    plt.ylabel("Account Balance ($)")
    plt.xlabel("Steps")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    run_backtest()