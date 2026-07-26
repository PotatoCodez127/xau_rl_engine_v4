import os
import sys
import joblib
import numpy as np
import pandas as pd
import onnxruntime as ort
import matplotlib.pyplot as plt

# Ensure imports work from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training.cpcv_validation import PurgedCombinatorialCV

def run_backtest():
    # 1. Setup Paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(project_root, 'data', 'oos_holdout_tensor.pkl')
    oracle_path = os.path.join(project_root, 'deployment', 'oracle.onnx')
    actor_path = os.path.join(project_root, 'deployment', 'actor.onnx')
    gatekeeper_path = os.path.join(project_root, 'deployment', 'gatekeeper.pkl')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Missing {data_path}. Please download it from Google Drive.")

    print("📊 Loading Master Dataset & CPCV Splits...")
    mtf_dict = joblib.load(data_path)
    
    # 2. Recreate the CPCV split to isolate Path 15's exact Out-Of-Sample Test Set
    cpcv = PurgedCombinatorialCV(n_folds=6, n_test_folds=2)
    paths = list(cpcv.split(mtf_dict["15m"]))
    _, test_idx = paths[-1] # Path 15

    # 3. Load ONNX Engines
    print("⚡ Initializing ONNX Inference Engines...")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    oracle_session = ort.InferenceSession(oracle_path, opts, providers=['CPUExecutionProvider'])
    actor_session = ort.InferenceSession(actor_path, opts, providers=['CPUExecutionProvider'])
    
    gatekeeper = None
    if os.path.exists(gatekeeper_path):
        gatekeeper = joblib.load(gatekeeper_path)

    # 4. Simulation Variables
    equity = 10000.0 # Starting Balance ($10,000)
    equity_curve = [equity]
    trade_log = []
    
    position = 0.0 # -1.0 (Short), 0.0 (Flat), 1.0 (Long)
    entry_price = 0.0
    unrealized_pnl = 0.0
    cooldown = 0
    
    CONVICTION_THRESHOLD = 0.60 # Requires 60% confidence to trigger trades
    
    print(f"🚀 Starting Deterministic Walk-Forward Simulation ({len(test_idx)} steps)...")

    # Safe Slicing Helper (Prevents Out-Of-Bounds Errors on Higher Timeframes)
    def slice_tf(tf, length, current_step):
        data = mtf_dict[tf]
        if hasattr(data, "values"):
            data = data.values
            
        idx = min(current_step, len(data))
        
        if idx < length:
            pad = np.zeros((length - idx, 11), dtype=np.float32)
            slice_data = data[:idx]
            return np.vstack([pad, slice_data]).astype(np.float32)
        
        return data[idx - length : idx].astype(np.float32)

    # 5. Step-by-Step Backtest Loop
    for i in range(128, len(test_idx)):
        idx = test_idx[i]
        
        # Get Current Price
        data_15m = mtf_dict["15m"].values if hasattr(mtf_dict["15m"], "iloc") else mtf_dict["15m"]
        current_price = data_15m[idx, 3] # Close price
        
        # Safely Slice Tensors
        m15 = slice_tf("15m", 128, idx)
        m30 = slice_tf("30m", 64, idx)
        h1 = slice_tf("1H", 32, idx)
        h4 = slice_tf("4H", 16, idx)
        
        state_vec = np.array([position, 1.0, unrealized_pnl, cooldown], dtype=np.float32)

        # ONNX Inference
        inputs_oracle = {
            "15m": np.expand_dims(m15, axis=0).astype(np.float32),
            "30m": np.expand_dims(m30, axis=0).astype(np.float32),
            "1H": np.expand_dims(h1, axis=0).astype(np.float32),
            "4H": np.expand_dims(h4, axis=0).astype(np.float32),
            "state": np.expand_dims(state_vec, axis=0).astype(np.float32)
        }
        
        oracle_probs = oracle_session.run(None, inputs_oracle)[0]
        action_outputs = actor_session.run(None, {"oracle_probs": oracle_probs, "state": inputs_oracle["state"]})
        
        # Squash raw ONNX logits via Tanh
        raw_action = action_outputs[0][0]
        direction_vol = np.tanh(raw_action[0])
        k_tp = (np.tanh(raw_action[1]) + 1.0) / 2.0
        k_sl = (np.tanh(raw_action[2]) + 1.0) / 2.0

        # Gatekeeper Veto
        if gatekeeper:
            macro_features = m15[-1, :2].reshape(1, -1)
            if gatekeeper.predict(macro_features)[0] == 2:
                direction_vol = 0.0

        prev_pos = position

        # ==========================================
        # INTRADAY EXECUTION LOGIC (NO HFT)
        # ==========================================
        if cooldown > 0:
            cooldown -= 1
        else:
            # 1. If currently FLAT, look for strong conviction to enter
            if position == 0.0:
                if direction_vol > CONVICTION_THRESHOLD:
                    position = 1.0  # Full Long
                    entry_price = current_price
                elif direction_vol < -CONVICTION_THRESHOLD:
                    position = -1.0 # Full Short
                    entry_price = current_price
                    
            # 2. If currently LONG, exit or flip only on strong negative signal
            elif position > 0.0:
                if direction_vol < -CONVICTION_THRESHOLD:
                    position = -1.0 # Flip to Short
                    entry_price = current_price
                elif direction_vol < 0.0:
                    position = 0.0  # Flat exit
                    
            # 3. If currently SHORT, exit or flip only on strong positive signal
            elif position < 0.0:
                if direction_vol > CONVICTION_THRESHOLD:
                    position = 1.0  # Flip to Long
                    entry_price = current_price
                elif direction_vol > 0.0:
                    position = 0.0  # Flat exit

       # ==========================================
        # REAL-WORLD XAUUSD PIP CALIBRATION
        # Rule: $10.00 price diff = 100 pips (1 pip = $0.10)
        # ==========================================
        PIP_SCALAR = 0.10
        SPREAD_PIPS = 2.0  # Standard 2 pip spread on Gold
        
        # Friction & PnL
        pos_delta = abs(position - prev_pos)
        transaction_cost_price = pos_delta * (SPREAD_PIPS * PIP_SCALAR) 
        
        # FIX: Calculate PnL based on the position we were just holding!
        if prev_pos != 0.0:
            step_pnl = prev_pos * (current_price - entry_price)
            unrealized_pnl = step_pnl - transaction_cost_price
        else:
            unrealized_pnl = 0.0

        # ==========================================
        # DYNAMIC TP & SL IN REAL PIPS
        # ==========================================
        sl_pips = 20.0 + (k_sl * 30.0)
        target_sl_price_dist = -1.0 * (sl_pips * PIP_SCALAR)

        tp_pips = 40.0 + (k_tp * 60.0)
        target_tp_price_dist = tp_pips * PIP_SCALAR

        # FIX: Check against prev_pos so SL/TP triggers correctly
        sl_hit = (prev_pos != 0.0) and (unrealized_pnl <= target_sl_price_dist)
        tp_hit = (prev_pos != 0.0) and (unrealized_pnl >= target_tp_price_dist)
        
        trade_closed = False
        reason = ""

        if position == 0.0 and prev_pos != 0.0:
            trade_closed, reason = True, "Network Exit"
        elif sl_hit:
            trade_closed, reason = True, "Stop Loss"
            position, cooldown = 0.0, 5
        elif tp_hit:
            trade_closed, reason = True, "Take Profit"
            position = 0.0
            
        if trade_closed:
            # 1 Standard Lot of XAUUSD (100 oz)
            # $1.00 price movement = $100 Profit/Loss
            realized = unrealized_pnl * 100.0
            equity += realized
            
            # Log exact pip distances for analytics
            pips_captured = unrealized_pnl / PIP_SCALAR
            
            trade_log.append({
                "Step": idx, 
                "Direction": "LONG" if prev_pos > 0 else "SHORT",
                "PnL": round(realized, 2),
                "Pips": round(pips_captured, 2),
                "Reason": reason,
                "Equity": round(equity, 2)
            })
            unrealized_pnl = 0.0
            
        equity_curve.append(equity)

    # 6. Tear Sheet & Logging
    log_df = pd.DataFrame(trade_log)
    log_df.to_csv("traders_log.csv", index=False)
    
    wins = len(log_df[log_df["PnL"] > 0]) if not log_df.empty else 0
    losses = len(log_df[log_df["PnL"] <= 0]) if not log_df.empty else 0
    total_trades = len(log_df)
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    gross_profit = log_df[log_df["PnL"] > 0]["PnL"].sum() if not log_df.empty else 0
    gross_loss = abs(log_df[log_df["PnL"] <= 0]["PnL"].sum()) if not log_df.empty else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    print("\n" + "="*40)
    print("📈 OUT-OF-SAMPLE TEAR SHEET (PATH 15)")
    print("="*40)
    print(f"Total Trades:   {total_trades}")
    print(f"Win Rate:       {win_rate:.2f}%")
    print(f"Profit Factor:  {profit_factor:.2f}")
    print(f"Final Equity:   ${equity:.2f}")
    print("="*40)
    print("✅ Full log saved to 'traders_log.csv'")

    plt.plot(equity_curve)
    plt.title("OOS Equity Curve (Path 15 - Intraday Mode)")
    plt.ylabel("Account Balance ($)")
    plt.xlabel("Steps")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    run_backtest()