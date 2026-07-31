import os
import sys
import joblib
import numpy as np
import pandas as pd
import onnxruntime as ort
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training.cpcv_validation import PurgedCombinatorialCV
from models.gatekeeper_hmm import ContextGatekeeper

def run_backtest():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(project_root, 'data', 'oos_holdout_tensor.pkl')
    oracle_path = os.path.join(project_root, 'deployment', 'oracle.onnx')
    actor_path = os.path.join(project_root, 'deployment', 'actor.onnx')
    gatekeeper_path = os.path.join(project_root, 'checkpoints', 'gatekeeper.pkl')
    
    # Fallback if the user downloaded gatekeeper directly to deployment/
    if not os.path.exists(gatekeeper_path):
        gatekeeper_path = os.path.join(project_root, 'deployment', 'gatekeeper.pkl')

    print("Loading Master Dataset & CPCV Splits...")
    mtf_dict = joblib.load(data_path)
    cpcv = PurgedCombinatorialCV(n_folds=6, n_test_folds=2)
    paths = list(cpcv.split(mtf_dict["15m"]))
    _, test_idx = paths[-1] 

    print("Initializing ONNX Inference Engines & Gatekeeper...")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    oracle_session = ort.InferenceSession(oracle_path, opts, providers=['CPUExecutionProvider'])
    actor_session = ort.InferenceSession(actor_path, opts, providers=['CPUExecutionProvider'])

    gatekeeper = ContextGatekeeper(n_components=3)
    if os.path.exists(gatekeeper_path):
        gatekeeper.load_model(gatekeeper_path)
        print("✅ Gatekeeper HMM restored and operational.")
    else:
        print("⚠️ WARNING: Gatekeeper HMM not found. Running un-gated.")

    equity = 10000.0 
    equity_curve = [equity]
    trade_log = []
    
    position, entry_price, unrealized_pnl, cooldown, bars_in_trade = 0.0, 0.0, 0.0, 0, 0

    # ==============================================================================
    # CALIBRATED HYPERPARAMETERS
    # ==============================================================================
    PIP_SCALAR = 0.10
    SPREAD_PIPS = 2.0
    CONVICTION_THRESHOLD = 0.55  
    EXIT_THRESHOLD = 0.20        # 🚀 The Escape Hatch boundary
    GATEKEEPER_THRESHOLD = 0.60  # Minimum Oracle probability to pass the veto
    POST_TRADE_COOLDOWN = 12     
    MAX_HOLD_BARS = 32           

    print(f"Starting Walk-Forward Simulation ({len(test_idx)} steps)...")

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
                bars_in_trade = 0
                
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
        
        raw_action = action_outputs[0][0]
        direction_vol = raw_action[0]
        k_tp = (raw_action[1] + 1.0) / 2.0
        k_sl = (raw_action[2] + 1.0) / 2.0

        # ==========================================
        # GATEKEEPER REGIME FILTER
        # ==========================================
        is_authorized = True
        if gatekeeper.is_fitted:
            # Extract recent 15m volatility/macro features (Index 0, 1)
            macro_features = m15[-1, :2].reshape(1, -1)
            current_regime = gatekeeper.predict_regime(macro_features)
            is_authorized = gatekeeper.authorize_execution(current_regime, oracle_probs[0], conviction_threshold=GATEKEEPER_THRESHOLD)

        # If Gatekeeper vetoes, zero out the conviction to stay flat or trigger Escape Hatch
        if not is_authorized:
            direction_vol = 0.0

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
        time_stop_hit = (prev_pos != 0.0) and (bars_in_trade >= MAX_HOLD_BARS)

        # ==========================================
        # THE HARD LOCK EXECUTION LOGIC + ESCAPE HATCH
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
                elif direction_vol < EXIT_THRESHOLD: target_pos = 0.0  # Safe Exit
            elif prev_pos < 0.0:
                if direction_vol > CONVICTION_THRESHOLD: target_pos = 1.0
                elif direction_vol > -EXIT_THRESHOLD: target_pos = 0.0 # Safe Exit

        # ==========================================
        # TRADE CLOSURE EVALUATION
        # ==========================================
        trade_closed, reason = False, ""
        
        if sl_hit:
            trade_closed, reason = True, "Stop Loss"
            target_pos, cooldown = 0.0, POST_TRADE_COOLDOWN
        elif tp_hit:
            trade_closed, reason = True, "Take Profit"
            target_pos, cooldown = 0.0, POST_TRADE_COOLDOWN
        elif time_stop_hit:
            trade_closed, reason = True, "Time Stop"
            target_pos, cooldown = 0.0, POST_TRADE_COOLDOWN
        elif prev_pos != 0.0 and target_pos == 0.0:
            trade_closed, reason = True, "Escape Hatch Exit"
            cooldown = POST_TRADE_COOLDOWN
        elif prev_pos != 0.0 and target_pos != prev_pos:
            trade_closed, reason = True, "Network Flip"
            cooldown = POST_TRADE_COOLDOWN

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
        # UPDATE ENTRY PRICE AND TIME IN TRADE
        # ==========================================
        if target_pos != 0.0 and target_pos != prev_pos:
            entry_price = current_price
            bars_in_trade = 1
        elif target_pos != 0.0 and target_pos == prev_pos:
            bars_in_trade += 1
        else:
            bars_in_trade = 0
            
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