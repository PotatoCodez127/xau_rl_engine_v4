
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
    UPDATE_FREQ = 100       # 🚀 Tuned to prevent excessive backprop overhead  
    UPDATE_PASSES = 2       # 🚀 Tuned down to prevent single-trajectory overfitting
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
            
            # 🚀 Cap the block to 20,000 steps to prevent single-epoch stagnation
            block_idx = block_idx[:20000]
            
            start_step = block_idx[0]
            max_steps = block_idx[-1]
            
            print(f"\n==================== CPCV PATH {fold_idx + 1}/{len(paths)} | EPOCH {epoch + 1}/{epochs_per_fold} ====================")
            print(f"▶️ Exploring Continuous Block: {len(block_idx)} candles")
            
            env = XAUMTFEnv(mtf_dict=mtf_dict, start_step=start_step, max_steps=max_steps)
            obs, _ = env.reset() 
            ep_reward = 0.0
            actor_losses, critic_losses, alpha_vals = [], [], []

            for step in range(len(block_idx)):
                # 🚀 Data is already in GPU. Only adding the batch dimension is required.
                tensor_obs = {k: v.unsqueeze(0) for k, v in obs.items()}

                # 🚀 Using inference_mode for faster continuous forward passes
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
