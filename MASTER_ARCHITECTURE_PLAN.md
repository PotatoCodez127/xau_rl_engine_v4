# MASTER_ARCHITECTURE_PLAN.md

## 1. Project Overview: Tri-Brain XAUUSD Engine

This project is an institutional-grade, decoupled Deep Reinforcement Learning (DRL) trading engine. It abandons traditional flat-array indicators and heuristic logic in favor of spatial market structure, volatility-normalized zone distances, and hardware-segregated deployment.

## 2. Core Architecture: The Tri-Brain System

The system prevents cognitive overload by splitting the environment into three distinct neural processing engines:

1. **The Spatial Oracle (Master):** A Temporal Attention Network (Transformer). Processes asynchronous MTF (Multi-Timeframe) zone distances and chronological history to output a directional probability matrix.


2. **The Context Gatekeeper (Filter):** A Hidden Markov Model (HMM) for Regime Detection. Categorizes the market state (trending, ranging, chop) and blocks the Manager from trading during "terrible" structure, strictly enforcing a 1-5 trades/day limit.


3. **The Risk Manager (Slave):** A Soft Actor-Critic (SAC) Agent. Uses the Oracle's probabilities to determine volumetric sizing and asymmetric Stop-Loss/Take-Profit bounds.



## 3. Phase 1: Data Integrity & Stateful Feature Synthesis

We do not feed raw price to the network. We calculate the distance from the current price ($P_t$) to dynamic wick-to-body zones across the 15m, 30m, 1H, and 4H timeframes.

* **Volatility-Normalized Context:** Distances are scaled by rolling ATR to ensure the network is not blind to absolute variance changes.



$$D_{zone} = \frac{P_t - Z_{boundary}}{ATR_t}$$


* **Non-Linear Feature Squashing:** We use $\tanh$ soft-clippers to rigidly bound vector inputs, preventing gradient collapse during Black Swan outliers.



$$x_{squashed} = \tanh(D_{zone})$$


* **Fractional Differentiation:** Applied to achieve strict statistical stationarity without destroying long-term market memory.


* **Temporal Completion Meters:** To prevent look-ahead bias on developing higher timeframes, we feed the network the exact lifecycle of the current candle using cyclical encodings:

$$C_{\sin} = \sin\left(2\pi \cdot \frac{\text{Elapsed Minutes}}{\text{Total Minutes}}\right)$$



## 4. Phase 2: Asymmetric Spatial Tensors

To prevent Sequential Amnesia and computational bloat, the Oracle receives an asymmetric 3D tensor where lookback windows scale inversely to the timeframe:

* **15m Channel:** 128 steps (Sniper resolution).
* **30m Channel:** 64 steps.
* **1H Channel:** 32 steps.
* **4H Channel:** 16 steps (Macro boundaries).

## 5. Phase 3: Reward Engineering & Action Space

* **Asymmetric Action Bounds:** The SAC Manager's continuous outputs lock Take Profits to the next structural zone and Stop Losses beyond the opposing wick, enforcing positive statistical expectancy.


* **Episodic Terminal Checkpoints:** Rewards are granted only when a trade sequence terminates, eliminating execution paralysis caused by step-by-step decay penalties.


* **Focal Loss / PER:** Heavily penalizes missed breakout signals and liquidity sweeps to cure class imbalance.



## 6. Phase 4: Validation & Edge Deployment

* **Combinatorial Purged Cross-Validation (CPCV):** Rolling evaluation windows with strict Out-Of-Sample embargoes to mathematically prove robustness and mitigate look-ahead bias.


* **Execution Hardware Segregation:** Massive PyTorch tensor training is isolated to Cloud GPUs. Live execution relies on lightweight ONNX computation graphs deployed on a local CPU.


* **Stateful Pre-loading:** A background daemon fetches the last 64 hours of MT5 data to pre-warm the $\tanh$-squashed zone buffers, preventing cold-boot amnesia upon launch.



---

---

# PROJECT_MEMORY.md

### **Current Project State:**

* **Phase:** 0 - Project Initialization
* **Status:** Architecture designed, mathematical encoding verified, and repository structure planned.
* **Frameworks Chosen:** `gymnasium` (Environment API), `PyTorch` (Neural Networks), `ONNX` (Edge Inference), `MetaTrader5` (Data Ingestion).

### **Completed Tasks:**

* [x] Define Tri-Brain Architecture (Oracle, Gatekeeper, Manager).
* [x] Finalize Data Sanitization Mathematics (ATR Scaling, Tanh Squashing, Fractional Diff).
* [x] Establish Asymmetric MTF Tensor Dimensions (15m, 30m, 1H, 4H).
* [x] Design Temporal Completion Meter logic to prevent Higher-Timeframe Look-Ahead Bias.

### **Active Objective:**

Initialize the codebase and build the foundational data preprocessing pipeline.

### **Next Steps (Queue):**

1. Setup local directory structure and virtual environment.
2. Develop `data/build_features.py` to extract MT5 data, detect wick-to-body zones, and apply $\tanh$ squashing.
3. Develop the `gymnasium` custom environment wrapper.

---

### Directory Structure Blueprint

I recommend setting up your local repository exactly like this to keep the "Master-Slave" decoupling perfectly organized:

* `data/`
* `build_features.py` *(Your zone calculations and squashing)*
* `mt5_streamer.py` *(Live data ingestion)*


* `envs/`
* `xau_mtf_env.py` *(The Gymnasium environment)*


* `models/`
* `oracle_transformer.py` *(The Spatial Oracle)*
* `manager_sac.py` *(The Risk Manager)*
* `gatekeeper_hmm.py` *(The Context Gatekeeper)*


* `training/`
* `train_pipeline.py`
* `cpcv_validation.py` *(Combinatorial Purged Cross-Validation)*


* `deployment/`
* `export_to_onnx.py`
* `live_inference.py` *(The script running on your i5 machine)*


* `MASTER_ARCHITECTURE_PLAN.md`
* `PROJECT_MEMORY.md`
