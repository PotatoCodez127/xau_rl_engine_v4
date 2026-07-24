```
xau_tribrain_engine/
├── data/
│   ├── __init__.py
│   ├── build_features.py         # MTF Wick-to-Body zone detection, ATR scaling, & tanh squashing
│   └── mt5_streamer.py          # Live MT5 data polling & pre-loading daemon
├── envs/
│   ├── __init__.py
│   └── xau_mtf_env.py           # Custom Gymnasium environment with Asymmetric Spatial Tensors
├── models/
│   ├── __init__.py
│   ├── oracle_transformer.py    # Spatial Oracle (Temporal Attention / Transformer)
│   ├── gatekeeper_hmm.py        # Context Gatekeeper (Regime Detection HMM)
│   └── manager_sac.py           # Risk Manager (Soft Actor-Critic with Asymmetric Action Space)
├── training/
│   ├── __init__.py
│   ├── cpcv_validation.py       # Combinatorial Purged Cross-Validation
│   └── train_pipeline.py        # Distributed PyTorch training loop
├── deployment/
│   ├── __init__.py
│   ├── export_to_onnx.py        # PyTorch to ONNX graph compiler
│   └── live_inference.py        # Edge ONNX runtime loop & WhatsApp live alert hooks
├── MASTER_ARCHITECTURE_PLAN.md
├── PROJECT_MEMORY.md
├── requirements.txt
└── .gitignore
```