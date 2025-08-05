# Project Chimera: Dense vs Mixture-of-Experts (MoE) Transformer Study

- Author: Cory Janowski
- Date: 2025-08-05

Abstract
This repository investigates architectural and computational trade-offs between dense Transformer decoders and Mixture-of-Experts (MoE) variants in autoregressive language modeling. We implement a nano-GPT style dense baseline and introduce a modular MoE interface with top-k gating and configurable expert ensembles. My goals are to: (1) establish a strong, reproducible dense baseline; (2) design an extensible MoE layer with clear interfaces for experts, gating, and routing; (3) evaluate training stability, loss convergence, and performance efficiency; and (4) provide open, testable code for further research on routing quality, auxiliary balancing objectives, and dispatch efficiency. Preliminary results are forthcoming; this document details the methodology and experimental protocol to ensure reproducibility and facilitate future extensions.

1. Introduction
Large-scale language models achieve strong performance but often require prohibitive compute. MoE architectures reduce per-token compute by routing tokens to a small subset of experts, potentially improving throughput and scaling efficiency. However, MoE introduces new challenges: routing stability, load balancing, dispatch overhead, and expert under/over-utilization. This project builds a rigorous experimental backbone to assess when and how MoE outperforms dense models at similar quality or latency.

Contributions:
- Dense baseline: nano-GPT style decoder with unit tests and training loop.
- MoE interface: modular design isolating experts, gating, and layer combination, with feature-flag integration.
- Reproducible pipeline: data preprocessing for AG News, CLI-driven training, device and seed utilities, tests for core functionality.
- Evaluation plan: metrics for model quality and system efficiency, plus ablations on top-k, expert count, and routing noise.

2. Related Work
Dense Transformers: The standard decoder-only Transformer architecture underpins autoregressive modeling across domains.
Mixture-of-Experts: Switch Transformer, GShard, and later works demonstrate sparse activation via routed experts, enabling larger model capacity without linearly increasing per-token FLOPs.
Routing and Load Balancing: Top-k routing with auxiliary balancing losses addresses collapsed utilization. Variants (noisy gating, soft routing, Sinkhorn routing) explore improved token-to-expert assignments.
System Optimizations: Expert parallelism and efficient gather/scatter implementations (e.g., in DeepSpeed or FSDP-based setups) reduce dispatch costs, enabling practical training.

References placeholders:
- Fedus et al., “Switch Transformers”
- Lepikhin et al., “GShard”
- Shazeer et al., “Outrageously Large Neural Networks”
- Recent MoE routing and load balancing advances

3. Methods
3.1 Dense Baseline
We implement a GPT-style Transformer decoder with:
- Token embeddings with weight tying to the output head.
- Sine/cos positional encoding.
- Pre-norm residual blocks comprising MultiheadAttention and MLP FFN.
- Causal attention masking and key_padding support.

Core components:
- Config and model: [src/project_chimera/baseline.py](src/project_chimera/baseline.py:10), GPTDecoder, TransformerBlock
- Training utilities: [src/project_chimera/trainer.py](src/project_chimera/trainer.py:1), BaselineTrainer with AdamW, AMP, gradient clipping
- Data preprocessing: [src/project_chimera/data/preprocess.py](src/project_chimera/data/preprocess.py:1) with GPT-2 tokenizer, lowercase optional, padding/truncation
- Device and seeding: [src/project_chimera/utils/device.py](src/project_chimera/utils/device.py:1), [src/project_chimera/utils/repro.py](src/project_chimera/utils/repro.py:1)
- Tests: [tests/test_model.py](tests/test_model.py:1), [tests/test_data.py](tests/test_data.py:1), [tests/test_device_and_seed.py](tests/test_device_and_seed.py:1)

3.2 MoE Architecture
The MoE layer is designed as an FFN replacement with clear separation of concerns:
- Experts: Parameterized FFN experts with configurable activation and dropout.
  - [src/project_chimera/moe/experts.py](src/project_chimera/moe/experts.py:1): ExpertConfig, FFNExpert, ExpertParallel
- Gating: Top-k gating to produce expert indices and weights per token.
  - [src/project_chimera/moe/gating.py](src/project_chimera/moe/gating.py:1): GatingConfig, TopKGating
- Layer: Combines experts and gating; returns output and routing diagnostics.
  - [src/project_chimera/moe/layer.py](src/project_chimera/moe/layer.py:1): MoEConfig, MoELayer, MoEFFNWrapper

Integration:
- Feature flag in BaselineConfig enables MoE FFN in each TransformerBlock.
- When disabled, the model behaves identically to the dense baseline.

Config flags in BaselineConfig:
- moe_enabled: bool
- moe_n_experts: int
- moe_top_k: int
- moe_activation: str
- moe_noisy_gate: bool

See integration points in [src/project_chimera/baseline.py](src/project_chimera/baseline.py:45).

3.3 Design Choices
- Pre-norm blocks for stability under both dense and MoE.
- Weight tying to reduce softmax head parameters, matching GPT-style conventions.
- Top-k gating stub prioritizes API clarity; dispatch efficiency to be added.
- AMP enabled optionally for CUDA to evaluate training throughput.

4. Experiments
4.1 Datasets
Primary: AG News (subset) as a compact, well-known dataset to validate end-to-end correctness and performance assumptions. Data is handled by scripts and loaders with tokenizer alignment.

4.2 Experimental Setup
- Dense Baseline:
  - d_model {128, 256}, n_layers {2, 4}, n_heads {2, 4}, ff_dim {4x d_model}, dropout {0.0, 0.1}
- MoE Variants:
  - n_experts {4, 8}, top_k {1, 2}, activation {"gelu"}, noisy_gate {False, True}
- Optimization: AdamW (lr 3e-4 default), gradient clipping, mixed precision on CUDA.
- Logging: periodic training and validation loss reporting, final validation loss.
- Compute: single GPU or CPU fallback.

4.3 Metrics
- Quality: training loss, validation perplexity (derived), stability (finite loss, no NaNs).
- Efficiency: steps/sec, memory footprint, and inference throughput (to be added).
- Routing Diagnostics: expert utilization histograms, top-k distributions (to be added).

4.4 Ablation Studies (Planned)
- Effect of top-k on loss and utilization.
- Number of experts vs. compute/quality trade-off.
- Noisy gating impact on load balancing and stability.
- Dispatch efficiency via gather/scatter vs naive all-expert compute.

5. Results
Placeholder: Results will include dense baseline curves, MoE comparisons at compute-matched settings, and routing diagnostics. We will report:
- Training and validation loss curves with confidence intervals (multiple seeds).
- Expert utilization histograms; load balancing metrics.
- Throughput and memory measurements across configurations.

6. Discussion
We anticipate that MoE models can match or exceed dense baseline quality at reduced per-token compute, but only with careful routing and load balancing. The naive soft-combine implementation serves as a correctness oracle but is inefficient. The central question is identifying the regimes (dataset size, model capacity, routing k) where MoE’s sparsity yields practical benefits without sacrificing convergence stability.

7. Limitations
- Current MoE layer computes all experts, then combines by top-k weights — not efficient and intended as a functional baseline.
- No distributed expert parallelism yet; single-process prototype.
- Load balancing / auxiliary objectives not yet implemented.
- Limited dataset scope at present; broader corpora planned.

8. Future Work
- Implement efficient token dispatch with capacity constraints and batch-wise gather/scatter.
- Add auxiliary load balancing losses and scheduling.
- Integrate FSDP/DeepSpeed expert parallelism for scaling.
- Expand datasets and conduct large-scale comparisons.
- Explore advanced routing (noisy top-k variants, Sinkhorn-based, task-aware routing).

9. Reproducibility
Environment
- Python 3.10+
- Install dependencies:
  - pip install -e .[dev]

Data
- AG News download handled by:
  - [scripts/download_ag_news.py](scripts/download_ag_news.py:1)
  - See [data/README.md](data/README.md:1) for dataset notes

Preprocessing
- Tokenizer: GPT-2 via HuggingFace; lowercase optional.
- Settings: max length, padding/truncation controlled by TokenizerConfig in [src/project_chimera/data/preprocess.py](src/project_chimera/data/preprocess.py:1).

Training (Dense Baseline)
- Command example:
  - python -u scripts/train_baseline_smoke.py --model_name gpt2 --lowercase --max_length 128 --batch_size 16 --train_limit 2048 --val_limit 512 --d_model 256 --n_layers 4 --n_heads 4 --ff_dim 1024 --dropout 0.1 --max_steps 200 --eval_every 100 --log_every 20 --lr 3e-4

Training (Enable MoE)
- Set flags via BaselineConfig:
  - moe_enabled=True, moe_n_experts=4, moe_top_k=1, moe_activation="gelu", moe_noisy_gate=False
- Example (pseudo-code):
  - cfg = BaselineConfig(vocab_size=tokenizer.vocab_size, d_model=256, n_layers=4, n_heads=4, ff_dim=1024, dropout=0.1, max_seq_len=128, tie_weights=True, moe_enabled=True, moe_n_experts=4, moe_top_k=1)
  - model = GPTDecoder(cfg)
  - Use BaselineTrainer as usual.

Testing
- Run unit tests:
  - pytest -q
- Tests include data pipeline smoke checks and model forward shape/finite loss tests.

10. References
- Placeholders for formal citations; recommended to add BibTeX entries for Switch Transformer, GShard, and subsequent MoE routing works when preparing manuscripts.

Appendix
A. Code Structure
- src/project_chimera/baseline.py: Config, decoder model, transformer block
- src/project_chimera/trainer.py: Training loop, evaluation
- src/project_chimera/data/: Dataset and preprocessing
- src/project_chimera/utils/: Device and reproducibility utilities
- src/project_chimera/moe/: Experts, gating, and MoE layer modules
- scripts/: Training and data scripts
- tests/: Unit tests

B. License
- See pyproject.toml for project metadata. Add a proper open-source license before public release.

C. Acknowledgements
- This work builds on the open community’s progress in dense and sparse Transformer research. Thanks to maintainers of PyTorch and HuggingFace ecosystems.
