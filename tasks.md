# Project Chimera – Task List

Status legend: [ ] pending  [-] in progress  [x] done

## Phase 1: Research and Planning
- [x] Compile literature review on MoE (Mixtral, Switch Transformer, GShard, recent routing methods)
- [x] Summarize routing mechanisms and trade-offs (top-k, load balancing, auxiliary losses)
- [x] Draft synthesis: key differences vs dense transformers
- [x] Prepare presentation of findings for team review

## Phase 2: Scope, Metrics, and Planning Artifacts
- [x] Finalize success metrics (training loss target vs dense baseline, inference throughput/latency)
- [x] Validate/adjust project scope against research takeaways
- [x] Define evaluation protocol (datasets, splits, metrics, logging)

## Phase 3: Environment Setup
- [x] Set up Python environment with PyTorch and dependencies (CUDA support)
- [x] Verify GPU availability and configure device selection
- [x] Establish project skeleton (src/, tests/, notebooks/, scripts/, data/README.md)
- [x] Configure reproducibility (seeds, deterministic flags as feasible)
- [x] Add basic CI to run unit tests and linting

## Phase 4: Data Preparation
- [x] Select and document dataset(s) (AG News; see data/README.md)
- [x] Implement data download and caching script (scripts/download_ag_news.py)
- [x] Implement preprocessing/tokenization pipeline (lowercase, GPT-2 tokenizer, max_seq_len=128)
- [x] Create data loaders with batching and sequence packing (causal LM targets, padding/truncation)
- [x] Smoke test data pipeline performance (scripts/smoke_test_data_pipeline.py and tests)

## Phase 5: Baseline Implementation (Dense Transformer Decoder)
- [x] Implement nano-GPT style Transformer decoder block in PyTorch
- [x] Implement model config (d_model, n_layers, n_heads, ff_dim, dropout, vocab_size)
- [x] Implement training loop (optimizer, scheduler, mixed precision optional)
- [x] Add unit tests for block shapes/forward pass
- [x] Run baseline training smoke test and record metrics

## Phase 6: MoE Layer Design and Implementation
- [x] Design MoE layer interface (experts, gating, top-k, load-balancing loss)
  - Scaffolding in place: src/project_chimera/moe/{experts.py,gating.py,layer.py}
  - BaselineConfig gained MoE flags (moe_enabled, moe_n_experts, moe_top_k, moe_activation, moe_noisy_gate)
  - TransformerBlock supports MoE FFN via MoEFFNWrapper behind feature flag
- [x] Implement expert networks (e.g., FFN experts) — basic FFNExpert implemented with activation/dropout; extendable for init scaling/weight tying
- [x] Implement gating network with routing and token dispatch — efficient gather/scatter with capacity implemented
- [x] Add load balancing/aux losses and configurables — load-balancing aux loss with coef added
- [x] Optimize routing (batch-wise gather/scatter efficiency) — dispatched compute path implemented
- [x] Unit tests for MoE layer (routing correctness, capacity adherence, shapes, finite loss, determinism, top-k distribution sanity)

## Phase 7: Integration of MoE into Transformer
- [x] Integrate MoE layer into Transformer block (replace/augment FFN)
- [x] Expose configuration flags to toggle dense vs MoE
- [x] Validate forward/backward correctness and stability
- [x] Micro-benchmarks to verify inference/training speed
- [x] Author unit tests for MoE dispatch and capacity behavior (routing correctness, shapes, finite loss) — validated via passing test suite

## Phase 8: Training and Evaluation
- [x] Prepare dataset artifacts (download/cache AG News parquet via scripts) — dataset download fixed and verified with single-parquet-per-split output
- [-] Implement/enable training CLI for real dataset runs (dense and MoE) using BaselineTrainer
- [ ] Train dense baseline with compute-matched settings
- [ ] Train MoE model on curated dataset (track losses, aux losses, expert utilization)
- [ ] Evaluate metrics (training loss curves, validation perplexity)
- [ ] Measure inference speed and memory footprint
- [ ] Compare MoE vs dense at similar compute/latency

## Phase 9: Analysis and Reporting
- [ ] Analyze results (loss, speed, utilization, load balancing effectiveness)
- [ ] Create visualizations (training curves, routing histograms, throughput)
- [ ] Draft final report (methods, experiments, results, discussion, future work)
- [ ] Prepare presentation deck summarizing findings

## Phase 10: Documentation and Packaging
- [x] Write README with quickstart, configs, and commands
- [ ] API docs for core modules (Transformer block, MoE layer, gating)
- [ ] Example configs for dense and MoE runs
- [ ] Reproducibility checklist and seed/results table

## Future Work and Extensions
- [ ] Explore advanced routing (noisy top-k, Sinkhorn/soft routing, task-aware)
- [ ] Evaluate expert parallelism strategies (sharding, FSDP/DeepSpeed integration)
- [ ] Scale experiments (larger datasets, deeper/wider models)
- [ ] Draft publication/presentation outline from report

## Administrative/Project Hygiene
- [ ] Define experiment tracking (e.g., WandB/MLflow) and naming conventions
- [ ] Set up automated checkpoints and best-model selection
- [x] Establish coding standards and pre-commit hooks (.pre-commit-config.yaml)
- [ ] Maintain CHANGELOG and version tags for key milestones