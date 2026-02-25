# LNN-exp: Hybrid Router with Dynamic Self-Control

This runnable CLI now includes a richer modular stack with **learned, context-aware control**:

1. **Neuro-symbolic hybrid** intent routing (semantic prototypes + symbolic plans).
2. **Memory-augmented module** (external key-value memory scoring).
3. **Cognitive architecture module** (working-buffer style rule priors).
4. **World-model planner module** (plan-utility prior from rollout proxy).
5. **Predictive-coding module** (prototype mismatch minimization).
6. **Learnable gating network** (online feature-to-module weighting policy).
7. **Context-dependent module arbitration** (input-sensitive reweighting of intent support).
8. **Meta-controller** (self-adaptive module trust calibrated by model confidence).

## Reproducible evaluation command

```bash
python src/hybrid_agent.py --dataset data/intent_benchmark.json --out results/eval.json
```

## Output metrics

`results/eval.json` includes:
- Intent routing metrics (`base_bert_style_accuracy`, `neuro_symbolic_accuracy`).
- Learnable control diagnostics:
  - `learnable_gating.feature_names`
  - `learnable_gating.final_weights`
  - `learnable_gating.trace` (per-example gate outputs and fused scores)
  - `meta_controller.module_trust`
- Next-token metrics (`next_token_metrics`).
- Reasoning metrics (`reasoning_metrics`, exact match + step-F1).
- **Autoregression test** (`autoregression_test`) including:
  - `coherence_rate`
  - `can_function_as_chatbot`

## Chatbot viability rule

The report marks `can_function_as_chatbot=true` when autoregressive coherence is at least `0.8` on the built-in benchmark.
