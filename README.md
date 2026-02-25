# LNN-exp: Hybrid Router with Dynamic Self-Control

This runnable CLI includes a modular neuro-symbolic router with dynamic control and iterative mutual reasoning:

1. **Neuro-symbolic hybrid routing** (semantic prototypes + symbolic plans).
2. **Memory-augmented module** (external key-value memory scoring).
3. **Cognitive architecture module** (working-buffer style priors).
4. **World-model planner module** (plan-utility prior).
5. **Predictive-coding module** (prototype mismatch minimization).
6. **Learnable gating network** (context-feature → module-weight policy with online updates).
7. **Context-dependent module arbitration** (input-sensitive intent evidence scaling).
8. **Meta-controller** (confidence-aware self-adjustment of module trust).
9. **Dynamic mutual reasoning engine** (iterative inter-module belief exchange with learned relation graph).

## Reproducible evaluation command

```bash
python src/hybrid_agent.py --dataset data/intent_benchmark.json --out results/eval.json
```

## Reproducible freeform conversation command

```bash
python src/hybrid_agent.py --cache-dir models --chat-prompt "Please reason over our symbolic memory and build an integration roadmap."
```

## Output metrics

`results/eval.json` includes:
- Intent routing metrics (`base_bert_style_accuracy`, `neuro_symbolic_accuracy`).
- Learnable control diagnostics:
  - `learnable_gating.feature_names`
  - `learnable_gating.final_weights`
  - `learnable_gating.trace` (per-example features, gate outputs, mutual reasoning trace, fused scores)
  - `meta_controller.module_trust`
- Mutual reasoning diagnostics:
  - `mutual_reasoning.average_rounds`
  - `mutual_reasoning.relation_graph`
- Next-token metrics (`next_token_metrics`).
- Reasoning metrics (`reasoning_metrics`, exact match + step-F1).
- Autoregression viability metrics (`autoregression_test`).

## Chatbot viability rule

The report marks `can_function_as_chatbot=true` when autoregressive coherence is at least `0.8` on the built-in benchmark.
