# LNN-exp: Hybrid + 4 Additional Architecture Features

This runnable CLI now includes your requested five-part stack:

1. **Neuro-symbolic hybrid** intent routing (existing core).
2. **Memory-augmented module** (external key-value memory scoring).
3. **Cognitive architecture module** (working-buffer style rule priors).
4. **World-model planner module** (plan-utility prior from action rollouts proxy).
5. **Predictive-coding module** (prototype mismatch minimization signal).

## Reproducible evaluation command

```bash
python src/hybrid_agent.py --dataset data/intent_benchmark.json --out results/eval.json
```

## Output metrics

`results/eval.json` includes:
- Intent routing metrics (`base_bert_style_accuracy`, `neuro_symbolic_accuracy`).
- Next-token metrics (`next_token_metrics`).
- Reasoning metrics (`reasoning_metrics`, exact match + step-F1).
- **Autoregression test** (`autoregression_test`) including:
  - `coherence_rate`
  - `can_function_as_chatbot`

## Chatbot viability rule

The report marks `can_function_as_chatbot=true` when autoregressive coherence is at least 0.8 on the built-in benchmark.
