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
  - `coherence_rate`
  - `can_function_as_chatbot`

## Chatbot viability rule

The report marks `can_function_as_chatbot=true` when autoregressive coherence is at least 0.8 on the built-in benchmark.

## Limitations and Next Steps

This repository contains an early-stage, runnable prototype that integrates several cognitive-inspired modules with a semantic router. It is a clear and useful exploratory implementation, but not yet positioned as a final or revolutionary system. Key limitations and recommended next steps:

- **Heuristic fusion:** Module outputs are combined with fixed, hand-tuned weights rather than a learned fusion layer. Consider adding a trainable fusion module and reporting generalization.
- **Prototype-level modules:** The memory, cognitive priors, and predictive-coding components are lightweight and rule-based. Replace them with learnable or retrieval-augmented counterparts for stronger claims.
- **Small synthetic evaluation:** Current benchmarks are toy-like and deterministic. Run larger-scale evaluations, ablations, and statistical tests against standard baselines.
- **No training pipeline:** There is no end-to-end training or fine-tuning harness. Add scripts to train and evaluate any learnable components and report scaling/latency.



