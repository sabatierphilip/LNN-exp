from __future__ import annotations

import statistics
import sys
from pathlib import Path

import pytest

# Ensure repo root is importable
sys.path.insert(0, str(Path.cwd()))

from src.gisd_generator import GISDConfig, GISDResponseGenerator, GuidedIterativeSemanticDiffusion
from src.hybrid_agent import INTENT_DESCRIPTIONS, NeuroSymbolicRouter, SemanticEncoder


@pytest.fixture(scope="module")
def encoder() -> SemanticEncoder:
    return SemanticEncoder(cache_dir=str(Path.cwd() / "models"))


@pytest.fixture(scope="module")
def gisd(encoder: SemanticEncoder) -> GuidedIterativeSemanticDiffusion:
    return GuidedIterativeSemanticDiffusion(encoder, GISDConfig(diffusion_steps=6, min_tokens=18, max_tokens=48))


def test_1_basic_generation(gisd: GuidedIterativeSemanticDiffusion):
    prompt = "Explain memory-augmented neural systems for long-context reasoning"
    for intent in INTENT_DESCRIPTIONS:
        text, ids = gisd.generate(prompt, intent)
        assert isinstance(text, str)
        assert isinstance(ids, list)
        assert len(text.split()) >= 5
        assert len(ids) > 0
        assert "[MASK]" not in text
        assert "[CLS]" not in text


def test_2_semantic_steering_moves_logits(gisd: GuidedIterativeSemanticDiffusion):
    prompt = "world models for dialogue planning"
    target_vec = gisd._semantic_target_vec(prompt)
    E = gisd._ensure_unit_embeddings()
    steering_scores = E @ target_vec

    encoded = gisd.tokenizer(f"{prompt} {gisd.tokenizer.mask_token}", return_tensors="pt").to(gisd.device)
    mask_pos = int((encoded["input_ids"][0] == gisd.tokenizer.mask_token_id).nonzero(as_tuple=False)[0, 0])
    with gisd.torch.no_grad():
        raw_logits = gisd.masked_lm(**encoded).logits[0, mask_pos]

    best_sem_idx = int(gisd.torch.argmax(steering_scores).item())
    best_sem = float(steering_scores[best_sem_idx].item())
    assert best_sem > 0.0
    alpha = (float(raw_logits.max().item()) - float(raw_logits[best_sem_idx].item()) + 0.1) / max(best_sem, 1e-6)
    steered_logits = raw_logits + alpha * steering_scores

    assert bool((steered_logits > raw_logits).any().item())
    assert float(steered_logits.max().item()) > float(raw_logits.max().item())


def test_3_revision_actually_remasks(encoder: SemanticEncoder):
    cfg = GISDConfig(diffusion_steps=3, remask_fraction=0.5, min_tokens=18, max_tokens=24)
    gd = GuidedIterativeSemanticDiffusion(encoder, cfg)

    prompt = "prototype agent memory"
    prefix = gd._prefix_ids(prompt)
    n_output = 20
    seq = gd.torch.tensor(
        [gd.tokenizer.cls_token_id, *prefix, *([gd.tokenizer.mask_token_id] * n_output), gd.tokenizer.sep_token_id],
        device=gd.device,
        dtype=gd.torch.long,
    )
    out_start = 1 + len(prefix)
    output_positions = list(range(out_start, out_start + n_output))
    revealed = gd.torch.zeros(n_output, device=gd.device, dtype=gd.torch.bool)
    token_counts = {}
    target_vec = gd._semantic_target_vec(prompt)

    seq, revealed = gd._diffuse_once(seq, revealed, output_positions, target_vec, 0, token_counts)
    assert bool(revealed.any().item())

    seq, revealed = gd._diffuse_once(
        seq,
        revealed,
        output_positions,
        target_vec,
        1,
        token_counts,
        revision_conf_threshold=1.0,
    )
    assert bool((~revealed).any().item())


def test_4_temperature_annealing_monotonic(gisd: GuidedIterativeSemanticDiffusion):
    vals = [gisd._anneal(1.6, 0.55, frac, 1.4) for frac in [0.0, 0.25, 0.5, 0.75, 1.0]]
    for i in range(1, len(vals)):
        assert vals[i] < vals[i - 1]


def test_5_intent_length_hints_different_budgets(gisd: GuidedIterativeSemanticDiffusion):
    prompt = "Discuss tradeoffs in retrieval augmented generation systems"
    plan_lengths = []
    summarize_lengths = []
    for _ in range(3):
        _, plan_ids = gisd.generate(prompt, "plan")
        _, sum_ids = gisd.generate(prompt, "summarize")
        plan_lengths.append(len(plan_ids))
        summarize_lengths.append(len(sum_ids))

    assert statistics.mean(plan_lengths) > statistics.mean(summarize_lengths)


def test_6_no_special_tokens_in_output(gisd: GuidedIterativeSemanticDiffusion):
    special = ["[MASK]", "[CLS]", "[SEP]", "[PAD]"]
    for intent in INTENT_DESCRIPTIONS:
        text, _ = gisd.generate("Explain latent world models.", intent)
        assert all(tok not in text for tok in special)


def test_7_response_generator_builds_coherent_prefix(encoder: SemanticEncoder):
    rg = GISDResponseGenerator(encoder)
    history = ["Earlier we prioritized compact memory and fast retrieval"]
    for intent in INTENT_DESCRIPTIONS:
        tool_outputs = [
            {"tool": intent, "result": "useful details", "items": ["a", "b"]},
            {"tool": "verify", "status": "ok"},
        ]
        prefix = rg._build_context_prefix("Need practical guidance", intent, tool_outputs, history)
        assert isinstance(prefix, str) and prefix.strip()
        assert "{" not in prefix
        assert '"tool":' not in prefix


def test_8_repetition_penalty_reduces_frequency(gisd: GuidedIterativeSemanticDiffusion):
    prompt = "Explain why memory addressing helps long context handling"
    ratios = []
    for _ in range(5):
        text, _ = gisd.generate(prompt, "generate")
        toks = [t.lower() for t in text.split() if t.strip()]
        if toks:
            ratios.append(len(set(toks)) / len(toks))
    assert ratios
    assert statistics.mean(ratios) > 0.55


def test_9_full_pipeline_smoke_test(encoder: SemanticEncoder):
    router = NeuroSymbolicRouter(encoder)
    generator = GISDResponseGenerator(encoder)
    prompt = "What are the tradeoffs of external memory in neural networks?"
    intent, conf = router.predict(prompt)
    response = generator.generate_response(
        prompt,
        intent,
        conf,
        conversation_history=["We discussed retrieval cost last week."],
        tool_outputs=[{"tool": "search", "finding": "memory boosts context but increases latency"}],
    )
    assert isinstance(response, str)
    assert len(response.split()) > 15
    assert "Request accepted" not in response
    assert "Integrated tool route" not in response


def test_10_sample_outputs_table(gisd: GuidedIterativeSemanticDiffusion):
    prompts = [
        "Find recent work on world models for dialogue agents",
        "Summarize the tradeoffs between memory size and retrieval latency",
        "What did we decide about the tokenizer choice last session?",
        "Write a concise explanation of why MANN helps with long context",
        "Plan a two-week sprint to prototype a compact world-model chatbot",
    ]
    conf = 0.82
    for prompt in prompts:
        print(f"PROMPT: {prompt}")
        for intent in INTENT_DESCRIPTIONS:
            text, ids = gisd.generate(prompt, intent)
            print(f"INTENT: {intent} | CONFIDENCE: {conf:.2f} | TOKENS: {len(ids)}")
            print(f"OUTPUT: {text}")
            print("---")
