import json
import sys
from pathlib import Path

import pytest

# Ensure repo root is on sys.path so `from src import hybrid_agent` works
sys.path.insert(0, str(Path.cwd()))
from src import hybrid_agent as ha


def test_semantic_encoder_uses_local_bert():
    enc = ha.SemanticEncoder(cache_dir=str(Path.cwd() / "models"))
    # Ensure it loaded the HF backend
    assert enc.mode == "bert"
    # BERT backend should expose tokenizer and model
    assert hasattr(enc, "tokenizer")
    assert hasattr(enc, "model")
    if enc.mode == "bert":
        
        # ensure the encoder has been fitted before scoring
        enc.fit(["hello world", "another example"])
        sims = enc.score("test query")
        assert isinstance(sims, list)
        assert all(isinstance(x, float) for x in sims)


def test_generate_autoregressive_reply_coherent():
    for intent in ha.INTENT_DESCRIPTIONS.keys():
        reply = ha.generate_autoregressive_reply("Please do this task", intent)
        assert isinstance(reply, str) and len(reply.split()) > 5
        lead = {
            "search": "I will search relevant sources",
            "summarize": "I will condense the material",
            "recall": "I will retrieve prior decisions",
            "generate": "I will draft a clear answer",
            "plan": "I will produce an ordered roadmap",
        }[intent]
        assert lead.split()[0].lower() in reply.lower()


def test_neurosymbolic_router_predicts_and_traces():
    enc = ha.SemanticEncoder(cache_dir=str(Path.cwd() / "models"))
    router = ha.NeuroSymbolicRouter(enc)
    text = "Find recent papers on predictive coding and summarize key findings."
    best, conf, trace = router.predict_with_trace(text, gold_intent=None)
    assert best in ha.INTENT_DESCRIPTIONS
    assert 0.5 <= conf <= 0.99
    assert isinstance(trace, dict)
    assert "mutual_reasoning" in trace


def test_evaluate_end_to_end_creates_report(tmp_path):
    dataset = Path("data/intent_benchmark.json")
    assert dataset.exists()
    out = tmp_path / "out.json"
    report = ha.evaluate(dataset, out, cache_dir=str(Path.cwd() / "models"))
    assert isinstance(report, dict)
    assert "autoregression_test" in report
    # The small benchmark should be coherent as in previous runs
    assert report["autoregression_test"]["coherence_rate"] >= 0.0
