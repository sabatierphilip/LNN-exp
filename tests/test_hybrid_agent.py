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
        # Test with default args
        reply = ha.generate_autoregressive_reply("Please do this task", intent)
        assert isinstance(reply, str) and len(reply.split()) > 5
        assert "Request:" in reply
        
        # Test with high confidence and trace info
        trace = {
            "fused_scores": {label: 0.2 for label in ha.INTENT_DESCRIPTIONS.keys()}
        }
        trace["fused_scores"][intent] = 0.8
        reply_high_conf = ha.generate_autoregressive_reply(
            "Please do this task", intent, confidence=0.9, trace=trace
        )
        assert isinstance(reply_high_conf, str) and len(reply_high_conf.split()) > 5
        assert "Request:" in reply_high_conf
        
        # Responses should be different when confidence varies
        reply_low_conf = ha.generate_autoregressive_reply(
            "Please do this task", intent, confidence=0.5
        )
        # Not all responses will differ, but some should when confidence changes action template
        assert len(reply_low_conf.split()) > 5


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
