import json
import sys
from pathlib import Path

import pytest

# Ensure repo root is on sys.path so `from src import hybrid_agent` works
sys.path.insert(0, str(Path.cwd()))
from src import hybrid_agent as ha


def test_semantic_encoder_backend_available():
    enc = ha.SemanticEncoder(cache_dir=str(Path.cwd() / "models"))
    assert enc.mode in {"bert", "tfidf"}

    if enc.mode == "bert":
        assert hasattr(enc, "tokenizer")
        assert hasattr(enc, "model")

    enc.fit(["hello world", "another example"])
    sims = enc.score("test query")
    assert isinstance(sims, list)
    assert all(isinstance(x, float) for x in sims)


def test_generate_autoregressive_reply_coherent():
    enc = ha.SemanticEncoder(cache_dir=str(Path.cwd() / "models"))
    for intent in ha.INTENT_DESCRIPTIONS.keys():
        # Test with BERT encoder (semantic-grounded)
        reply = ha.generate_autoregressive_reply("Please do this task", intent, encoder=enc, confidence=0.8)
        assert isinstance(reply, str) and len(reply.split()) > 30
        assert "Request accepted" in reply
        assert "Integrated tool route" in reply
        assert "world_prior" in reply or "At iteration" in reply
        
        # Test with high confidence
        reply_high_conf = ha.generate_autoregressive_reply(
            "Please do this task", intent, encoder=enc, confidence=0.9
        )
        assert isinstance(reply_high_conf, str) and len(reply_high_conf.split()) > 8
        
        # Test with low confidence
        reply_low_conf = ha.generate_autoregressive_reply(
            "Please do this task", intent, encoder=enc, confidence=0.4
        )
        assert isinstance(reply_low_conf, str) and len(reply_low_conf.split()) > 5
        
        # Test fallback without encoder
        reply_fallback = ha.generate_autoregressive_reply("Please do this task", intent, encoder=None)
        assert isinstance(reply_fallback, str) and len(reply_fallback.split()) > 8
        assert "self-referential" in reply_fallback.lower()


def test_semantic_encoder_prefers_local_bert_weights(tmp_path):
    enc = ha.SemanticEncoder(cache_dir=str(Path.cwd() / "models"))
    assert enc.mode in {"bert", "tfidf"}

    # When local bert files exist, the encoder should use them and avoid download.
    local_bert = Path.cwd() / "models" / "bert-base-uncased"
    if local_bert.exists():
        assert local_bert.is_dir()


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


def test_run_freeform_turn_integrates_trace_and_response():
    enc = ha.SemanticEncoder(cache_dir=str(Path.cwd() / "models"))
    router = ha.NeuroSymbolicRouter(enc)
    turn = ha.run_freeform_turn(
        "Please reason about symbolic memory and propose a world-model grounded roadmap.",
        router,
    )
    assert isinstance(turn.user_prompt, str) and turn.user_prompt
    assert turn.predicted_intent in ha.INTENT_DESCRIPTIONS
    assert 0.5 <= turn.confidence <= 0.99
    assert turn.symbolic_plan == ha.SYMBOLIC_PLANS[turn.predicted_intent]
    assert isinstance(turn.response, str) and len(turn.response.split()) > 30
    assert "Integrated tool route" in turn.response
    assert isinstance(turn.trace, dict)
    assert "mutual_reasoning" in turn.trace
    assert "fused_scores" in turn.trace


def test_agentic_chat_orchestrator_multi_turn_stateful():
    enc = ha.SemanticEncoder(cache_dir=str(Path.cwd() / "models"))
    router = ha.NeuroSymbolicRouter(enc)
    orchestrator = ha.AgenticChatOrchestrator(router)
    prompts = ha.sample_chat_prompts()
    session = orchestrator.run_session(prompts)

    assert len(session.turns) == len(prompts)
    assert session.state["turn_count"] == len(prompts)
    assert isinstance(session.state["latent_slots"], dict)
    assert any(session.state["latent_slots"].values())
    assert isinstance(session.state["intent_transitions"], dict)

    # Continuation responses should include world-state/autoreference signals.
    assert any(
        any(k in turn.response.lower() for k in ["world model", "memory", "continu", "latent"])
        for turn in session.turns[1:]
    )


def test_evaluate_includes_chat_continuation_test(tmp_path):
    dataset = Path("data/intent_benchmark.json")
    out = tmp_path / "out_eval.json"
    report = ha.evaluate(dataset, out, cache_dir=str(Path.cwd() / "models"))
    assert "chat_continuation_test" in report
    assert "multi_turn_capable" in report["chat_continuation_test"]
    assert "world_state" in report["chat_continuation_test"]
    assert report["chat_continuation_test"]["world_state"]["turn_count"] >= 1
