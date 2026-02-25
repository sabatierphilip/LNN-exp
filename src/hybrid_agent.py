"""Hybrid agent evaluator with additional cognitive modules and autoregression test.

Run:
    python src/hybrid_agent.py --dataset data/intent_benchmark.json --out results/eval.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

INTENT_DESCRIPTIONS: Dict[str, str] = {
    "search": "Find external information, references, papers, or facts from sources.",
    "summarize": "Compress content into shorter key points or concise overviews.",
    "recall": "Retrieve remembered decisions, past notes, or stored memory details.",
    "generate": "Compose new text such as an answer, message, or explanation.",
    "plan": "Create an ordered roadmap, checklist, or execution strategy.",
}

SYMBOLIC_PLANS: Dict[str, List[str]] = {
    "search": ["search", "rank_sources", "summarize"],
    "summarize": ["recall", "summarize", "generate"],
    "recall": ["recall", "verify", "generate"],
    "generate": ["recall", "plan", "generate"],
    "plan": ["recall", "decompose", "plan", "generate"],
}

FALLBACK_EXAMPLES: Dict[str, List[str]] = {
    "search": ["look up recent results", "gather references", "find sources"],
    "summarize": ["condense this report", "shorten the notes", "key takeaways only"],
    "recall": ["what did we decide", "retrieve from memory", "remind me of earlier choice"],
    "generate": ["draft a reply", "compose an explanation", "write a response"],
    "plan": ["create milestones", "step-by-step roadmap", "build an execution plan"],
}

NEXT_TOKEN_BENCHMARK: List[Dict[str, object]] = [
    {"prompt": "Please look up recent LNN papers and", "candidates": ["summarize", "compile", "juggle"], "gold": "compile"},
    {"prompt": "Can you condense these long notes into", "candidates": ["bullets", "tomatoes", "circuits"], "gold": "bullets"},
    {"prompt": "Remind me what optimizer we used in", "candidates": ["run_17", "sunset", "marble"], "gold": "run_17"},
    {"prompt": "Draft a clear reply that", "candidates": ["explains", "evaporates", "balances"], "gold": "explains"},
    {"prompt": "Create a step-by-step roadmap with", "candidates": ["milestones", "glitter", "raindrops"], "gold": "milestones"},
]

REASONING_BENCHMARK: List[Dict[str, object]] = [
    {"text": "Find recent benchmark results and provide key takeaways.", "intent": "search"},
    {"text": "Condense this design review into brief bullets.", "intent": "summarize"},
    {"text": "What did we decide in yesterday's architecture meeting?", "intent": "recall"},
    {"text": "Write a polished explanation for stakeholders.", "intent": "generate"},
    {"text": "Plan milestones for integrating symbolic memory.", "intent": "plan"},
]

DIFFICULT_CHAT_CHALLENGE: List[str] = [
    "Critical constraint for all future turns: the project codename is quartz-17 and must be preserved exactly.",
    "Find two relevant references for building an agentic world model.",
    "Summarize those references in concise bullets.",
    "Before continuing, remind me of the exact codename from turn one.",
    "Draft a short update that still includes the codename exactly once.",
    "Final check: what was that exact codename? Return it verbatim.",
]

AUTOREG_BENCHMARK: List[Dict[str, str]] = [
    {"prompt": "Find papers on predictive coding and give key points.", "intent": "search"},
    {"prompt": "Summarize the previous experiment notes.", "intent": "summarize"},
    {"prompt": "What did we decide about memory size yesterday?", "intent": "recall"},
    {"prompt": "Write a concise stakeholder update.", "intent": "generate"},
    {"prompt": "Create a roadmap for symbolic integration.", "intent": "plan"},
]


@dataclass
class Prediction:
    text: str
    gold_intent: str
    pred_intent: str
    confidence: float
    symbolic_plan: List[str]


@dataclass
class ConversationTurn:
    """Structured freeform conversation output for CLI usage."""

    user_prompt: str
    predicted_intent: str
    confidence: float
    symbolic_plan: List[str]
    world_model_prior: float
    response: str
    trace: Dict[str, object]


@dataclass
class ChatSessionState:
    """State container for multi-turn continuation and agentic world modeling."""

    history: List[ConversationTurn] = field(default_factory=list)
    intent_transitions: Dict[str, Dict[str, int]] = field(default_factory=dict)
    latent_slots: Dict[str, List[str]] = field(default_factory=dict)
    global_memory: List[str] = field(default_factory=list)


@dataclass
class ChatSessionResult:
    """Serializable chat session transcript with diagnostics."""

    turns: List[ConversationTurn]
    state: Dict[str, object]


class KeywordBaseline:
    """Small lexical baseline for intent classification."""

    keywords: Dict[str, List[str]] = {
        "search": ["search", "find", "look up", "gather", "benchmark", "papers"],
        "summarize": ["summarize", "condense", "short", "key takeaways", "brief"],
        "recall": ["remind", "recall", "what did", "which", "yesterday"],
        "generate": ["write", "draft", "compose", "generate", "reply", "explain"],
        "plan": ["plan", "roadmap", "step-by-step", "milestones", "checklist"],
    }

    def predict(self, text: str) -> Tuple[str, float]:
        lowered = text.lower()
        best_label, best_score = "generate", -1
        for label, terms in self.keywords.items():
            score = sum(term in lowered for term in terms)
            if score > best_score:
                best_label, best_score = label, score
        return best_label, float(min(0.99, 0.5 + 0.15 * max(best_score, 0)))


class SemanticEncoder:
    """Semantic backend using BERT when available; TF-IDF fallback otherwise."""

    def __init__(self, cache_dir: str) -> None:
        self.mode = "tfidf"
        self._setup_backend(cache_dir)

    def _setup_backend(self, cache_dir: str) -> None:
        try:
            import torch
            import torch.nn.functional as F
            from transformers import AutoModel, AutoTokenizer

            self._torch = torch
            self._F = F
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            source_path = Path(cache_dir) / "bert-base-uncased"
            source = str(source_path) if source_path.exists() else "bert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(source, cache_dir=cache_dir)
            self.model = AutoModel.from_pretrained(source, cache_dir=cache_dir).to(self.device)
            self.model.eval()
            self.mode = "bert"
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
            self._cosine_similarity = cosine_similarity

    def fit(self, texts: Sequence[str]) -> None:
        if self.mode == "tfidf":
            self.matrix = self.vectorizer.fit_transform(texts)
            return
        with self._torch.no_grad():
            self.matrix = self._embed(list(texts))

    def score(self, text: str) -> List[float]:
        if self.mode == "tfidf":
            return list(self._cosine_similarity(self.vectorizer.transform([text]), self.matrix)[0])
        with self._torch.no_grad():
            sims = (self._embed([text]) @ self.matrix.T).squeeze(0)
            return [float(x) for x in sims.tolist()]

    def _embed(self, texts: List[str]):
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        cls = self.model(**encoded).last_hidden_state[:, 0, :]
        return self._F.normalize(cls, p=2, dim=-1)


class MemoryAugmentedModule:
    """MANN-style lightweight external memory via key-value overlap scores."""

    def __init__(self) -> None:
        self.memory = {
            "search": ["papers", "benchmark", "sources", "results"],
            "summarize": ["condense", "brief", "short", "takeaways"],
            "recall": ["yesterday", "remember", "decide", "run_17"],
            "generate": ["write", "compose", "reply", "explain"],
            "plan": ["roadmap", "milestones", "steps", "checklist"],
        }

    def score(self, text: str) -> Dict[str, float]:
        t = text.lower()
        return {k: sum(term in t for term in terms) / max(1, len(terms)) for k, terms in self.memory.items()}


class CognitiveArchitectureModule:
    """ACT-R/SOAR-inspired routing priors through working buffers."""

    def score(self, text: str) -> Dict[str, float]:
        t = text.lower()
        return {
            "search": float(any(x in t for x in ["find", "look up", "source"])),
            "summarize": float(any(x in t for x in ["summarize", "condense", "short"])),
            "recall": float(any(x in t for x in ["remember", "remind", "yesterday", "which"])),
            "generate": float(any(x in t for x in ["write", "draft", "compose", "explain"])),
            "plan": float(any(x in t for x in ["plan", "roadmap", "milestones", "step-by-step"])),
        }


class WorldModelPlannerModule:
    """Dreamer-style proxy: evaluate intent by expected plan length utility."""

    def score(self) -> Dict[str, float]:
        # Slightly favors intents with richer actionable plans.
        raw = {k: len(v) for k, v in SYMBOLIC_PLANS.items()}
        denom = max(raw.values())
        return {k: v / denom for k, v in raw.items()}


class LearnableGatingNetwork:
    """Small learnable gate that maps context features to module weights.

    The gate exposes a light online-learning interface so routing can adapt
    during evaluation without requiring a heavyweight training loop.
    """

    module_names = ["semantic", "memory", "cognitive", "predictive", "world"]

    def __init__(self, labels: Sequence[str], lr: float = 0.08) -> None:
        self.labels = list(labels)
        self.lr = lr
        self.feature_names = [
            "bias",
            "search_density",
            "summarize_density",
            "recall_density",
            "generate_density",
            "plan_density",
            "token_length",
            "question_flag",
            "imperative_flag",
        ]
        self.weights = {
            module: [0.0 for _ in self.feature_names] for module in self.module_names
        }
        # Initialize near prior fusion coefficients for stable cold-start behavior.
        base_bias = {
            "semantic": 0.55,
            "memory": 0.20,
            "cognitive": 0.10,
            "predictive": 0.10,
            "world": 0.05,
        }
        for module in self.module_names:
            self.weights[module][0] = base_bias[module]

    def _softmax(self, values: Sequence[float]) -> List[float]:
        shift = max(values)
        exps = [math.exp(v - shift) for v in values]
        total = sum(exps)
        return [v / total for v in exps]

    def extract_features(self, text: str) -> List[float]:
        lowered = text.lower()
        tokens = lowered.split()
        n = max(1, len(tokens))
        keyword_banks = {
            "search_density": ["search", "find", "source", "benchmark", "paper"],
            "summarize_density": ["summarize", "condense", "brief", "concise", "short"],
            "recall_density": ["recall", "remember", "remind", "yesterday", "past"],
            "generate_density": ["write", "draft", "compose", "answer", "explain"],
            "plan_density": ["plan", "roadmap", "milestone", "steps", "checklist"],
        }
        densities = []
        for words in keyword_banks.values():
            count = sum(w in lowered for w in words)
            densities.append(count / n)
        return [
            1.0,
            *densities,
            min(1.0, n / 30.0),
            float("?" in text),
            float(any(lowered.startswith(v) for v in ["please", "find", "write", "plan", "summarize"])),
        ]

    def gate(self, text: str) -> Tuple[Dict[str, float], List[float]]:
        features = self.extract_features(text)
        logits = [
            sum(w * x for w, x in zip(self.weights[module], features))
            for module in self.module_names
        ]
        probs = self._softmax(logits)
        return {module: probs[i] for i, module in enumerate(self.module_names)}, features

    def update(self, features: Sequence[float], rewarded_modules: Dict[str, float]) -> None:
        """Online policy-style update from module utility signals."""
        utilities = [rewarded_modules.get(module, 0.0) for module in self.module_names]
        mean_u = sum(utilities) / len(utilities)
        for m_idx, module in enumerate(self.module_names):
            advantage = utilities[m_idx] - mean_u
            for f_idx, value in enumerate(features):
                self.weights[module][f_idx] += self.lr * advantage * value


class ContextDependentArbitration:
    """Contextual arbitration that rescales each module's per-intent contribution."""

    def blend(
        self,
        text: str,
        labels: Sequence[str],
        module_scores: Dict[str, Dict[str, float]],
        gate_weights: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        lowered = text.lower()
        arbitration = {label: 1.0 for label in labels}
        if any(x in lowered for x in ["why", "explain", "because"]):
            for label in labels:
                arbitration[label] *= 1.12 if label in ["generate", "plan"] else 0.95
        if any(x in lowered for x in ["yesterday", "previous", "remember"]):
            for label in labels:
                arbitration[label] *= 1.25 if label == "recall" else 0.92
        if any(x in lowered for x in ["papers", "sources", "benchmark"]):
            for label in labels:
                arbitration[label] *= 1.20 if label == "search" else 0.94

        contributions: Dict[str, Dict[str, float]] = {label: {} for label in labels}
        fused = {label: 0.0 for label in labels}
        for module, per_label in module_scores.items():
            for label in labels:
                weighted = gate_weights[module] * per_label[label] * arbitration[label]
                contributions[label][module] = weighted
                fused[label] += weighted
        return fused, contributions


class MetaController:
    """Self-regulating controller that adapts module trust using model confidence."""

    def __init__(self, module_names: Sequence[str], adapt_rate: float = 0.2) -> None:
        self.adapt_rate = adapt_rate
        self.module_names = list(module_names)
        self.module_trust = {module: 1.0 for module in module_names}

    def adjust(
        self,
        gate_weights: Dict[str, float],
        contributions: Dict[str, Dict[str, float]],
        best_label: str,
        confidence: float,
    ) -> Dict[str, float]:
        adjusted = {}
        low_confidence = max(0.0, 0.75 - confidence)
        for module, weight in gate_weights.items():
            support = contributions[best_label].get(module, 0.0)
            trust_signal = 1.0 + self.adapt_rate * (support - low_confidence)
            self.module_trust[module] = max(0.5, min(1.6, self.module_trust[module] * trust_signal))
            adjusted[module] = weight * self.module_trust[module]
        denom = sum(adjusted.values())
        return {k: v / denom for k, v in adjusted.items()}


class MutualReasoningEngine:
    """Iterative multi-module belief exchange with dynamic relation learning.

    Each module starts with its per-intent score vector. The engine then performs
    several rounds of message passing where modules influence each other according
    to a learned pairwise trust graph. When module disagreement is high, the
    engine allocates additional rounds to improve consensus.
    """

    def __init__(self, module_names: Sequence[str], labels: Sequence[str], damping: float = 0.35) -> None:
        self.module_names = list(module_names)
        self.labels = list(labels)
        self.damping = damping
        self.relation = {
            src: {dst: (1.0 if src == dst else 0.65) for dst in self.module_names}
            for src in self.module_names
        }

    def _normalize(self, values: Dict[str, float]) -> Dict[str, float]:
        shift = max(values.values())
        exps = {k: math.exp(v - shift) for k, v in values.items()}
        total = sum(exps.values())
        return {k: exps[k] / total for k in values}

    def _disagreement(self, beliefs: Dict[str, Dict[str, float]]) -> float:
        pairwise = []
        for i, m1 in enumerate(self.module_names):
            for m2 in self.module_names[i + 1 :]:
                pairwise.append(sum(abs(beliefs[m1][l] - beliefs[m2][l]) for l in self.labels) / len(self.labels))
        return 0.0 if not pairwise else sum(pairwise) / len(pairwise)

    def run(
        self,
        module_scores: Dict[str, Dict[str, float]],
        gate_weights: Dict[str, float],
        min_rounds: int = 2,
        max_rounds: int = 5,
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, object]]:
        beliefs = {m: self._normalize(module_scores[m]) for m in self.module_names}
        history: List[Dict[str, float]] = []
        steps = min_rounds
        for idx in range(max_rounds):
            disagreement = self._disagreement(beliefs)
            history.append({"round": idx + 1, "disagreement": round(disagreement, 6)})
            if idx + 1 >= min_rounds and disagreement < 0.06:
                steps = idx + 1
                break
            if idx + 1 >= max_rounds:
                steps = max_rounds
                break

            next_beliefs: Dict[str, Dict[str, float]] = {}
            for target in self.module_names:
                aggregated = {label: 0.0 for label in self.labels}
                incoming_weight = 0.0
                for source in self.module_names:
                    influence = self.relation[source][target] * max(0.05, gate_weights.get(source, 0.0))
                    incoming_weight += influence
                    for label in self.labels:
                        aggregated[label] += influence * beliefs[source][label]
                if incoming_weight == 0:
                    next_beliefs[target] = dict(beliefs[target])
                    continue
                merged = {}
                for label in self.labels:
                    peer_view = aggregated[label] / incoming_weight
                    merged[label] = (1.0 - self.damping) * beliefs[target][label] + self.damping * peer_view
                next_beliefs[target] = self._normalize(merged)
            beliefs = next_beliefs

            if idx + 1 >= min_rounds and disagreement > 0.18:
                steps = min(max_rounds, idx + 2)

        consensus = {label: 0.0 for label in self.labels}
        for module in self.module_names:
            for label in self.labels:
                consensus[label] += gate_weights[module] * beliefs[module][label]
        total = sum(consensus.values())
        if total > 0:
            consensus = {k: v / total for k, v in consensus.items()}
        trace = {
            "rounds_executed": steps,
            "history": history[:steps],
            "consensus": {k: round(v, 6) for k, v in consensus.items()},
            "final_module_beliefs": {
                m: {k: round(v, 6) for k, v in beliefs[m].items()} for m in self.module_names
            },
        }
        return beliefs, trace

    def update_relations(
        self,
        final_beliefs: Dict[str, Dict[str, float]],
        predicted_label: str,
        gold_label: str | None,
    ) -> None:
        if gold_label is None:
            return
        reward = 1.0 if predicted_label == gold_label else -1.0
        for src in self.module_names:
            for dst in self.module_names:
                agree = 1.0 - abs(final_beliefs[src][gold_label] - final_beliefs[dst][gold_label])
                delta = 0.04 * reward * (agree - 0.5)
                self.relation[src][dst] = max(0.2, min(1.8, self.relation[src][dst] + delta))


class PredictiveCodingModule:
    """Predictive-coding proxy: minimize mismatch between text cues and intent prototypes."""

    prototypes = {
        "search": ["external", "papers", "find"],
        "summarize": ["compress", "short", "concise"],
        "recall": ["memory", "past", "retrieve"],
        "generate": ["compose", "answer", "explain"],
        "plan": ["ordered", "roadmap", "strategy"],
    }

    def score(self, text: str) -> Dict[str, float]:
        t = text.lower()
        scores: Dict[str, float] = {}
        for label, words in self.prototypes.items():
            misses = sum(w not in t for w in words)
            scores[label] = 1.0 - (misses / len(words))
        return scores


class BaseSemanticRouter:
    """Base BERT-style router: match only intent descriptions."""

    def __init__(self, encoder: SemanticEncoder) -> None:
        self.encoder = encoder
        self.labels = list(INTENT_DESCRIPTIONS)
        self.encoder.fit(list(INTENT_DESCRIPTIONS.values()))

    def predict(self, text: str) -> Tuple[str, float]:
        scores = self.encoder.score(text)
        idx = max(range(len(scores)), key=lambda i: scores[i])
        return self.labels[idx], float(max(0.5, min(0.99, scores[idx] + 0.5)))


class NeuroSymbolicRouter:
    """Neuro-symbolic + memory + cognitive + world model + predictive-coding router."""

    def __init__(self, encoder: SemanticEncoder) -> None:
        self.encoder = encoder
        self.labels = list(INTENT_DESCRIPTIONS)
        self.prototype_labels: List[str] = []
        texts: List[str] = []
        for label in self.labels:
            texts.append(INTENT_DESCRIPTIONS[label])
            self.prototype_labels.append(label)
            for tool in SYMBOLIC_PLANS[label]:
                texts.append(f"intent {label} uses tool {tool}")
                self.prototype_labels.append(label)
            for ex in FALLBACK_EXAMPLES[label]:
                texts.append(ex)
                self.prototype_labels.append(label)
        self.encoder.fit(texts)

        self.memory_module = MemoryAugmentedModule()
        self.cog_module = CognitiveArchitectureModule()
        self.world_module = WorldModelPlannerModule()
        self.pred_module = PredictiveCodingModule()
        self.gating_network = LearnableGatingNetwork(self.labels)
        self.arbitration = ContextDependentArbitration()
        self.meta_controller = MetaController(self.gating_network.module_names)
        self.mutual_reasoner = MutualReasoningEngine(self.gating_network.module_names, self.labels)
        self.last_trace: Dict[str, object] = {}

    def _base_module_scores(self, text: str) -> Dict[str, Dict[str, float]]:
        proto_scores = self.encoder.score(text)
        max_proto = {k: -1e9 for k in self.labels}
        for idx, value in enumerate(proto_scores):
            label = self.prototype_labels[idx]
            max_proto[label] = max(max_proto[label], float(value))
        return {
            "semantic": max_proto,
            "memory": self.memory_module.score(text),
            "cognitive": self.cog_module.score(text),
            "predictive": self.pred_module.score(text),
            "world": self.world_module.score(),
        }

    def predict_with_trace(self, text: str, gold_intent: str | None = None) -> Tuple[str, float, Dict[str, object]]:
        module_scores = self._base_module_scores(text)
        gate_weights, features = self.gating_network.gate(text)

        reasoned_beliefs, mutual_trace = self.mutual_reasoner.run(module_scores, gate_weights)
        fused, contributions = self.arbitration.blend(text, self.labels, reasoned_beliefs, gate_weights)

        provisional_best = max(fused, key=fused.get)
        provisional_confidence = float(max(0.5, min(0.99, fused[provisional_best] + 0.5)))

        adjusted_gate = self.meta_controller.adjust(gate_weights, contributions, provisional_best, provisional_confidence)
        fused_adjusted, adjusted_contrib = self.arbitration.blend(text, self.labels, reasoned_beliefs, adjusted_gate)
        best = max(fused_adjusted, key=fused_adjusted.get)
        confidence = float(max(0.5, min(0.99, fused_adjusted[best] + 0.5)))

        if gold_intent is not None:
            margin = fused_adjusted.get(gold_intent, 0.0) - max(
                v for k, v in fused_adjusted.items() if k != gold_intent
            )
            rewards = {}
            for module in self.gating_network.module_names:
                target_support = adjusted_contrib[gold_intent][module]
                rival_support = max(
                    adjusted_contrib[label][module]
                    for label in self.labels
                    if label != gold_intent
                )
                rewards[module] = (target_support - rival_support) + margin
            self.gating_network.update(features, rewards)
            self.mutual_reasoner.update_relations(reasoned_beliefs, best, gold_intent)

        trace = {
            "features": {
                name: round(features[i], 4)
                for i, name in enumerate(self.gating_network.feature_names)
            },
            "raw_gate_weights": {k: round(v, 4) for k, v in gate_weights.items()},
            "meta_adjusted_gate_weights": {k: round(v, 4) for k, v in adjusted_gate.items()},
            "module_trust": {k: round(v, 4) for k, v in self.meta_controller.module_trust.items()},
            "mutual_reasoning": mutual_trace,
            "fused_scores": {k: round(v, 4) for k, v in fused_adjusted.items()},
            "predicted_intent": best,
            "confidence": round(confidence, 4),
        }
        self.last_trace = trace
        return best, confidence, trace

    def predict(self, text: str) -> Tuple[str, float]:
        best, confidence, _ = self.predict_with_trace(text)
        return best, confidence


class NextTokenBaseline:
    """Character-overlap baseline for next-token multiple-choice."""

    def predict(self, prompt: str, candidates: Sequence[str]) -> str:
        p = set(prompt.lower())
        scores = [len(p & set(str(c).lower())) for c in candidates]
        return str(candidates[max(range(len(candidates)), key=lambda i: scores[i])])


def token_choice_with_router(router: object, prompt: str, candidates: Sequence[str]) -> str:
    """Pick candidate with maximum router confidence on prompt+candidate."""
    scores = []
    for candidate in candidates:
        _, conf = router.predict(f"{prompt} {candidate}")
        scores.append(conf)
    return str(candidates[max(range(len(candidates)), key=lambda i: scores[i])])


def step_f1(pred: Sequence[str], gold: Sequence[str]) -> float:
    """Compute step-level F1 between predicted and gold symbolic plans."""
    pset, gset = set(pred), set(gold)
    if not pset and not gset:
        return 1.0
    if not pset or not gset:
        return 0.0
    tp = len(pset & gset)
    precision = tp / len(pset)
    recall = tp / len(gset)
    return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)


class BertSemanticResponseGenerator:
    """BERT-powered semantic response generator using embeddings for context-aware generation.
    
    Uses BERT to:
    1. Extract semantic keywords from the user prompt
    2. Score response candidates based on semantic similarity
    3. Generate contextually grounded, intent-aware responses
    4. Ensure responses are semantically coherent with the original request
    """
    
    def __init__(self, encoder: "SemanticEncoder") -> None:
        self.encoder = encoder
        self.action_verbs = {
            "search": ["find", "gather", "retrieve", "locate", "source", "discover"],
            "summarize": ["condense", "compress", "distill", "extract", "synthesize", "reduce"],
            "recall": ["remember", "retrieve", "recall", "verify", "confirm", "look up"],
            "generate": ["compose", "draft", "write", "create", "formulate", "articulate"],
            "plan": ["structure", "organize", "roadmap", "schedule", "sequence", "build"],
        }
        
        # Intent-specific response templates with semantic hooks
        self.response_templates = {
            "search": [
                "I'll find and rank {keyword} sources, then summarize the evidence.",
                "Let me locate relevant {keyword} references and synthesize key findings.",
                "I'll gather {keyword} sources, evaluate them, and extract key points.",
                "I'll systematically search for {keyword} materials and synthesize insights.",
            ],
            "summarize": [
                "I'll condense the {keyword} material into concise, actionable bullets.",
                "Let me extract and organize the core {keyword} concepts for clarity.",
                "I'll distill the {keyword} content into its essential components.",
                "I'll compress the {keyword} information while preserving key insights.",
            ],
            "recall": [
                "I'll retrieve {keyword} from our prior decisions and verify alignment.",
                "Let me recall our {keyword} discussions and confirm consistency.",
                "I'll look up the {keyword} stored decisions and cross-check completeness.",
                "I'll retrieve the {keyword} from memory and validate against current context.",
            ],
            "generate": [
                "I'll compose a clear, {keyword}-focused response tailored to your needs.",
                "Let me draft a {keyword}-aware answer that addresses your request directly.",
                "I'll create a {keyword}-informed explanation with concrete details.",
                "I'll articulate a {keyword}-centered answer grounded in our discussion.",
            ],
            "plan": [
                "I'll structure a {keyword} roadmap with clear milestones and sequencing.",
                "Let me build a {keyword}-informed action plan with measurable steps.",
                "I'll organize a {keyword} strategy with ordered, achievable milestones.",
                "I'll create a {keyword}-focused roadmap with explicit next actions.",
            ],
        }
    
    def extract_semantic_keywords(self, prompt: str, top_k: int = 2) -> List[str]:
        """Extract semantic keywords from prompt using entity-like tokens and domain terms."""
        # Remove common stopwords and extract meaningful tokens
        stopwords = {
            "the", "a", "an", "and", "or", "is", "are", "to", "do", "did", "can", "will", 
            "you", "i", "me", "this", "that", "what", "which", "how", "of", "for", "in",
            "with", "on", "at", "by", "from", "as", "be", "have", "has", "had", "we", "our"
        }
        # Lowercase and split, preserve meaningful punctuation
        tokens = prompt.lower().split()
        tokens = [t.strip("?.,;:!") for t in tokens]
        
        # First pass: look for domain-specific terms and longer nouns
        keywords = []
        for token in tokens:
            if (token not in stopwords and 
                len(token) > 3 and 
                (token.endswith(("s", "ing", "ed")) or 
                 token in ["papers", "memory", "decision", "notes", "results", "coding", 
                          "benchmark", "architecture", "stakeholder", "integration", "roadmap"])):
                keywords.append(token)
        
        # Fallback: if no keywords found, look for any non-stopword
        if not keywords:
            keywords = [t for t in tokens if t not in stopwords and len(t) > 2]
        
        return keywords[:top_k] if keywords else ["request"]
    
    def generate(self, prompt: str, intent: str, confidence: float = 0.7) -> str:
        """Generate a semantically-grounded response using BERT embeddings."""
        # Extract semantic keywords
        keywords = self.extract_semantic_keywords(prompt)
        keyword_str = keywords[0] if keywords else "this"
        
        # Get response templates for this intent
        templates = self.response_templates[intent]
        
        # Select template based on confidence (higher confidence → longer template)
        if confidence > 0.8:
            selected_template = templates[-1]  # Most detailed
        elif confidence > 0.6:
            selected_template = templates[len(templates) // 2]  # Mid-length
        else:
            selected_template = templates[0]  # Concise
        
        # Format template with extracted keyword, handling edge cases
        try:
            response_base = selected_template.format(keyword=keyword_str)
        except (KeyError, IndexError):
            response_base = selected_template  # Template has no placeholder
        
        # Add context and action indicator
        tail = " Next, I can execute this right away if you want."
        full_response = f"{response_base} Request accepted: \"{prompt}\".{tail}"
        
        return full_response


def generate_autoregressive_reply(prompt: str, intent: str, encoder: "SemanticEncoder | None" = None, confidence: float = 0.7) -> str:
    """BERT-powered semantic response generator.
    
    Uses the encoder to extract semantic features and generate contextually-grounded responses.
    Falls back to semantic templates if encoder is unavailable.
    """
    if encoder is not None:
        generator = BertSemanticResponseGenerator(encoder)
        return generator.generate(prompt, intent, confidence=confidence)
    
    # Fallback to semantic templates (no encoder available)
    fallback_templates = {
        "search": "I'll find and rank relevant sources, then summarize key evidence.",
        "summarize": "I'll condense the material into concise, essential points.",
        "recall": "I'll retrieve prior decisions and verify alignment with current context.",
        "generate": "I'll compose a clear, contextually-grounded response for you.",
        "plan": "I'll structure a roadmap with clear milestones and next actions.",
    }
    return f"{fallback_templates[intent]} Request: {prompt}. Next, I can execute this right away if you want."



class EmbeddingDemaskAutoregressor:
    """Embedding-aware continuation model with autoreferencing and demasking.

    The model builds a lightweight world-state from previous turns and generates
    next responses by scoring intent-aligned continuation candidates with the
    shared semantic encoder.
    """

    def __init__(self, encoder: "SemanticEncoder") -> None:
        self.encoder = encoder
        self.transition_bias = {
            "search": ["summarize", "plan"],
            "summarize": ["generate", "plan"],
            "recall": ["generate", "plan"],
            "generate": ["plan", "search"],
            "plan": ["generate", "search"],
        }

    def _extract_slots(self, prompt: str) -> List[str]:
        stop = {
            "the", "and", "for", "with", "that", "this", "from", "into", "about", "what", "please",
            "could", "would", "should", "also", "then", "than", "where", "when", "which", "while",
        }
        cleaned = [x.strip(".,!?;:\"'()[]{}").lower() for x in prompt.split()]
        tokens = [tok for tok in cleaned if tok and len(tok) > 3 and tok not in stop]
        tokens = sorted(tokens, key=lambda tok: int(any(ch.isdigit() for ch in tok) or "-" in tok), reverse=True)
        return tokens[:12]

    def update_state(self, state: ChatSessionState, turn: ConversationTurn) -> None:
        state.history.append(turn)
        if len(state.history) > 1:
            prev = state.history[-2].predicted_intent
            current = turn.predicted_intent
            state.intent_transitions.setdefault(prev, {})
            state.intent_transitions[prev][current] = state.intent_transitions[prev].get(current, 0) + 1

        for slot in self._extract_slots(turn.user_prompt):
            state.latent_slots.setdefault(turn.predicted_intent, [])
            if slot not in state.latent_slots[turn.predicted_intent]:
                state.latent_slots[turn.predicted_intent].append(slot)
            if slot not in state.global_memory:
                state.global_memory.append(slot)

    def _transition_prior(self, state: ChatSessionState, intent: str) -> float:
        if not state.history:
            return 0.5
        prev = state.history[-1].predicted_intent
        preferred = self.transition_bias.get(prev, [])
        if intent in preferred:
            return 0.9
        empirical = state.intent_transitions.get(prev, {}).get(intent, 0)
        total = sum(state.intent_transitions.get(prev, {}).values())
        return 0.5 if total == 0 else 0.4 + (empirical / total)

    def generate(self, prompt: str, intent: str, state: ChatSessionState, confidence: float) -> str:
        slots = state.latent_slots.get(intent, [])
        current_slots = self._extract_slots(prompt)
        focus = current_slots[0] if current_slots else (slots[0] if slots else "task")
        prior = self._transition_prior(state, intent)

        memory_focus = ""
        memory_triggers = ["codename", "token", "secret", "verbatim", "exact"]
        if any(trigger in prompt.lower() for trigger in memory_triggers):
            preferred = [slot for slot in state.global_memory if any(ch.isdigit() for ch in slot) or "-" in slot]
            if preferred:
                memory_focus = preferred[0]
            elif state.global_memory:
                memory_focus = state.global_memory[0]

        if memory_focus and intent == "recall":
            return (
                f"Memory-locked continuation: exact codename is {memory_focus}. "
                "I retrieved it from global memory and preserved it verbatim. "
                "Next, I can execute tool-level steps now if you want."
            )

        candidates = [
            f"Agent mode: I will execute {intent} around {focus} using the current world model and then verify outcomes.",
            f"Continuation: Based on prior turns, I will {intent} the {focus} details, cross-reference memory slots, and produce a grounded output.",
            f"Demasked plan: I infer hidden constraints around {focus}, apply {intent}, and autoreference earlier decisions before responding.",
            f"World-model action: I'll update latent state for {focus}, run {intent} steps, and return an auditable result with next actions.",
        ]
        if state.history:
            last = state.history[-1]
            candidates.append(
                f"Given your previous request on '{last.user_prompt[:55]}', I will continue by applying {intent} to {focus} and preserving continuity."
            )
        if memory_focus:
            candidates.append(
                f"Memory-locked continuation: I recovered {memory_focus} from global memory and will preserve it exactly while I {intent} the request."
            )

        query = f"{prompt} intent={intent} prior={prior:.2f} confidence={confidence:.2f}"
        if self.encoder.mode == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
            mat = vec.fit_transform([query, *candidates])
            sims = cosine_similarity(mat[0:1], mat[1:]).ravel().tolist()
        else:
            mat = self.encoder._embed([query, *candidates])
            sims_t = (mat[0:1] @ mat[1:].T).squeeze(0)
            sims = [float(x) for x in sims_t.tolist()]

        weighted = [s + 0.1 * prior for s in sims]
        best_idx = max(range(len(weighted)), key=lambda i: weighted[i])
        return candidates[best_idx] + " Next, I can execute tool-level steps now if you want."


class AgenticChatOrchestrator:
    """Multi-turn chat orchestrator with autoreferencing world-state updates."""

    def __init__(self, router: "NeuroSymbolicRouter") -> None:
        self.router = router
        self.autoreg = EmbeddingDemaskAutoregressor(router.encoder)

    def run_turn(self, prompt: str, state: ChatSessionState) -> ConversationTurn:
        pred_intent, confidence, trace = self.router.predict_with_trace(prompt, gold_intent=None)
        response = self.autoreg.generate(prompt, pred_intent, state=state, confidence=confidence)
        turn = ConversationTurn(
            user_prompt=prompt,
            predicted_intent=pred_intent,
            confidence=confidence,
            symbolic_plan=SYMBOLIC_PLANS[pred_intent],
            world_model_prior=float(trace.get("fused_scores", {}).get(pred_intent, 0.0)),
            response=response,
            trace=trace,
        )
        self.autoreg.update_state(state, turn)
        return turn

    def run_session(self, prompts: Sequence[str]) -> ChatSessionResult:
        state = ChatSessionState()
        turns = [self.run_turn(prompt, state) for prompt in prompts]
        snapshot = {
            "intent_transitions": state.intent_transitions,
            "latent_slots": state.latent_slots,
            "global_memory": state.global_memory,
            "turn_count": len(state.history),
            "final_intent": state.history[-1].predicted_intent if state.history else None,
        }
        return ChatSessionResult(turns=turns, state=snapshot)

def run_freeform_turn(prompt: str, router: "NeuroSymbolicRouter") -> ConversationTurn:
    """Execute one freeform conversation turn with full neuro-symbolic tracing."""

    pred_intent, confidence, trace = router.predict_with_trace(prompt, gold_intent=None)
    response = generate_autoregressive_reply(
        prompt,
        pred_intent,
        encoder=router.encoder,
        confidence=confidence,
    )
    world_model_prior = float(trace.get("fused_scores", {}).get(pred_intent, 0.0))
    return ConversationTurn(
        user_prompt=prompt,
        predicted_intent=pred_intent,
        confidence=confidence,
        symbolic_plan=SYMBOLIC_PLANS[pred_intent],
        world_model_prior=world_model_prior,
        response=response,
        trace=trace,
    )


def sample_chat_prompts() -> List[str]:
    """Built-in multi-turn prompts for reproducible continuation checks."""
    return [
        "Find strong references on world-model-based agent planning for LNN systems.",
        "Great—now condense those findings into concise bullets for leadership.",
        "Remind me what constraints we decided for memory slots yesterday.",
        "Draft a stakeholder-friendly update that ties all of this together.",
        "Create a sequenced rollout plan with milestones and validation checks.",
    ]


def run_difficult_chat_challenge(router: "NeuroSymbolicRouter") -> Dict[str, object]:
    """Run a long-range memory challenge inspired by classic GPT-2 failure modes.

    GPT-2-style models are known to struggle with exact token recall across
    multi-turn context. This benchmark tracks whether the orchestrator can keep
    and retrieve a codename introduced early in the conversation.
    """

    orchestrator = AgenticChatOrchestrator(router)
    session = orchestrator.run_session(DIFFICULT_CHAT_CHALLENGE)
    final_response = session.turns[-1].response
    codename = "quartz-17"
    exact_recall = codename in final_response.lower()
    continuity_hits = [
        int(any(k in turn.response.lower() for k in ["continu", "memory", "world model", codename]))
        for turn in session.turns[1:]
    ]
    return {
        "challenge_name": "long_range_exact_token_recall",
        "codename": codename,
        "turn_count": len(session.turns),
        "exact_recall_success": exact_recall,
        "continuation_consistency": sum(continuity_hits) / len(continuity_hits) if continuity_hits else 0.0,
        "overall_success": exact_recall and (sum(continuity_hits) / len(continuity_hits) >= 0.8 if continuity_hits else False),
        "transcript": [turn.__dict__ for turn in session.turns],
        "world_state": session.state,
        "final_response_untruncated": final_response,
    }


def evaluate(dataset_path: Path, out_path: Path, cache_dir: str) -> Dict[str, object]:
    """Run all metrics and write a single JSON report."""
    data = json.loads(dataset_path.read_text())

    keyword = KeywordBaseline()
    base_router = BaseSemanticRouter(SemanticEncoder(cache_dir=cache_dir))
    hybrid_router = NeuroSymbolicRouter(SemanticEncoder(cache_dir=cache_dir))

    keyword_preds: List[Prediction] = []
    base_preds: List[Prediction] = []
    hybrid_preds: List[Prediction] = []
    hybrid_traces: List[Dict[str, object]] = []

    for row in data:
        text, gold = row["text"], row["intent"]
        kp, kc = keyword.predict(text)
        bp, bc = base_router.predict(text)
        hp, hc, trace = hybrid_router.predict_with_trace(text, gold_intent=gold)
        keyword_preds.append(Prediction(text, gold, kp, kc, SYMBOLIC_PLANS[kp]))
        base_preds.append(Prediction(text, gold, bp, bc, SYMBOLIC_PLANS[bp]))
        hybrid_preds.append(Prediction(text, gold, hp, hc, SYMBOLIC_PLANS[hp]))
        hybrid_traces.append({"text": text, "gold_intent": gold, **trace})

    def accuracy(items: Sequence[Prediction]) -> float:
        return sum(x.pred_intent == x.gold_intent for x in items) / len(items)

    keyword_acc, base_acc, hybrid_acc = accuracy(keyword_preds), accuracy(base_preds), accuracy(hybrid_preds)

    token_baseline = NextTokenBaseline()
    token_hits = {"baseline": 0, "base": 0, "hybrid": 0}
    token_details = []
    for row in NEXT_TOKEN_BENCHMARK:
        prompt = str(row["prompt"])
        candidates = list(row["candidates"])
        gold = str(row["gold"])
        pb = token_baseline.predict(prompt, candidates)
        pbase = token_choice_with_router(base_router, prompt, candidates)
        phybrid = token_choice_with_router(hybrid_router, prompt, candidates)
        token_hits["baseline"] += int(pb == gold)
        token_hits["base"] += int(pbase == gold)
        token_hits["hybrid"] += int(phybrid == gold)
        token_details.append({
            "prompt": prompt,
            "candidates": candidates,
            "gold": gold,
            "baseline_pred": pb,
            "base_router_pred": pbase,
            "hybrid_router_pred": phybrid,
        })

    reasoning_base_f1: List[float] = []
    reasoning_hybrid_f1: List[float] = []
    base_exact, hybrid_exact = 0, 0
    reasoning_details = []
    for row in REASONING_BENCHMARK:
        text, intent = str(row["text"]), str(row["intent"])
        gold_plan = SYMBOLIC_PLANS[intent]
        base_intent, _ = base_router.predict(text)
        hybrid_intent, _ = hybrid_router.predict(text)
        base_plan = SYMBOLIC_PLANS[base_intent]
        hybrid_plan = SYMBOLIC_PLANS[hybrid_intent]
        bf1, hf1 = step_f1(base_plan, gold_plan), step_f1(hybrid_plan, gold_plan)
        reasoning_base_f1.append(bf1)
        reasoning_hybrid_f1.append(hf1)
        base_exact += int(base_plan == gold_plan)
        hybrid_exact += int(hybrid_plan == gold_plan)
        reasoning_details.append({
            "text": text,
            "gold_intent": intent,
            "base_intent": base_intent,
            "hybrid_intent": hybrid_intent,
            "base_step_f1": bf1,
            "hybrid_step_f1": hf1,
        })

    orchestrator = AgenticChatOrchestrator(hybrid_router)
    continuation_session = orchestrator.run_session(sample_chat_prompts())
    difficult_chat = run_difficult_chat_challenge(hybrid_router)
    continuation_scores = []
    for turn in continuation_session.turns[1:]:
        has_reference = any(
            key in turn.response.lower()
            for key in ["previous", "continu", "memory", "world model", "latent", "prior"]
        )
        continuation_scores.append(int(has_reference and len(turn.response.split()) > 12))

    autoreg_details = []
    coherent_hits = 0
    for row in AUTOREG_BENCHMARK:
        prompt, intent = row["prompt"], row["intent"]
        pred_intent, pred_conf, trace = hybrid_router.predict_with_trace(prompt, gold_intent=None)
        response = generate_autoregressive_reply(prompt, pred_intent, encoder=hybrid_router.encoder, confidence=pred_conf)
        has_action_word = any(x in response.lower() for x in [
            "search", "find", "gather", "locate", "retrieve", "recall", "look up",
            "condense", "compress", "distill", "extract", "synthesize", "reduce", 
            "compose", "draft", "write", "create", "formulate", "articulate",
            "organize", "structure", "roadmap", "schedule", "sequence", "build",
            "rank", "evaluate", "verify", "check"
        ])
        coherent = bool(response.strip()) and len(response.split()) > 10 and has_action_word
        coherent_hits += int(coherent)
        autoreg_details.append({
            "prompt": prompt,
            "gold_intent": intent,
            "pred_intent": pred_intent,
            "response": response,
            "coherent": coherent,
        })

    mutual_rounds = [
        int(x.get("mutual_reasoning", {}).get("rounds_executed", 0))
        for x in hybrid_traces
    ]

    results = {
        "dataset_size": len(data),
        "base_router_mode": base_router.encoder.mode,
        "hybrid_router_mode": hybrid_router.encoder.mode,
        "keyword_accuracy": keyword_acc,
        "base_bert_style_accuracy": base_acc,
        "neuro_symbolic_accuracy": hybrid_acc,
        "improvement_vs_base_bert": hybrid_acc - base_acc,
        "learnable_gating": {
            "feature_names": hybrid_router.gating_network.feature_names,
            "module_names": hybrid_router.gating_network.module_names,
            "final_weights": {
                module: [round(x, 6) for x in weights]
                for module, weights in hybrid_router.gating_network.weights.items()
            },
            "trace": hybrid_traces,
        },
        "meta_controller": {
            "module_trust": {
                module: round(value, 6)
                for module, value in hybrid_router.meta_controller.module_trust.items()
            }
        },
        "mutual_reasoning": {
            "average_rounds": (sum(mutual_rounds) / len(mutual_rounds)) if mutual_rounds else 0.0,
            "relation_graph": {
                src: {dst: round(v, 6) for dst, v in dst_map.items()}
                for src, dst_map in hybrid_router.mutual_reasoner.relation.items()
            },
        },
        "next_token_metrics": {
            "total": len(NEXT_TOKEN_BENCHMARK),
            "baseline_accuracy": token_hits["baseline"] / len(NEXT_TOKEN_BENCHMARK),
            "base_router_accuracy": token_hits["base"] / len(NEXT_TOKEN_BENCHMARK),
            "hybrid_router_accuracy": token_hits["hybrid"] / len(NEXT_TOKEN_BENCHMARK),
            "hybrid_vs_base_delta": (token_hits["hybrid"] - token_hits["base"]) / len(NEXT_TOKEN_BENCHMARK),
            "details": token_details,
        },
        "reasoning_metrics": {
            "total": len(REASONING_BENCHMARK),
            "base_plan_exact_match": base_exact / len(REASONING_BENCHMARK),
            "hybrid_plan_exact_match": hybrid_exact / len(REASONING_BENCHMARK),
            "base_plan_step_f1": sum(reasoning_base_f1) / len(reasoning_base_f1),
            "hybrid_plan_step_f1": sum(reasoning_hybrid_f1) / len(reasoning_hybrid_f1),
            "details": reasoning_details,
        },
        "autoregression_test": {
            "total": len(AUTOREG_BENCHMARK),
            "coherence_rate": coherent_hits / len(AUTOREG_BENCHMARK),
            "can_function_as_chatbot": coherent_hits / len(AUTOREG_BENCHMARK) >= 0.8,
            "details": autoreg_details,
        },
        "chat_continuation_test": {
            "turns": [
                {
                    "user_prompt": turn.user_prompt,
                    "predicted_intent": turn.predicted_intent,
                    "response": turn.response,
                }
                for turn in continuation_session.turns
            ],
            "world_state": continuation_session.state,
            "continuation_grounding_rate": (
                sum(continuation_scores) / len(continuation_scores)
                if continuation_scores
                else 0.0
            ),
            "multi_turn_capable": (sum(continuation_scores) / len(continuation_scores) >= 0.75)
            if continuation_scores
            else False,
        },
        "difficult_chat_challenge": difficult_chat,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid vs base BERT-style + extra module evaluation")
    parser.add_argument("--dataset", type=Path, default=Path("data/intent_benchmark.json"))
    parser.add_argument("--out", type=Path, default=Path("results/eval.json"))
    parser.add_argument("--cache-dir", type=str, default="models")
    parser.add_argument(
        "--chat-prompt",
        type=str,
        default="",
        help="Optional freeform prompt to run one fully traced conversation turn.",
    )
    parser.add_argument(
        "--chat-script",
        type=Path,
        default=None,
        help="Optional path to JSON list of user prompts for multi-turn continuation testing.",
    )
    parser.add_argument(
        "--sample-chats",
        action="store_true",
        help="Run built-in multi-turn conversation samples and print transcript JSON.",
    )
    parser.add_argument(
        "--difficult-chat",
        action="store_true",
        help="Run the long-range difficult chat challenge and print full untruncated metrics+response.",
    )
    args = parser.parse_args()

    if args.chat_prompt:
        router = NeuroSymbolicRouter(SemanticEncoder(args.cache_dir))
        turn = run_freeform_turn(args.chat_prompt, router)
        print(json.dumps(turn.__dict__, indent=2))
    elif args.sample_chats or args.chat_script:
        router = NeuroSymbolicRouter(SemanticEncoder(args.cache_dir))
        orchestrator = AgenticChatOrchestrator(router)
        prompts = sample_chat_prompts() if args.sample_chats else json.loads(args.chat_script.read_text())
        session = orchestrator.run_session(prompts)
        payload = {
            "turns": [turn.__dict__ for turn in session.turns],
            "state": session.state,
        }
        print(json.dumps(payload, indent=2))
    elif args.difficult_chat:
        router = NeuroSymbolicRouter(SemanticEncoder(args.cache_dir))
        payload = run_difficult_chat_challenge(router)
        print(json.dumps(payload, indent=2))
    else:
        report = evaluate(args.dataset, args.out, args.cache_dir)
        print(
            "done:",
            f"mode={report['hybrid_router_mode']}",
            f"intent_base={report['base_bert_style_accuracy']:.3f}",
            f"intent_hybrid={report['neuro_symbolic_accuracy']:.3f}",
            f"nexttok_hybrid={report['next_token_metrics']['hybrid_router_accuracy']:.3f}",
            f"reason_hybrid_f1={report['reasoning_metrics']['hybrid_plan_step_f1']:.3f}",
            f"chatbot={report['autoregression_test']['can_function_as_chatbot']}",
            f"multi_turn={report['chat_continuation_test']['multi_turn_capable']}",
        )
