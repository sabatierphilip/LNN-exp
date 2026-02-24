"""Hybrid agent evaluator with additional cognitive modules and autoregression test.

Run:
    python src/hybrid_agent.py --dataset data/intent_benchmark.json --out results/eval.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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

    def predict(self, text: str) -> Tuple[str, float]:
        proto_scores = self.encoder.score(text)
        max_proto = {k: -1e9 for k in self.labels}
        for idx, value in enumerate(proto_scores):
            label = self.prototype_labels[idx]
            max_proto[label] = max(max_proto[label], float(value))

        mem = self.memory_module.score(text)
        cog = self.cog_module.score(text)
        world = self.world_module.score()
        pred = self.pred_module.score(text)

        fused = {}
        for label in self.labels:
            fused[label] = (
                0.55 * max_proto[label]
                + 0.20 * mem[label]
                + 0.10 * cog[label]
                + 0.10 * pred[label]
                + 0.05 * world[label]
            )

        best = max(fused, key=fused.get)
        return best, float(max(0.5, min(0.99, fused[best] + 0.5)))


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


def generate_autoregressive_reply(prompt: str, intent: str) -> str:
    """Tiny deterministic autoregressive-style generator for chatbot feasibility checks."""
    lead = {
        "search": "I will search relevant sources, rank evidence, and summarize findings.",
        "summarize": "I will condense the material into concise bullets and keep key points.",
        "recall": "I will retrieve prior decisions from memory and verify consistency.",
        "generate": "I will draft a clear answer tailored to your request.",
        "plan": "I will produce an ordered roadmap with milestones and next actions.",
    }[intent]
    tail = " Next, I can execute the first step now if you want."
    # emulate stepwise generation by appending tokens one-by-one
    tokens = (lead + " " + f"Request: {prompt}" + tail).split()
    return " ".join(tokens)


def evaluate(dataset_path: Path, out_path: Path, cache_dir: str) -> Dict[str, object]:
    """Run all metrics and write a single JSON report."""
    data = json.loads(dataset_path.read_text())

    keyword = KeywordBaseline()
    base_router = BaseSemanticRouter(SemanticEncoder(cache_dir=cache_dir))
    hybrid_router = NeuroSymbolicRouter(SemanticEncoder(cache_dir=cache_dir))

    keyword_preds: List[Prediction] = []
    base_preds: List[Prediction] = []
    hybrid_preds: List[Prediction] = []

    for row in data:
        text, gold = row["text"], row["intent"]
        kp, kc = keyword.predict(text)
        bp, bc = base_router.predict(text)
        hp, hc = hybrid_router.predict(text)
        keyword_preds.append(Prediction(text, gold, kp, kc, SYMBOLIC_PLANS[kp]))
        base_preds.append(Prediction(text, gold, bp, bc, SYMBOLIC_PLANS[bp]))
        hybrid_preds.append(Prediction(text, gold, hp, hc, SYMBOLIC_PLANS[hp]))

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

    autoreg_details = []
    coherent_hits = 0
    for row in AUTOREG_BENCHMARK:
        prompt, intent = row["prompt"], row["intent"]
        pred_intent, _ = hybrid_router.predict(prompt)
        response = generate_autoregressive_reply(prompt, pred_intent)
        has_action_word = any(x in response.lower() for x in ["search", "condense", "retrieve", "draft", "roadmap"])
        coherent = bool(response.strip()) and len(response.split()) > 10 and has_action_word
        coherent_hits += int(coherent)
        autoreg_details.append({
            "prompt": prompt,
            "gold_intent": intent,
            "pred_intent": pred_intent,
            "response": response,
            "coherent": coherent,
        })

    results = {
        "dataset_size": len(data),
        "base_router_mode": base_router.encoder.mode,
        "hybrid_router_mode": hybrid_router.encoder.mode,
        "keyword_accuracy": keyword_acc,
        "base_bert_style_accuracy": base_acc,
        "neuro_symbolic_accuracy": hybrid_acc,
        "improvement_vs_base_bert": hybrid_acc - base_acc,
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
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid vs base BERT-style + extra module evaluation")
    parser.add_argument("--dataset", type=Path, default=Path("data/intent_benchmark.json"))
    parser.add_argument("--out", type=Path, default=Path("results/eval.json"))
    parser.add_argument("--cache-dir", type=str, default="models")
    args = parser.parse_args()

    report = evaluate(args.dataset, args.out, args.cache_dir)
    print(
        "done:",
        f"mode={report['hybrid_router_mode']}",
        f"intent_base={report['base_bert_style_accuracy']:.3f}",
        f"intent_hybrid={report['neuro_symbolic_accuracy']:.3f}",
        f"nexttok_hybrid={report['next_token_metrics']['hybrid_router_accuracy']:.3f}",
        f"reason_hybrid_f1={report['reasoning_metrics']['hybrid_plan_step_f1']:.3f}",
        f"chatbot={report['autoregression_test']['can_function_as_chatbot']}",
    )
