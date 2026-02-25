"""Guided Iterative Semantic Diffusion (GISD) text generation for BERT masked LMs.

GISD is a non-autoregressive decoder that starts from a fully masked suffix and
iteratively denoises all output positions in parallel while allowing revisions.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple


@dataclass
class GISDConfig:
    """Hyperparameters controlling Guided Iterative Semantic Diffusion behavior."""

    diffusion_steps: int = 8  # T — more steps = more coherent but slower
    reveal_fraction: float = 0.35  # fraction of masked positions revealed per step
    remask_fraction: float = 0.12  # fraction of revealed tokens re-opened for revision
    temp_start: float = 1.6  # exploration temperature at step 0
    temp_end: float = 0.55  # commitment temperature at step T
    temp_gamma: float = 1.4  # annealing curve sharpness
    steering_alpha_start: float = 0.80
    steering_alpha_end: float = 0.15
    top_k_vocab: int = 512  # restrict steering to top-k vocab tokens
    top_p: float = 0.90
    repetition_penalty: float = 1.25
    min_tokens: int = 18
    max_tokens: int = 80
    intent_length_hints: Dict[str, float] = field(
        default_factory=lambda: {
            "search": 1.0,
            "summarize": 0.72,
            "recall": 0.85,
            "generate": 1.2,
            "plan": 1.3,
        }
    )
    context_max_tokens: int = 64  # prompt prefix budget
    revision_conf_base: float = 0.25  # baseline remask confidence threshold
    entropy_boost: float = 0.30  # boosts reveal/remask aggressiveness when uncertainty is high
    prompt_complexity_weight: float = 0.25  # scales output budget by prompt complexity


class GuidedIterativeSemanticDiffusion:
    """Generate text using parallel iterative denoising inside a masked-LM.

    Why the semantic steering term ``E @ target_vec`` is principled:
    - ``E`` contains one embedding vector per vocabulary token learned during BERT
      pretraining, and those rows encode latent semantic/syntactic regularities.
    - ``target_vec`` is the prompt-level meaning vector (CLS embedding) projected
      into the exact same embedding space.
    - By unit-normalizing both, the dot product becomes cosine similarity, yielding
      a vocabulary-wide semantic relevance score for the current prompt.
    - Adding this score to masked-token logits injects global intent alignment at
      *every position simultaneously*, which leverages bidirectional context and is
      fundamentally different from left-to-right autoregressive decoding.
    - Alpha annealing starts with strong steering (topic coherence) and gradually
      decays so BERT's native token-compatibility and grammar priors dominate late
      commitment steps.
    """

    def __init__(self, encoder: object, config: GISDConfig | None = None) -> None:
        self.encoder = encoder
        self.config = config or GISDConfig()
        self.torch = encoder._torch
        self.F = encoder._F
        self.tokenizer = encoder.tokenizer
        self.masked_lm = encoder.masked_lm
        self.model = encoder.model
        self.device = encoder.device
        self._unit_embeddings = None

    def _ensure_unit_embeddings(self):
        if self._unit_embeddings is None:
            emb = self.masked_lm.get_input_embeddings().weight.detach()
            self._unit_embeddings = self.F.normalize(emb, p=2, dim=-1)
        return self._unit_embeddings

    @staticmethod
    def _anneal(start: float, end: float, frac: float, gamma: float = 1.0) -> float:
        frac = min(max(frac, 0.0), 1.0)
        return float(end + (start - end) * ((1.0 - frac) ** gamma))

    @staticmethod
    def _linear_anneal(start: float, end: float, frac: float) -> float:
        frac = min(max(frac, 0.0), 1.0)
        return float(start + (end - start) * frac)

    def _prefix_ids(self, prompt: str) -> List[int]:
        ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        return list(ids[: self.config.context_max_tokens])

    def _prompt_complexity(self, prompt: str) -> float:
        words = [w.strip(".,!?;:\"'()[]{}") for w in prompt.split() if w.strip()]
        if not words:
            return 0.0
        unique_ratio = len({w.lower() for w in words}) / len(words)
        avg_len = sum(len(w) for w in words) / len(words)
        punctuation = sum(ch in ",;:.!?" for ch in prompt) / max(1, len(prompt))
        raw = 0.45 * unique_ratio + 0.45 * min(1.0, avg_len / 8.0) + 0.10 * min(1.0, punctuation * 20)
        return float(min(1.0, max(0.0, raw)))

    def _intent_budget(self, intent: str, prompt: str) -> int:
        multiplier = self.config.intent_length_hints.get(intent, 1.0)
        base = 0.5 * (self.config.min_tokens + self.config.max_tokens)
        complexity = self._prompt_complexity(prompt)
        dynamic = base * (1.0 + self.config.prompt_complexity_weight * (complexity - 0.5))
        n = int(round(dynamic * multiplier))
        return max(self.config.min_tokens, min(self.config.max_tokens, n))

    def _semantic_target_vec(self, prompt: str):
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.config.context_max_tokens)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with self.torch.no_grad():
            cls = self.model(**encoded).last_hidden_state[:, 0, :]
        return self.F.normalize(cls.squeeze(0), p=2, dim=-1)

    def _compute_entropy(self, probs) -> float:
        eps = 1e-12
        entropy = -(probs * self.torch.log(probs + eps)).sum().item()
        max_entropy = math.log(max(2, probs.numel()))
        return float(entropy / max_entropy)

    def _apply_top_p(self, logits, top_p: float):
        sorted_logits, sorted_idx = self.torch.sort(logits, descending=True)
        probs = self.torch.softmax(sorted_logits, dim=-1)
        csum = self.torch.cumsum(probs, dim=-1)
        keep = csum <= top_p
        if keep.numel() and not bool(keep[0]):
            keep[0] = True
        filtered = self.torch.full_like(logits, float("-inf"))
        filtered[sorted_idx[keep]] = logits[sorted_idx[keep]]
        return filtered

    def _dynamic_top_p(self, frac: float, entropy_ratio: float) -> float:
        # higher uncertainty -> retain more mass; late steps shrink support.
        base = self.config.top_p
        delta = 0.08 * (entropy_ratio - 0.5) - 0.10 * frac
        return float(min(0.98, max(0.72, base + delta)))

    def _sample_position(self, raw_logits, token_counts: Dict[int, int], steering_scores, alpha: float, temp: float, frac: float):
        logits = raw_logits / max(1e-6, temp)

        if token_counts:
            penalty_scale = math.log(max(1.01, self.config.repetition_penalty))
            for tok, freq in token_counts.items():
                if 0 <= tok < logits.shape[0]:
                    logits[tok] = logits[tok] - penalty_scale * (1.0 + math.sqrt(freq))

        if self.config.top_k_vocab > 0:
            top_vals, top_idx = self.torch.topk(logits, k=min(self.config.top_k_vocab, logits.shape[-1]))
            steered = top_vals + alpha * steering_scores[top_idx]
            logits2 = self.torch.full_like(logits, float("-inf"))
            logits2[top_idx] = steered
            logits = logits2
        else:
            logits = logits + alpha * steering_scores

        prelim_probs = self.torch.softmax(logits, dim=-1)
        entropy_ratio = self._compute_entropy(prelim_probs)
        dynamic_top_p = self._dynamic_top_p(frac, entropy_ratio)
        filtered = self._apply_top_p(logits, dynamic_top_p)
        probs = self.torch.softmax(filtered, dim=-1)

        chosen = int(self.torch.multinomial(probs, 1).item())
        confidence = float(probs[chosen].item())
        return chosen, confidence, entropy_ratio

    def _compute_reveal_count(self, masked_left: int, frac: float, mean_entropy: float) -> int:
        if masked_left <= 0:
            return 0
        ramp = 0.5 + 0.5 * frac
        uncertainty_boost = 1.0 + self.config.entropy_boost * (mean_entropy - 0.5)
        reveal_ratio = self.config.reveal_fraction * ramp * uncertainty_boost
        reveal_ratio = min(0.95, max(0.05, reveal_ratio))
        return max(1, int(math.ceil(masked_left * reveal_ratio)))

    def _compute_revision_threshold(self, frac: float, mean_conf: float, mean_entropy: float) -> float:
        # higher entropy and lower confidence increase remasking pressure.
        confidence_term = 0.35 * max(0.0, 0.45 - mean_conf)
        entropy_term = 0.20 * max(0.0, mean_entropy - 0.55)
        late_step_relax = 0.12 * frac
        threshold = self.config.revision_conf_base + confidence_term + entropy_term - late_step_relax
        return float(min(0.85, max(0.10, threshold)))

    def _clean_text(self, text: str) -> str:
        text = text.replace(" ##", "")
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()
        if text and text[0].isalpha():
            text = text[0].upper() + text[1:]
        return text

    def _diffuse_once(
        self,
        seq,
        revealed,
        output_positions,
        target_vec,
        step_index: int,
        token_counts: Dict[int, int],
        revision_conf_threshold: float | None = None,
    ):
        steps = max(1, self.config.diffusion_steps)
        frac = step_index / max(1, steps - 1)
        temp = self._anneal(self.config.temp_start, self.config.temp_end, frac, self.config.temp_gamma)
        alpha = self._linear_anneal(self.config.steering_alpha_start, self.config.steering_alpha_end, frac)

        mask_id = int(self.tokenizer.mask_token_id)
        steering_scores = self._ensure_unit_embeddings() @ target_vec

        with self.torch.no_grad():
            logits = self.masked_lm(input_ids=seq.unsqueeze(0)).logits[0]

        candidates: List[Tuple[int, int, float, float]] = []
        sampled_conf: Dict[int, float] = {}
        sampled_entropy: List[float] = []

        for local_idx, pos in enumerate(output_positions):
            if not bool(revealed[local_idx]):
                tok, conf, entropy_ratio = self._sample_position(logits[pos], token_counts, steering_scores, alpha, temp, frac)
                seq[pos] = tok
                candidates.append((local_idx, tok, conf, entropy_ratio))
                sampled_conf[local_idx] = conf
                sampled_entropy.append(entropy_ratio)

        mean_entropy = float(sum(sampled_entropy) / len(sampled_entropy)) if sampled_entropy else 0.5
        masked_left = int((~revealed).sum().item())
        reveal_count = self._compute_reveal_count(masked_left, frac, mean_entropy)

        candidates.sort(key=lambda x: x[2], reverse=True)
        chosen = candidates[:reveal_count]
        for local_idx, tok, _, _ in chosen:
            revealed[local_idx] = True
            token_counts[tok] = token_counts.get(tok, 0) + 1

        if step_index >= 1:
            open_revealed = [i for i in range(len(output_positions)) if bool(revealed[i])]
            mean_conf = float(sum(sampled_conf.values()) / len(sampled_conf)) if sampled_conf else 1.0
            threshold = revision_conf_threshold
            if threshold is None:
                threshold = self._compute_revision_threshold(frac, mean_conf, mean_entropy)

            if open_revealed:
                max_by_fraction = max(1, int(math.ceil(len(open_revealed) * self.config.remask_fraction)))
                low_conf = sorted([(i, sampled_conf.get(i, 1.0)) for i in open_revealed], key=lambda x: x[1])
                remaskable = [x for x in low_conf if x[1] < threshold][:max_by_fraction]
                for local_idx, _ in remaskable:
                    pos = output_positions[local_idx]
                    old = int(seq[pos].item())
                    if old in token_counts:
                        token_counts[old] = max(0, token_counts[old] - 1)
                    seq[pos] = mask_id
                    revealed[local_idx] = False

        return seq, revealed

    def generate(self, prompt: str, intent: str = "generate") -> Tuple[str, List[int]]:
        if getattr(self.encoder, "mode", "tfidf") != "bert":
            fallback = (
                f"I can help with {intent}: {prompt}. "
                "I will provide a concise, practical response grounded in the request context."
            )
            words = fallback.split()
            return " ".join(words[: max(5, min(len(words), self.config.min_tokens))]), []

        with self.torch.no_grad():
            prefix = self._prefix_ids(prompt)
            n_output = self._intent_budget(intent, prompt)
            cls_id = int(self.tokenizer.cls_token_id)
            sep_id = int(self.tokenizer.sep_token_id)
            mask_id = int(self.tokenizer.mask_token_id)

            seq_ids = [cls_id, *prefix, *([mask_id] * n_output), sep_id]
            seq = self.torch.tensor(seq_ids, device=self.device, dtype=self.torch.long)
            out_start = 1 + len(prefix)
            output_positions = list(range(out_start, out_start + n_output))
            revealed = self.torch.zeros(n_output, device=self.device, dtype=self.torch.bool)
            token_counts: Dict[int, int] = {}
            target_vec = self._semantic_target_vec(prompt)

            for step in range(self.config.diffusion_steps):
                seq, revealed = self._diffuse_once(seq, revealed, output_positions, target_vec, step, token_counts)

            if int((~revealed).sum().item()) > 0:
                logits = self.masked_lm(input_ids=seq.unsqueeze(0)).logits[0]
                for local_idx, pos in enumerate(output_positions):
                    if not bool(revealed[local_idx]):
                        seq[pos] = int(self.torch.argmax(logits[pos]).item())
                        revealed[local_idx] = True

            token_ids = [int(seq[p].item()) for p in output_positions]
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            return self._clean_text(text), token_ids


class GISDResponseGenerator:
    """Integration shim for using GISD inside the neuro-symbolic router pipeline."""

    def __init__(self, encoder: object | None, config: GISDConfig | None = None) -> None:
        self.encoder = encoder
        self.config = config or GISDConfig()
        self.gisd = None
        if encoder is not None and getattr(encoder, "mode", None) == "bert":
            self.gisd = GuidedIterativeSemanticDiffusion(encoder, self.config)

    def _tool_outputs_to_english(self, tool_outputs: Sequence[object] | None) -> str:
        if not tool_outputs:
            return "no external tool findings were available"

        def flatten(value: object) -> str:
            if isinstance(value, dict):
                pairs = [f"{k} {flatten(v)}" for k, v in value.items()]
                return "; ".join(pairs)
            if isinstance(value, (list, tuple)):
                return ", ".join(flatten(v) for v in value[:4])
            return str(value)

        parts: List[str] = []
        for item in tool_outputs:
            if isinstance(item, dict):
                tool = str(item.get("tool", "tool"))
                payload = {k: v for k, v in item.items() if k != "tool"}
                details = flatten(payload) if payload else "usable intermediate findings"
                parts.append(f"{tool} reported {details}")
            else:
                parts.append(flatten(item))

        text = "; ".join(parts)
        return re.sub(r"[{}\[\]\"]", "", text)

    def _build_context_prefix(
        self,
        prompt: str,
        intent: str,
        tool_outputs: Sequence[object] | None,
        history: Sequence[str] | None,
    ) -> str:
        intro = {
            "search": "Searching for relevant evidence",
            "summarize": "Summarizing the core points",
            "recall": "Recalling earlier decisions",
            "generate": "Drafting a clear explanation",
            "plan": "Planning concrete next steps",
        }.get(intent, "Responding helpfully")

        history_snippet = ""
        if history:
            recent = [h.strip() for h in history if str(h).strip()][-2:]
            history_snippet = f" while preserving continuity with {'; '.join(recent)}"

        tools_plain = self._tool_outputs_to_english(tool_outputs)
        prefix = f"{intro}{history_snippet}, I found that {tools_plain}; regarding {prompt},"
        return re.sub(r"\s+", " ", prefix).strip()

    def generate_response(
        self,
        prompt: str,
        intent: str,
        confidence: float,
        conversation_history: Sequence[str] | None = None,
        tool_outputs: Sequence[object] | None = None,
    ) -> str:
        prefix = self._build_context_prefix(prompt, intent, tool_outputs, conversation_history)
        if self.gisd is None:
            return (
                f"Given intent {intent} at confidence {confidence:.2f}, {prefix} "
                "I will provide a concise response focused on actionable details and semantic clarity."
            )
        generated, _ = self.gisd.generate(prefix, intent=intent)
        if not generated.strip():
            return f"For {intent}, {prompt}, I recommend a pragmatic answer with explicit assumptions and next steps."
        return generated
