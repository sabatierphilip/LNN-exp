"""Guided Iterative Semantic Diffusion (GISD) text generation for BERT masked LMs.

This module implements a non-autoregressive generation algorithm that starts from a
fully masked suffix and iteratively reveals/revises tokens in parallel.
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
    steering_alpha_start: float = 0.80  # semantic guidance strength at step 0
    steering_alpha_end: float = 0.15  # semantic guidance strength at final step
    top_k_vocab: int = 512  # restrict steering to top-k vocab tokens before top-p
    top_p: float = 0.90  # nucleus mass retained for sampling
    repetition_penalty: float = 1.25  # down-weight already chosen tokens
    min_tokens: int = 18  # global lower bound for generated suffix length
    max_tokens: int = 80  # global upper bound for generated suffix length
    revision_conf_threshold: float = 0.25  # low-confidence revealed tokens are remasked
    context_max_tokens: int = 64  # prompt prefix budget
    intent_length_hints: Dict[str, float] = field(
        default_factory=lambda: {
            "search": 1.0,
            "summarize": 0.72,
            "recall": 0.85,
            "generate": 1.2,
            "plan": 1.3,
        }
    )


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

    def _intent_budget(self, intent: str) -> int:
        multiplier = self.config.intent_length_hints.get(intent, 1.0)
        base = int(round((self.config.min_tokens + self.config.max_tokens) / 2))
        n = int(round(base * multiplier))
        return max(self.config.min_tokens, min(self.config.max_tokens, n))

    def _prefix_ids(self, prompt: str) -> List[int]:
        ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        return list(ids[: self.config.context_max_tokens])

    def _semantic_target_vec(self, prompt: str):
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.config.context_max_tokens)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with self.torch.no_grad():
            cls = self.model(**encoded).last_hidden_state[:, 0, :]
        return self.F.normalize(cls.squeeze(0), p=2, dim=-1)

    def _apply_top_p(self, logits):
        sorted_logits, sorted_idx = self.torch.sort(logits, descending=True)
        probs = self.torch.softmax(sorted_logits, dim=-1)
        csum = self.torch.cumsum(probs, dim=-1)
        keep = csum <= self.config.top_p
        if keep.numel() and not bool(keep[0]):
            keep[0] = True
        filtered = self.torch.full_like(logits, float("-inf"))
        kept_idx = sorted_idx[keep]
        filtered[kept_idx] = logits[kept_idx]
        return filtered

    def _sample_position(self, raw_logits, token_counts: Dict[int, int], steering_scores, alpha: float, temp: float):
        logits = raw_logits / max(1e-6, temp)

        if token_counts:
            for tok, freq in token_counts.items():
                if tok >= 0 and tok < logits.shape[0]:
                    logits[tok] = logits[tok] - math.log(self.config.repetition_penalty) * float(freq)

        if self.config.top_k_vocab > 0:
            top_vals, top_idx = self.torch.topk(logits, k=min(self.config.top_k_vocab, logits.shape[-1]))
            steered = top_vals + alpha * steering_scores[top_idx]
            logits2 = self.torch.full_like(logits, float("-inf"))
            logits2[top_idx] = steered
            logits = logits2
        else:
            logits = logits + alpha * steering_scores

        filtered = self._apply_top_p(logits)
        probs = self.torch.softmax(filtered, dim=-1)
        chosen = int(self.torch.multinomial(probs, 1).item())
        conf = float(probs[chosen].item())
        return chosen, conf

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
        alpha = self._anneal(self.config.steering_alpha_start, self.config.steering_alpha_end, frac, 1.0)

        mask_id = int(self.tokenizer.mask_token_id)
        unit_E = self._ensure_unit_embeddings()
        steering_scores = unit_E @ target_vec

        with self.torch.no_grad():
            logits = self.masked_lm(input_ids=seq.unsqueeze(0)).logits[0]

        candidates: List[Tuple[int, int, float]] = []
        sampled_conf: Dict[int, float] = {}

        for local_idx, pos in enumerate(output_positions):
            if not bool(revealed[local_idx]):
                tok, conf = self._sample_position(logits[pos], token_counts, steering_scores, alpha, temp)
                seq[pos] = tok
                candidates.append((local_idx, tok, conf))
                sampled_conf[local_idx] = conf

        masked_left = int((~revealed).sum().item())
        ramp = 0.5 + 0.5 * frac
        reveal_count = max(1, int(math.ceil(masked_left * self.config.reveal_fraction * ramp))) if masked_left > 0 else 0
        candidates.sort(key=lambda x: x[2], reverse=True)
        chosen = candidates[:reveal_count]

        for local_idx, tok, _ in chosen:
            revealed[local_idx] = True
            token_counts[tok] = token_counts.get(tok, 0) + 1

        conf_thr = self.config.revision_conf_threshold if revision_conf_threshold is None else revision_conf_threshold
        if step_index >= 1:
            open_revealed = [i for i in range(len(output_positions)) if bool(revealed[i])]
            low_conf = sorted(
                [(i, sampled_conf.get(i, 1.0)) for i in open_revealed], key=lambda x: x[1]
            )
            if low_conf:
                max_by_fraction = max(1, int(math.ceil(len(open_revealed) * self.config.remask_fraction)))
                remaskable = [x for x in low_conf if x[1] < conf_thr][:max_by_fraction]
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
            fallback = f"I can help with {intent}: {prompt}. I will provide a concise, practical response grounded in the request context."
            words = fallback.split()
            return " ".join(words[: max(5, min(len(words), self.config.min_tokens))]), []

        with self.torch.no_grad():
            prefix = self._prefix_ids(prompt)
            n_output = self._intent_budget(intent)
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
            cleaned = self._clean_text(text)
            return cleaned, token_ids


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
            return "no external tool findings were provided"
        parts: List[str] = []
        for item in tool_outputs:
            if isinstance(item, dict):
                tool = str(item.get("tool", "tool"))
                detail_chunks = []
                for k, v in item.items():
                    if k == "tool":
                        continue
                    if isinstance(v, (list, tuple)):
                        vtxt = ", ".join(str(x) for x in v[:3])
                    else:
                        vtxt = str(v)
                    detail_chunks.append(f"{k} {vtxt}")
                detail = "; ".join(detail_chunks) if detail_chunks else "with usable intermediate findings"
                parts.append(f"{tool} reported {detail}")
            else:
                parts.append(str(item))
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
            history_snippet = f" while keeping continuity with earlier notes about {history[-1][:80]}"

        tools_plain = self._tool_outputs_to_english(tool_outputs)
        prefix = f"{intro}{history_snippet}, I found that {tools_plain}; regarding '{prompt}',"
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
                f"I will provide a concise response focused on actionable details and semantic clarity."
            )
        text, _ = self.gisd.generate(prefix, intent=intent)
        if not text.strip():
            return f"For {intent}, {prompt}, I recommend a pragmatic answer with explicit assumptions and next steps."
        return text
