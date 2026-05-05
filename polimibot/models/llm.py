"""LLM wrapper. The only file that imports `transformers`."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch


@dataclass(frozen=True)
class LLMSpec:
    """Everything needed to load a model. Pass to LLM.load()."""
    model_id: str                          # HF repo, e.g. "Qwen/Qwen2.5-7B-Instruct"
    load_in_4bit: bool = True              # NF4 via bitsandbytes; set False on CPU
    torch_dtype: str = "bfloat16"         # ignored when load_in_4bit=True
    device_map: str = "auto"              # "auto" lets accelerate place layers
    max_new_tokens: int = 16              # generation cap; overridable per-call
    trust_remote_code: bool = False


@dataclass
class AnswerProbabilities:
    """Result of score_options: softmax'd probs over A/B/C/D."""
    probs: Dict[str, float]               # {"A": 0.82, "B": 0.07, ...}
    top_letter: str
    top_prob: float
    margin: float                         # top_prob - second_prob; confidence proxy
    elapsed_seconds: float

    @classmethod
    def from_probs(cls, probs: Dict[str, float], elapsed: float) -> "AnswerProbabilities":
        sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_letter, top_prob = sorted_items[0]
        margin = top_prob - sorted_items[1][1] if len(sorted_items) > 1 else top_prob
        return cls(probs=probs, top_letter=top_letter,
                   top_prob=top_prob, margin=margin, elapsed_seconds=elapsed)


@dataclass
class LLMResponse:
    text: str
    elapsed_seconds: float
    input_tokens: int = 0
    output_tokens: int = 0


class LLM:
    """Thin wrapper over a HuggingFace causal LM.

    Usage:
        spec = LLMSpec(model_id="Qwen/Qwen2.5-7B-Instruct")
        llm = LLM.load(spec)
        probs = llm.score_options(messages)   # preferred for MCQ
        resp  = llm.generate(messages)        # for CoT / free-text
    """

    def __init__(self, model, tokenizer, spec: LLMSpec) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self.spec = spec

    @property
    def name(self) -> str:
        # "Qwen/Qwen2.5-7B-Instruct" → "qwen2.5-7b-instruct"
        return self.spec.model_id.split("/")[-1].lower()

    @classmethod
    def load(cls, spec: LLMSpec) -> "LLM":
        """Load model + tokenizer. Slow (1–3 min on Colab); call once."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"Loading {spec.model_id}  (4bit={spec.load_in_4bit})…")
        t0 = time.monotonic()

        tokenizer = AutoTokenizer.from_pretrained(
            spec.model_id, trust_remote_code=spec.trust_remote_code
        )
        # Ensure padding token exists (some models omit it)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        bnb_cfg = None
        if spec.load_in_4bit:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,   # extra 0.4 bits/weight saving
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        dtype = getattr(torch, spec.torch_dtype) if not spec.load_in_4bit else None
        model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            quantization_config=bnb_cfg,
            torch_dtype=dtype,
            device_map=spec.device_map,
            trust_remote_code=spec.trust_remote_code,
        )
        model.eval()
        print(f"Loaded in {time.monotonic()-t0:.1f}s")
        return cls(model, tokenizer, spec)

    # ── public API ──────────────────────────────────────────────────────────

    def generate(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Generate free text from a chat-formatted message list."""
        prompt = self._apply_template(messages, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        n_in = inputs["input_ids"].shape[-1]

        t0 = time.monotonic()
        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.spec.max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        elapsed = time.monotonic() - t0

        new_tokens = out[0][n_in:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return LLMResponse(text=text, elapsed_seconds=elapsed,
                           input_tokens=n_in, output_tokens=len(new_tokens))

    def score_options(
        self,
        messages: Sequence[Dict[str, str]],
        letters: Sequence[str] = ("A", "B", "C", "D"),
    ) -> AnswerProbabilities:
        """One forward pass → probabilities over answer letters.

        Appends "Answer:" to the prompt so the model's next token
        is the answer letter. Reads logits[letters] directly — no generation.
        """
        # Append the answer-elicitation suffix BEFORE tokenizing
        suffix_messages = list(messages) + [
            {"role": "assistant", "content": "Answer:"}
        ]
        prompt = self._apply_template(suffix_messages, add_generation_prompt=False)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        t0 = time.monotonic()
        with torch.inference_mode():
            logits = self._model(**inputs).logits  # (1, seq_len, vocab_size)
        elapsed = time.monotonic() - t0

        # Last token position = where the next token would be predicted
        last_logits = logits[0, -1, :]  # (vocab_size,)

        # Resolve token IDs for each letter (e.g. "A" → token 32)
        letter_ids = {
            l: self._tokenizer.encode(l, add_special_tokens=False)[0]
            for l in letters
        }
        raw = torch.tensor([last_logits[tid] for tid in letter_ids.values()])
        probs_tensor = torch.softmax(raw, dim=0)
        probs = {l: probs_tensor[i].item() for i, l in enumerate(letters)}

        return AnswerProbabilities.from_probs(probs, elapsed)

    # ── private ─────────────────────────────────────────────────────────────

    def _apply_template(
        self,
        messages: Sequence[Dict[str, str]],
        add_generation_prompt: bool,
    ) -> str:
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )