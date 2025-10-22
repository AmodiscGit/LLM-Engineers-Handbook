from __future__ import annotations

from typing import Any, Dict
import json
from loguru import logger

try:
    from transformers import (
        AutoTokenizer,
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
    )
    import torch
except Exception:
    logger.warning("Transformers or torch not available; local inference won't work.")

from llm_engineering.domain.inference import Inference
from llm_engineering.settings import settings


class LocalModelInference(Inference):
    """Local model inference wrapper that auto-detects causal vs seq2seq models.

    It looks for a model in `output/trained_model_long` then `output/trained_model`.
    If the model config has `is_encoder_decoder=True` we load a Seq2SeqLM, otherwise
    we load a CausalLM. The interface exposes `set_payload(inputs, parameters)` and
    `inference()` which returns a list with a dict {'generated_text': ...} to match
    the existing SageMaker wrapper contract.
    """

    def __init__(self, model_dir: str | None = None):
        super().__init__()
        self.model_dir = model_dir or "output/trained_model_long"
        # fallback
        self._try_dirs = [self.model_dir, "output/trained_model"]

        self._load_model()

    def _load_model(self):
        for d in self._try_dirs:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(d)
                config = AutoConfig.from_pretrained(d)

                # choose model class based on config
                if getattr(config, "is_encoder_decoder", False):
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(d)
                    self.model_type = "seq2seq"
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(d)
                    self.model_type = "causal"

                # move to device
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device)

                # ensure pad token for tokenizer
                if self.tokenizer.pad_token is None:
                    self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

                logger.info(f"Loaded local model ({self.model_type}) from {d}")
                return
            except Exception:
                continue

        raise RuntimeError("No local finetuned model found in output/trained_model_long or output/trained_model")

    def set_payload(self, inputs: str, parameters: Dict[str, Any] | None = None):
        self.inputs = inputs
        self.parameters = parameters or {}

    def inference(self):
        """Generate text. For seq2seq models we use the same generate API but keep
        treatment of inputs appropriate for encoder-decoder models.
        """
        max_new_tokens = int(self.parameters.get("max_new_tokens", settings.MAX_NEW_TOKENS_INFERENCE))
        temperature = float(self.parameters.get("temperature", settings.TEMPERATURE_INFERENCE))

        # tokenize inputs
        inputs = self.tokenizer(self.inputs, return_tensors="pt", truncation=True).to(self.device)

        with torch.no_grad():
            # modern transformers support `max_new_tokens` for both model types
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=False,
            )

        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return [{"generated_text": text}]
