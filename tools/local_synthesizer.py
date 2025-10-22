from __future__ import annotations
from typing import List, Dict
import os
from loguru import logger



def synthesize_from_provenance(bullets: List[str], max_tokens: int = 256) -> str:
    """Synthesize a short, cited answer from extractive provenance bullets.

    bullets: list of strings already containing citation markers like [1] or (source: ...)
    Returns a short paragraph and preserves citation markers.
    """
    if not bullets:
        return "No provenance bullets to synthesize from."

    prompt = (
        "You are a helpful summarization assistant. Given the following numbered provenance bullets, "
        "write a concise (2-4 sentence) answer to the user's question, and keep citation markers like '[1]'.\n\n"
        "Bullets:\n"
    )

    for b in bullets:
        prompt += f"- {b}\n"

    prompt += "\nAnswer:"

    try:
        # import here to avoid hard dependency at module import-time if transformers isn't installed
        from llm_engineering.model.inference.local_inference import LocalModelInference

        llm = LocalModelInference()
        llm.set_payload(inputs=prompt, parameters={"max_new_tokens": max_tokens, "temperature": 0.0})
        out = llm.inference()
        return out[0]["generated_text"]
    except Exception as e:
        logger.warning(f"Local synthesis failed: {e}")
        # fallback: join the bullets
        return "\n".join(bullets[:5])
