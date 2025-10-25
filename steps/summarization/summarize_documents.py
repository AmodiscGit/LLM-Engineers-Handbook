
from zenml import step
from zenml.client import Client
import openai
from openai import OpenAIError
import os
import math


def _chunk_text(text: str, max_chars: int = 4000, overlap: int = 200) -> list:
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    parts = []
    start = 0
    L = len(text)
    while start < L:
        end = min(L, start + max_chars)
        parts.append(text[start:end])
        if end == L:
            break
        start = max(0, end - overlap)
    return parts


@step(enable_cache=True)
def summarize_documents(documents: list) -> list:
    """Summarize a list of documents (abstractive). Improvements:
    - Uses a clear system + user prompt with an expected output format
    - Chunks long documents and combines chunk summaries
    - Exposes simple env-config for model, temperature and max tokens
    """
    # Try to load OpenAI credentials from ZenML secret store, fall back to env vars
    try:
        secret = Client().get_secret("openai_secret")
        api_key = secret.secret_values.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        model_id = secret.secret_values.get("OPENAI_MODEL_ID", os.getenv("OPENAI_MODEL_ID", "gpt-4o-mini"))
    except Exception:
        api_key = os.getenv("OPENAI_API_KEY")
        model_id = os.getenv("OPENAI_MODEL_ID", "gpt-4o-mini")

    if not api_key:
        raise RuntimeError("OpenAI API key not found in ZenML secret store or in environment variable OPENAI_API_KEY")

    openai.api_key = api_key
    fallback_model = os.getenv("SUMMARIZE_FALLBACK_MODEL", "gpt-3.5-turbo")
    max_tokens = int(os.getenv("SUMMARIZE_MAX_TOKENS", "400"))
    temperature = float(os.getenv("SUMMARIZE_TEMPERATURE", "0.2"))
    bullets = int(os.getenv("SUMMARIZE_BULLETS", "3"))

    # Prepare system + user prompt template
    system_msg = (
        "You are a concise, factual summarization assistant.\n"
        "When asked to summarize, produce exactly {bullets} short bullet points that capture the main ideas, followed by a single provenance line in the format: Provenance: <SOURCE>.\n"
        "Be precise, avoid hallucination, and prefer to say 'unknown' when provenance is not available."
    ).format(bullets=bullets)

    # few-shot examples to encourage format
    example_user = (
        "Text: In 2020 the project released version 1.2 which fixed security bugs and improved performance."
        "\nSummarize:"
    )
    example_assistant = "- Project released v1.2 with security fixes.\n- Performance improvements in 2020.\n- No breaking API changes.\nProvenance: release_notes.txt"

    summaries = []

    for doc in documents:
        if not doc:
            summaries.append("")
            continue

        # chunk long docs, summarize each chunk, then combine
        chunks = _chunk_text(doc, max_chars=4000, overlap=200)
        chunk_summaries = []
        for c in chunks:
            prompt = (
                f"Summarize the following text into {bullets} concise bullets and one provenance line.\n\nText:\n{c}"
            )
            try:
                response = openai.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": example_user},
                        {"role": "assistant", "content": example_assistant},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except OpenAIError as e:
                # try fallback model once
                if fallback_model:
                    response = openai.chat.completions.create(
                        model=fallback_model,
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                else:
                    raise

            text_out = ""
            try:
                text_out = response.choices[0].message.content.strip()
            except Exception:
                text_out = ""

            chunk_summaries.append(text_out)

        # If multiple chunk summaries, combine them into final bullets
        if len(chunk_summaries) == 1:
            final = chunk_summaries[0]
        else:
            combine_prompt = (
                f"Combine the following {len(chunk_summaries)} chunk summaries into exactly {bullets} concise bullets and one provenance line.\n\n" +
                "\n\n".join(chunk_summaries)
            )
            try:
                resp2 = openai.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": combine_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                final = resp2.choices[0].message.content.strip()
            except Exception:
                # Fall back to concatenation of chunk summaries
                final = "\n\n".join(chunk_summaries)

        summaries.append(final)

    return summaries