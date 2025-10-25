
from zenml import step
from zenml.client import Client
import openai
from openai import OpenAIError
import os

@step(enable_cache=True)
def summarize_documents(documents: list) -> list:
    # Try to load OpenAI credentials from ZenML secret store, fall back to env vars
    try:
        secret = Client().get_secret("openai_secret")
        openai.api_key = secret.secret_values.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        model_id = secret.secret_values.get("OPENAI_MODEL_ID", os.getenv("OPENAI_MODEL_ID", "gpt-4o-mini"))
    except Exception:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        model_id = os.getenv("OPENAI_MODEL_ID", "gpt-4o-mini")
    if not openai.api_key:
        raise RuntimeError("OpenAI API key not found in ZenML secret store or in environment variable OPENAI_API_KEY")
    fallback_model = "gpt-3.5-turbo"
    summaries = []
    for doc in documents:
        prompt = f"Summarize the following document:\n\n{doc[:4000]}"
        try:
            response = openai.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.5,
            )
        except OpenAIError as e:
            if "insufficient_quota" in str(e):
                response = openai.chat.completions.create(
                    model=fallback_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=256,
                    temperature=0.5,
                )
            else:
                raise
        summary = response.choices[0].message.content.strip()
        summaries.append(summary)
    return summaries