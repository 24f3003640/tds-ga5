import os
import json
import re
import asyncio
import httpx

def get_llm_credentials():
    api_key = os.environ.get("AIPIPE_TOKEN") or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LLM_API_KEY") or ""
    base_url = os.environ.get("LLM_BASE_URL")
    if not base_url:
        if os.environ.get("AIPIPE_TOKEN"):
            base_url = "https://aipipe.org/v1"
        elif os.environ.get("OPENROUTER_API_KEY"):
            base_url = "https://openrouter.ai/api/v1"
        else:
            base_url = "https://api.openai.com/v1"
    model = os.environ.get("LLM_MODEL") or os.environ.get("OPENROUTER_MODEL") or ("gpt-4o-mini" if os.environ.get("AIPIPE_TOKEN") or os.environ.get("OPENAI_API_KEY") else "nvidia/nemotron-3-ultra-550b-a55b:free")
    return api_key.strip(), base_url.rstrip("/"), model

def available() -> bool:
    api_key, _, _ = get_llm_credentials()
    return bool(api_key)

async def call_llm_json(prompt: str, timeout: float = 15.0) -> dict:
    """
    Calls configured LLM (AIPipe, OpenRouter, OpenAI) and parses JSON output.
    Returns parsed dict or list.
    """
    api_key, base_url, model = get_llm_credentials()
    if not api_key:
        print("⚠️ No LLM API key configured (AIPIPE_TOKEN/OPENAI_API_KEY/OPENROUTER_API_KEY)", flush=True)
        return {}

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 2048,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json=payload,
            )
        if resp.status_code >= 400:
            print(f"⚠️ LLM call failed with status {resp.status_code}: {resp.text[:200]}", flush=True)
            return {}
        
        data = resp.json()
        text = (data["choices"][0]["message"]["content"] or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text).strip()
        return json.loads(text)
    except Exception as e:
        print(f"⚠️ LLM call failed or timed out: {e}", flush=True)
        return {}
