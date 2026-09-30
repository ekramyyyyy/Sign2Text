"""Gloss tokens -> English + Arabic sentence using a FREE LLM.

Provider is chosen with env vars (see README):
  LLM_PROVIDER=gemini  (default)  needs GEMINI_API_KEY   [optional GEMINI_MODEL]
  LLM_PROVIDER=openai             any OpenAI-compatible endpoint, e.g. Qwen via DashScope / OpenRouter / local Ollama
                                  LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
"""
import json, os, re

SYSTEM = (
    "You convert recognized American Sign Language (ASL) glosses into natural sentences. "
    "Each numbered line is ONE sign with the recognizer's top candidates and confidences; pick the most plausible "
    "candidate per position using sentence context. ASL gloss order and grammar differ from English "
    "(e.g. 'YOU NAME WHAT' means 'What is your name?'). Add only the function words needed; do not invent content. "
    'Reply with ONLY JSON: {"glosses": [chosen glosses], "english": "...", "arabic": "..."}. '
    "The Arabic must be natural Modern Standard Arabic and must preserve the EXACT meaning of the English "
    "sentence you produced - translate meaning, not similar-sounding words. Before answering, double-check "
    "that each Arabic word's root matches the intended meaning and is not a different, similar-sounding root "
    "(e.g. Arabic مشى 'to walk' vs مشط 'to comb' are unrelated roots that must not be confused)."
)

def _prompt(tokens):
    lines = [f"{i}. " + " | ".join(f"{_clean(g)} ({c:.2f})" for g, c in cands) for i, cands in enumerate(tokens, 1)]
    return "Recognized signs, in order:\n" + "\n".join(lines)


def _clean(g):
    return g[:-1] if g.endswith("1") else g   # hide the ASL-LEX "1" suffix (e.g. bee1 -> bee)


def _parse(text):
    try:
        return json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    except Exception:
        return {"english": text.strip(), "arabic": ""}


def _gemini(prompt):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    r = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), contents=prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM, temperature=0.2,
                                           response_mime_type="application/json"))
    return r.text


def _groq(prompt):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")
    r = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=0.2,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
    return r.choices[0].message.content


def _openai_compat(prompt):
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("LLM_API_KEY", "ollama"),
                    base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"))
    r = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "qwen2.5:7b"), temperature=0.2,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
    return r.choices[0].message.content


_PROVIDERS = {"gemini": _gemini, "groq": _groq, "openai": _openai_compat}


def gloss_to_sentence(tokens):
    """tokens: list (one per sign) of [(gloss, confidence), ...] candidates, best first."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    try:
        text = _PROVIDERS.get(provider, _groq)(_prompt(tokens))
        out = _parse(text)
        out.setdefault("english", ""); out.setdefault("arabic", "")
        return out
    except Exception as e:        # offline fallback: just join the top-1 glosses
        return {"english": " ".join(_clean(c[0][0]) for c in tokens).lower(), "arabic": "", "error": str(e)}


if __name__ == "__main__":
    demo = [[("you", .9), ("your", .05)], [("name", .8), ("call", .1)], [("what", .7), ("who", .2)]]
    print(gloss_to_sentence(demo))