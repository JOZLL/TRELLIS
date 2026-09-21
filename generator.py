# generator.py – defensive edition
import os, sys, pathlib
from typing import Any, Dict, List

# ------------------------------------------------------------------
# Make sure we can import the TRELLIS package even if the host runs us
# from a sub‑directory.  This walks two levels up (repo root) and
# adds it to sys.path.
# ------------------------------------------------------------------
repo_root = pathlib.Path(__file__).parents[2]          # …/app_my_trellis/..
sys.path.append(str(repo_root))

try:
    from trellis.models import ModelFactory
except Exception as exc:
    raise ImportError(
        "TRELLIS could not be imported. Did you run `pip install -e .` "
        "from the repo root? Original error: " + str(exc)
    ) from exc

# ------------------------------------------------------------------
# Model loading (once)
# ------------------------------------------------------------------
_MODEL_NAME: str = os.getenv("TRELLIS_MODEL", "gpt2")
_factory = ModelFactory.from_pretrained(_MODEL_NAME)

# ------------------------------------------------------------------
# Public generate() – robust against missing/incorrect prompt key
# ------------------------------------------------------------------
def generate(
    prompt: str = "",
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    **extra: Any,
) -> Dict[str, Any]:
    # If the caller gave us a non‑string or used a different key, coerce it.
    if not isinstance(prompt, str):
        # Look for common alternatives
        prompt = (
            extra.get("input")
            or extra.get("text")
            or ""
        )
        if not isinstance(prompt, str):
            prompt = str(prompt)

    outputs: List[str] = _factory.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=1,
    )
    generated = outputs[0] if outputs else ""

    return {
        "generated_text": generated,
        "model": _MODEL_NAME,
        "prompt": prompt,
        "settings": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        },
    }

# ------------------------------------------------------------------
# Optional CLI (unchanged)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(
        description="Simple CLI wrapper around the `generate` function."
    )
    parser.add_argument("prompt", help="Prompt text to seed generation")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()

    result = generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
