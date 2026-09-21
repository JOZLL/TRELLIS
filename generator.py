# ------------------------------------------------------------
# generator.py
# Minimal TRELLIS‑based generator required by the host platform.
# ------------------------------------------------------------
#   • Imports the TRELLIS library (already part of the cloned repo)
#   • Loads a model once at import time (cached for subsequent calls)
#   • Exposes a `generate(...)` function the host will invoke
#   • Optional CLI for quick local testing
# ------------------------------------------------------------

import os
from typing import Any, Dict, List

# ------------------------------------------------------------------
# 1️⃣  Import TRELLIS
# ------------------------------------------------------------------
# The `trellis` package lives next to this file because we are inside the
# cloned TRELLIS repository, so Python can resolve it automatically.
try:
    from trellis.models import ModelFactory
except ImportError as exc:
    raise ImportError(
        "TRELLIS could not be imported. Make sure the repository is "
        "installed (`pip install -r requirements.txt`) and that this "
        "file lives inside the TRELLIS clone."
    ) from exc


# ------------------------------------------------------------------
# 2️⃣  Load the model (once) ---------------------------------------
# ------------------------------------------------------------------
# The model name can be overridden with the environment variable
# `TRELLIS_MODEL`.  If not set we fall back to the very small `gpt2`
# checkpoint – fast to download and works on any machine.
_MODEL_NAME: str = os.getenv("TRELLIS_MODEL", "gpt2")

# `ModelFactory.from_pretrained` returns a callable that can be used
# repeatedly to generate text.  The first call will download/cache the
# model; subsequent calls are instant.
_factory = ModelFactory.from_pretrained(_MODEL_NAME)


# ------------------------------------------------------------------
# 3️⃣  Public API – the function the host will call ----------------
# ------------------------------------------------------------------
def generate(
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    **_: Any,
) -> Dict[str, Any]:
    """
    Generate text using TRELLIS.

    Parameters
    ----------
    prompt : str
        Seed text for the model.
    max_new_tokens : int, default 128
        Number of new tokens to generate.
    temperature : float, default 0.8
        Sampling temperature – higher = more random.
    top_k : int, default 50
        Keep only the top‑k tokens at each step.
    top_p : float, default 0.95
        Nucleus sampling – keep the smallest set of tokens with
        cumulative probability >= top_p.
    **_ : Any
        Catch‑all for any extra arguments the host may send;
        they are ignored but kept for a flexible signature.

    Returns
    -------
    dict
        A JSON‑serialisable dictionary with:
        * ``generated_text`` – the model’s continuation.
        * ``model`` – which model was used.
        * ``prompt`` – the input prompt (echoed back).
        * ``settings`` – the generation hyper‑parameters.
    """
    # TRELLIS `generate` returns a list of strings (one per requested
    # sample).  We ask for a single sample (`num_return_sequences=1`).
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
# 4️⃣  Optional command‑line interface for local testing ----------
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Simple CLI wrapper around the `generate` function."
    )
    parser.add_argument("prompt", help="Prompt text to seed generation")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate (default: 128)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top‑k sampling (default: 50)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top‑p (nucleus) sampling (default: 0.95)",
    )
    args = parser.parse_args()

    result = generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    # Pretty‑print the JSON result for quick inspection
    print(json.dumps(result, ensure_ascii=False, indent=2))
