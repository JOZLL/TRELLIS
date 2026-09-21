# ------------------------------------------------------------
# generator.py
# Minimal TRELLIS wrapper required by the host platform.
# ------------------------------------------------------------
# The host will import this file, look for the class name given
# in manifest["generator_class"] (“Generator”), instantiate it,
# and call its .generate(prompt, **kwargs) method.
# ------------------------------------------------------------

import os
import sys
import pathlib
from typing import Any, Dict, List

# ------------------------------------------------------------------
# Make sure the TRELLIS package can be imported even if the host runs
# this file from a different working directory.
# ------------------------------------------------------------------
repo_root = pathlib.Path(__file__).parents[2]   # <repo‑root>/app_folder/..
sys.path.append(str(repo_root))

try:
    from trellis.models import ModelFactory
except Exception as exc:
    raise ImportError(
        "Could not import TRELLIS. Make sure the repository is "
        "installed (`pip install -r requirements.txt`)."
    ) from exc

# ------------------------------------------------------------------
# Load a model once (global singleton).  Default is the tiny “gpt2”
# which works on any machine.  Override with the env var
# TRELLIS_MODEL if you want a larger model.
# ------------------------------------------------------------------
_MODEL_NAME: str = os.getenv("TRELLIS_MODEL", "gpt2")
_factory = ModelFactory.from_pretrained(_MODEL_NAME)


# ------------------------------------------------------------------
# The class name must match the value in manifest["generator_class"]
# ------------------------------------------------------------------
class Generator:
    """
    Host‑side generator class.

    The host will do roughly:
        from generator import Generator
        gen = Generator()
        result = gen.generate(prompt="…", max_new_tokens=128, …)

    The method returns a JSON‑serialisable dict containing the generated
    text and a few meta‑fields.
    """

    def __init__(self) -> None:
        # Nothing to initialise – the model is already loaded globally.
        pass

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        **extra: Any,
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        prompt : str
            Text that seeds generation.
        max_new_tokens, temperature, top_k, top_p : generation knobs.
        **extra : Any
            Catch‑all for any additional arguments the host may pass.

        Returns
        -------
        dict
            {
                "generated_text": <text>,
                "model": <model‑name>,
                "prompt": <prompt>,
                "settings": {max_new_tokens, temperature, top_k, top_p}
            }
        """
        # Guard against a non‑string prompt (some hosts use `input` instead)
        if not isinstance(prompt, str):
            prompt = str(extra.get("input", ""))

        # TRELLIS generate returns a list of strings; we ask for one.
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
# Optional CLI for local testing (does not affect the host)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(
        description="Local test for the TRELLIS Generator wrapper"
    )
    parser.add_argument("prompt", help="Prompt text")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)

    args = parser.parse_args()
    gen = Generator()
    result = gen.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
