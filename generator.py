import os
import sys
import pathlib
from typing import Any, Dict, List

# Ensure the TRELLIS package can be imported even if run from a different cwd
repo_root = pathlib.Path(__file__).parents[2]
sys.path.append(str(repo_root))

try:
    from trellis.models import ModelFactory
except Exception as exc:
    raise ImportError(
        "Could not import TRELLIS. Make sure the repository is installed "
        "(pip install -r requirements.txt)."
    ) from exc

# Load a default model once (global singleton). Env var TRELLIS_MODEL can override.
_MODEL_NAME: str = os.getenv("TRELLIS_MODEL", "gpt2")
_factory = ModelFactory.from_pretrained(_MODEL_NAME)


class Generator:
    """Host‑side generator class expected by the platform.

    The platform will instantiate this class and call ``generate``.
    ``generate`` returns a JSON‑serialisable dict.
    """

    def __init__(self) -> None:
        # No per‑instance state needed – the model is loaded globally.
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
        """Generate text from a prompt."""
        if not isinstance(prompt, str):
            prompt = str(extra.get("input", ""))

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


# Optional CLI for quick local testing
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
