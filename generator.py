# trellis_fork/generator.py
import os
import sys
import pathlib
from typing import Any, Dict, List

# --------------------------------------------------------------
# Make the repository root the *first* entry on sys.path.
# This ensures that a local stub of Kaolin (if you keep one) is found
# before the placeholder wheel in site‑packages.
# --------------------------------------------------------------
repo_root = pathlib.Path(__file__).parents[2].resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))   # <‑‑ PREPEND, not append

# -----------------------------------------------------------------
# The heavy TRELLIS import is now done lazily inside generate().
# If the import fails we fall back to a very simple echo implementation.
# -----------------------------------------------------------------
_MODEL_NAME: str = os.getenv("TRELLIS_MODEL", "gpt2")
_factory = None               # will be set lazily
_TRELLIS_AVAILABLE = False   # flag set after a successful import
_IMPORT_ERROR = None          # keep the original exception for debugging


class Generator:
    """Modly‑compatible generator.

    If the full TRELLIS stack can be imported, we delegate to its
    ``ModelFactory``.  Otherwise we return a `[fallback]` string so that the
    extension loads without crashing.
    """

    def __init__(self) -> None:
        pass

    def _ensure_trellis(self) -> None:
        """Import TRELLIS on first use.  Sets globals _factory and _TRELLIS_AVAILABLE."""
        global _factory, _TRELLIS_AVAILABLE, _IMPORT_ERROR
        if _TRELLIS_AVAILABLE:
            return
        try:
            # Import *after* the repo root has been prepended.
            from trellis.models import ModelFactory  # type: ignore
            _factory = ModelFactory.from_pretrained(_MODEL_NAME)  # may raise
            _TRELLIS_AVAILABLE = True
        except Exception as exc:  # pragma: no cover
            _IMPORT_ERROR = exc
            _TRELLIS_AVAILABLE = False
            _factory = None

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Generate text (or fallback)."""
        if not isinstance(prompt, str):
            prompt = str(extra.get("input", ""))

        # Try the real TRELLIS implementation first.
        self._ensure_trellis()
        if _TRELLIS_AVAILABLE and _factory is not None:
            outputs: List[str] = _factory.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=1,
            )
            generated = outputs[0] if outputs else ""
            model_name = _MODEL_NAME
        else:
            # -----------------------------------------------------------------
            # Fallback path – works on any machine, no native dependencies needed.
            # -----------------------------------------------------------------
            generated = f"[fallback] {prompt}"
            model_name = "fallback"

        result: Dict[str, Any] = {
            "generated_text": generated,
            "model": model_name,
            "prompt": prompt,
            "settings": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            },
        }

        # Attach the original traceback only when we are in fallback mode,
        # so you can inspect why the real stack failed (useful for debugging).
        if not _TRELLIS_AVAILABLE and _IMPORT_ERROR is not None:
            result["import_error"] = str(_IMPORT_ERROR)

        return result


# -----------------------------------------------------------------
# Small CLI – handy for manual testing.
# -----------------------------------------------------------------
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(
        description="Test the TRELLIS‑wrapper (fallback works out‑of‑the‑box)."
    )
    parser.add_argument("prompt", help="Prompt text")
    args = parser.parse_args()
    gen = Generator()
    out = gen.generate(args.prompt)
    print(json.dumps(out, ensure_ascii=False, indent=2))
