"""
gemini_router.py — Resilient Gemini Model Manager with Automatic Fallback

FIXES:
  - Flat sequential scan prevents circular fallback to already-failed models
  - Non-retriable errors (bad API key, invalid request) propagate immediately
  - on_fallback callback safely wrapped in try/except
  - Removed stale `_is_retriable` false negatives for "not found" / 404

Model Priority (strongest → lightest):
  1. gemini-2.5-flash       ⭐⭐⭐⭐⭐
  2. gemini-2.0-flash       ⭐⭐⭐⭐
  3. gemini-1.5-flash       ⭐⭐⭐
  4. gemini-2.0-flash-lite  ⭐⭐
"""

import os
import time
import logging
from typing import Optional, List, Callable, Any
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

MODELS_PRIORITY: List[str] = [
    "gemini-2.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-2.0-flash-lite",
]

_RETRIABLE_KEYWORDS = [
    "429", "500", "503", "404", "not_found", "not found",
    "resource_exhausted", "service_unavailable",
    "internal", "quota", "overloaded",
    "deadline_exceeded", "too many",
]

_NON_RETRIABLE_KEYWORDS = [
    "api_key", "invalid", "permission", "403",
    "unauthorized",
]


def _is_retriable(e: Exception) -> bool:
    err_str = str(e).lower()
    # Non-retriable takes priority
    if any(kw in err_str for kw in _NON_RETRIABLE_KEYWORDS):
        return False
    if any(kw in err_str for kw in _RETRIABLE_KEYWORDS):
        return True
    type_name = type(e).__name__
    return type_name in (
        "ResourceExhausted", "ServiceUnavailable",
        "InternalServerError", "DeadlineExceeded", "Aborted",
    )


class GeminiRouter:
    """
    Wraps ChatGoogleGenerativeAI with automatic model fallback.

    Usage:
        router = GeminiRouter()
        response = router.invoke("Hello!")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        models: Optional[List[str]] = None,
        on_fallback: Optional[Callable[[str, str, str], None]] = None,
    ):
        self._api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self._temperature = temperature
        self._models = models or list(MODELS_PRIORITY)
        self._on_fallback = on_fallback
        self._current_idx = 0

        if not self._api_key:
            raise ValueError(
                "GOOGLE_API_KEY is required — set it in .env or pass api_key="
            )

        self._llms = {
            name: ChatGoogleGenerativeAI(
                api_key=self._api_key,
                model=name,
                temperature=self._temperature,
                max_retries=0,
            )
            for name in self._models
        }

    @property
    def current_model(self) -> str:
        return self._models[self._current_idx]

    @property
    def llm(self) -> ChatGoogleGenerativeAI:
        return self._llms[self.current_model]

    def invoke(self, prompt, **kwargs) -> Any:
        """
        Invoke the LLM with automatic fallback.
        Tries each model sequentially — never loops back to a failed model.
        """
        errors: List[str] = []
        n = len(self._models)

        for attempt in range(n):
            idx = (self._current_idx + attempt) % n
            model_name = self._models[idx]
            backoff = min(1.0 * (2 ** attempt), 16.0)

            try:
                response = self._llms[model_name].invoke(prompt, **kwargs)
                self._current_idx = idx
                return response

            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                errors.append(f"[{model_name}] {error_msg}")

                if not _is_retriable(e):
                    raise

                logger.warning(
                    f"[GeminiRouter] {model_name} failed ({error_msg}). "
                    f"Backing off {backoff:.1f}s then trying next model..."
                )

                # Notify caller before switching
                if self._on_fallback and attempt < n - 1:
                    next_idx = (self._current_idx + attempt + 1) % n
                    new_model = self._models[next_idx]
                    try:
                        self._on_fallback(model_name, new_model, error_msg)
                    except Exception:
                        pass

                if attempt < n - 1:
                    time.sleep(backoff)

        raise RuntimeError(
            f"All {n} Gemini models exhausted.\n" + "\n".join(errors)
        )

    def reset(self):
        """Reset to the primary (strongest) model."""
        self._current_idx = 0

    def __repr__(self) -> str:
        return (
            f"GeminiRouter(active={self.current_model!r}, "
            f"models={self._models})"
        )