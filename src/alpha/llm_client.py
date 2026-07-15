"""
OpenAI-compatible LLM API wrapper with SHA256-based file cache.

Supports two separate providers for narrative vs alpha generation.
Cache stores each unique (model, messages, kwargs) combination as a
JSON file under cache_dir/<sha256>.json.
"""
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


def _make_client(api_key_env: str, api_base: str) -> Optional[object]:
    if OpenAI is None:
        return None
    key = os.environ.get(api_key_env, '')
    return OpenAI(api_key=key, base_url=api_base, timeout=60.0) if key else None


class LLMClient:
    def __init__(self, config: dict):
        """
        config keys (all optional, with sensible defaults):

        Narrative model (fast, cheap - e.g. Gemini Flash):
            narrative_model       : model name
            narrative_api_base    : provider base URL
            narrative_api_key_env : env var holding the API key

        Alpha model (strong - e.g. DeepSeek Chat):
            alpha_model           : model name
            alpha_api_base        : provider base URL
            alpha_api_key_env     : env var holding the API key

        Shared:
            cache_dir : directory for cached responses
        """
        cache_dir = config.get('cache_dir', 'data/llm_cache')
        os.makedirs(cache_dir, exist_ok=True)
        self._cache_dir = cache_dir

        self._narrative_model = config.get('narrative_model', 'gemini-2.0-flash-lite')
        narrative_key_env = config.get('narrative_api_key_env', 'GEMINI_API_KEY')
        narrative_base = config.get(
            'narrative_api_base',
            'https://generativelanguage.googleapis.com/v1beta/openai/',
        )
        self._narrative_client = _make_client(narrative_key_env, narrative_base)
        self._narrative_key_env = narrative_key_env

        self._alpha_model = config.get('alpha_model', 'deepseek-chat')
        alpha_key_env = config.get('alpha_api_key_env', 'DEEPSEEK_API_KEY')
        alpha_base = config.get('alpha_api_base', 'https://api.deepseek.com')
        self._alpha_client = _make_client(alpha_key_env, alpha_base)
        self._alpha_key_env = alpha_key_env

        # Cumulative token counters {model: {'in': int, 'out': int}}
        self.usage: Dict[str, Dict[str, int]] = {}

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, messages: List[dict], model: str, **kwargs) -> str:
        payload = json.dumps(
            {'model': model, 'messages': messages, **kwargs},
            sort_keys=True, ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_path(self, key: str) -> str:
        return os.path.join(self._cache_dir, f"{key}.json")

    def _load_from_cache(self, key: str) -> Optional[str]:
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('response')
        except (json.JSONDecodeError, KeyError):
            os.remove(path)  # remove corrupted cache entry
            return None

    def _save_to_cache(self, key: str, model: str, response: str) -> None:
        path = self._cache_path(key)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'prompt_hash': key,
                'model': model,
                'response': response,
                'timestamp': datetime.now(timezone.utc).isoformat(),
            }, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Core call
    # ------------------------------------------------------------------

    def _call(
        self,
        client: object,
        key_env: str,
        messages: List[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        use_cache: bool,
        json_mode: bool,
    ) -> str:
        kwargs: Dict[str, Any] = {
            'temperature': temperature,
            'max_tokens': max_tokens,
        }
        if json_mode:
            kwargs['response_format'] = {'type': 'json_object'}

        cache_key = self._cache_key(messages, model, **kwargs)

        if use_cache:
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                return cached

        if client is None:
            raise RuntimeError(
                f"LLM client not initialised - set {key_env} env var "
                "and ensure the openai package is installed."
            )

        text = ''
        for _retry in range(3):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,
            )
            if not response.choices:
                raise RuntimeError(f"Empty choices in API response: {response}")
            raw = response.choices[0].message.content or ''
            text = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            if not text:
                m = re.search(r'<think>(.*?)</think>', raw, re.DOTALL)
                if m:
                    text = m.group(1).strip()
            if response.usage:
                u = self.usage.setdefault(model, {'in': 0, 'out': 0})
                u['in'] += response.usage.prompt_tokens
                u['out'] += response.usage.completion_tokens
            if text:
                break
            logger.warning(f"[{model}] empty response (retry {_retry + 1}/3)")

        if use_cache and text:  # don't cache empty responses
            self._save_to_cache(cache_key, model, text)

        logger.debug(
            f"[{model}] total {self.usage.get(model, {}).get('in', 0)}in "
            f"{self.usage.get(model, {}).get('out', 0)}out"
        )

        return text

    # ------------------------------------------------------------------
    # Public wrappers
    # ------------------------------------------------------------------

    def call_narrative(
        self,
        prompt: str,
        temperature: float = 0.5,
        use_cache: bool = True,
    ) -> str:
        """Call the narrative model (fast/cheap) for a ≤30-word market summary."""
        messages = [
            {
                'role': 'system',
                'content': (
                    'You are a senior CFA analyst writing a concise one-sentence '
                    'market commentary.  You must mention exactly one technical signal '
                    'AND one sentiment signal.  Maximum 30 words.'
                ),
            },
            {'role': 'user', 'content': prompt},
        ]
        return self._call(
            client=self._narrative_client,
            key_env=self._narrative_key_env,
            messages=messages,
            model=self._narrative_model,
            temperature=temperature,
            max_tokens=100,
            use_cache=use_cache,
            json_mode=False,
        )

    def call_alpha(
        self,
        prompt: str,
        temperature: float = 0.7,
        use_cache: bool = True,
    ) -> str:
        """Call the alpha model (strong) to generate 5 alpha expressions as JSON."""
        messages = [
            {
                'role': 'system',
                'content': (
                    'You are a quantitative researcher generating formulaic alpha '
                    'expressions.  Respond ONLY with valid JSON matching the exact '
                    'schema specified in the prompt.  No markdown, no explanations '
                    'outside the JSON object.'
                ),
            },
            {'role': 'user', 'content': prompt},
        ]
        return self._call(
            client=self._alpha_client,
            key_env=self._alpha_key_env,
            messages=messages,
            model=self._alpha_model,
            temperature=temperature,
            max_tokens=2048,
            use_cache=use_cache,
            json_mode=True,
        )
