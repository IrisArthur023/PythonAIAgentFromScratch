"""Provider-agnostic LLM interface, factory, and concrete providers.

Supports:
- OpenAI (GPT-4o, etc.)
- Anthropic (Claude 3.5 Sonnet, etc.)
- Local Ollama (llama3.2, etc.)
- OpenRouter (unified gateway to 100+ models)
"""

import os
import json
import urllib.request
import urllib.error
import urllib.parse
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage


class LLMProviderError(Exception):
    """Raised when an LLM provider fails with a user-facing explanation."""

    def __init__(self, message: str, status_code: int = 500, provider: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.provider = provider


class LLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    provider_name: str = "generic"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key

    @abstractmethod
    def get_chat_model(self) -> BaseChatModel:
        """Return an initialized LangChain chat model suitable for tool calling and agents."""
        pass

    def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        """Unified text generation method for standard queries."""
        try:
            chat_model = self.get_chat_model()
            messages = []
            if system:
                messages.append(SystemMessage(content=system))
            messages.append(HumanMessage(content=prompt))

            response = chat_model.invoke(messages, **kwargs)
            content = response.content
            if isinstance(content, list):
                return "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            return str(content)
        except LLMProviderError:
            raise
        except Exception as error:
            raise self._wrap_error(error)

    @abstractmethod
    def _wrap_error(self, error: Exception) -> LLMProviderError:
        """Convert provider-specific SDK exceptions into a user-friendly LLMProviderError."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation."""

    provider_name = "openai"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        selected_model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        super().__init__(model=selected_model, api_key=resolved_key)

    def get_chat_model(self) -> BaseChatModel:
        if not self.api_key:
            raise LLMProviderError(
                "OpenAI API key is missing. Please set OPENAI_API_KEY in your .env file or user settings.",
                status_code=400,
                provider=self.provider_name,
            )
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=self.model,
                api_key=self.api_key,
                temperature=0.2,
                request_timeout=60,
            )
        except ImportError:
            raise LLMProviderError(
                "langchain-openai package is not installed. Please run pip install langchain-openai.",
                status_code=500,
                provider=self.provider_name,
            )

    def _wrap_error(self, error: Exception) -> LLMProviderError:
        msg = str(error)
        status = 500
        if "AuthenticationError" in type(error).__name__ or "Incorrect API key" in msg or "401" in msg:
            status = 401
            friendly = "OpenAI authentication failed: Invalid or expired API key."
        elif "RateLimitError" in type(error).__name__ or "insufficient_quota" in msg or "429" in msg:
            status = 429
            friendly = "OpenAI rate limit or quota exceeded. Please check your plan and usage limits."
        elif "APITimeoutError" in type(error).__name__ or "timeout" in msg.lower():
            status = 504
            friendly = f"OpenAI request timed out while communicating with model '{self.model}'."
        elif "NotFoundError" in type(error).__name__ or "model_not_found" in msg:
            status = 404
            friendly = f"OpenAI model '{self.model}' not found or access is not permitted."
        else:
            friendly = f"OpenAI error: {msg}"
        return LLMProviderError(friendly, status_code=status, provider=self.provider_name)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider implementation."""

    provider_name = "anthropic"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        selected_model = model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        super().__init__(model=selected_model, api_key=resolved_key)

    def get_chat_model(self) -> BaseChatModel:
        if not self.api_key:
            raise LLMProviderError(
                "Anthropic API key is missing. Please set ANTHROPIC_API_KEY in your .env file or user settings.",
                status_code=400,
                provider=self.provider_name,
            )
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=self.model,
                api_key=self.api_key,
                temperature=0.2,
                timeout=60,
            )
        except ImportError:
            raise LLMProviderError(
                "langchain-anthropic package is not installed. Please run pip install langchain-anthropic.",
                status_code=500,
                provider=self.provider_name,
            )

    def _wrap_error(self, error: Exception) -> LLMProviderError:
        msg = str(error)
        status = 500
        if "AuthenticationError" in type(error).__name__ or "invalid x-api-key" in msg.lower() or "401" in msg:
            status = 401
            friendly = "Anthropic authentication failed: Invalid or expired API key."
        elif "RateLimitError" in type(error).__name__ or "rate_limit_error" in msg.lower() or "429" in msg:
            status = 429
            friendly = "Anthropic rate limit exceeded. Please wait a moment and try again."
        elif "APITimeoutError" in type(error).__name__ or "timeout" in msg.lower():
            status = 504
            friendly = f"Anthropic request timed out while communicating with model '{self.model}'."
        elif "NotFoundError" in type(error).__name__ or "not_found" in msg.lower():
            status = 404
            friendly = f"Anthropic model '{self.model}' was not found."
        else:
            friendly = f"Anthropic error: {msg}"
        return LLMProviderError(friendly, status_code=status, provider=self.provider_name)


class OllamaProvider(LLMProvider):
    """Local open-source model provider served via Ollama's OpenAI-compatible API.

    Uses:
        from openai import OpenAI
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama" # A placeholder key is required, but ignored by Ollama
        )
        response = client.chat.completions.create(
            model="qwen2.5-coder",
            messages=[{"role": "user", "content": "..."}]
        )
    """

    provider_name = "ollama"

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        selected_model = model or os.getenv("OLLAMA_MODEL", "qwen2.5-coder")
        resolved_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        super().__init__(model=selected_model)
        self.base_url = resolved_url

    @property
    def v1_url(self) -> str:
        """Return the OpenAI-compatible v1 endpoint for Ollama."""
        return f"{self.base_url}/v1" if not self.base_url.endswith("/v1") else self.base_url

    def get_client(self):
        """Return an OpenAI client configured for Ollama's local OpenAI-compatible endpoint."""
        from openai import OpenAI
        return OpenAI(
            base_url=self.v1_url,
            api_key="ollama",  # A placeholder key is required, but ignored by Ollama
        )

    def _check_server_health(self) -> None:
        """Verify the local Ollama instance is reachable."""
        req_url = f"{self.base_url}/api/tags"
        try:
            req = urllib.request.Request(req_url, headers={"User-Agent": "FieldNotes/1.0"})
            with urllib.request.urlopen(req, timeout=2.0) as response:
                if response.status == 200:
                    payload = json.loads(response.read().decode("utf-8"))
                    models = [m.get("name", "") for m in payload.get("models", [])]
                    model_found = any(
                        m == self.model or m.startswith(f"{self.model}:") or self.model.startswith(m)
                        for m in models
                    )
                    if not model_found and models:
                        pass
                    return
        except urllib.error.URLError:
            raise LLMProviderError(
                f"Local Ollama server is not running at {self.base_url}. "
                "Please start Ollama locally (e.g. run 'ollama serve' or launch the Ollama app).",
                status_code=503,
                provider=self.provider_name,
            )
        except Exception as err:
            raise LLMProviderError(
                f"Unable to reach Ollama at {self.base_url}: {err}",
                status_code=503,
                provider=self.provider_name,
            )

    def get_chat_model(self) -> BaseChatModel:
        """Return LangChain chat model targeting Ollama's OpenAI-compatible endpoint."""
        self._check_server_health()
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=self.model,
                base_url=self.v1_url,
                api_key="ollama",  # Placeholder key required, ignored by Ollama
                temperature=0.2,
            )
        except ImportError:
            try:
                from langchain_ollama import ChatOllama
                return ChatOllama(
                    model=self.model,
                    base_url=self.base_url,
                    temperature=0.2,
                )
            except ImportError:
                raise LLMProviderError(
                    "Ollama support requires 'langchain-openai' or 'langchain-ollama'. Please run: pip install langchain-openai",
                    status_code=500,
                    provider=self.provider_name,
                )

    def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        """Generate a response using Ollama via the OpenAI client."""
        self._check_server_health()
        try:
            client = self.get_client()
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs,
            )
            return response.choices[0].message.content or ""
        except Exception as error:
            raise self._wrap_error(error)

    def _wrap_error(self, error: Exception) -> LLMProviderError:
        msg = str(error)
        status = 500
        if "connection refused" in msg.lower() or "failed to connect" in msg.lower() or "connecterror" in type(error).__name__.lower():
            status = 503
            friendly = f"Local Ollama server is offline at {self.base_url}. Start Ollama to use local models."
        elif "model" in msg.lower() and "not found" in msg.lower():
            status = 404
            friendly = f"Model '{self.model}' was not found in Ollama. Run 'ollama run {self.model}' to pull it."
        elif "timeout" in msg.lower():
            status = 504
            friendly = f"Ollama generation timed out on model '{self.model}'."
        else:
            friendly = f"Ollama error: {msg}"
        return LLMProviderError(friendly, status_code=status, provider=self.provider_name)


class OpenRouterProvider(LLMProvider):
    """OpenRouter provider — unified gateway to 100+ models via OpenAI-compatible API.

    API docs: https://openrouter.ai/docs
    Model list: https://openrouter.ai/models
    Default model: openai/gpt-4o
    """

    provider_name = "openrouter"
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        selected_model = model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")
        resolved_key = api_key or os.getenv("OPENROUTER_API_KEY")
        super().__init__(model=selected_model, api_key=resolved_key)

    def get_chat_model(self) -> BaseChatModel:
        if not self.api_key:
            raise LLMProviderError(
                "OpenRouter API key is missing. Please set OPENROUTER_API_KEY in your .env file or user settings.",
                status_code=400,
                provider=self.provider_name,
            )
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=self.model,
                api_key=self.api_key,
                base_url=self.BASE_URL,
                temperature=0.2,
                request_timeout=60,
                default_headers={
                    # OpenRouter recommends these for analytics and routing
                    "HTTP-Referer": "http://localhost:3000",
                    "X-Title": "Field Notes",
                },
            )
        except ImportError:
            raise LLMProviderError(
                "langchain-openai package is not installed. Please run pip install langchain-openai.",
                status_code=500,
                provider=self.provider_name,
            )

    def _wrap_error(self, error: Exception) -> LLMProviderError:
        msg = str(error)
        status = 500
        if "401" in msg or "invalid_api_key" in msg.lower() or "no auth" in msg.lower():
            status = 401
            friendly = "OpenRouter authentication failed. Check your OPENROUTER_API_KEY."
        elif "429" in msg or "rate_limit" in msg.lower():
            status = 429
            friendly = "OpenRouter rate limit hit. Please wait and try again."
        elif "timeout" in msg.lower():
            status = 504
            friendly = f"OpenRouter request timed out on model '{self.model}'."
        elif "404" in msg or "not found" in msg.lower():
            status = 404
            friendly = (
                f"OpenRouter model '{self.model}' was not found. "
                "Browse available models at https://openrouter.ai/models"
            )
        elif "402" in msg or "insufficient" in msg.lower() or "credits" in msg.lower():
            status = 402
            friendly = "OpenRouter account has insufficient credits. Top up at https://openrouter.ai/credits"
        else:
            friendly = f"OpenRouter error: {msg}"
        return LLMProviderError(friendly, status_code=status, provider=self.provider_name)


def _http_json(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: int = 120) -> tuple[Dict[str, Any], int]:
    """POST JSON and return (body, status)."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8")), response.status
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", errors="replace")
        try:
            return json.loads(raw), error.code
        except json.JSONDecodeError:
            return {"error": raw or str(error)}, error.code
    except urllib.error.URLError as error:
        raise LLMProviderError(f"Could not reach the model API: {error.reason}", status_code=503)


def complete_chat(
    provider_name: str,
    model: str,
    system: str,
    messages: List[Dict[str, str]],
    user_settings: Optional[Dict[str, Any]] = None,
) -> str:
    """Multi-turn chat used by the Pixel frontend. Keys stay on the backend."""
    settings = user_settings or {}
    chosen = (provider_name or "").strip().lower()
    if not model or not re_model_id(model):
        raise LLMProviderError("Invalid model id.", status_code=400)
    if not isinstance(messages, list) or not messages:
        raise LLMProviderError("messages must be a non-empty list.", status_code=400)

    history = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content", "")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        history.append({"role": role, "content": content})
    if not history:
        raise LLMProviderError("No valid chat messages were provided.", status_code=400)

    if chosen == "anthropic":
        api_key = settings.get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMProviderError(
                "Anthropic API key is missing. Add ANTHROPIC_API_KEY in Backend/.env or the Access tab.",
                status_code=400,
                provider="anthropic",
            )
        body, status = _http_json(
            "https://api.anthropic.com/v1/messages",
            {"model": model, "max_tokens": 8000, "system": system or "", "messages": history},
            {
                "content-type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        if status >= 400:
            err = body.get("error") if isinstance(body, dict) else body
            message = err.get("message") if isinstance(err, dict) else str(err or "Anthropic request failed")
            raise LLMProviderError(message, status_code=status, provider="anthropic")
        parts = body.get("content") or []
        return "".join(p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text")

    if chosen == "gemini":
        api_key = settings.get("gemini_api_key") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise LLMProviderError(
                "Gemini API key is missing. Add GEMINI_API_KEY in Backend/.env or the Access tab.",
                status_code=400,
                provider="gemini",
            )
        contents = [
            {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": m["content"]}],
            }
            for m in history
        ]
        payload: Dict[str, Any] = {"contents": contents}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            f"?key={urllib.parse.quote(api_key)}"
        )
        body, status = _http_json(
            url,
            payload,
            {"content-type": "application/json", "x-goog-api-key": api_key},
        )
        if status >= 400:
            err = body.get("error") if isinstance(body, dict) else body
            message = err.get("message") if isinstance(err, dict) else str(err or "Gemini request failed")
            raise LLMProviderError(message, status_code=status, provider="gemini")
        cand = (body.get("candidates") or [{}])[0]
        parts = (cand.get("content") or {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts if isinstance(p, dict))

    if chosen == "openai":
        api_key = settings.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMProviderError(
                "OpenAI API key is missing. Add OPENAI_API_KEY in Backend/.env or the Access tab.",
                status_code=400,
                provider="openai",
            )
        openai_messages = []
        if system:
            openai_messages.append({"role": "system", "content": system})
        openai_messages.extend(history)
        body, status = _http_json(
            "https://api.openai.com/v1/chat/completions",
            {"model": model, "messages": openai_messages, "max_tokens": 8000},
            {"content-type": "application/json", "Authorization": f"Bearer {api_key}"},
        )
        if status >= 400:
            err = body.get("error") if isinstance(body, dict) else body
            message = err.get("message") if isinstance(err, dict) else str(err or "OpenAI request failed")
            raise LLMProviderError(message, status_code=status, provider="openai")
        choices = body.get("choices") or [{}]
        return ((choices[0].get("message") or {}).get("content")) or ""

    raise LLMProviderError(
        f"Unsupported chat provider '{chosen}'. Use anthropic, gemini, or openai.",
        status_code=400,
    )


def re_model_id(model: str) -> bool:
    return bool(model) and all(ch.isalnum() or ch in "._/-" for ch in model)


class GeminiProvider(LLMProvider):
    """Google Gemini via the public generateContent API."""

    provider_name = "gemini"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        selected_model = model or os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        super().__init__(model=selected_model, api_key=resolved_key)

    def get_chat_model(self) -> BaseChatModel:
        raise LLMProviderError(
            "Gemini tool-calling agents are not enabled. Use Plan/Code/Debug or Pixel chat instead.",
            status_code=400,
            provider=self.provider_name,
        )

    def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        return complete_chat(
            "gemini",
            self.model,
            system or "",
            [{"role": "user", "content": prompt}],
            user_settings={"gemini_api_key": self.api_key},
        )

    def _wrap_error(self, error: Exception) -> LLMProviderError:
        return LLMProviderError(str(error), status_code=500, provider=self.provider_name)


AVAILABLE_PROVIDERS = {
    "gemini": {
        "id": "gemini",
        "name": "Google Gemini",
        "default_model": "gemini-3.7-flash",
        "badge": "Gemini",
        "requires_key": True,
    },
    "anthropic": {
        "id": "anthropic",
        "name": "Anthropic Claude",
        "default_model": "claude-3-5-sonnet-20241022",
        "badge": "Claude",
        "requires_key": True,
    },
    "openai": {
        "id": "openai",
        "name": "OpenAI",
        "default_model": "gpt-4o",
        "badge": "GPT",
        "requires_key": True,
    },
    "openrouter": {
        "id": "openrouter",
        "name": "OpenRouter",
        "default_model": "openai/gpt-4o",
        "badge": "Router",
        "requires_key": True,
    },
    "ollama": {
        "id": "ollama",
        "name": "Local (Ollama)",
        "default_model": "qwen2.5-coder",
        "badge": "Local",
        "requires_key": False,
    },
}


def get_llm_provider(
    provider_name: Optional[str] = None,
    user_settings: Optional[Dict[str, Any]] = None,
) -> LLMProvider:
    """Factory returning the configured LLMProvider instance.

    Precedence:
    1. Explicit provider_name argument
    2. Preference from user_settings (Neon / in-memory)
    3. LLM_PROVIDER environment variable
    4. Default: 'anthropic'
    """
    settings = user_settings or {}
    chosen = (
        (provider_name or "").strip().lower()
        or (settings.get("preferred_provider") or "").strip().lower()
        or os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
    )

    custom_model = settings.get("preferred_model")

    if chosen == "openai":
        return OpenAIProvider(
            model=custom_model,
            api_key=settings.get("openai_api_key"),
        )
    elif chosen == "gemini":
        return GeminiProvider(
            model=custom_model,
            api_key=settings.get("gemini_api_key"),
        )
    elif chosen == "openrouter":
        return OpenRouterProvider(
            model=custom_model,
            api_key=settings.get("openrouter_api_key"),
        )
    elif chosen == "ollama":
        return OllamaProvider(
            model=custom_model,
            base_url=settings.get("ollama_base_url"),
        )
    elif chosen == "anthropic":
        return AnthropicProvider(
            model=custom_model,
            api_key=settings.get("anthropic_api_key"),
        )
    else:
        valid_options = ", ".join(AVAILABLE_PROVIDERS.keys())
        raise LLMProviderError(
            f"Unsupported LLM provider '{chosen}'. Choose one of: {valid_options}.",
            status_code=400,
        )
 
 