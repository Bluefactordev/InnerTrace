"""Synthesis providers for summarizing LLM prompts/responses.

This module provides a plugin architecture for text synthesis:
- TruncationProvider: Simple truncation (no network calls, default)
- OpenAICompatibleProvider: Uses OpenAI-compatible API (OpenAI, vLLM, Ollama, etc.)
"""

import asyncio
import logging
import time
from typing import List, Protocol


class SynthesisProvider(Protocol):
    """Interface for text synthesis providers."""

    async def synthesize_batch(self, texts: List[str], max_words: int = 15) -> List[str]:
        """Synthesize a batch of texts into concise summaries.

        Args:
            texts: List of text strings to synthesize
            max_words: Maximum words per summary

        Returns:
            List of summaries (same length as input texts)
        """
        ...


class TruncationProvider:
    """Default provider: simple truncation (no network calls)."""

    async def synthesize_batch(self, texts: List[str], max_words: int = 15) -> List[str]:
        """Truncate texts to fixed length."""
        max_chars = max_words * 6  # Rough estimate: ~6 chars per word
        return [text[:max_chars] if text else "" for text in texts]


class OpenAICompatibleProvider:
    """Provider using OpenAI-compatible API (OpenAI, vLLM, Ollama, etc.).

    Uses /v1/chat/completions endpoint with standard Bearer token auth.
    Compatible with:
    - OpenAI API (https://api.openai.com/v1)
    - vLLM OpenAI server (http://localhost:8000/v1)
    - Ollama OpenAI compatibility (http://localhost:11434/v1)
    - Any other OpenAI-compatible endpoint
    """

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.3,
        max_tokens: int = 50,
        timeout: float = 180.0,
        max_concurrent: int = 10,
    ):
        """Initialize OpenAI-compatible provider.

        Args:
            base_url: Base URL for API (e.g., https://api.openai.com/v1)
            api_key: API key for authentication (Bearer token)
            model: Model ID to use
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens per response
            timeout: Request timeout in seconds (default: 60, configurable via config.json)
                     Note: Requests go through the app's planner system which adds overhead,
                     so timeout needs to account for planner processing time, not just LLM inference.
            max_concurrent: Maximum concurrent requests (default: 10)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_concurrent = max_concurrent

    async def synthesize_batch(self, texts: List[str], max_words: int = 15) -> List[str]:
        """Synthesize texts using OpenAI-compatible API.

        Args:
            texts: List of text strings to synthesize
            max_words: Maximum words per summary

        Returns:
            List of summaries (same length as input texts)
        """
        if not texts:
            return []

        try:
            import aiohttp
        except ImportError:
            logging.error(
                "aiohttp is required for OpenAI-compatible synthesis. "
                "Install with: pip install 'bf-trace[synthesis]'"
            )
            # Fallback to truncation
            return [text[:100] if text else "" for text in texts]

        system_prompt = (
            f"You are a text summarizer. Summarize the following text in maximum {max_words} words. "
            "Return ONLY the summary, nothing else."
        )

        # Per HTTP, non usare SSL. Per HTTPS, usa SSL normale
        # 🔧 FIX: Usa 127.0.0.1 invece di localhost per evitare problemi IPv6
        base_url_fixed = self.base_url.replace("localhost", "127.0.0.1")
        connector = aiohttp.TCPConnector(ssl=False) if base_url_fixed.startswith("http://") else None
        
        # Create timeout object
        timeout_obj = aiohttp.ClientTimeout(total=self.timeout)
        
        # 🔧 FIX: Salva base_url_fixed come variabile locale per accesso nella funzione annidata
        base_url_for_request = base_url_fixed
        
        async def synthesize_one(session: aiohttp.ClientSession, text: str, index: int) -> tuple[int, str]:
            """Synthesize a single text. Returns (index, summary)."""
            # Skip very short texts
            if not text or len(text) < 100:
                return (index, text[:100] if text else "")
            
            # Truncate input to avoid excessive costs
            text_to_summarize = text[:500] if len(text) > 500 else text
            
            # Prepare API request
            # 🔧 FIX: Usa base_url_for_request invece di self.base_url
            url = f"{base_url_for_request}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Summarize this text in maximum {max_words} words. Return ONLY the summary:\n\n{text_to_summarize}",
                    },
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            
            try:
                start_time = time.time()
                
                async with session.post(url, headers=headers, json=payload, timeout=timeout_obj) as response:
                    elapsed = time.time() - start_time
                    response.raise_for_status()
                    result = await response.json()
                    
                    # Extract content from OpenAI-style response
                    content = ""
                    if isinstance(result, dict):
                        choices = result.get("choices", [])
                        if choices and len(choices) > 0:
                            message = choices[0].get("message", {})
                            content = message.get("content", "")
                    
                    if content and content.strip():
                        logging.debug(f"Synthesis success for text {index} (len={len(text)}): {elapsed:.2f}s, summary_len={len(content)}")
                        return (index, content.strip())
                    else:
                        # Fallback to truncation
                        logging.warning(f"Synthesis empty response for text {index} (len={len(text)}): {elapsed:.2f}s. Result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
                        if isinstance(result, dict) and "choices" in result:
                            logging.warning(f"Synthesis: choices={result['choices']}")
                        return (index, text[:100])
            
            except asyncio.TimeoutError as e:
                elapsed = time.time() - start_time if 'start_time' in locals() else 0
                logging.warning(
                    f"Synthesis TIMEOUT for text {index} (len={len(text)}, elapsed={elapsed:.2f}s, timeout={self.timeout}s): {type(e).__name__}"
                )
                # Fallback to truncation on error
                return (index, text[:100])
            except Exception as e:
                elapsed = time.time() - start_time if 'start_time' in locals() else 0
                logging.warning(
                    f"Synthesis API call failed for text {index} (len={len(text)}, elapsed={elapsed:.2f}s): {type(e).__name__}: {str(e)[:200]}"
                )
                # Fallback to truncation on error
                return (index, text[:100])
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout_obj) as session:
            # Process texts in parallel batches
            # 
            # APPROACH: Multiple parallel HTTP requests (standard for OpenAI-compat APIs)
            # - OpenAI-compat APIs don't support batch requests (multiple prompts in one call)
            # - Standard practice: make parallel HTTP requests, one per prompt
            # - Flask/Uvicorn handles concurrent requests in parallel threads
            # - vLLM automatically batches simultaneous requests using continuous batching
            # - This leverages vLLM's GPU parallelism without needing custom batch endpoints
            #
            # Use semaphore to limit concurrent requests and avoid overwhelming the server
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async def synthesize_with_semaphore(text: str, index: int) -> tuple[int, str]:
                async with semaphore:
                    return await synthesize_one(session, text, index)
            
            # Create all tasks
            tasks = [synthesize_with_semaphore(text, i) for i, text in enumerate(texts)]
            
            # Execute in parallel (with concurrency limit via semaphore)
            results = await asyncio.gather(*tasks)
            
            # Sort by index to maintain original order
            results.sort(key=lambda x: x[0])
            summaries = [summary for _, summary in results]
        
        return summaries


def get_synthesis_provider(provider_type: str = "truncation", **kwargs) -> SynthesisProvider:
    """Factory function to create synthesis providers.

    Args:
        provider_type: Type of provider ('truncation' or 'openai')
        **kwargs: Provider-specific configuration

    Returns:
        SynthesisProvider instance

    Example:
        >>> # Truncation provider (default, no network)
        >>> provider = get_synthesis_provider("truncation")

        >>> # OpenAI provider
        >>> provider = get_synthesis_provider(
        ...     "openai",
        ...     base_url="https://api.openai.com/v1",
        ...     api_key="sk-...",
        ...     model="gpt-3.5-turbo"
        ... )

        >>> # vLLM provider (OpenAI-compatible)
        >>> provider = get_synthesis_provider(
        ...     "openai",
        ...     base_url="http://localhost:8000/v1",
        ...     api_key="EMPTY",  # vLLM doesn't require real API key
        ...     model="qwen2.5-7b-instruct"
        ... )
    """
    if provider_type == "truncation":
        return TruncationProvider()
    elif provider_type == "openai":
        return OpenAICompatibleProvider(**kwargs)
    else:
        raise ValueError(
            f"Unknown synthesis provider: {provider_type}. "
            f"Supported providers: 'truncation', 'openai'"
        )
