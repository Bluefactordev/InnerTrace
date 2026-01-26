"""Synthesis providers for summarizing LLM prompts/responses.

This module provides a plugin architecture for text synthesis:
- TruncationProvider: Simple truncation (no network calls, default)
- OpenAICompatibleProvider: Uses OpenAI-compatible API (OpenAI, vLLM, Ollama, etc.)
"""

import asyncio
import logging
import os
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
        timeout: float = 240.0,
        max_concurrent: int = 10,
        verbose: bool = False,
    ):
        """Initialize OpenAI-compatible provider.

        Args:
            base_url: Base URL for API (e.g., https://api.openai.com/v1)
            api_key: API key for authentication (Bearer token)
            model: Model ID to use
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens per response
            timeout: Request timeout in seconds (default: 240, configurable via config.json)
                     Note: Requests go through the app's planner system which adds overhead,
                     so timeout needs to account for planner processing time, not just LLM inference.
            max_concurrent: Maximum concurrent requests (default: 10)
            verbose: If True, enable debug logging for synthesis requests
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.verbose = verbose

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
            "Return ONLY the summary text, nothing else. Do NOT return code, JSON, or any structured data. "
            "Return ONLY plain text summary."
        )

        # Per HTTP, non usare SSL. Per HTTPS, usa SSL normale
        # 🔧 FIX: Usa 127.0.0.1 invece di localhost per evitare problemi IPv6
        base_url_fixed = self.base_url.replace("localhost", "127.0.0.1")
        connector = aiohttp.TCPConnector(ssl=False) if base_url_fixed.startswith("http://") else None
        
        # 🔧 FIX: Test rapido di connettività (2 secondi) prima di procedere
        # Se il server non è raggiungibile, fallback immediato a truncation
        import time
        test_start = time.time()
        quick_test_timeout = aiohttp.ClientTimeout(total=2.0, connect=1.0)
        test_connector = aiohttp.TCPConnector(ssl=False) if base_url_fixed.startswith("http://") else None
        server_reachable = False
        try:
            async with aiohttp.ClientSession(connector=test_connector, timeout=quick_test_timeout) as test_session:
                # Prova a connettersi al server con una richiesta HEAD veloce
                # Include API key per evitare 401 Unauthorized (che significa server raggiungibile)
                test_url = f"{base_url_fixed}/models"  # Endpoint standard OpenAI-compat
                test_headers = {
                    "Authorization": f"Bearer {self.api_key}",
                }
                async with test_session.head(test_url, headers=test_headers, allow_redirects=True) as test_response:
                    # Qualsiasi risposta HTTP (anche 401/403) significa che il server è raggiungibile
                    # Solo errori di connessione/network indicano server non raggiungibile
                    status = test_response.status
                    server_reachable = True
                    test_elapsed = time.time() - test_start
                    logging.debug(f"Synthesis connectivity test: server reachable (status={status}, elapsed={test_elapsed:.3f}s)")
                    if status >= 400 and status < 500:
                        # 4xx = server raggiungibile ma errore client (auth, not found, etc.)
                        # Questo è OK, significa che il server risponde
                        pass
                    elif status >= 500:
                        # 5xx = server raggiungibile ma errore server
                        # Anche questo è OK per il test di connettività
                        pass
                    # 2xx/3xx = tutto OK
            # Chiudi esplicitamente il test connector dopo il test
            if test_connector:
                await test_connector.close()
        except (aiohttp.ClientConnectorError, asyncio.TimeoutError, OSError) as e:
            # Solo errori di connessione/network = server non raggiungibile
            # 401/403/404/etc. non vengono catturati qui (sono risposte HTTP valide)
            test_elapsed = time.time() - test_start
            logging.warning(
                f"Synthesis server not reachable at {base_url_fixed}: {type(e).__name__} (test elapsed={test_elapsed:.3f}s). "
                f"Using truncation fallback (no LLM synthesis)."
            )
            # Chiudi il connector di test se esiste
            if test_connector:
                try:
                    await test_connector.close()
                except:
                    pass
            # 🎯 v0.2: Fallback intelligente - estrai formato PROMPT/RESP dal meta-prompt
            # Il nuovo prompt contiene "Richiesta:" e "Risposta:", estraiamo in formato separato
            fallback_results = []
            for text in texts:
                if not text:
                    fallback_results.append("")
                    continue
                # Cerca sezioni Richiesta/Risposta e costruisci formato PROMPT: ... | RESP: ...
                if "Richiesta:" in text and "Risposta:" in text:
                    try:
                        req_start = text.index("Richiesta:") + len("Richiesta:")
                        resp_start = text.index("Risposta:")
                        resp_end = text.index("Sintesi separata:") if "Sintesi separata:" in text else len(text)

                        prompt_snippet = text[req_start:resp_start].strip()[:40]  # Max 40 chars
                        response_snippet = text[resp_start + len("Risposta:"):resp_end].strip()[:50]  # Max 50 chars

                        fallback = f"PROMPT: {prompt_snippet} | RESP: {response_snippet}"
                        fallback_results.append(fallback[:max_words * 6])
                    except (ValueError, IndexError):
                        # Parsing fallito, usa troncamento semplice
                        max_chars = max_words * 6
                        fallback_results.append(text[:max_chars])
                else:
                    # Fallback al fallback: tronca tutto
                    max_chars = max_words * 6
                    fallback_results.append(text[:max_chars])
            return fallback_results
        except Exception as e:
            # Altri errori (non di connessione) = server raggiungibile ma problema diverso
            # Procediamo comunque con le chiamate reali
            test_elapsed = time.time() - test_start
            logging.debug(f"Synthesis connectivity test: unexpected error {type(e).__name__} (elapsed={test_elapsed:.3f}s), proceeding with synthesis")
            if test_connector:
                try:
                    await test_connector.close()
                except:
                    pass
        
        # Create timeout object per le richieste reali
        # 🔧 FIX: Crea un nuovo connector per le chiamate reali (non riutilizzare quello del test)
        # 🔧 FIX: Timeout di 240s totale per permettere richieste lunghe in sistemi multiagente
        #        Un timeout troppo basso può vanificare un intero run, quindi preferiamo attendere
        timeout_obj = aiohttp.ClientTimeout(
            total=self.timeout,  # 240s totale (per sistemi multiagente)
            connect=10.0,  # 10s per connettersi (abbondante)
            sock_read=self.timeout - 5.0  # 235s per leggere risposta (abbondante per richieste lunghe)
        )
        connector = aiohttp.TCPConnector(ssl=False) if base_url_fixed.startswith("http://") else None
        
        # 🔧 FIX: Salva base_url_fixed come variabile locale per accesso nella funzione annidata
        base_url_for_request = base_url_fixed
        
        async def synthesize_one(session: aiohttp.ClientSession, text: str, index: int) -> tuple[int, str]:
            """Synthesize a single text. Returns (index, summary)."""
            # Skip very short texts
            if not text or len(text) < 100:
                return (index, text[:100] if text else "")
            
            # 🔧 FIX: NON troncare il testo - l'utente ha esplicitamente richiesto di non porre limiti
            text_to_summarize = text
            
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
                        "content": f"Summarize this text in maximum {max_words} words. Return ONLY the summary text (no code, no JSON, no structured data):\n\n{text_to_summarize}",
                    },
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            
            # 🔧 RETRY INTELLIGENTE: Configurazione
            max_retries = 2  # Numero massimo di retry per errori transitori
            retry_delays = [0.5, 1.0]  # Backoff esponenziale: 0.5s, 1.0s
            transient_errors = {500, 502, 503, 504}  # Errori HTTP transitori
            
            # 🔧 RETRY INTELLIGENTE: Loop di retry per errori transitori
            last_exception = None
            for retry_attempt in range(max_retries + 1):  # 0 = primo tentativo, 1-N = retry
                # 🔧 DEBUG: Dizionario per tracciare le fasi (accessibile anche nel blocco except)
                phases = {"start": time.time()}
                
                # 🔧 DEBUG DETTAGLIATO per text 64 (errore 500 ricorrente) solo al primo tentativo
                is_text_64 = (index == 64)
                if is_text_64 and retry_attempt == 0:
                    import json
                    payload_str = json.dumps(payload, ensure_ascii=False)
                    payload_size = len(payload_str.encode('utf-8'))
                    text_preview = text[:500] if len(text) > 500 else text
                    text_tail = text[-200:] if len(text) > 200 else text
                    # Controlla caratteri problematici
                    problematic_chars = []
                    for char in text[:1000]:
                        if ord(char) > 127 and char not in problematic_chars[:10]:
                            problematic_chars.append(f"U+{ord(char):04X} ('{char}')")
                    
                    logging.error(
                        f"[SYNTHESIS DEBUG TEXT 64] ⚠️ DETTAGLI COMPLETI:\n"
                        f"  - text_len={len(text)}, text_bytes={len(text.encode('utf-8'))}\n"
                        f"  - text_preview (primi 500): {text_preview}\n"
                        f"  - text_tail (ultimi 200): {text_tail}\n"
                        f"  - payload_size={payload_size} bytes\n"
                        f"  - payload_preview (primi 1000): {payload_str[:1000]}\n"
                        f"  - url={url}\n"
                        f"  - model={self.model}, max_tokens={self.max_tokens}, temperature={self.temperature}\n"
                        f"  - timeout_obj: total={timeout_obj.total}s, connect={timeout_obj.connect}s, sock_read={timeout_obj.sock_read}s\n"
                        f"  - headers_keys={list(headers.keys())}\n"
                        f"  - problematic_chars (primi 10): {problematic_chars[:10] if problematic_chars else 'none'}\n"
                        f"  - text_encoding_check: valid_utf8={text.encode('utf-8', errors='strict') is not None}"
                    )
                
                # Se è un retry, aspetta con backoff esponenziale
                if retry_attempt > 0:
                    delay = retry_delays[retry_attempt - 1] if retry_attempt <= len(retry_delays) else retry_delays[-1]
                    logging.warning(f"[SYNTHESIS RETRY] text {index}: Retry attempt {retry_attempt}/{max_retries} dopo {delay}s (errore precedente: {type(last_exception).__name__})")
                    await asyncio.sleep(delay)
                
                try:
                    # 🔧 FIX: Mostra log di debug solo se verbose=True
                    if self.verbose or is_text_64:
                        logging.warning(f"[SYNTHESIS DEBUG] text {index}: Inizio richiesta - url={url}, text_len={len(text)}, payload_size={len(str(payload))}")
                    
                    # Fase 1: Invio richiesta HTTP e stabilimento connessione
                    phases["phase1_start"] = time.time()
                    if self.verbose:
                        logging.warning(f"[SYNTHESIS DEBUG] text {index}: Fase 1 - Invio POST request (elapsed={phases['phase1_start'] - phases['start']:.3f}s)")
                    
                    async with session.post(url, headers=headers, json=payload, timeout=timeout_obj) as response:
                        # 🔧 FIX: Imposta phase1_end immediatamente dopo l'apertura del context manager
                        phases["phase1_end"] = time.time()
                        phases["phase1_elapsed"] = phases["phase1_end"] - phases["phase1_start"]
                        if self.verbose:
                            logging.warning(f"[SYNTHESIS DEBUG] text {index}: Fase 1 completata - Connessione stabilita, status={response.status}, elapsed={phases['phase1_elapsed']:.3f}s")
                        
                        # Fase 2: Verifica status HTTP
                        phases["phase2_start"] = time.time()
                        if self.verbose:
                            logging.warning(f"[SYNTHESIS DEBUG] text {index}: Fase 2 - Verifica status HTTP (elapsed={phases['phase2_start'] - phases['start']:.3f}s)")
                        response.raise_for_status()
                        phases["phase2_end"] = time.time()
                        phases["phase2_elapsed"] = phases["phase2_end"] - phases["phase2_start"]
                        if self.verbose:
                            logging.warning(f"[SYNTHESIS DEBUG] text {index}: Fase 2 completata - Status OK, elapsed={phases['phase2_elapsed']:.3f}s")
                        
                        # Fase 3: Lettura risposta JSON
                        phases["phase3_start"] = time.time()
                        if self.verbose:
                            logging.warning(f"[SYNTHESIS DEBUG] text {index}: Fase 3 - Lettura response.json() (elapsed={phases['phase3_start'] - phases['start']:.3f}s)")
                        result = await response.json()
                        phases["phase3_end"] = time.time()
                        phases["phase3_elapsed"] = phases["phase3_end"] - phases["phase3_start"]
                        elapsed = phases["phase3_end"] - phases["start"]
                        if self.verbose:
                            logging.warning(f"[SYNTHESIS DEBUG] text {index}: Fase 3 completata - JSON ricevuto, size={len(str(result))}, elapsed={phases['phase3_elapsed']:.3f}s, total={elapsed:.3f}s")
                        
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
            
                except aiohttp.ClientResponseError as e:
                    # Errori HTTP (401, 403, 404, 500, etc.)
                    elapsed = time.time() - phases.get("start", time.time())
                    phase_info = f"phase1_elapsed={phases.get('phase1_elapsed', 0):.3f}s" if "phase1_elapsed" in phases else "phase1_not_started"
                    
                    # 🔧 RETRY INTELLIGENTE: Distingui tra errori transitori e permanenti
                    is_transient = e.status in transient_errors
                    
                    # 🔧 DEBUG DETTAGLIATO per text 64 in caso di errore 500
                    is_text_64 = (index == 64)
                    if is_text_64 and e.status == 500:
                        import json
                        try:
                            # Prova a leggere il body della risposta di errore se disponibile
                            error_body = ""
                            if hasattr(e, 'request_info') and hasattr(e.request_info, 'real_url'):
                                error_body = f"Request URL: {e.request_info.real_url}"
                            if hasattr(e, 'message'):
                                error_body += f", Error message: {e.message}"
                        except:
                            pass
                        
                        # response potrebbe non essere definito se l'errore avviene prima del context manager
                        response_status = "N/A"
                        try:
                            if 'response' in locals():
                                response_status = response.status
                        except:
                            pass
                        
                        logging.error(
                            f"[SYNTHESIS ERROR TEXT 64] ⚠️ ERRORE 500 DETTAGLIATO (attempt {retry_attempt + 1}/{max_retries + 1}):\n"
                            f"  - status={e.status}, message={e.message}\n"
                            f"  - elapsed={elapsed:.3f}s, {phase_info}\n"
                            f"  - text_len={len(text)}, text_bytes={len(text.encode('utf-8'))}\n"
                            f"  - url={url}\n"
                            f"  - payload_size={len(json.dumps(payload, ensure_ascii=False).encode('utf-8'))} bytes\n"
                            f"  - response_status={response_status}\n"
                            f"  - {error_body}\n"
                            f"  - phases_tracked={list(phases.keys())}\n"
                            f"  - is_transient={is_transient}, will_retry={is_transient and retry_attempt < max_retries}"
                        )
                    
                    logging.error(
                        f"[SYNTHESIS ERROR] text {index}: HTTP error - status={e.status}, message={e.message}, "
                        f"elapsed={elapsed:.3f}s, text_len={len(text)}, url={url}, {phase_info}, "
                        f"attempt={retry_attempt + 1}/{max_retries + 1}, is_transient={is_transient}"
                    )
                    
                    # 🔧 RETRY INTELLIGENTE: Se è un errore transitorio e ci sono ancora retry disponibili, riprova
                    if is_transient and retry_attempt < max_retries:
                        last_exception = e
                        continue  # Riprova con backoff
                    else:
                        # Errore permanente o retry esauriti: solleva per permettere fallback a MultiProvider
                        raise
                except (aiohttp.ClientConnectorError, aiohttp.ClientError, OSError) as e:
                    # Errori di connessione - potrebbero essere transitori
                    elapsed = time.time() - phases.get("start", time.time())
                    # 🔧 DEBUG: Determina in quale fase si è bloccata la richiesta
                    phase_info_parts = []
                    if "phase1_elapsed" in phases:
                        phase_info_parts.append(f"phase1_elapsed={phases['phase1_elapsed']:.3f}s")
                    elif "phase1_start" in phases:
                        phase1_current = time.time() - phases["phase1_start"]
                        phase_info_parts.append(f"phase1_stuck_at={phase1_current:.3f}s")
                    else:
                        phase_info_parts.append("phase1_not_started")
                    
                    if "phase2_elapsed" in phases:
                        phase_info_parts.append(f"phase2_elapsed={phases['phase2_elapsed']:.3f}s")
                    elif "phase2_start" in phases:
                        phase2_current = time.time() - phases["phase2_start"]
                        phase_info_parts.append(f"phase2_stuck_at={phase2_current:.3f}s")
                    
                    if "phase3_elapsed" in phases:
                        phase_info_parts.append(f"phase3_elapsed={phases['phase3_elapsed']:.3f}s")
                    elif "phase3_start" in phases:
                        phase3_current = time.time() - phases["phase3_start"]
                        phase_info_parts.append(f"phase3_stuck_at={phase3_current:.3f}s")
                    
                    phase_info = ", ".join(phase_info_parts) if phase_info_parts else "unknown_phase"
                    logging.error(
                        f"[SYNTHESIS ERROR] text {index}: Connection error - type={type(e).__name__}, "
                        f"error={str(e)[:200]}, elapsed={elapsed:.3f}s, text_len={len(text)}, url={url}, {phase_info}, "
                        f"attempt={retry_attempt + 1}/{max_retries + 1}"
                    )
                    
                    # 🔧 RETRY INTELLIGENTE: Errori di connessione sono transitori, ritenta se possibile
                    if retry_attempt < max_retries:
                        last_exception = e
                        continue  # Riprova con backoff
                    else:
                        # Retry esauriti: solleva per permettere fallback a MultiProvider
                        raise
                except asyncio.TimeoutError as e:
                    elapsed = time.time() - phases.get("start", time.time())
                    # 🔧 DEBUG: Log dettagliato per capire dove si blocca
                    phase_info_parts = []
                    if "phase1_elapsed" in phases:
                        phase_info_parts.append(f"phase1_elapsed={phases['phase1_elapsed']:.3f}s")
                    elif "phase1_start" in phases:
                        phase1_current = time.time() - phases["phase1_start"]
                        phase_info_parts.append(f"phase1_stuck_at={phase1_current:.3f}s")
                    else:
                        phase_info_parts.append("phase1_not_started")
                    
                    if "phase2_elapsed" in phases:
                        phase_info_parts.append(f"phase2_elapsed={phases['phase2_elapsed']:.3f}s")
                    elif "phase2_start" in phases:
                        phase2_current = time.time() - phases["phase2_start"]
                        phase_info_parts.append(f"phase2_stuck_at={phase2_current:.3f}s")
                    
                    if "phase3_elapsed" in phases:
                        phase_info_parts.append(f"phase3_elapsed={phases['phase3_elapsed']:.3f}s")
                    elif "phase3_start" in phases:
                        phase3_current = time.time() - phases["phase3_start"]
                        phase_info_parts.append(f"phase3_stuck_at={phase3_current:.3f}s")
                    
                    phase_info = ", ".join(phase_info_parts)
                    logging.error(
                        f"[SYNTHESIS TIMEOUT] text {index}: TIMEOUT - elapsed={elapsed:.3f}s, timeout={self.timeout}s, "
                        f"text_len={len(text)}, url={url}, {phase_info}, timeout_obj=total={timeout_obj.total}s,connect={timeout_obj.connect}s,sock_read={timeout_obj.sock_read}s, "
                        f"attempt={retry_attempt + 1}/{max_retries + 1}"
                    )
                    
                    # 🔧 RETRY INTELLIGENTE: Timeout sono transitori, ritenta se possibile
                    if retry_attempt < max_retries:
                        last_exception = e
                        continue  # Riprova con backoff
                    else:
                        # Retry esauriti: solleva per permettere fallback a MultiProvider
                        raise
                except Exception as e:
                    elapsed = time.time() - phases.get("start", time.time())
                    import traceback
                    tb_str = traceback.format_exc()[:500]
                    logging.error(
                        f"[SYNTHESIS ERROR] text {index}: Unexpected error - type={type(e).__name__}, "
                        f"error={str(e)[:300]}, elapsed={elapsed:.3f}s, text_len={len(text)}, url={url}, "
                        f"traceback={tb_str}, attempt={retry_attempt + 1}/{max_retries + 1}"
                    )
                    # 🔧 RETRY INTELLIGENTE: Per errori inaspettati, ritenta se possibile (potrebbero essere transitori)
                    if retry_attempt < max_retries:
                        last_exception = e
                        continue  # Riprova con backoff
                    else:
                        # Retry esauriti: solleva per permettere fallback a MultiProvider
                        raise
            
            # Se arriviamo qui, tutti i retry sono falliti
            if last_exception:
                raise last_exception
            else:
                # Questo non dovrebbe mai accadere, ma per sicurezza
                raise RuntimeError(f"All retry attempts failed for text {index}")
        
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
            # 🔧 FIX: Usa return_exceptions=True per catturare tutte le eccezioni
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verifica se tutte le richieste sono fallite
            http_errors = []
            connection_errors = []
            successful_results = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    if isinstance(result, aiohttp.ClientResponseError):
                        http_errors.append((i, result))
                    else:
                        connection_errors.append((i, result))
                else:
                    successful_results.append(result)
            
            # Se TUTTE le richieste sono fallite (HTTP errors o connection errors), solleva eccezione
            # per permettere a MultiProvider di provare il prossimo provider
            total_errors = len(http_errors) + len(connection_errors)
            if total_errors == len(texts) and len(successful_results) == 0:
                # Preferisci mostrare errori HTTP se presenti, altrimenti connection errors
                if http_errors:
                    error_status = http_errors[0][1].status
                    error_msg = http_errors[0][1].message
                    raise ConnectionError(
                        f"All synthesis requests failed with HTTP error {error_status}: {error_msg}. "
                        f"Provider {self.base_url} is not available or failed."
                    )
                elif connection_errors:
                    error_type = type(connection_errors[0][1]).__name__
                    error_msg = str(connection_errors[0][1])[:100]
                    raise ConnectionError(
                        f"All synthesis requests failed with {error_type}: {error_msg}. "
                        f"Provider {self.base_url} is not available."
                    )
            
            # Se la MAGGIORANZA delle richieste fallisce con errori 500 (server error), 
            # considera il provider non disponibile e solleva eccezione per fallback
            server_errors = [e for e in http_errors if e[1].status >= 500]
            if len(server_errors) > len(texts) * 0.5 and len(successful_results) < len(texts) * 0.5:
                error_status = server_errors[0][1].status
                error_msg = server_errors[0][1].message
                raise ConnectionError(
                    f"Most synthesis requests failed with HTTP error {error_status}: {error_msg}. "
                    f"Provider {self.base_url} is experiencing server errors ({len(server_errors)}/{len(texts)} failed)."
                )
            
            # Se ci sono alcuni successi, restituisci quelli + fallback per gli errori
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Fallback a truncation per errori individuali
                    text = texts[i] if i < len(texts) else ""
                    final_results.append((i, text[:100] if text else ""))
                else:
                    final_results.append(result)
            
            # Sort by index to maintain original order
            final_results.sort(key=lambda x: x[0])
            summaries = [summary for _, summary in final_results]
        
        return summaries

    async def evaluate_prompt_response_coherence(
        self,
        prompt: str,
        response: str,
    ) -> dict:
        """Evaluate if response semantically satisfies the prompt.

        This is a DIAGNOSTIC function for post-facto analysis.
        Does NOT modify events. Returns metadata only.

        Args:
            prompt: The original LLM prompt
            response: The LLM response to evaluate

        Returns:
            {
                "satisfaction": "full|partial|none",
                "reason": "one sentence explanation",
                "missing": ["field1", "field2"]  # if partial/none
            }

        Example:
            >>> provider = OpenAICompatibleProvider(...)
            >>> result = await provider.evaluate_prompt_response_coherence(
            ...     prompt="Extract fields: name, age, city",
            ...     response='{"name": "John"}'
            ... )
            >>> print(result)
            {
                "satisfaction": "partial",
                "reason": "Response contains name but missing age and city",
                "missing": ["age", "city"]
            }
        """
        try:
            import aiohttp
        except ImportError:
            logging.error(
                "aiohttp is required for coherence evaluation. "
                "Install with: pip install 'innertrace[synthesis]'"
            )
            return {
                "satisfaction": "none",
                "reason": "aiohttp not installed",
                "missing": []
            }

        # Truncate inputs to avoid excessive costs
        prompt_truncated = prompt[:800] if len(prompt) > 800 else prompt
        response_truncated = response[:800] if len(response) > 800 else response

        evaluation_prompt = f"""Evaluate if the RESPONSE adequately satisfies the REQUEST.

REQUEST:
{prompt_truncated}

RESPONSE:
{response_truncated}

Output ONLY valid JSON with this exact schema (no markdown, no code blocks):
{{
  "satisfaction": "full|partial|none",
  "reason": "one sentence explanation",
  "missing": []
}}

Rules:
- satisfaction="full" if response completely satisfies the request
- satisfaction="partial" if response partially satisfies (some fields missing/incomplete)
- satisfaction="none" if response fails to satisfy the request
- missing: array of missing elements (empty if full, list if partial/none)
- reason: concise 1-sentence explanation

Output JSON now:"""

        # Prepare API request
        base_url_fixed = self.base_url.replace("localhost", "127.0.0.1")
        url = f"{base_url_fixed}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": evaluation_prompt},
            ],
            "temperature": 0.0,  # Deterministic evaluation
            "max_tokens": 150,
        }

        connector = aiohttp.TCPConnector(ssl=False) if base_url_fixed.startswith("http://") else None
        # 🔧 FIX: Timeout separati per connect e sock_read per evitare blocchi
        timeout_obj = aiohttp.ClientTimeout(
            total=self.timeout,
            connect=5.0,  # Timeout per stabilire connessione
            sock_read=30.0  # Timeout per leggere risposta
        )

        try:
            async with aiohttp.ClientSession(connector=connector, timeout=timeout_obj) as session:
                async with session.post(url, headers=headers, json=payload) as api_response:
                    api_response.raise_for_status()
                    result = await api_response.json()

                    # Extract content from OpenAI-style response
                    content = ""
                    if isinstance(result, dict):
                        choices = result.get("choices", [])
                        if choices and len(choices) > 0:
                            message = choices[0].get("message", {})
                            content = message.get("content", "")

                    if content and content.strip():
                        # Parse JSON response
                        import json
                        import re

                        # Clean response (remove markdown code blocks if present)
                        content_clean = content.strip()
                        if content_clean.startswith("```"):
                            # Extract JSON from markdown code block
                            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content_clean, re.DOTALL)
                            if match:
                                content_clean = match.group(1)

                        try:
                            evaluation = json.loads(content_clean)

                            # Validate schema
                            if "satisfaction" in evaluation and "reason" in evaluation:
                                # Ensure missing field exists
                                if "missing" not in evaluation:
                                    evaluation["missing"] = []

                                # Validate satisfaction value
                                if evaluation["satisfaction"] not in ["full", "partial", "none"]:
                                    evaluation["satisfaction"] = "none"

                                return evaluation
                            else:
                                logging.warning(f"Invalid coherence evaluation schema: {evaluation}")
                                return {
                                    "satisfaction": "none",
                                    "reason": "Invalid evaluation response schema",
                                    "missing": []
                                }
                        except json.JSONDecodeError as e:
                            logging.warning(f"Failed to parse coherence evaluation JSON: {e}. Content: {content_clean[:200]}")
                            return {
                                "satisfaction": "none",
                                "reason": "Failed to parse evaluation response",
                                "missing": []
                            }
                    else:
                        return {
                            "satisfaction": "none",
                            "reason": "Empty evaluation response",
                            "missing": []
                        }

        except (aiohttp.ClientConnectorError, aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logging.warning(f"Coherence evaluation request failed: {type(e).__name__}: {str(e)[:100]}")
            return {
                "satisfaction": "none",
                "reason": f"Evaluation request failed: {type(e).__name__}",
                "missing": []
            }
        except Exception as e:
            logging.error(f"Unexpected error in coherence evaluation: {e}")
            return {
                "satisfaction": "none",
                "reason": f"Unexpected error: {type(e).__name__}",
                "missing": []
            }


class VLLMProvider:
    """Provider for direct vLLM endpoints (no API key required)."""
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        model: str = "qwen2.5-7b-instruct",
        timeout: float = 240.0,
        max_tokens: int = 500,
        max_concurrent: int = 20,
        connectivity_test_timeout: float = 5.0,
        verbose: bool = False,
    ):
        """Initialize vLLM provider.
        
        Args:
            host: vLLM server host
            port: vLLM server port
            model: Model name
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens per response
            max_concurrent: Maximum concurrent requests
            connectivity_test_timeout: Timeout for connectivity test
            verbose: If True, enable debug logging for synthesis requests
        """
        self.base_url = f"http://{host}:{port}/v1"
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.max_concurrent = max_concurrent
        self.connectivity_test_timeout = connectivity_test_timeout
        self.verbose = verbose
    
    async def synthesize_batch(self, texts: List[str], max_words: int = 15) -> List[str]:
        """Synthesize texts using vLLM endpoint."""
        # vLLMProvider è identico a OpenAICompatibleProvider ma senza API key
        provider = OpenAICompatibleProvider(
            base_url=self.base_url,
            api_key="",  # vLLM non richiede API key
            model=self.model,
            timeout=self.timeout,
            max_tokens=self.max_tokens,
            max_concurrent=self.max_concurrent,
            verbose=self.verbose,
        )
        return await provider.synthesize_batch(texts, max_words)


class MultiProvider:
    """Provider that tries multiple providers in order with fallback."""
    
    def __init__(self, providers: List[SynthesisProvider], verbose: bool = False, fallback_callback=None):
        """Initialize multi-provider.
        
        Args:
            providers: List of providers to try in order
            verbose: If True, log fallback messages
            fallback_callback: Optional callback function(message: str) for fallback messages
        """
        self.providers = providers
        self.verbose = verbose
        self.fallback_callback = fallback_callback
    
    async def synthesize_batch(self, texts: List[str], max_words: int = 15) -> List[str]:
        """Try each provider in order until one succeeds."""
        last_error = None
        
        for i, provider in enumerate(self.providers):
            try:
                results = await provider.synthesize_batch(texts, max_words)
                # Verifica che almeno alcuni risultati siano validi (non vuoti e non solo truncation)
                if results:
                    # Conta risultati validi (non vuoti e più lunghi di 50 caratteri per evitare truncation)
                    valid_results = [r for r in results if r and r.strip() and len(r.strip()) > 50]
                    if len(valid_results) >= len(texts) * 0.8:  # Almeno 80% di risultati validi
                        if i > 0 and self.verbose:
                            msg = f"Fallback: usando provider {i+1}/{len(self.providers)}"
                            if self.fallback_callback:
                                self.fallback_callback(msg)
                            logging.info(msg)
                        return results
                    else:
                        # Troppi risultati vuoti/truncation = provider non funziona correttamente
                        if self.verbose:
                            msg = f"Provider {i+1}/{len(self.providers)} ha prodotto troppi risultati vuoti ({len(valid_results)}/{len(texts)} validi), provando provider successivo"
                            if self.fallback_callback:
                                self.fallback_callback(msg)
                            logging.warning(msg)
                        # Solleva eccezione per provare provider successivo
                        raise ConnectionError(
                            f"Provider {i+1} produced too many empty/invalid results "
                            f"({len(valid_results)}/{len(texts)} valid). Trying next provider."
                        )
            except Exception as e:
                last_error = e
                if self.verbose:
                    msg = f"Provider {i+1}/{len(self.providers)} fallito: {type(e).__name__}"
                    if self.fallback_callback:
                        self.fallback_callback(msg)
                    logging.warning(msg)
                continue
        
        # Se tutti i provider falliscono, usa truncation come ultimo fallback
        logging.error(f"Tutti i provider falliti, usando truncation. Ultimo errore: {last_error}")
        truncation = TruncationProvider()
        return await truncation.synthesize_batch(texts, max_words)


def create_multi_provider_from_quality_config(quality_config: dict, verbose: bool = False) -> MultiProvider:
    """Create MultiProvider from quality level configuration.
    
    Args:
        quality_config: Configuration dict with 'providers' list
        verbose: If True, enable verbose logging
        
    Returns:
        MultiProvider instance
    """
    providers = []
    
    for provider_config in quality_config.get("providers", []):
        provider_type = provider_config.get("type")
        
        if provider_type == "truncation":
            providers.append(TruncationProvider())
        elif provider_type == "openai_compatible":
            # Leggi API key da config.json o variabile d'ambiente
            api_key = provider_config.get("api_key", "")
            if not api_key:
                # Fallback: leggi da variabile d'ambiente
                api_key = os.environ.get("BF_TRACE_SYNTHESIS_API_KEY", "")
            
            providers.append(OpenAICompatibleProvider(
                base_url=provider_config.get("base_url", "https://api.openai.com/v1"),
                api_key=api_key,
                model=provider_config.get("model", "gpt-3.5-turbo"),
                timeout=provider_config.get("timeout", 240.0),
                max_tokens=provider_config.get("max_tokens", 500),
                max_concurrent=provider_config.get("max_concurrent", 10),
                verbose=verbose,
            ))
        elif provider_type == "vllm":
            providers.append(VLLMProvider(
                host=provider_config.get("host", "127.0.0.1"),
                port=provider_config.get("port", 8000),
                model=provider_config.get("model", "qwen2.5-7b-instruct"),
                timeout=provider_config.get("timeout", 240.0),
                max_tokens=provider_config.get("max_tokens", 500),
                max_concurrent=provider_config.get("max_concurrent", 20),
                connectivity_test_timeout=provider_config.get("connectivity_test_timeout", 5.0),
                verbose=verbose,
            ))
        else:
            logging.warning(f"Provider type '{provider_type}' non riconosciuto, saltato")
    
    if not providers:
        logging.warning("Nessun provider valido trovato, usando truncation come fallback")
        providers.append(TruncationProvider())
    
    return MultiProvider(providers, verbose=verbose)


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
