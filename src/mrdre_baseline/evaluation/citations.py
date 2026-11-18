"""
Citation-based evaluation module.
Evaluates reports by checking if claims are supported by citations.
"""

import json
import os
import logging
import re
import hashlib
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from mrdre_baseline.evaluation.prompts import (
    get_citation_claim_extraction_prompts,
    get_citation_supported_judge_prompts,
    get_content_summarization_prompt,
)
from mrdre_baseline.evaluation.crawler.jina import fetch_webpage_content_jina
from mrdre_baseline.core.io import model_name

logger = logging.getLogger(__name__)

VALID_JUDGE_RESULTS = {"supported", "insufficient", "contradictory", "no_url"}


class CitationRequestRejected(RuntimeError):
    """Non-retriable Ollama HTTP 400; retains the server response for diagnosis."""


def _save_progress(path: Path, data: Any) -> None:
    """Atomic, crash-safe per-step checkpoint inside output/ (included in archive)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _read_progress(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (ValueError, OSError) as exc:
        logger.warning("Invalid checkpoint %s: %s", path, exc)
        return None


def _progress_key(report: str, *models: Any) -> str:
    # Includes judge implementation version and real model fingerprint/base URL;
    # does not silently reuse measurements from an altered evaluation setup.
    parts = ["citation-checkpoint-v2", report, os.getenv("MRDRE_LLM_BACKEND", ""),
             os.getenv("MRDRE_LOCAL_MODEL", ""),
             os.getenv("MRDRE_MODEL_FINGERPRINT", ""), os.getenv("OLLAMA_BASE_URL", "")]
    for model in models:
        parts += [str(model_name(model)) if model is not None else "none",
                  str(getattr(model, "system_prompt", ""))]
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()


def _ollama_error_text(response: Any) -> str:
    """Expose Ollama's actual error instead of hiding it behind requests.HTTPError."""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            detail = payload.get("error", payload.get("message", ""))
            if isinstance(detail, (dict, list)):
                detail = json.dumps(detail, ensure_ascii=False)
            if detail:
                return str(detail)[:1000]
    except (ValueError, TypeError):
        pass
    return str(getattr(response, "text", ""))[:1000]


def _judge_local_ollama(prompt: str, judge_model: Any) -> dict[str, str]:
    """Use the original judgment rubric, with full error reporting on HTTP 400."""
    import requests
    base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("MRDRE_LOCAL_MODEL", "qwen3:4b-instruct")
    schema = {"type": "object", "properties": {
        "result": {"type": "string", "enum": ["supported", "insufficient", "contradictory"]}},
        "required": ["result"], "additionalProperties": False}
    payload = {"model": model, "stream": False, "think": False, "format": schema,
               "messages": [{"role": "system", "content": judge_model.system_prompt},
                            {"role": "user", "content": prompt}],
               "options": {"temperature": 0, "num_predict": 128}}
    timeout = int(os.getenv("MRDRE_CITATION_REQUEST_TIMEOUT", "300"))
    attempts = max(1, int(os.getenv("MRDRE_CITATION_CLAIM_RETRIES", "4")))
    last_error = None
    for attempt in range(attempts):
        try:
            response = requests.post(base + "/api/chat", json=payload, timeout=timeout)
            if getattr(response, "status_code", None) == 400:
                detail = _ollama_error_text(response)
                raise CitationRequestRejected(
                    f"Ollama HTTP 400: {detail}; prompt_chars={len(prompt)}, "
                    f"system_chars={len(judge_model.system_prompt)}, model={model}, base={base}")
            response.raise_for_status()
            body = response.json()
            raw = body.get("message", {}).get("content", "")
            parsed = json.loads(raw)
            label = parsed.get("result") if isinstance(parsed, dict) else None
            if label not in ("supported", "insufficient", "contradictory"):
                raise ValueError(f"Invalid citation label {label!r}; model output={raw[:250]!r}")
            return {"result": label}
        except CitationRequestRejected:
            raise  # Repeating an identical invalid request cannot solve HTTP 400.
        except requests.HTTPError as exc:
            response = getattr(exc, "response", None)
            detail = _ollama_error_text(response) if response is not None else ""
            if response is not None and getattr(response, "status_code", None) == 400:
                raise CitationRequestRejected(f"Ollama HTTP 400: {detail}; "
                    f"prompt_chars={len(prompt)}; model={model}") from exc
            last_error = RuntimeError(f"{exc}; ollama_detail={detail}")
        except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
            last_error = exc
        logger.warning("Citation judge failed attempt %s/%s: %s: %s",
                       attempt + 1, attempts, type(last_error).__name__, str(last_error)[:500])
        if attempt + 1 < attempts:
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Citation judge exhausted {attempts} attempts: "
                       f"{type(last_error).__name__}: {last_error}") from last_error


def _partition_evidence(content: str, budget: int) -> List[str]:
    """Deterministically split ALL evidence without discarding any characters."""
    if not content:
        return []
    chunks = []
    cursor = 0
    while cursor < len(content):
        target = min(cursor + max(256, budget), len(content))
        end = target
        if target < len(content):
            lower = cursor + int((target - cursor) * 0.65)
            # Prefer paragraph boundaries, otherwise word boundaries.
            for delim in ("\n\n", "\n", " "):
                found = content.rfind(delim, lower, target)
                if found > cursor:
                    end = found + len(delim)
                    break
        if end <= cursor:
            end = target
        chunks.append(content[cursor:end])
        cursor = end
    assert "".join(chunks) == content
    return chunks


def _merge_piece_judgments(labels: List[str]) -> str:
    """Evidence from any piece may support a claim; conflicting pieces are insufficient."""
    has_support = "supported" in labels
    has_contradiction = "contradictory" in labels
    if has_support and not has_contradiction:
        return "supported"
    if has_contradiction and not has_support:
        return "contradictory"
    return "insufficient"


def _judge_local_bounded(claim_data: Dict[str, Any], judge_model: Any,
                         piece_cache: Path | None = None) -> dict[str, str]:
    """Evaluate full evidence in bounded pieces, splitting further on Ollama HTTP 400.

    Each piece uses the UNCHANGED original rubric. Pieces are separately cached, so
    a crash/resume never rejudges earlier completed portions of a long citation.
    """
    evidence = claim_data["combined_content"]
    claim = claim_data["claim"]
    budget = max(1500, int(os.getenv("MRDRE_CITATION_MAX_EVIDENCE_CHARS", "16000")))
    model_key = [model_name(judge_model), getattr(judge_model, "system_prompt", ""),
                 os.getenv("MRDRE_MODEL_FINGERPRINT", ""), os.getenv("OLLAMA_BASE_URL", ""),
                 os.getenv("MRDRE_LOCAL_MODEL", "")]

    def judge_piece(piece: str, depth: int = 0) -> str:
        prompt = get_citation_supported_judge_prompts([
            {"combined_content": piece, "claim": claim}])[0]
        checkpoint = None
        if piece_cache is not None:
            key = hashlib.sha256(json.dumps(["bounded-citation-v1", *model_key, prompt],
                                    ensure_ascii=False).encode()).hexdigest()
            checkpoint = piece_cache / (key + ".json")
            cached = _read_progress(checkpoint)
            if isinstance(cached, dict) and cached.get("result") in (
                    "supported", "insufficient", "contradictory"):
                return cached["result"]
        try:
            label = _judge_local_ollama(prompt, judge_model)["result"]
        except CitationRequestRejected as exc:
            if len(piece) <= 1500 or depth > 12:
                raise RuntimeError(f"Citation rejected even at {len(piece)} evidence chars: {exc}") from exc
            sections = _partition_evidence(piece, max(750, len(piece) // 2))
            if len(sections) < 2:
                raise RuntimeError(f"Cannot further split citation: {exc}") from exc
            logger.warning("Ollama rejected citation evidence (%d chars): %s. "
                           "Splitting into %d smaller segments, without discarding evidence.",
                           len(piece), str(exc)[:280], len(sections))
            label = _merge_piece_judgments([judge_piece(part, depth + 1) for part in sections])
        if checkpoint is not None:
            _save_progress(checkpoint, {"result": label})
        return label

    parts = _partition_evidence(evidence, budget)
    if len(parts) > 1:
        logger.info("Citation with %d evidence chars: %d bounded judging pieces",
                    len(evidence), len(parts))
    return {"result": _merge_piece_judgments([judge_piece(part) for part in parts])}

def _split_report_into_sections(report: str) -> List[str]:
    """
    Split the report into sections using double newlines. This excludes the references/sources section.
    Adaptively splits into 4 parts, with each part containing at least 2 \n\n-separated sections.
    
    From ResearcherBench.
    """
    reference_titles_hash = ['References', 'references', 'Key Citations']
    reference_titles_nohash = ['**References**', '参考资料', '**Sources:**', '**Works cited**', '**引用的著作**', '<div>⁂</div>', '<div style="text-align: center">⁂</div>']

    hash_pattern = '|'.join(re.escape(title) for title in reference_titles_hash)
    nohash_pattern = '|'.join(re.escape(title) for title in reference_titles_nohash)
    reference_pattern = fr'(?i)(?:^#+\s*(?:{hash_pattern})|^#*\s*(?:{nohash_pattern}))'
    
    # Remove references section if it exists
    report = re.split(reference_pattern, report, flags=re.MULTILINE)[0]
    
    sections = re.split(r'\n\n', report)
    sections = [s for s in sections if s.strip() != ""]

    # Adaptively choose step to split into 4 parts, with minimum 2 sections per part
    step = 1
    
    processed_sections = []
    for i in range(0, len(sections), step):
        processed_sections.append("\n\n".join(sections[i:i+step]))
    
    if not processed_sections:
        return [report.strip()]
        
    return processed_sections


def extract_claims(
    report: str,
    extraction_model: Any,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Extract claims from a report.
    
    Args:
        report: The generated report text
        extraction_model: Callable LLM used for claim extraction
        **kwargs: Additional arguments
    
    Returns:
        List of claim dictionaries with structure:
        [
            {
                "claim": str,
                "url": List[str]  # list of citation URLs (empty list if no citation)
            }
        ]
    """
    # Get all sections ready
    sections = _split_report_into_sections(report)
    if not sections:
        sections = [report]
    
    # Get all prompts
    prompts = get_citation_claim_extraction_prompts(report, sections)
    
    # Multi-threaded LLM calls
    def call_llm(prompt):
        if os.getenv("MRDRE_LLM_BACKEND", "").lower() != "ollama":
            return json.loads(extraction_model(prompt))

        import hashlib
        import requests
        from mrdre_baseline.agents.cache import DEFAULT_CACHE

        base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        model = os.getenv("MRDRE_LOCAL_MODEL", "qwen3.5:4b")
        system_prompt = extraction_model.system_prompt

        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "url": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["claim", "url"],
                "additionalProperties": False
            }
        }

        cache_input = json.dumps(
            [model, base, system_prompt, prompt, "structured-v1"],
            ensure_ascii=False
        )
        key = "structured_claims:" + hashlib.sha256(
            cache_input.encode()
        ).hexdigest()

        cached = DEFAULT_CACHE.get(key)
        if cached is not None:
            return cached

        payload = {"model": model, "messages": [
                   {"role": "system", "content": system_prompt},
                   {"role": "user", "content": prompt}],
                   "stream": False, "think": False, "format": schema,
                   "options": {"temperature": 0, "num_predict": 8192}}
        attempts = max(1, int(os.getenv("MRDRE_CLAIM_EXTRACTION_RETRIES", "3")))
        last_error = None
        for attempt in range(attempts):
            try:
                response = requests.post(base + "/api/chat", json=payload,
                                         timeout=int(os.getenv("MRDRE_CLAIM_EXTRACTION_TIMEOUT", "600")))
                response.raise_for_status()
                claims = json.loads(response.json()["message"]["content"])
                if not isinstance(claims, list) or not all(
                    isinstance(c, dict)
                    and isinstance(c.get("claim"), str)
                    and isinstance(c.get("url"), list)
                    and all(isinstance(u, str) for u in c["url"])
                    for c in claims
                ):
                    raise ValueError("Invalid claim extraction output")
                DEFAULT_CACHE.set(key, claims)
                return claims
            except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
                last_error = exc
                logger.warning("Claim extraction attempt %d/%d failed: %s: %s",
                               attempt + 1, attempts, type(exc).__name__, str(exc)[:350])
                if attempt + 1 < attempts:
                    time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"Claim extraction failed after {attempts} attempts: "
                           f"{type(last_error).__name__}: {last_error}") from last_error
    progress_dir = kwargs.get("progress_dir")
    parts_cache = Path(progress_dir) / "extraction_parts" if progress_dir else None
    results = [None] * len(prompts)
    missing = []
    for idx, prompt in enumerate(prompts):
        checkpoint = None
        if parts_cache is not None:
            digest = hashlib.sha256(json.dumps(["v2", prompt, model_name(extraction_model),
                getattr(extraction_model, "system_prompt", ""),
                os.getenv("MRDRE_MODEL_FINGERPRINT", "")], ensure_ascii=False).encode()).hexdigest()
            checkpoint = parts_cache / (digest + ".json")
            cached = _read_progress(checkpoint)
            if isinstance(cached, list):
                results[idx] = cached
                continue
        missing.append((idx, prompt, checkpoint))

    with ThreadPoolExecutor(max_workers=max(1, min(len(missing) or 1, int(os.getenv(
            "MRDRE_EVAL_WORKERS", "1" if os.getenv("MRDRE_LLM_BACKEND") == "ollama" else "20"))))) as executor:
        future_to_prompt = {executor.submit(call_llm, prompt): (idx, checkpoint)
                           for idx, prompt, checkpoint in missing}
        failures = []
        with tqdm(total=len(missing), desc="Extracting claims", disable=len(missing) < 4, leave=False) as pbar:
            for future in as_completed(future_to_prompt):
                idx, checkpoint = future_to_prompt[future]
                try:
                    results[idx] = future.result()
                    if checkpoint:
                        _save_progress(checkpoint, results[idx])
                except Exception as exc:
                    logger.exception("Claim extraction part %d failed", idx)
                    failures.append((idx, exc))
                pbar.update(1)
        if failures:
            idx, exc = failures[0]
            raise RuntimeError(f"Claim extraction failed for part {idx}: "
                               f"{type(exc).__name__}: {exc}") from exc
    # Flatten and normalize url field to always be a list
    claims = sum(results, [])
    for claim in claims:
        url = claim.get("url", [])
        if isinstance(url, str):
            claim["url"] = [url] if url else []
    return claims


def judge_claims(
    claims: List[Dict[str, Any]],
    judge_model: Any,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Judge whether claims are supported by their combined URL content.
    
    Args:
        claims: List of claim dictionaries, where each dict has:
            - "id": int
            - "claim": str
            - "url": List[str]
            - "combined_content": str (concatenated content/summaries from all URLs)
        judge_model: Model to use for judgment
        **kwargs: Additional arguments
    
    Returns:
        Same list with "result" field added to each claim
    """
    if not claims:
        return claims
    
    # Filter claims that have content to judge
    claims_to_judge = [(i, c) for i, c in enumerate(claims) if c.get("combined_content")]
    
    if not claims_to_judge:
        for claim in claims:
            claim["result"] = "no_url" if not claim.get("url") else "insufficient"
        return claims
    
    # Build prompts - one per claim
    prompts = get_citation_supported_judge_prompts([c for _, c in claims_to_judge])
    
    local = os.getenv("MRDRE_LLM_BACKEND", "").lower() == "ollama"
    progress_dir = kwargs.get("progress_dir")
    judge_cache = Path(progress_dir) / "judgments" if progress_dir else None

    def call_llm(prompt: str, claim_data: Dict[str, Any]) -> dict:
        if local:
            return _judge_local_bounded(claim_data, judge_model, judge_cache / "pieces" if judge_cache else None)
        return json.loads(judge_model(prompt))

    responses = [None] * len(prompts)
    missing = []
    for idx, prompt in enumerate(prompts):
        checkpoint = None
        if judge_cache is not None:
            # Key includes the COMPLETE evidence, claim and current scoring model.
            key = hashlib.sha256(json.dumps(["v2", prompt, model_name(judge_model),
                getattr(judge_model, "system_prompt", ""),
                os.getenv("MRDRE_MODEL_FINGERPRINT", "")], ensure_ascii=False).encode()).hexdigest()
            checkpoint = judge_cache / (key + ".json")
            cached = _read_progress(checkpoint)
            if isinstance(cached, dict) and cached.get("result") in (
                    "supported", "insufficient", "contradictory"):
                responses[idx] = cached
                continue
        missing.append((idx, prompt, checkpoint, claims_to_judge[idx][1]))
    if judge_cache is not None:
        logger.info("Citation judge: resumed %d/%d claims, remaining %d",
                    len(prompts) - len(missing), len(prompts), len(missing))

    with ThreadPoolExecutor(max_workers=max(1, min(len(missing) or 1, int(os.getenv(
            "MRDRE_EVAL_WORKERS", "1" if local else "10"))))) as executor:
        future_to_idx = {executor.submit(call_llm, prompt, claim_data): (idx, checkpoint)
                         for idx, prompt, checkpoint, claim_data in missing}
        with tqdm(total=len(missing), desc="Judging claims", disable=len(missing) < 5, leave=False) as pbar:
            errors = []
            for future in as_completed(future_to_idx):
                idx, checkpoint = future_to_idx[future]
                try:
                    response = future.result()
                    label = response.get("result") if isinstance(response, dict) else None
                    if label not in ("supported", "insufficient", "contradictory"):
                        raise ValueError(f"Invalid citation judge output: {str(response)[:400]}")
                    responses[idx] = response
                    if checkpoint:
                        _save_progress(checkpoint, response)
                except Exception as exc:
                    logger.exception("Citation claim %d failed", idx)
                    errors.append((idx, exc))
                pbar.update(1)
            if errors and local:
                idx, exc = errors[0]
                raise RuntimeError(f"Local citation judgement failed for claim {idx}: "
                                   f"{type(exc).__name__}: {exc}") from exc

    # Parse results
    for (orig_idx, _), response in zip(claims_to_judge, responses):
        if response and isinstance(response, dict):
            result = response.get("result", "error").lower()
            claims[orig_idx]["result"] = result if result in VALID_JUDGE_RESULTS else "error"
        else:
            claims[orig_idx]["result"] = "error"
    
    # Handle claims without URLs
    for claim in claims:
        if "result" not in claim:
            claim["result"] = "no_url" if not claim.get("url") else "insufficient"
    
    return claims


def crawl_urls(urls: List[str]) -> Dict[str, str]:
    """
    Crawl a list of URLs and return a mapping of URL -> content.
    Uses caching so duplicate URLs are only fetched once.
    
    Args:
        urls: List of URLs to crawl
    
    Returns:
        Dict mapping URL to its content
    """
    if not urls:
        return {}
    
    def crawl_url(url: str) -> Tuple[str, str]:
        try:
            result = fetch_webpage_content_jina(url)
            return (url, result.get("content", ""))
        except Exception as e:
            logger.warning(f"Crawl failed: {url[:50]}...")
            return (url, "")
    
    url_to_content = {}
    with ThreadPoolExecutor(max_workers=min(len(urls), 20)) as executor:
        futures = {executor.submit(crawl_url, url): url for url in urls}
        with tqdm(total=len(urls), desc="Crawling URLs", disable=len(urls) < 3, leave=False) as pbar:
            for future in as_completed(futures):
                url, content = future.result()
                url_to_content[url] = content
                pbar.update(1)
    
    return url_to_content


def summarize_url_content(
    url_to_content: Dict[str, str],
    url_to_claims: Dict[str, List[Dict[str, Any]]],
    summarization_model: Any,
) -> Dict[str, str]:
    """
    Summarize URL content to extract parts relevant to the claims citing each URL.
    
    Args:
        url_to_content: Mapping of URL to raw content
        url_to_claims: Mapping of URL to list of claims that cite it
        summarization_model: Model to use for summarization
    
    Returns:
        Mapping of URL to summarized content
    """
    if not url_to_content or summarization_model is None:
        return url_to_content
    
    # Filter URLs that have content to summarize
    urls_with_content = [url for url, content in url_to_content.items() if content]
    
    if not urls_with_content:
        return url_to_content
    
    def summarize(url: str) -> Tuple[str, str]:
        try:
            content = url_to_content[url]
            claims = url_to_claims.get(url, [])
            prompt = get_content_summarization_prompt(content, claims)
            summary = summarization_model(prompt)
            return (url, summary)
        except Exception as e:
            logger.error(f"Summarization failed: {url[:50]}...")
            return (url, url_to_content[url])  # Return original on error
    
    url_to_summary = dict(url_to_content)  # Start with original content
    with ThreadPoolExecutor(max_workers=min(
        len(urls_with_content),
        max(1, int(os.getenv("MRDRE_SUMMARY_WORKERS", os.getenv("MRDRE_EVAL_WORKERS", "2"))))
        if os.getenv("MRDRE_LLM_BACKEND") == "ollama" else 10
    )) as executor:
        futures = {executor.submit(summarize, url): url for url in urls_with_content}
        with tqdm(total=len(futures), desc="Summarizing", disable=len(urls_with_content) < 3, leave=False) as pbar:
            for future in as_completed(futures):
                url, summary = future.result()
                url_to_summary[url] = summary
                pbar.update(1)
    
    return url_to_summary


def evaluate_citations(
    report_obj: Dict[str, Any],
    claim_extraction_model: Any,
    supported_judge_model: Any,
    webpage_summarization_model: Any = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Evaluate a report's citations for faithfulness and groundedness.
    
    Args:
        report_obj: Generated report object
        claim_extraction_model: Model for claim extraction
        supported_judge_model: Model for judging claim support
        webpage_summarization_model: Optional model for summarizing webpage content
        **kwargs: Additional arguments
    
    Returns:
        Dict with structure:
        {
            "id": int,
            "claim_extraction_model": str,
            "supported_judge_model": str,
            "faithfulness_score": float,
            "groundedness_score": float,
            "total_claims": int,
            "supported_claims": int,
            "claims_details": List[Dict]
        }
    """
    report = report_obj.get("full_response", {}).get("report", "") or ""

    progress_root = kwargs.get("progress_dir")
    progress = (Path(progress_root) / _progress_key(
        report, claim_extraction_model, supported_judge_model, webpage_summarization_model)
        if progress_root is not None else None)

    # Checkpoint each expensive phase, keyed to report+model+prompt identity.
    claims_checkpoint = progress / "extracted_claims.json" if progress else None
    extracted_claims = _read_progress(claims_checkpoint) if claims_checkpoint else None
    if not isinstance(extracted_claims, list):
        extracted_claims = extract_claims(
            report=report, extraction_model=claim_extraction_model,
            progress_dir=progress,
            **{k: v for k, v in kwargs.items() if k != "progress_dir"})
        if claims_checkpoint:
            _save_progress(claims_checkpoint, extracted_claims)
    else:
        logger.info("Resumed %d extracted claims", len(extracted_claims))

    all_urls = set()
    url_to_claims = {}
    for claim in extracted_claims:
        for url in claim.get("url", []):
            all_urls.add(url)
            url_to_claims.setdefault(url, []).append(
                {"id": len(url_to_claims.get(url, [])), "claim": claim["claim"]})

    raw_checkpoint = progress / "web_content.json" if progress else None
    url_to_content = _read_progress(raw_checkpoint) if raw_checkpoint else None
    if not isinstance(url_to_content, dict):
        url_to_content = crawl_urls(sorted(all_urls))
        if raw_checkpoint:
            _save_progress(raw_checkpoint, url_to_content)
    if webpage_summarization_model is not None:
        summary_checkpoint = progress / "web_summaries.json" if progress else None
        summaries = _read_progress(summary_checkpoint) if summary_checkpoint else None
        if isinstance(summaries, dict):
            url_to_content = summaries
            logger.info("Resumed summaries for %d source URLs", len(summaries))
        else:
            url_to_content = summarize_url_content(url_to_content, url_to_claims,
                                                   webpage_summarization_model)
            if summary_checkpoint:
                _save_progress(summary_checkpoint, url_to_content)

    # Build claims with combined content for judging
    claims_for_judging = []
    for i, claim in enumerate(extracted_claims):
        urls = claim.get("url", [])
        # Concatenate content/summaries from all URLs for this claim
        content_parts = [url_to_content.get(url, "").strip() for url in urls]
        combined_content = "\n\n---\n\n".join([c for c in content_parts if c])
        
        claims_for_judging.append({
            "id": i,
            "claim": claim["claim"].strip(),
            "url": urls,
            "combined_content": combined_content,
        })

    # Judge claims
    claims_with_judgments = judge_claims(
        claims=claims_for_judging,
        judge_model=supported_judge_model,
        progress_dir=progress,
        **{k: v for k, v in kwargs.items() if k != "progress_dir"},
    )

    # Build claims_details for output (save summaries instead of raw url_content)
    claims_details = []
    for claim in claims_with_judgments:
        urls = claim.get("url", [])
        url_summaries = [url_to_content.get(url, "") for url in urls]
        claims_details.append({
            "claim": claim["claim"],
            "url": urls,
            "url_summaries": url_summaries,
            "result": claim.get("result", "no_url"),
        })

    # exclude claims with result "error" for metrics calculation since it comes from judgement model errors
    claims_without_errors = [item for item in claims_details if item.get("result") != "error"]

    total_claims = len(claims_without_errors)
    supported_claims = sum(1 for item in claims_without_errors if item.get("result") == "supported")
    claims_with_citations = [item for item in claims_without_errors if item.get("url")]

    faithfulness_score = (supported_claims / len(claims_with_citations)) if claims_with_citations else 0.0
    groundedness_score = (supported_claims / total_claims) if total_claims > 0 else 0.0
    return {
        "id": report_obj.get("id"),
        "claim_extraction_model": model_name(claim_extraction_model),
        "supported_judge_model": model_name(supported_judge_model),
        "faithfulness_score": round(faithfulness_score, 4),
        "groundedness_score": round(groundedness_score, 4),
        "total_claims": total_claims,
        "supported_claims": supported_claims,
        "claims_details": claims_details,
    }


def aggregate_citation_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate citation metrics across multiple samples.
    
    Args:
        metrics: List of sample-level citation metric objects
    
    Returns:
        Dict with aggregated metrics
    """
    if not metrics:
        return {}
    
    total_samples = len(metrics)
    avg_faithfulness = sum(metric.get("faithfulness_score", 0.0) for metric in metrics) / total_samples
    avg_groundedness = sum(metric.get("groundedness_score", 0.0) for metric in metrics) / total_samples
    total_claims = sum(metric.get("total_claims", 0) for metric in metrics)
    supported_claims = sum(metric.get("supported_claims", 0) for metric in metrics)

    return {
        "avg_faithfulness_score": round(avg_faithfulness, 4),
        "avg_groundedness_score": round(avg_groundedness, 4),
        "total_samples": total_samples,
        "total_claims": total_claims,
        "total_supported_claims": supported_claims,
    }

