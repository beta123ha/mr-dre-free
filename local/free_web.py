"""Free web search and direct article retrieval.

This module replaces a *service*, never the DRA decision-making algorithm.
Content is untrusted external text. URLs are validated to exclude local/private
hosts before retrieval; the web reader does not follow redirects.
"""

import ipaddress
import re
import io
import socket
from html import unescape
from urllib.parse import urlsplit, urljoin

import requests


def _validate_public_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in ("https", "http") or not parsed.hostname:
        raise ValueError("Only public HTTP(S) URLs are allowed")
    if parsed.username or parsed.password:
        raise ValueError("URLs with credentials are not allowed")
    host = parsed.hostname
    if host.lower() in ("localhost", "localhost.localdomain") or host.lower().endswith(".local"):
        raise ValueError("Private host is not allowed")
    try:
        resolved = socket.getaddrinfo(host, None)
    except OSError as exc:
        raise ValueError(f"Unresolvable URL hostname: {host}") from exc
    for address in resolved:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ValueError("Local/private/reserved addresses are not allowed")


def search_web(query: str, num_results: int = 5) -> list[dict]:
    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise RuntimeError("Install the ddgs package for free web search") from exc
    if not query.strip():
        raise ValueError("Search query must not be empty")
    searcher = DDGS()
    data = list(searcher.text(query, max_results=min(max(num_results, 1), 10)))
    return [{"title": str(r.get("title") or ""),
             "url": str(r.get("href") or ""),
             "snippet": str(r.get("body") or "")}
            for r in data if r.get("href", "").startswith(("https://", "http://"))]


def read_public_webpage(url: str, timeout: int = 30, max_chars: int = 50000) -> dict:
    # Validate each redirect target. Do not let a public URL redirect to an
    # internal service, e.g. Ollama on localhost.
    original_url = url
    for _ in range(6):
        _validate_public_url(url)
        response = requests.get(
            url, headers={"User-Agent": "Mozilla/5.0 (compatible; MrDreAcademicResearch/1.0)"},
            timeout=timeout, allow_redirects=False,
        )
        if 300 <= response.status_code < 400:
            location = response.headers.get("location")
            if not location:
                raise ValueError("Redirect without Location header")
            url = urljoin(url, location)
            continue
        response.raise_for_status()
        break
    else:
        raise ValueError("Too many redirects")

    typ = response.headers.get("content-type", "").lower()
    if len(response.content) > 8_000_000:
        raise ValueError("Webpage too large")
    title = ""
    if "pdf" in typ or urlsplit(url).path.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Install pypdf for PDF sources") from exc
        reader = PdfReader(io.BytesIO(response.content))
        body = []
        for page in reader.pages:
            body.append(page.extract_text() or "")
            if sum(len(x) for x in body) >= max_chars:
                break
        raw = "\n\n".join(body)
    elif "html" in typ:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()
        raw = soup.get_text("\n", strip=True)
    elif "text/plain" in typ or "text/markdown" in typ:
        raw = response.text
    else:
        raise ValueError(f"Unsupported webpage media type: {typ}")
    text = unescape(re.sub(r"\n{3,}", "\n\n", raw))[:max_chars]
    if not text.strip():
        raise ValueError("Page has no extractable text")
    return {"url": url, "title": title, "content": text, "description": "",
            "publishedTime": "", "metadata": {}, "success": True,
            "requested_url": original_url}
