"""Drop-in replacement for paid Jina Reader API."""

from typing import Dict, Any
from local.free_web import read_public_webpage


def fetch_webpage_content_direct(url: str, timeout: int = 30) -> Dict[str, Any]:
    return read_public_webpage(url, timeout=timeout)
