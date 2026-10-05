import os
import re

import requests
from bs4 import BeautifulSoup
from typing_extensions import TypedDict

from .cache import cached


TIMEOUT = int(os.getenv("API_TIMEOUT", 30))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 Chrome/130 Safari/537.36"
    )
}


class WebpageResponse(TypedDict, total=False):
    url: str
    title: str
    content: str
    success: bool
    error: str


@cached()
def fetch_webpage_content(
    url: str,
    timeout: int = TIMEOUT,
) -> WebpageResponse:
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=timeout,
            allow_redirects=True,
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(
            [
                "script",
                "style",
                "nav",
                "footer",
                "header",
                "noscript",
                "svg",
                "form",
            ]
        ):
            tag.decompose()

        title = soup.title.get_text(" ", strip=True) if soup.title else ""

        main = soup.find("main") or soup.find("article") or soup.body or soup

        content = main.get_text("\n", strip=True)
        content = re.sub(r"\n{3,}", "\n\n", content)

        # Tránh nhét một trang web quá lớn vào model local.
        content = content[:50000]

        return {
            "url": response.url,
            "title": title,
            "content": content,
            "success": True,
        }

    except Exception as e:
        return {
            "url": url,
            "title": "",
            "content": "",
            "success": False,
            "error": str(e),
        }