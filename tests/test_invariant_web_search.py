import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.web_search import web_search

INTERNAL_PAYLOADS = [
    "http://169.254.169.254/latest/meta-data/",  # AWS metadata endpoint (exact exploit)
    "http://192.168.1.1/admin",                   # Private network boundary case
    "https://www.google.com/search?q=test",       # Valid public URL
    "http://10.0.0.1/internal-service",           # RFC1918 private range
]

INTERNAL_PATTERNS = [
    "169.254.",
    "192.168.",
    "10.",
    "172.16.", "172.17.", "172.18.", "172.19.",
    "172.20.", "172.21.", "172.22.", "172.23.",
    "172.24.", "172.25.", "172.26.", "172.27.",
    "172.28.", "172.29.", "172.30.", "172.31.",
    "127.",
    "::1",
    "localhost",
]


@pytest.mark.parametrize("url", INTERNAL_PAYLOADS)
def test_web_search_blocks_internal_network_access(url):
    """Invariant: web_search must never successfully fetch internal/private network resources."""
    is_internal = any(pattern in url for pattern in INTERNAL_PATTERNS)

    if is_internal:
        try:
            result = web_search(url)
            # If a result is returned, it must not contain sensitive internal data indicators
            assert result is None or "meta-data" not in str(result).lower(), (
                f"SSRF: web_search returned internal resource content for URL: {url}"
            )
            # The function should raise, return error, or return empty — not silently succeed
            assert not result or "error" in str(result).lower() or len(str(result)) == 0, (
                f"SSRF: web_search did not block internal URL: {url}"
            )
        except (ValueError, PermissionError, ConnectionError):
            pass  # Blocking via exception is acceptable
    else:
        # Valid public URL should not raise a blocking error
        try:
            result = web_search(url)
            # Just verify it doesn't crash unexpectedly for valid input
        except Exception:
            pass  # Network errors in test env are acceptable