"""Secure HTTP client with HTTPS enforcement and certificate validation.

Requirements: 20.3
"""

from __future__ import annotations

import ssl
from typing import Any
from urllib.parse import urlparse

import httpx


class InsecureURLError(Exception):
    """Raised when a URL does not use HTTPS."""

    def __init__(self, url: str) -> None:
        self.url = url
        super().__init__(f"URL must use HTTPS scheme, got: {url}")


def validate_url(url: str) -> str:
    """Validate that *url* uses the HTTPS scheme.

    Args:
        url: The URL to validate.

    Returns:
        The validated URL string.

    Raises:
        InsecureURLError: If the URL does not use ``https``.
        ValueError: If the URL is empty or malformed.
    """
    if not url or not url.strip():
        raise ValueError("URL cannot be empty")

    parsed = urlparse(url)

    if not parsed.scheme:
        raise ValueError(f"URL is missing a scheme: {url}")

    if parsed.scheme.lower() != "https":
        raise InsecureURLError(url)

    if not parsed.hostname:
        raise ValueError(f"URL is missing a hostname: {url}")

    return url


def _build_ssl_context(ca_cert_path: str | None = None) -> ssl.SSLContext:
    """Build an SSL context with certificate verification.

    Args:
        ca_cert_path: Optional path to a custom CA certificate bundle.
            When *None*, the system default CA certificates are used.

    Returns:
        A configured :class:`ssl.SSLContext`.
    """
    ctx = ssl.create_default_context()
    if ca_cert_path is not None:
        ctx.load_verify_locations(ca_cert_path)
    return ctx


def create_secure_client(
    ca_cert_path: str | None = None,
    timeout: float = 30.0,
) -> httpx.AsyncClient:
    """Create an :class:`httpx.AsyncClient` configured for secure HTTPS communication.

    The returned client:
    * Enables SSL certificate verification (using system CAs or a custom bundle).
    * Applies a default timeout.

    Callers are still expected to use :func:`validate_url` (or :class:`SecureHTTPClient`)
    to ensure only HTTPS URLs are requested.

    Args:
        ca_cert_path: Optional path to a custom CA certificate bundle.
        timeout: Request timeout in seconds.

    Returns:
        A configured :class:`httpx.AsyncClient`.
    """
    ssl_context = _build_ssl_context(ca_cert_path)
    return httpx.AsyncClient(
        verify=ssl_context,
        timeout=httpx.Timeout(timeout),
    )


class SecureHTTPClient:
    """Wrapper around :class:`httpx.AsyncClient` that enforces HTTPS and certificate validation.

    Usage::

        async with SecureHTTPClient() as client:
            resp = await client.get("https://example.com/api")
    """

    def __init__(
        self,
        ca_cert_path: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._ca_cert_path = ca_cert_path
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> SecureHTTPClient:
        self._client = create_secure_client(
            ca_cert_path=self._ca_cert_path,
            timeout=self._timeout,
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _ensure_open(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                "SecureHTTPClient is not open. Use 'async with SecureHTTPClient() as client:'"
            )
        return self._client

    # ------------------------------------------------------------------
    # HTTP verb helpers – each validates the URL before delegating.
    # ------------------------------------------------------------------

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        validate_url(url)
        return await self._ensure_open().get(url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        validate_url(url)
        return await self._ensure_open().post(url, **kwargs)

    async def put(self, url: str, **kwargs: Any) -> httpx.Response:
        validate_url(url)
        return await self._ensure_open().put(url, **kwargs)

    async def patch(self, url: str, **kwargs: Any) -> httpx.Response:
        validate_url(url)
        return await self._ensure_open().patch(url, **kwargs)

    async def delete(self, url: str, **kwargs: Any) -> httpx.Response:
        validate_url(url)
        return await self._ensure_open().delete(url, **kwargs)

    async def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        validate_url(url)
        return await self._ensure_open().request(method, url, **kwargs)
