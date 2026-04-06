"""Unit tests for the secure HTTP client module.

Requirements: 20.3
"""

import ssl
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from agentlegatus.security.http_client import (
    InsecureURLError,
    SecureHTTPClient,
    _build_ssl_context,
    create_secure_client,
    validate_url,
)


# ---------------------------------------------------------------------------
# validate_url
# ---------------------------------------------------------------------------

class TestValidateUrl:
    """Req 20.3 — only HTTPS URLs are allowed."""

    def test_https_url_accepted(self):
        url = "https://example.com/api/v1"
        assert validate_url(url) == url

    def test_http_url_rejected(self):
        with pytest.raises(InsecureURLError):
            validate_url("http://example.com/api")

    def test_ftp_url_rejected(self):
        with pytest.raises(InsecureURLError):
            validate_url("ftp://files.example.com/data")

    def test_empty_url_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            validate_url("")

    def test_whitespace_only_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            validate_url("   ")

    def test_missing_scheme_rejected(self):
        with pytest.raises(ValueError, match="missing a scheme"):
            validate_url("example.com/path")

    def test_missing_hostname_rejected(self):
        with pytest.raises(ValueError, match="missing a hostname"):
            validate_url("https://")

    def test_https_with_port_accepted(self):
        url = "https://example.com:8443/api"
        assert validate_url(url) == url

    def test_https_with_query_accepted(self):
        url = "https://example.com/api?key=val"
        assert validate_url(url) == url


# ---------------------------------------------------------------------------
# _build_ssl_context
# ---------------------------------------------------------------------------

class TestBuildSslContext:

    def test_default_context_verifies_certs(self):
        ctx = _build_ssl_context()
        assert ctx.verify_mode == ssl.CERT_REQUIRED
        assert ctx.check_hostname is True

    def test_custom_ca_cert_path(self, tmp_path):
        # Create a dummy PEM file (won't be a real cert, but load_verify_locations
        # is called — we just verify the path is accepted without crashing for
        # a real PEM).  We test the wiring, not OpenSSL itself.
        import certifi
        ctx = _build_ssl_context(ca_cert_path=certifi.where())
        assert ctx.verify_mode == ssl.CERT_REQUIRED

    def test_invalid_ca_path_raises(self):
        with pytest.raises(OSError):
            _build_ssl_context(ca_cert_path="/nonexistent/ca-bundle.pem")


# ---------------------------------------------------------------------------
# create_secure_client
# ---------------------------------------------------------------------------

class TestCreateSecureClient:

    def test_returns_async_client(self):
        client = create_secure_client()
        assert isinstance(client, httpx.AsyncClient)

    def test_default_timeout(self):
        client = create_secure_client()
        assert client.timeout.connect == 30.0

    def test_custom_timeout(self):
        client = create_secure_client(timeout=10.0)
        assert client.timeout.connect == 10.0

    def test_ssl_verification_enabled(self):
        client = create_secure_client()
        # The verify parameter should be an ssl.SSLContext (not False/True)
        assert isinstance(client._transport._pool._ssl_context, ssl.SSLContext)


# ---------------------------------------------------------------------------
# SecureHTTPClient
# ---------------------------------------------------------------------------

class TestSecureHTTPClient:

    async def test_context_manager_opens_and_closes(self):
        async with SecureHTTPClient() as client:
            assert client._client is not None
        assert client._client is None

    async def test_get_rejects_http(self):
        async with SecureHTTPClient() as client:
            with pytest.raises(InsecureURLError):
                await client.get("http://example.com")

    async def test_post_rejects_http(self):
        async with SecureHTTPClient() as client:
            with pytest.raises(InsecureURLError):
                await client.post("http://example.com")

    async def test_put_rejects_http(self):
        async with SecureHTTPClient() as client:
            with pytest.raises(InsecureURLError):
                await client.put("http://example.com")

    async def test_patch_rejects_http(self):
        async with SecureHTTPClient() as client:
            with pytest.raises(InsecureURLError):
                await client.patch("http://example.com")

    async def test_delete_rejects_http(self):
        async with SecureHTTPClient() as client:
            with pytest.raises(InsecureURLError):
                await client.delete("http://example.com")

    async def test_request_rejects_http(self):
        async with SecureHTTPClient() as client:
            with pytest.raises(InsecureURLError):
                await client.request("GET", "http://example.com")

    async def test_not_open_raises_runtime_error(self):
        client = SecureHTTPClient()
        with pytest.raises(RuntimeError, match="not open"):
            await client.get("https://example.com")

    async def test_custom_ca_and_timeout(self):
        import certifi
        async with SecureHTTPClient(ca_cert_path=certifi.where(), timeout=5.0) as client:
            inner = client._client
            assert inner is not None
            assert inner.timeout.connect == 5.0


# ---------------------------------------------------------------------------
# Integration-style: validate_url is called before every request verb
# ---------------------------------------------------------------------------

class TestSecureHTTPClientValidation:
    """Ensure every HTTP verb goes through validate_url."""

    async def test_get_validates(self):
        async with SecureHTTPClient() as client:
            with pytest.raises(InsecureURLError):
                await client.get("http://evil.com")

    async def test_post_validates(self):
        async with SecureHTTPClient() as client:
            with pytest.raises(InsecureURLError):
                await client.post("http://evil.com", json={"a": 1})
