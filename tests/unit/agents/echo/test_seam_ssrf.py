"""OSIRIS-ssrf (2026-07-06 red-team, Step 1) — tests for the seam-owned SSRF
guard replacing the upstream web_tools.fetch_url handler (which had ZERO SSRF
protection: httpx.get(url, follow_redirects=True) fetched 169.254.169.254 /
127.0.0.1 / ::1 / fc00::/7 with no check).

Two layers landed in one commit:
  1. ``seam_safe_fetch.check_url`` — the in-process floor-gate entry, called from
     ``execute_tools`` for any tool call with a ``url`` param (after guard-source,
     before cert). Refuses reserved/private ranges + per-IP rate limit.
  2. ``seam_safe_fetch.safe_fetch_wrapper`` — the sandboxed handler re-registered
     as fetch_url (replacing upstream); re-checks the initial url + every
     redirect hop (manual redirect loop, follow_redirects=False) — catches
     SSRF-by-redirect the floor gate cannot see.

Unit tier (no network/Ollama — run in apply_seam.sh post-deploy
attestation): check_url / is_url_reserved on IP literals + name blocklist (no
DNS), canonical-form obfuscation defeat (ipaddress normalization), the
execute_tools floor gate via a fake registry (routes through the real
metadata-driven gates), handler re-registration, rate limit, + the handler
redirect re-validation with a MOCKED httpx.get (no network).

Integration tier (``@pytest.mark.integration``, skipped by apply_seam.sh's
``-m 'not integration'``): one live fetch of a public url (needs DNS + network;
``pytest.skip`` when unreachable).

Leak-probe neutrality: no leak-probe arm supplies a ``url`` param (verified
2026-07-06), so the floor gate never fires for an arm — the 12/12 leak-probe
suite is unaffected (regression-guarded by the separate ``test_leak_probe.py``).
"""
import pytest

from hermes_cli.agents.echo.agent import SeamedTool, execute_tools, _build_registry
from hermes_cli.agents.echo.state import EchoState
from hermes_cli.agents.echo.tools.registry import Tool
from hermes_cli.agents.echo.tools.seam_safe_fetch import (
    check_url, is_url_reserved, safe_fetch_wrapper, _RATE_STATE,
)


@pytest.fixture(autouse=True)
def _clear_rate_state():
    """Hermeticity: the in-process rate-limit state is module-level + persists
    across tests. Clear it before each test so a prior test's hits don't trip
    the limit (and so the rate-limit test starts from zero)."""
    _RATE_STATE.clear()
    yield
    _RATE_STATE.clear()


# ---------------------------------------------------------------------------
# check_url / is_url_reserved — reserved-range refusal (no DNS: IP literals +
# name blocklist short-circuit before getaddrinfo)
# ---------------------------------------------------------------------------

class TestCheckUrlReservedRefusal:
    def test_refuses_cloud_metadata_ipv4(self):
        # 169.254.169.254 — the AWS/GCP/Azure cloud-metadata endpoint.
        r = check_url("http://169.254.169.254/latest/meta-data/iam/security-credentials/")
        assert r is not None
        assert "169.254" in r

    def test_refuses_loopback_ipv4(self):
        assert check_url("http://127.0.0.1:8080/") is not None
        assert check_url("http://127.0.0.1/") is not None

    def test_refuses_rfc1918_private(self):
        assert check_url("http://10.0.0.1/") is not None
        assert check_url("http://172.16.0.1/") is not None
        assert check_url("http://192.168.1.1/") is not None

    def test_refuses_cgnat(self):
        # 100.64/10 CGNAT (RFC6598) — only folded into is_private in Py3.13, so
        # the explicit _REJECTED_V4 list is the version-independent stop.
        r = check_url("http://100.64.0.1/")
        assert r is not None, "CGNAT 100.64/10 must be refused"

    def test_refuses_ipv6_loopback_and_ula(self):
        assert check_url("http://[::1]/") is not None
        assert check_url("http://[fc00::1]/") is not None
        assert check_url("http://[fe80::1]/") is not None

    def test_refuses_name_blocklist(self):
        # metadata.google.internal — the GCP metadata hostname (blocklist
        # short-circuits BEFORE DNS, so no network needed).
        assert check_url("http://metadata.google.internal/computeMetadata/v1/") is not None
        assert check_url("http://localhost/") is not None

    def test_refuses_zero_and_multicast(self):
        assert check_url("http://0.0.0.0/") is not None
        assert check_url("http://224.0.0.1/") is not None  # multicast
        assert check_url("http://240.0.0.1/") is not None  # reserved

    def test_refuses_deprecated_6to4_anycast(self):
        # 192.88.99.0/24 — deprecated 6to4 anycast (RFC 7526); not always flagged
        # by is_private/is_reserved, so the explicit _REJECTED_V4 list catches it.
        # 4-lens lens B-finding-f: this gap was open before the re-verify fix.
        assert check_url("http://192.88.99.1/") is not None

    def test_refuses_ipv6_documentation_prefix(self):
        # 2001:db8::/32 — the documentation prefix; 4-lens lens B-finding-f.
        assert check_url("http://[2001:db8::1]/") is not None


class TestSchemeAllowlist:
    """A non-http(s) scheme is refused fail-closed INDEPENDENT of httpx's scheme
    handling — defense-in-depth so a hostless / local-file target cannot be
    reached via a redirect Location. 4-lens lens B-finding-b."""

    def test_refuses_file_scheme(self):
        r = check_url("file:///etc/passwd")
        assert r is not None
        assert "scheme" in r

    def test_refuses_data_scheme(self):
        assert check_url("data:text/html,<script>") is not None

    def test_refuses_gopher_scheme(self):
        assert check_url("gopher://127.0.0.1/") is not None

    def test_handler_refuses_redirect_to_file_scheme(self, monkeypatch):
        """A public url that 302-redirects to a file:/// target is refused at the
        redirect hop (the scheme allowlist fires before any fetch of the target)."""
        def _fake_get(url, **_kw):
            if url == "http://93.184.216.34/":
                return _FakeResp(302, location="file:///etc/passwd")
            return _FakeResp(200, text="x")

        monkeypatch.setattr("hermes_cli.agents.echo.tools.seam_safe_fetch.httpx.get", _fake_get)
        _out = safe_fetch_wrapper("http://93.184.216.34/")
        assert "SSRF guard on redirect" in _out
        assert "scheme" in _out


class TestCanonicalFormObfuscation:
    """The classic SSRF bypass: encode the reserved IP in a non-canonical form
    (decimal/hex/octal/leading-zero/IPv4-mapped) so a naive string-blocklist
    misses it. IPv4-mapped is NORMALIZED by ipaddress + unwrapped to its v4;
    decimal/hex/octal/leading-zero are REJECTED by strict ipaddress + refused
    fail-closed by the _IPISH check (no hand-off to a loose platform resolver).
    All paths refuse — the canonical-form bypass is closed either way."""

    def test_refuses_hex_ip_literal(self):
        # 0xA9FEA9FE == 169.254.169.254 (cloud metadata) in hex.
        r = check_url("http://0xA9FEA9FE/")
        assert r is not None, "hex IP literal must normalize + be refused"

    def test_refuses_decimal_ip_literal(self):
        # 2852039166 == 169.254.169.254 in pure decimal.
        r = check_url("http://2852039166/")
        assert r is not None, "decimal IP literal must normalize + be refused"

    def test_refuses_octal_leading_zero(self):
        # 0127.0.0.1 == 127.0.0.1 (leading-zero octal-ish form).
        r = check_url("http://0127.0.0.1/")
        assert r is not None, "leading-zero IP literal must normalize + be refused"

    def test_refuses_ipv4_mapped_ipv6(self):
        # ::ffff:169.254.169.254 — IPv4-mapped IPv6, unwrapped to the v4 + checked.
        r = check_url("http://[::ffff:169.254.169.254]/")
        assert r is not None, "IPv4-mapped IPv6 must unwrap + be refused"

    def test_refuses_hex_loopback(self):
        # 0x7F000001 == 127.0.0.1.
        assert check_url("http://0x7F000001/") is not None


class TestCheckUrlAllow:
    def test_allows_public_ip_literal(self):
        # 93.184.216.34 is example.com's public IP — an IP literal, so NO DNS
        # call (ipaddress.ip_address parses it directly). Public -> None.
        assert check_url("http://93.184.216.34/") is None

    def test_is_url_reserved_none_for_public_literal(self):
        assert is_url_reserved("http://93.184.216.34/path?q=1") is None

    def test_is_url_reserved_empty_and_nonstring(self):
        # No host -> no check -> None (the floor gate treats no-url as no-op).
        assert is_url_reserved("") is None
        assert is_url_reserved(None) is None

    def test_check_url_empty_and_nonstring(self):
        assert check_url("") is None
        assert check_url(None) is None


# ---------------------------------------------------------------------------
# handler re-registration (two-version rule: upstream web_tools.py untouched,
# fetch_url re-registered with safe_fetch_wrapper)
# ---------------------------------------------------------------------------

class TestHandlerReRegistration:
    def test_fetch_url_handler_is_safe_fetch_wrapper(self):
        """The live registry must register fetch_url with the seam-owned
        safe_fetch_wrapper (NOT the upstream web_tools.fetch_url). This is the
        two-version re-registration — upstream web_tools.py is untouched."""
        reg = _build_registry()
        _t = reg.get("fetch_url")
        assert _t is not None
        assert _t.handler is safe_fetch_wrapper, (
            f"fetch_url handler must be safe_fetch_wrapper, got {_t.handler!r}"
        )
        assert _t.handler.__name__ == "safe_fetch_wrapper"
        assert _t._affect_cert_ok is True  # construction cert passed (clean)
        assert _t.execution_sandbox == "none"

    def test_upstream_fetch_url_symbol_not_dispatched(self):
        """The upstream web_tools.fetch_url must NOT be the registered handler
        (it has zero SSRF protection). Belt-and-braces: import the upstream
        symbol + assert it is a DIFFERENT callable than safe_fetch_wrapper."""
        from hermes_cli.agents.echo.tools.web_tools import fetch_url as _upstream
        assert _upstream is not safe_fetch_wrapper
        reg = _build_registry()
        assert reg.get("fetch_url").handler is not _upstream


# ---------------------------------------------------------------------------
# execute_tools floor gate (via a fake registry — routes through the real
# metadata-driven gates; no sandbox/network needed)
# ---------------------------------------------------------------------------

def _clean_url_fetcher(url: str = "") -> str:
    """Affect-clean dummy handler for the floor-gate tests (the fake registry's
    execute() ignores the handler, but the SeamedTool construction cert scans
    it, so it must be clean)."""
    return f"fetched {url}"


class _FakeReg:
    """Minimal fake ToolRegistry mirroring the leak-probe injection pattern.
    execute_tools calls _build_registry() (patched to return this), so the fake
    routes through the REAL metadata-driven floor gate."""

    def __init__(self, tool, call):
        self._tool = tool
        self._call = call
        self.executed = False

    def parse_tool_calls(self, _text):
        return [self._call]

    def has_tool_calls(self, text):
        return True

    def get(self, name):
        return self._tool if name == self._call["name"] else None

    def list_tools(self):
        return [{"name": self._call["name"], "description": "x", "parameters": []}]

    def execute(self, _n, _p):
        self.executed = True
        return {"name": self._call["name"], "success": True,
                "output": "RAN", "error": None}


def _url_tool():
    return SeamedTool(
        name="fetch_url",
        description="test url tool",
        parameters=[{"name": "url", "type": "string", "required": True}],
        handler=_clean_url_fetcher,
        guard_source_policy="none",  # no `path` param -> sanity permits "none"
        requires_affect_cert=True,
        execution_sandbox="none",   # routes to in-process registry.execute
        execution_sandbox_rationale="test: in-process dummy",
    )


def _run(state_response="<ignored>"):
    state = EchoState(config={"memory_dir": "/tmp/echomem_ssrf"},
                      messages=[], user_input="")
    state["response"] = state_response
    return execute_tools(state)


class TestFloorGate:
    def test_refuses_ssrf_url_at_dispatch(self, monkeypatch):
        """A fetch_url call with url=169.254.169.254 is refused by the floor gate
        BEFORE the handler runs (fake.executed stays False). The error names the
        SSRF guard."""
        fake = _FakeReg(_url_tool(), {"name": "fetch_url",
                                       "parameters": {"url": "http://169.254.169.254/latest/meta-data/"}})
        monkeypatch.setattr("hermes_cli.agents.echo.agent._build_registry",
                            lambda _md=None: fake)
        out = _run()
        r0 = out["tool_results"][0]
        assert not r0["success"]
        assert "SSRF" in r0["error"]
        assert "169.254" in r0["error"]
        assert not fake.executed  # floor gate fired BEFORE dispatch

    def test_refuses_canonical_form_at_dispatch(self, monkeypatch):
        """The hex IP literal 0xA9FEA9FE (== 169.254.169.254) is normalized by
        ipaddress + refused at the floor gate — the canonical-form bypass is
        defeated at dispatch."""
        fake = _FakeReg(_url_tool(), {"name": "fetch_url",
                                       "parameters": {"url": "http://0xA9FEA9FE/"}})
        monkeypatch.setattr("hermes_cli.agents.echo.agent._build_registry",
                            lambda _md=None: fake)
        out = _run()
        r0 = out["tool_results"][0]
        assert not r0["success"]
        assert "SSRF" in r0["error"]
        assert not fake.executed

    def test_refuses_loopback_at_dispatch(self, monkeypatch):
        fake = _FakeReg(_url_tool(), {"name": "fetch_url",
                                       "parameters": {"url": "http://127.0.0.1/"}})
        monkeypatch.setattr("hermes_cli.agents.echo.agent._build_registry",
                            lambda _md=None: fake)
        out = _run()
        r0 = out["tool_results"][0]
        assert not r0["success"]
        assert "SSRF" in r0["error"]
        assert not fake.executed

    def test_allows_public_url_at_dispatch(self, monkeypatch):
        """A fetch_url call with a PUBLIC IP-literal url passes the floor gate
        (check_url -> None) + the cert gate + reaches registry.execute (the
        fake's execute sets executed=True). IP literal -> no DNS -> no network."""
        fake = _FakeReg(_url_tool(), {"name": "fetch_url",
                                       "parameters": {"url": "http://93.184.216.34/"}})
        monkeypatch.setattr("hermes_cli.agents.echo.agent._build_registry",
                            lambda _md=None: fake)
        out = _run()
        r0 = out["tool_results"][0]
        assert r0["success"], f"public url should pass the floor gate, got {r0!r}"
        assert fake.executed

    def test_no_url_param_is_noop(self, monkeypatch):
        """A tool call with NO 'url' param must NOT trip the floor gate (the
        param-key scan is a no-op when url is absent). A memory call (no url)
        is unaffected — leak-probe-neutral."""
        tool = SeamedTool(
            name="memory",
            description="mem",
            parameters=[{"name": "action", "type": "string", "required": True}],
            handler=None,
        )
        fake = _FakeReg(tool, {"name": "memory", "parameters": {"action": "search", "query": "x"}})
        monkeypatch.setattr("hermes_cli.agents.echo.agent._build_registry",
                            lambda _md=None: fake)
        out = _run()
        # memory is dispatched via the MemoryTool special-case (handler=None);
        # it must NOT be refused by the SSRF floor gate (no url param).
        r0 = out["tool_results"][0]
        # memory handler=None routes through the memory special-case, not the
        # floor gate. Assert it was not SSRF-refused.
        assert "SSRF" not in (r0.get("error") or "")


# ---------------------------------------------------------------------------
# rate limit (in-process, module-level state; cleared by the autouse fixture)
# ---------------------------------------------------------------------------

class TestRateLimit:
    def test_rate_limit_enforced_after_max(self):
        # 20 requests to the same public IP are allowed; the 21st is refused.
        _url = "http://93.184.216.34/"  # public IP literal -> no DNS
        for _i in range(20):
            assert check_url(_url) is None, f"request {_i+1} should be allowed"
        _r21 = check_url(_url)
        assert _r21 is not None, "21st request must be rate-limited"
        assert "rate limit" in _r21

    def test_rate_limit_per_ip_not_global(self):
        # Two different PUBLIC IPs each get their own 20-request budget. (Both
        # are IP literals -> no DNS -> no network.)
        for _ in range(20):
            assert check_url("http://93.184.216.34/") is None  # example.com, public
        # A different public IP (1.1.1.1, Cloudflare) is NOT rate-limited by the
        # first IP's 20 hits — the limit is per-IP, not global.
        assert check_url("http://1.1.1.1/") is None, "per-IP limit must not be global"

    def test_rate_limit_v4_and_v4_mapped_share_bucket(self):
        """4-lens lens B-finding-e: a public IP given as a v4 literal vs an
        IPv4-mapped-v6 literal must key the SAME rate-limit bucket (the canonical
        unwrapped form). Otherwise alternating forms doubles the limit to 40/60s.
        ::ffff:93.184.216.34 unwraps to 93.184.216.34 — same bucket."""
        _v4 = "http://93.184.216.34/"
        _v4m = "http://[::ffff:93.184.216.34]/"
        for _ in range(20):
            assert check_url(_v4) is None
        # The 21st hit via the v4-MAPPED form must STILL be rate-limited — it is
        # the same canonical key, not a fresh bucket.
        _r = check_url(_v4m)
        assert _r is not None, "v4 + v4-mapped must share the rate-limit bucket"
        assert "rate limit" in _r


# ---------------------------------------------------------------------------
# handler redirect re-validation (safe_fetch_wrapper called directly with a
# MOCKED httpx.get — no network)
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, status_code, location=None, text=""):
        self.status_code = status_code
        self.headers = {"location": location} if location else {}
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class TestHandlerRedirectRevalidation:
    def test_refuses_initial_reserved_url_without_fetch(self, monkeypatch):
        """safe_fetch_wrapper refuses a reserved initial url WITHOUT calling
        httpx.get (the is_url_reserved check short-circuits before the fetch)."""
        _called = []

        def _fake_get(_url, **_kw):
            _called.append(_url)
            return _FakeResp(200, text="should not reach")

        monkeypatch.setattr("hermes_cli.agents.echo.tools.seam_safe_fetch.httpx.get", _fake_get)
        _out = safe_fetch_wrapper("http://169.254.169.254/latest/meta-data/")
        assert "SSRF guard" in _out
        assert "169.254" in _out
        assert _called == [], "httpx.get must NOT be called for a reserved initial url"

    def test_refuses_redirect_to_metadata(self, monkeypatch):
        """A public initial url that 302-redirects to the cloud-metadata IP is
        refused at the redirect hop (the SSRF-by-redirect vector — the floor gate
        cannot see this; only the handler's redirect re-validation catches it)."""
        _calls = []

        def _fake_get(url, **_kw):
            _calls.append(url)
            if url == "http://93.184.216.34/":
                return _FakeResp(302, location="http://169.254.169.254/latest/meta-data/")
            return _FakeResp(200, text="oops")

        monkeypatch.setattr("hermes_cli.agents.echo.tools.seam_safe_fetch.httpx.get", _fake_get)
        _out = safe_fetch_wrapper("http://93.184.216.34/")
        assert "SSRF guard on redirect" in _out
        assert "169.254" in _out
        assert _calls == ["http://93.184.216.34/"], "only the initial fetch happened; redirect was refused"

    def test_follows_safe_redirect(self, monkeypatch):
        """A public initial url that 302-redirects to ANOTHER public IP is
        followed, and the final 200 response text is returned (stripped)."""
        _calls = []

        def _fake_get(url, **_kw):
            _calls.append(url)
            if url == "http://93.184.216.34/":
                return _FakeResp(302, location="http://93.184.216.34/page")
            return _FakeResp(200, text="<html><body>ok content</body></html>")

        monkeypatch.setattr("hermes_cli.agents.echo.tools.seam_safe_fetch.httpx.get", _fake_get)
        _out = safe_fetch_wrapper("http://93.184.216.34/")
        assert _out == "ok content", f"expected stripped redirect target text, got {_out!r}"
        assert _calls == ["http://93.184.216.34/", "http://93.184.216.34/page"]

    def test_redirect_to_loopback_refused(self, monkeypatch):
        def _fake_get(url, **_kw):
            if url == "http://93.184.216.34/":
                return _FakeResp(302, location="http://127.0.0.1/")
            return _FakeResp(200, text="x")

        monkeypatch.setattr("hermes_cli.agents.echo.tools.seam_safe_fetch.httpx.get", _fake_get)
        _out = safe_fetch_wrapper("http://93.184.216.34/")
        assert "SSRF guard on redirect" in _out

    def test_too_many_redirects_refused(self, monkeypatch):
        """A redirect loop that exceeds _MAX_REDIRECTS is refused (bounded loop)."""
        def _fake_get(url, **_kw):
            # Always redirect to the same public url -> infinite loop, bounded
            # by _MAX_REDIRECTS + 1.
            return _FakeResp(302, location="http://93.184.216.34/")

        monkeypatch.setattr("hermes_cli.agents.echo.tools.seam_safe_fetch.httpx.get", _fake_get)
        _out = safe_fetch_wrapper("http://93.184.216.34/")
        assert "too many redirects" in _out


# ---------------------------------------------------------------------------
# v2 pin-and-fetch (closes the v1 DNS-rebinding TOCTOU): the handler resolves
# the host ONCE per hop, pins the resolved IP into the connect URL, and fetches
# by connecting to that IP (httpx never re-resolves). HTTP stays on the module-
# level httpx.get; HTTPS uses httpx.Client with extensions={"sni_hostname": host}.
# These tests call safe_fetch_wrapper DIRECTLY (not via execute_tools) so the
# floor gate's separate resolve is NOT call #1 — the handler's single resolve is
# the only getaddrinfo (the F1 guardrail).
# ---------------------------------------------------------------------------

class TestPinAndFetch:
    def test_pin_uses_resolved_ip_not_hostname(self, monkeypatch):
        """The connect URL pins the resolved IP; the DNS name never reaches httpx
        (if it did, httpx would re-resolve -> the rebind window re-opens)."""
        import socket as _s
        monkeypatch.setattr(
            "hermes_cli.agents.echo.tools.seam_safe_fetch.socket.getaddrinfo",
            lambda _h, _p: [(_s.AF_INET, _s.SOCK_STREAM, 0, "", ("93.184.216.34", 0))])
        _got = []
        monkeypatch.setattr("hermes_cli.agents.echo.tools.seam_safe_fetch.httpx.get",
            lambda url, **_kw: (_got.append(url) or _FakeResp(200, text="ok")))
        safe_fetch_wrapper("http://rebind.example.com/")
        assert _got, "httpx.get must be called"
        assert "93.184.216.34" in _got[0]
        assert "rebind.example.com" not in _got[0], f"name leaked into connect URL: {_got[0]!r}"

    def test_dns_rebinding_toctou_closed(self, monkeypatch):
        """F1 guardrail: getaddrinfo returns PUBLIC on call #1, PRIVATE on call #2.
        v2 must call getaddrinfo EXACTLY ONCE (the single resolve supplies verdict
        + pin) and fetch the PUBLIC IP — the private rebind is never reached."""
        import socket as _s
        _gai = []
        _seq = ["93.184.216.34", "169.254.169.254"]  # call#1 public, call#2 private

        def _fake_gai(_h, _p):
            _ip = _seq[min(len(_gai), len(_seq) - 1)]
            _gai.append(_h)
            return [(_s.AF_INET, _s.SOCK_STREAM, 0, "", (_ip, 0))]

        monkeypatch.setattr("hermes_cli.agents.echo.tools.seam_safe_fetch.socket.getaddrinfo", _fake_gai)
        _got = []
        monkeypatch.setattr("hermes_cli.agents.echo.tools.seam_safe_fetch.httpx.get",
            lambda url, **_kw: (_got.append(url) or _FakeResp(200, text="ok")))
        _out = safe_fetch_wrapper("http://rebind.example.com/")
        assert len(_gai) == 1, f"getaddrinfo must be called exactly once (F1), got {len(_gai)}"
        assert _got, "httpx.get must be called"
        assert "93.184.216.34" in _got[0], f"must pin the PUBLIC (call#1) IP, got {_got[0]!r}"
        assert "169.254" not in _got[0], "the private rebind IP must NOT reach the fetch"
        assert "SSRF" not in _out

    def test_pin_refuses_if_handler_resolves_private(self, monkeypatch):
        """The handler's OWN reserved check refuses when the resolved IP is private
        — defense-in-depth with the floor gate (a direct call still gets refused)."""
        import socket as _s
        monkeypatch.setattr(
            "hermes_cli.agents.echo.tools.seam_safe_fetch.socket.getaddrinfo",
            lambda _h, _p: [(_s.AF_INET, _s.SOCK_STREAM, 0, "", ("169.254.169.254", 0))])
        _got = []
        monkeypatch.setattr("hermes_cli.agents.echo.tools.seam_safe_fetch.httpx.get",
            lambda url, **_kw: (_got.append(url) or _FakeResp(200, text="oops")))
        _out = safe_fetch_wrapper("http://rebind.example.com/")
        assert "SSRF guard" in _out
        assert "169.254" in _out
        assert _got == [], "httpx.get must NOT be called when the host resolves reserved"

    def test_pin_preserves_port_and_host_header(self, monkeypatch):
        """F3: the connect URL keeps the explicit port + path + query, and the Host
        header is the original DNS vhost netloc (host:port)."""
        import socket as _s
        monkeypatch.setattr(
            "hermes_cli.agents.echo.tools.seam_safe_fetch.socket.getaddrinfo",
            lambda _h, _p: [(_s.AF_INET, _s.SOCK_STREAM, 0, "", ("93.184.216.34", 0))])
        _got = {}
        def _fake_get(url, **_kw):
            _got["url"] = url
            _got["headers"] = _kw.get("headers")
            return _FakeResp(200, text="ok")
        monkeypatch.setattr("hermes_cli.agents.echo.tools.seam_safe_fetch.httpx.get", _fake_get)
        safe_fetch_wrapper("http://rebind.example.com:8080/path?q=1")
        assert _got.get("url") == "http://93.184.216.34:8080/path?q=1", \
            f"port+path+query must be preserved, got {_got.get('url')!r}"
        _h = _got.get("headers") or {}
        assert _h.get("Host") == "rebind.example.com:8080", \
            f"Host must be the vhost netloc (F3), got {_h!r}"

    def test_pin_ipv6_bracketed(self, monkeypatch):
        """A resolved public IPv6 is bracketed in the connect URL netloc."""
        import socket as _s
        _v6 = "2606:2800:220:1:248:1893:25c8:1946"  # global unicast (not reserved)
        monkeypatch.setattr(
            "hermes_cli.agents.echo.tools.seam_safe_fetch.socket.getaddrinfo",
            lambda _h, _p: [(_s.AF_INET6, _s.SOCK_STREAM, 0, "", (_v6, 0, 0, 0))])
        _got = []
        monkeypatch.setattr("hermes_cli.agents.echo.tools.seam_safe_fetch.httpx.get",
            lambda url, **_kw: (_got.append(url) or _FakeResp(200, text="ok")))
        safe_fetch_wrapper("http://rebind.example.com/path")
        assert _got, "httpx.get must be called"
        assert _got[0].startswith("http://[2606:2800:220:1:248:1893:25c8:1946]"), \
            f"IPv6 must be bracketed in the connect URL, got {_got[0]!r}"

    def test_pin_carries_vhost_across_relative_redirect(self, monkeypatch):
        """F2: a same-server RELATIVE redirect keeps the original DNS vhost as the
        Host header on the re-fetch (a vhosted server sees the right Host)."""
        import socket as _s
        monkeypatch.setattr(
            "hermes_cli.agents.echo.tools.seam_safe_fetch.socket.getaddrinfo",
            lambda _h, _p: [(_s.AF_INET, _s.SOCK_STREAM, 0, "", ("93.184.216.34", 0))])
        _hosts = []

        def _fake_get(url, **_kw):
            _h = _kw.get("headers")
            _hosts.append(_h.get("Host") if _h else None)
            if url == "http://93.184.216.34/":
                return _FakeResp(302, location="/page2")
            return _FakeResp(200, text="ok")

        monkeypatch.setattr("hermes_cli.agents.echo.tools.seam_safe_fetch.httpx.get", _fake_get)
        _out = safe_fetch_wrapper("http://rebind.example.com/")
        assert _out == "ok", f"relative redirect should be followed, got {_out!r}"
        assert _hosts == ["rebind.example.com", "rebind.example.com"], \
            f"vhost Host must carry across the relative redirect (F2), got {_hosts}"

    def test_https_pin_uses_sni_hostname(self, monkeypatch):
        """HTTPS pins the IP in the connect URL + sets sni_hostname to the real
        vhost (so SNI + cert-verify use the vhost, connect to the IP)."""
        import socket as _s
        monkeypatch.setattr(
            "hermes_cli.agents.echo.tools.seam_safe_fetch.socket.getaddrinfo",
            lambda _h, _p: [(_s.AF_INET, _s.SOCK_STREAM, 0, "", ("93.184.216.34", 0))])
        _got = {}

        class _FakeCli:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return None

            def get(self, url, **_kw):
                _got["url"] = url
                _got["extensions"] = _kw.get("extensions")
                _got["headers"] = _kw.get("headers")
                return _FakeResp(200, text="ok")

        monkeypatch.setattr("hermes_cli.agents.echo.tools.seam_safe_fetch.httpx.Client", _FakeCli)
        _out = safe_fetch_wrapper("https://rebind.example.com/path?q=1")
        assert "SSRF" not in _out, f"https pin should succeed, got {_out!r}"
        assert _got.get("url") == "https://93.184.216.34/path?q=1", \
            f"https connect URL must pin the IP, got {_got.get('url')!r}"
        assert _got.get("extensions") == {"sni_hostname": "rebind.example.com"}, \
            f"HTTPS must set sni_hostname to the real vhost, got {_got.get('extensions')!r}"
        assert (_got.get("headers") or {}).get("Host") == "rebind.example.com", \
            f"Host must be the vhost, got {_got.get('headers')!r}"


# ---------------------------------------------------------------------------
# integration tier (needs network + DNS; skipped by apply_seam.sh's
# -m 'not integration'; run in a separate live gate)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIntegrationLiveFetch:
    def test_live_fetch_public_url(self):
        """Live fetch of a public url through the real handler (DNS + HTTP).
        Skip when offline (no network / DNS unreachable)."""
        try:
            import socket
            socket.getaddrinfo("example.com", None)
        except Exception:
            pytest.skip("no network / DNS unreachable")
        _out = safe_fetch_wrapper("http://example.com/")
        # Either real content OR a graceful network error string (NOT an SSRF
        # refusal — example.com is public).
        assert "SSRF guard" not in _out, f"public url was SSRF-refused: {_out!r}"

    def test_live_metadata_refused(self):
        """Live: the cloud-metadata IP is refused even when reachable (the
        handler's is_url_reserved check fires before any fetch)."""
        _out = safe_fetch_wrapper("http://169.254.169.254/latest/meta-data/")
        assert "SSRF guard" in _out

    def test_live_https_pinned_fetch(self):
        """Live e2e: a real HTTPS url is fetched via pin-and-fetch (resolve host
        -> pin IP -> connect to the IP with sni_hostname=vhost so SNI + cert-verify
        use the real host). The pin must NOT break certificate verification: a
        successful fetch returns content; an SSL/cert error would mean the pin
        broke cert-verify (the e2e backstop for the sni_hostname claim)."""
        try:
            import socket
            socket.getaddrinfo("example.com", None)
        except Exception:
            pytest.skip("no network / DNS unreachable")
        _out = safe_fetch_wrapper("https://example.com/")
        assert "SSRF guard" not in _out, f"public https url was SSRF-refused: {_out!r}"
        # A graceful network error (timeout/unreachable) is tolerated; an SSL/cert
        # error is NOT — it would mean the pinned-IP connect broke cert-verify.
        _low = _out.lower()
        assert "ssl" not in _low and "cert" not in _low, \
            f"HTTPS pin must keep cert-verify against the real vhost; got {_out!r}"