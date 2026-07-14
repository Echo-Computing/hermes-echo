"""OSIRIS-ssrf seam guard (2026-07-06 red-team) — SSRF defense for fetch_url.

Replaces the upstream ``web_tools.fetch_url`` handler (ZERO SSRF protection:
``httpx.get(url, follow_redirects=True)`` resolved + fetched ANY url including
169.254.169.254 cloud-metadata, 127.0.0.1, ::1, fc00::/7) with a seam-owned
``safe_fetch_wrapper`` plus an in-process floor-gate entry ``check_url``.

Two layers (landed in one commit):
  1. FLOOR GATE (in-process, ``agent.execute_tools``): any tool call with a
     ``url`` param is run through ``check_url`` — refuses if the url's host
     resolves to a reserved/private range OR exceeds the per-IP rate limit.
     Sits after the guard-source gate, before the cert gate (a param-shape
     sibling of the guard-source gate; inspects ``url`` where guard-source
     inspects ``path``). Catches the obvious SSRF at dispatch, before the
     handler runs.
  2. HANDLER (in-process, ``execution_sandbox="none"``): ``safe_fetch_wrapper``
     re-checks the initial url AND every redirect hop (manual redirect loop,
     ``follow_redirects=False``) — catches SSRF-by-redirect (a public url that
     302-> 169.254.169.254) which the floor gate cannot see (it inspects only
     the LLM-supplied initial url).

Defenses:
  - reserved-range block: explicit IPv4/IPv6 rejected-network list (loopback,
    RFC1918, CGNAT 100.64/10, link-local incl 169.254.169.254 cloud-metadata,
    ULA fc00::/7, multicast, reserved, TEST-NET, benchmarking) + stdlib
    ``is_loopback/is_link_local/is_multicast/is_unspecified/is_private/
    is_reserved`` as belt-and-braces. The explicit list is the version-
    independent primary (CGNAT only folded into ``is_private`` in Py3.13).
  - canonical-form obfuscation defeat: IPv4-mapped IPv6 is normalized by
    ``ipaddress.ip_address()``; decimal/hex/octal/leading-zero IP literals
    (``http://0xA9FEA9FE``, ``http://2852039166``, ``http://0127.0.0.1``) are
    REJECTED by strict ipaddress and refused fail-closed by the ``_IPISH`` check
    (an IP-ish hostname ipaddress rejected is NOT handed to a loose resolver like
    glibc getaddrinfo, which would interpret it platform-dependently). So no
    obfuscated reserved-IP literal reaches the fetch.
  - DNS resolution: a hostname is resolved via ``socket.getaddrinfo`` and EVERY
    resolved IP is checked; any reserved -> refuse. Fail-closed on DNS error.
  - redirect re-validation: ``follow_redirects=False`` + a bounded manual loop
    (``_MAX_REDIRECTS``=5 hops) re-runs the reserved check + scheme allowlist on
    every Location header (relative redirects resolved via ``urljoin`` against the
    current url).
  - per-IP rate limit: ``_RATE_LIMIT_MAX`` requests per IP per ``_RATE_LIMIT_WINDOW``
    (in-process, ``check_url`` only — the rate limit lives in the in-process
    floor gate where it survives across fetch_url calls for the agent process
    lifetime).
  - NAME blocklist: localhost / metadata.google.internal / metadata.aws /
    metadata / kernel — defense-in-depth before DNS resolution.

v2 (this module) closes the DNS-rebinding TOCTOU: the handler resolves the host
ONCE per fetch/redirect-hop (the single ``_resolve_host`` supplies BOTH the
reserved verdict AND the pin — F1 invariant; a second resolve would re-open the
window), pins the resolved IP into the connect URL, and fetches with
``Host: <original vhost netloc>`` (HTTPS adds ``extensions={"sni_hostname": host}``
so httpcore uses the real vhost as TLS ``server_hostname`` — SNI + cert-verify
against the vhost, not the pinned IP; ``httpcore/_sync/connection.py:151``).
httpx connects to the IP literal and does NOT re-resolve, so a name rebinding
public->private between check and connect cannot redirect the fetch. v1's floor
(``check_url`` reserved + rate-limit gate, redirect re-validation) is unchanged.

Residuals (documented, v2): F4 — the per-IP rate limit is keyed by the FLOOR
GATE's resolved IP, so under a rebind the handler's pinned IP can differ and is
not separately rate-limited (secondary defense; unchanged from v1). F7 — httpx
defaults ``trust_env=True`` (honors ``HTTP(S)_PROXY``); a compliant proxy still
connects to the pinned IP (pin holds), but a malicious proxy could re-resolve the
``Host`` and rebind. Low-risk for the local-LLM single-machine profile; documented.

UPSTREAM ``web_tools.py`` is LEFT UNTOUCHED (two-version rule): the seam re-
registers ``fetch_url`` with ``handler=safe_fetch_wrapper`` in ``_build_registry``;
the upstream ``fetch_url`` symbol is no longer dispatched. No cloud LLM, no
attribution (the public "OSIRIS" repo is an unrelated Solana token-pump ad —
never credited; this is a seam-owned SSRF guard, not a port of that repo).

Public-safe: stdlib ``ipaddress``/``socket``/``re``/``time`` + ``httpx`` only, no
private substrate import, no private substrate read (cert-clean). Import-pure
(module-level = imports + defs + the rejected-network constants; no I/O on
import). ``fetch_url`` runs ``execution_sandbox="none"`` in-process (needs
network; the protected-store path gates still apply).
"""
from __future__ import annotations

import ipaddress
import re
import socket
import time
from typing import List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import httpx

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
_MAX_REDIRECTS = 5
_FETCH_TIMEOUT = 30
_MAX_BODY_CHARS = 8000
_RATE_LIMIT_MAX = 20          # max requests per IP within the window
_RATE_LIMIT_WINDOW = 60.0     # seconds

# Hostnames blocked before DNS resolution (defense-in-depth; the IP-reserved
# check is the primary stop). Lowercased; matched against the parsed hostname.
_NAME_BLOCKLIST = frozenset({
    "localhost",
    "metadata.google.internal",
    "metadata.aws",
    "metadata",
    "kernel",
})

# Explicitly rejected IPv4 ranges. ipaddress.ip_address(...).is_private covers
# most of these in Py3.13, but CGNAT 100.64/10 was only folded into is_private in
# 3.13, and TEST-NET / benchmarking / IETF-assignment ranges are not always
# flagged — so this explicit list is the version-independent primary check,
# with the stdlib properties as belt-and-braces in _is_reserved_ip.
_REJECTED_V4 = [
    ipaddress.ip_network("0.0.0.0/8"),        # "this network"
    ipaddress.ip_network("10.0.0.0/8"),        # RFC1918
    ipaddress.ip_network("100.64.0.0/10"),    # CGNAT (RFC6598)
    ipaddress.ip_network("127.0.0.0/8"),       # loopback
    ipaddress.ip_network("169.254.0.0/16"),    # link-local (incl cloud metadata)
    ipaddress.ip_network("172.16.0.0/12"),     # RFC1918
    ipaddress.ip_network("192.0.0.0/24"),      # IETF protocol assignments
    ipaddress.ip_network("192.0.2.0/24"),     # TEST-NET-1
    ipaddress.ip_network("192.168.0.0/16"),   # RFC1918
    ipaddress.ip_network("198.18.0.0/15"),    # benchmarking
    ipaddress.ip_network("198.51.100.0/24"),  # TEST-NET-2
    ipaddress.ip_network("203.0.113.0/24"),   # TEST-NET-3
    ipaddress.ip_network("192.88.99.0/24"),    # deprecated 6to4 anycast (RFC 7526)
    ipaddress.ip_network("224.0.0.0/4"),      # multicast
    ipaddress.ip_network("240.0.0.0/4"),      # reserved
]

_REJECTED_V6 = [
    ipaddress.ip_network("::/128"),           # unspecified
    ipaddress.ip_network("::1/128"),           # loopback
    ipaddress.ip_network("fc00::/7"),          # ULA private
    ipaddress.ip_network("fe80::/10"),         # link-local
    ipaddress.ip_network("ff00::/8"),          # multicast
    ipaddress.ip_network("2001:db8::/32"),    # documentation prefix
    ipaddress.ip_network("::ffff:0:0/96"),     # IPv4-mapped (unwrap + recheck v4)
]

# In-process per-IP rate-limit state (floor gate only). The rate limit lives
# HERE, in-process, where it survives across fetch_url calls for the agent
# process lifetime (fresh per `echo` invocation — documented v1 scope). Keyed
# by the resolved-IP string (or the IP literal).
_RATE_STATE: dict = {}

# An "IP-ish" hostname: only chars valid in an IP literal (digits, dots, colons,
# hex digits, 'x' for the 0x hex prefix). Used to detect AMBIGUOUS IP-literal
# forms that strict ``ipaddress.ip_address()`` REJECTS but a loose resolver
# (glibc ``getaddrinfo``) would interpret — e.g. "0127.0.0.1" (leading-zero:
# ipaddress rejects; glibc parses octal 0127=87 -> 87.0.0.1 on Linux, but other
# platforms differ). Parser disagreement is an SSRF vector, so an IP-ish host
# that ipaddress rejected is REFUSED fail-closed rather than handed to
# getaddrinfo where the interpretation is platform-dependent. Real DNS names
# almost always contain a letter outside [a-f] (g-z), so they do NOT match this
# pattern and route to DNS normally. (Rare all-hexish single-label names like
# "cafe" would false-positive — fail-closed is the right security posture.)
_IPISH = re.compile(r"^[0-9a-fA-F.:x]+$")

# Allowed URL schemes. A non-http(s) scheme (file:///, data:, gopher://, ftp://,
# javascript:) is refused fail-closed by ``_scheme_refused`` INDEPENDENT of httpx's
# scheme handling — defense-in-depth so a hostless / local-file target cannot be
# reached via a redirect Location even if a future httpx gains a scheme handler
# (4-lens lens B-finding-b). Empty scheme (schemeless url) is allowed (it has no
# host to resolve -> _resolve_host returns (None, []) -> no-op).
_ALLOWED_SCHEMES = frozenset({"http", "https", ""})


def _scheme_refused(url: str) -> Optional[str]:
    """Return a reason string if ``url``'s scheme is not http/https (or empty),
    else None."""
    try:
        _sch = urlparse(url).scheme.lower()
    except Exception:
        return f"unparseable url {url!r}"
    if _sch not in _ALLOWED_SCHEMES:
        return (f"scheme {_sch!r} not allowed (http/https only; refused to avoid "
                f"a hostless/local-file target via redirect)")
    return None


def _is_reserved_ip(ip: ipaddress._BaseAddress) -> Optional[str]:
    """Return a short reason string if ``ip`` is a reserved/private/rejected
    range, else None. Canonical-form obfuscations are defeated BEFORE this check
    by ``_resolve_host``: IPv4-mapped IPv6 (::ffff:a.b.c.d) is normalized by
    ``ipaddress.ip_address()`` to its v4 (unwrapped here too as belt-and-braces);
    decimal/hex/octal/leading-zero forms (http://0xA9FEA9FE, http://2852039166,
    http://0127.0.0.1) are REJECTED by strict ipaddress and refused fail-closed by
    the ``_IPISH`` check in ``_resolve_host`` (closing the parser-disagreement gap
    with a loose resolver like glibc getaddrinfo, which would interpret them).
    So no obfuscated reserved-IP literal reaches this check un-normalized.
    """
    # Unwrap IPv4-mapped IPv6 (::ffff:a.b.c.d) -> check the embedded v4.
    if isinstance(ip, ipaddress.IPv6Address):
        _mapped = ip.ipv4_mapped
        if _mapped is not None:
            ip = _mapped
    _nets = _REJECTED_V4 if isinstance(ip, ipaddress.IPv4Address) else _REJECTED_V6
    for _n in _nets:
        if ip in _n:
            return f"{ip} in reserved range {_n}"
    # Belt-and-braces: stdlib properties (covers edge cases the explicit list
    # might miss on a given Python version; all exist across 3.12/3.13).
    if (ip.is_loopback or ip.is_link_local or ip.is_multicast
            or ip.is_unspecified or ip.is_private or ip.is_reserved):
        return f"{ip} is_loopback/link-local/private/reserved"
    return None


def _resolve_host(host: Optional[str]) -> Tuple[Optional[str], List[str]]:
    """Resolve ``host`` (a parsed URL hostname) to ``(reserved_reason, ips)``.

      - host None/empty -> ``(None, [])`` (no hostname to check).
      - NAME blocklist hit -> ``(reason, [])``.
      - host is an IP literal -> ``ipaddress.ip_address()`` accepts it (only the
        canonical forms: dotted-quad, hex groups, IPv4-mapped v6). Decimal/hex/
        octal/leading-zero forms are REJECTED here and refused fail-closed by
        ``_IPISH`` below. -> ``_is_reserved_ip``. ``(reason, [])`` if reserved,
        else ``(None, [str(canonical_key)])`` (the rate-limit key is the
        canonical unwrapped form: IPv4-mapped IPv6 unwraps to its v4 so a public
        IP given as v4 vs v4-mapped-v6 keys the SAME bucket).
      - host is a DNS name -> ``socket.getaddrinfo`` -> check EACH resolved IP;
        if ANY is reserved -> ``(reason, [])``. Else ``(None, [ip_strings])``.
        Fail-closed on DNS error (an unresolvable name must not fall through to
        a fetch that might resolve differently inside the handler).

    DNS-rebinding TOCTOU (a name resolving public here, private at connect time)
    is CLOSED in v2: ``safe_fetch_wrapper`` calls this ONCE per fetch/redirect-hop,
    pins the returned IP into the connect URL, and fetches with ``Host: <vhost>``
    so httpx does NOT re-resolve. This function is the single DNS chokepoint — one
    call per fetch/redirect-hop supplies both the reserved verdict and the pin
    (F1 invariant; do not pair it with a separate ``is_url_reserved`` call, which
    would be a second ``getaddrinfo`` and re-open the window).
    """
    if not host:
        return None, []
    _h = host.lower().strip("[]")
    if _h in _NAME_BLOCKLIST:
        return f"hostname {_h!r} in name blocklist", []
    # IP literal? ipaddress.ip_address accepts ONLY canonical forms (dotted-quad,
    # hex groups, IPv4-mapped v6). Decimal/hex/octal/leading-zero are REJECTED ->
    # fall to the _IPISH fail-closed refusal below.
    try:
        _ip = ipaddress.ip_address(_h)
    except ValueError:
        _ip = None
    if _ip is not None:
        _r = _is_reserved_ip(_ip)
        if _r is not None:
            return _r, []
        # Canonical rate-limit key: unwrap IPv4-mapped IPv6 to its v4 so a public
        # IP given as a v4 literal vs an IPv4-mapped-v6 literal keys the SAME
        # _RATE_STATE bucket (4-lens lens B-finding-e: otherwise alternating
        # forms doubles the effective limit to 40/60s).
        _key = _ip
        if isinstance(_ip, ipaddress.IPv6Address) and _ip.ipv4_mapped is not None:
            _key = _ip.ipv4_mapped
        return None, [str(_key)]
    # Ambiguous IP-literal form: strict ipaddress REJECTED this host, but it
    # looks like an IP literal (only IP-ish chars). A loose resolver like glibc
    # getaddrinfo would interpret it (leading-zero -> octal, etc.) — and the
    # interpretation is platform-dependent. Parser disagreement is an SSRF
    # vector, so refuse fail-closed rather than hand it to getaddrinfo.
    if _IPISH.match(_h):
        return (f"ambiguous IP-literal form {_h!r}: strict ipaddress rejected it; "
                f"refusing rather than hand to a loose platform-dependent resolver", [])
    # DNS name -> resolve + check every IP.
    try:
        _infos = socket.getaddrinfo(_h, None)
    except socket.gaierror as _e:
        return f"DNS resolution failed for {_h!r}: {_e}", []
    _ips: List[str] = []
    for _fam, _stype, _proto, _canon, _sockaddr in _infos:
        _ipstr = _sockaddr[0]
        try:
            _ipobj = ipaddress.ip_address(_ipstr)
        except ValueError:
            continue
        _r = _is_reserved_ip(_ipobj)
        if _r is not None:
            return f"{_h} resolves to {_ipstr}: {_r}", []
        if _ipstr not in _ips:
            _ips.append(_ipstr)
    if not _ips:
        return f"DNS returned no usable IPs for {_h!r}", []
    return None, _ips


def is_url_reserved(url: str) -> Optional[str]:
    """Stateless reserved-range check on ``url`` (the LLM-supplied initial url OR
    a redirect Location). Returns a reason string if the url's host resolves to a
    reserved/private range, else None. Used by:
      - ``safe_fetch_wrapper`` (the in-process handler): re-checks the initial url
        + EVERY redirect hop (SSRF-by-redirect vector).
      - ``check_url`` (the in-process floor gate): the reserved half (the rate-
        limit half is added by ``check_url``, which holds the persistent state).
    """
    if not isinstance(url, str) or not url:
        return None
    _sch_r = _scheme_refused(url)
    if _sch_r is not None:
        return _sch_r
    try:
        _host = urlparse(url).hostname
    except Exception:
        return f"unparseable url {url!r}"
    _reason, _ = _resolve_host(_host)
    return _reason


def _rate_limit_check(ip_key: str) -> Optional[str]:
    """In-process per-IP rate limit: refuse if more than ``_RATE_LIMIT_MAX``
    requests to the same IP within ``_RATE_LIMIT_WINDOW`` seconds. State lives in
    ``_RATE_STATE`` (module-level, persists for the agent process lifetime; fresh
    per ``echo`` invocation — a documented v1 scope). Called ONLY by
    ``check_url`` (the in-process floor gate); the handler re-checks the url
    but the persistent cross-call limit is enforced by the floor gate."""
    _now = time.time()
    _hist = [t for t in _RATE_STATE.get(ip_key, []) if _now - t < _RATE_LIMIT_WINDOW]
    if len(_hist) >= _RATE_LIMIT_MAX:
        return (f"rate limit: {len(_hist)} requests to {ip_key} within "
                f"{int(_RATE_LIMIT_WINDOW)}s (max {_RATE_LIMIT_MAX})")
    _hist.append(_now)
    _RATE_STATE[ip_key] = _hist
    return None


def check_url(url: str) -> Optional[str]:
    """The in-process FLOOR-GATE entry (called from ``agent.execute_tools`` for
    any tool call with a ``url`` param). Returns a reason string if the url is
    refused (reserved range OR rate-limited), else None. Composes
    ``is_url_reserved`` (the reserved check, via ``_resolve_host``) with
    ``_rate_limit_check`` (the per-IP limit, keyed by the resolved IP — or the
    IP literal). This is the seam-level chokepoint that catches the obvious SSRF
    at dispatch; ``safe_fetch_wrapper`` is the runtime guard that catches
    redirect-based SSRF the floor gate cannot see (the floor gate inspects only
    the LLM-supplied initial url)."""
    if not isinstance(url, str) or not url:
        return None
    _sch_r = _scheme_refused(url)
    if _sch_r is not None:
        return _sch_r
    try:
        _host = urlparse(url).hostname
    except Exception:
        return f"unparseable url {url!r}"
    _reason, _ips = _resolve_host(_host)
    if _reason is not None:
        return _reason
    if not _ips:
        return None  # no host / nothing to rate-limit
    return _rate_limit_check(_ips[0])


def _strip_html(content: str) -> str:
    """Strip HTML to visible text (mirrors upstream web_tools.fetch_url)."""
    _t = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL | re.IGNORECASE)
    _t = re.sub(r"<style[^>]*>.*?</style>", "", _t, flags=re.DOTALL | re.IGNORECASE)
    _t = re.sub(r"<[^>]+>", " ", _t)
    _t = re.sub(r"\s+", " ", _t).strip()
    return _t


def _connect_url(parsed, ip: str) -> str:
    """Rebuild the URL with the pinned IP as the netloc (IPv6 bracketed), keeping
    scheme + port + path + query. httpx connects to the IP literal and does NOT
    re-resolve — this is what closes the DNS-rebinding TOCTOU (the connect target
    is the already-reserved-checked IP, not a fresh resolve that could rebind
    private between the check and the connect)."""
    _h = f"[{ip}]" if ":" in ip else ip
    try:
        _port = parsed.port
    except ValueError:
        _port = None  # malformed port — let httpx refuse it downstream
    _netloc = f"{_h}:{_port}" if _port else _h
    return parsed._replace(netloc=_netloc).geturl()


def _pin_and_fetch(parsed, vhost_netloc: str):
    """Resolve ``parsed.hostname`` ONCE and fetch by connecting to the pinned IP.

    F1 single-resolve invariant: this ONE ``_resolve_host`` call supplies BOTH the
    reserved verdict AND the pin. Do NOT add a separate ``is_url_reserved`` call
    — that would be a second ``getaddrinfo`` and re-open the TOCTOU (the reserved
    check + pin must come from the SAME resolve, so a rebind between two resolves
    cannot pin a private IP that escaped the check).

    Returns ``(response, reserved_reason)``. ``reserved_reason`` is non-None iff
    the host resolves to a reserved range -> the caller refuses (the handler's own
    reserved check; defense-in-depth with the floor gate). On success ``response``
    is the httpx.Response (possibly a 3xx the caller redirects on) and
    ``reserved_reason`` is None. Network errors raise (caught by the caller).

    HTTP uses the module-level ``httpx.get`` (no ``extensions`` param on the
    module-level API — ``httpx/_api.py:174``); HTTPS uses ``httpx.Client.get``
    with ``extensions={"sni_hostname": host}`` so httpcore uses the real vhost as
    TLS ``server_hostname`` (``httpcore/_sync/connection.py:151``) — SNI + cert-
    verify against the vhost, connect to the pinned IP. The ``Host`` header is the
    original DNS vhost netloc (``host`` or ``host:port`` — F3)."""
    _host = parsed.hostname
    _scheme = parsed.scheme.lower()
    _reason, _ips = _resolve_host(_host)
    if _reason is not None:
        return None, _reason
    if not _ips:
        return None, "no resolved IPs (unexpected for a non-empty host)"
    _conn = _connect_url(parsed, _ips[0])
    _headers = {"Host": vhost_netloc}
    if _scheme == "https":
        with httpx.Client() as _cli:
            _resp = _cli.get(_conn, extensions={"sni_hostname": _host},
                             headers=_headers, timeout=_FETCH_TIMEOUT,
                             follow_redirects=False)
    else:
        _resp = httpx.get(_conn, headers=_headers, timeout=_FETCH_TIMEOUT,
                          follow_redirects=False)
    return _resp, None


def safe_fetch_wrapper(url: str) -> str:
    """The ``fetch_url`` handler (registered in ``_build_registry`` as the handler
    for the ``fetch_url`` tool, replacing the upstream ``web_tools.fetch_url``
    which had ZERO SSRF protection). v2 pin-and-fetch: per fetch/redirect-hop it
    resolves the host ONCE, pins the resolved IP into the connect URL, and fetches
    by connecting to that IP with ``Host: <original vhost netloc>`` (+ SNI for
    HTTPS) so httpx does NOT re-resolve — closing the v1 DNS-rebinding TOCTOU.
    Redirects use ``follow_redirects=False`` + a bounded manual loop, re-pinning
    per hop (each hop's single resolve supplies its own reserved check). Returns
    extracted text (first ``_MAX_BODY_CHARS`` chars) or an error string, matching
    the upstream ``fetch_url`` return shape so the tool-result channel + leak-probe
    assertions are unchanged.

    The floor gate (``agent.execute_tools``, in-process) already refused a
    reserved initial url; this handler re-checks (defense-in-depth: a direct call,
    or a registry rebuilt without the floor gate, still gets the reserved check)
    AND owns the redirect re-validation + the pin the floor gate cannot see.

    v2 closes the v1 DNS-rebinding residual: a name resolving public at the
    floor-gate/handler check + private at httpx's connect. The single per-hop
    ``_resolve_host`` supplies both verdict + pin; httpx connects to the pinned IP
    literal and does not re-resolve. See the module docstring for the F4 (rate-
    limit key) + F7 (proxy trust_env) residuals.
    """
    _cur = url
    _on_redirect = False
    for _ in range(_MAX_REDIRECTS + 1):
        _p = urlparse(_cur)
        _sch_r = _scheme_refused(_cur)
        if _sch_r is not None:
            return (f"fetch_url refused (SSRF guard on redirect): {_sch_r}"
                    if _on_redirect else f"fetch_url refused (SSRF guard): {_sch_r}")
        # The DNS vhost netloc (host:port) — _cur stays the LOGICAL url (never the
        # connect IP), so a same-server relative redirect keeps the right Host
        # naturally (F2); an absolute cross-host Location re-derives it next hop.
        _vhost = _p.netloc
        if not _p.hostname:
            # Hostless (schemeless/relative) url: no DNS-rebind surface; fetch the
            # url directly (v1 pass-through).
            try:
                _resp = httpx.get(_cur, timeout=_FETCH_TIMEOUT,
                                  follow_redirects=False)
            except Exception as _e:
                return f"Failed to fetch URL — network error: {_e}"
        else:
            try:
                _resp, _reason = _pin_and_fetch(_p, _vhost)
            except Exception as _e:
                return f"Failed to fetch URL — network error: {_e}"
            if _reason is not None:
                return (f"fetch_url refused (SSRF guard on redirect): {_reason}"
                        if _on_redirect else f"fetch_url refused (SSRF guard): {_reason}")
        if _resp.status_code in (301, 302, 303, 307, 308):
            _loc = _resp.headers.get("location", "").strip()
            if not _loc:
                return "Failed to fetch URL — redirect with no Location header."
            _cur = urljoin(_cur, _loc)
            _on_redirect = True
            continue
        try:
            _resp.raise_for_status()
        except Exception as _e:
            return f"Failed to fetch URL — HTTP {_resp.status_code}: {_e}"
        _text = _strip_html(_resp.text)
        return _text[:_MAX_BODY_CHARS] if _text else "(empty page)"
    return f"fetch_url refused — too many redirects (>{_MAX_REDIRECTS})."