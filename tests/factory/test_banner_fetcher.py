"""Hermetic test for banner.default_fetcher's HTTP contract.

Wikimedia's User-Agent policy 403s requests that send no UA or a generic
library UA (https://meta.wikimedia.org/wiki/User-Agent_policy). The real
Commons fetch therefore fails unless we send an identifying header. This
test pins that header without touching the network.
"""

import httpx

import janus
from janus.factory.banner import default_fetcher


class _Resp:
    content = b"ok"

    def raise_for_status(self):
        return None


def test_default_fetcher_sends_identifying_user_agent(monkeypatch):
    seen: dict = {}

    def fake_get(url, **kwargs):
        seen["url"] = url
        seen["headers"] = kwargs.get("headers")
        return _Resp()

    monkeypatch.setattr(httpx, "get", fake_get)

    body = default_fetcher("https://commons.wikimedia.org/w/api.php", {"a": "b"})

    assert body == b"ok"
    headers = seen["headers"] or {}
    ua = headers.get("User-Agent", "")
    assert "Janus" in ua                 # identifies the tool
    assert janus.__version__ in ua       # ...and its version (policy asks for it)
    assert "python-httpx" not in ua      # not the blocked generic default
    assert "http" in ua or "@" in ua     # policy asks for a contact URL/address
