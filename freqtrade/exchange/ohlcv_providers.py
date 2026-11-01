"""
Keyless, rotating OHLCV providers for the IMT/WBNB PancakeSwap pool (BSC).

Drop this file next to immortality.py (freqtrade/exchange/ohlcv_providers.py).

Every provider returns HOURLY candles priced in BNB per IMT, shaped exactly like
the original server feed: [[ts_ms, open, high, low, close, volume], ...].
immortality.py already interpolates hourly data down to 1m/5m/15m/30m, so nothing
downstream changes. Because the pool is low-volume, empty hours are filled with flat
zero-volume candles up to the current hour (as the original server feed did).

Run `python ohlcv_providers.py` on your bot machine for a live self-check that
compares every provider against an independent DexScreener price.
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import requests


POOL = "0xfa56e9abcaa45207be5e43cf475ee061768ca915"  # IMT/WBNB (lowercase)
IMT_TOKEN = "0x2bf2141ed175f3236903cf07de33d7324871802d"  # noqa: S105 — token contract address, not a secret

GECKO_BASE = "https://api.geckoterminal.com/api/v2"
PAPRIKA_BASE = "https://api.dexpaprika.com"
DEXSCREENER_PAIR = "https://api.dexscreener.com/latest/dex/pairs/bsc/" + POOL

DEFAULT_SERVER_URL = "http://14.180.249.77/ohlcv.json"
TIMEOUT = (3.05, 8.0)  # connect, read

# Reject a provider's answer if its last close is wildly off the on-chain price.
# Wide on purpose: it only exists to catch unit/orientation errors (USD vs BNB,
# inverted pair), not to second-guess real volatility.
MAX_PRICE_RATIO = 4.0

Candle = list  # [ts_ms, open, high, low, close, volume]


class RateLimited(Exception):
    """Provider answered 429 (or equivalent)."""

    def __init__(self, retry_after: float | None = None):
        super().__init__(f"rate limited (retry_after={retry_after})")
        self.retry_after = retry_after


# --------------------------------------------------------------------------- #
# HTTP helper
# --------------------------------------------------------------------------- #
def _get_json(session: requests.Session, url: str, params=None, headers=None):
    r = session.get(url, params=params, headers=headers, timeout=TIMEOUT)
    if r.status_code == 429:
        ra = r.headers.get("Retry-After")
        try:
            retry_after = float(ra) if ra is not None else None
        except ValueError:
            retry_after = None
        raise RateLimited(retry_after)
    r.raise_for_status()
    return r.json()


# --------------------------------------------------------------------------- #
# Normalisation / validation
# --------------------------------------------------------------------------- #
def _orient_to_ref(candles: list[Candle], ref_price: float | None) -> list[Candle]:
    """Invert candles if they look like IMT-per-BNB instead of BNB-per-IMT."""
    if not candles or not ref_price or ref_price <= 0:
        return candles
    last = candles[-1][4]
    if last <= 0:
        return candles
    if abs(math.log((1.0 / last) / ref_price)) < abs(math.log(last / ref_price)):
        return [
            [ts, 1.0 / o, 1.0 / lo, 1.0 / h, 1.0 / c, v]  # high/low swap on inversion
            for ts, o, h, lo, c, v in candles
        ]
    return candles


def _clean(raw: list[Candle], ref_price: float | None) -> list[Candle]:
    out: dict[int, Candle] = {}
    for c in raw:
        try:
            ts, o, h, lo, cl, v = (
                int(c[0]),
                float(c[1]),
                float(c[2]),
                float(c[3]),
                float(c[4]),
                float(c[5]),
            )
        except (TypeError, ValueError, IndexError):
            continue
        if ts <= 0 or not all(math.isfinite(x) for x in (o, h, lo, cl, v)):
            continue
        if min(o, h, lo, cl) <= 0 or v < 0 or h < lo:
            continue
        out[ts] = [ts, o, h, lo, cl, v]
    candles = [out[k] for k in sorted(out)]
    if not candles:
        raise ValueError("no valid candles")
    if ref_price and ref_price > 0:
        ratio = candles[-1][4] / ref_price
        if not (1.0 / MAX_PRICE_RATIO) < ratio < MAX_PRICE_RATIO:
            raise ValueError(f"price sanity check failed (close/ref = {ratio:.4g})")
    return candles


HOUR_MS = 3_600_000
MAX_TAIL_FILL_HOURS = 24 * 30  # never manufacture more than 30 days of trailing candles


def _flat(ts: int, price: float) -> Candle:
    return [ts, price, price, price, price, 0.0]


def _fill_to_now(
    candles: list[Candle], ref_price: float | None, now_ms: int | None = None
) -> list[Candle]:
    """
    Recreate what the original server feed did for this low-volume pool: indexers
    only emit hours that had trades, so fill every empty hour with a flat zero-volume
    candle at the previous close, all the way up to the current hour. The current
    (still-open) hour is pinned to the on-chain price when it is known.
    """
    if not candles:
        return candles
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    cur = now_ms // HOUR_MS * HOUR_MS
    out: list[Candle] = []
    prev: Candle | None = None
    for c in candles:
        if prev is not None:
            t = prev[0] + HOUR_MS
            while t < c[0]:
                prev = _flat(t, prev[4])
                out.append(prev)
                t += HOUR_MS
        out.append(c)
        prev = c
    prev = candles[-1]
    t = max(prev[0] + HOUR_MS, cur - MAX_TAIL_FILL_HOURS * HOUR_MS)
    while t <= cur:
        prev = _flat(t, prev[4])
        out.append(prev)
        t += HOUR_MS
    last = out[-1]
    if ref_price and ref_price > 0 and last[0] == cur and last[5] == 0.0:
        last[4] = ref_price
        last[2] = max(last[2], ref_price)
        last[3] = min(last[3], ref_price)
    return out


# --------------------------------------------------------------------------- #
# Providers: fn(session, ref_price) -> raw candles (hourly, BNB per IMT)
# --------------------------------------------------------------------------- #
def fetch_geckoterminal(session: requests.Session, ref_price: float | None) -> list[Candle]:
    """
    GeckoTerminal (CoinGecko). Keyless. hour/aggregate=1, up to 1000 candles.

    `token=<IMT address>` + `currency=token` => price of IMT in the pool's other
    token (WBNB). Do NOT use token=quote: that returns WBNB priced in IMT.
    Gaps appear for hours with no trades (the exchange already fill_missing's).
    """
    data = _get_json(
        session,
        f"{GECKO_BASE}/networks/bsc/pools/{POOL}/ohlcv/hour",
        params={
            "aggregate": 1,
            "limit": 200,
            "currency": "token",
            "token": IMT_TOKEN,
        },
        headers={"Accept": "application/json;version=20230203"},
    )
    rows = data["data"]["attributes"]["ohlcv_list"]  # newest first, ts in seconds
    return [[int(r[0]) * 1000, r[1], r[2], r[3], r[4], r[5]] for r in rows]


def fetch_dexpaprika(session: requests.Session, ref_price: float | None) -> list[Candle]:
    """
    DexPaprika. Keyless tier = last 24h at >=1h intervals (more needs a free key).
    Candles are the raw pool price ratio, so orientation is auto-corrected against
    the on-chain reference price. Volume is in the pool's base-token units.
    """
    start = int(time.time()) - 23 * 3600  # stay inside the keyless 24h window
    rows = _get_json(
        session,
        f"{PAPRIKA_BASE}/networks/bsc/pools/{POOL}/ohlcv",
        params={"start": start, "interval": "1h", "limit": 24},
    )
    candles: list[Candle] = []
    for r in rows:
        t = datetime.fromisoformat(str(r["time_open"]))
        if t.tzinfo is None:
            t = t.replace(tzinfo=UTC)
        candles.append(
            [int(t.timestamp() * 1000), r["open"], r["high"], r["low"], r["close"], r["volume"]]
        )
    candles.sort(key=lambda c: c[0])
    return _orient_to_ref(candles, ref_price)


def make_server_fetcher(url: str) -> Callable[[requests.Session, float | None], list[Candle]]:
    """Your existing JSON feed: {"data": [{timestamp(s), open, high, low, close, volume}]}."""

    def fetch_server(session: requests.Session, ref_price: float | None) -> list[Candle]:
        rows = _get_json(session, url).get("data", [])
        return [
            [int(b["timestamp"]) * 1000, b["open"], b["high"], b["low"], b["close"], b["volume"]]
            for b in rows
        ]

    return fetch_server


# --------------------------------------------------------------------------- #
# Provider bookkeeping + rotation pool
# --------------------------------------------------------------------------- #
@dataclass
class Provider:
    name: str
    fn: Callable[[requests.Session, float | None], list[Candle]]
    min_spacing: float  # seconds between calls to this provider
    depth: int  # max candles it can return (used to pick a provider for cold starts)
    daily_cap: int | None = None  # hard ceiling on calls per UTC day
    # runtime state
    cooldown_until: float = 0.0
    next_ok: float = 0.0
    fails: int = 0
    calls_today: int = 0
    day: str = ""
    ok_count: int = 0
    err_count: int = 0


class OhlcvProviderPool:
    """
    Alternates providers on every request, skips any that are cooling down or
    spaced out, and fails over to the next one inside the same request.
    """

    def __init__(
        self,
        providers: list[Provider],
        *,
        rotation: str = "round_robin",  # or "random"
        cache_ttl: float = 30.0,
        max_stale: float = 1800.0,
        logger: logging.Logger | None = None,
    ):
        if not providers:
            raise ValueError("at least one OHLCV provider is required")
        self.providers = providers
        self.rotation = rotation
        self.cache_ttl = cache_ttl
        self.max_stale = max_stale
        self.log = logger or logging.getLogger(__name__)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "Freqtrade-Immortality/1.0"
        self.last_provider: str | None = None
        self._rr = 0
        self._cache: list[Candle] = []
        self._cache_t = -1e9
        self._good_raw: list[Candle] = []  # last validated provider answer (before gap-fill)
        self._good_t = -1e9
        self._stats_t = time.monotonic()
        self._lock = threading.Lock()

    # -- selection ----------------------------------------------------------
    def _order(self, need: int) -> list[Provider]:
        provs = list(self.providers)
        if self.rotation == "random":
            random.shuffle(provs)
        else:
            k = self._rr % len(provs)
            provs = provs[k:] + provs[:k]
        self._rr += 1
        # Cold start: providers with enough history go first (stable otherwise).
        if need:
            provs.sort(key=lambda p: p.depth < need)
        return provs

    def _reserve(self, p: Provider, now: float) -> bool:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if p.day != today:
            p.day, p.calls_today = today, 0
        if now < p.cooldown_until or now < p.next_ok:
            return False
        if p.daily_cap is not None and p.calls_today >= p.daily_cap:
            return False
        p.next_ok = now + p.min_spacing
        p.calls_today += 1
        return True

    # -- public -------------------------------------------------------------
    def fetch(self, ref_price: float | None = None, need: int = 0) -> list[Candle]:
        """Return validated hourly candles (oldest first), or [] if every provider failed."""
        with self._lock:
            now = time.monotonic()
            if self._cache and now - self._cache_t < self.cache_ttl:
                return list(self._cache)
            order = self._order(need)

        errors: list[str] = []
        for p in order:
            with self._lock:
                if not self._reserve(p, time.monotonic()):
                    continue
            try:
                clean = _clean(p.fn(self.session, ref_price), ref_price)
                candles = _fill_to_now(clean, ref_price)
            except RateLimited as e:
                with self._lock:
                    p.fails += 1
                    p.err_count += 1
                    wait = e.retry_after or min(60.0 * 2 ** (p.fails - 1), 900.0)
                    p.cooldown_until = time.monotonic() + wait
                errors.append(f"{p.name}: 429, cooling {wait:.0f}s")
                continue
            except Exception as e:  # network, bad JSON, validation, plan limits...
                with self._lock:
                    p.fails += 1
                    p.err_count += 1
                    wait = min(15.0 * 2 ** (p.fails - 1), 300.0)
                    p.cooldown_until = time.monotonic() + wait
                errors.append(f"{p.name}: {type(e).__name__}: {e} (cooling {wait:.0f}s)")
                continue
            with self._lock:
                p.fails = 0
                p.ok_count += 1
                self._cache, self._cache_t = candles, time.monotonic()
                self._good_raw, self._good_t = clean, self._cache_t
                self.last_provider = p.name
            self.log.debug("OHLCV from %s (%d candles)", p.name, len(candles))
            if time.monotonic() - self._stats_t >= 600:
                self.log.info("OHLCV provider stats (10 min): %s", self.stats())
                self._stats_t = time.monotonic()
            return list(candles)

        self.log.warning(
            "All OHLCV providers failed or are cooling down: %s", "; ".join(errors) or "none ready"
        )
        # Never leave the bot with no chart: re-project the last good answer up to "now"
        # (flat zero-volume candles), but only for a bounded time so it never trades blind.
        with self._lock:
            age = time.monotonic() - self._good_t
            good = [c[:] for c in self._good_raw]
        if good and age <= self.max_stale:
            self.log.warning("Serving stale OHLCV (last good fetch %.0fs ago)", age)
            return _fill_to_now(good, ref_price)
        return []

    def stats(self) -> dict[str, dict]:
        return {
            p.name: {
                "ok": p.ok_count,
                "err": p.err_count,
                "fails_in_row": p.fails,
                "calls_today": p.calls_today,
            }
            for p in self.providers
        }


def build_default_pool(config: dict, logger: logging.Logger | None = None) -> OhlcvProviderPool:
    """
    Optional freqtrade config keys (all optional):
      "ohlcv_url":       your server feed; set "" to disable it
      "ohlcv_providers": subset/order of ["server", "geckoterminal", "dexpaprika"]
      "ohlcv_rotation":  "round_robin" (default) | "random"
      "ohlcv_cache_ttl": seconds to reuse the last good answer (default 30)
      "ohlcv_max_stale": seconds the last good data may be re-served when every provider
                         fails (default 1800); after that fetch() returns []
    """
    server_url = config.get("ohlcv_url", DEFAULT_SERVER_URL)
    catalog: dict[str, Provider] = {
        # Conservative spacing: GeckoTerminal officially allows 30/min, but
        # CoinGecko's keyless docs say ~10/min under load.
        "geckoterminal": Provider(
            "geckoterminal", fetch_geckoterminal, min_spacing=6.0, depth=1000
        ),
        # Free tier is 10,000 requests/day; stay well under it.
        "dexpaprika": Provider(
            "dexpaprika", fetch_dexpaprika, min_spacing=10.0, depth=24, daily_cap=9000
        ),
    }
    if server_url:
        catalog["server"] = Provider(
            "server", make_server_fetcher(server_url), min_spacing=5.0, depth=1000
        )

    wanted = config.get("ohlcv_providers") or ["server", "geckoterminal", "dexpaprika"]
    providers = [catalog[n] for n in wanted if n in catalog]
    return OhlcvProviderPool(
        providers,
        rotation=config.get("ohlcv_rotation", "round_robin"),
        cache_ttl=float(config.get("ohlcv_cache_ttl", 30.0)),
        max_stale=float(config.get("ohlcv_max_stale", 1800.0)),
        logger=logger,
    )


# --------------------------------------------------------------------------- #
# Live self-check:  python ohlcv_providers.py
# --------------------------------------------------------------------------- #
def _dexscreener_price(session: requests.Session) -> float | None:
    """Independent reference: current IMT price in WBNB (DexScreener has no candles)."""
    data = _get_json(session, DEXSCREENER_PAIR)
    pair = data.get("pair") or (data.get("pairs") or [None])[0]
    return float(pair["priceNative"]) if pair else None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    s = requests.Session()
    s.headers["User-Agent"] = "Freqtrade-Immortality/1.0"
    ref = None
    try:
        ref = _dexscreener_price(s)
        print(f"DexScreener reference price: {ref:.6e} BNB/IMT")
    except Exception as e:
        print(f"DexScreener reference unavailable: {e}")

    for name, fn in (
        ("server", make_server_fetcher(DEFAULT_SERVER_URL)),
        ("geckoterminal", fetch_geckoterminal),
        ("dexpaprika", fetch_dexpaprika),
    ):
        try:
            c = _clean(fn(s, ref), ref)
            last = c[-1]
            when = datetime.fromtimestamp(last[0] / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M")
            drift = f"  vs ref x{last[4] / ref:.3f}" if ref else ""
            print(
                f"[OK ] {name:14s} {len(c):4d} candles, last {when}Z "
                f"close={last[4]:.6e} vol={last[5]:.4g}{drift}"
            )
        except Exception as e:
            print(f"[ERR] {name:14s} {type(e).__name__}: {e}")
        time.sleep(1.5)
